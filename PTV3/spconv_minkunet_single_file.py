from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from easydict import EasyDict

from loss_segmentation import Losses
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor, SparseModule


def ReplaceFeature(x: SparseConvTensor, features: Tensor) -> SparseConvTensor:
    if hasattr(x, 'replace_feature'):
        return x.replace_feature(features)
    x.features = features
    return x


def PadOrTrimFeatures(points: Tensor, in_channels: int) -> Tensor:
    ## 把点特征维度整理成固定的 in_channels。
    if points.shape[1] == in_channels:
        return points

    if points.shape[1] > in_channels:
        return points[:, :in_channels]

    pad_channels = in_channels - points.shape[1]
    pad_tensor = points.new_zeros((points.shape[0], pad_channels))
    return torch.cat((points, pad_tensor), dim=1)


def SparseQuantize(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ## 对重复 voxel 坐标去重，并建立 point -> voxel 映射。
    coords = coords - np.min(coords, axis=0)
    coords = coords.astype(np.uint64, copy=False)
    xmax = np.max(coords, axis=0).astype(np.uint64) + 1

    hashed = np.zeros(coords.shape[0], dtype=np.uint64)
    for axis_idx in range(coords.shape[1] - 1):
        hashed += coords[:, axis_idx]
        hashed *= xmax[axis_idx + 1]
    hashed += coords[:, -1]

    _, unique_indices, inverse_indices = np.unique(
        hashed,
        return_index=True,
        return_inverse=True,
    )
    return unique_indices, inverse_indices


def BuildBatchVoxels(
    points: Tensor,
    voxel_size: Any,
    point_cloud_range: Any,
    in_channels: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    把 collate 后的点云转成 spconv 稀疏体素输入。
    输入：
    - points: [N, C_total]
      - 第 0 列是 batch_idx
      - 第 1:4 列是点坐标 [x, y, z]，坐标系为 LiDAR 车体坐标系，单位 m
    - voxel_size: [3]，顺序是 [x_size, y_size, z_size]
    - point_cloud_range: [6]，顺序是 [x_min, y_min, z_min, x_max, y_max, z_max]
    输出：
    - voxel_features: [M, C_in]
    - voxel_coords: [M, 4]，严格采用 spconv 约定 [batch_idx, z_idx, y_idx, x_idx]
    - point2voxel_map: [N_valid]，把每个范围内点映射到其所属 voxel 的行号
    - b_valid_mask: [N]，标记哪些点落在固定 point_cloud_range 内
    - spatial_shape_zyx: [3]，固定网格大小 [Z, Y, X]
    '''
    if points.ndim != 2:
        raise ValueError(f'points must be [N, C], but got {points.shape}.')

    point_features_all = points[:, 1:]
    point_coords_xyz_all = point_features_all[:, :3]
    voxel_size_xyz = point_features_all.new_tensor(voxel_size)
    point_cloud_range_xyz = point_features_all.new_tensor(point_cloud_range)
    point_cloud_min_xyz = point_cloud_range_xyz[:3]
    point_cloud_max_xyz = point_cloud_range_xyz[3:]

    b_valid_mask = (
        (point_coords_xyz_all >= point_cloud_min_xyz)
        & (point_coords_xyz_all < point_cloud_max_xyz)
    ).all(dim=1)
    if not torch.any(b_valid_mask):
        raise ValueError(f'No points fall inside point_cloud_range={point_cloud_range}.')

    valid_points = points[b_valid_mask]
    batch_indices = valid_points[:, 0].to(torch.int32)
    point_features = valid_points[:, 1:]
    point_coords_xyz = point_features[:, :3]
    coords_xyz = torch.floor((point_coords_xyz - point_cloud_min_xyz) / voxel_size_xyz).to(torch.int32)

    ## spconv 3D 稀疏坐标顺序固定为 [batch, z, y, x]，这里把点坐标的 [x, y, z] 显式重排为 [z, y, x]。
    coords_zyx = coords_xyz[:, [2, 1, 0]]
    coords_with_batch = torch.cat((batch_indices.unsqueeze(1), coords_zyx), dim=1)

    coords_numpy = coords_with_batch.cpu().numpy()
    unique_indices, inverse_indices = SparseQuantize(coords_numpy)

    unique_indices_tensor = torch.from_numpy(unique_indices).to(device=points.device, dtype=torch.long)
    point2voxel_map = torch.from_numpy(inverse_indices).to(device=points.device, dtype=torch.long)
    voxel_features = PadOrTrimFeatures(point_features[unique_indices_tensor], in_channels)
    voxel_coords = coords_with_batch[unique_indices_tensor]
    spatial_shape_xyz = torch.round((point_cloud_max_xyz - point_cloud_min_xyz) / voxel_size_xyz).to(torch.int32)
    spatial_shape_zyx = spatial_shape_xyz[[2, 1, 0]]

    return voxel_features, voxel_coords, point2voxel_map, b_valid_mask, spatial_shape_zyx


class SparseConvBNReLU(SparseModule):
    ## 稀疏卷积基础块：conv + bn + relu。
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        conv_type: str = 'subm',
        indice_key: Optional[str] = None,
        b_activate: bool = True,
    ) -> None:
        super().__init__()

        if conv_type == 'subm':
            self.conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)
        elif conv_type == 'spconv':
            self.conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, indice_key=indice_key)
        elif conv_type == 'inverse':
            self.conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size=kernel_size, bias=False, indice_key=indice_key)
        else:
            raise ValueError(f'Unsupported conv_type={conv_type}.')

        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.b_activate = b_activate

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        x = self.conv(x)
        x = ReplaceFeature(x, self.norm(x.features))
        if self.b_activate:
            x = ReplaceFeature(x, self.relu(x.features))
        return x


class SparseBasicBlock(SparseModule):
    ## 稀疏残差块：两层 submanifold conv 加残差连接。
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        indice_key: str,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = spconv.SubMConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = spconv.SubMConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: SparseConvTensor) -> SparseConvTensor:
        identity = x.features

        out = self.conv1(x)
        out = ReplaceFeature(out, self.norm1(out.features))
        out = ReplaceFeature(out, self.relu(out.features))

        out = self.conv2(out)
        out = ReplaceFeature(out, self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x).features

        out = ReplaceFeature(out, out.features + identity)
        out = ReplaceFeature(out, self.relu(out.features))
        return out


class SpconvMinkUNet(nn.Module):
    ## 单帧语义分割版 MinkUNet：
    ## - 结构写法对齐多任务主模型中的语义分支
    ## - 只保留语义所需的 sparse encoder / decoder、point restore 和 segmentation loss
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        if not isinstance(config, EasyDict):
            config = EasyDict(config)
        self.config = config

        if not (
            len(self.config.encoder_channels) == len(self.config.encoder_blocks)
            == len(self.config.decoder_channels) == len(self.config.decoder_blocks)
        ):
            raise ValueError('Encoder/decoder channel and block settings must match in length.')

        encoder_channels = [self.config.base_channels, *self.config.encoder_channels]
        decoder_channels = [self.config.encoder_channels[-1], *self.config.decoder_channels]

        self.conv_input = nn.Sequential(
            SparseConvBNReLU(self.config.in_channels, self.config.base_channels, kernel_size=3, padding=1, conv_type='subm', indice_key='subm0'),
            SparseConvBNReLU(self.config.base_channels, self.config.base_channels, kernel_size=3, padding=1, conv_type='subm', indice_key='subm0'),
        )

        self.encoder_stage1 = self._BuildEncoderStage(in_channels=encoder_channels[0], out_channels=encoder_channels[1], num_blocks=self.config.encoder_blocks[0], stage_idx=1)
        self.encoder_stage2 = self._BuildEncoderStage(in_channels=encoder_channels[1], out_channels=encoder_channels[2], num_blocks=self.config.encoder_blocks[1], stage_idx=2)
        self.encoder_stage3 = self._BuildEncoderStage(in_channels=encoder_channels[2], out_channels=encoder_channels[3], num_blocks=self.config.encoder_blocks[2], stage_idx=3)
        self.encoder_stage4 = self._BuildEncoderStage(in_channels=encoder_channels[3], out_channels=encoder_channels[4], num_blocks=self.config.encoder_blocks[3], stage_idx=4)

        self.decoder_stage4_up = SparseConvBNReLU(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2, conv_type='inverse', indice_key='spconv4')
        self.decoder_stage4_blocks = self._BuildDecoderStage(merged_channels=decoder_channels[1] + encoder_channels[3], out_channels=decoder_channels[1], num_blocks=self.config.decoder_blocks[0], stage_idx=3)

        self.decoder_stage3_up = SparseConvBNReLU(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2, conv_type='inverse', indice_key='spconv3')
        self.decoder_stage3_blocks = self._BuildDecoderStage(merged_channels=decoder_channels[2] + encoder_channels[2], out_channels=decoder_channels[2], num_blocks=self.config.decoder_blocks[1], stage_idx=2)

        self.decoder_stage2_up = SparseConvBNReLU(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2, conv_type='inverse', indice_key='spconv2')
        self.decoder_stage2_blocks = self._BuildDecoderStage(merged_channels=decoder_channels[3] + encoder_channels[1], out_channels=decoder_channels[3], num_blocks=self.config.decoder_blocks[2], stage_idx=1)

        self.decoder_stage1_up = SparseConvBNReLU(decoder_channels[3], decoder_channels[4], kernel_size=2, stride=2, conv_type='inverse', indice_key='spconv1')
        self.decoder_stage1_blocks = self._BuildDecoderStage(merged_channels=decoder_channels[4] + encoder_channels[0], out_channels=decoder_channels[4], num_blocks=self.config.decoder_blocks[3], stage_idx=0)

        self.head = nn.Linear(self.config.decoder_channels[-1], self.config.num_classes)
        self.criterion_losses = self.BuildCriterionLosses()

    def BuildCriterionLosses(self) -> Losses:
        default_loss_types = ('DiceLossV1', 'LovLoss', 'CELoss')

        loss_types = tuple(getattr(self.config, 'loss_types', ()))
        if len(loss_types) == 0:
            loss_types = default_loss_types

        loss_weights = tuple(getattr(self.config, 'loss_weights', ()))
        if len(loss_weights) == 0:
            loss_weights = tuple(1.0 for _ in loss_types)

        if len(loss_types) != len(loss_weights):
            raise ValueError(
                'LOSS_TYPES and LOSS_WEIGHTS must have the same length. '
                f'Got {len(loss_types)} vs {len(loss_weights)}.'
            )

        class_weight = None if self.config.class_weight is None else torch.tensor(self.config.class_weight, dtype=torch.float32)
        cls_num_pts = None if self.config.cls_num_pts is None else list(self.config.cls_num_pts)
        class_names = None if self.config.class_names is None else list(self.config.class_names)

        self.loss_types = list(loss_types)
        self.loss_weights = list(loss_weights)
        return Losses(
            loss_types=self.loss_types,
            loss_weights=self.loss_weights,
            cls_num_pts=cls_num_pts,
            ignore_index=self.config.ignore_index,
            knn=self.config.knn,
            label_smoothing=self.config.label_smoothing,
            class_weight=class_weight,
            class_names=class_names,
        )

    def ForwardNetwork(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(batch_dict, dict):
            raise TypeError(f'batch_dict must be dict, but got {type(batch_dict)}')
        if 'points' not in batch_dict:
            raise KeyError("Batch dict must contain key 'points'.")

        point_tensor_all = batch_dict['points']
        if not isinstance(point_tensor_all, torch.Tensor):
            raise TypeError("Batch dict key 'points' must be a torch.Tensor.")

        voxel_features, voxel_coords, point2voxel_map, b_valid_mask, spatial_shape_zyx = BuildBatchVoxels(
            point_tensor_all,
            self.config.voxel_size,
            self.config.point_cloud_range,
            self.config.in_channels,
        )
        point_tensor = point_tensor_all[b_valid_mask]
        batch_indices = point_tensor[:, 0].to(torch.long)

        batch_size = len(batch_dict['token']) if 'token' in batch_dict else int(point_tensor_all[:, 0].max().item()) + 1
        x = SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=spatial_shape_zyx.tolist(),
            batch_size=batch_size,
        )

        x0 = self.conv_input(x)
        x1 = self.encoder_stage1(x0)
        x2 = self.encoder_stage2(x1)
        x3 = self.encoder_stage3(x2)
        x4 = self.encoder_stage4(x3)

        y3 = self.decoder_stage4_up(x4)
        y3 = ReplaceFeature(y3, torch.cat((y3.features, x3.features), dim=1))
        y3 = self.decoder_stage4_blocks(y3)

        y2 = self.decoder_stage3_up(y3)
        y2 = ReplaceFeature(y2, torch.cat((y2.features, x2.features), dim=1))
        y2 = self.decoder_stage3_blocks(y2)

        y1 = self.decoder_stage2_up(y2)
        y1 = ReplaceFeature(y1, torch.cat((y1.features, x1.features), dim=1))
        y1 = self.decoder_stage2_blocks(y1)

        y0 = self.decoder_stage1_up(y1)
        y0 = ReplaceFeature(y0, torch.cat((y0.features, x0.features), dim=1))
        y0 = self.decoder_stage1_blocks(y0)

        voxel_logits = self.head(y0.features)
        point_logits = voxel_logits[point2voxel_map]
        _, point_count_per_batch = torch.unique_consecutive(batch_indices, return_counts=True)
        point_logits_list = [chunk.contiguous() for chunk in point_logits.split(point_count_per_batch.tolist(), dim=0)]

        return {
            'voxel_logits': voxel_logits,
            'point_logits': point_logits,
            'point_logits_list': point_logits_list,
            'voxel_coords': voxel_coords,
            'point2voxel_map': point2voxel_map,
            'b_valid_mask': b_valid_mask,
            'point_tensor': point_tensor,
            'batch_indices': batch_indices,
        }

    def ComputeLoss(
        self,
        batch_dict: Dict[str, Any],
        forward_dict: Dict[str, Any],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if 'semantic_label' not in batch_dict:
            raise KeyError("Training batch must contain key 'semantic_label'.")

        point_logits = forward_dict['point_logits']
        point_tensor = forward_dict['point_tensor']
        b_valid_mask = forward_dict['b_valid_mask']
        batch_indices = forward_dict['batch_indices']
        target = batch_dict['semantic_label'][b_valid_mask].to(point_logits.device).long()

        if point_logits.shape[0] != target.shape[0]:
            raise ValueError(
                'point logits count does not match semantic label count. '
                f'Got {point_logits.shape[0]} vs {target.shape[0]}.'
            )

        coords_xyz = point_tensor[:, 1:4]
        _, point_count_per_batch = torch.unique_consecutive(batch_indices, return_counts=True)
        offset = point_count_per_batch.to(device=target.device, dtype=torch.long).cumsum(dim=0)

        total_loss, loss_dict = self.criterion_losses(
            point_logits,
            target,
            xyz=coords_xyz,
            offset=offset,
        )
        return total_loss, loss_dict

    def forward(
        self,
        batch_dict: Dict[str, Any],
        b_return_details: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        forward_dict = self.ForwardNetwork(batch_dict)

        if self.training:
            total_loss, loss_dict = self.ComputeLoss(batch_dict, forward_dict)
            logs = {
                'task': ['seg'],
                'loss': total_loss.detach(),
                **loss_dict,
            }
            return total_loss, logs

        if b_return_details:
            return forward_dict

        point_logits = forward_dict['point_logits']
        point_pred = point_logits.argmax(dim=1)
        point_tensor = forward_dict['point_tensor']
        batch_indices = forward_dict['batch_indices']

        if 'token' in batch_dict:
            result_dict = {}
            for batch_idx, token in enumerate(batch_dict['token']):
                b_mask = batch_indices == batch_idx
                result_dict[token] = {
                    'point_logits': point_logits[b_mask].detach().cpu(),
                    'point_pred': point_pred[b_mask].detach().cpu(),
                    'point_coords': point_tensor[b_mask, 1:4].detach().cpu(),
                }
            return result_dict

        return point_logits

    def _BuildEncoderStage(self, in_channels: int, out_channels: int, num_blocks: int, stage_idx: int) -> nn.Sequential:
        stage_layers = [SparseConvBNReLU(in_channels, in_channels, kernel_size=2, stride=2, conv_type='spconv', indice_key=f'spconv{stage_idx}')]
        for block_idx in range(num_blocks):
            if block_idx == 0 and in_channels != out_channels:
                stage_layers.append(
                    SparseBasicBlock(
                        in_channels,
                        out_channels,
                        indice_key=f'subm{stage_idx}',
                        downsample=SparseConvBNReLU(in_channels, out_channels, kernel_size=1, conv_type='subm', indice_key=f'subm{stage_idx}', b_activate=False),
                    )
                )
            else:
                stage_layers.append(SparseBasicBlock(out_channels, out_channels, indice_key=f'subm{stage_idx}'))
        return nn.Sequential(*stage_layers)

    def _BuildDecoderStage(self, merged_channels: int, out_channels: int, num_blocks: int, stage_idx: int) -> nn.Sequential:
        stage_layers = []
        for block_idx in range(num_blocks):
            if block_idx == 0:
                stage_layers.append(
                    SparseBasicBlock(
                        merged_channels,
                        out_channels,
                        indice_key=f'subm{stage_idx}',
                        downsample=SparseConvBNReLU(merged_channels, out_channels, kernel_size=1, conv_type='subm', indice_key=f'subm{stage_idx}', b_activate=False),
                    )
                )
            else:
                stage_layers.append(SparseBasicBlock(out_channels, out_channels, indice_key=f'subm{stage_idx}'))
        return nn.Sequential(*stage_layers)
