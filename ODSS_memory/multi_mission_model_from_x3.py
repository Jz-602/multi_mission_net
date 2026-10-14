from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import spconv.pytorch as spconv
import torch
from easydict import EasyDict
from spconv.pytorch import SparseConvTensor, SparseModule
from torch import Tensor, nn
from ASPPNeck import ASPPNeck
from CenterHead import CenterHead
from loss_segmentation import Losses



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

    _, unique_indices, inverse_indices = np.unique(hashed,return_index=True,return_inverse=True,)
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

    b_valid_mask = ((point_coords_xyz_all >= point_cloud_min_xyz) & (point_coords_xyz_all < point_cloud_max_xyz)).all(dim=1)
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
            self.conv = spconv.SubMConv3d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False,indice_key=indice_key,)
        elif conv_type == 'spconv':
            self.conv = spconv.SparseConv3d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False,indice_key=indice_key,)
        elif conv_type == 'inverse':
            self.conv = spconv.SparseInverseConv3d(in_channels,out_channels,kernel_size=kernel_size,bias=False,indice_key=indice_key,)
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
        self.conv1 = spconv.SubMConv3d(in_channels,out_channels,kernel_size=3,padding=1,bias=False,indice_key=indice_key,)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = spconv.SubMConv3d(out_channels,out_channels,kernel_size=3,padding=1,bias=False,indice_key=indice_key,)
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


class BEVResidualBlock(nn.Module):
    ## 2D BEV 残差块：
    ## - 输入/输出: [B, C, H, W]
    ## - 坐标系: LiDAR 车体坐标系下的 BEV 网格，H/W 分别对应 y/x
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, bev_feature_map: Tensor) -> Tensor:
        identity_feature_map = bev_feature_map
        bev_feature_map = self.conv1(bev_feature_map)
        bev_feature_map = self.norm1(bev_feature_map)
        bev_feature_map = self.relu(bev_feature_map)
        bev_feature_map = self.conv2(bev_feature_map)
        bev_feature_map = self.norm2(bev_feature_map)
        bev_feature_map = self.relu(bev_feature_map + identity_feature_map)
        return bev_feature_map


class DetectionBEVProjectorFromX3(nn.Module):
    ## 把 x3 的 3D sparse voxel 特征整理成检测可直接使用的 2D BEV 特征图。
    ##
    ## 输入：
    ## - x: SparseConvTensor
    ##   - x.features: [M, C_in]
    ##   - x.indices: [M, 4]，每行是 [batch_idx, z_idx, y_idx, x_idx]
    ##   - 坐标系：LiDAR 车体坐标系离散后的 voxel 坐标，空间维顺序严格对应 [z, y, x]
    ## 输出：
    ## - bev_feature_map: [B, C_bev, H, W]
    ##   - 先在 z 方向额外做一次 stride=2 的 sparse 卷积，把 x3 的高度层数减半
    ##   - 再通过一个 sparse 残差块整理压缩后的 3D 特征
    ##   - 最后 dense 到 [B, C, Z, Y, X]，展平高度维并压成固定 proj_channels 的 BEV 特征
    ##   - H/W 对应 BEV 的 y/x 平面，坐标系仍为 LiDAR 车体坐标系下的 BEV 网格
    def __init__(
        self,
        in_channels: int,
        proj_channels: int,
        stacked_bev_in_channels: int,
    ) -> None:
        super().__init__()
        self.z_downsample = SparseConvBNReLU(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            conv_type='spconv',
            indice_key='spconv_det_z_x3',
        )
        self.z_refine = SparseBasicBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            indice_key='subm_det_z_x3',
        )
        self.pre_bev_conv = SparseConvBNReLU(
            in_channels=in_channels,
            out_channels=proj_channels,
            kernel_size=1,
            padding=0,
            conv_type='subm',
            indice_key='subm_bev_proj',
        )

        ## 把 dense 后展平得到的 [B, C * Z, H, W] 压回固定 proj_channels。
        self.height_compress = nn.Sequential(
            nn.Conv2d(
                in_channels=stacked_bev_in_channels,
                out_channels=proj_channels * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(proj_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=proj_channels * 2,
                out_channels=proj_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )

        ## 用少量 2D 残差块细化 BEV 表达：
        ## - 主要职责是把 3D->2D 压缩后的特征整理成更适合检测 head 的 256 维 BEV 特征
        self.bev_refine = nn.Sequential(
            BEVResidualBlock(proj_channels),
            BEVResidualBlock(proj_channels),
        )

    def forward(self, x: SparseConvTensor) -> Tensor:
        x = self.z_downsample(x)
        x = self.z_refine(x)
        x = self.pre_bev_conv(x)
        ## spconv dense() 的输出维度为 [B, C, Z, Y, X]，其中 Z 对应高度方向。
        dense_feature_map = x.dense()
        batch_size, num_channels, depth_size, height_size, width_size = dense_feature_map.shape
        ## 把高度维 Z 直接并到 channel，得到原始 BEV 平面特征 [B, C * Z, Y, X]。
        stacked_bev_feature_map = dense_feature_map.view(batch_size, num_channels * depth_size, height_size, width_size)

        ## 先沿 channel 维融合高度信息，再做轻量 2D BEV 细化。
        bev_feature_map = self.height_compress(stacked_bev_feature_map)
        bev_feature_map = self.bev_refine(bev_feature_map)

        return bev_feature_map



class MultiMissionModel(nn.Module):
    """Single-file MinkUNet-like model with spconv backend only.

    This file intentionally removes trainer / logger / registry / loss plumbing.
    The kept path is:
    points -> voxelize -> sparse UNet -> voxel logits -> per-point logits
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        if not isinstance(config, EasyDict):
            config = EasyDict(config)
        self.config = config
        class_names = None if self.config.class_names is None else list(self.config.class_names)

        ## 这里直接把 backbone 结构并进主模型，不再额外跳到 backbone 类里看。
        self.conv_input = nn.Sequential(
            SparseConvBNReLU(self.config.in_channels, self.config.base_channels, kernel_size=3, padding=1, conv_type='subm', indice_key='subm0'),
            SparseConvBNReLU(self.config.base_channels, self.config.base_channels, kernel_size=3, padding=1, conv_type='subm', indice_key='subm0'),
        )

        encoder_channels = [self.config.base_channels, *self.config.encoder_channels]
        decoder_channels = [self.config.encoder_channels[-1], *self.config.decoder_channels]

        ## 编码器
        self.encoder_stage1 = self._BuildEncoderStage(in_channels=encoder_channels[0], out_channels=encoder_channels[1], num_blocks=self.config.encoder_blocks[0], stage_idx=1)
        self.encoder_stage2 = self._BuildEncoderStage(in_channels=encoder_channels[1], out_channels=encoder_channels[2], num_blocks=self.config.encoder_blocks[1], stage_idx=2)
        self.encoder_stage3 = self._BuildEncoderStage(in_channels=encoder_channels[2], out_channels=encoder_channels[3], num_blocks=self.config.encoder_blocks[2], stage_idx=3)
        self.encoder_stage4 = self._BuildEncoderStage(in_channels=encoder_channels[3], out_channels=encoder_channels[4], num_blocks=self.config.encoder_blocks[3], stage_idx=4)

        ## 检测分支：从 x3 开始做目标检测。
        bev_projection_channels = int(self.config.neck.in_channels)
        z_depth_after_encoder = int(round((self.config.point_cloud_range[5] - self.config.point_cloud_range[2]) / self.config.voxel_size[2]))
        for _ in range(len(self.config.encoder_blocks) - 1):
            z_depth_after_encoder = max(z_depth_after_encoder // 2, 1)
        stacked_bev_in_channels = bev_projection_channels * z_depth_after_encoder
        self.detection_bev_projector = DetectionBEVProjectorFromX3(
            in_channels=encoder_channels[-2],
            proj_channels=bev_projection_channels,
            stacked_bev_in_channels=stacked_bev_in_channels,
        )
        self.detection_neck = ASPPNeck(in_channels=int(self.config.neck.in_channels))
        self.detection_head = CenterHead(
            in_channels=int(self.config.head.in_channels),
            tasks=[list(task) for task in self.config.tasks],
            weight=self.config.head.weight,
            code_weights=list(self.config.head.code_weights),
            common_heads=self.config.head.common_heads,
            strides=list(self.config.head.strides),
            with_reg_iou=bool(self.config.head.with_reg_iou),
            voxel_size=list(self.config.voxel_size),
            pc_range=list(self.config.point_cloud_range),
            out_size_factor=list(self.config.out_size_factor),
            rectifier=self.config.head.rectifier,
        )
        self.detection_post_processing = EasyDict(self.config.post_processing)
        self.detection_post_processing.voxel_size = list(self.config.voxel_size)
        self.detection_post_processing.pc_range = list(self.config.point_cloud_range)
        self.detection_post_processing.out_size_factor = list(self.config.out_size_factor)

        ## 语义分支
        self.decoder_stage4_up = SparseConvBNReLU(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2, conv_type='inverse', indice_key='spconv4')
        self.decoder_stage4_blocks = self._BuildDecoderStage(merged_channels=decoder_channels[1] + encoder_channels[3], out_channels=decoder_channels[1], num_blocks=self.config.decoder_blocks[0], stage_idx=3)

        self.decoder_stage3_up = SparseConvBNReLU(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2, conv_type='inverse', indice_key='spconv3')
        self.decoder_stage3_blocks = self._BuildDecoderStage(merged_channels=decoder_channels[2] + encoder_channels[2], out_channels=decoder_channels[2], num_blocks=self.config.decoder_blocks[1], stage_idx=2)

        self.decoder_stage2_up = SparseConvBNReLU(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2, conv_type='inverse', indice_key='spconv2')
        self.decoder_stage2_blocks = self._BuildDecoderStage(merged_channels=decoder_channels[3] + encoder_channels[1], out_channels=decoder_channels[3], num_blocks=self.config.decoder_blocks[2], stage_idx=1)

        self.decoder_stage1_up = SparseConvBNReLU(decoder_channels[3], decoder_channels[4], kernel_size=2, stride=2, conv_type='inverse', indice_key='spconv1')
        self.decoder_stage1_blocks = self._BuildDecoderStage(merged_channels=decoder_channels[4] + encoder_channels[0], out_channels=decoder_channels[4], num_blocks=self.config.decoder_blocks[3], stage_idx=0)

        ## head 和 loss 仍然放在主模型尾部，和训练分支保持一处汇总。
        self.head = nn.Linear(self.config.decoder_channels[-1], self.config.num_classes)
        self.loss_types = list(self.config.loss_types)
        self.loss_weights = list(self.config.loss_weights)
        self.criterion_losses = Losses(
            loss_types=self.loss_types,
            loss_weights=self.loss_weights,
            cls_num_pts=self.config.cls_num_pts,
            ignore_index=self.config.ignore_index,
            knn=self.config.knn,
            label_smoothing=self.config.label_smoothing,
            class_weight=None if self.config.class_weight is None else torch.tensor(self.config.class_weight,dtype=torch.float32,),
            class_names=class_names,
        )

    def forward(self, batch_dict: Dict[str, Any]) -> Union[Tensor, Dict[str, Tensor]]:
        ## 先把 collate 后的点云做体素化，并保留点到体素的映射。
        point_tensor_all = batch_dict['points']
        voxel_features, voxel_coords, point2voxel_map, b_valid_mask, spatial_shape_zyx = BuildBatchVoxels(
            point_tensor_all,
            self.config.voxel_size,
            self.config.point_cloud_range,
            self.config.in_channels,
        )
        point_tensor = point_tensor_all[b_valid_mask]

        ## 这里直接展开 sparse UNet 主干，后面接检测分支或 memory 时不用再进 backbone.forward。
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

        detection_feature_map = self.detection_bev_projector(x3)
        detection_feature_map = self.detection_neck(detection_feature_map)
        detection_pred_dicts = self.detection_head(detection_feature_map)

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

        ## 先得到 voxel 级预测，再按 point2voxel_map 还原回点级预测。
        voxel_logits = self.head(y0.features)
        point_logits = voxel_logits[point2voxel_map]
        batch_indices = point_tensor[:, 0].to(torch.long)

        if self.training:
            ## 训练态直接在这里计算分割 loss，保持 trainer 侧接口不变。
            target = batch_dict['semantic_label'][b_valid_mask].to(point_logits.device).long()
            coords_xyz = point_tensor[:, 1:4]
            _, counts = torch.unique_consecutive(batch_indices, return_counts=True)
            offset = counts.to(device=target.device, dtype=torch.long).cumsum(dim=0)
            seg_loss, loss_dict = self.criterion_losses(
                point_logits,
                target,
                xyz=coords_xyz,
                offset=offset,
            )
            det_loss, det_logs = self.detection_head.loss(batch_dict, detection_pred_dicts)
            total_loss = seg_loss + det_loss
            logs: List[Dict[str, Any]] = [{
                'task': ['seg'],
                'loss': seg_loss.detach(),
                **loss_dict,
            }]
            logs.extend(det_logs)
            return total_loss, logs

        ## 非训练态按 token 返回最终分割结果，直接给 trainer 保存。
        point_pred = point_logits.argmax(dim=1)
        detection_outputs = self.detection_head.predict(batch_dict, detection_pred_dicts, self.detection_post_processing)
        result_dict = {}
        point_coords = point_tensor[:, 1:4]
        b_return_bev_feature_map = bool(batch_dict.get('b_return_bev_feature_map', False))
        for batch_idx, token in enumerate(batch_dict['token']):
            b_mask = batch_indices == batch_idx
            result_dict[token] = {
                'seg': {
                    'point_pred': point_pred[b_mask].detach().cpu(),
                    'point_logits': point_logits[b_mask].detach().cpu(),
                    'point_coords': point_coords[b_mask].detach().cpu(),
                }}
            if b_return_bev_feature_map:
                result_dict[token]['bev'] = {
                    'feature_map': detection_feature_map[batch_idx].detach().cpu(),
                }
        for detection_output in detection_outputs:
            token = detection_output['token']
            if token not in result_dict:
                result_dict[token] = {}
            result_dict[token]['det'] = {
                'box3d_lidar': detection_output['box3d_lidar'].detach().cpu(),
                'scores': detection_output['scores'].detach().cpu(),
                'label_preds': detection_output['label_preds'].detach().cpu(),
            }
        return result_dict

    def _BuildEncoderStage(self, in_channels: int, out_channels: int, num_blocks: int, stage_idx: int) -> nn.Sequential:
        stage_layers = [SparseConvBNReLU(in_channels, in_channels, kernel_size=2, stride=2, conv_type='spconv', indice_key=f'spconv{stage_idx}')]
        for block_idx in range(num_blocks):
            if block_idx == 0 and in_channels != out_channels:
                stage_layers.append(SparseBasicBlock(in_channels, out_channels, indice_key=f'subm{stage_idx}', downsample=SparseConvBNReLU(in_channels, out_channels, kernel_size=1, conv_type='subm', indice_key=f'subm{stage_idx}', b_activate=False)))
            else:
                stage_layers.append(SparseBasicBlock(out_channels, out_channels, indice_key=f'subm{stage_idx}'))
        return nn.Sequential(*stage_layers)

    def _BuildDecoderStage(self, merged_channels: int, out_channels: int, num_blocks: int, stage_idx: int) -> nn.Sequential:
        stage_layers = []
        for block_idx in range(num_blocks):
            if block_idx == 0:
                stage_layers.append(SparseBasicBlock(merged_channels, out_channels, indice_key=f'subm{stage_idx}', downsample=SparseConvBNReLU(merged_channels, out_channels, kernel_size=1, conv_type='subm', indice_key=f'subm{stage_idx}', b_activate=False)))
            else:
                stage_layers.append(SparseBasicBlock(out_channels, out_channels, indice_key=f'subm{stage_idx}'))
        return nn.Sequential(*stage_layers)
