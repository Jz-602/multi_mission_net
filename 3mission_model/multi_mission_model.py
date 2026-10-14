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
from terrain_head import TerrainHead, TerrainLoss
from occ_head import GeneratedOCCHead



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
    coords_xyz = torch.minimum(coords_xyz.clamp_min(0), torch.round((point_cloud_max_xyz - point_cloud_min_xyz) / voxel_size_xyz).to(torch.int32) - 1)

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


class DetectionBEVProjector(nn.Module):
    ## 把 x4 的 3D sparse voxel 特征整理成检测可直接使用的 2D BEV 特征图。
    '''输入：
    - x: SparseConvTensor
      - x.features: [M, C_in]
      - x.indices: [M, 4]，每行是 [batch_idx, z_idx, y_idx, x_idx]
      - 坐标系：LiDAR 车体坐标系离散后的 voxel 坐标，空间维顺序严格对应 [z, y, x]
    输出：
    - bev_feature_map: [B, C_bev, H, W]
      - 先把 dense 后的 [B, C, Z, Y, X] 展平成 [B, C * Z, Y, X]
      - 再通过 2D BEV adapter 融合高度信息
      - 经过一次上采样前细化、x/y 上采样、上采样后细化
      - 输出给后续 ASPPNeck / CenterHead 的检测 BEV 特征
    '''

    def __init__(
        self,
        in_channels: int,
        proj_channels: int,
        stacked_bev_in_channels: int,
    ) -> None:
        super().__init__()
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

        ## 上采样前先细化一次，避免直接把较粗的 x4 BEV 特征硬放大。
        self.bev_refine_before_upsample = BEVResidualBlock(proj_channels)

        ## 把 x4 对应的 stride=16 BEV 特征提升到 stride=8，和后续检测头的输出步长配合后回到 stride=4。
        self.xy_upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=proj_channels,
                out_channels=proj_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(proj_channels),
            nn.ReLU(inplace=True),
        )

        ## 上采样后再细化一次，补一些局部几何与边界信息。
        self.bev_refine_after_upsample = BEVResidualBlock(proj_channels)

    def forward(self, x: SparseConvTensor) -> Tensor:
        x = self.pre_bev_conv(x)
        ## spconv dense() 的输出维度为 [B, C, Z, Y, X]，其中 Z 对应高度方向。
        dense_feature_map = x.dense()
        batch_size, num_channels, depth_size, height_size, width_size = dense_feature_map.shape
        ## 把高度维 Z 直接并到 channel，得到原始 BEV 平面特征 [B, C * Z, Y, X]。
        stacked_bev_feature_map = dense_feature_map.view(batch_size, num_channels * depth_size, height_size, width_size)

        ## 先沿 channel 维融合高度信息，再做一次上采样前细化、x/y 上采样和上采样后细化。
        bev_feature_map = self.height_compress(stacked_bev_feature_map)
        bev_feature_map_before_upsample = self.bev_refine_before_upsample(bev_feature_map)
        bev_feature_map = self.xy_upsample(bev_feature_map_before_upsample)
        bev_feature_map = self.bev_refine_after_upsample(bev_feature_map)

        return bev_feature_map


class TaskUncertaintyLoss(nn.Module):
    '''
    按 LidarMultiNet 的 task-level uncertainty weighting 聚合多任务 loss。
    输入：
    - task_losses: dict[str, Tensor]，例如 {'seg': LSEG, 'det': LDET}
      每个 task loss 已经是该任务内部各 loss component 的固定权重和。
    输出：
    - total_loss: 标量 Tensor，用于 backward
    - log_dict: 每个任务的可复查权重信息
    '''
    def __init__(self, task_names: List[str]) -> None:
        super().__init__()
        if len(task_names) == 0:
            raise ValueError('task_names must not be empty.')
        self.task_names = list(task_names)
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in self.task_names})

    def forward(self, task_losses: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Dict[str, Tensor]]]:
        total_loss: Optional[Tensor] = None
        log_dict: Dict[str, Dict[str, Tensor]] = {}

        for task_idx, task_name in enumerate(self.task_names):
            if task_name not in task_losses:
                continue

            task_loss = task_losses[task_name]
            log_var = self.log_vars[task_name]
            task_weight = 0.5 * torch.exp(-log_var)
            weighted_loss = task_weight * task_loss + 0.5 * log_var
            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
            log_dict[task_name] = {
                'weighted_loss': weighted_loss.detach(),
                'log_var': log_var.detach(),
                'loss_weight': task_weight.detach(),
            }

        if total_loss is None:
            raise ValueError(f'No task loss matched task_names={self.task_names}.')

        return total_loss, log_dict


class MultiMissionModel(nn.Module):
    """共享稀疏编码器上的检测、地形高度与生成式静态 OCC。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        if not isinstance(config, EasyDict):
            config = EasyDict(config)
        self.config = config

        ## 这里直接把 backbone 结构并进主模型，不再额外跳到 backbone 类里看。
        self.conv_input = nn.Sequential(
            SparseConvBNReLU(self.config.in_channels, self.config.base_channels, kernel_size=3, padding=1, conv_type='subm', indice_key='subm0'),
            SparseConvBNReLU(self.config.base_channels, self.config.base_channels, kernel_size=3, padding=1, conv_type='subm', indice_key='subm0'),
        )

        encoder_channels = [self.config.base_channels, *self.config.encoder_channels]

        ## 编码器
        self.encoder_stage1 = self._BuildEncoderStage(in_channels=encoder_channels[0], out_channels=encoder_channels[1], num_blocks=self.config.encoder_blocks[0], stage_idx=1)
        self.encoder_stage2 = self._BuildEncoderStage(in_channels=encoder_channels[1], out_channels=encoder_channels[2], num_blocks=self.config.encoder_blocks[1], stage_idx=2)
        self.encoder_stage3 = self._BuildEncoderStage(in_channels=encoder_channels[2], out_channels=encoder_channels[3], num_blocks=self.config.encoder_blocks[2], stage_idx=3)
        self.encoder_stage4 = self._BuildEncoderStage(in_channels=encoder_channels[3], out_channels=encoder_channels[4], num_blocks=self.config.encoder_blocks[3], stage_idx=4)

        ## 检测分支
        bev_projection_channels = int(self.config.neck.in_channels)
        z_depth_after_encoder = int(round((self.config.point_cloud_range[5] - self.config.point_cloud_range[2]) / self.config.voxel_size[2]))
        for _ in range(len(self.config.encoder_blocks)):
            z_depth_after_encoder = max(z_depth_after_encoder // 2, 1)
        stacked_bev_in_channels = bev_projection_channels * z_depth_after_encoder
        self.detection_bev_projector = DetectionBEVProjector(
            in_channels=encoder_channels[-1],
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

        terrain_config = self.config.get('terrain', {})
        self.terrain_head = TerrainHead(in_channels=bev_projection_channels,
            b_predict_validity=bool(terrain_config.get('b_predict_validity', False)))
        self.terrain_criterion = TerrainLoss(validity_weight=float(terrain_config.get('validity_weight', 1.0)))
        occ_config = dict(self.config.get('occ', {}))
        occ_config['pc_range'] = list(self.config.point_cloud_range)
        occ_config['voxel_size_xyz'] = [float(v) * 4 for v in self.config.voxel_size]
        self.occ_head = GeneratedOCCHead(
            occ_config, sparse_basic_block_cls=SparseBasicBlock,
            sparse_conv_bn_relu_cls=SparseConvBNReLU,
        )
        self.task_uncertainty_loss = TaskUncertaintyLoss(['det', 'terrain', 'occ'])
        self.b_three_task = True

    def EncodeCurrent(self, batch_dict):
        # points: [N, batch+x+y+z], 当前车体系；稀疏坐标为 BZYX。
        features, coords, _, _, shape = BuildBatchVoxels(
            batch_dict['points'], self.config.voxel_size,
            self.config.point_cloud_range, self.config.in_channels,
        )
        batch_size = len(batch_dict['token'])
        x = SparseConvTensor(features, coords, shape.tolist(), batch_size)
        x0 = self.conv_input(x)
        x1 = self.encoder_stage1(x0)
        x2 = self.encoder_stage2(x1)
        return x2, self.encoder_stage3(x2)

    def ForwardHeads(self, x4, x3, x2):
        # 先完成稀疏分支的临时工作区，再保存二维分支激活，降低联合训练峰值。
        occ = self.occ_head(x4, x3, x2)
        shared_bev = self.detection_bev_projector(x4)
        terrain = self.terrain_head(shared_bev)
        det = self.detection_head(self.detection_neck(shared_bev))
        return {'det': det, 'terrain': terrain, 'occ': occ,
                'counts': {'x2': len(x2.indices), 'x3': len(x3.indices), 'x4': len(x4.indices)}}

    def ForwardNetwork(self, batch_dict):
        x2, x3 = self.EncodeCurrent(batch_dict)
        x4 = self.encoder_stage4(x3)
        return self.ForwardHeads(x4, x3, x2)

    def ComputeLoss(self, outputs, batch_dict):
        det_loss, det_logs = self.detection_head.loss(batch_dict, outputs['det'])
        terrain_loss, terrain_logs, b_terrain = self.terrain_criterion.ComputeLoss(outputs['terrain'], batch_dict)
        occ_loss, occ_logs, b_occ = self.occ_head.ComputeLoss(outputs['occ'], batch_dict)
        task_losses = {'det': det_loss}
        zero_loss = det_loss.new_zeros(())
        for name, loss, b_valid in [('terrain', terrain_loss, b_terrain), ('occ', occ_loss, b_occ)]:
            if b_valid:
                task_losses[name] = loss
            else:
                # 无监督不加入 uncertainty 正则，同时所有任务参数仍参与 DDP 图。
                zero_loss = zero_loss + loss + self.task_uncertainty_loss.log_vars[name] * 0
        total, weights = self.task_uncertainty_loss(task_losses)
        logs = [{'task': ['det'], 'loss': det_loss.detach(), **weights['det']}]
        for key in ('hm_loss', 'loc_loss', 'iou_loss', 'iou_reg_loss'):
            values = [entry[key] for entry in det_logs if key in entry]
            if values:
                logs[0][key] = sum(values)
        for name, loss, details in [('terrain', terrain_loss, terrain_logs), ('occ', occ_loss, occ_logs)]:
            logs.append({**details, 'task': [name], 'loss': loss.detach(), **weights.get(name, {})})
        return total + zero_loss, logs

    def ExportPredictions(self, outputs, batch_dict):
        detections = self.detection_head.predict(batch_dict, outputs['det'], self.detection_post_processing)
        occ_frames = self.occ_head.ExportPredictions(outputs['occ'], batch_dict)
        result = {}
        for index, token in enumerate(batch_dict['token']):
            result[token] = {
                'terrain': {
                    'height': outputs['terrain']['terrain_height_pred'][index].detach().cpu(),
                    'xy_range': list(self.config.point_cloud_range[i] for i in [0, 1, 3, 4]),
                    'resolution': float(self.config.voxel_size[0]) * 4,
                },
                'occ': occ_frames[index],
            }
            for output_name in ('validity_logits', 'validity_probability', 'validity_mask', 'height_masked'):
                key = 'terrain_' + output_name
                if key in outputs['terrain']:
                    result[token]['terrain'][output_name] = outputs['terrain'][key][index].detach().cpu()
        for output in detections:
            result[output['token']]['det'] = {
                key: output[key].detach().cpu()
                for key in ('box3d_lidar', 'scores', 'label_preds')
            }
        return result

    def forward(self, batch_dict):
        outputs = self.ForwardNetwork(batch_dict)
        if self.training:
            return self.ComputeLoss(outputs, batch_dict)
        return self.ExportPredictions(outputs, batch_dict)

    def _BuildEncoderStage(self, in_channels: int, out_channels: int, num_blocks: int, stage_idx: int) -> nn.Sequential:
        stage_layers = [SparseConvBNReLU(in_channels, in_channels, kernel_size=2, stride=2, conv_type='spconv', indice_key=f'spconv{stage_idx}')]
        for block_idx in range(num_blocks):
            if block_idx == 0 and in_channels != out_channels:
                stage_layers.append(SparseBasicBlock(in_channels, out_channels, indice_key=f'subm{stage_idx}', downsample=SparseConvBNReLU(in_channels, out_channels, kernel_size=1, conv_type='subm', indice_key=f'subm{stage_idx}', b_activate=False)))
            else:
                stage_layers.append(SparseBasicBlock(out_channels, out_channels, indice_key=f'subm{stage_idx}'))
        return nn.Sequential(*stage_layers)
