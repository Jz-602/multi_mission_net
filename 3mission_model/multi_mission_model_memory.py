from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn.functional as F
from easydict import EasyDict
from spconv.pytorch import SparseConvTensor, SparseModule
from torch import Tensor, nn
from multi_mission_model import MultiMissionModel as SingleFrameModel


X3_STRIDE = 8
MEMORY_DISTANCE_MARGIN = 1.0
MEMORY_ALPHA_NEAR = 0.7
MEMORY_ALPHA_FAR = 0.2
MEMORY_ALPHA_EQUAL = 0.5
MEMORY_ATTENTION_CHUNK_SIZE = 4096



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


def EncodeSparseCoords(coords: Tensor, spatial_shape: Any) -> Tensor:
    ## 把 spconv [batch, z, y, x] 坐标编码成一维 key，用于稀疏坐标匹配。
    coords = coords.to(torch.long)
    spatial_shape_tensor = coords.new_tensor(spatial_shape, dtype=torch.long)
    z_size, y_size, x_size = spatial_shape_tensor[0], spatial_shape_tensor[1], spatial_shape_tensor[2]
    return (((coords[:, 0] * z_size + coords[:, 1]) * y_size + coords[:, 2]) * x_size + coords[:, 3])


class VoxelFourierPositionEncoding(nn.Module):
    """
    给 x3 voxel center 加连续 3D 位置编码。

    xyz: shape [N, 3]，当前帧增强后 LiDAR/车体坐标系下的 voxel center，单位 m。
    point_cloud_range: shape [6]，[x_min, y_min, z_min, x_max, y_max, z_max]。
    """

    def __init__(
        self,
        embed_channels: int,
        num_frequencies: int = 4,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        raw_channels = 3 + 3 * 2 * num_frequencies
        if hidden_channels is None:
            hidden_channels = max(embed_channels, raw_channels)

        self.proj = nn.Sequential(
            nn.Linear(raw_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, embed_channels),
        )
        freq_bands = torch.pow(
            torch.tensor(2.0, dtype=torch.float32),
            torch.arange(num_frequencies, dtype=torch.float32),
        ) * math.pi
        self.register_buffer("freq_bands", freq_bands, persistent=False)

    def forward(self, xyz: Tensor, point_cloud_range: Tensor) -> Tensor:
        point_cloud_range = point_cloud_range.to(device=xyz.device, dtype=xyz.dtype)
        point_cloud_min = point_cloud_range[:3]
        point_cloud_max = point_cloud_range[3:]
        point_cloud_extent = (point_cloud_max - point_cloud_min).clamp_min(1e-3)

        xyz_normalized = (xyz - point_cloud_min) / point_cloud_extent
        xyz_normalized = xyz_normalized * 2.0 - 1.0

        embed_parts = [xyz_normalized]
        freq_bands = self.freq_bands.to(device=xyz.device, dtype=xyz.dtype)
        for freq in freq_bands:
            embed_parts.append(torch.sin(xyz_normalized * freq))
            embed_parts.append(torch.cos(xyz_normalized * freq))

        pos_raw = torch.cat(embed_parts, dim=1)
        return self.proj(pos_raw)


class X3LocalMemoryCrossAttention(nn.Module):
    """
    x3 当前 voxel -> aligned memory voxel 的局部 topk cross-attention。

    current_coords: shape [N, 4]，[batch_idx, z_idx, y_idx, x_idx]，当前帧增强后
      LiDAR/车体坐标系下的 x3 sparse index。
    memory_coords: shape [M, 4]，已经 pose 对齐到当前帧增强后 LiDAR/车体坐标系下的 x3 sparse index。
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        topk: int = 16,
        coord_radius: Tuple[int, int, int] = (1, 2, 2),
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        cosine_weight: float = 1.0,
        coord_weight: float = 0.2,
        b_use_selection_score_bias: bool = True,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(
                f"channels must be divisible by num_heads. "
                f"Got channels={channels}, num_heads={num_heads}."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.topk = topk
        self.coord_radius_zyx = tuple(int(radius) for radius in coord_radius)
        self.scale = self.head_dim ** -0.5
        self.cosine_weight = cosine_weight
        self.coord_weight = coord_weight
        self.b_use_selection_score_bias = b_use_selection_score_bias

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        relative_hidden_channels = max(channels // 4, 16)
        self.relative_position_bias = nn.Sequential(
            nn.Linear(3, relative_hidden_channels),
            nn.GELU(),
            nn.Linear(relative_hidden_channels, num_heads),
        )
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        coord_radius_xyz = torch.tensor(
            [self.coord_radius_zyx[2], self.coord_radius_zyx[1], self.coord_radius_zyx[0]],
            dtype=torch.float32,
        )
        self.register_buffer("coord_radius_xyz_tensor", coord_radius_xyz, persistent=False)

        offset_z_values = torch.arange(-self.coord_radius_zyx[0], self.coord_radius_zyx[0] + 1, dtype=torch.long)
        offset_y_values = torch.arange(-self.coord_radius_zyx[1], self.coord_radius_zyx[1] + 1, dtype=torch.long)
        offset_x_values = torch.arange(-self.coord_radius_zyx[2], self.coord_radius_zyx[2] + 1, dtype=torch.long)
        offset_z, offset_y, offset_x = torch.meshgrid(
            offset_z_values,
            offset_y_values,
            offset_x_values,
            indexing="ij",
        )
        neighbor_offsets_zyx = torch.stack(
            (
                offset_z.reshape(-1),
                offset_y.reshape(-1),
                offset_x.reshape(-1),
            ),
            dim=1,
        )
        self.register_buffer("neighbor_offsets_zyx", neighbor_offsets_zyx, persistent=False)

    def BuildCandidateIndices(
        self,
        current_coords: Tensor,
        memory_coords: Tensor,
        spatial_shape: Any,
    ) -> Tensor:
        current_coords = current_coords.to(torch.long)
        memory_coords = memory_coords.to(torch.long)

        num_current = current_coords.shape[0]
        num_memory = memory_coords.shape[0]
        num_offsets = self.neighbor_offsets_zyx.shape[0]
        candidate_memory_indices = current_coords.new_full(
            (num_current, num_offsets),
            -1,
            dtype=torch.long,
        )
        if num_current == 0 or num_memory == 0:
            return candidate_memory_indices

        spatial_shape_tensor = current_coords.new_tensor(spatial_shape, dtype=torch.long)
        z_size, y_size, x_size = spatial_shape_tensor[0], spatial_shape_tensor[1], spatial_shape_tensor[2]
        memory_keys = EncodeSparseCoords(memory_coords, spatial_shape)
        sorted_memory_keys, sorted_order = torch.sort(memory_keys)

        offsets = self.neighbor_offsets_zyx.to(device=current_coords.device)
        candidate_coords = current_coords[:, None, :].repeat(1, num_offsets, 1)
        candidate_coords[:, :, 1:4] = candidate_coords[:, :, 1:4] + offsets[None, :, :]
        b_valid_candidate = (
            (candidate_coords[:, :, 1] >= 0)
            & (candidate_coords[:, :, 1] < z_size)
            & (candidate_coords[:, :, 2] >= 0)
            & (candidate_coords[:, :, 2] < y_size)
            & (candidate_coords[:, :, 3] >= 0)
            & (candidate_coords[:, :, 3] < x_size)
        )

        candidate_keys = EncodeSparseCoords(candidate_coords.reshape(-1, 4), spatial_shape)
        insert_pos = torch.searchsorted(sorted_memory_keys, candidate_keys)
        b_inside = insert_pos < sorted_memory_keys.shape[0]
        safe_insert_pos = insert_pos.clamp(max=sorted_memory_keys.shape[0] - 1)
        b_matched = (
            b_inside
            & (sorted_memory_keys[safe_insert_pos] == candidate_keys)
            & b_valid_candidate.reshape(-1)
        )

        flat_candidate_memory_indices = candidate_memory_indices.reshape(-1)
        flat_candidate_memory_indices[b_matched] = sorted_order[safe_insert_pos[b_matched]]
        return flat_candidate_memory_indices.reshape(num_current, num_offsets)

    def SelectTopKMemory(
        self,
        query_features: Tensor,
        memory_features: Tensor,
        candidate_memory_indices: Tensor,
        x3_voxel_size: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        num_current, num_candidates = candidate_memory_indices.shape
        if num_current == 0 or memory_features.shape[0] == 0:
            empty_indices = candidate_memory_indices.new_full((num_current, 0), -1)
            empty_mask = torch.zeros((num_current, 0), device=query_features.device, dtype=torch.bool)
            empty_score = query_features.new_zeros((num_current, 0))
            return empty_indices, empty_mask, empty_score

        b_candidate_valid = candidate_memory_indices >= 0
        safe_candidate_indices = candidate_memory_indices.clamp_min(0)
        candidate_memory_features = memory_features[safe_candidate_indices]

        query_norm = F.normalize(query_features, dim=-1, eps=1e-6)
        memory_norm = F.normalize(candidate_memory_features, dim=-1, eps=1e-6)
        cosine_score = (query_norm[:, None, :] * memory_norm).sum(dim=-1)

        offsets_xyz = self.neighbor_offsets_zyx[:, [2, 1, 0]].to(
            device=query_features.device,
            dtype=query_features.dtype,
        )
        x3_voxel_size = x3_voxel_size.to(device=query_features.device, dtype=query_features.dtype)
        coord_distance = torch.linalg.norm(offsets_xyz * x3_voxel_size[None, :], dim=1)
        coord_distance = coord_distance[None, :].expand(num_current, -1)
        coord_radius_xyz = self.coord_radius_xyz_tensor.to(device=query_features.device, dtype=query_features.dtype)
        coord_normalizer = torch.linalg.norm(x3_voxel_size * coord_radius_xyz.clamp_min(1.0)).clamp_min(1e-3)
        coord_distance = coord_distance / coord_normalizer

        selection_score = self.cosine_weight * cosine_score - self.coord_weight * coord_distance
        selection_score = selection_score.masked_fill(~b_candidate_valid, -10000.0)
        k_top = min(self.topk, num_candidates)
        topk_score, topk_candidate_pos = torch.topk(selection_score, k=k_top, dim=1)
        topk_memory_indices = candidate_memory_indices.gather(1, topk_candidate_pos)
        b_topk_valid = topk_memory_indices >= 0
        return topk_memory_indices, b_topk_valid, topk_score

    def forward(
        self,
        query_features: Tensor,
        memory_features: Tensor,
        current_coords: Tensor,
        memory_coords: Tensor,
        current_xyz: Tensor,
        memory_xyz: Tensor,
        current_pos: Tensor,
        memory_pos: Tensor,
        spatial_shape: Any,
        x3_voxel_size: Tensor,
    ) -> Tensor:
        if query_features.shape[0] == 0 or memory_features.shape[0] == 0:
            return query_features.new_zeros(query_features.shape)

        candidate_memory_indices = self.BuildCandidateIndices(
            current_coords=current_coords,
            memory_coords=memory_coords,
            spatial_shape=spatial_shape,
        )
        num_current = query_features.shape[0]
        out = query_features.new_zeros(query_features.shape)
        x3_voxel_size = x3_voxel_size.to(device=query_features.device, dtype=query_features.dtype)
        coord_radius_xyz = self.coord_radius_xyz_tensor.to(device=query_features.device, dtype=query_features.dtype)
        relative_scale = (x3_voxel_size * coord_radius_xyz.clamp_min(1.0)).view(1, 1, 3).clamp_min(1e-3)

        for start_idx in range(0, num_current, MEMORY_ATTENTION_CHUNK_SIZE):
            end_idx = min(start_idx + MEMORY_ATTENTION_CHUNK_SIZE, num_current)
            topk_memory_indices, b_topk_valid, topk_score = self.SelectTopKMemory(
                query_features=query_features[start_idx:end_idx],
                memory_features=memory_features,
                candidate_memory_indices=candidate_memory_indices[start_idx:end_idx],
                x3_voxel_size=x3_voxel_size,
            )
            if topk_memory_indices.shape[1] == 0:
                continue

            safe_topk_indices = topk_memory_indices.clamp_min(0)
            topk_memory_features = memory_features[safe_topk_indices]
            topk_memory_pos = memory_pos[safe_topk_indices]
            topk_memory_xyz = memory_xyz[safe_topk_indices]

            chunk_query_features = query_features[start_idx:end_idx]
            chunk_current_pos = current_pos[start_idx:end_idx]
            chunk_current_xyz = current_xyz[start_idx:end_idx]
            chunk_size = chunk_query_features.shape[0]
            k_top = topk_memory_indices.shape[1]

            query_with_pos = chunk_query_features + chunk_current_pos
            memory_with_pos = topk_memory_features + topk_memory_pos
            q = self.q_proj(query_with_pos)
            k = self.k_proj(memory_with_pos)
            v = self.v_proj(topk_memory_features)

            q = q.view(chunk_size, self.num_heads, self.head_dim).unsqueeze(2)
            k = k.view(chunk_size, k_top, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(chunk_size, k_top, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            attn_logits = (q * k).sum(dim=-1) * self.scale
            relative_xyz = (topk_memory_xyz - chunk_current_xyz[:, None, :]) / relative_scale
            relative_bias = self.relative_position_bias(relative_xyz).permute(0, 2, 1)
            attn_logits = attn_logits + relative_bias
            if self.b_use_selection_score_bias:
                attn_logits = attn_logits + topk_score.detach().clamp(-5.0, 5.0).unsqueeze(1)

            b_attn_valid = b_topk_valid.unsqueeze(1)
            attn_logits = attn_logits.masked_fill(~b_attn_valid, -10000.0)
            attn = torch.softmax(attn_logits, dim=-1)
            attn = attn * b_attn_valid.to(dtype=attn.dtype)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attn = self.attn_dropout(attn)

            context = (attn.unsqueeze(-1) * v).sum(dim=2)
            context = context.reshape(chunk_size, self.channels)
            chunk_out = self.out_proj(context)
            chunk_out = self.proj_dropout(chunk_out)
            b_has_memory = b_topk_valid.any(dim=1, keepdim=True)
            out[start_idx:end_idx] = chunk_out * b_has_memory.to(dtype=chunk_out.dtype)

        return out


class X3MemoryCrossAttentionTransformer(nn.Module):
    """
    x3 memory cross-attention transformer。

    x3_features: shape [N, C3]。
    x3_coords: shape [N, 4]，当前帧增强后 LiDAR/车体坐标系下的 x3 sparse index。
    aligned_memory["coords"]: shape [M, 4]，已经对齐到当前帧增强后 LiDAR/车体坐标系。
    aligned_memory["feats"]: shape [M, C3]。
    feature_stride 默认 8；X4 调用显式传 16，并使用对应通道数及栅格大小。
    """

    def __init__(
        self,
        channels: int,
        point_cloud_range: Any,
        voxel_size: Any,
        depth: int = 2,
        num_heads: int = 4,
        topk: int = 16,
        coord_radius: Tuple[int, int, int] = (1, 2, 2),
        ffn_ratio: float = 2.0,
        pos_num_frequencies: int = 4,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        layer_scale_init: float = 1e-2,
        cosine_weight: float = 1.0,
        coord_weight: float = 0.2,
        feature_stride: int = X3_STRIDE,
    ) -> None:
        super().__init__()

        point_cloud_range_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)
        x3_voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32) * feature_stride
        self.register_buffer("point_cloud_range_tensor", point_cloud_range_tensor, persistent=False)
        self.register_buffer("x3_voxel_size_tensor", x3_voxel_size_tensor, persistent=False)

        self.position_encoding = VoxelFourierPositionEncoding(
            embed_channels=channels,
            num_frequencies=pos_num_frequencies,
        )
        hidden_channels = int(channels * ffn_ratio)
        self.norm_query_list = nn.ModuleList([nn.LayerNorm(channels) for _ in range(depth)])
        self.norm_memory_list = nn.ModuleList([nn.LayerNorm(channels) for _ in range(depth)])
        self.cross_attention_list = nn.ModuleList([
            X3LocalMemoryCrossAttention(
                channels=channels,
                num_heads=num_heads,
                topk=topk,
                coord_radius=coord_radius,
                attn_dropout=attn_dropout,
                proj_dropout=ffn_dropout,
                cosine_weight=cosine_weight,
                coord_weight=coord_weight,
            )
            for _ in range(depth)
        ])
        self.norm_ffn_list = nn.ModuleList([nn.LayerNorm(channels) for _ in range(depth)])
        self.ffn_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(ffn_dropout),
                nn.Linear(hidden_channels, channels),
                nn.Dropout(ffn_dropout),
            )
            for _ in range(depth)
        ])
        self.attn_layer_scale = nn.Parameter(torch.ones(depth, channels) * layer_scale_init)
        self.ffn_layer_scale = nn.Parameter(torch.ones(depth, channels) * layer_scale_init)

    def CoordsToXYZ(self, coords: Tensor, dtype: torch.dtype) -> Tensor:
        ## x3 sparse index [batch, z, y, x] -> 当前帧增强后 LiDAR/车体坐标系 voxel center xyz。
        coords_xyz = coords[:, [3, 2, 1]].to(dtype=dtype)
        point_cloud_min = self.point_cloud_range_tensor[:3].to(device=coords.device, dtype=dtype)
        x3_voxel_size = self.x3_voxel_size_tensor.to(device=coords.device, dtype=dtype)
        return point_cloud_min + (coords_xyz + 0.5) * x3_voxel_size

    def forward(
        self,
        x3_features: Tensor,
        x3_coords: Tensor,
        aligned_memory: Optional[Dict[str, Tensor]],
        spatial_shape: Any,
    ) -> Tensor:
        if aligned_memory is None or aligned_memory["coords"].shape[0] == 0:
            return x3_features

        memory_features = aligned_memory["feats"].to(device=x3_features.device, dtype=x3_features.dtype)
        memory_coords = aligned_memory["coords"].to(device=x3_features.device, dtype=torch.long)
        if memory_features.shape[0] == 0:
            return x3_features

        x3_coords = x3_coords.to(device=x3_features.device, dtype=torch.long)
        current_xyz = self.CoordsToXYZ(x3_coords, dtype=x3_features.dtype)
        memory_xyz = self.CoordsToXYZ(memory_coords, dtype=x3_features.dtype)
        point_cloud_range = self.point_cloud_range_tensor.to(device=x3_features.device, dtype=x3_features.dtype)
        x3_voxel_size = self.x3_voxel_size_tensor.to(device=x3_features.device, dtype=x3_features.dtype)
        current_pos = self.position_encoding(current_xyz, point_cloud_range)
        memory_pos = self.position_encoding(memory_xyz, point_cloud_range)

        x = x3_features
        for layer_idx, (norm_query, norm_memory, cross_attention, norm_ffn, ffn) in enumerate(zip(
            self.norm_query_list,
            self.norm_memory_list,
            self.cross_attention_list,
            self.norm_ffn_list,
            self.ffn_list,
        )):
            attn_delta = cross_attention(
                query_features=norm_query(x),
                memory_features=norm_memory(memory_features),
                current_coords=x3_coords,
                memory_coords=memory_coords,
                current_xyz=current_xyz,
                memory_xyz=memory_xyz,
                current_pos=current_pos,
                memory_pos=memory_pos,
                spatial_shape=spatial_shape,
                x3_voxel_size=x3_voxel_size,
            )
            x = x + self.attn_layer_scale[layer_idx].view(1, -1) * attn_delta
            x = x + self.ffn_layer_scale[layer_idx].view(1, -1) * ffn(norm_ffn(x))
        return x


class MultiMissionModel(SingleFrameModel):
    """完整融合 x3 坐标和特征继续编码 x4，三任务头与单帧共用。"""

    def __init__(self, config):
        super().__init__(config)
        self.x3_memory_transformer = X3MemoryCrossAttentionTransformer(
            channels=self.config.encoder_channels[2],
            point_cloud_range=self.config.point_cloud_range,
            voxel_size=self.config.voxel_size,
            depth=int(self.config.memory_transformer_depth),
            num_heads=int(self.config.memory_num_heads),
            topk=int(self.config.memory_topk),
            coord_radius=self.config.memory_coord_radius,
        )

    def ForwardNetwork(self, batch_dict, memory=None, b_memory_only=False):
        x2, x3 = self.EncodeCurrent(batch_dict)
        aligned_memory = self.AlignMemoryToCurrentX3(batch_dict, memory, x3)
        if aligned_memory is not None and aligned_memory['coords'].shape[0] > 0:
            features = self.x3_memory_transformer(
                x3_features=x3.features, x3_coords=x3.indices,
                aligned_memory=aligned_memory, spatial_shape=x3.spatial_shape,
            )
            x3 = ReplaceFeature(x3, features)
        else:
            # 第一帧/全部历史出界时，保留 attention 参数的零梯度连接。
            zero = sum(param.reshape(-1)[0] * 0 for param in self.x3_memory_transformer.parameters())
            x3 = ReplaceFeature(x3, x3.features + zero)
        memory_next = self.MemoryUpdate(x3, aligned_memory, x3.batch_size)
        if b_memory_only:
            return {'memory': memory_next}
        # 不回采当前帧坐标；历史独有位置也进入 encoder_stage4。重新建立 indice 缓存。
        x3_fused = SparseConvTensor(
            memory_next['feats'], memory_next['coords'].int(),
            memory_next['spatial_shape'], memory_next['batch_size'],
        )
        x4 = self.encoder_stage4(x3_fused)
        result = self.ForwardHeads(x4, x3_fused, x2)
        result['memory'] = memory_next
        result['counts']['x3_current'] = len(x3.indices)
        return result

    def forward(self, batch_dict, memory=None, b_memory_only=False):
        outputs = self.ForwardNetwork(batch_dict, memory, b_memory_only)
        if b_memory_only:
            return outputs['memory']
        if self.training:
            loss, logs = self.ComputeLoss(outputs, batch_dict)
            return loss, logs, outputs['memory']
        return self.ExportPredictions(outputs, batch_dict)

    def AlignMemoryToCurrentX3(
        self,
        batch_dict: Dict[str, Any],
        memory: Optional[Dict[str, Any]],
        x3: SparseConvTensor,
        feature_stride: int = X3_STRIDE,
    ) -> Optional[Dict[str, Tensor]]:
        if memory is None or memory.get("b_empty", False):
            return None
        if "pose_to_prev" not in batch_dict:
            return None

        memory_feats = memory["feats"]
        memory_coords = memory["coords"].to(device=memory_feats.device)
        if memory_coords.shape[0] == 0:
            return None

        ## memory_coords: [M, 4]，上一帧增强坐标系下的 sparse index [batch, z, y, x]。
        ## feature_stride=8 为原 X3；X4 调用传 16，中心与量化必须使用同一物理尺度。
        ## pose_to_prev: [B, 4, 4]，上一帧增强后 LiDAR/车体坐标系 -> 当前帧增强后 LiDAR/车体坐标系。
        pose_to_prev = batch_dict["pose_to_prev"].to(device=memory_feats.device, dtype=memory_feats.dtype)
        point_cloud_range = memory_feats.new_tensor(self.config.point_cloud_range)
        voxel_size = memory_feats.new_tensor(self.config.voxel_size) * feature_stride
        point_cloud_min = point_cloud_range[:3]
        point_cloud_max = point_cloud_range[3:]

        batch_indices = memory_coords[:, 0].long()
        coords_xyz = memory_coords[:, [3, 2, 1]].to(device=memory_feats.device, dtype=memory_feats.dtype)
        memory_xyz = point_cloud_min + (coords_xyz + 0.5) * voxel_size
        memory_distances = memory.get("distances", None)
        if memory_distances is None:
            memory_distances = torch.linalg.norm(memory_xyz, dim=1)
        else:
            memory_distances = memory_distances.to(device=memory_feats.device, dtype=memory_feats.dtype).view(-1)

        memory_xyz_homo = torch.cat((memory_xyz, memory_xyz.new_ones((memory_xyz.shape[0], 1))), dim=1)
        current_xyz = torch.bmm(
            pose_to_prev[batch_indices],
            memory_xyz_homo.unsqueeze(-1),
        ).squeeze(-1)[:, :3]

        b_in_range = ((current_xyz >= point_cloud_min) & (current_xyz < point_cloud_max)).all(dim=1)
        if not torch.any(b_in_range):
            return {
                "coords": memory_coords.new_zeros((0, 4)),
                "feats": memory_feats.new_zeros((0, memory_feats.shape[1])),
                "distances": memory_feats.new_zeros((0,)),
            }

        current_xyz = current_xyz[b_in_range]
        current_feats = memory_feats[b_in_range]
        current_distances = memory_distances[b_in_range]
        current_batch_indices = batch_indices[b_in_range]
        current_coords_xyz = torch.floor((current_xyz - point_cloud_min) / voxel_size).long()
        spatial_shape = current_coords_xyz.new_tensor(x3.spatial_shape)
        b_valid_coords = (
            (current_coords_xyz >= 0)
            & (current_coords_xyz[:, [2, 1, 0]] < spatial_shape)
        ).all(dim=1)
        if not torch.any(b_valid_coords):
            return {
                "coords": memory_coords.new_zeros((0, 4)),
                "feats": memory_feats.new_zeros((0, memory_feats.shape[1])),
                "distances": memory_feats.new_zeros((0,)),
            }

        current_coords_xyz = current_coords_xyz[b_valid_coords]
        current_feats = current_feats[b_valid_coords]
        current_distances = current_distances[b_valid_coords]
        current_batch_indices = current_batch_indices[b_valid_coords]
        current_coords = torch.cat(
            (
                current_batch_indices.unsqueeze(1),
                current_coords_xyz[:, [2, 1, 0]],
            ),
            dim=1,
        ).to(dtype=memory_coords.dtype)

        unique_coords, inverse_indices = torch.unique(current_coords, dim=0, return_inverse=True)
        unique_feats = current_feats.new_zeros((unique_coords.shape[0], current_feats.shape[1]))
        unique_feats.index_add_(0, inverse_indices, current_feats)
        unique_count = current_feats.new_zeros((unique_coords.shape[0], 1))
        unique_count.index_add_(0, inverse_indices, current_feats.new_ones((current_feats.shape[0], 1)))
        unique_feats = unique_feats / unique_count.clamp_min(1.0)
        unique_distances = current_feats.new_zeros((unique_coords.shape[0],))
        unique_distances.index_add_(0, inverse_indices, current_distances)
        unique_distances = unique_distances / unique_count.squeeze(1).clamp_min(1.0)

        return {
            "coords": unique_coords,
            "feats": unique_feats,
            "distances": unique_distances,
        }

    def MemoryUpdate(
        self,
        x3: SparseConvTensor,
        aligned_memory: Optional[Dict[str, Tensor]],
        batch_size: int,
        feature_stride: int = X3_STRIDE,
    ) -> Dict[str, Any]:
        """
        用当前帧增强后的 x3 sparse 特征更新 rolling memory。

        x3.indices: shape [N, 4]，当前帧增强后 LiDAR/车体坐标系下的 [batch, z, y, x]。
        x3.features: shape [N, C3]，当前帧 x3 voxel feature。
        X4 复用时输入为当前 X4，并显式传 feature_stride=16。
        """
        current_feats = x3.features
        current_coords = x3.indices.to(device=current_feats.device)

        point_cloud_range = current_feats.new_tensor(self.config.point_cloud_range)
        voxel_size = current_feats.new_tensor(self.config.voxel_size) * feature_stride
        point_cloud_min = point_cloud_range[:3]
        current_coords_xyz = current_coords[:, [3, 2, 1]].to(device=current_feats.device, dtype=current_feats.dtype)
        current_xyz = point_cloud_min + (current_coords_xyz + 0.5) * voxel_size
        current_distances = torch.linalg.norm(current_xyz, dim=1)

        b_memory_empty = (
            aligned_memory is None
            or aligned_memory["coords"].shape[0] == 0
        )
        if b_memory_empty:
            return {
                "coords": current_coords,
                "feats": current_feats,
                "distances": current_distances,
                "spatial_shape": x3.spatial_shape,
                "batch_size": batch_size,
                "b_empty": current_coords.shape[0] == 0,
                "b_from_empty": True,
            }

        memory_coords = aligned_memory["coords"].to(device=current_coords.device, dtype=current_coords.dtype)
        memory_feats = aligned_memory["feats"].to(device=current_feats.device, dtype=current_feats.dtype)
        memory_distances = aligned_memory.get("distances", None)
        if memory_distances is None:
            memory_coords_xyz = memory_coords[:, [3, 2, 1]].to(device=current_feats.device, dtype=current_feats.dtype)
            memory_xyz = point_cloud_min + (memory_coords_xyz + 0.5) * voxel_size
            memory_distances = torch.linalg.norm(memory_xyz, dim=1)
        else:
            memory_distances = memory_distances.to(device=current_feats.device, dtype=current_feats.dtype).view(-1)

        current_keys = EncodeSparseCoords(current_coords, x3.spatial_shape)
        memory_keys = EncodeSparseCoords(memory_coords, x3.spatial_shape)
        all_keys = torch.cat((memory_keys, current_keys), dim=0)
        _, inverse_indices = torch.unique(all_keys, sorted=True, return_inverse=True)
        memory_inverse = inverse_indices[:memory_keys.shape[0]]
        current_inverse = inverse_indices[memory_keys.shape[0]:]
        num_memory = memory_coords.shape[0]
        num_next = int(inverse_indices.max().item()) + 1

        b_has_memory = torch.zeros((num_next,), device=current_feats.device, dtype=torch.bool)
        b_has_current = torch.zeros((num_next,), device=current_feats.device, dtype=torch.bool)
        b_has_memory[memory_inverse] = True
        b_has_current[current_inverse] = True

        next_coords = current_coords.new_zeros((num_next, 4))
        next_coords[memory_inverse] = memory_coords
        next_coords[current_inverse] = current_coords

        memory_feats_full = current_feats.new_zeros((num_next, current_feats.shape[1]))
        current_feats_full = current_feats.new_zeros((num_next, current_feats.shape[1]))
        memory_feats_full[memory_inverse] = memory_feats
        current_feats_full[current_inverse] = current_feats

        memory_distances_full = current_feats.new_full((num_next,), float("inf"))
        current_distances_full = current_feats.new_full((num_next,), float("inf"))
        memory_distances_full[memory_inverse] = memory_distances
        current_distances_full[current_inverse] = current_distances

        alpha = current_feats.new_zeros((num_next, 1))
        b_current_only = b_has_current & ~b_has_memory
        b_both = b_has_current & b_has_memory
        alpha[b_current_only] = 1.0
        if torch.any(b_both):
            current_distances_both = current_distances_full[b_both]
            memory_distances_both = memory_distances_full[b_both]
            alpha_both = current_distances_both.new_full(current_distances_both.shape, MEMORY_ALPHA_EQUAL)
            alpha_both = torch.where(
                current_distances_both + MEMORY_DISTANCE_MARGIN < memory_distances_both,
                current_distances_both.new_full(current_distances_both.shape, MEMORY_ALPHA_NEAR),
                alpha_both,
            )
            alpha_both = torch.where(
                current_distances_both > memory_distances_both + MEMORY_DISTANCE_MARGIN,
                current_distances_both.new_full(current_distances_both.shape, MEMORY_ALPHA_FAR),
                alpha_both,
            )
            alpha[b_both, 0] = alpha_both

        next_feats = (1.0 - alpha) * memory_feats_full + alpha * current_feats_full
        next_distances = torch.minimum(memory_distances_full, current_distances_full)

        return {
            "coords": next_coords,
            "feats": next_feats,
            "distances": next_distances,
            "spatial_shape": x3.spatial_shape,
            "batch_size": batch_size,
            "b_empty": num_next == 0,
            "b_from_empty": num_memory == 0,
        }
