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
from ASPPNeck import ASPPNeck
from CenterHead import CenterHead
from loss_segmentation import Losses


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
    ) -> None:
        super().__init__()

        point_cloud_range_tensor = torch.tensor(point_cloud_range, dtype=torch.float32)
        x3_voxel_size_tensor = torch.tensor(voxel_size, dtype=torch.float32) * X3_STRIDE
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
        self.log_vars = nn.Parameter(torch.zeros(len(self.task_names), dtype=torch.float32))

    def forward(self, task_losses: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Dict[str, Tensor]]]:
        total_loss: Optional[Tensor] = None
        log_dict: Dict[str, Dict[str, Tensor]] = {}

        for task_idx, task_name in enumerate(self.task_names):
            if task_name not in task_losses:
                continue

            task_loss = task_losses[task_name]
            log_var = self.log_vars[task_idx]
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
        self.x3_memory_transformer = X3MemoryCrossAttentionTransformer(
            channels=self.config.encoder_channels[2],
            point_cloud_range=self.config.point_cloud_range,
            voxel_size=self.config.voxel_size,
            depth=int(self.config.memory_transformer_depth),
            num_heads=int(self.config.memory_num_heads),
            topk=int(self.config.memory_topk),
            coord_radius=self.config.memory_coord_radius,
        )

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
        self.task_uncertainty_loss = TaskUncertaintyLoss(['seg', 'det'])

    def ForwardNetwork(
        self,
        batch_dict: Dict[str, Any],
        memory: Optional[Dict[str, Any]] = None,
        b_memory_only: bool = False,
    ) -> Dict[str, Any]:
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
        aligned_memory = self.AlignMemoryToCurrentX3(
            batch_dict=batch_dict,
            memory=memory,
            x3=x3,
        )
        if aligned_memory is not None and aligned_memory["coords"].shape[0] > 0:
            enhanced_features = self.x3_memory_transformer(
                x3_features=x3.features,
                x3_coords=x3.indices,
                aligned_memory=aligned_memory,
                spatial_shape=x3.spatial_shape,
            )
            x3 = ReplaceFeature(x3, enhanced_features)

        memory_next = self.MemoryUpdate(
            x3=x3,
            aligned_memory=aligned_memory,
            batch_size=batch_size,
        )
        if b_memory_only:
            return {"memory": memory_next}

        x4 = self.encoder_stage4(x3)

        detection_feature_map = self.detection_bev_projector(x4)
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
            'detection_feature_map': detection_feature_map,
            'detection_pred_dicts': detection_pred_dicts,
            'memory': memory_next,
        }

    def forward(
        self,
        batch_dict: Dict[str, Any],
        memory: Optional[Dict[str, Any]] = None,
        b_memory_only: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        forward_dict = self.ForwardNetwork(
            batch_dict=batch_dict,
            memory=memory,
            b_memory_only=b_memory_only,
        )
        memory_next = forward_dict["memory"]

        if b_memory_only:
            return memory_next

        point_logits = forward_dict["point_logits"]
        b_valid_mask = forward_dict["b_valid_mask"]
        point_tensor = forward_dict["point_tensor"]
        batch_indices = forward_dict["batch_indices"]
        detection_feature_map = forward_dict["detection_feature_map"]
        detection_pred_dicts = forward_dict["detection_pred_dicts"]

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
            total_loss, task_weight_logs = self.task_uncertainty_loss({
                'seg': seg_loss,
                'det': det_loss,
            })
            logs: List[Dict[str, Any]] = [{
                'task': ['seg'],
                'loss': seg_loss.detach(),
                **task_weight_logs['seg'],
                **loss_dict,
            }]
            if len(det_logs) > 0:
                det_logs[0].update(task_weight_logs['det'])
            logs.extend(det_logs)
            return total_loss, logs, memory_next

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

    def AlignMemoryToCurrentX3(
        self,
        batch_dict: Dict[str, Any],
        memory: Optional[Dict[str, Any]],
        x3: SparseConvTensor,
    ) -> Optional[Dict[str, Tensor]]:
        if memory is None or memory.get("b_empty", False):
            return None
        if "pose_to_prev" not in batch_dict:
            return None

        memory_feats = memory["feats"]
        memory_coords = memory["coords"].to(device=memory_feats.device)
        if memory_coords.shape[0] == 0:
            return None

        ## memory_coords: [M, 4]，上一帧增强坐标系下的 x3 sparse index [batch, z, y, x]。
        ## pose_to_prev: [B, 4, 4]，上一帧增强后 LiDAR/车体坐标系 -> 当前帧增强后 LiDAR/车体坐标系。
        pose_to_prev = batch_dict["pose_to_prev"].to(device=memory_feats.device, dtype=memory_feats.dtype)
        point_cloud_range = memory_feats.new_tensor(self.config.point_cloud_range)
        voxel_size = memory_feats.new_tensor(self.config.voxel_size) * X3_STRIDE
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
    ) -> Dict[str, Any]:
        """
        用当前帧增强后的 x3 sparse 特征更新 rolling memory。

        x3.indices: shape [N, 4]，当前帧增强后 LiDAR/车体坐标系下的 [batch, z, y, x]。
        x3.features: shape [N, C3]，当前帧 x3 voxel feature。
        """
        current_feats = x3.features
        current_coords = x3.indices.to(device=current_feats.device)

        point_cloud_range = current_feats.new_tensor(self.config.point_cloud_range)
        voxel_size = current_feats.new_tensor(self.config.voxel_size) * X3_STRIDE
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
