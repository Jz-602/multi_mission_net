from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from easydict import EasyDict

from loss_segmentation import Losses
import spconv.pytorch as spconv
from spconv.pytorch import SparseConvTensor

try:
    import flash_attn
except ImportError:
    flash_attn = None


def ReplaceFeature(x: SparseConvTensor, features: Tensor) -> SparseConvTensor:
    # 功能：兼容不同 spconv 版本，替换稀疏张量的特征。
    if hasattr(x, "replace_feature"):
        return x.replace_feature(features)
    x.features = features
    return x


def PadOrTrimFeatures(points: Tensor, in_channels: int) -> Tensor:
    # 功能：把输入点特征裁剪或补零到固定通道数。
    if points.shape[1] == in_channels:
        return points
    if points.shape[1] > in_channels:
        return points[:, :in_channels]
    pad = points.new_zeros((points.shape[0], in_channels - points.shape[1]))
    return torch.cat((points, pad), dim=1)


def DropPath(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    # 功能：实现 transformer block 中的随机深度残差丢弃。
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    return x.div(keep_prob) * mask


def RightShift(binary: Tensor, shift: int, axis: int = -1) -> Tensor:
    # 功能：对二进制 bit 张量沿指定维度右移并补零。
    if binary.shape[axis] <= shift:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -shift)
    return nn.functional.pad(binary[tuple(slicing)], (shift, 0), mode="constant", value=0)


def GrayToBinary(gray: Tensor, axis: int = -1) -> Tensor:
    # 功能：把 Gray code bit 张量转换成普通二进制 bit 张量。
    shift = 1 << (int(math.ceil(math.log2(gray.shape[axis]))) - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, RightShift(gray, shift, axis=axis))
        shift //= 2
    return gray


def HilbertEncode(grid_coord: Tensor, depth: int) -> Tensor:
    # 功能：把 3D voxel 网格坐标编码成 Hilbert 曲线序列码。
    num_dims = 3
    if grid_coord.shape[-1] != num_dims:
        raise ValueError(f"grid_coord must be [N, 3], but got {grid_coord.shape}.")
    if num_dims * depth > 63:
        raise ValueError(f"Hilbert depth={depth} exceeds int64 support.")

    bitpack_mask = 1 << torch.arange(0, 8, device=grid_coord.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)
    locs_uint8 = grid_coord.long().contiguous().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -depth:]
    )

    for bit in range(depth):
        for dim in range(num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], mask[:, None])
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(gray[:, dim, bit + 1 :], to_flip)
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    gray = gray.swapaxes(1, 2).reshape((-1, depth * num_dims))
    binary = GrayToBinary(gray)
    extra_dims = 64 - depth * num_dims
    padded = nn.functional.pad(binary, (extra_dims, 0), "constant", 0)
    packed_uint8 = (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask).sum(2).squeeze().type(torch.uint8)
    return packed_uint8.contiguous().view(torch.int64).reshape(-1)


def ZOrderEncode(grid_coord: Tensor, batch: Tensor, depth: int, order: str) -> Tensor:
    # 功能：按 z-order 或 Hilbert order 生成带 batch 信息的序列化编码。
    if order not in ("z", "z-trans", "hilbert", "hilbert-trans"):
        raise ValueError(f"Unsupported PTv3 serialization order={order}.")
    if order in ("z-trans", "hilbert-trans"):
        grid_coord = grid_coord[:, [1, 0, 2]]
    coord = grid_coord.long().clamp_min(0)
    if order in ("hilbert", "hilbert-trans"):
        code = HilbertEncode(coord, depth)
        code |= batch.long() << (3 * depth)
        return code

    x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
    code = torch.zeros_like(x, dtype=torch.long)
    for bit in range(depth):
        code |= ((x >> bit) & 1) << (3 * bit + 2)
        code |= ((y >> bit) & 1) << (3 * bit + 1)
        code |= ((z >> bit) & 1) << (3 * bit)
    code |= batch.long() << (3 * depth)
    return code


class Point(dict):
    def __getattr__(self, key: str) -> Any:
        # 功能：支持用 point.xxx 形式读取字典字段。
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        # 功能：支持用 point.xxx 形式写入字典字段。
        self[key] = value

    def serialization(self, order: Tuple[str, ...], shuffle_orders: bool) -> None:
        # 功能：为当前 voxel 点集生成多种空间序列化顺序。
        depth = int(self.grid_coord.max().item() + 1).bit_length()
        depth = max(depth, 1)
        if depth > 16:
            raise ValueError(
                f"PTv3 serialization supports depth <= 16, got {depth}. "
                "Increase voxel_size or reduce point_cloud_range."
            )
        code = torch.stack(
            [ZOrderEncode(self.grid_coord, self.batch, depth, order_name) for order_name in order],
            dim=0,
        )
        serialized_order = torch.argsort(code, dim=1)
        serialized_inverse = torch.zeros_like(serialized_order).scatter_(
            dim=1,
            index=serialized_order,
            src=torch.arange(code.shape[1], device=code.device).repeat(code.shape[0], 1),
        )
        if shuffle_orders:
            perm = torch.randperm(code.shape[0], device=code.device)
            code = code[perm]
            serialized_order = serialized_order[perm]
            serialized_inverse = serialized_inverse[perm]
        self.serialized_depth = depth
        self.serialized_code = code
        self.serialized_order = serialized_order
        self.serialized_inverse = serialized_inverse

    def sparsify(self, pad: int = 96) -> None:
        # 功能：把 Point 中的 voxel 特征转换成 spconv 稀疏张量。
        sparse_shape = (self.grid_coord.max(dim=0).values + pad).tolist()
        self.sparse_shape = sparse_shape
        self.sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat([self.batch.int().unsqueeze(1), self.grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=int(self.batch.max().item()) + 1,
        )


class PointBatchNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        # 功能：初始化作用在点/voxel 特征上的 BatchNorm。
        super().__init__()
        self.norm = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

    def forward(self, point: Point) -> Point:
        # 功能：对 point.feat 做 BatchNorm 并同步 sparse tensor 特征。
        point.feat = self.norm(point.feat)
        point.sparse_conv_feat = ReplaceFeature(point.sparse_conv_feat, point.feat)
        return point


class PointLayerNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        # 功能：初始化作用在点/voxel 特征上的 LayerNorm。
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, point: Point) -> Point:
        # 功能：对 point.feat 做 LayerNorm 并同步 sparse tensor 特征。
        point.feat = self.norm(point.feat)
        point.sparse_conv_feat = ReplaceFeature(point.sparse_conv_feat, point.feat)
        return point


class PointActivation(nn.Module):
    def __init__(self) -> None:
        # 功能：初始化点特征使用的 GELU 激活。
        super().__init__()
        self.act = nn.GELU()

    def forward(self, point: Point) -> Point:
        # 功能：对 point.feat 做 GELU 激活并同步 sparse tensor 特征。
        point.feat = self.act(point.feat)
        point.sparse_conv_feat = ReplaceFeature(point.sparse_conv_feat, point.feat)
        return point


class Embedding(nn.Module):
    def __init__(self, in_channels: int, embed_channels: int) -> None:
        # 功能：初始化 PTv3 输入 stem，将原始 voxel 特征映射到 embedding 通道。
        super().__init__()
        self.conv = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=5,
            padding=1,
            bias=False,
            indice_key="stem",
        )
        self.norm = nn.BatchNorm1d(embed_channels, eps=1e-3, momentum=0.01)
        self.act = nn.GELU()

    def forward(self, point: Point) -> Point:
        # 功能：用 SubMConv、BatchNorm 和 GELU 生成初始 voxel embedding。
        sparse = self.conv(point.sparse_conv_feat)
        feat = self.act(self.norm(sparse.features))
        point.feat = feat
        point.sparse_conv_feat = ReplaceFeature(sparse, feat)
        return point


class ConditionalPositionEncoding(nn.Module):
    def __init__(self, channels: int, indice_key: str) -> None:
        # 功能：初始化基于稀疏卷积的条件位置编码模块。
        super().__init__()
        self.conv = spconv.SubMConv3d(channels, channels, kernel_size=3, bias=True, indice_key=indice_key)
        self.linear = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, point: Point) -> Point:
        # 功能：用局部稀疏卷积为 voxel 特征加入位置相关信息。
        sparse = self.conv(point.sparse_conv_feat)
        feat = self.norm(self.linear(sparse.features))
        point.feat = feat
        point.sparse_conv_feat = ReplaceFeature(sparse, feat)
        return point


class MLP(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float, drop: float) -> None:
        # 功能：初始化 transformer block 内的前馈网络。
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        # 功能：对输入特征执行 Linear、GELU、Dropout、Linear 的前馈变换。
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SerializedAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        order_index: int,
        qkv_bias: bool,
        attn_drop: float,
        proj_drop: float,
        enable_flash: bool,
    ) -> None:
        # 功能：初始化按空间序列分块计算的 self-attention。
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}.")
        self.channels = channels
        self.num_heads = num_heads
        self.patch_size_max = int(patch_size)
        self.patch_size = int(patch_size)
        self.order_index = order_index
        self.scale = (channels // num_heads) ** -0.5
        self.enable_flash_config = bool(enable_flash)
        self.attn_drop_p = float(attn_drop)
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    @torch.no_grad()
    def get_padding_and_inverse(self, point: Point) -> Tuple[Tensor, Tensor, Tensor]:
        # 功能：为变长 batch 点序列生成 attention padding、反向索引和长度表。
        pad_key = f"pad_{self.patch_size}"
        unpad_key = f"unpad_{self.patch_size}"
        cu_key = f"cu_seqlens_{self.patch_size}"
        if pad_key in point and unpad_key in point and cu_key in point:
            return point[pad_key], point[unpad_key], point[cu_key]

        offset = point.offset
        bincount = torch.diff(offset, prepend=offset.new_zeros(1))
        bincount_pad = torch.div(bincount + self.patch_size - 1, self.patch_size, rounding_mode="trunc") * self.patch_size
        mask_pad = bincount > self.patch_size
        bincount_pad = torch.where(mask_pad, bincount_pad, bincount)
        offset_raw = nn.functional.pad(offset, (1, 0))
        offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
        pad = torch.arange(offset_pad[-1], device=offset.device)
        unpad = torch.arange(offset_raw[-1], device=offset.device)
        cu_seqlens = []

        for i in range(len(offset)):
            unpad[offset_raw[i] : offset_raw[i + 1]] += offset_pad[i] - offset_raw[i]
            if bincount[i] != bincount_pad[i]:
                tail = bincount[i] % self.patch_size
                pad[offset_pad[i + 1] - self.patch_size + tail : offset_pad[i + 1]] = pad[
                    offset_pad[i + 1] - 2 * self.patch_size + tail : offset_pad[i + 1] - self.patch_size
                ]
            pad[offset_pad[i] : offset_pad[i + 1]] -= offset_pad[i] - offset_raw[i]
            cu_seqlens.append(
                torch.arange(offset_pad[i], offset_pad[i + 1], step=self.patch_size, dtype=torch.int32, device=offset.device)
            )
        cu_seqlens = nn.functional.pad(torch.cat(cu_seqlens), (0, 1), value=offset_pad[-1])
        point[pad_key], point[unpad_key], point[cu_key] = pad, unpad, cu_seqlens
        return pad, unpad, cu_seqlens

    def forward(self, point: Point) -> Point:
        # 功能：按序列化顺序对 voxel 特征分 patch 做 self-attention。
        use_flash = self.enable_flash_config and flash_attn is not None and point.feat.is_cuda
        if use_flash:
            self.patch_size = self.patch_size_max
        else:
            min_count = int(torch.diff(point.offset, prepend=point.offset.new_zeros(1)).min().item())
            self.patch_size = max(1, min(min_count, self.patch_size_max))

        num_heads = self.num_heads
        patch_size = self.patch_size
        channels = self.channels
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]
        qkv = self.qkv(point.feat)[order]

        if use_flash:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, num_heads, channels // num_heads),
                cu_seqlens,
                max_seqlen=patch_size,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                softmax_scale=self.scale,
            ).reshape(-1, channels)
            feat = feat.to(qkv.dtype)
        else:
            q, k, v = qkv.reshape(-1, patch_size, 3, num_heads, channels // num_heads).permute(2, 0, 3, 1, 4).unbind(0)
            attn = (q.float() * self.scale) @ k.float().transpose(-2, -1)
            attn = torch.softmax(attn, dim=-1).to(qkv.dtype)
            attn = self.attn_drop(attn)
            feat = (attn @ v).transpose(1, 2).reshape(-1, channels)

        point.feat = self.proj_drop(self.proj(feat[inverse]))
        return point


class Block(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        patch_size: int,
        order_index: int,
        indice_key: str,
        mlp_ratio: float,
        qkv_bias: bool,
        drop_path: float,
        attn_drop: float,
        proj_drop: float,
        enable_flash: bool,
    ) -> None:
        # 功能：初始化 PTv3 基础 block，包括位置编码、attention 和 MLP。
        super().__init__()
        self.cpe = ConditionalPositionEncoding(channels, indice_key=indice_key)
        self.norm1 = PointLayerNorm(channels)
        self.attn = SerializedAttention(channels, num_heads, patch_size, order_index, qkv_bias, attn_drop, proj_drop, enable_flash)
        self.norm2 = PointLayerNorm(channels)
        self.mlp = MLP(channels, mlp_ratio=mlp_ratio, drop=proj_drop)
        self.drop_path = float(drop_path)

    def forward(self, point: Point) -> Point:
        # 功能：执行 CPE、序列化 self-attention、MLP 及对应残差连接。
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        point.sparse_conv_feat = ReplaceFeature(point.sparse_conv_feat, point.feat)

        shortcut = point.feat
        point = self.norm1(point)
        point = self.attn(point)
        point.feat = shortcut + DropPath(point.feat, self.drop_path, self.training)
        point.sparse_conv_feat = ReplaceFeature(point.sparse_conv_feat, point.feat)

        shortcut = point.feat
        point = self.norm2(point)
        point.feat = shortcut + DropPath(self.mlp(point.feat), self.drop_path, self.training)
        point.sparse_conv_feat = ReplaceFeature(point.sparse_conv_feat, point.feat)
        return point


def ScatterMean(src: Tensor, cluster: Tensor, num_clusters: int) -> Tensor:
    # 功能：按 cluster 对特征做均值聚合。
    out = src.new_zeros((num_clusters, src.shape[1]))
    out.index_add_(0, cluster, src)
    counts = torch.bincount(cluster, minlength=num_clusters).to(src.dtype).clamp_min(1)
    return out / counts.unsqueeze(1)


def ScatterMax(src: Tensor, cluster: Tensor, num_clusters: int) -> Tensor:
    # 功能：按 cluster 对特征做最大值聚合。
    out = src.new_full((num_clusters, src.shape[1]), -torch.inf)
    for cluster_idx in range(num_clusters):
        mask = cluster == cluster_idx
        out[cluster_idx] = src[mask].max(dim=0).values
    return torch.nan_to_num(out, neginf=0.0)


class SerializedPooling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        reduce: str,
        shuffle_orders: bool,
    ) -> None:
        # 功能：初始化基于序列化 cluster 的下采样 pooling 模块。
        super().__init__()
        self.stride = int(stride)
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.proj = nn.Linear(in_channels, out_channels)
        self.norm = PointBatchNorm(out_channels)
        self.act = PointActivation()

    def reduce_feat(self, src: Tensor, cluster: Tensor, num_clusters: int) -> Tensor:
        # 功能：根据配置对同一 cluster 内的 voxel 特征做 mean 或 max 聚合。
        if self.reduce == "mean":
            return ScatterMean(src, cluster, num_clusters)
        if self.reduce == "max":
            if hasattr(torch.Tensor, "scatter_reduce_"):
                out = src.new_full((num_clusters, src.shape[1]), -torch.inf)
                index = cluster.unsqueeze(1).expand(-1, src.shape[1])
                out.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
                return torch.nan_to_num(out, neginf=0.0)
            return ScatterMax(src, cluster, num_clusters)
        raise ValueError(f"Unsupported reduce={self.reduce}.")

    def forward(self, point: Point) -> Point:
        # 功能：将高分辨率 voxel 聚合到低分辨率 voxel，并保存反向映射。
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0

        code = point.serialized_code >> (pooling_depth * 3)
        _, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
        _, indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        head_indices = indices[idx_ptr[:-1]]
        num_clusters = int(counts.numel())

        feat = self.reduce_feat(self.proj(point.feat), cluster, num_clusters)
        coord = ScatterMean(point.coord, cluster, num_clusters)
        grid_coord = point.grid_coord[head_indices] >> pooling_depth
        batch = point.batch[head_indices]
        offset = torch.bincount(batch.long(), minlength=len(point.offset)).cumsum(dim=0)

        pooled_code = code[:, head_indices]
        order = torch.argsort(pooled_code, dim=1)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(pooled_code.shape[1], device=pooled_code.device).repeat(pooled_code.shape[0], 1),
        )
        if self.shuffle_orders:
            perm = torch.randperm(pooled_code.shape[0], device=pooled_code.device)
            pooled_code = pooled_code[perm]
            order = order[perm]
            inverse = inverse[perm]

        new_point = Point(
            feat=feat,
            coord=coord,
            grid_coord=grid_coord,
            batch=batch,
            offset=offset,
            serialized_code=pooled_code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            pooling_inverse=cluster,
            pooling_parent=point,
        )
        new_point.sparsify()
        new_point = self.norm(new_point)
        new_point = self.act(new_point)
        return new_point


class SerializedUnpooling(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        # 功能：初始化 decoder 中的上采样和 skip 特征投影模块。
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.GELU(),
        )

    def forward(self, point: Point) -> Point:
        # 功能：利用 pooling_inverse 把低分辨率特征恢复到上一层 voxel，并融合 skip 特征。
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        low_feat = self.proj(point.feat)
        parent.feat = self.proj_skip(parent.feat) + low_feat[inverse]
        parent.sparse_conv_feat = ReplaceFeature(parent.sparse_conv_feat, parent.feat)
        return parent


def BuildPointBatch(
    points: Tensor,
    voxel_size: Any,
    point_cloud_range: Any,
    in_channels: int,
) -> Tuple[Point, Tensor, Tensor, Tensor, Tensor]:
    # 功能：把 batch 点云过滤、体素化，并构建 PTv3 使用的 Point 对象。
    if points.ndim != 2:
        raise ValueError(f"points must be [N, C], but got {points.shape}.")

    point_features_all = points[:, 1:]
    point_coords_xyz_all = point_features_all[:, :3]
    voxel_size_xyz = point_features_all.new_tensor(voxel_size)
    point_cloud_range_xyz = point_features_all.new_tensor(point_cloud_range)
    point_cloud_min_xyz = point_cloud_range_xyz[:3]
    point_cloud_max_xyz = point_cloud_range_xyz[3:]

    b_valid_mask = ((point_coords_xyz_all >= point_cloud_min_xyz) & (point_coords_xyz_all < point_cloud_max_xyz)).all(dim=1)
    if not torch.any(b_valid_mask):
        raise ValueError(f"No points fall inside point_cloud_range={point_cloud_range}.")

    valid_points = points[b_valid_mask]
    batch_indices = valid_points[:, 0].long()
    point_features = PadOrTrimFeatures(valid_points[:, 1:], in_channels)
    point_coords_xyz = valid_points[:, 1:4]
    grid_coord = torch.floor((point_coords_xyz - point_cloud_min_xyz) / voxel_size_xyz).int()
    coords_with_batch = torch.cat([batch_indices.int().unsqueeze(1), grid_coord], dim=1)

    unique_coords, point2voxel_map = torch.unique(coords_with_batch, sorted=True, return_inverse=True, dim=0)
    num_voxels = unique_coords.shape[0]
    feat = point_features.new_zeros((num_voxels, point_features.shape[1]))
    coord = point_coords_xyz.new_zeros((num_voxels, 3))
    feat.index_add_(0, point2voxel_map, point_features)
    coord.index_add_(0, point2voxel_map, point_coords_xyz)
    counts = torch.bincount(point2voxel_map, minlength=num_voxels).to(point_features.dtype).clamp_min(1)
    feat = feat / counts.unsqueeze(1)
    coord = coord / counts.unsqueeze(1)

    voxel_batch = unique_coords[:, 0].long()
    batch_size = int(points[:, 0].max().item()) + 1
    point = Point(
        feat=feat,
        coord=coord,
        grid_coord=unique_coords[:, 1:4].int(),
        batch=voxel_batch,
        offset=torch.bincount(voxel_batch.long(), minlength=batch_size).cumsum(dim=0),
    )
    return point, point2voxel_map.long(), b_valid_mask, valid_points, batch_indices


class PTv3WaymoSegmentor(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        # 功能：初始化 Waymo 点云语义分割模型、分类头和 loss。
        super().__init__()
        if not isinstance(config, EasyDict):
            config = EasyDict(config)
        self.config = config
        self.order = tuple(getattr(config, "ptv3_order", ("z", "z-trans")))
        self.shuffle_orders = bool(getattr(config, "ptv3_shuffle_orders", True))
        self.enable_flash = bool(getattr(config, "ptv3_enable_flash", False))

        enc_channels = tuple(getattr(config, "ptv3_enc_channels", (32, 64, 128, 256, 512)))
        enc_depths = tuple(getattr(config, "ptv3_enc_depths", (2, 2, 2, 6, 2)))
        enc_heads = tuple(getattr(config, "ptv3_enc_num_head", (2, 4, 8, 16, 32)))
        dec_channels = tuple(getattr(config, "ptv3_dec_channels", (64, 64, 128, 256)))
        dec_depths = tuple(getattr(config, "ptv3_dec_depths", (2, 2, 2, 2)))
        dec_heads = tuple(getattr(config, "ptv3_dec_num_head", (4, 4, 8, 16)))
        stride = tuple(getattr(config, "ptv3_stride", (2, 2, 2, 2)))
        default_patch = 1024 if self.enable_flash else 128
        enc_patch = tuple(getattr(config, "ptv3_enc_patch_size", (default_patch,) * len(enc_channels)))
        dec_patch = tuple(getattr(config, "ptv3_dec_patch_size", (default_patch,) * len(dec_channels)))
        mlp_ratio = float(getattr(config, "ptv3_mlp_ratio", 4.0))
        drop_path = float(getattr(config, "ptv3_drop_path", 0.3))
        attn_drop = float(getattr(config, "ptv3_attn_drop", 0.0))
        proj_drop = float(getattr(config, "ptv3_proj_drop", 0.0))
        qkv_bias = bool(getattr(config, "ptv3_qkv_bias", True))
        reduce = str(getattr(config, "ptv3_pool_reduce", "max"))

        self.embedding = Embedding(config.in_channels, enc_channels[0])

        enc_drop_path = torch.linspace(0, drop_path, sum(enc_depths)).tolist()
        encoder_stages = []
        cursor = 0
        for stage_idx, depth in enumerate(enc_depths):
            layers = []
            if stage_idx > 0:
                layers.append(SerializedPooling(enc_channels[stage_idx - 1], enc_channels[stage_idx], stride[stage_idx - 1], reduce, self.shuffle_orders))
            for block_idx in range(depth):
                layers.append(
                    Block(
                        channels=enc_channels[stage_idx],
                        num_heads=enc_heads[stage_idx],
                        patch_size=enc_patch[stage_idx],
                        order_index=block_idx % len(self.order),
                        indice_key=f"stage{stage_idx}",
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=enc_drop_path[cursor],
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        enable_flash=self.enable_flash,
                    )
                )
                cursor += 1
            encoder_stages.append(nn.Sequential(*layers))
        self.encoder_stage0 = encoder_stages[0]
        self.encoder_stage1 = encoder_stages[1]
        self.encoder_stage2 = encoder_stages[2]
        self.encoder_stage3 = encoder_stages[3]
        self.encoder_stage4 = encoder_stages[4]

        dec_drop_path = torch.linspace(0, drop_path, sum(dec_depths)).tolist()
        dec_channels_full = list(dec_channels) + [enc_channels[-1]]
        decoder_stages = []
        for stage_idx in reversed(range(len(enc_depths) - 1)):
            layers = []
            layers.append(SerializedUnpooling(dec_channels_full[stage_idx + 1], enc_channels[stage_idx], dec_channels_full[stage_idx]))
            stage_drop_path = dec_drop_path[sum(dec_depths[:stage_idx]) : sum(dec_depths[: stage_idx + 1])]
            stage_drop_path.reverse()
            for block_idx in range(dec_depths[stage_idx]):
                layers.append(
                    Block(
                        channels=dec_channels_full[stage_idx],
                        num_heads=dec_heads[stage_idx],
                        patch_size=dec_patch[stage_idx],
                        order_index=block_idx % len(self.order),
                        indice_key=f"stage{stage_idx}",
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=stage_drop_path[block_idx],
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        enable_flash=self.enable_flash,
                    )
                )
            decoder_stages.append(nn.Sequential(*layers))
        self.decoder_stage3 = decoder_stages[0]
        self.decoder_stage2 = decoder_stages[1]
        self.decoder_stage1 = decoder_stages[2]
        self.decoder_stage0 = decoder_stages[3]

        self.head = nn.Linear(dec_channels[0], self.config.num_classes)
        self.criterion_losses = self.BuildCriterionLosses()

    def BuildCriterionLosses(self) -> Losses:
        # 功能：根据配置构建语义分割训练损失组合。
        default_loss_types = ("DiceLossV1", "LovLoss", "CELoss")
        loss_types = tuple(getattr(self.config, "loss_types", ())) or default_loss_types
        loss_weights = tuple(getattr(self.config, "loss_weights", ())) or tuple(1.0 for _ in loss_types)
        if len(loss_types) != len(loss_weights):
            raise ValueError("LOSS_TYPES and LOSS_WEIGHTS must have the same length.")

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

    def ComputeLoss(self, batch_dict: Dict[str, Any], forward_dict: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]:
        # 功能：使用点级 logits 和语义标签计算训练 loss。
        if "semantic_label" not in batch_dict:
            raise KeyError("Training batch must contain key 'semantic_label'.")

        point_logits = forward_dict["point_logits"]
        point_tensor = forward_dict["point_tensor"]
        b_valid_mask = forward_dict["b_valid_mask"]
        batch_indices = forward_dict["batch_indices"]
        target = batch_dict["semantic_label"][b_valid_mask].to(point_logits.device).long()

        if point_logits.shape[0] != target.shape[0]:
            raise ValueError(
                "point logits count does not match semantic label count. "
                f"Got {point_logits.shape[0]} vs {target.shape[0]}."
            )

        coords_xyz = point_tensor[:, 1:4]
        _, point_count_per_batch = torch.unique_consecutive(batch_indices, return_counts=True)
        offset = point_count_per_batch.to(device=target.device, dtype=torch.long).cumsum(dim=0)
        total_loss, loss_dict = self.criterion_losses(point_logits, target, xyz=coords_xyz, offset=offset)
        return total_loss, loss_dict

    def ForwardNetwork(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        # 功能：执行点云体素化、PTv3 主干、语义分类头和 voxel 到点的 logits 还原。
        if not isinstance(batch_dict, dict):
            raise TypeError(f"batch_dict must be dict, but got {type(batch_dict)}")
        if "points" not in batch_dict:
            raise KeyError("Batch dict must contain key 'points'.")

        point_tensor_all = batch_dict["points"]
        if not isinstance(point_tensor_all, torch.Tensor):
            raise TypeError("Batch dict key 'points' must be a torch.Tensor.")

        point, point2voxel_map, b_valid_mask, point_tensor, batch_indices = BuildPointBatch(
            point_tensor_all,
            self.config.voxel_size,
            self.config.point_cloud_range,
            self.config.in_channels,
        )
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        x0 = self.embedding(point)
        x0 = self.encoder_stage0(x0)
        x1 = self.encoder_stage1(x0)
        x2 = self.encoder_stage2(x1)
        x3 = self.encoder_stage3(x2)
        x4 = self.encoder_stage4(x3)

        y3 = self.decoder_stage3(x4)
        y2 = self.decoder_stage2(y3)
        y1 = self.decoder_stage1(y2)
        y0 = self.decoder_stage0(y1)

        voxel_logits = self.head(y0.feat)
        point_logits = voxel_logits[point2voxel_map]
        _, point_count_per_batch = torch.unique_consecutive(batch_indices, return_counts=True)
        point_logits_list = [chunk.contiguous() for chunk in point_logits.split(point_count_per_batch.tolist(), dim=0)]

        forward_dict = {
            "voxel_logits": voxel_logits,
            "point_logits": point_logits,
            "point_logits_list": point_logits_list,
            "voxel_coords": torch.cat([y0.batch.int().unsqueeze(1), y0.grid_coord.int()], dim=1),
            "point2voxel_map": point2voxel_map,
            "b_valid_mask": b_valid_mask,
            "point_tensor": point_tensor,
            "batch_indices": batch_indices,
        }
        return forward_dict

    def forward(
        self,
        batch_dict: Dict[str, Any],
        b_return_details: bool = False,
    ) -> Union[Tensor, Dict[str, Any]]:
        # 功能：统一训练和推理入口，训练返回 loss，推理返回 logits 或详细结果。
        forward_dict = self.ForwardNetwork(batch_dict)

        if self.training:
            total_loss, loss_dict = self.ComputeLoss(batch_dict, forward_dict)
            logs = {"task": ["seg"], "loss": total_loss.detach(), **loss_dict}
            return total_loss, logs

        if b_return_details:
            return forward_dict

        point_logits = forward_dict["point_logits"]
        point_pred = point_logits.argmax(dim=1)
        point_tensor = forward_dict["point_tensor"]
        batch_indices = forward_dict["batch_indices"]

        if "token" in batch_dict:
            result_dict = {}
            for batch_idx, token in enumerate(batch_dict["token"]):
                b_mask = batch_indices == batch_idx
                result_dict[token] = {
                    "point_logits": point_logits[b_mask].detach().cpu(),
                    "point_pred": point_pred[b_mask].detach().cpu(),
                    "point_coords": point_tensor[b_mask, 1:4].detach().cpu(),
                }
            return result_dict

        return point_logits


SpconvMinkUNet = PTv3WaymoSegmentor
