

import numpy as np
import numba as nb


def VoxelizeWhilePreservingLabels(xyz, labels, voxel_size):
    # too slow to use
    # 计算每个点所属的体素网格
    voxel_indices = np.floor(xyz / voxel_size).astype(int)

    # 取唯一体素网格及对应点的索引
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)

    # 对每个体素内的点取平均坐标和平均颜色
    downsampled_xyz = np.zeros((len(unique_voxels), 3))
    downsampled_labels = np.zeros((len(unique_voxels), 1))

    for i in range(len(unique_voxels)):
        mask = (inverse == i)
        downsampled_xyz[i] = np.mean(xyz[mask], axis=0)
        tmp = labels[mask]
        downsampled_labels[i] = tmp[0]

    return downsampled_xyz, downsampled_labels


def VoxelizeWhilePreservingLabelsV2(xyz, labels, voxel_size):
    # 计算每个点所属的体素索引
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)

    # 哈希每个体素，用于快速分组（比 np.unique(axis=0) 快）
    keys, inv = np.unique(voxel_indices, axis=0, return_inverse=True)

    # 每个体素里点的数量
    counts = np.bincount(inv)

    # xyz 累加后除以数量 -> 平均
    sums = np.zeros((len(keys), 3), dtype=np.float64)
    np.add.at(sums, inv, xyz)
    downsampled_xyz = sums / counts[:, None]

    # labels 只取第一个即可（保持标签）
    # 利用 np.zeros + advanced indexing 来避免 for 循环
    downsampled_labels = labels[np.unique(inv, return_index=True)[1]]

    # return downsampled_xyz, downsampled_labels.reshape(-1, 1)
    return downsampled_xyz, downsampled_labels


def VoxelizeWhilePreservingLabelsV3(xyz, labels, voxel_size):
    voxels = np.floor(xyz / voxel_size).astype(np.int32)
    keys = voxels[:, 0] + voxels[:, 1] * 19349653 + voxels[:, 2] * 83492791

    sort_idx = np.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_xyz = xyz[sort_idx]
    sorted_labels = labels[sort_idx]

    mask = np.concatenate(([True], sorted_keys[1:] != sorted_keys[:-1]))
    group_starts = np.where(mask)[0]
    group_ends = np.concatenate((group_starts[1:], [len(keys)]))
    group_counts = group_ends - group_starts

    result_xyz = np.add.reduceat(sorted_xyz, group_starts, axis=0) / group_counts[:, None]
    result_labels = sorted_labels[group_starts]

    return result_xyz, result_labels

def VoxelizeWhilePreservingLabelsV4(xyz, labels, voxel_size):
    # 1) 计算体素坐标
    vox = np.floor(xyz / voxel_size).astype(np.int64)

    # 2) 将 (x, y, z) 映射成无冲突 1D key（前提：点云 voxel 坐标有限）
    mins = vox.min(axis=0)
    vox -= mins  # shift，使坐标非负并压缩范围
    maxs = vox.max(axis=0) + 1
    keys = vox[:, 0] + maxs[0] * (vox[:, 1] + maxs[1] * vox[:, 2])

    # 3) 排序，分组
    sort_idx = np.argsort(keys)
    sk = keys[sort_idx]
    sxyz = xyz[sort_idx]
    slabels = labels[sort_idx]

    # 4) 找到每个 group 起始
    mask = np.concatenate(([True], sk[1:] != sk[:-1]))
    group_starts = np.where(mask)[0]
    group_ends = np.append(group_starts[1:], len(keys))
    counts = group_ends - group_starts

    # 5) 每组 xyz 求均值
    result_xyz = np.add.reduceat(sxyz, group_starts, axis=0) / counts[:, None]

    # 6) label 取每组第一个
    result_labels = slabels[group_starts]

    return result_xyz, result_labels

# @nb.njit(parallel=True)
# def VoxelizeWhilePreservingLabels_numba(xyz, labels, voxel_size):
#     voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)
#
#     # 哈希：转为一维 key，避免 unique(axis=0) 慢
#     # 这里假设点云坐标不超过 1e6
#     primes = np.array([73856093, 19349663, 83492791], dtype=np.int64)
#     keys = (voxel_indices * primes).sum(axis=1)
#
#     uniq_keys, inv = np.unique(keys, return_inverse=True)
#     n = len(uniq_keys)
#
#     sums = np.zeros((n, 3))
#     counts = np.zeros(n, dtype=np.int64)
#     out_labels = np.zeros(n, dtype=labels.dtype)
#
#     for i in nb.prange(len(inv)):
#         idx = inv[i]
#         sums[idx, 0] += xyz[i, 0]
#         sums[idx, 1] += xyz[i, 1]
#         sums[idx, 2] += xyz[i, 2]
#         counts[idx] += 1
#         if counts[idx] == 1:  # 第一次赋值标签
#             out_labels[idx] = labels[i]
#
#     for i in nb.prange(n):
#         sums[i] /= counts[i]
#
#     return sums, out_labels.reshape(-1, 1)