

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