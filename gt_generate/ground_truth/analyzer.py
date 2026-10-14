from collections import deque

import numpy as np
from scipy import ndimage

from gt_generate.common.bgk import BgkSmoother
from gt_generate.common.constants import INVALID_HEIGHT, INVALID_VALUE


def BuildRawTerrainMaps(stats):
    count_map = stats["count_map"].astype(np.float64)
    sum_z_map = stats["sum_z_map"].astype(np.float64)
    sum_z2_map = stats["sum_z2_map"].astype(np.float64)
    valid_mask = count_map > 0.0

    mean_map = np.full(count_map.shape, INVALID_HEIGHT, dtype=np.float64)
    std_map = np.full(count_map.shape, INVALID_HEIGHT, dtype=np.float64)
    mean_map[valid_mask] = sum_z_map[valid_mask] / count_map[valid_mask]
    variance = np.zeros(count_map.shape, dtype=np.float64)
    variance[valid_mask] = sum_z2_map[valid_mask] / count_map[valid_mask] - mean_map[valid_mask] * mean_map[valid_mask]
    variance = np.maximum(variance, 0.0)
    std_map[valid_mask] = np.sqrt(variance[valid_mask])
    return mean_map, std_map


def BuildReliableSeedMap(mean_map, std_map, std_thresh_m):
    valid_mask = (
        (mean_map != INVALID_HEIGHT)
        & (std_map != INVALID_HEIGHT)
        & (std_map <= float(std_thresh_m))
    )
    seed_map = np.full(mean_map.shape, INVALID_HEIGHT, dtype=np.float64)
    seed_map[valid_mask] = mean_map[valid_mask]
    return seed_map


def CompleteTerrain(seed_map, map_resolution_m, bgk_radius_m, bgk_min_valid, bgk_bilateral_std):
    smoother = BgkSmoother(
        map_resolution=map_resolution_m,
        prediction_kernel_radius=bgk_radius_m,
        valid_points_num_thresh=bgk_min_valid,
        bilateral_std=bgk_bilateral_std,
    )
    completed_map, confidence_map = smoother.Run(seed_map, invalid_value=INVALID_HEIGHT, b_perform_bf=True)
    return completed_map, confidence_map


def BuildInpaintDistanceMap(seed_map, inpainted_map, map_resolution_m):
    seed_valid_mask = seed_map != INVALID_HEIGHT
    inpainted_valid_mask = inpainted_map != INVALID_HEIGHT
    distance_map = np.full(seed_map.shape, INVALID_HEIGHT, dtype=np.float64)
    if not np.any(seed_valid_mask):
        return distance_map
    distance_map[inpainted_valid_mask] = ndimage.distance_transform_edt(~seed_valid_mask)[inpainted_valid_mask] * float(map_resolution_m)
    return distance_map


def CalculateSlopeDeg(terrain_map, map_resolution_m):
    slope_map = np.full(terrain_map.shape, INVALID_VALUE, dtype=np.float64)
    valid_mask = terrain_map != INVALID_HEIGHT
    center_valid = valid_mask[1:-1, 1:-1]
    neighbor_valid = (
        center_valid
        & valid_mask[1:-1, 2:]
        & valid_mask[1:-1, :-2]
        & valid_mask[2:, 1:-1]
        & valid_mask[:-2, 1:-1]
    )
    dz_dx = (terrain_map[1:-1, 2:] - terrain_map[1:-1, :-2]) / (2.0 * float(map_resolution_m))
    dz_dy = (terrain_map[:-2, 1:-1] - terrain_map[2:, 1:-1]) / (2.0 * float(map_resolution_m))
    slope_rad = np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy))
    slope_map[1:-1, 1:-1][neighbor_valid] = slope_rad[neighbor_valid] * 180.0 / np.pi
    return slope_map


def BuildTrajectorySeedMask(poses, rows, cols, map_min_x_cm, map_min_y_cm, map_resolution_m, slope_map):
    seed_mask = np.zeros((rows, cols), dtype=np.uint8)
    map_min_x_m = float(map_min_x_cm) / 100.0
    map_min_y_m = float(map_min_y_cm) / 100.0
    for pose in poses:
        col_idx = int(np.floor((float(pose[0]) - map_min_x_m) / float(map_resolution_m)))
        row_idx = int(rows - 1 - np.floor((float(pose[1]) - map_min_y_m) / float(map_resolution_m)))
        if 0 <= row_idx < rows and 0 <= col_idx < cols and slope_map[row_idx, col_idx] != INVALID_VALUE:
            seed_mask[row_idx, col_idx] = 1
    return seed_mask


def GrowRoiFromTrajectory(slope_map, trajectory_seed_mask, slope_thresh_deg):
    roi_mask = np.zeros(slope_map.shape, dtype=np.uint8)
    queue = deque()
    seed_rows, seed_cols = np.nonzero(trajectory_seed_mask != 0)
    for row_idx, col_idx in zip(seed_rows, seed_cols):
        if slope_map[row_idx, col_idx] == INVALID_VALUE:
            continue
        roi_mask[row_idx, col_idx] = 1
        queue.append((int(row_idx), int(col_idx)))

    rows, cols = slope_map.shape
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        row_idx, col_idx = queue.popleft()
        for row_offset, col_offset in offsets:
            nr = row_idx + row_offset
            nc = col_idx + col_offset
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if roi_mask[nr, nc] != 0:
                continue
            if slope_map[nr, nc] == INVALID_VALUE:
                continue
            if slope_map[nr, nc] >= float(slope_thresh_deg):
                continue
            roi_mask[nr, nc] = 1
            queue.append((nr, nc))
    return roi_mask


def BuildFinalSeedMap(raw_mean_map, raw_std_map, provisional_map, args, roi_mask=None):
    if roi_mask is None:
        roi_mask = np.ones(raw_mean_map.shape, dtype=np.uint8)
    valid_mask = (
        (raw_mean_map != INVALID_HEIGHT)
        & (raw_std_map != INVALID_HEIGHT)
        & (provisional_map != INVALID_HEIGHT)
        & (roi_mask != 0)
        & (raw_std_map <= float(args.final_seed_std_max_m))
        & (np.abs(raw_mean_map - provisional_map) <= float(args.final_seed_raw_provisional_delta_max_m))
    )
    final_seed_map = np.full(raw_mean_map.shape, INVALID_HEIGHT, dtype=np.float64)
    final_seed_map[valid_mask] = raw_mean_map[valid_mask]
    return final_seed_map
