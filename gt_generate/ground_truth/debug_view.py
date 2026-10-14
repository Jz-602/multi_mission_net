import numpy as np

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.open3d_debug_view import _BuildUniformPointCloud, _RunVisualizerGrid


def BuildTerrainMapPoints(terrain_map, map_resolution_m=0.2, valid_mask=None):
    terrain_map = np.asarray(terrain_map)
    if valid_mask is None:
        valid_mask = np.isfinite(terrain_map) & (terrain_map != INVALID_HEIGHT)
    grid_rows, grid_cols = np.nonzero(valid_mask)
    if grid_rows.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64), grid_rows, grid_cols
    rows, cols = terrain_map.shape
    x_values = (grid_cols.astype(np.float64) - float(cols) * 0.5 + 0.5) * float(map_resolution_m)
    y_values = (float(rows) * 0.5 - grid_rows.astype(np.float64) - 0.5) * float(map_resolution_m)
    z_values = terrain_map[grid_rows, grid_cols].astype(np.float64)
    return np.column_stack((x_values, y_values, z_values)), grid_rows, grid_cols


def BuildValueColors(values, invalid_value=INVALID_HEIGHT):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    valid_mask = np.isfinite(values) & (values != invalid_value)
    colors = np.tile(np.asarray((0.55, 0.55, 0.55), dtype=np.float64), (values.shape[0], 1))
    if not np.any(valid_mask):
        return colors
    valid_values = values[valid_mask]
    min_value = float(np.min(valid_values))
    max_value = float(np.max(valid_values))
    if max_value <= min_value:
        normalized = np.full(valid_values.shape, 0.5, dtype=np.float64)
    else:
        normalized = (valid_values - min_value) / (max_value - min_value)
    colors_valid = np.zeros((valid_values.shape[0], 3), dtype=np.float64)
    colors_valid[:, 0] = normalized
    colors_valid[:, 1] = 1.0 - np.abs(normalized - 0.5) * 2.0
    colors_valid[:, 2] = 1.0 - normalized
    colors[valid_mask] = colors_valid
    return np.clip(colors, 0.0, 1.0)


def BuildColoredTerrainCloud(terrain_map, value_map=None, invalid_value=INVALID_HEIGHT, color_rgb=None):
    valid_mask = np.isfinite(terrain_map) & (terrain_map != INVALID_HEIGHT)
    points, grid_rows, grid_cols = BuildTerrainMapPoints(terrain_map, valid_mask=valid_mask)
    if points.shape[0] == 0:
        return None
    if color_rgb is not None:
        colors = np.tile(np.asarray(color_rgb, dtype=np.float64), (points.shape[0], 1))
    else:
        if value_map is None:
            value_map = terrain_map
            invalid_value = INVALID_HEIGHT
        values = np.asarray(value_map)[grid_rows, grid_cols]
        colors = BuildValueColors(values, invalid_value=invalid_value)
    from gt_generate.common.open3d_debug_view import _RequireOpen3d

    o3d = _RequireOpen3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors, dtype=np.float64))
    return cloud


def ShowTerrainOnlyMaps(stage_name, map_items, b_block=True):
    del b_block
    window_specs = []
    for title, layer_map, invalid_value in map_items:
        if invalid_value is None:
            terrain_map = np.where(layer_map != 0, 0.0, INVALID_HEIGHT).astype(np.float64)
            cloud = BuildColoredTerrainCloud(terrain_map, color_rgb=(0.0, 0.9, 0.2))
        elif np.issubdtype(np.asarray(layer_map).dtype, np.integer) and invalid_value == 0.0:
            terrain_map = np.where(layer_map != 0, 0.0, INVALID_HEIGHT).astype(np.float64)
            cloud = BuildColoredTerrainCloud(terrain_map, value_map=layer_map.astype(np.float64), invalid_value=0.0)
        else:
            cloud = BuildColoredTerrainCloud(layer_map, value_map=layer_map, invalid_value=invalid_value)
        window_specs.append((f"{stage_name}-{title}", [cloud]))
    _RunVisualizerGrid(window_specs, width=860, height=680)


def ShowTerrainOnlyPointCloud(stage_name, point_sets):
    geometries = []
    for points_xyz, color_rgb in point_sets:
        cloud = _BuildUniformPointCloud(points_xyz[:, :3], color_rgb)
        if cloud is not None:
            geometries.append(cloud)
    _RunVisualizerGrid([(str(stage_name), geometries)], width=1280, height=860)
