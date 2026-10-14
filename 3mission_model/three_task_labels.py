import json
import os

import numpy as np

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.io_utils import ReadMat


TERRAIN_FILENAMES = {
    "terrain_height": ("terrain_z_inpainted.bin", np.dtype(np.float32)),
    "terrain_inpainted_mask": ("terrain_inpainted_mask.bin", np.dtype(np.uint8)),
    "terrain_train_weight": ("terrain_train_weight.bin", np.dtype(np.float32)),
    "terrain_inpaint_distance": ("terrain_inpaint_distance.bin", np.dtype(np.float32)),
}
OCC_REQUIRED_FIELDS = {
    "voxel_indices",
    "labels",
    "voxel_size_xyz",
    "pc_range",
    "grid_shape",
}
GRID_TOLERANCE = 1e-4


def ValidateRange(target_pc_range, context):
    target_pc_range = np.asarray(target_pc_range, dtype=np.float64)
    if target_pc_range.shape != (6,):
        raise ValueError(f"{context}: pc_range must have shape [6], got {target_pc_range.shape}")
    if not np.all(np.isfinite(target_pc_range)):
        raise ValueError(f"{context}: pc_range contains non-finite values: {target_pc_range}")
    if np.any(target_pc_range[3:] <= target_pc_range[:3]):
        raise ValueError(f"{context}: pc_range min must be smaller than max: {target_pc_range}")
    return target_pc_range


def ComputeAlignedBounds(source_min, source_max, target_min, target_max, resolution, context):
    if target_min < source_min - GRID_TOLERANCE or target_max > source_max + GRID_TOLERANCE:
        raise ValueError(
            f"{context}: target range [{target_min}, {target_max}) is outside "
            f"source range [{source_min}, {source_max})"
        )

    start_float = (target_min - source_min) / resolution
    end_float = (target_max - source_min) / resolution
    start_index = int(round(start_float))
    end_index = int(round(end_float))
    if not np.isclose(start_float, start_index, rtol=0.0, atol=GRID_TOLERANCE):
        raise ValueError(f"{context}: target minimum {target_min} is not aligned to {resolution} m grid")
    if not np.isclose(end_float, end_index, rtol=0.0, atol=GRID_TOLERANCE):
        raise ValueError(f"{context}: target maximum {target_max} is not aligned to {resolution} m grid")
    if end_index <= start_index:
        raise ValueError(f"{context}: aligned crop is empty")
    return start_index, end_index


def RequireMetadataNumber(metadata, key, b_integer, context):
    if key not in metadata:
        raise KeyError(f"{context}: frame_meta.json is missing '{key}'")
    value = metadata[key]
    if isinstance(value, bool):
        raise TypeError(f"{context}: metadata '{key}' must be numeric, got bool")
    if b_integer:
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{context}: metadata '{key}' must be an integer, got {type(value).__name__}")
        return int(value)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{context}: metadata '{key}' must be numeric, got {type(value).__name__}")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{context}: metadata '{key}' is not finite")
    return value


def LoadTerrainLabelData(root_path, sequence_name, timestamp_ms, terrain_label_path, target_pc_range):
    context = f"terrain labels for {sequence_name} timestamp {timestamp_ms}"
    target_pc_range = ValidateRange(target_pc_range, context)
    frame_dir = os.path.join(
        root_path,
        sequence_name,
        terrain_label_path,
        f"{timestamp_ms:013d}",
    )
    metadata_filename = os.path.join(frame_dir, "frame_meta.json")
    terrain_dir = os.path.join(frame_dir, "base", "terrain_completion")

    try:
        with open(metadata_filename, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except Exception as exception:
        raise RuntimeError(f"{context}: failed to read {metadata_filename}: {exception}") from exception
    if not isinstance(metadata, dict):
        raise TypeError(f"{context}: {metadata_filename} must contain a JSON object")

    metadata_timestamp = RequireMetadataNumber(metadata, "timestamp", True, context)
    patch_rows = RequireMetadataNumber(metadata, "patch_rows", True, context)
    patch_cols = RequireMetadataNumber(metadata, "patch_cols", True, context)
    resolution = RequireMetadataNumber(metadata, "map_resolution_m", False, context)
    forward_range = RequireMetadataNumber(metadata, "forward_range_m", False, context)
    backward_range = RequireMetadataNumber(metadata, "backward_range_m", False, context)
    left_range = RequireMetadataNumber(metadata, "left_range_m", False, context)
    right_range = RequireMetadataNumber(metadata, "right_range_m", False, context)

    if metadata_timestamp != timestamp_ms:
        raise ValueError(
            f"{context}: metadata timestamp {metadata_timestamp} does not match target timestamp {timestamp_ms} "
            f"in {metadata_filename}"
        )
    if patch_rows <= 0 or patch_cols <= 0 or resolution <= 0.0:
        raise ValueError(
            f"{context}: invalid terrain shape/resolution rows={patch_rows}, cols={patch_cols}, "
            f"resolution={resolution}"
        )
    expected_rows = (forward_range + backward_range) / resolution
    expected_cols = (left_range + right_range) / resolution
    if not np.isclose(expected_rows, patch_rows, rtol=0.0, atol=GRID_TOLERANCE):
        raise ValueError(f"{context}: x range/resolution gives {expected_rows} rows, metadata says {patch_rows}")
    if not np.isclose(expected_cols, patch_cols, rtol=0.0, atol=GRID_TOLERANCE):
        raise ValueError(f"{context}: y range/resolution gives {expected_cols} cols, metadata says {patch_cols}")

    terrain_arrays = {}
    for field_name, (filename, expected_dtype) in TERRAIN_FILENAMES.items():
        full_filename = os.path.join(terrain_dir, filename)
        try:
            array = ReadMat(full_filename)
        except Exception as exception:
            raise RuntimeError(f"{context}: failed to read {full_filename}: {exception}") from exception
        if array.shape != (patch_rows, patch_cols):
            raise ValueError(
                f"{context}: {full_filename} has shape {array.shape}, expected {(patch_rows, patch_cols)}"
            )
        if array.dtype != expected_dtype:
            raise TypeError(
                f"{context}: {full_filename} has dtype {array.dtype}, expected {expected_dtype}"
            )
        terrain_arrays[field_name] = array

    terrain_mask = terrain_arrays.pop("terrain_inpainted_mask")
    unique_mask_values = np.unique(terrain_mask)
    if not np.all(np.isin(unique_mask_values, np.array([0, 1], dtype=np.uint8))):
        raise ValueError(f"{context}: terrain mask contains values outside {{0, 1}}: {unique_mask_values}")

    source_x_min = -backward_range
    source_x_max = forward_range
    source_y_min = -right_range
    source_y_max = left_range
    x_start_ascending, x_end_ascending = ComputeAlignedBounds(
        source_x_min,
        source_x_max,
        target_pc_range[0],
        target_pc_range[3],
        resolution,
        f"{context} x",
    )
    y_start_ascending, y_end_ascending = ComputeAlignedBounds(
        source_y_min,
        source_y_max,
        target_pc_range[1],
        target_pc_range[4],
        resolution,
        f"{context} y",
    )
    # 地形原始排列是 x 递减的行、y 递减的列，因此由递增坐标边界换算为反向切片。
    row_start = patch_rows - x_end_ascending
    row_end = patch_rows - x_start_ascending
    col_start = patch_cols - y_end_ascending
    col_end = patch_cols - y_start_ascending
    crop_slice = (slice(row_start, row_end), slice(col_start, col_end))

    terrain_height = terrain_arrays["terrain_height"][crop_slice]
    terrain_mask = terrain_mask[crop_slice]
    terrain_train_weight = terrain_arrays["terrain_train_weight"][crop_slice]
    terrain_inpaint_distance = terrain_arrays["terrain_inpaint_distance"][crop_slice]
    terrain_valid_mask = (
        (terrain_mask != 0)
        & np.isfinite(terrain_height)
        & (terrain_height != np.float32(INVALID_HEIGHT))
    )
    # 正常空洞是有限INVALID_HEIGHT且原mask=0；损坏/矛盾不得作为空洞负例。
    b_height_valid = np.isfinite(terrain_height) & (terrain_height != np.float32(INVALID_HEIGHT))
    terrain_ignore_mask = (~np.isfinite(terrain_height)) | ((terrain_mask != 0) != b_height_valid)

    return {
        "terrain_height": terrain_height[np.newaxis, ...].astype(np.float32, copy=False),
        "terrain_valid_mask": terrain_valid_mask[np.newaxis, ...],
        "terrain_ignore_mask": terrain_ignore_mask[np.newaxis, ...],
        "terrain_train_weight": terrain_train_weight[np.newaxis, ...].astype(np.float32, copy=False),
        "terrain_inpaint_distance": terrain_inpaint_distance[np.newaxis, ...].astype(np.float32, copy=False),
        "terrain_resolution": np.array([resolution], dtype=np.float32),
        "terrain_xy_range": target_pc_range[[0, 1, 3, 4]].astype(np.float32),
    }


def LoadOCCLabelData(root_path, sequence_name, timestamp_ms, occ_label_path, target_pc_range):
    context = f"OCC labels for {sequence_name} timestamp {timestamp_ms}"
    target_pc_range = ValidateRange(target_pc_range, context)
    occ_filename = os.path.join(root_path, sequence_name, occ_label_path, f"{timestamp_ms}.npz")
    try:
        with np.load(occ_filename, allow_pickle=False) as occ_file:
            missing_fields = OCC_REQUIRED_FIELDS.difference(occ_file.files)
            if missing_fields:
                raise KeyError(f"missing fields {sorted(missing_fields)}")
            voxel_indices = occ_file["voxel_indices"]
            labels = occ_file["labels"]
            voxel_size_xyz = occ_file["voxel_size_xyz"]
            source_pc_range = occ_file["pc_range"]
            source_grid_shape = occ_file["grid_shape"]
    except Exception as exception:
        raise RuntimeError(f"{context}: failed to read {occ_filename}: {exception}") from exception

    expected_fields = {
        "voxel_indices": (voxel_indices, (None, 3), np.dtype(np.int32)),
        "labels": (labels, (None,), np.dtype(np.uint16)),
        "voxel_size_xyz": (voxel_size_xyz, (3,), np.dtype(np.float32)),
        "pc_range": (source_pc_range, (6,), np.dtype(np.float32)),
        "grid_shape": (source_grid_shape, (3,), np.dtype(np.int64)),
    }
    for field_name, (array, expected_shape, expected_dtype) in expected_fields.items():
        if array.ndim != len(expected_shape) or any(
            expected_size is not None and array.shape[axis] != expected_size
            for axis, expected_size in enumerate(expected_shape)
        ):
            raise ValueError(
                f"{context}: {occ_filename} field '{field_name}' has shape {array.shape}, "
                f"expected {expected_shape}"
            )
        if array.dtype != expected_dtype:
            raise TypeError(
                f"{context}: {occ_filename} field '{field_name}' has dtype {array.dtype}, "
                f"expected {expected_dtype}"
            )

    if voxel_indices.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{context}: coordinate count {voxel_indices.shape[0]} does not match label count {labels.shape[0]} "
            f"in {occ_filename}"
        )
    if not np.all(np.isfinite(voxel_size_xyz)) or np.any(voxel_size_xyz <= 0.0):
        raise ValueError(f"{context}: invalid voxel_size_xyz {voxel_size_xyz} in {occ_filename}")
    if not np.all(np.isfinite(source_pc_range)) or np.any(source_pc_range[3:] <= source_pc_range[:3]):
        raise ValueError(f"{context}: invalid pc_range {source_pc_range} in {occ_filename}")
    if np.any(source_grid_shape <= 0):
        raise ValueError(f"{context}: invalid grid_shape {source_grid_shape} in {occ_filename}")
    expected_grid_shape = (source_pc_range[3:] - source_pc_range[:3]) / voxel_size_xyz
    if not np.allclose(expected_grid_shape, source_grid_shape, rtol=0.0, atol=GRID_TOLERANCE):
        raise ValueError(
            f"{context}: range/resolution gives grid {expected_grid_shape}, file says {source_grid_shape} "
            f"in {occ_filename}"
        )
    if voxel_indices.size > 0:
        if np.any(voxel_indices < 0) or np.any(voxel_indices >= source_grid_shape[None, :]):
            raise ValueError(f"{context}: voxel indices fall outside grid {source_grid_shape} in {occ_filename}")
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, np.array([4, 5, 6], dtype=np.uint16))):
        raise ValueError(f"{context}: undefined OCC labels {unique_labels} in {occ_filename}")

    crop_start = np.empty(3, dtype=np.int64)
    crop_end = np.empty(3, dtype=np.int64)
    for axis, axis_name in enumerate(("x", "y", "z")):
        crop_start[axis], crop_end[axis] = ComputeAlignedBounds(
            float(source_pc_range[axis]),
            float(source_pc_range[axis + 3]),
            float(target_pc_range[axis]),
            float(target_pc_range[axis + 3]),
            float(voxel_size_xyz[axis]),
            f"{context} {axis_name}",
        )

    keep_mask = np.all(
        (voxel_indices >= crop_start[None, :]) & (voxel_indices < crop_end[None, :]),
        axis=1,
    )
    cropped_indices = voxel_indices[keep_mask].astype(np.int64) - crop_start[None, :]
    cropped_labels = labels[keep_mask].astype(np.int64) - 3
    cropped_grid_shape = crop_end - crop_start

    return {
        "occ_voxel_indices": cropped_indices,
        "occ_labels": cropped_labels,
        "occ_voxel_size_xyz": voxel_size_xyz.astype(np.float32, copy=False),
        "occ_pc_range": target_pc_range.astype(np.float32),
        "occ_grid_shape": cropped_grid_shape.astype(np.int64),
    }
