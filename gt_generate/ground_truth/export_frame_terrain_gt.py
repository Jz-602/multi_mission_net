import os
import sys
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from scipy.ndimage import map_coordinates

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.geometry import YprToRotationMatrixV2
from gt_generate.common.io_utils import (
    EnsureDir,
    LoadMapInfo,
    LoadPoseDb,
    ReadMat,
    SaveJson,
    UpdateNamespaceFromCli,
    WriteMat,
)
from gt_generate.ground_truth.data_paths import (
    DEFAULT_FH_WAYMO_ROOT,
    DEFAULT_FH_WAYMO_SEQUENCE_NAME,
    ApplyResolvedDataPaths,
    BuildDefaultTerrainDir,
    BuildTimestampFile,
    ResolveTerrainOnlyDataPaths,
)


def BuildProgressBar(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def BuildArgs():
    return Namespace(
        # FH_waymo_small 根目录，或其中某一条 sequence 目录。
        data_path=DEFAULT_FH_WAYMO_ROOT,
        sequence_name=DEFAULT_FH_WAYMO_SEQUENCE_NAME,
        pcd_dir="",
        semantic_label_dir="",
        pose_filename="",
        # 全局 terrain-only 地图目录，由 build_waymo_terrain_only.py 生成。
        terrain_dir="",
        # 每帧 terrain GT 输出目录。
        output_dir="",
        # 当前帧 patch 前向范围，单位 m。
        forward_range_m=80.0,
        # 当前帧 patch 后向范围，单位 m。
        backward_range_m=80.0,
        # 当前帧 patch 左侧范围，单位 m。
        left_range_m=80.0,
        # 当前帧 patch 右侧范围，单位 m。
        right_range_m=80.0,
        # patch 采样分辨率，必须与全局 terrain-only 地图分辨率一致。
        map_resolution_m=0.2,
        # 每隔多少帧导出一次。1 表示导出有点云文件的每一帧。
        frame_stride=1,
        # 默认只导出到最后一个 Waymo semantic label 对应的 lidar 帧，避免尾段无语义约束的 GT。
        b_clip_to_last_semantic_frame=True,
        # 调试用最大导出帧数；0 表示不限制。
        max_frames=0,
        # 仅筛选逐帧导出，逗号分隔的毫秒时间戳；空字符串保留原有行为。
        target_timestamps="",
        # 用 inpaint distance 生成训练权重的衰减尺度，单位 m；只写入 summary 供训练端使用。
        terrain_weight_distance_sigma_m=3.0,
    )


def LoadGlobalTerrainMaps(terrain_dir):
    required_names = [
        "terrain_z_final_seed",
        "terrain_z_inpainted",
        "terrain_inpaint_distance",
    ]
    maps = {}
    for field_name in required_names:
        filename = os.path.join(terrain_dir, f"{field_name}.bin")
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        maps[field_name] = ReadMat(filename)
    return maps


def BuildLocalGrid(args, patch_rows, patch_cols):
    row_indices = np.arange(patch_rows, dtype=np.float64)
    col_indices = np.arange(patch_cols, dtype=np.float64)
    local_x = float(args.forward_range_m) - (row_indices + 0.5) * float(args.map_resolution_m)
    local_y = float(args.left_range_m) - (col_indices + 0.5) * float(args.map_resolution_m)
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="ij")
    return grid_x, grid_y


def WorldToGridIndex(world_x, world_y, map_min_x_cm, map_min_y_cm, rows, cols, map_resolution_m):
    col_float = (world_x * 100.0 - float(map_min_x_cm)) / (float(map_resolution_m) * 100.0) - 0.5
    row_float = float(rows) - 0.5 - (world_y * 100.0 - float(map_min_y_cm)) / (float(map_resolution_m) * 100.0)
    row_idx = np.rint(row_float).astype(np.int32)
    col_idx = np.rint(col_float).astype(np.int32)
    in_bounds = (row_idx >= 0) & (row_idx < int(rows)) & (col_idx >= 0) & (col_idx < int(cols))
    safe_rows = np.clip(row_idx, 0, int(rows) - 1)
    safe_cols = np.clip(col_idx, 0, int(cols) - 1)
    return safe_rows, safe_cols, in_bounds


def SampleMap(layer_map, rows, cols, valid_mask, fill_value):
    output_map = np.full(rows.shape, fill_value, dtype=layer_map.dtype)
    output_map[valid_mask] = layer_map[rows[valid_mask], cols[valid_mask]]
    return output_map


def BuildFrameTerrainGt(global_maps, pose, map_info, args, patch_rows, patch_cols):
    map_min_x_cm, map_min_y_cm, rows, cols = map_info
    # pose [6] 为当前帧到世界系的 [x, y, z, roll, pitch, yaw]。
    # 输出 [patch_rows, patch_cols] 的 XY 网格和 Z 均在当前帧点云坐标系。
    grid_x, grid_y = BuildLocalGrid(args, patch_rows, patch_cols)
    rotation = YprToRotationMatrixV2((pose[5], pose[4], pose[3]))
    base_x = pose[0] + rotation[0, 0] * grid_x + rotation[0, 1] * grid_y
    base_y = pose[1] + rotation[1, 0] * grid_x + rotation[1, 1] * grid_y
    base_z = pose[2] + rotation[2, 0] * grid_x + rotation[2, 1] * grid_y
    world_height = global_maps["terrain_z_inpainted"].astype(np.float64).copy()
    world_height[world_height == INVALID_HEIGHT] = np.nan
    local_z = np.zeros_like(grid_x)
    # 沿当前帧 Z 轴求与全局地形的交点；倾斜会同时影响世界 X、Y、Z。
    for _ in range(16):
        world_x = base_x + rotation[0, 2] * local_z
        world_y = base_y + rotation[1, 2] * local_z
        col_float = (world_x - map_min_x_cm / 100.0) / args.map_resolution_m - 0.5
        row_float = rows - 0.5 - (world_y - map_min_y_cm / 100.0) / args.map_resolution_m
        sampled_z = map_coordinates(world_height, [row_float, col_float], order=1, mode="constant", cval=np.nan)
        valid_mask = np.isfinite(sampled_z)
        local_z[valid_mask] = (sampled_z[valid_mask] - base_z[valid_mask]) / rotation[2, 2]
    terrain_z_inpainted = np.where(valid_mask, local_z, INVALID_HEIGHT).astype(np.float32)
    grid_rows, grid_cols, in_bounds = WorldToGridIndex(
        world_x, world_y, map_min_x_cm, map_min_y_cm, rows, cols, args.map_resolution_m,
    )
    in_bounds &= valid_mask
    terrain_z_final_seed = SampleMap(global_maps["terrain_z_final_seed"], grid_rows, grid_cols, in_bounds, INVALID_HEIGHT)
    terrain_inpaint_distance = SampleMap(global_maps["terrain_inpaint_distance"], grid_rows, grid_cols, in_bounds, INVALID_HEIGHT)
    terrain_inpainted_mask = (terrain_z_inpainted != INVALID_HEIGHT).astype(np.uint8)
    reliable_mask = (terrain_z_final_seed != INVALID_HEIGHT).astype(np.uint8)
    distance_valid_mask = terrain_inpaint_distance != INVALID_HEIGHT
    terrain_train_weight = np.zeros(terrain_inpaint_distance.shape, dtype=np.float32)
    terrain_train_weight[distance_valid_mask] = np.exp(
        -terrain_inpaint_distance[distance_valid_mask].astype(np.float32) / float(args.terrain_weight_distance_sigma_m)
    )
    terrain_train_weight[reliable_mask != 0] = 1.0

    return {
        "base/terrain_completion": {
            "terrain_z_inpainted": terrain_z_inpainted,
            "terrain_inpainted_mask": terrain_inpainted_mask,
            "terrain_inpaint_distance": terrain_inpaint_distance.astype(np.float32),
            "terrain_train_weight": terrain_train_weight.astype(np.float32),
        },
    }


def SaveFrameTerrainGt(frame_dir, frame_field_groups):
    for group_name, fields in frame_field_groups.items():
        group_dir = os.path.join(frame_dir, group_name)
        EnsureDir(group_dir)
        for field_name, field_value in fields.items():
            output_filename = os.path.join(group_dir, f"{field_name}.bin")
            if field_value.dtype == np.uint8:
                WriteMat(output_filename, field_value.astype(np.uint8))
            else:
                WriteMat(output_filename, field_value.astype(np.float32))


def CleanupDeprecatedFrameTerrainGtFiles(frame_dir):
    deprecated_filenames = [
        os.path.join(frame_dir, "base", "terrain_raw", "terrain_z_mean_raw.bin"),
        os.path.join(frame_dir, "base", "terrain_raw", "terrain_z_std_raw.bin"),
        os.path.join(frame_dir, "base", "terrain_completion", "terrain_z_reliable.bin"),
    ]
    for filename in deprecated_filenames:
        if os.path.exists(filename):
            os.remove(filename)
    terrain_raw_dir = os.path.join(frame_dir, "base", "terrain_raw")
    if os.path.isdir(terrain_raw_dir) and not os.listdir(terrain_raw_dir):
        os.rmdir(terrain_raw_dir)


def ResolveLastSemanticFrameIndex(data_path, timestamps):
    args = Namespace(data_path=data_path, sequence_name="", pcd_dir="", semantic_label_dir="", pose_filename="")
    data_paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=True, b_require_pose=False)
    semantic_dir = data_paths.semantic_label_dir
    if not os.path.isdir(semantic_dir):
        raise FileNotFoundError(f"Waymo semantic label dir not found: {semantic_dir}")
    timestamp_to_idx = {int(timestamp): int(frame_idx) for frame_idx, timestamp in enumerate(timestamps)}
    matched_indices = []
    for filename in os.listdir(semantic_dir):
        if not filename.endswith(".npy"):
            continue
        timestamp_text = os.path.splitext(filename)[0]
        try:
            timestamp = int(timestamp_text)
        except ValueError:
            continue
        if timestamp in timestamp_to_idx:
            matched_indices.append(timestamp_to_idx[timestamp])
    if not matched_indices:
        raise RuntimeError(f"No semantic label timestamp matches lidar pose timestamps under {semantic_dir}")
    return int(max(matched_indices)), int(len(matched_indices))


def ExportFrameTerrainGt(args):
    map_info = LoadMapInfo(os.path.join(args.terrain_dir, "map_info.txt"))
    global_maps = LoadGlobalTerrainMaps(args.terrain_dir)
    timestamps, poses = LoadPoseDb(args.pose_filename)
    patch_rows = int(round((float(args.forward_range_m) + float(args.backward_range_m)) / float(args.map_resolution_m)))
    patch_cols = int(round((float(args.left_range_m) + float(args.right_range_m)) / float(args.map_resolution_m)))
    last_export_frame_idx = int(timestamps.shape[0]) - 1
    semantic_label_num = 0
    if bool(args.b_clip_to_last_semantic_frame):
        last_export_frame_idx, semantic_label_num = ResolveLastSemanticFrameIndex(args.data_path, timestamps)
    EnsureDir(args.output_dir)
    target_text = str(getattr(args, "target_timestamps", "")).strip()
    target_timestamps = {int(value) for value in target_text.split(",")} if target_text else None

    frame_manifest = []
    exported_num = 0
    for frame_idx, timestamp in BuildProgressBar(enumerate(timestamps), total=int(timestamps.shape[0]), desc="TerrainOnlyFrameGt"):
        if target_timestamps is not None and int(timestamp) not in target_timestamps:
            continue
        if frame_idx > last_export_frame_idx:
            break
        if int(args.frame_stride) > 1 and frame_idx % int(args.frame_stride) != 0:
            continue
        pcd_filename = BuildTimestampFile(args.pcd_dir, timestamp, ".pcd")
        if not os.path.exists(pcd_filename):
            continue
        frame_dir = os.path.join(args.output_dir, f"{int(timestamp):013d}")
        frame_field_groups = BuildFrameTerrainGt(global_maps, poses[frame_idx], map_info, args, patch_rows, patch_cols)
        CleanupDeprecatedFrameTerrainGtFiles(frame_dir)
        SaveFrameTerrainGt(frame_dir, frame_field_groups)
        SaveJson(
            os.path.join(frame_dir, "frame_meta.json"),
            {
                "timestamp": int(timestamp),
                "frame_index": int(frame_idx),
                "pose_xyz_rpy": [float(item) for item in poses[frame_idx]],
                "patch_rows": int(patch_rows),
                "patch_cols": int(patch_cols),
                "map_resolution_m": float(args.map_resolution_m),
                "forward_range_m": float(args.forward_range_m),
                "backward_range_m": float(args.backward_range_m),
                "left_range_m": float(args.left_range_m),
                "right_range_m": float(args.right_range_m),
                "last_semantic_frame_index": int(last_export_frame_idx),
            },
        )
        frame_manifest.append(
            {
                "timestamp": int(timestamp),
                "frame_index": int(frame_idx),
                "frame_dir": os.path.relpath(frame_dir, args.terrain_dir),
            }
        )
        exported_num += 1
        if int(args.max_frames) > 0 and exported_num >= int(args.max_frames):
            break

    if target_timestamps is not None and {frame["timestamp"] for frame in frame_manifest} != target_timestamps:
        raise ValueError("Terrain target export incomplete; check timestamps, clipping, stride and max_frames")
    manifest = {
        "schema_version": "terrain_only_frame_gt_v1",
        "data_path": args.data_path,
        "terrain_dir": args.terrain_dir,
        "output_dir": args.output_dir,
        "patch_rows": int(patch_rows),
        "patch_cols": int(patch_cols),
        "map_resolution_m": float(args.map_resolution_m),
        "forward_range_m": float(args.forward_range_m),
        "backward_range_m": float(args.backward_range_m),
        "left_range_m": float(args.left_range_m),
        "right_range_m": float(args.right_range_m),
        "clip_to_last_semantic_frame": bool(args.b_clip_to_last_semantic_frame),
        "semantic_label_num": int(semantic_label_num),
        "last_export_frame_index": int(last_export_frame_idx),
        "last_export_timestamp": int(timestamps[last_export_frame_idx]),
        "terrain_reference_field": "terrain_z_inpainted",
        "exported_frame_fields": [
            "base/terrain_completion/terrain_z_inpainted.bin",
            "base/terrain_completion/terrain_inpainted_mask.bin",
            "base/terrain_completion/terrain_inpaint_distance.bin",
            "base/terrain_completion/terrain_train_weight.bin",
        ],
        "weight_field": "terrain_train_weight",
        "weight_note": "terrain_train_weight = exp(-terrain_inpaint_distance / terrain_weight_distance_sigma_m), reliable seed cells are clamped to 1.0",
        "terrain_weight_distance_sigma_m": float(args.terrain_weight_distance_sigma_m),
        "frames": frame_manifest,
    }
    SaveJson(os.path.join(args.terrain_dir, "terrain_only_frame_gt_manifest.json"), manifest)
    SaveJson(
        os.path.join(args.terrain_dir, "terrain_only_frame_gt_summary.json"),
        {
            "schema_version": manifest["schema_version"],
            "data_path": args.data_path,
            "terrain_dir": args.terrain_dir,
            "output_dir": args.output_dir,
            "exported_frame_num": int(len(frame_manifest)),
            "patch_rows": int(patch_rows),
            "patch_cols": int(patch_cols),
            "map_resolution_m": float(args.map_resolution_m),
            "clip_to_last_semantic_frame": bool(args.b_clip_to_last_semantic_frame),
            "semantic_label_num": int(semantic_label_num),
            "last_export_frame_index": int(last_export_frame_idx),
            "last_export_timestamp": int(timestamps[last_export_frame_idx]),
        },
    )
    return manifest


def Main():
    args = BuildArgs()
    UpdateNamespaceFromCli(args, sys.argv)
    data_paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=True, b_require_pose=True)
    ApplyResolvedDataPaths(args, data_paths)
    if not str(args.terrain_dir).strip():
        args.terrain_dir = BuildDefaultTerrainDir(args)
    if not str(args.output_dir).strip():
        args.output_dir = os.path.join(args.terrain_dir, "frames")
    manifest = ExportFrameTerrainGt(args)
    print(f"[TerrainOnlyFrameGt] saved frames: {len(manifest['frames'])}, root: {args.output_dir}")


if __name__ == "__main__":
    Main()
