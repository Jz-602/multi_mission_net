import json
import os
import sys
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.io_utils import LoadPcdXyzi, LoadPoseDb, ReadMat, UpdateNamespaceFromCli
from gt_generate.ground_truth.data_paths import (
    DEFAULT_FH_WAYMO_ROOT,
    DEFAULT_FH_WAYMO_SEQUENCE_NAME,
    ApplyResolvedDataPaths,
    BuildDefaultTerrainDir,
    BuildTimestampFile,
    ResolveTerrainOnlyDataPaths,
)
from gt_generate.common.open3d_debug_view import _RequireOpen3d, _UpdatePointCloudGeometry


def BuildArgs():
    return Namespace(
        # 输入数据根目录，用于读取当前帧原始 pcd。
        data_path=DEFAULT_FH_WAYMO_ROOT,
        sequence_name=DEFAULT_FH_WAYMO_SEQUENCE_NAME,
        pcd_dir="",
        semantic_label_dir="",
        pose_filename="",
        # terrain-only 目录，要求已经运行 export_frame_terrain_gt.py。
        terrain_dir="",
        # 指定 timestamp；小于 0 时使用 frame_index。
        timestamp=-1,
        # 指定 manifest 中的帧序号；timestamp 有效时忽略该参数。
        frame_index=0,
        # 点云抽样步长，真实数据点多时可调大。
        point_stride=1,
        # 兼容旧 frame_meta：如果 frame_meta 没写入这些范围，就使用这里的值。
        forward_range_m=80.0,
        backward_range_m=80.0,
        left_range_m=80.0,
        right_range_m=80.0,
        # 与逐帧 GT 导出保持一致：默认只显示到最后一个 semantic label 对应的 lidar 帧。
        b_clip_to_last_semantic_frame=True,
        # 仅显示 patch 范围内且有有效地形真值的点。
        b_show_only_points_with_terrain=True,
        # 点到地形的高度残差显示范围，单位 m。超过该范围会饱和显示。
        color_abs_height_m=0.5,
        # 残差绝对值小于该值的点显示为绿色，单位 m。
        close_height_m=0.15,
    )


def LoadManifest(terrain_dir):
    manifest_filename = os.path.join(terrain_dir, "terrain_only_frame_gt_manifest.json")
    if not os.path.exists(manifest_filename):
        raise FileNotFoundError(f"Frame GT manifest not found: {manifest_filename}")
    with open(manifest_filename, "r", encoding="utf-8") as fp:
        return json.load(fp)


def ResolveFrameItem(manifest, args):
    frame_items = list(manifest["frames"])
    return frame_items[ResolveStartFrameIndex(frame_items, args)]


def ResolveStartFrameIndex(frame_items, args):
    if int(args.timestamp) >= 0:
        for item_idx, frame_item in enumerate(frame_items):
            if int(frame_item["timestamp"]) == int(args.timestamp):
                return int(item_idx)
        raise ValueError(f"timestamp not found in manifest: {int(args.timestamp)}")
    if int(args.frame_index) < 0 or int(args.frame_index) >= len(frame_items):
        raise IndexError(f"frame_index out of range: {int(args.frame_index)}, frame_num={len(frame_items)}")
    return int(args.frame_index)


def ResolveLastSemanticFrameIndex(data_path):
    data_paths = ResolveTerrainOnlyDataPaths(
        Namespace(data_path=data_path, sequence_name="", pcd_dir="", semantic_label_dir="", pose_filename=""),
        b_require_semantic=True,
        b_require_pose=True,
    )
    pose_filename = data_paths.pose_filename
    semantic_dir = data_paths.semantic_label_dir
    if not os.path.isdir(semantic_dir):
        raise FileNotFoundError(f"Waymo semantic label dir not found: {semantic_dir}")
    timestamps, _ = LoadPoseDb(pose_filename)
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


def FilterFrameItemsBySemanticClip(manifest, args):
    frame_items = list(manifest["frames"])
    if not bool(args.b_clip_to_last_semantic_frame):
        return frame_items
    if "last_export_frame_index" in manifest:
        last_frame_idx = int(manifest["last_export_frame_index"])
        semantic_label_num = int(manifest.get("semantic_label_num", 0))
    else:
        last_frame_idx, semantic_label_num = ResolveLastSemanticFrameIndex(args.data_path)
    clipped_items = [item for item in frame_items if int(item["frame_index"]) <= last_frame_idx]
    print(
        "[TerrainGTAlignmentReplay] "
        f"semantic_label_num={semantic_label_num}, "
        f"last_semantic_frame_index={last_frame_idx}, "
        f"display_frame_num={len(clipped_items)}/{len(frame_items)}"
    )
    return clipped_items


def LoadFrameMeta(frame_dir):
    with open(os.path.join(frame_dir, "frame_meta.json"), "r", encoding="utf-8") as fp:
        return json.load(fp)


def GetFrameRange(frame_meta, args, key):
    return float(frame_meta.get(key, getattr(args, key)))


def LocalToPatchIndex(local_x, local_y, frame_meta, args):
    local_x = np.asarray(local_x, dtype=np.float64)
    local_y = np.asarray(local_y, dtype=np.float64)
    resolution_m = float(frame_meta["map_resolution_m"])
    forward_range_m = GetFrameRange(frame_meta, args, "forward_range_m")
    left_range_m = GetFrameRange(frame_meta, args, "left_range_m")
    row_idx = np.floor((forward_range_m - local_x) / resolution_m).astype(np.int32)
    col_idx = np.floor((left_range_m - local_y) / resolution_m).astype(np.int32)
    valid_mask = (
        (row_idx >= 0)
        & (row_idx < int(frame_meta["patch_rows"]))
        & (col_idx >= 0)
        & (col_idx < int(frame_meta["patch_cols"]))
    )
    safe_rows = np.clip(row_idx, 0, int(frame_meta["patch_rows"]) - 1)
    safe_cols = np.clip(col_idx, 0, int(frame_meta["patch_cols"]) - 1)
    return safe_rows, safe_cols, valid_mask


def BuildTerrainLocalPoints(terrain_map, frame_meta, args):
    valid_mask = np.isfinite(terrain_map) & (terrain_map != INVALID_HEIGHT)
    grid_rows, grid_cols = np.nonzero(valid_mask)
    if grid_rows.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    resolution_m = float(frame_meta["map_resolution_m"])
    forward_range_m = GetFrameRange(frame_meta, args, "forward_range_m")
    left_range_m = GetFrameRange(frame_meta, args, "left_range_m")
    local_x = forward_range_m - (grid_rows.astype(np.float64) + 0.5) * resolution_m
    local_y = left_range_m - (grid_cols.astype(np.float64) + 0.5) * resolution_m
    local_z = terrain_map[grid_rows, grid_cols].astype(np.float64)
    return np.column_stack((local_x, local_y, local_z))


def BuildHeightResidualColors(height_residual, close_height_m, color_abs_height_m):
    height_residual = np.asarray(height_residual, dtype=np.float64).reshape(-1)
    colors = np.zeros((height_residual.shape[0], 3), dtype=np.float64)
    close_mask = np.abs(height_residual) <= float(close_height_m)
    below_mask = height_residual < -float(close_height_m)
    above_mask = height_residual > float(close_height_m)
    colors[close_mask] = np.asarray((0.0, 0.9, 0.2), dtype=np.float64)
    if np.any(below_mask):
        alpha = np.clip((np.abs(height_residual[below_mask]) - float(close_height_m)) / max(float(color_abs_height_m), 1e-6), 0.0, 1.0)
        green = np.asarray((0.0, 0.9, 0.2), dtype=np.float64)
        blue = np.asarray((0.1, 0.35, 1.0), dtype=np.float64)
        colors[below_mask] = (1.0 - alpha[:, None]) * green[None, :] + alpha[:, None] * blue[None, :]
    if np.any(above_mask):
        alpha = np.clip((height_residual[above_mask] - float(close_height_m)) / max(float(color_abs_height_m), 1e-6), 0.0, 1.0)
        yellow = np.asarray((1.0, 0.85, 0.1), dtype=np.float64)
        red = np.asarray((1.0, 0.15, 0.05), dtype=np.float64)
        colors[above_mask] = (1.0 - alpha[:, None]) * yellow[None, :] + alpha[:, None] * red[None, :]
    return np.clip(colors, 0.0, 1.0)


def BuildFrameTerrainAlignmentRecord(args, frame_item):
    frame_dir = os.path.join(args.terrain_dir, frame_item["frame_dir"])
    frame_meta = LoadFrameMeta(frame_dir)
    terrain_map = ReadMat(os.path.join(frame_dir, "base", "terrain_completion", "terrain_z_inpainted.bin")).astype(np.float32)
    terrain_points = BuildTerrainLocalPoints(terrain_map, frame_meta, args)
    terrain_colors = np.tile(np.asarray((0.55, 0.55, 0.55), dtype=np.float64), (terrain_points.shape[0], 1))

    pcd_filename = BuildTimestampFile(args.pcd_dir, frame_meta["timestamp"], ".pcd")
    raw_points_xyzi = LoadPcdXyzi(pcd_filename)
    sampled_points = raw_points_xyzi[:: max(1, int(args.point_stride)), :3]
    row_idx, col_idx, in_patch_mask = LocalToPatchIndex(sampled_points[:, 0], sampled_points[:, 1], frame_meta, args)
    terrain_z_at_points = terrain_map[row_idx, col_idx].astype(np.float64)
    terrain_valid_mask = terrain_z_at_points != INVALID_HEIGHT
    point_valid_mask = in_patch_mask & terrain_valid_mask
    if not bool(args.b_show_only_points_with_terrain):
        point_valid_mask = in_patch_mask
    shown_points = sampled_points[point_valid_mask]
    shown_terrain_z = terrain_z_at_points[point_valid_mask]
    shown_terrain_valid = shown_terrain_z != INVALID_HEIGHT
    residual = np.zeros(shown_points.shape[0], dtype=np.float64)
    residual[shown_terrain_valid] = shown_points[shown_terrain_valid, 2] - shown_terrain_z[shown_terrain_valid]
    point_colors = BuildHeightResidualColors(residual, args.close_height_m, args.color_abs_height_m)
    point_colors[~shown_terrain_valid] = np.asarray((0.25, 0.25, 0.25), dtype=np.float64)

    valid_residual = residual[shown_terrain_valid]
    if valid_residual.shape[0] > 0:
        residual_mean = float(np.mean(valid_residual))
        residual_abs_p50 = float(np.percentile(np.abs(valid_residual), 50))
        residual_abs_p90 = float(np.percentile(np.abs(valid_residual), 90))
    else:
        residual_mean = 0.0
        residual_abs_p50 = 0.0
        residual_abs_p90 = 0.0
    return {
        "timestamp": int(frame_meta["timestamp"]),
        "terrain_points": terrain_points,
        "terrain_colors": terrain_colors,
        "points": shown_points,
        "colors": point_colors,
        "shown_point_num": int(shown_points.shape[0]),
        "valid_residual_num": int(valid_residual.shape[0]),
        "residual_mean": residual_mean,
        "residual_abs_p50": residual_abs_p50,
        "residual_abs_p90": residual_abs_p90,
    }


def ShowFrameTerrainGtAlignment(args):
    manifest = LoadManifest(args.terrain_dir)
    frame_items = FilterFrameItemsBySemanticClip(manifest, args)
    if len(frame_items) == 0:
        raise ValueError("Frame GT manifest has no frames.")
    o3d = _RequireOpen3d()
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="TerrainGTAlignmentReplay  E:next  Q:prev", width=1600, height=960, visible=True)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray((0.04, 0.04, 0.04), dtype=np.float64)
    render_option.point_size = 2.0

    state = {
        "frame_idx": ResolveStartFrameIndex(frame_items, args),
        "pending_step": 0,
        "initialized": False,
    }
    initial_record = BuildFrameTerrainAlignmentRecord(args, frame_items[state["frame_idx"]])
    terrain_cloud = o3d.geometry.PointCloud()
    point_cloud = o3d.geometry.PointCloud()
    _UpdatePointCloudGeometry(terrain_cloud, initial_record["terrain_points"], initial_record["terrain_colors"])
    _UpdatePointCloudGeometry(point_cloud, initial_record["points"], initial_record["colors"])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    visualizer.add_geometry(terrain_cloud)
    visualizer.add_geometry(point_cloud)
    visualizer.add_geometry(coordinate_frame)

    def UpdateFrame(vis):
        record = BuildFrameTerrainAlignmentRecord(args, frame_items[state["frame_idx"]])
        _UpdatePointCloudGeometry(terrain_cloud, record["terrain_points"], record["terrain_colors"])
        _UpdatePointCloudGeometry(point_cloud, record["points"], record["colors"])
        vis.update_geometry(terrain_cloud)
        vis.update_geometry(point_cloud)
        if not state["initialized"]:
            vis.reset_view_point(True)
            state["initialized"] = True
        if record["valid_residual_num"] > 0:
            print(
                "[TerrainGTAlignmentReplay] "
                f"frame={state['frame_idx'] + 1}/{len(frame_items)} "
                f"timestamp={record['timestamp']} "
                f"shown_points={record['shown_point_num']} "
                f"residual_mean={record['residual_mean']:.3f}m "
                f"residual_abs_p50={record['residual_abs_p50']:.3f}m "
                f"residual_abs_p90={record['residual_abs_p90']:.3f}m"
            )
        else:
            print(
                "[TerrainGTAlignmentReplay] "
                f"frame={state['frame_idx'] + 1}/{len(frame_items)} "
                f"timestamp={record['timestamp']} no point with valid terrain"
            )

    def NextFrame(_vis):
        state["pending_step"] = +1
        return False

    def PrevFrame(_vis):
        state["pending_step"] = -1
        return False

    visualizer.register_key_callback(ord("E"), NextFrame)
    visualizer.register_key_callback(ord("Q"), PrevFrame)

    try:
        UpdateFrame(visualizer)
        while True:
            if state["pending_step"] != 0:
                state["frame_idx"] = (state["frame_idx"] + state["pending_step"]) % len(frame_items)
                state["pending_step"] = 0
                UpdateFrame(visualizer)
            if not visualizer.poll_events():
                break
            visualizer.update_renderer()
    finally:
        visualizer.destroy_window()


def Main():
    args = BuildArgs()
    UpdateNamespaceFromCli(args, sys.argv)
    data_paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=True, b_require_pose=True)
    ApplyResolvedDataPaths(args, data_paths)
    if not str(args.terrain_dir).strip():
        args.terrain_dir = BuildDefaultTerrainDir(args)
    ShowFrameTerrainGtAlignment(args)


if __name__ == "__main__":
    Main()
