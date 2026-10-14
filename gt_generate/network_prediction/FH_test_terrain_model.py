import os
import sys
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

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
from gt_generate.ground_truth.debug_frame_terrain_gt_alignment import (
    BuildHeightResidualColors,
    BuildTerrainLocalPoints,
    FilterFrameItemsBySemanticClip,
    LoadFrameMeta,
    LoadManifest,
    LocalToPatchIndex,
    ResolveStartFrameIndex,
)
from gt_generate.network_prediction.training_data import BuildTerrainOnlyInputArgs, TerrainOnlyObservationInputBuilder
from gt_generate.network_prediction.training_model import TerrainOnlyNet
from gt_generate.common.open3d_debug_view import _RequireOpen3d, _UpdatePointCloudGeometry


def BuildArgs():
    return Namespace(
        data_path=DEFAULT_FH_WAYMO_ROOT,
        sequence_name=DEFAULT_FH_WAYMO_SEQUENCE_NAME,
        pcd_dir="",
        semantic_label_dir="",
        pose_filename="",
        terrain_dir="",
        checkpoint_filename="",
        replay_mode="auto",
        timestamp=-1,
        frame_index=0,
        point_stride=1,
        input_point_stride=1,
        map_resolution_m=0.2,
        forward_range_m=80.0,
        backward_range_m=80.0,
        left_range_m=80.0,
        right_range_m=80.0,
        b_cache_observation_inputs=True,
        b_clip_to_last_semantic_frame=True,
        b_show_only_points_with_terrain=True,
        color_abs_height_m=0.5,
        close_height_m=0.15,
        device="cuda",
    )


def BuildInputTensor(input_dict, input_field_names, device):
    tensor_items = []
    for field_name in input_field_names:
        tensor_items.append(torch.from_numpy(input_dict[field_name][np.newaxis, np.newaxis, ...]).to(device=device, dtype=torch.float32))
    return torch.cat(tensor_items, dim=1)


def LoadModel(checkpoint_filename, device):
    checkpoint = torch.load(checkpoint_filename, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    first_weight = state_dict["enc1.conv.block.0.weight"]
    input_channel_num = int(first_weight.shape[1])
    hidden_channels = int(first_weight.shape[0])
    model = TerrainOnlyNet(input_channel_num=input_channel_num, hidden_channels=hidden_channels).to(device=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, list(checkpoint.get("input_field_names", [])), input_channel_num


def BuildFrameMeta(timestamp, pose_xyz_rpy, args):
    patch_rows = int(round((float(args.forward_range_m) + float(args.backward_range_m)) / float(args.map_resolution_m)))
    patch_cols = int(round((float(args.left_range_m) + float(args.right_range_m)) / float(args.map_resolution_m)))
    return {
        "timestamp": int(timestamp),
        "pose_xyz_rpy": [float(item) for item in pose_xyz_rpy],
        "map_resolution_m": float(args.map_resolution_m),
        "patch_rows": int(patch_rows),
        "patch_cols": int(patch_cols),
        "forward_range_m": float(args.forward_range_m),
        "backward_range_m": float(args.backward_range_m),
        "left_range_m": float(args.left_range_m),
        "right_range_m": float(args.right_range_m),
    }


def LoadRawFrameItems(args):
    lidar_dir = args.pcd_dir
    if not os.path.isdir(lidar_dir):
        raise FileNotFoundError(f"pcd dir not found: {lidar_dir}")
    pcd_timestamps = []
    for filename in sorted(os.listdir(lidar_dir)):
        if not filename.endswith(".pcd"):
            continue
        try:
            pcd_timestamps.append(int(os.path.splitext(filename)[0]))
        except ValueError:
            continue
    if not pcd_timestamps:
        raise RuntimeError(f"No pcd files found under {lidar_dir}")

    pose_filename = args.pose_filename
    pose_dict = {}
    if os.path.exists(pose_filename):
        pose_timestamps, poses = LoadPoseDb(pose_filename)
        pose_dict = {int(timestamp): poses[item_idx] for item_idx, timestamp in enumerate(pose_timestamps)}

    frame_items = []
    for frame_idx, timestamp in enumerate(pcd_timestamps):
        pose_xyz_rpy = pose_dict.get(int(timestamp), np.zeros(6, dtype=np.float64))
        frame_items.append(
            {
                "frame_index": int(frame_idx),
                "timestamp": int(timestamp),
                "frame_meta": BuildFrameMeta(timestamp, pose_xyz_rpy, args),
                "frame_dir": "",
            }
        )
    return frame_items


def LoadReplayFrameItems(args):
    b_manifest_exists = os.path.exists(os.path.join(args.terrain_dir, "terrain_only_frame_gt_manifest.json"))
    if str(args.replay_mode) == "manifest" or (str(args.replay_mode) == "auto" and b_manifest_exists):
        manifest = LoadManifest(args.terrain_dir)
        frame_items = FilterFrameItemsBySemanticClip(manifest, args)
        if len(frame_items) == 0:
            raise ValueError("Frame GT manifest has no frames.")
        for frame_item in frame_items:
            frame_dir = os.path.join(args.terrain_dir, frame_item["frame_dir"])
            frame_item["frame_meta"] = LoadFrameMeta(frame_dir)
        return frame_items, True
    if str(args.replay_mode) not in ("auto", "raw"):
        raise ValueError(f"Unsupported replay_mode: {args.replay_mode}")
    return LoadRawFrameItems(args), False


def BuildPredictionRecord(args, model, input_builder, input_field_names, frame_item, b_use_manifest):
    frame_meta = frame_item["frame_meta"]
    frame_dir = os.path.join(args.terrain_dir, frame_item["frame_dir"]) if bool(b_use_manifest) else ""
    if bool(b_use_manifest):
        input_dict = input_builder.BuildObservationInputDict(frame_dir, input_field_names)
    else:
        input_dict = input_builder.BuildObservationMaps(frame_meta)
        input_dict = {field_name: input_dict[field_name] for field_name in input_field_names}
    input_tensor = BuildInputTensor(input_dict, input_field_names, args.device)
    with torch.no_grad():
        pred_map = model(input_tensor)["terrain_z_inpainted"][0, 0].detach().cpu().numpy().astype(np.float32)

    pred_terrain_points = BuildTerrainLocalPoints(pred_map, frame_meta, args)
    pred_terrain_colors = np.tile(np.asarray((0.55, 0.55, 0.55), dtype=np.float64), (pred_terrain_points.shape[0], 1))

    pcd_filename = BuildTimestampFile(args.pcd_dir, frame_meta["timestamp"], ".pcd")
    raw_points_xyzi = LoadPcdXyzi(pcd_filename)
    sampled_points = raw_points_xyzi[:: max(1, int(args.point_stride)), :3]
    row_idx, col_idx, in_patch_mask = LocalToPatchIndex(sampled_points[:, 0], sampled_points[:, 1], frame_meta, args)
    terrain_z_at_points = pred_map[row_idx, col_idx].astype(np.float64)
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

    gt_filename = os.path.join(frame_dir, "base", "terrain_completion", "terrain_z_inpainted.bin") if frame_dir else ""
    if gt_filename and os.path.exists(gt_filename):
        gt_map = ReadMat(gt_filename).astype(np.float32)
        gt_valid_mask = gt_map != INVALID_HEIGHT
        pred_valid_mask = pred_map != INVALID_HEIGHT
        eval_mask = gt_valid_mask & pred_valid_mask
        terrain_mae = float(np.mean(np.abs(pred_map[eval_mask] - gt_map[eval_mask]))) if np.any(eval_mask) else None
    else:
        terrain_mae = None
    valid_residual = residual[shown_terrain_valid]
    return {
        "timestamp": int(frame_meta["timestamp"]),
        "terrain_points": pred_terrain_points,
        "terrain_colors": pred_terrain_colors,
        "points": shown_points,
        "colors": point_colors,
        "shown_point_num": int(shown_points.shape[0]),
        "terrain_mae": terrain_mae,
        "residual_abs_p50": float(np.percentile(np.abs(valid_residual), 50)) if valid_residual.shape[0] > 0 else 0.0,
        "residual_abs_p90": float(np.percentile(np.abs(valid_residual), 90)) if valid_residual.shape[0] > 0 else 0.0,
    }


def ShowPredictionReplay(args):
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    frame_items, b_use_manifest = LoadReplayFrameItems(args)
    input_args = BuildTerrainOnlyInputArgs()
    input_args.input_point_stride = int(args.input_point_stride)
    input_args.b_cache_observation_inputs = bool(args.b_cache_observation_inputs)
    input_args.data_path = str(args.data_path)
    input_args.pcd_dir = str(args.pcd_dir)
    input_builder = TerrainOnlyObservationInputBuilder(args.terrain_dir, input_args)
    input_field_names = list(input_args.default_input_field_names)
    model, checkpoint_input_field_names, input_channel_num = LoadModel(args.checkpoint_filename, args.device)
    if checkpoint_input_field_names:
        input_field_names = checkpoint_input_field_names
    else:
        input_field_names = list(input_args.default_input_field_names)[:input_channel_num]

    o3d = _RequireOpen3d()
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="TerrainOnlyModelTest  E:next  Q:prev  W:terrain-only", width=1600, height=960, visible=True)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray((0.04, 0.04, 0.04), dtype=np.float64)
    render_option.point_size = 2.0

    state = {"frame_idx": ResolveStartFrameIndex(frame_items, args), "pending_step": 0, "initialized": False, "b_show_points": True}
    initial_record = BuildPredictionRecord(args, model, input_builder, input_field_names, frame_items[state["frame_idx"]], b_use_manifest)
    state["record"] = initial_record
    terrain_cloud = o3d.geometry.PointCloud()
    point_cloud = o3d.geometry.PointCloud()
    _UpdatePointCloudGeometry(terrain_cloud, initial_record["terrain_points"], initial_record["terrain_colors"])
    _UpdatePointCloudGeometry(point_cloud, initial_record["points"], initial_record["colors"])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
    visualizer.add_geometry(terrain_cloud)
    visualizer.add_geometry(point_cloud)
    visualizer.add_geometry(coordinate_frame)

    def ApplyPointVisibility(vis, record):
        if bool(state["b_show_points"]):
            _UpdatePointCloudGeometry(point_cloud, record["points"], record["colors"])
        else:
            _UpdatePointCloudGeometry(point_cloud, np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64))
        vis.update_geometry(point_cloud)

    def UpdateFrame(vis):
        record = BuildPredictionRecord(args, model, input_builder, input_field_names, frame_items[state["frame_idx"]], b_use_manifest)
        state["record"] = record
        _UpdatePointCloudGeometry(terrain_cloud, record["terrain_points"], record["terrain_colors"])
        vis.update_geometry(terrain_cloud)
        ApplyPointVisibility(vis, record)
        if not state["initialized"]:
            vis.reset_view_point(True)
            state["initialized"] = True
        terrain_mae_text = "none" if record["terrain_mae"] is None else f"{record['terrain_mae']:.3f}m"
        print(
            "[TerrainOnlyModelTest] "
            f"mode={'manifest' if b_use_manifest else 'raw'} "
            f"display={'terrain+points' if bool(state['b_show_points']) else 'terrain_only'} "
            f"frame={state['frame_idx'] + 1}/{len(frame_items)} "
            f"timestamp={record['timestamp']} "
            f"shown_points={record['shown_point_num']} "
            f"terrain_mae={terrain_mae_text} "
            f"point_abs_p50={record['residual_abs_p50']:.3f}m "
            f"point_abs_p90={record['residual_abs_p90']:.3f}m"
        )

    def NextFrame(_vis):
        state["pending_step"] = +1
        return False

    def PrevFrame(_vis):
        state["pending_step"] = -1
        return False

    def TogglePointCloud(vis):
        state["b_show_points"] = not bool(state["b_show_points"])
        ApplyPointVisibility(vis, state["record"])
        print(f"[TerrainOnlyModelTest] display={'terrain+points' if bool(state['b_show_points']) else 'terrain_only'}")
        return False

    visualizer.register_key_callback(ord("E"), NextFrame)
    visualizer.register_key_callback(ord("Q"), PrevFrame)
    visualizer.register_key_callback(ord("W"), TogglePointCloud)
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
    data_paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=False, b_require_pose=False)
    ApplyResolvedDataPaths(args, data_paths)
    if not str(args.terrain_dir).strip():
        args.terrain_dir = BuildDefaultTerrainDir(args)
    if not str(args.checkpoint_filename).strip():
        args.checkpoint_filename = os.path.join(args.terrain_dir, "training_runs", "terrain_only_v1", "checkpoints", "best.pt")
    ShowPredictionReplay(args)


if __name__ == "__main__":
    Main()
