import argparse
import glob
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.io_utils import LoadMapInfo, LoadPcdXyzi, LoadPoseDb, ReadMat
from gt_generate.ground_truth.data_paths import BuildTimestampFile, FindExistingFile
from gt_generate.ground_truth.export_frame_terrain_gt import (
    BuildArgs,
    BuildFrameTerrainGt,
    LoadGlobalTerrainMaps,
    SampleMap,
    WorldToGridIndex,
)


TERRAIN_FIELD_NAMES = (
    "terrain_z_inpainted",
    "terrain_inpainted_mask",
    "terrain_train_weight",
    "terrain_inpaint_distance",
)


def ParseArgs():
    parser = argparse.ArgumentParser(
        description="Audit terrain/OCC geometry on a real high-roll/pitch three-task frame."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("WAYMO_DATA_ROOT", ""),
        help="FH Waymo data root. Defaults to WAYMO_DATA_ROOT.",
    )
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument(
        "--info-path",
        default="",
        help="Anchor info pickle relative to data root. Defaults from --split.",
    )
    parser.add_argument(
        "--sequence-name",
        default="",
        help="Optional sequence restriction. Without timestamp, the highest roll/pitch frame is selected.",
    )
    parser.add_argument("--timestamp-ms", type=int, default=-1, help="Optional exact frame timestamp in ms.")
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--occ-sample-size", type=int, default=50000)
    parser.add_argument("--output", default="", help="Optional JSON output filename.")
    return parser.parse_args()


def LoadInfos(data_root, info_path):
    info_filename = os.path.join(data_root, info_path)
    with open(info_filename, "rb") as file:
        infos = pickle.load(file)
    if not isinstance(infos, list):
        raise TypeError(f"info file must contain a list: {info_filename}")
    return infos


def BuildFramePaths(data_root, sequence_name, timestamp_ms):
    sequence_root = os.path.join(data_root, sequence_name)
    terrain_dir = os.path.join(sequence_root, "output", "label", "ground_truth")
    frame_dir = os.path.join(terrain_dir, "frames", f"{timestamp_ms:013d}")
    return sequence_root, terrain_dir, frame_dir


def LoadFrameMetadata(data_root, sequence_name, timestamp_ms):
    _, _, frame_dir = BuildFramePaths(data_root, sequence_name, timestamp_ms)
    metadata_filename = os.path.join(frame_dir, "frame_meta.json")
    with open(metadata_filename, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return metadata, metadata_filename


def SelectAuditFrame(args, infos):
    if args.timestamp_ms >= 0 and not args.sequence_name:
        raise ValueError("--timestamp-ms requires --sequence-name")
    if args.timestamp_ms >= 0:
        metadata, metadata_filename = LoadFrameMetadata(
            args.data_root,
            args.sequence_name,
            args.timestamp_ms,
        )
        return args.sequence_name, int(args.timestamp_ms), metadata, metadata_filename

    candidates = []
    for info in infos:
        sequence_name = str(info["sequence_name"])
        if args.sequence_name and sequence_name != args.sequence_name:
            continue
        timestamp_ms = int(info["frame_timestamp_micros"]) // 1000
        try:
            metadata, metadata_filename = LoadFrameMetadata(args.data_root, sequence_name, timestamp_ms)
        except FileNotFoundError:
            continue
        pose = np.asarray(metadata.get("pose_xyz_rpy", []), dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            continue
        attitude_score = float(np.hypot(pose[3], pose[4]))
        candidates.append((attitude_score, sequence_name, timestamp_ms, metadata, metadata_filename))
    if not candidates:
        raise RuntimeError("no frame with valid pose metadata matched the requested index/sequence")
    _, sequence_name, timestamp_ms, metadata, metadata_filename = max(candidates, key=lambda item: item[0])
    return sequence_name, timestamp_ms, metadata, metadata_filename


def BuildExportArgs(sequence_root, terrain_dir, metadata):
    args = BuildArgs()
    output_dir = os.path.join(sequence_root, "output")
    args.data_path = sequence_root
    args.sequence_name = ""
    args.pcd_dir = os.path.join(output_dir, "pcd")
    args.semantic_label_dir = os.path.join(output_dir, "SemanticSegmentation")
    args.pose_filename = FindExistingFile(
        [
            os.path.join(output_dir, "pose_slam_euler_waymo.txt"),
            os.path.join(output_dir, "pose_slam_euler.txt"),
            os.path.join(output_dir, "pose_global.txt"),
        ]
    )
    args.terrain_dir = terrain_dir
    for key in (
        "forward_range_m",
        "backward_range_m",
        "left_range_m",
        "right_range_m",
        "map_resolution_m",
    ):
        setattr(args, key, float(metadata[key]))
    return args


def LoadStoredTerrain(frame_dir):
    terrain_dir = os.path.join(frame_dir, "base", "terrain_completion")
    return {
        field_name: ReadMat(os.path.join(terrain_dir, f"{field_name}.bin"))
        for field_name in TERRAIN_FIELD_NAMES
    }


def CompareTerrainArrays(stored_terrain, recomputed_terrain):
    comparison = {}
    for field_name in TERRAIN_FIELD_NAMES:
        stored = stored_terrain[field_name]
        recomputed = recomputed_terrain[field_name]
        if stored.shape != recomputed.shape or stored.dtype != recomputed.dtype:
            comparison[field_name] = {
                "array_equal": False,
                "stored_shape": list(stored.shape),
                "recomputed_shape": list(recomputed.shape),
                "stored_dtype": str(stored.dtype),
                "recomputed_dtype": str(recomputed.dtype),
            }
            continue
        difference = np.abs(stored.astype(np.float64) - recomputed.astype(np.float64))
        finite_mask = np.isfinite(difference)
        comparison[field_name] = {
            "array_equal": bool(np.array_equal(stored, recomputed, equal_nan=True)),
            "different_cells": int(np.count_nonzero(difference > 0.0)),
            "max_abs_diff": float(np.max(difference[finite_mask])) if np.any(finite_mask) else None,
        }
    return comparison


def BuildYawOnlyTerrain(global_maps, pose, map_info, args, patch_rows, patch_cols):
    row_indices = np.arange(patch_rows, dtype=np.float64)
    col_indices = np.arange(patch_cols, dtype=np.float64)
    local_x = float(args.forward_range_m) - (row_indices + 0.5) * float(args.map_resolution_m)
    local_y = float(args.left_range_m) - (col_indices + 0.5) * float(args.map_resolution_m)
    grid_x, grid_y = np.meshgrid(local_x, local_y, indexing="ij")
    cos_yaw = np.cos(float(pose[5]))
    sin_yaw = np.sin(float(pose[5]))
    world_x = float(pose[0]) + cos_yaw * grid_x - sin_yaw * grid_y
    world_y = float(pose[1]) + sin_yaw * grid_x + cos_yaw * grid_y
    map_min_x_cm, map_min_y_cm, rows, cols = map_info
    grid_rows, grid_cols, in_bounds = WorldToGridIndex(
        world_x,
        world_y,
        map_min_x_cm,
        map_min_y_cm,
        rows,
        cols,
        args.map_resolution_m,
    )
    terrain = SampleMap(
        global_maps["terrain_z_inpainted"],
        grid_rows,
        grid_cols,
        in_bounds,
        INVALID_HEIGHT,
    ).astype(np.float32)
    valid_mask = terrain != INVALID_HEIGHT
    terrain[valid_mask] -= float(pose[2])
    return terrain


def BuildPointMapIndices(points, terrain_shape, args):
    patch_rows, patch_cols = terrain_shape
    row_indices = np.floor(
        (float(args.forward_range_m) - points[:, 0]) / float(args.map_resolution_m)
    ).astype(np.int64)
    col_indices = np.floor(
        (float(args.left_range_m) - points[:, 1]) / float(args.map_resolution_m)
    ).astype(np.int64)
    in_patch = (
        (row_indices >= 0)
        & (row_indices < patch_rows)
        & (col_indices >= 0)
        & (col_indices < patch_cols)
    )
    safe_rows = np.clip(row_indices, 0, patch_rows - 1)
    safe_cols = np.clip(col_indices, 0, patch_cols - 1)
    return safe_rows, safe_cols, in_patch


def ComputePointResidualMetrics(points, terrain, args):
    safe_rows, safe_cols, in_patch = BuildPointMapIndices(points, terrain.shape, args)
    terrain_z = terrain[safe_rows, safe_cols].astype(np.float64)
    valid_mask = in_patch & np.isfinite(terrain_z) & (terrain_z != INVALID_HEIGHT)
    absolute_residual = np.abs(points[valid_mask, 2] - terrain_z[valid_mask])
    if absolute_residual.size == 0:
        return {"count": 0}
    return {
        "count": int(absolute_residual.size),
        "abs_p10_m": float(np.percentile(absolute_residual, 10)),
        "abs_p25_m": float(np.percentile(absolute_residual, 25)),
        "abs_p50_m": float(np.percentile(absolute_residual, 50)),
        "abs_p90_m": float(np.percentile(absolute_residual, 90)),
        "fraction_le_0_15m": float(np.mean(absolute_residual <= 0.15)),
        "fraction_le_0_30m": float(np.mean(absolute_residual <= 0.30)),
    }


def ComputeCellMinimumResidualMetrics(points, terrain, args):
    safe_rows, safe_cols, in_patch = BuildPointMapIndices(points, terrain.shape, args)
    terrain_z = terrain[safe_rows, safe_cols].astype(np.float64)
    valid_mask = in_patch & np.isfinite(terrain_z) & (terrain_z != INVALID_HEIGHT)
    point_keys = safe_rows[valid_mask] * terrain.shape[1] + safe_cols[valid_mask]
    residuals = np.abs(points[valid_mask, 2] - terrain_z[valid_mask])
    if residuals.size == 0:
        return {"cells": 0}
    order = np.argsort(point_keys)
    point_keys = point_keys[order]
    residuals = residuals[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(point_keys)) + 1]
    minimum_residuals = np.minimum.reduceat(residuals, group_starts)
    return {
        "cells": int(minimum_residuals.size),
        "abs_p50_m": float(np.percentile(minimum_residuals, 50)),
        "abs_p90_m": float(np.percentile(minimum_residuals, 90)),
        "fraction_le_0_15m": float(np.mean(minimum_residuals <= 0.15)),
        "fraction_le_0_30m": float(np.mean(minimum_residuals <= 0.30)),
    }


def ComputeOCCMetrics(sequence_root, timestamp_ms, points, terrain, args, sample_size):
    occ_filename = os.path.join(
        sequence_root,
        "output",
        "label",
        "background_occ",
        f"{timestamp_ms}.npz",
    )
    with np.load(occ_filename, allow_pickle=False) as occ_file:
        indices = occ_file["voxel_indices"].astype(np.float64)
        labels = occ_file["labels"]
        voxel_size = occ_file["voxel_size_xyz"].astype(np.float64)
        pc_range = occ_file["pc_range"].astype(np.float64)
    centers = pc_range[:3] + (indices + 0.5) * voxel_size
    safe_rows, safe_cols, in_patch = BuildPointMapIndices(centers, terrain.shape, args)
    terrain_z = terrain[safe_rows, safe_cols].astype(np.float64)
    valid_mask = in_patch & np.isfinite(terrain_z) & (terrain_z != INVALID_HEIGHT)
    height_from_terrain = centers[valid_mask, 2] - terrain_z[valid_mask]

    selected_count = min(max(0, int(sample_size)), centers.shape[0])
    if selected_count > 0 and points.shape[0] > 0:
        selected_indices = np.random.default_rng(0).choice(
            centers.shape[0],
            size=selected_count,
            replace=False,
        )
        nearest_distance = cKDTree(points).query(centers[selected_indices], k=1, workers=1)[0]
    else:
        nearest_distance = np.zeros((0,), dtype=np.float64)

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    metrics = {
        "voxel_count": int(centers.shape[0]),
        "class_counts": {
            str(int(label)): int(count) for label, count in zip(unique_labels, label_counts)
        },
        "terrain_comparable_count": int(height_from_terrain.size),
    }
    if height_from_terrain.size > 0:
        metrics.update(
            {
                "center_minus_terrain_z_p01_m": float(np.percentile(height_from_terrain, 1)),
                "center_minus_terrain_z_p50_m": float(np.percentile(height_from_terrain, 50)),
                "fraction_center_below_terrain_by_more_than_0_2m": float(
                    np.mean(height_from_terrain < -0.2)
                ),
            }
        )
    if nearest_distance.size > 0:
        metrics.update(
            {
                "sampled_center_count": int(nearest_distance.size),
                "sampled_center_to_current_point_distance_p50_m": float(
                    np.percentile(nearest_distance, 50)
                ),
                "sampled_center_to_current_point_distance_p90_m": float(
                    np.percentile(nearest_distance, 90)
                ),
                "sampled_center_within_0_35m_fraction": float(np.mean(nearest_distance <= 0.35)),
            }
        )
    return metrics


def CheckSplitOverlap(data_root):
    sequence_sets = {}
    for split in ("training", "validation"):
        info_filename = os.path.join(data_root, f"LYF_waymo_infos_{split}_semantic.pkl")
        if not os.path.isfile(info_filename):
            return {"available": False, "missing_info_file": info_filename}
        with open(info_filename, "rb") as file:
            infos = pickle.load(file)
        sequence_sets[split] = {str(info["sequence_name"]) for info in infos}
    return {
        "available": True,
        "training_sequence_count": len(sequence_sets["training"]),
        "validation_sequence_count": len(sequence_sets["validation"]),
        "overlap_sequences": sorted(sequence_sets["training"] & sequence_sets["validation"]),
    }


def GetExporterCommit():
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "-1",
                "--format=%H|%cs",
                "--",
                "gt_generate/ground_truth/export_frame_terrain_gt.py",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        commit_hash, commit_date = result.stdout.strip().split("|", 1)
        return commit_hash, commit_date
    except Exception:
        return "", ""


def FindBatchDoneRecord(data_root, sequence_name):
    pattern = os.path.join(
        data_root,
        ".three_task_gt_runs",
        "*",
        sequence_name,
        "terrain_frames.done.json",
    )
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not candidates:
        return {}
    done_filename = candidates[0]
    with open(done_filename, "r", encoding="utf-8") as file:
        done_record = json.load(file)
    return {
        "batch_done_record": done_filename,
        "batch_done_mtime": os.path.getmtime(done_filename),
        "batch_done_contents": done_record,
    }


def ReadRecordedCommit(terrain_dir, metadata):
    candidate_keys = ("git_commit", "commit_hash", "code_commit", "source_commit")
    for key in candidate_keys:
        if key in metadata and str(metadata[key]).strip():
            return str(metadata[key]).strip()
    for filename in (
        "terrain_only_frame_gt_manifest.json",
        "terrain_only_frame_gt_summary.json",
        "terrain_only_summary.json",
    ):
        full_filename = os.path.join(terrain_dir, filename)
        if not os.path.isfile(full_filename):
            continue
        with open(full_filename, "r", encoding="utf-8") as file:
            record = json.load(file)
        for key in candidate_keys:
            if key in record and str(record[key]).strip():
                return str(record[key]).strip()
    return ""


def BuildAudit(args):
    if not args.data_root:
        raise ValueError("--data-root or WAYMO_DATA_ROOT is required")
    info_path = args.info_path or f"LYF_waymo_infos_{args.split}_semantic.pkl"
    infos = LoadInfos(args.data_root, info_path)
    sequence_name, timestamp_ms, metadata, metadata_filename = SelectAuditFrame(args, infos)
    sequence_root, terrain_dir, frame_dir = BuildFramePaths(
        args.data_root,
        sequence_name,
        timestamp_ms,
    )
    export_args = BuildExportArgs(sequence_root, terrain_dir, metadata)
    timestamps, poses = LoadPoseDb(export_args.pose_filename)
    matching_indices = np.flatnonzero(timestamps == timestamp_ms)
    if matching_indices.size != 1:
        raise ValueError(f"timestamp {timestamp_ms} matched {matching_indices.size} poses")
    frame_index = int(matching_indices[0])
    pose = poses[frame_index]
    metadata_pose = np.asarray(metadata["pose_xyz_rpy"], dtype=np.float64)

    map_info = LoadMapInfo(os.path.join(terrain_dir, "map_info.txt"))
    global_maps = LoadGlobalTerrainMaps(terrain_dir)
    stored_terrain = LoadStoredTerrain(frame_dir)
    recomputed_terrain = BuildFrameTerrainGt(
        global_maps,
        pose,
        map_info,
        export_args,
        int(metadata["patch_rows"]),
        int(metadata["patch_cols"]),
    )["base/terrain_completion"]
    recompute_comparison = CompareTerrainArrays(stored_terrain, recomputed_terrain)
    yaw_only_terrain = BuildYawOnlyTerrain(
        global_maps,
        pose,
        map_info,
        export_args,
        int(metadata["patch_rows"]),
        int(metadata["patch_cols"]),
    )
    stored_height = stored_terrain["terrain_z_inpainted"]
    common_valid = (stored_height != INVALID_HEIGHT) & (yaw_only_terrain != INVALID_HEIGHT)
    yaw_only_difference = np.abs(
        stored_height[common_valid].astype(np.float64)
        - yaw_only_terrain[common_valid].astype(np.float64)
    )

    pcd_filename = BuildTimestampFile(export_args.pcd_dir, timestamp_ms, ".pcd")
    points = LoadPcdXyzi(pcd_filename)[:: max(1, int(args.point_stride)), :3].astype(np.float64)
    exporter_commit, exporter_commit_date = GetExporterCommit()
    recorded_commit = ReadRecordedCommit(terrain_dir, metadata)
    b_source_version_certain = bool(recorded_commit) and exporter_commit.startswith(recorded_commit)
    b_geometry_consistent = all(
        comparison.get("array_equal", False) for comparison in recompute_comparison.values()
    )

    audit = {
        "selected_frame": {
            "split": args.split,
            "info_path": info_path,
            "sequence_name": sequence_name,
            "timestamp_ms": int(timestamp_ms),
            "frame_index": frame_index,
            "frame_metadata": metadata_filename,
            "pose_xyz_rpy": [float(value) for value in pose],
            "roll_pitch_norm_rad": float(np.hypot(pose[3], pose[4])),
            "roll_pitch_norm_deg": float(np.degrees(np.hypot(pose[3], pose[4]))),
            "metadata_pose_exact_match": bool(np.array_equal(metadata_pose, pose)),
        },
        "current_export_recompute": recompute_comparison,
        "yaw_only_contrast": {
            "common_valid_cells": int(yaw_only_difference.size),
            "abs_diff_p50_m": float(np.percentile(yaw_only_difference, 50)),
            "abs_diff_p90_m": float(np.percentile(yaw_only_difference, 90)),
            "abs_diff_max_m": float(np.max(yaw_only_difference)),
        },
        "point_alignment": {
            "stored_full_pose": ComputePointResidualMetrics(points, stored_height, export_args),
            "yaw_only_reference": ComputePointResidualMetrics(points, yaw_only_terrain, export_args),
            "stored_cell_min": ComputeCellMinimumResidualMetrics(points, stored_height, export_args),
            "yaw_only_cell_min": ComputeCellMinimumResidualMetrics(points, yaw_only_terrain, export_args),
        },
        "occ_alignment": ComputeOCCMetrics(
            sequence_root,
            timestamp_ms,
            points,
            stored_height,
            export_args,
            args.occ_sample_size,
        ),
        "split_check": CheckSplitOverlap(args.data_root),
        "provenance": {
            "current_exporter_commit": exporter_commit,
            "current_exporter_commit_date": exporter_commit_date,
            "recorded_export_commit": recorded_commit,
            "commit_hash_recorded_in_export_metadata": bool(recorded_commit),
            **FindBatchDoneRecord(args.data_root, sequence_name),
        },
        "verdict": {
            "geometry_consistent": b_geometry_consistent,
            "source_version_certain": b_source_version_certain,
            "note": (
                "Stored terrain arrays reproduce exactly with the current full-roll/pitch/yaw exporter. "
                "Source version is certain only when the export metadata records a matching commit."
            ),
        },
    }
    return audit


def Main():
    args = ParseArgs()
    audit = BuildAudit(args)
    audit_text = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(audit_text + "\n", encoding="utf-8")
        print(f"saved audit: {output_path}")
    print(audit_text)


if __name__ == "__main__":
    Main()
