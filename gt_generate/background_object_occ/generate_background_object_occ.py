import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gt_generate.common.geometry import YprToRotationMatrixV2
from gt_generate.common.io_utils import EnsureDir, LoadPcdXyzi, LoadPoseDb, UpdateNamespaceFromCli
from gt_generate.ground_truth.data_paths import (
    DEFAULT_FH_WAYMO_ROOT,
    BuildTimestampFile,
    ListFhSequenceDirs,
    ResolveTerrainOnlyDataPaths,
)
from lyf_tools import visualize_point_cloud, voxel_downsample


DEFAULT_LABEL_MAPPING_PATH = PROJECT_ROOT / "Scene_Tag_Completion" / "label_mapping" / "waymo_merge_example.yaml"
DEFAULT_SEQUENCE_NAME = "16102220208346880_1420_000_1440_000"
DEFAULT_VOXEL_SIZE_XYZ = "0.2,0.2,0.2"
DEFAULT_OUTPUT_NAME = "background_occ"
EMPTY_LABEL = 0
SOURCE_SEMANTIC = 1
SOURCE_PSEUDO = 2
DEBUG_CLASS_FILTERED_LABEL = 1
DEBUG_CLUSTER_FILTERED_LABEL = 2
DEBUG_KEPT_LABEL = 3


def BuildArgs():
    return Namespace(
        # WAYMO_DATA_ROOT 根目录，或单条 sequence 目录。
        data_path=DEFAULT_FH_WAYMO_ROOT,
        # 单条序列名；传 all 时批处理 data_path 下所有 FH_waymo sequence。
        sequence_name=DEFAULT_SEQUENCE_NAME,
        pcd_dir="",
        semantic_label_dir="",
        pose_filename="",
        output_dir="",
        label_mapping_path=str(DEFAULT_LABEL_MAPPING_PATH),
        # 伪标签目录，相对 sequence/output。伪标签已经是映射后的训练标签空间。
        pseudo_label_dirname="label/pseudo_label_2",
        pseudo_label_dtype="uint32",
        # 3D OCC voxel 尺寸，单位 m。点云会先转到世界坐标系，再按这个尺寸聚合。
        voxel_size_xyz=DEFAULT_VOXEL_SIZE_XYZ,
        # 当前帧车体/LiDAR坐标系 3D OCC 范围: x_min,y_min,z_min,x_max,y_max,z_max。
        pc_range="-80,-80,-7,80,80,5.8",
        # 这些类别不会写入背景 OCC；其它 voxel 默认 EMPTY=0。
        ignore_class_names="UNKNOWN,VEHICLE,CYCLIST,PEDESTRIAN,ROAD,LOW_GROUND",
        # 对全局地图先过滤类别，再 0.05m 下采样，之后做点级聚类去掉零星虚点。
        b_filter_static_clusters=True,
        static_cluster_downsample_voxel_size_xyz="0.05,0.05,0.05",
        static_cluster_eps_m=0.6,
        static_cluster_min_points=5,
        # 调试用最大处理帧数；0 表示不限制。
        max_frames=0,
        # 只筛选最后逐帧输出；全序列点云仍参与静态世界地图融合。
        target_timestamps="",
    )


def BuildProgressBar(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def ParseFloatList(text, expected_num):
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != int(expected_num):
        raise ValueError(f"Expected {expected_num} comma-separated floats, got: {text}")
    return np.asarray(values, dtype=np.float64)


def ParseStringList(text):
    return [str(item).strip() for item in str(text).split(",") if str(item).strip()]


def LoadLabelMapping(label_mapping_path):
    with open(label_mapping_path, "r", encoding="utf-8") as fp:
        label_mapping_cfg = yaml.safe_load(fp)
    class_names = list(label_mapping_cfg["CLASS_NAMES"])
    learning_map = {int(key): int(value) for key, value in label_mapping_cfg["learning_map"].items()}
    max_src_label = max(learning_map.keys())
    label_remap_lut = np.zeros((max_src_label + 1,), dtype=np.int32)
    for src_label, dst_label in learning_map.items():
        label_remap_lut[int(src_label)] = int(dst_label)
    class_name_to_label = {str(class_name): int(class_idx) for class_idx, class_name in enumerate(class_names)}
    return class_names, label_remap_lut, class_name_to_label


def RemapWaymoLabels(semantic_labels, label_remap_lut):
    semantic_labels = np.asarray(semantic_labels, dtype=np.int32).reshape(-1)
    mapped_labels = np.zeros(semantic_labels.shape, dtype=np.int32)
    valid_mask = (semantic_labels >= 0) & (semantic_labels < label_remap_lut.shape[0])
    mapped_labels[valid_mask] = label_remap_lut[semantic_labels[valid_mask]]
    return mapped_labels


def BuildIgnoreLabelIds(ignore_class_names, class_name_to_label):
    ignore_label_ids = []
    for class_name in ParseStringList(ignore_class_names):
        if class_name not in class_name_to_label:
            raise KeyError(f"ignore class name not found in label mapping: {class_name}")
        ignore_label_ids.append(int(class_name_to_label[class_name]))
    return np.asarray(sorted(set(ignore_label_ids)), dtype=np.int32)


def BuildPseudoLabelPath(output_dir, args, timestamp):
    return BuildTimestampFile(os.path.join(output_dir, args.pseudo_label_dirname), timestamp, ".label")


def LoadFrameLabels(data_paths, output_dir, args, timestamp, raw_points_xyzi, label_remap_lut):
    semantic_filename = BuildTimestampFile(data_paths.semantic_label_dir, timestamp, ".npy")
    if os.path.exists(semantic_filename):
        labels = np.load(semantic_filename)
        if labels.shape[0] != raw_points_xyzi.shape[0]:
            raise ValueError(
                f"Waymo label/point count mismatch: {semantic_filename}, "
                f"labels={labels.shape[0]}, points={raw_points_xyzi.shape[0]}"
            )
        mapped_labels = RemapWaymoLabels(labels[:, 1].astype(np.int32), label_remap_lut)
        return mapped_labels, SOURCE_SEMANTIC, semantic_filename

    pseudo_filename = BuildPseudoLabelPath(output_dir, args, timestamp)
    if os.path.exists(pseudo_filename):
        mapped_labels = np.fromfile(pseudo_filename, dtype=np.dtype(str(args.pseudo_label_dtype))).astype(np.int32)
        if mapped_labels.shape[0] != raw_points_xyzi.shape[0]:
            raise ValueError(
                f"pseudo label/point count mismatch: {pseudo_filename}, "
                f"labels={mapped_labels.shape[0]}, points={raw_points_xyzi.shape[0]}"
            )
        return mapped_labels, SOURCE_PSEUDO, pseudo_filename

    return None, 0, ""


def TransformPointsToWorld(points_xyz, pose):
    rotation = YprToRotationMatrixV2([pose[5], pose[4], pose[3]])
    translation = np.asarray(pose[:3], dtype=np.float64)
    return points_xyz @ rotation.T + translation


def TransformPointsToLocal(points_world_xyz, pose):
    rotation = YprToRotationMatrixV2([pose[5], pose[4], pose[3]])
    translation = np.asarray(pose[:3], dtype=np.float64)
    return (points_world_xyz - translation.reshape(1, 3)) @ rotation


def AccumulateVoxelVotes(vote_dict, world_points_xyz, mapped_labels, source_types, voxel_size_xyz, class_num):
    voxel_indices = np.floor(world_points_xyz / voxel_size_xyz.reshape(1, 3)).astype(np.int64)
    label_column = mapped_labels.astype(np.int64).reshape(-1, 1)
    source_column = source_types.astype(np.int64).reshape(-1, 1)
    voxel_label_items = np.concatenate([voxel_indices, label_column, source_column], axis=1)
    unique_items, counts = np.unique(voxel_label_items, axis=0, return_counts=True)
    for item, count in zip(unique_items, counts):
        voxel_key = (int(item[0]), int(item[1]), int(item[2]))
        label_id = int(item[3])
        source_type = int(item[4])
        if label_id < 0 or label_id >= int(class_num):
            continue
        if voxel_key not in vote_dict:
            vote_dict[voxel_key] = [
                np.zeros((class_num,), dtype=np.uint32),
                np.zeros((class_num,), dtype=np.uint32),
            ]
        if source_type == SOURCE_SEMANTIC:
            vote_dict[voxel_key][0][label_id] += np.uint32(count)
        else:
            vote_dict[voxel_key][1][label_id] += np.uint32(count)


def SelectVoxelLabels(vote_dict):
    world_voxel_indices = []
    labels = []
    sources = []
    semantic_counts = []
    pseudo_counts = []
    total_counts = []
    for voxel_key, (semantic_votes, pseudo_votes) in vote_dict.items():
        semantic_total = int(np.sum(semantic_votes))
        pseudo_total = int(np.sum(pseudo_votes))
        if semantic_total > 0:
            label_id = int(np.argmax(semantic_votes))
            source = SOURCE_SEMANTIC
            selected_count = int(semantic_votes[label_id])
        elif pseudo_total > 0:
            label_id = int(np.argmax(pseudo_votes))
            source = SOURCE_PSEUDO
            selected_count = int(pseudo_votes[label_id])
        else:
            continue
        if label_id == EMPTY_LABEL:
            continue
        world_voxel_indices.append(voxel_key)
        labels.append(label_id)
        sources.append(source)
        semantic_counts.append(semantic_total)
        pseudo_counts.append(pseudo_total)
        total_counts.append(selected_count)

    if not world_voxel_indices:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros((0,), dtype=np.uint16),
            np.zeros((0,), dtype=np.uint8),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=np.uint32),
            np.zeros((0,), dtype=np.uint32),
        )
    return (
        np.asarray(world_voxel_indices, dtype=np.int64),
        np.asarray(labels, dtype=np.uint16),
        np.asarray(sources, dtype=np.uint8),
        np.asarray(semantic_counts, dtype=np.uint32),
        np.asarray(pseudo_counts, dtype=np.uint32),
        np.asarray(total_counts, dtype=np.uint32),
    )


def BuildPcRangeMask(points_xyz, pc_range):
    return (
        (points_xyz[:, 0] >= pc_range[0])
        & (points_xyz[:, 1] >= pc_range[1])
        & (points_xyz[:, 2] >= pc_range[2])
        & (points_xyz[:, 0] < pc_range[3])
        & (points_xyz[:, 1] < pc_range[4])
        & (points_xyz[:, 2] < pc_range[5])
    )


def BuildGridShape(pc_range, voxel_size_xyz):
    grid_size_xyz = (pc_range[3:] - pc_range[:3]) / voxel_size_xyz
    return np.ceil(grid_size_xyz).astype(np.int64)


def BuildFrameOccFromLocalPoints(local_points_xyz, point_labels, pc_range, voxel_size_xyz, grid_shape):
    range_mask = BuildPcRangeMask(local_points_xyz, pc_range)
    local_points_xyz = local_points_xyz[range_mask]
    point_labels = point_labels[range_mask]
    if local_points_xyz.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.uint16),
        )

    local_voxel_indices = np.floor(
        (local_points_xyz - pc_range[:3].reshape(1, 3)) / voxel_size_xyz.reshape(1, 3)
    ).astype(np.int32)
    in_grid_mask = (
        (local_voxel_indices[:, 0] >= 0)
        & (local_voxel_indices[:, 1] >= 0)
        & (local_voxel_indices[:, 2] >= 0)
        & (local_voxel_indices[:, 0] < grid_shape[0])
        & (local_voxel_indices[:, 1] < grid_shape[1])
        & (local_voxel_indices[:, 2] < grid_shape[2])
    )
    local_voxel_indices = local_voxel_indices[in_grid_mask]
    point_labels = point_labels[in_grid_mask]
    if local_voxel_indices.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.uint16),
        )

    _, selected_indices = np.unique(local_voxel_indices, axis=0, return_index=True)
    selected_indices = np.sort(selected_indices)
    return local_voxel_indices[selected_indices], point_labels[selected_indices].astype(np.uint16)


def FilterSingleLabelClusters(points_xyz, eps_m, min_points):
    if points_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if float(eps_m) <= 0.0 or int(min_points) <= 1:
        return np.ones((points_xyz.shape[0],), dtype=bool)

    if points_xyz.shape[0] < int(min_points):
        return np.zeros((points_xyz.shape[0],), dtype=bool)

    # 世界系同类别点 (N,3)。这里只需要 DBSCAN 的非噪声掩码，不需要簇编号。
    # 核心点的第 min_points 个邻居（含自身）必须在半径内；边界点邻接任一核心点。
    # Open3D 的半径查询采用严格小于，保留该边界语义。
    points = np.asarray(points_xyz, dtype=np.float64)
    distances, _ = cKDTree(points).query(points, k=[int(min_points)], workers=-1)
    b_core = distances[:, 0] < float(eps_m)
    if not np.any(b_core) or np.all(b_core):
        return b_core
    distances, _ = cKDTree(points[b_core]).query(points, k=1, workers=-1)
    return distances < float(eps_m)


def FilterStaticClusters(points_xyz, labels, sources, eps_m, min_points):
    if points_xyz.shape[0] == 0:
        return points_xyz, labels, sources, np.zeros((0, 3), dtype=points_xyz.dtype), 0
    keep_mask = np.zeros((points_xyz.shape[0],), dtype=bool)
    for label_id in np.unique(labels):
        label_mask = labels == label_id
        label_keep_mask = FilterSingleLabelClusters(points_xyz[label_mask], eps_m, min_points)
        keep_mask[np.flatnonzero(label_mask)[label_keep_mask]] = True
    removed_point_num = int(np.count_nonzero(~keep_mask))
    return points_xyz[keep_mask], labels[keep_mask], sources[keep_mask], points_xyz[~keep_mask], removed_point_num


def VisualizeFilterDebug(class_filtered_points_xyz, cluster_filtered_points_xyz, kept_points_xyz):
    point_blocks = []
    label_blocks = []
    if class_filtered_points_xyz.shape[0] > 0:
        point_blocks.append(class_filtered_points_xyz)
        label_blocks.append(
            np.full((class_filtered_points_xyz.shape[0],), DEBUG_CLASS_FILTERED_LABEL, dtype=np.int32)
        )
    if cluster_filtered_points_xyz.shape[0] > 0:
        point_blocks.append(cluster_filtered_points_xyz)
        label_blocks.append(
            np.full((cluster_filtered_points_xyz.shape[0],), DEBUG_CLUSTER_FILTERED_LABEL, dtype=np.int32)
        )
    if kept_points_xyz.shape[0] > 0:
        point_blocks.append(kept_points_xyz)
        label_blocks.append(np.full((kept_points_xyz.shape[0],), DEBUG_KEPT_LABEL, dtype=np.int32))
    if not point_blocks:
        return

    debug_points_xyz = np.concatenate(point_blocks, axis=0)
    debug_labels = np.concatenate(label_blocks, axis=0)
    # visualize_point_cloud(
    #     debug_points_xyz,
    #     debug_labels,
    #     label_colors={
    #         DEBUG_CLASS_FILTERED_LABEL: [0.55, 0.55, 0.55],
    #         DEBUG_CLUSTER_FILTERED_LABEL: [1.0, 0.0, 0.0],
    #         DEBUG_KEPT_LABEL: [0.0, 0.25, 1.0],
    #     },
    # )


def SaveFrameOcc(output_dir, timestamp, local_voxel_indices, labels, voxel_size_xyz, pc_range, grid_shape):
    EnsureDir(output_dir)
    np.savez_compressed(
        os.path.join(output_dir, f"{int(timestamp)}.npz"),
        voxel_indices=local_voxel_indices,
        labels=labels,
        voxel_size_xyz=voxel_size_xyz.astype(np.float32),
        pc_range=pc_range.astype(np.float32),
        grid_shape=grid_shape.astype(np.int64),
    )


def DownsampleGlobalMap(points_xyz, labels, sources, voxel_size_xyz):
    ## 全局地图点: points_xyz 为世界坐标系 (N, 3)，labels/sources 为 (N,)。
    if points_xyz.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.uint8),
        )
    downsampled_points_xyz, selected_indices, _ = voxel_downsample(
        points_xyz,
        voxel_size_xyz,
        return_mapping=True,
        pick_closest_to_center=True,
    )
    return (
        downsampled_points_xyz,
        labels[selected_indices].astype(np.int32, copy=False),
        sources[selected_indices].astype(np.uint8, copy=False),
    )


def BuildLabelHistogram(labels, class_names):
    label_histogram = {}
    for label_id in sorted(set(int(item) for item in labels.tolist())):
        label_name = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        label_histogram[label_name] = int(np.count_nonzero(labels == label_id))
    return label_histogram


def SaveFrameOccSummary(output_dir, class_names, voxel_size_xyz, pc_range, grid_shape, args, meta, frame_summaries, labels, sources):
    summary = {
        "schema_version": "background_object_frame_occ_v1",
        "description": "Per-frame sparse static background occupancy in current-frame LiDAR coordinates.",
        "output_dir": output_dir,
        "data_path": meta["data_path"],
        "sequence_name": meta["sequence_name"],
        "voxel_size_xyz": [float(item) for item in voxel_size_xyz],
        "pc_range": [float(item) for item in pc_range],
        "grid_shape": [int(item) for item in grid_shape],
        "global_static_voxel_num": int(labels.shape[0]),
        "global_filtered_downsample_voxel_num": int(meta["global_filtered_downsample_voxel_num"]),
        "global_semantic_source_voxel_num": int(np.count_nonzero(sources == SOURCE_SEMANTIC)),
        "global_pseudo_source_voxel_num": int(np.count_nonzero(sources == SOURCE_PSEUDO)),
        "global_label_histogram": BuildLabelHistogram(labels, class_names),
        "empty_label": int(EMPTY_LABEL),
        "source_semantic": int(SOURCE_SEMANTIC),
        "source_pseudo": int(SOURCE_PSEUDO),
        "class_names": list(class_names),
        "ignore_class_names": ParseStringList(args.ignore_class_names),
        "processed_frame_num": int(meta["processed_frame_num"]),
        "semantic_frame_num": int(meta["semantic_frame_num"]),
        "pseudo_frame_num": int(meta["pseudo_frame_num"]),
        "skipped_no_label_frame_num": int(meta["skipped_no_label_frame_num"]),
        "input_point_num": int(meta["input_point_num"]),
        "zero_label_point_num": int(meta["zero_label_point_num"]),
        "nonzero_point_num": int(meta["nonzero_point_num"]),
        "kept_point_num": int(meta["kept_point_num"]),
        "static_cluster_removed_point_num": int(meta["static_cluster_removed_point_num"]),
        "b_filter_static_clusters": bool(args.b_filter_static_clusters),
        "static_cluster_downsample_voxel_size_xyz": [
            float(item) for item in ParseFloatList(args.static_cluster_downsample_voxel_size_xyz, 3)
        ],
        "static_cluster_eps_m": float(args.static_cluster_eps_m),
        "static_cluster_min_points": int(args.static_cluster_min_points),
        "frame_occ": frame_summaries,
    }
    with open(os.path.join(output_dir, "frame_occ_summary.json"), "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)


def ResolveOutputDir(data_paths, args):
    if str(args.output_dir).strip():
        return str(args.output_dir)
    return os.path.join(data_paths.sequence_root, "output", "label", DEFAULT_OUTPUT_NAME)


def GenerateSequenceBackgroundObjectOcc(args):
    data_paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=False, b_require_pose=True)
    output_dir = os.path.dirname(data_paths.pcd_dir)
    occ_output_dir = ResolveOutputDir(data_paths, args)
    class_names, label_remap_lut, class_name_to_label = LoadLabelMapping(args.label_mapping_path)
    ignore_label_ids = BuildIgnoreLabelIds(args.ignore_class_names, class_name_to_label)
    voxel_size_xyz = ParseFloatList(args.voxel_size_xyz, 3)
    if np.any(voxel_size_xyz <= 0.0):
        raise ValueError(f"voxel_size_xyz must be positive, got {args.voxel_size_xyz}")
    cluster_voxel_size_xyz = ParseFloatList(args.static_cluster_downsample_voxel_size_xyz, 3)
    if np.any(cluster_voxel_size_xyz <= 0.0):
        raise ValueError(
            "static_cluster_downsample_voxel_size_xyz must be positive, "
            f"got {args.static_cluster_downsample_voxel_size_xyz}"
        )
    pc_range_text = str(getattr(args, "pc_range", "")).strip()
    if not pc_range_text:
        raise ValueError("pc_range must be set when exporting per-frame static OCC.")
    pc_range = ParseFloatList(pc_range_text, 6)
    grid_shape = BuildGridShape(pc_range, voxel_size_xyz)

    timestamps, poses = LoadPoseDb(data_paths.pose_filename)
    global_point_blocks = []
    global_label_blocks = []
    global_source_blocks = []
    frame_items = []
    meta = {
        "data_path": data_paths.sequence_root,
        "sequence_name": os.path.basename(data_paths.sequence_root),
        "processed_frame_num": 0,
        "semantic_frame_num": 0,
        "pseudo_frame_num": 0,
        "skipped_no_label_frame_num": 0,
        "input_point_num": 0,
        "zero_label_point_num": 0,
        "nonzero_point_num": 0,
        "kept_point_num": 0,
        "global_filtered_downsample_voxel_num": 0,
        "static_cluster_removed_point_num": 0,
    }

    progress_iter = BuildProgressBar(enumerate(timestamps), total=int(timestamps.shape[0]), desc=f"BuildRawWorldMap-{meta['sequence_name']}")
    for frame_idx, timestamp in progress_iter:
        if int(args.max_frames) > 0 and len(frame_items) >= int(args.max_frames):
            break
        pcd_filename = BuildTimestampFile(data_paths.pcd_dir, timestamp, ".pcd")
        if not os.path.exists(pcd_filename):
            continue
        raw_points_xyzi = LoadPcdXyzi(pcd_filename)
        mapped_labels, source_type, _ = LoadFrameLabels(data_paths, output_dir, args, timestamp, raw_points_xyzi, label_remap_lut)
        if mapped_labels is None:
            meta["skipped_no_label_frame_num"] += 1
            del raw_points_xyzi
            continue

        meta["processed_frame_num"] += 1
        meta["input_point_num"] += int(raw_points_xyzi.shape[0])
        frame_items.append((int(timestamp), poses[frame_idx].copy()))
        if int(source_type) == SOURCE_SEMANTIC:
            meta["semantic_frame_num"] += 1
        else:
            meta["pseudo_frame_num"] += 1

        nonzero_mask = mapped_labels != EMPTY_LABEL
        meta["zero_label_point_num"] += int(np.count_nonzero(~nonzero_mask))
        meta["nonzero_point_num"] += int(np.count_nonzero(nonzero_mask))
        if not np.any(nonzero_mask):
            del raw_points_xyzi, mapped_labels, nonzero_mask
            continue

        nonzero_points_xyzi = raw_points_xyzi[nonzero_mask]
        nonzero_labels = mapped_labels[nonzero_mask]
        nonzero_world_points_xyz = TransformPointsToWorld(
            nonzero_points_xyzi[:, :3].astype(np.float64),
            poses[frame_idx],
        )
        source_types = np.full(nonzero_labels.shape, int(source_type), dtype=np.uint8)
        global_point_blocks.append(nonzero_world_points_xyz.astype(np.float32, copy=False))
        global_label_blocks.append(nonzero_labels.astype(np.int32, copy=False))
        global_source_blocks.append(source_types)
        del raw_points_xyzi
        del mapped_labels, nonzero_mask, nonzero_points_xyzi, nonzero_labels
        del nonzero_world_points_xyz, source_types

    if not frame_items:
        raise RuntimeError(f"No labeled frames found for sequence: {data_paths.sequence_root}")

    if global_point_blocks:
        global_world_points_xyz = np.concatenate(global_point_blocks, axis=0)
        global_labels = np.concatenate(global_label_blocks, axis=0)
        global_sources = np.concatenate(global_source_blocks, axis=0)
    else:
        global_world_points_xyz = np.zeros((0, 3), dtype=np.float32)
        global_labels = np.zeros((0,), dtype=np.int32)
        global_sources = np.zeros((0,), dtype=np.uint8)
    del global_point_blocks, global_label_blocks, global_source_blocks

    static_mask = ~np.isin(global_labels, ignore_label_ids)
    class_filtered_world_points_xyz = global_world_points_xyz[~static_mask]
    static_world_points_xyz = global_world_points_xyz[static_mask]
    static_labels = global_labels[static_mask]
    static_sources = global_sources[static_mask]
    del global_world_points_xyz, global_labels, global_sources, static_mask
    # visualize_point_cloud(static_world_points_xyz, static_labels)

    static_world_points_xyz, static_labels, static_sources = DownsampleGlobalMap(
        static_world_points_xyz,
        static_labels,
        static_sources,
        cluster_voxel_size_xyz,
    )
    # visualize_point_cloud(static_world_points_xyz, static_labels)
    meta["global_filtered_downsample_voxel_num"] = int(static_world_points_xyz.shape[0])

    cluster_filtered_world_points_xyz = np.zeros((0, 3), dtype=static_world_points_xyz.dtype)
    if bool(args.b_filter_static_clusters):
        (
            static_world_points_xyz,
            static_labels,
            static_sources,
            cluster_filtered_world_points_xyz,
            removed_point_num,
        ) = FilterStaticClusters(
            static_world_points_xyz,
            static_labels,
            static_sources,
            float(args.static_cluster_eps_m),
            int(args.static_cluster_min_points),
        )
        meta["static_cluster_removed_point_num"] = int(removed_point_num)
        # visualize_point_cloud(static_world_points_xyz, static_labels)
    # VisualizeFilterDebug(class_filtered_world_points_xyz, cluster_filtered_world_points_xyz, static_world_points_xyz)
    del class_filtered_world_points_xyz, cluster_filtered_world_points_xyz
    meta["kept_point_num"] = int(static_world_points_xyz.shape[0])

    vote_dict = {}
    if static_world_points_xyz.shape[0] > 0:
        AccumulateVoxelVotes(
            vote_dict,
            static_world_points_xyz,
            static_labels,
            static_sources,
            voxel_size_xyz,
            len(class_names),
        )
    world_voxel_indices, labels, sources, _, _, _ = SelectVoxelLabels(vote_dict)
    world_voxel_centers_xyz = (world_voxel_indices.astype(np.float64) + 0.5) * voxel_size_xyz.reshape(1, 3)
    vote_dict.clear()
    del vote_dict, static_world_points_xyz, static_labels, static_sources
    # visualize_point_cloud(world_voxel_centers_xyz, labels)

    frame_summaries = []
    target_text = str(getattr(args, "target_timestamps", "")).strip()
    target_timestamps = {int(value) for value in target_text.split(",")} if target_text else None
    frame_iter = BuildProgressBar(frame_items, total=len(frame_items), desc=f"SaveFrameOcc-{meta['sequence_name']}")
    for timestamp, pose in frame_iter:
        if target_timestamps is not None and int(timestamp) not in target_timestamps:
            continue
        local_voxel_centers_xyz = TransformPointsToLocal(world_voxel_centers_xyz, pose)
        local_voxel_indices, frame_labels = BuildFrameOccFromLocalPoints(
            local_voxel_centers_xyz,
            labels,
            pc_range,
            voxel_size_xyz,
            grid_shape,
        )

        SaveFrameOcc(
            occ_output_dir,
            timestamp,
            local_voxel_indices,
            frame_labels,
            voxel_size_xyz,
            pc_range,
            grid_shape,
        )
        frame_summaries.append(
            {
                "timestamp": int(timestamp),
                "filename": f"{int(timestamp)}.npz",
                "occupied_voxel_num": int(frame_labels.shape[0]),
                "label_histogram": BuildLabelHistogram(frame_labels, class_names),
            }
        )

    if target_timestamps is not None and {frame["timestamp"] for frame in frame_summaries} != target_timestamps:
        raise ValueError("OCC target export incomplete; check timestamps and max_frames")
    SaveFrameOccSummary(
        occ_output_dir,
        class_names,
        voxel_size_xyz,
        pc_range,
        grid_shape,
        args,
        meta,
        frame_summaries,
        labels,
        sources,
    )
    print(
        f"[BackgroundObjectOcc] saved {len(frame_summaries)} frame OCC files, "
        f"global_static_voxels={int(labels.shape[0])}: {occ_output_dir}"
    )


def GenerateAllSequences(args):
    sequence_dirs = ListFhSequenceDirs(str(args.data_path))
    if not sequence_dirs:
        raise FileNotFoundError(f"No FH_waymo sequence dirs found under {args.data_path}")
    for sequence_dir in sequence_dirs:
        sequence_args = Namespace(**vars(args))
        sequence_args.sequence_name = os.path.basename(sequence_dir)
        if str(args.output_dir).strip():
            sequence_args.output_dir = os.path.join(str(args.output_dir), sequence_args.sequence_name)
        else:
            sequence_args.output_dir = ""
        GenerateSequenceBackgroundObjectOcc(sequence_args)


def Main():
    args = BuildArgs()
    UpdateNamespaceFromCli(args, sys.argv)
    if str(args.sequence_name).strip().lower() == "all":
        GenerateAllSequences(args)
    else:
        GenerateSequenceBackgroundObjectOcc(args)


if __name__ == "__main__":
    Main()
