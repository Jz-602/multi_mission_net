import os
import sys
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import yaml
from scipy.spatial import cKDTree

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.geometry import YprToRotationMatrixV2
from gt_generate.common.io_utils import (
    EnsureDir,
    LoadPcdXyzi,
    LoadPoseDb,
    SaveJson,
    SaveMapInfo,
    UpdateNamespaceFromCli,
    WriteMat,
)
from gt_generate.ground_truth.analyzer import (
    BuildFinalSeedMap,
    BuildInpaintDistanceMap,
    BuildRawTerrainMaps,
    BuildReliableSeedMap,
    BuildTrajectorySeedMask,
    CalculateSlopeDeg,
    CompleteTerrain,
    GrowRoiFromTrajectory,
)
from gt_generate.ground_truth.block_store import TerrainBlockStore
from gt_generate.ground_truth.data_paths import (
    DEFAULT_FH_WAYMO_ROOT,
    DEFAULT_FH_WAYMO_SEQUENCE_NAME,
    ApplyResolvedDataPaths,
    BuildDefaultTerrainDir,
    BuildTimestampFile,
    ResolveTerrainOnlyDataPaths,
)


TERRAIN_GROUND_UNKNOWN = 0
TERRAIN_GROUND_ROAD = 1
TERRAIN_GROUND_ROAD2 = 2
DEFAULT_LABEL_MAPPING_PATH = os.path.join(
    PROJECT_ROOT,
    "Scene_Tag_Completion",
    "label_mapping",
    "waymo_merge_example.yaml",
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
        # data_path 指向 FH_waymo_small 根目录时必须指定；data_path 已经指向单条序列时可留空。
        sequence_name=DEFAULT_FH_WAYMO_SEQUENCE_NAME,
        # 可选：直接覆盖 FH output 下的点云、语义标签和 pose 路径。
        pcd_dir="",
        semantic_label_dir="",
        pose_filename="",
        # terrain-only 输出目录，包含 map_info、各层 bin 地图和 block 缓存。
        output_dir="",
        # 2D 地形栅格分辨率，单位 m。越小细节越多，但地图更大、BGK 更慢。
        map_resolution_m=0.2,
        # 小块地图边长，单位为栅格数。只影响分块存储和大范围数据内存占用，不改变结果语义。
        block_rows=200,
        # 单帧点云只使用雷达坐标系 xy 距离小于该值的点，单位 m。
        point_max_range_m=80.0,
        # 当前帧车体/LiDAR坐标系 3D 范围: x_min,y_min,z_min,x_max,y_max,z_max；不为空时优先于 point_max_range_m。
        pc_range="-80,-80,-7,80,80,5.8",
        # 只有伪标签的帧按该位移间隔采样；有人工语义标注的帧必定加入。
        keyframe_sampling_distance_m=0.5,
        # 语义标签映射配置。人工 Waymo 原始标签会先映射到该训练标签空间，伪标签已经在该空间内。
        label_mapping_path=DEFAULT_LABEL_MAPPING_PATH,
        # 伪标签目录，相对 sequence/output；只有缺少人工 SemanticSegmentation 的帧才读取。
        pseudo_label_dirname="label/pseudo_label_2",
        pseudo_label_dtype="uint32",
        # 当前训练标签空间中被认为是地面的类别名。
        ground_class_names="ROAD,LOW_GROUND",
        # raw 阶段 seed 的最大 z 标准差，单位 m。只按 std 过滤，不再使用 count 阈值。
        raw_seed_std_max_m=0.18,
        # 第一遍 BGK 补全半径，单位 m，用于生成临时参考地形 terrain_z_provisional。
        provisional_bgk_radius_m=1.5,
        # 第一遍 BGK 每个待补全栅格附近所需的最少有效 seed 数。
        provisional_bgk_min_valid=3,
        # 第一遍 BGK bilateral 高度相似性尺度，单位 m。越小越不容易跨高度突变传播。
        provisional_bgk_bilateral_std=0.05,
        # 是否启用 C++ 风格的轨迹种子 ROI 生长。Waymo 标签主线默认不启用。
        b_use_trajectory_roi=False,
        # 轨迹 ROI 生长时允许通过的最大坡度角，单位 degree；仅 b_use_trajectory_roi=True 时使用。
        slope_angle_thresh_deg=15.0,
        # 最终 seed 的最大 raw z 标准差，单位 m。该值越小，最终地形越可靠但覆盖更小。
        final_seed_std_max_m=0.16,
        # raw 高度与 provisional 地形的最大允许差值，单位 m，用于剔除明显不连续或异常的 raw 栅格。
        final_seed_raw_provisional_delta_max_m=0.25,
        # 最终 BGK 补全半径，单位 m，决定 terrain_z_inpainted 的外扩和平滑范围。
        final_bgk_radius_m=3.0,
        # 最终 BGK 每个待补全栅格附近所需的最少有效 seed 数。
        final_bgk_min_valid=5,
        # 最终 BGK bilateral 高度相似性尺度，单位 m。越小越保守，越大越平滑但可能跨障碍传播。
        final_bgk_bilateral_std=0.05,
    )


def ParseStringList(text):
    return [str(item).strip() for item in str(text).split(",") if str(item).strip()]


def ParseFloatList(text, expected_num):
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) != int(expected_num):
        raise ValueError(f"Expected {expected_num} comma-separated floats, got: {text}")
    return np.asarray(values, dtype=np.float64)


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
    return label_remap_lut, class_name_to_label


def RemapWaymoLabels(semantic_labels, label_remap_lut):
    semantic_labels = np.asarray(semantic_labels, dtype=np.int32).reshape(-1)
    mapped_labels = np.zeros(semantic_labels.shape, dtype=np.int32)
    valid_mask = (semantic_labels >= 0) & (semantic_labels < label_remap_lut.shape[0])
    mapped_labels[valid_mask] = label_remap_lut[semantic_labels[valid_mask]]
    return mapped_labels


def BuildGroundLabelConfig(args, class_name_to_label):
    ground_label_ids = []
    for class_name in ParseStringList(args.ground_class_names):
        if class_name not in class_name_to_label:
            raise KeyError(f"ground class name not found in label mapping: {class_name}")
        ground_label_ids.append(class_name_to_label[class_name])
    return {
        "ground_label_ids": np.asarray(ground_label_ids, dtype=np.int32),
        "road_label_id": int(class_name_to_label["ROAD"]),
        "low_ground_label_id": int(class_name_to_label["LOW_GROUND"]),
    }


def FilterPointsByRange(points_xyz, max_range_m):
    if points_xyz.shape[0] == 0:
        return points_xyz
    xy_range = np.sqrt(points_xyz[:, 0] * points_xyz[:, 0] + points_xyz[:, 1] * points_xyz[:, 1])
    return points_xyz[xy_range <= float(max_range_m)]


def BuildPointRangeMask(points_xyz, args):
    pc_range_text = str(getattr(args, "pc_range", "")).strip()
    if pc_range_text:
        pc_range = ParseFloatList(pc_range_text, 6)
        return (
            (points_xyz[:, 0] >= pc_range[0])
            & (points_xyz[:, 1] >= pc_range[1])
            & (points_xyz[:, 2] >= pc_range[2])
            & (points_xyz[:, 0] <= pc_range[3])
            & (points_xyz[:, 1] <= pc_range[4])
            & (points_xyz[:, 2] <= pc_range[5])
        )
    xy_range = np.sqrt(points_xyz[:, 0] * points_xyz[:, 0] + points_xyz[:, 1] * points_xyz[:, 1])
    return xy_range <= float(args.point_max_range_m)


def TransformPoints(points_xyz, pose):
    rotation = YprToRotationMatrixV2([pose[5], pose[4], pose[3]])
    translation = np.asarray(pose[:3], dtype=np.float64)
    return points_xyz @ rotation.T + translation


def BuildPseudoLabelPath(args, timestamp):
    output_dir = os.path.dirname(args.pcd_dir)
    return BuildTimestampFile(os.path.join(output_dir, args.pseudo_label_dirname), timestamp, ".label")


def SelectGroundPoints(args, raw_points_xyzi, mapped_labels, ground_label_config):
    mapped_labels = np.asarray(mapped_labels, dtype=np.int32).reshape(-1)
    if mapped_labels.shape[0] != raw_points_xyzi.shape[0]:
        raise ValueError(f"label/point count mismatch: labels={mapped_labels.shape[0]}, points={raw_points_xyzi.shape[0]}")
    ground_mask = np.isin(mapped_labels, ground_label_config["ground_label_ids"])
    ground_points = raw_points_xyzi[ground_mask, :3].astype(np.float64)
    ground_labels = mapped_labels[ground_mask]
    ground_type = np.zeros((ground_labels.shape[0],), dtype=np.uint8)
    ground_type[ground_labels == ground_label_config["road_label_id"]] = TERRAIN_GROUND_ROAD
    ground_type[ground_labels == ground_label_config["low_ground_label_id"]] = TERRAIN_GROUND_ROAD2
    range_mask = BuildPointRangeMask(ground_points, args)
    return ground_points[range_mask], ground_type[range_mask]


def LoadSemanticGroundPoints(args, timestamp, raw_points_xyzi, label_remap_lut, ground_label_config):
    label_filename = BuildTimestampFile(args.semantic_label_dir, timestamp, ".npy")
    if not os.path.exists(label_filename):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.uint8)
    labels = np.load(label_filename)
    if labels.shape[0] != raw_points_xyzi.shape[0]:
        raise ValueError(f"Waymo label/point count mismatch: {label_filename}, labels={labels.shape[0]}, points={raw_points_xyzi.shape[0]}")
    semantic_labels = labels[:, 1].astype(np.int32)
    mapped_labels = RemapWaymoLabels(semantic_labels, label_remap_lut)
    return SelectGroundPoints(args, raw_points_xyzi, mapped_labels, ground_label_config)


def LoadPseudoGroundPoints(args, timestamp, raw_points_xyzi, ground_label_config):
    pseudo_label_filename = BuildPseudoLabelPath(args, timestamp)
    if not os.path.exists(pseudo_label_filename):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.uint8)
    mapped_labels = np.fromfile(pseudo_label_filename, dtype=np.dtype(str(args.pseudo_label_dtype))).astype(np.int32)
    if mapped_labels.shape[0] != raw_points_xyzi.shape[0]:
        raise ValueError(
            f"pseudo label/point count mismatch: {pseudo_label_filename}, "
            f"labels={mapped_labels.shape[0]}, points={raw_points_xyzi.shape[0]}"
        )
    return SelectGroundPoints(args, raw_points_xyzi, mapped_labels, ground_label_config)


def BuildTerrainGroundTypeRawMap(stats):
    ground_type_map = np.zeros((int(stats["rows"]), int(stats["cols"])), dtype=np.uint8)
    road_mask = stats["road_count_map"] > 0
    road2_mask = stats["road2_count_map"] > 0
    ground_type_map[road_mask] = TERRAIN_GROUND_ROAD
    # 对齐旧 C++ RefineTerrainPointCloud：同一 cell 中 road2 优先于 road。
    ground_type_map[road2_mask] = TERRAIN_GROUND_ROAD2
    return ground_type_map


def BuildTerrainGroundTypeMap(raw_ground_type_map, final_seed_map, final_map, map_resolution_m, fill_radius_m):
    output_map = np.zeros(final_map.shape, dtype=np.uint8)
    seed_label_mask = (final_seed_map != INVALID_HEIGHT) & (raw_ground_type_map != TERRAIN_GROUND_UNKNOWN)
    output_map[seed_label_mask] = raw_ground_type_map[seed_label_mask]
    fill_mask = (final_map != INVALID_HEIGHT) & (output_map == TERRAIN_GROUND_UNKNOWN)
    if not np.any(seed_label_mask) or not np.any(fill_mask):
        return output_map

    seed_rows, seed_cols = np.nonzero(seed_label_mask)
    fill_rows, fill_cols = np.nonzero(fill_mask)
    seed_xy = np.column_stack((seed_cols.astype(np.float64), seed_rows.astype(np.float64))) * float(map_resolution_m)
    fill_xy = np.column_stack((fill_cols.astype(np.float64), fill_rows.astype(np.float64))) * float(map_resolution_m)
    kdtree = cKDTree(seed_xy)
    distances, indices = kdtree.query(fill_xy, k=1, workers=-1)
    assign_mask = distances <= float(fill_radius_m)
    output_map[fill_rows[assign_mask], fill_cols[assign_mask]] = raw_ground_type_map[seed_rows[indices[assign_mask]], seed_cols[indices[assign_mask]]]
    return output_map


def AccumulateWaymoGroundBlocks(args):
    timestamps, poses = LoadPoseDb(args.pose_filename)
    label_remap_lut, class_name_to_label = LoadLabelMapping(args.label_mapping_path)
    ground_label_config = BuildGroundLabelConfig(args, class_name_to_label)
    block_store = TerrainBlockStore(args.output_dir, args.map_resolution_m, args.block_rows)

    accepted_num = 0
    semantic_accepted_num = 0
    pseudo_accepted_num = 0
    semantic_candidate_num = 0
    pseudo_candidate_num = 0
    pseudo_skipped_by_distance_num = 0
    ground_point_num = 0
    semantic_ground_point_num = 0
    pseudo_ground_point_num = 0
    last_pseudo_accepted_position = None
    for frame_idx, timestamp in BuildProgressBar(enumerate(timestamps), total=int(timestamps.shape[0]), desc="TerrainOnlyWaymoReplay"):
        label_filename = BuildTimestampFile(args.semantic_label_dir, timestamp, ".npy")
        pseudo_label_filename = BuildPseudoLabelPath(args, timestamp)
        b_has_semantic = os.path.exists(label_filename)
        b_has_pseudo = os.path.exists(pseudo_label_filename)
        if b_has_semantic:
            semantic_candidate_num += 1
        elif b_has_pseudo:
            pseudo_candidate_num += 1
        else:
            continue

        current_position = poses[frame_idx, :2]
        if (not b_has_semantic) and last_pseudo_accepted_position is not None:
            move_distance = float(np.linalg.norm(current_position - last_pseudo_accepted_position))
            if move_distance < float(args.keyframe_sampling_distance_m):
                pseudo_skipped_by_distance_num += 1
                continue
        pcd_filename = BuildTimestampFile(args.pcd_dir, timestamp, ".pcd")
        if not os.path.exists(pcd_filename):
            continue
        raw_points_xyzi = LoadPcdXyzi(pcd_filename)
        if b_has_semantic:
            ground_points, ground_type = LoadSemanticGroundPoints(args, timestamp, raw_points_xyzi, label_remap_lut, ground_label_config)
        else:
            ground_points, ground_type = LoadPseudoGroundPoints(args, timestamp, raw_points_xyzi, ground_label_config)
        if ground_points.shape[0] == 0:
            continue
        global_ground_points = TransformPoints(ground_points, poses[frame_idx])
        block_store.AddWorldPoints(global_ground_points, ground_type)
        accepted_num += 1
        ground_point_num += int(global_ground_points.shape[0])
        if b_has_semantic:
            semantic_accepted_num += 1
            semantic_ground_point_num += int(global_ground_points.shape[0])
        else:
            pseudo_accepted_num += 1
            pseudo_ground_point_num += int(global_ground_points.shape[0])
            last_pseudo_accepted_position = current_position.copy()

        if 0:
            # PyCharm debug: inspect one accepted keyframe after Waymo semantic filtering.
            # Gray = raw points; green = Waymo ground labels.
            from gt_generate.ground_truth.debug_view import ShowTerrainOnlyPointCloud

            ShowTerrainOnlyPointCloud(
                f"WaymoGroundFrame timestamp={int(timestamp)}",
                [
                    (raw_points_xyzi[:, :3], (0.45, 0.45, 0.45)),
                    (ground_points[:, :3], (0.0, 0.9, 0.2)),
                ],
            )

    block_store.SaveBlocks()
    return block_store, {
        "timestamp_num": int(timestamps.shape[0]),
        "accepted_keyframe_num": int(accepted_num),
        "semantic_candidate_frame_num": int(semantic_candidate_num),
        "pseudo_candidate_frame_num": int(pseudo_candidate_num),
        "semantic_accepted_frame_num": int(semantic_accepted_num),
        "pseudo_accepted_frame_num": int(pseudo_accepted_num),
        "pseudo_skipped_by_distance_num": int(pseudo_skipped_by_distance_num),
        "ground_point_num": int(ground_point_num),
        "semantic_ground_point_num": int(semantic_ground_point_num),
        "pseudo_ground_point_num": int(pseudo_ground_point_num),
        "ground_label_ids": [int(item) for item in ground_label_config["ground_label_ids"]],
        "ground_class_names": ParseStringList(args.ground_class_names),
        "pseudo_label_dirname": str(args.pseudo_label_dirname),
        "pseudo_label_dtype": str(args.pseudo_label_dtype),
        "keyframe_sampling_distance_m": float(args.keyframe_sampling_distance_m),
        "pc_range": ParseFloatList(args.pc_range, 6).tolist() if str(getattr(args, "pc_range", "")).strip() else None,
        "timestamps": timestamps,
        "poses": poses,
    }


def SaveTerrainOnlyOutputs(args, stats, output_maps, meta):
    EnsureDir(args.output_dir)
    SaveMapInfo(args.output_dir + "/map_info.txt", stats["map_min_x_cm"], stats["map_min_y_cm"], stats["rows"], stats["cols"])
    for name, layer_map in output_maps.items():
        WriteMat(os.path.join(args.output_dir, f"{name}.bin"), layer_map)
    distance_valid_mask = output_maps["terrain_inpaint_distance"] != INVALID_HEIGHT
    summary = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "map_resolution_m": float(args.map_resolution_m),
        "block_rows": int(args.block_rows),
        "map_min_x_cm": int(stats["map_min_x_cm"]),
        "map_min_y_cm": int(stats["map_min_y_cm"]),
        "rows": int(stats["rows"]),
        "cols": int(stats["cols"]),
        "raw_valid_cell_num": int(np.count_nonzero(output_maps["terrain_z_raw"] != INVALID_HEIGHT)),
        "raw_seed_cell_num": int(np.count_nonzero(output_maps["terrain_z_raw_seed"] != INVALID_HEIGHT)),
        "roi_enabled": bool(args.b_use_trajectory_roi),
        "roi_cell_num": int(np.count_nonzero(output_maps["terrain_roi_mask"])),
        "final_seed_cell_num": int(np.count_nonzero(output_maps["terrain_z_final_seed"] != INVALID_HEIGHT)),
        "final_valid_cell_num": int(np.count_nonzero(output_maps["terrain_z_inpainted"] != INVALID_HEIGHT)),
        "terrain_ground_type_road_cell_num": int(np.count_nonzero(output_maps["terrain_ground_type_map"] == TERRAIN_GROUND_ROAD)) if "terrain_ground_type_map" in output_maps else 0,
        "terrain_ground_type_road2_cell_num": int(np.count_nonzero(output_maps["terrain_ground_type_map"] == TERRAIN_GROUND_ROAD2)) if "terrain_ground_type_map" in output_maps else 0,
        "inpaint_distance_mean_m": float(np.mean(output_maps["terrain_inpaint_distance"][distance_valid_mask])) if np.any(distance_valid_mask) else 0.0,
        "inpaint_distance_max_m": float(np.max(output_maps["terrain_inpaint_distance"][distance_valid_mask])) if np.any(distance_valid_mask) else 0.0,
        "waymo": {key: value for key, value in meta.items() if key not in {"timestamps", "poses"}},
    }
    SaveJson(os.path.join(args.output_dir, "terrain_only_summary.json"), summary)


def Main():
    args = BuildArgs()
    UpdateNamespaceFromCli(args, sys.argv)
    data_paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=True, b_require_pose=True)
    ApplyResolvedDataPaths(args, data_paths)
    if not str(args.output_dir).strip():
        args.output_dir = BuildDefaultTerrainDir(args)
    EnsureDir(args.output_dir)

    block_store, meta = AccumulateWaymoGroundBlocks(args)
    stats = block_store.BuildDenseStats()
    raw_mean_map, raw_std_map = BuildRawTerrainMaps(stats)
    terrain_ground_type_raw_map = BuildTerrainGroundTypeRawMap(stats)

    if 0:
        # PyCharm debug: inspect raw Waymo terrain statistics before any smoothing.
        from gt_generate.ground_truth.debug_view import ShowTerrainOnlyMaps

        ShowTerrainOnlyMaps(
            "TerrainOnly-RawStats",
            [
                ("terrain_z_raw", raw_mean_map, INVALID_HEIGHT),
                ("terrain_z_std", raw_std_map, INVALID_HEIGHT),
            ],
        )

    raw_seed_map = BuildReliableSeedMap(raw_mean_map, raw_std_map, args.raw_seed_std_max_m)
    provisional_map, _ = CompleteTerrain(
        raw_seed_map,
        args.map_resolution_m,
        args.provisional_bgk_radius_m,
        args.provisional_bgk_min_valid,
        args.provisional_bgk_bilateral_std,
    )

    if 0:
        # PyCharm debug: inspect strict raw seed and first BGK completion.
        from gt_generate.ground_truth.debug_view import ShowTerrainOnlyMaps

        ShowTerrainOnlyMaps(
            "TerrainOnly-ProvisionalCompletion",
            [
                ("terrain_z_raw_seed", raw_seed_map, INVALID_HEIGHT),
                ("terrain_z_provisional", provisional_map, INVALID_HEIGHT),
            ],
        )

    slope_map = CalculateSlopeDeg(provisional_map, args.map_resolution_m)
    if bool(args.b_use_trajectory_roi):
        trajectory_seed_mask = BuildTrajectorySeedMask(
            meta["poses"],
            stats["rows"],
            stats["cols"],
            stats["map_min_x_cm"],
            stats["map_min_y_cm"],
            args.map_resolution_m,
            slope_map,
        )
        roi_mask = GrowRoiFromTrajectory(slope_map, trajectory_seed_mask, args.slope_angle_thresh_deg)
    else:
        trajectory_seed_mask = np.zeros(slope_map.shape, dtype=np.uint8)
        roi_mask = np.ones(slope_map.shape, dtype=np.uint8)

    if 0:
        # PyCharm debug: inspect optional C++-style trajectory-seeded low-slope ROI.
        from gt_generate.ground_truth.debug_view import ShowTerrainOnlyMaps

        ShowTerrainOnlyMaps(
            "TerrainOnly-Roi",
            [
                ("terrain_slope_deg", slope_map, -123456.0),
                ("trajectory_seed_mask", trajectory_seed_mask, None),
                ("terrain_roi_mask", roi_mask, None),
            ],
        )

    final_seed_map = BuildFinalSeedMap(raw_mean_map, raw_std_map, provisional_map, args, roi_mask=roi_mask)
    final_map, final_confidence_map = CompleteTerrain(
        final_seed_map,
        args.map_resolution_m,
        args.final_bgk_radius_m,
        args.final_bgk_min_valid,
        args.final_bgk_bilateral_std,
    )
    final_map[final_seed_map != INVALID_HEIGHT] = final_seed_map[final_seed_map != INVALID_HEIGHT]
    final_inpaint_distance_map = BuildInpaintDistanceMap(final_seed_map, final_map, args.map_resolution_m)
    terrain_ground_type_map = BuildTerrainGroundTypeMap(
        terrain_ground_type_raw_map,
        final_seed_map,
        final_map,
        args.map_resolution_m,
        args.final_bgk_radius_m,
    )

    if 0:
        # PyCharm debug: inspect final reliable seed and final completed terrain.
        from gt_generate.ground_truth.debug_view import ShowTerrainOnlyMaps

        ShowTerrainOnlyMaps(
            "TerrainOnly-Final",
            [
                ("terrain_z_final_seed", final_seed_map, INVALID_HEIGHT),
                ("terrain_z_inpainted", final_map, INVALID_HEIGHT),
                ("terrain_inpaint_distance", final_inpaint_distance_map, INVALID_HEIGHT),
                ("terrain_ground_type_map", terrain_ground_type_map, None),
                ("final_confidence", final_confidence_map, 0.0),
            ],
        )

    output_maps = {
        "terrain_z_raw": raw_mean_map.astype(np.float32),
        "terrain_z_std": raw_std_map.astype(np.float32),
        "terrain_z_raw_seed": raw_seed_map.astype(np.float32),
        "terrain_z_provisional": provisional_map.astype(np.float32),
        "terrain_slope_deg": slope_map.astype(np.float32),
        "trajectory_seed_mask": trajectory_seed_mask.astype(np.uint8),
        "terrain_roi_mask": roi_mask.astype(np.uint8),
        "terrain_z_final_seed": final_seed_map.astype(np.float32),
        "terrain_z_inpainted": final_map.astype(np.float32),
        "terrain_inpaint_distance": final_inpaint_distance_map.astype(np.float32),
        "terrain_ground_type_raw_map": terrain_ground_type_raw_map.astype(np.uint8),
        "terrain_ground_type_map": terrain_ground_type_map.astype(np.uint8),
    }
    SaveTerrainOnlyOutputs(args, stats, output_maps, meta)
    print(f"[TerrainOnly] saved: {args.output_dir}")


if __name__ == "__main__":
    Main()
