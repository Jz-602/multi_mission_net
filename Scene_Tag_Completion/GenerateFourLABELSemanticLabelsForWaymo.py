import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import box_np_ops
from lyf_tools import CarToWorld
from lyf_tools import LoadPosesFromPoseTxt
from lyf_tools import visualize_point_cloud

WAYMO_DATA_ROOT = os.path.expandvars("${WAYMO_DATA_ROOT}")  # Waymo 数据根目录，下面按 sequence/output 组织。
OUTPUT_DIRNAME = "sequence_SemanticPrior"  # 传播语义先验输出目录名，位于每个 sequence output 目录下。

B_RUN_GENERATE = True  # True: 重新生成传播先验 label；False: 不生成，直接使用已有结果。
B_RUN_VISUALIZE = False  # True: 打开整段序列浏览器；False: 不查看。
B_VISUALIZE_GENERATION_DEBUG = False  # True: 生成过程中额外弹窗查看稀疏语义地图和逐帧先验。
B_RUN_MULTIPROCESS = True  # True: 按 sequence 多进程生成；可视化模式下自动退回串行。
MULTIPROCESS_WORKERS = 0  # 多进程 worker 数；0 表示自动使用一半 CPU，正整数表示手动指定进程数。

PROPAGATE_RADIUS_METERS = 0.5  # 最近邻传播半径，超过该距离的点先验标签置 UNKNOWN_LABEL。
UNKNOWN_LABEL = 0  # 未知/无先验标签值，传播失败的点写 0。
KD_TREE_QUERY_WORKERS = 1  # 多进程按 sequence 并行，单个 KDTree query 固定单线程，避免 CPU 过度抢占。
POINT_STRIDE = 1  # 可视化抽点间隔，1 表示显示所有点。
POINT_SIZE = 2.0  # Open3D 可视化点大小。
WINDOW_WIDTH = 1600  # Open3D 可视化窗口宽度。
WINDOW_HEIGHT = 900  # Open3D 可视化窗口高度。
FRAME_INTERVAL_SEC = 0.08  # 自动播放时相邻帧间隔秒数。
B_LOOP_SEQUENCE = False  # True: 播放到最后一帧后循环；False: 到最后一帧后暂停。

WAYMO_DETECTION_TYPE_TO_RAW_SEMANTIC_LABEL = {
    1: 1,  # vehicle -> CAR
    2: 7,  # pedestrian -> PEDESTRIAN
    3: 8,  # sign -> SIGN
    4: 5,  # cyclist/rider -> MOTORCYCLIST
}


def BuildSortKey(file_name: str):
    stem = os.path.splitext(os.path.basename(file_name))[0]
    try:
        return 0, int(stem)
    except ValueError:
        return 1, stem


def FindSequenceOutputDirs(waymo_data_root: str):
    """
    从 WAYMO_DATA_ROOT 搜索全部可生成传播语义先验的 sequence output 目录。

    返回 sequence_dir_list，每个元素都是 `/.../sequence_name/output`。
    """
    sequence_dir_list = []
    for current_dir, dir_name_list, _ in os.walk(waymo_data_root):
        if "output" not in dir_name_list:
            continue

        sequence_dir = os.path.join(current_dir, "output")
        if (
            os.path.isdir(os.path.join(sequence_dir, "pcd"))
            and os.path.isdir(os.path.join(sequence_dir, "SemanticSegmentation"))
            and os.path.isfile(os.path.join(sequence_dir, "pose_slam_euler.txt"))
        ):
            sequence_dir_list.append(sequence_dir)

        ## 找到 sequence/output 后不再进入 output 内部，避免递归扫描 pcd 等大目录。
        dir_name_list.remove("output")

    sequence_dir_list.sort()
    return sequence_dir_list


def LoadLabelMappingAssets():
    """
    从 LYF_segmentation_xyz.yaml 指定的 label_mapping 读取训练标签映射和显示颜色。

    传播保存使用 learning_map 之后的训练标签，不保存原始 Waymo 标签。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "LYF_segmentation_xyz.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    label_mapping_name = config["common_variables"]["label_mapping"]
    label_mapping_path = os.path.join(script_dir, "label_mapping", label_mapping_name)
    with open(label_mapping_path, "r", encoding="utf-8") as file:
        label_mapping_cfg = yaml.safe_load(file)

    class_names = list(label_mapping_cfg["CLASS_NAMES"])
    visualization_colors = label_mapping_cfg["VISUALIZATION_COLORS"]
    color_table = np.asarray(
        [visualization_colors[class_name] for class_name in class_names],
        dtype=np.float64,
    )
    if "UNKNOWN" in class_names:
        color_table[class_names.index("UNKNOWN")] = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

    learning_map = {int(src_label): int(dst_label) for src_label, dst_label in label_mapping_cfg["learning_map"].items()}
    max_src_label = max(learning_map.keys())
    label_remap_lut = np.full((max_src_label + 1,), int(UNKNOWN_LABEL), dtype=np.int32)
    for src_label, dst_label in learning_map.items():
        label_remap_lut[src_label] = dst_label

    return class_names, color_table, label_remap_lut


def RemapRawLabelsToTraining(point_label: np.ndarray, label_remap_lut: np.ndarray) -> np.ndarray:
    """
    point_label: shape [N]，原始 Waymo 标签。
    返回 mapped_label: shape [N]，训练类别标签。
    """
    point_label = np.asarray(point_label, dtype=np.int64).reshape(-1)
    mapped_label = np.full(point_label.shape, int(UNKNOWN_LABEL), dtype=np.int64)
    b_valid = (point_label >= 0) & (point_label < label_remap_lut.shape[0])
    mapped_label[b_valid] = label_remap_lut[point_label[b_valid]]
    return mapped_label


def ReadPCDXYZ(pcd_path: str) -> np.ndarray:
    """
    用 Open3D 读取单帧点云的 xyz。

    返回:
        points_xyz: (N, 3), float32
    """
    if not os.path.isfile(pcd_path):
        raise FileNotFoundError(f"未找到 pcd 文件: {pcd_path}")

    point_cloud = o3d.t.io.read_point_cloud(pcd_path)
    if "positions" not in point_cloud.point:
        raise ValueError(f"点云里没有 positions 字段: {pcd_path}")

    points_xyz = point_cloud.point["positions"].numpy().astype(np.float32, copy=False)
    return points_xyz


def LoadSemanticLabelsForFrame(semantic_npy_path: str, point_count: int) -> np.ndarray:
    """
    读取单帧语义标签。

    当前数据中 `.npy` 约定为 shape (N, 2)，
    每个点一行，第二列为语义标签。

    返回:
        semantic_labels: (N,), int32
    """
    if not os.path.isfile(semantic_npy_path):
        raise FileNotFoundError(f"未找到语义标签文件: {semantic_npy_path}")

    semantic_data = np.load(semantic_npy_path)
    semantic_data = np.asarray(semantic_data, dtype=np.int32)

    if semantic_data.ndim != 2 or semantic_data.shape[1] < 2:
        raise ValueError(f"不支持的语义标签格式: {semantic_npy_path}, shape={semantic_data.shape}")
    if semantic_data.shape[0] != point_count:
        raise ValueError(
            f"语义标签行数与点数不一致: {semantic_npy_path}, "
            f"labels={semantic_data.shape[0]}, points={point_count}"
        )

    return semantic_data[:, 1].astype(np.int32, copy=False)


def LoadDetectionInfoDict(detection_info_path: str):
    """
    读取当前 sequence 的 detection_info.pkl。

    返回 detection_info_dict:
        key 为 frame_id 字符串，value 为当前帧检测真值；若文件不存在则返回空 dict。
    """
    if not os.path.isfile(detection_info_path):
        return {}

    with open(detection_info_path, "rb") as file:
        detection_info = pickle.load(file)

    if not isinstance(detection_info, list):
        raise TypeError(f"detection_info.pkl 不是 list，而是 {type(detection_info)}")

    detection_info_dict = {}
    for frame_info in detection_info:
        frame_id = str(frame_info["frame_timestamp_micros"])[:-3]
        detection_info_dict[frame_id] = frame_info

    return detection_info_dict


def FillUnknownPriorLabelsByDetectionBoxes(
    car_points_xyz: np.ndarray,
    frame_prior_labels: np.ndarray,
    frame_detection_info,
    detection_type_to_training_label: dict,
    unknown_label: int,
) -> int:
    """
    用当前帧检测框补充传播失败的语义先验。

    car_points_xyz: shape [N, 3]，当前帧 LiDAR/车体坐标系 xyz。
    frame_prior_labels: shape [N]，训练标签空间语义先验；本函数只修改值为 unknown_label 的点。
    frame_detection_info: detection_info.pkl 中当前帧 dict，objects 里的 box 为当前帧 LiDAR/车体坐标系。
    返回 box_filled_count: 被检测框补充成非 0 标签的点数。
    """
    if frame_detection_info is None:
        return 0

    objects = frame_detection_info.get("objects", [])
    if len(objects) == 0:
        return 0

    gt_box_list = []
    gt_label_list = []
    for object_info in objects:
        detection_type = int(object_info["type"])
        if detection_type not in detection_type_to_training_label:
            continue
        training_label = int(detection_type_to_training_label[detection_type])
        if training_label == int(unknown_label):
            continue
        gt_box_list.append(np.asarray(object_info["box"], dtype=np.float32).reshape(7))
        gt_label_list.append(training_label)

    if len(gt_box_list) == 0:
        return 0

    gt_boxes = np.stack(gt_box_list, axis=0).astype(np.float32, copy=False)
    gt_labels = np.asarray(gt_label_list, dtype=np.uint16)
    point_in_box_mask = box_np_ops.points_in_rbbox(car_points_xyz, gt_boxes)

    b_unknown_before = frame_prior_labels == int(unknown_label)
    filled_count = 0
    for box_idx, training_label in enumerate(gt_labels):
        b_fill = b_unknown_before & point_in_box_mask[:, box_idx]
        fill_count = int(np.count_nonzero(b_fill))
        if fill_count == 0:
            continue
        frame_prior_labels[b_fill] = training_label
        b_unknown_before[b_fill] = False
        filled_count += fill_count

    return filled_count


def BuildSparseSemanticFrameBlocks(
    sequence_dir: str,
    pcd_file_name_list,
    pose_list,
    unknown_label: int,
    label_remap_lut: np.ndarray,
):
    """
    构建有语义真值帧的稀疏语义地图块。

    每个 block 对应一个有 SemanticSegmentation 真值的帧：
    - points: shape [M, 3]，当前帧车体坐标系点经过 pose 对齐后的世界坐标系点。
    - labels: shape [M]，与 points 一一对应，已经映射到训练标签空间；这里只保留非 0 标签点。
    """
    semantic_dir = os.path.join(sequence_dir, "SemanticSegmentation")
    pcd_dir = os.path.join(sequence_dir, "pcd")

    semantic_frame_id_set = {
        os.path.splitext(file_name)[0]
        for file_name in os.listdir(semantic_dir)
        if file_name.endswith(".npy")
    }  # 有单帧语义结果的帧 id 集合

    semantic_frame_block_dict = {}

    ## 逐帧读取有真值的点云，并变换到统一世界坐标系。
    for frame_idx, (pcd_file_name, pose6) in enumerate(zip(pcd_file_name_list, pose_list)):
        frame_id = os.path.splitext(pcd_file_name)[0]
        if frame_id not in semantic_frame_id_set:
            continue

        pcd_path = os.path.join(pcd_dir, pcd_file_name)
        semantic_npy_path = os.path.join(semantic_dir, f"{frame_id}.npy")
        car_points_xyz = ReadPCDXYZ(pcd_path)  # shape [N, 3]，当前帧车体坐标系 xyz。
        world_points_xyz = CarToWorld(car_points_xyz, pose6)  # shape [N, 3]，世界坐标系 xyz。
        raw_point_labels = LoadSemanticLabelsForFrame(
            semantic_npy_path=semantic_npy_path,
            point_count=car_points_xyz.shape[0],
        )
        point_labels = RemapRawLabelsToTraining(raw_point_labels, label_remap_lut)
        b_labeled = point_labels != unknown_label
        if np.any(b_labeled):
            semantic_frame_block_dict[frame_id] = {
                "points": world_points_xyz[b_labeled].astype(np.float32, copy=False),
                "labels": point_labels[b_labeled].astype(np.int32, copy=False),
            }

    if len(semantic_frame_block_dict) == 0:
        raise RuntimeError(f"没有可用于传播的语义帧: {semantic_dir}")

    return semantic_frame_block_dict


def MergeSemanticFrameBlocks(semantic_frame_block_dict, exclude_frame_id=None):
    """
    合并语义地图块。

    exclude_frame_id:
        若不为 None，则排除当前有真值帧自身，避免训练输入先验泄漏本帧真值。
    """
    semantic_point_block_list = []
    semantic_label_block_list = []

    for frame_id, semantic_block in semantic_frame_block_dict.items():
        if frame_id == exclude_frame_id:
            continue
        semantic_point_block_list.append(semantic_block["points"])
        semantic_label_block_list.append(semantic_block["labels"])

    if len(semantic_point_block_list) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )

    semantic_map_points_xyz = np.concatenate(semantic_point_block_list, axis=0).astype(np.float32, copy=False)
    semantic_map_labels = np.concatenate(semantic_label_block_list, axis=0).astype(np.int32, copy=False)
    return semantic_map_points_xyz, semantic_map_labels


def PropagateSemanticPrior(
    query_points_world_xyz: np.ndarray,
    semantic_tree,
    semantic_map_labels: np.ndarray,
    propagate_radius_meters: float,
    unknown_label: int,
) -> np.ndarray:
    """
    给当前帧点生成传播语义先验。

    query_points_world_xyz: shape [N, 3]，当前帧点云对齐后的世界坐标系点。
    semantic_tree: 由稀疏语义地图点构建的 cKDTree；None 表示没有可传播语义点。
    semantic_map_labels: shape [M]，与 semantic_map_points_xyz 一一对应。
    返回 prior_labels: shape [N]，距离超过阈值的点为 unknown_label。
    """
    prior_labels = np.full(query_points_world_xyz.shape[0], unknown_label, dtype=np.uint16)
    if semantic_tree is None or semantic_map_labels.shape[0] == 0:
        return prior_labels

    distances, indices = semantic_tree.query(query_points_world_xyz, k=1, workers=KD_TREE_QUERY_WORKERS)
    b_valid = distances <= propagate_radius_meters
    prior_labels[b_valid] = semantic_map_labels[indices[b_valid]].astype(np.uint16, copy=False)
    return prior_labels


def GenerateSequenceSemanticPriorLabels(
    sequence_dir: str,
    propagate_radius_meters: float = 0.5,
    unknown_label: int = 0,
    output_dirname: str = "sequence_SemanticPrior",
):
    """
    输入一个序列 output 目录，生成每帧点云的传播语义先验 `sequence_SemanticPrior/*.label`。

    - 无语义真值帧：用所有语义真值帧构建稀疏语义地图并传播。
    - 有语义真值帧：排除当前帧自身后再传播，避免训练输入泄漏本帧监督标签。

    输入:
        sequence_dir:
            例如 `/.../FH_waymo/.../output`
    """
    pcd_dir = os.path.join(sequence_dir, "pcd")
    semantic_dir = os.path.join(sequence_dir, "SemanticSegmentation")
    pose_txt_path = os.path.join(sequence_dir, "pose_slam_euler.txt")
    detection_info_path = os.path.join(sequence_dir, "detection_info.pkl")
    output_dir = os.path.join(sequence_dir, output_dirname)

    if not os.path.isdir(pcd_dir):
        raise FileNotFoundError(f"未找到 pcd 目录: {pcd_dir}")
    if not os.path.isdir(semantic_dir):
        raise FileNotFoundError(f"未找到 SemanticSegmentation 目录: {semantic_dir}")
    if not os.path.isfile(pose_txt_path):
        raise FileNotFoundError(f"未找到 pose 文件: {pose_txt_path}")
    os.makedirs(output_dir, exist_ok=True)

    ## 读取序列文件列表和位姿。
    pcd_file_name_list = sorted(
        file_name for file_name in os.listdir(pcd_dir)
        if file_name.endswith(".pcd")
    )
    frame_id_list = [os.path.splitext(file_name)[0] for file_name in pcd_file_name_list]
    pose_list = LoadPosesFromPoseTxt(pose_txt_path, frame_id_list=frame_id_list)

    _, _, label_remap_lut = LoadLabelMappingAssets()
    detection_info_dict = LoadDetectionInfoDict(detection_info_path)
    detection_type_to_training_label = {
        detection_type: int(label_remap_lut[raw_semantic_label])
        for detection_type, raw_semantic_label in WAYMO_DETECTION_TYPE_TO_RAW_SEMANTIC_LABEL.items()
        if raw_semantic_label < label_remap_lut.shape[0]
    }

    ## 构建所有有真值帧的稀疏语义地图块；后续有真值帧会 leave-one-frame-out 传播。
    semantic_frame_block_dict = BuildSparseSemanticFrameBlocks(
        sequence_dir=sequence_dir,
        pcd_file_name_list=pcd_file_name_list,
        pose_list=pose_list,
        unknown_label=unknown_label,
        label_remap_lut=label_remap_lut,
    )

    semantic_map_points_xyz, semantic_map_labels = MergeSemanticFrameBlocks(semantic_frame_block_dict)
    semantic_tree = cKDTree(semantic_map_points_xyz)
    if B_VISUALIZE_GENERATION_DEBUG:
        visualize_point_cloud(semantic_map_points_xyz, semantic_map_labels)

    ## 对每一帧生成传播语义先验；有真值帧排除自身，普通帧使用全量语义地图。
    for frame_idx, (pcd_file_name, pose6) in enumerate(zip(pcd_file_name_list, pose_list)):
        frame_id = os.path.splitext(pcd_file_name)[0]
        pcd_path = os.path.join(pcd_dir, pcd_file_name)
        car_points_xyz = ReadPCDXYZ(pcd_path)  # shape [N, 3]，当前帧车体坐标系 xyz。
        world_points_xyz = CarToWorld(car_points_xyz, pose6)  # shape [N, 3]，世界坐标系 xyz。

        if frame_id in semantic_frame_block_dict:
            current_map_points_xyz, current_map_labels = MergeSemanticFrameBlocks(
                semantic_frame_block_dict,
                exclude_frame_id=frame_id,
            )
            current_semantic_tree = None
            if current_map_points_xyz.shape[0] > 0:
                current_semantic_tree = cKDTree(current_map_points_xyz)
        else:
            current_map_labels = semantic_map_labels
            current_semantic_tree = semantic_tree

        frame_prior_labels = PropagateSemanticPrior(
            query_points_world_xyz=world_points_xyz,
            semantic_tree=current_semantic_tree,
            semantic_map_labels=current_map_labels,
            propagate_radius_meters=propagate_radius_meters,
            unknown_label=unknown_label,
        )
        FillUnknownPriorLabelsByDetectionBoxes(
            car_points_xyz=car_points_xyz,
            frame_prior_labels=frame_prior_labels,
            frame_detection_info=detection_info_dict.get(frame_id),
            detection_type_to_training_label=detection_type_to_training_label,
            unknown_label=unknown_label,
        )

        output_path = os.path.join(output_dir, f"{frame_id}.label")
        frame_prior_labels.tofile(output_path)

        if B_VISUALIZE_GENERATION_DEBUG:
            visualize_point_cloud(car_points_xyz, frame_prior_labels)


def LoadPriorLabel(label_path: str, point_count: int) -> np.ndarray:
    """
    读取传播语义先验 label。

    label_path: `.label` 文件，uint16，每个点一个训练标签空间的先验语义 id。
    返回 point_label: shape [N]，与当前帧 pcd 点顺序一致。
    """
    point_label = np.fromfile(label_path, dtype=np.uint16).astype(np.int64)
    if point_label.shape[0] != point_count:
        raise ValueError(
            f"label count mismatch: label={point_label.shape[0]}, "
            f"points={point_count}, path={label_path}"
        )
    return point_label


def BuildPriorPointCloud(
    points_xyz_lidar: np.ndarray,
    point_label: np.ndarray,
    color_table: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    points_xyz_lidar: shape [N, 3]，当前帧 LiDAR/车体坐标系 xyz，单位 m。
    point_label: shape [N]，训练标签空间的传播语义先验标签，0 表示没有先验。
    """
    k_stride = max(int(POINT_STRIDE), 1)
    points_show = points_xyz_lidar[::k_stride].astype(np.float64, copy=False)
    label_show = np.clip(point_label[::k_stride], 0, color_table.shape[0] - 1)
    colors_show = color_table[label_show]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_show)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_show)
    return point_cloud


def BuildGroundTruthFrameMarker(points_xyz_lidar: np.ndarray) -> o3d.geometry.TriangleMesh:
    """
    构建真值帧标记。

    points_xyz_lidar: shape [N, 3]，当前帧 LiDAR/车体坐标系 xyz。
    返回红色立方体，放在点云中心上方，用于表示当前帧本身有 SemanticSegmentation 真值。
    """
    if points_xyz_lidar.shape[0] == 0:
        marker_center = np.array([0.0, 0.0, 2.5], dtype=np.float64)
    else:
        point_min = np.nanmin(points_xyz_lidar, axis=0).astype(np.float64)
        point_max = np.nanmax(points_xyz_lidar, axis=0).astype(np.float64)
        marker_center = np.array([
            0.5 * (point_min[0] + point_max[0]),
            0.5 * (point_min[1] + point_max[1]),
            point_max[2] + 2.0,
        ], dtype=np.float64)

    marker_size = 1.5
    marker = o3d.geometry.TriangleMesh.create_box(
        width=marker_size,
        height=marker_size,
        depth=marker_size,
    )
    marker.paint_uniform_color([1.0, 0.0, 0.0])
    marker.translate(marker_center - marker_size * 0.5)
    return marker


def PrintPriorHistogram(point_label: np.ndarray) -> None:
    labeled_count = int(np.count_nonzero(point_label))
    total_count = int(point_label.shape[0])
    unique_labels, counts = np.unique(point_label, return_counts=True)
    hist_text = ", ".join(
        f"{int(label)}:{int(count)}"
        for label, count in zip(unique_labels, counts)
    )
    print(f"[PriorHist] labeled={labeled_count}/{total_count} | {hist_text}", flush=True)


def PrintTrainingHistogram(point_label: np.ndarray, class_names) -> None:
    class_hist = np.bincount(
        np.clip(point_label, 0, len(class_names) - 1),
        minlength=len(class_names),
    )
    hist_text = ", ".join(
        f"{class_name}:{int(class_hist[class_idx])}"
        for class_idx, class_name in enumerate(class_names)
    )
    print(f"[TrainingHist] {hist_text}", flush=True)


class FrameController:
    def __init__(self, num_frames: int) -> None:
        self.num_frames = int(num_frames)
        self.frame_idx = 0
        self.b_paused = True
        self.b_dirty = True
        self.last_frame_time = 0.0

    def PrevFrame(self, vis) -> bool:
        if self.frame_idx > 0:
            self.frame_idx -= 1
        elif B_LOOP_SEQUENCE:
            self.frame_idx = self.num_frames - 1
        self.b_paused = True
        self.b_dirty = True
        print(f"[Control] previous frame: {self.frame_idx + 1}/{self.num_frames}", flush=True)
        return False

    def NextFrame(self, vis) -> bool:
        if self.frame_idx + 1 < self.num_frames:
            self.frame_idx += 1
        elif B_LOOP_SEQUENCE:
            self.frame_idx = 0
        self.b_paused = True
        self.b_dirty = True
        print(f"[Control] next frame: {self.frame_idx + 1}/{self.num_frames}", flush=True)
        return False

    def TogglePause(self, vis) -> bool:
        self.b_paused = not self.b_paused
        self.last_frame_time = time.perf_counter()
        print(f"[Control] {'pause' if self.b_paused else 'play'}", flush=True)
        return False

    def AutoStep(self) -> None:
        if self.b_paused or self.b_dirty:
            return
        elapsed_time = time.perf_counter() - self.last_frame_time
        if elapsed_time < FRAME_INTERVAL_SEC:
            return
        if self.frame_idx + 1 < self.num_frames:
            self.frame_idx += 1
            self.b_dirty = True
        elif B_LOOP_SEQUENCE:
            self.frame_idx = 0
            self.b_dirty = True
        else:
            self.b_paused = True


def RegisterKeys(vis: o3d.visualization.VisualizerWithKeyCallback, controller: FrameController) -> None:
    ## 空格：暂停/继续；q/Q：上一帧；e/E：下一帧。
    vis.register_key_callback(ord("q"), controller.PrevFrame)
    vis.register_key_callback(ord("Q"), controller.PrevFrame)
    vis.register_key_callback(ord("e"), controller.NextFrame)
    vis.register_key_callback(ord("E"), controller.NextFrame)
    vis.register_key_callback(32, controller.TogglePause)


def BuildFrameListForVisualization(sequence_dir: str, output_dirname: str):
    pcd_dir = os.path.join(sequence_dir, "pcd")
    prior_dir = os.path.join(sequence_dir, output_dirname)
    semantic_dir = os.path.join(sequence_dir, "SemanticSegmentation")

    if not os.path.isdir(pcd_dir):
        raise FileNotFoundError(f"未找到 pcd 目录: {pcd_dir}")
    if not os.path.isdir(prior_dir):
        raise FileNotFoundError(f"未找到传播先验目录: {prior_dir}")

    semantic_frame_id_set = set()
    if os.path.isdir(semantic_dir):
        semantic_frame_id_set = {
            os.path.splitext(file_name)[0]
            for file_name in os.listdir(semantic_dir)
            if file_name.endswith(".npy")
        }

    frame_list = []
    for pcd_file_name in sorted(
        [file_name for file_name in os.listdir(pcd_dir) if file_name.endswith(".pcd")],
        key=BuildSortKey,
    ):
        frame_id = os.path.splitext(pcd_file_name)[0]
        pcd_path = os.path.join(pcd_dir, pcd_file_name)
        label_path = os.path.join(prior_dir, f"{frame_id}.label")
        if os.path.isfile(label_path):
            frame_list.append({
                "frame_id": frame_id,
                "pcd_path": pcd_path,
                "label_path": label_path,
                "b_has_gt": frame_id in semantic_frame_id_set,
            })

    if len(frame_list) == 0:
        raise RuntimeError(f"没有找到可视化帧: {pcd_dir} {prior_dir}")
    return frame_list


def VisualizeSequenceSemanticPrior(
    sequence_dir: str,
    output_dirname: str = OUTPUT_DIRNAME,
) -> None:
    """
    浏览整段序列的传播语义先验。

    q/e 切换上一帧/下一帧；空格播放/暂停。有真值帧会额外显示红色立方体。
    """
    frame_list = BuildFrameListForVisualization(sequence_dir, output_dirname)
    class_names, color_table, _ = LoadLabelMappingAssets()
    controller = FrameController(num_frames=len(frame_list))

    print(f"[Dataset] frames={len(frame_list)}", flush=True)
    print(f"[Dataset] sequence_dir={sequence_dir}", flush=True)
    print(f"[Dataset] prior_dir={os.path.join(sequence_dir, output_dirname)}", flush=True)
    print("[Keys] Space pause/play, q previous frame, e next frame", flush=True)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{os.path.basename(os.path.dirname(sequence_dir))} {output_dirname}",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        render_option.point_size = float(POINT_SIZE)
    RegisterKeys(vis, controller)

    b_window_alive = True
    b_first_frame = True
    try:
        while b_window_alive:
            if controller.b_dirty:
                frame_item = frame_list[controller.frame_idx]
                points_xyz_lidar = ReadPCDXYZ(frame_item["pcd_path"])
                point_label = LoadPriorLabel(frame_item["label_path"], points_xyz_lidar.shape[0])
                point_cloud = BuildPriorPointCloud(points_xyz_lidar, point_label, color_table)

                vis.clear_geometries()
                vis.add_geometry(point_cloud, reset_bounding_box=b_first_frame)
                if frame_item["b_has_gt"]:
                    marker = BuildGroundTruthFrameMarker(points_xyz_lidar)
                    vis.add_geometry(marker, reset_bounding_box=False)
                b_first_frame = False

                frame_type = "GT" if frame_item["b_has_gt"] else "NO_GT"
                print(
                    f"[Frame] {controller.frame_idx + 1}/{len(frame_list)} "
                    f"type={frame_type} timestamp={frame_item['frame_id']} "
                    f"points={points_xyz_lidar.shape[0]}",
                    flush=True,
                )
                PrintPriorHistogram(point_label)
                PrintTrainingHistogram(point_label, class_names)

                b_window_alive = vis.poll_events()
                vis.update_renderer()
                controller.b_dirty = False
                controller.last_frame_time = time.perf_counter()

            b_window_alive = vis.poll_events()
            vis.update_renderer()
            controller.AutoStep()
            time.sleep(0.01)
    finally:
        vis.destroy_window()


def GenerateOneSequenceTask(task):
    sequence_idx, sequence_count, sequence_dir = task

    try:
        GenerateSequenceSemanticPriorLabels(
            sequence_dir,
            propagate_radius_meters=PROPAGATE_RADIUS_METERS,
            unknown_label=UNKNOWN_LABEL,
            output_dirname=OUTPUT_DIRNAME,
        )
    except Exception as err:
        return sequence_idx, sequence_count, sequence_dir, repr(err)

    return sequence_idx, sequence_count, sequence_dir, None


def Main():
    waymo_data_root = os.path.expandvars(WAYMO_DATA_ROOT)
    if waymo_data_root == "${WAYMO_DATA_ROOT}" or waymo_data_root == "":
        raise EnvironmentError("环境变量 WAYMO_DATA_ROOT 未设置，无法搜索 Waymo 序列根目录。")

    sequence_dir_list = FindSequenceOutputDirs(waymo_data_root)
    if len(sequence_dir_list) == 0:
        raise RuntimeError(f"没有在 WAYMO_DATA_ROOT 下找到可处理的 sequence/output 目录: {waymo_data_root}")

    failed_sequence_list = []
    b_run_parallel = (
        B_RUN_GENERATE
        and B_RUN_MULTIPROCESS
        and not B_RUN_VISUALIZE
        and len(sequence_dir_list) > 1
    )

    if b_run_parallel:
        if MULTIPROCESS_WORKERS > 0:
            worker_count = min(int(MULTIPROCESS_WORKERS), len(sequence_dir_list))
        else:
            worker_count = min(max(1, (os.cpu_count() or 1) // 2), len(sequence_dir_list))
        task_list = [
            (sequence_idx, len(sequence_dir_list), sequence_dir)
            for sequence_idx, sequence_dir in enumerate(sequence_dir_list)
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_task = {
                executor.submit(GenerateOneSequenceTask, task): task
                for task in task_list
            }
            for future in as_completed(future_to_task):
                sequence_idx, sequence_count, sequence_dir = future_to_task[future]
                sequence_name = os.path.basename(os.path.dirname(sequence_dir))
                sequence_idx, sequence_count, sequence_dir, error_text = future.result()
                if error_text is not None:
                    failed_sequence_list.append((sequence_dir, error_text))
                    print(
                        f"[SequenceError] {sequence_idx + 1}/{sequence_count} "
                        f"name={sequence_name} error={error_text}",
                        flush=True,
                    )
                else:
                    print(
                        f"[SequenceOK] {sequence_idx + 1}/{sequence_count} name={sequence_name}",
                        flush=True,
                    )
    else:
        for sequence_idx, sequence_dir in enumerate(sequence_dir_list):
            sequence_name = os.path.basename(os.path.dirname(sequence_dir))
            try:
                if B_RUN_GENERATE:
                    GenerateSequenceSemanticPriorLabels(
                        sequence_dir,
                        propagate_radius_meters=PROPAGATE_RADIUS_METERS,
                        unknown_label=UNKNOWN_LABEL,
                        output_dirname=OUTPUT_DIRNAME,
                    )
                if B_RUN_VISUALIZE:
                    VisualizeSequenceSemanticPrior(
                        sequence_dir,
                        output_dirname=OUTPUT_DIRNAME,
                    )
            except Exception as err:
                failed_sequence_list.append((sequence_dir, repr(err)))
                print(
                    f"[SequenceError] {sequence_idx + 1}/{len(sequence_dir_list)} "
                    f"name={sequence_name} error={repr(err)}",
                    flush=True,
                )
                continue

            print(
                f"[SequenceOK] {sequence_idx + 1}/{len(sequence_dir_list)} name={sequence_name}",
                flush=True,
            )

    if len(failed_sequence_list) > 0:
        raise RuntimeError(
            f"{len(failed_sequence_list)}/{len(sequence_dir_list)} sequences failed."
        )


if __name__ == "__main__":
    Main()
