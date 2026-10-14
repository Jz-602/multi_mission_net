import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
for path in [str(PROJECT_DIR), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from lyf_tools import voxel_downsample
from pcd_io import ReadCleanPcdXYZ


PCD_DIR = Path(os.environ["WAYMO_DATA_ROOT"]) / "16102220208346880_1420_000_1440_000/output/pcd"
POSE_PATH = PCD_DIR.parent / "pose_slam_euler.txt"
MODEL_NAME = "xyzl_epoch60"
LABEL_DIR = PCD_DIR.parent / "label" / MODEL_NAME
LABEL_MAPPING_PATH = PROJECT_DIR / "label_mapping" / "waymo_merge_example.yaml"

VOXEL_SIZE = 0.05
MAX_FRAMES = -1                 # 最多读取多少帧；-1 表示读取 pcd 目录下全部帧。
POSE_MAX_TIME_DIFF = 200        # pcd 时间戳找不到完全匹配 pose 时，允许匹配最近 pose 的最大时间差；<0 表示不限制。
CHUNK_FRAME_COUNT = 300         # 每累计多少帧做一次中间下采样；数值越大越快但峰值内存越高。
POINT_STRIDE = 1                # 可视化抽点间隔；1 表示显示所有下采样后的点，2 表示隔一个点显示一个。
POINT_SIZE = 1.0                # Open3D 显示点大小。
WINDOW_WIDTH = 1600             # Open3D 窗口宽度。
WINDOW_HEIGHT = 900             # Open3D 窗口高度。
B_SHOW_AXIS = True              # True 显示 world 坐标系坐标轴；False 不显示。


def BuildRotationXYZ(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def BuildPoseMatrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    pose 行含义: timestamp x y z roll pitch yaw。
    返回 pose_lidar_to_world: [4, 4]，当前帧 LiDAR/车体坐标系 -> world 坐标系。
    """
    pose_lidar_to_world = np.eye(4, dtype=np.float32)
    pose_lidar_to_world[:3, :3] = BuildRotationXYZ(roll, pitch, yaw).astype(np.float32)
    pose_lidar_to_world[:3, 3] = np.asarray([x, y, z], dtype=np.float32)
    return pose_lidar_to_world


def LoadPoseDict(pose_path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    if not pose_path.is_file():
        raise FileNotFoundError(f"POSE_PATH not found: {pose_path}")

    pose_dict: Dict[str, np.ndarray] = {}
    timestamp_list: List[str] = []
    with pose_path.open("r", encoding="utf-8") as file_object:
        for line in file_object:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            timestamp = str(parts[-7])
            try:
                x, y, z = float(parts[-6]), float(parts[-5]), float(parts[-4])
                roll, pitch, yaw = float(parts[-3]), float(parts[-2]), float(parts[-1])
            except ValueError:
                continue
            pose_dict[timestamp] = BuildPoseMatrix(x, y, z, roll, pitch, yaw)
            timestamp_list.append(timestamp)

    if len(pose_dict) == 0:
        raise ValueError(f"No valid pose rows found in {pose_path}")

    pose_time_array = np.asarray([int(timestamp) for timestamp in timestamp_list], dtype=np.int64)
    return pose_dict, pose_time_array, timestamp_list


def FindPoseForTimestamp(
    timestamp: str,
    pose_dict: Dict[str, np.ndarray],
    pose_time_array: np.ndarray,
    timestamp_list: List[str],
) -> np.ndarray:
    if timestamp in pose_dict:
        return pose_dict[timestamp]

    timestamp_value = int(timestamp)
    nearest_idx = int(np.argmin(np.abs(pose_time_array - timestamp_value)))
    matched_timestamp = timestamp_list[nearest_idx]
    time_diff = int(abs(int(matched_timestamp) - timestamp_value))
    if POSE_MAX_TIME_DIFF >= 0 and time_diff > POSE_MAX_TIME_DIFF:
        raise KeyError(
            f"pose for timestamp={timestamp} not found; nearest={matched_timestamp}, "
            f"diff={time_diff} > {POSE_MAX_TIME_DIFF}"
        )
    return pose_dict[matched_timestamp]


def BuildSortKey(path: Path):
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem


def TransformLidarToWorld(points_xyz_lidar: np.ndarray, pose_lidar_to_world: np.ndarray) -> np.ndarray:
    """
    points_xyz_lidar: [N, 3]，当前帧 LiDAR/车体坐标系 xyz，单位 m。
    pose_lidar_to_world: [4, 4]，当前帧 LiDAR/车体坐标系 -> world 坐标系。
    返回 points_xyz_world: [N, 3]，world 坐标系 xyz，单位 m。
    """
    rotation_lidar_to_world = pose_lidar_to_world[:3, :3]
    translation_lidar_to_world = pose_lidar_to_world[:3, 3]
    points_xyz_world = points_xyz_lidar @ rotation_lidar_to_world.T + translation_lidar_to_world
    return points_xyz_world.astype(np.float32, copy=False)


def DownsamplePoints(points_xyz_world: np.ndarray) -> np.ndarray:
    if points_xyz_world.shape[0] == 0:
        return points_xyz_world
    return voxel_downsample(points_xyz_world, VOXEL_SIZE)


def DownsamplePointsAndLabels(
    points_xyz_world: np.ndarray,
    point_labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if points_xyz_world.shape[0] == 0:
        return points_xyz_world, point_labels
    points_downsampled, selected_indices, _ = voxel_downsample(
        points_xyz_world,
        VOXEL_SIZE,
        return_mapping=True,
    )
    return points_downsampled, point_labels[selected_indices]


def LoadLabel(label_path: Path, point_count: int) -> np.ndarray:
    """
    label_path: uint32 `.label`，模型预测类别 id，与当前帧清洗后 PCD 点顺序一致。
    返回 point_label: [N]。
    """
    if not label_path.is_file():
        raise FileNotFoundError(f"label not found: {label_path}")
    point_label = np.fromfile(label_path, dtype=np.uint32).astype(np.int64)
    if point_label.shape[0] != point_count:
        raise ValueError(
            f"label count mismatch: label={point_label.shape[0]}, "
            f"points={point_count}, path={label_path}"
        )
    return point_label


def LoadAccumulatedMap() -> Tuple[np.ndarray, np.ndarray]:
    if not PCD_DIR.is_dir():
        raise FileNotFoundError(f"PCD_DIR not found: {PCD_DIR}")
    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"LABEL_DIR not found: {LABEL_DIR}")

    pcd_path_list = sorted(PCD_DIR.glob("*.pcd"), key=BuildSortKey)
    if MAX_FRAMES >= 0:
        pcd_path_list = pcd_path_list[:MAX_FRAMES]
    if len(pcd_path_list) == 0:
        raise RuntimeError(f"No pcd files found: {PCD_DIR}")

    pose_dict, pose_time_array, timestamp_list = LoadPoseDict(POSE_PATH)

    map_points_xyz_world = np.empty((0, 3), dtype=np.float32)
    map_labels = np.empty((0,), dtype=np.int64)
    chunk_points_list = []
    chunk_label_list = []
    for frame_idx, pcd_path in enumerate(pcd_path_list):
        points_xyz_lidar = ReadCleanPcdXYZ(pcd_path)
        point_label = LoadLabel(LABEL_DIR / f"{pcd_path.stem}.label", points_xyz_lidar.shape[0])
        pose_lidar_to_world = FindPoseForTimestamp(
            timestamp=pcd_path.stem,
            pose_dict=pose_dict,
            pose_time_array=pose_time_array,
            timestamp_list=timestamp_list,
        )
        points_xyz_world = TransformLidarToWorld(points_xyz_lidar, pose_lidar_to_world)
        chunk_points_list.append(points_xyz_world)
        chunk_label_list.append(point_label)

        b_flush_chunk = (
            len(chunk_points_list) >= CHUNK_FRAME_COUNT
            or frame_idx + 1 == len(pcd_path_list)
        )
        if b_flush_chunk:
            chunk_points_xyz_world = np.concatenate(chunk_points_list, axis=0)
            chunk_labels = np.concatenate(chunk_label_list, axis=0)
            if map_points_xyz_world.shape[0] > 0:
                chunk_points_xyz_world = np.concatenate(
                    [map_points_xyz_world, chunk_points_xyz_world],
                    axis=0,
                )
                chunk_labels = np.concatenate([map_labels, chunk_labels], axis=0)
            map_points_xyz_world, map_labels = DownsamplePointsAndLabels(
                chunk_points_xyz_world,
                chunk_labels,
            )
            chunk_points_list = []
            chunk_label_list = []

        print(
            f"[Map] frame={frame_idx + 1}/{len(pcd_path_list)} "
            f"pcd={pcd_path.name} raw_points={points_xyz_lidar.shape[0]} "
            f"map_points={map_points_xyz_world.shape[0]}",
            flush=True,
        )

    return map_points_xyz_world, map_labels


def LoadLabelColors() -> np.ndarray:
    with LABEL_MAPPING_PATH.open("r", encoding="utf-8") as file_object:
        label_mapping_cfg = yaml.safe_load(file_object)
    class_names = list(label_mapping_cfg["CLASS_NAMES"])
    visualization_colors = label_mapping_cfg["VISUALIZATION_COLORS"]
    return np.asarray(
        [visualization_colors[class_name] for class_name in class_names],
        dtype=np.float64,
    )


def VisualizeMap(points_xyz_world: np.ndarray, point_labels: np.ndarray) -> None:
    """
    points_xyz_world: [N, 3]，整段序列拼接后的 world 坐标系地图点，单位 m。
    point_labels: [N]，与 points_xyz_world 一一对应的模型预测类别 id。
    """
    k_stride = max(int(POINT_STRIDE), 1)
    points_show = points_xyz_world[::k_stride].astype(np.float64, copy=False)
    labels_show = point_labels[::k_stride]
    color_table = LoadLabelColors()
    colors_show = color_table[np.clip(labels_show, 0, color_table.shape[0] - 1)]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_show)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_show)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"{PCD_DIR.parent.parent.name} {MODEL_NAME} map voxel={VOXEL_SIZE}",
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
    )
    if B_SHOW_AXIS:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=5.0,
            origin=points_show.mean(axis=0),
        )
        vis.add_geometry(axis)
    vis.add_geometry(point_cloud)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        render_option.point_size = float(POINT_SIZE)
    vis.run()
    vis.destroy_window()


def Main() -> None:
    print(f"[Config] pcd_dir={PCD_DIR}", flush=True)
    print(f"[Config] pose_path={POSE_PATH}", flush=True)
    print(f"[Config] label_dir={LABEL_DIR}", flush=True)
    print(f"[Config] voxel_size={VOXEL_SIZE}", flush=True)
    map_points_xyz_world, map_labels = LoadAccumulatedMap()
    print(f"[Done] map_points={map_points_xyz_world.shape[0]}", flush=True)
    VisualizeMap(map_points_xyz_world, map_labels)


if __name__ == "__main__":
    Main()
