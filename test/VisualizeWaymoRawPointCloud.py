from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lyf_tools import CarToWorld
from lyf_tools import LoadPosesFromPoseTxt

SEQUENCE_NAME = "10023947602400723454_1120_000_1140_000"
START_FRAME = 0
END_FRAME = -1
FPS = 20
VOXEL_SIZE = 0.0
POINT_SIZE = 2.0
B_USE_WORLD = False
B_LOOP = False
BACKGROUND_COLOR = (0.0, 0.0, 0.0)


def BuildSortKey(file_path: Path):
    stem = file_path.stem
    try:
        return 0, int(stem)
    except ValueError:
        return 1, stem


def FindPoseFile(sequence_output_dir: Path) -> Path | None:
    candidate_path_list = [
        sequence_output_dir / "pose_slam_euler_hba_refined.txt",
        sequence_output_dir / "pose_slam_euler.txt",
        sequence_output_dir / "pose_slam_euler_waymo.txt",
    ]
    for pose_path in candidate_path_list:
        if pose_path.is_file():
            return pose_path
    return None


def ResolveSequenceOutputDir(waymo_data_root: Path, sequence_name: str | None) -> Path:
    if sequence_name is not None:
        sequence_output_dir = waymo_data_root / sequence_name / "output"
        if not sequence_output_dir.is_dir():
            raise FileNotFoundError(f"未找到 sequence output 目录: {sequence_output_dir}")
        return sequence_output_dir

    sequence_output_dir_list = sorted(
        path / "output"
        for path in waymo_data_root.iterdir()
        if path.is_dir() and (path / "output" / "pcd").is_dir()
    )
    if not sequence_output_dir_list:
        raise RuntimeError(f"WAYMO_DATA_ROOT 下没有找到包含 output/pcd 的序列: {waymo_data_root}")
    return sequence_output_dir_list[0]


def LoadPointsXYZ(pcd_path: Path) -> np.ndarray:
    ## 读取 Waymo 单帧原始点云。
    ## 输入 pcd_path: sequence/output/pcd/<timestamp_ms>.pcd。
    ## 输出 points_xyz: shape [N, 3]，每行 [x, y, z]，坐标系为当前帧车体/LiDAR 坐标系。
    point_cloud = o3d.io.read_point_cloud(str(pcd_path))
    points_xyz = np.asarray(point_cloud.points, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"点云必须是 shape [N, 3]，但 {pcd_path} 读到 {points_xyz.shape}")
    return points_xyz


def DownsamplePoints(points_xyz: np.ndarray, voxel_size: float | None) -> np.ndarray:
    if voxel_size is None or voxel_size <= 0:
        return points_xyz

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
    point_cloud = point_cloud.voxel_down_sample(voxel_size)
    return np.asarray(point_cloud.points, dtype=np.float32)


def ColorizeByHeight(points_xyz: np.ndarray) -> np.ndarray:
    ## 输入 points_xyz: shape [N, 3]，当前显示坐标系下的点云，按 z 高度赋 RGB 颜色。
    z_values = points_xyz[:, 2]
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    if z_max <= z_min:
        return np.ones((points_xyz.shape[0], 3), dtype=np.float32)

    height_ratio = (z_values - z_min) / (z_max - z_min)
    colors = np.zeros((points_xyz.shape[0], 3), dtype=np.float32)
    colors[:, 0] = height_ratio
    colors[:, 1] = 1.0 - np.abs(height_ratio - 0.5) * 2.0
    colors[:, 2] = 1.0 - height_ratio
    return colors


def StreamPointCloudSequence(
    pcd_path_list: list[Path],
    pose_list: list[np.ndarray] | None,
    voxel_size: float | None,
    fps: int,
    point_size: float,
    background_color: tuple[float, float, float],
    start_frame: int,
    end_frame: int,
    b_loop: bool,
    window_name: str,
) -> None:
    """
    单窗口播放 Waymo 原始点云序列。

    pcd_path_list: 每个元素是 sequence/output/pcd/<timestamp_ms>.pcd。
    pose_list: None 时显示当前帧车体/LiDAR 坐标系；非 None 时 pose6 shape [6]，
        表示当前帧车体坐标系 -> 世界坐标系，用于将 points_xyz shape [N, 3] 转到世界坐标系。
    """
    if not pcd_path_list:
        raise RuntimeError("pcd_path_list 为空。")

    if end_frame < 0 or end_frame >= len(pcd_path_list):
        end_frame = len(pcd_path_list) - 1
    if start_frame < 0 or start_frame > end_frame:
        raise ValueError(f"帧范围错误: start_frame={start_frame}, end_frame={end_frame}")

    pcd_path_list = pcd_path_list[start_frame:end_frame + 1]
    if pose_list is not None:
        pose_list = pose_list[start_frame:end_frame + 1]

    frame_count = len(pcd_path_list)
    frame_interval_sec = 1.0 / max(1, fps)

    print(f"播放帧范围: [{start_frame}, {end_frame}]，共 {frame_count} 帧")
    print("按键: 空格=暂停/继续, q=上一帧, e=下一帧, ESC=退出")

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name=window_name)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray(background_color, dtype=np.float32)
    render_option.point_size = float(point_size)

    point_cloud = o3d.geometry.PointCloud()
    state = {
        "frame_idx": 0,
        "b_paused": False,
        "step": 0,
        "b_quit": False,
    }

    def LoadFrame(frame_idx: int) -> None:
        points_xyz = LoadPointsXYZ(pcd_path_list[frame_idx])
        if pose_list is not None:
            points_xyz = CarToWorld(points_xyz, pose_list[frame_idx])
        points_xyz = DownsamplePoints(points_xyz, voxel_size)

        point_cloud.points = o3d.utility.Vector3dVector(points_xyz)
        point_cloud.colors = o3d.utility.Vector3dVector(ColorizeByHeight(points_xyz))

        global_frame_idx = start_frame + frame_idx
        coord_name = "world" if pose_list is not None else "car/lidar"
        print(
            f"\r显示帧 {global_frame_idx}/{end_frame} | "
            f"{pcd_path_list[frame_idx].name} | points={points_xyz.shape[0]} | coord={coord_name}",
            end="",
            flush=True,
        )

    def TogglePause(vis) -> bool:
        state["b_paused"] = not state["b_paused"]
        print(f"\n暂停: {state['b_paused']}")
        return False

    def StepPrev(vis) -> bool:
        state["step"] = -1
        state["b_paused"] = True
        return False

    def StepNext(vis) -> bool:
        state["step"] = 1
        state["b_paused"] = True
        return False

    def Quit(vis) -> bool:
        state["b_quit"] = True
        return False

    visualizer.register_key_callback(32, TogglePause)
    visualizer.register_key_callback(ord("q"), StepPrev)
    visualizer.register_key_callback(ord("Q"), StepPrev)
    visualizer.register_key_callback(ord("e"), StepNext)
    visualizer.register_key_callback(ord("E"), StepNext)
    visualizer.register_key_callback(256, Quit)

    LoadFrame(state["frame_idx"])
    visualizer.add_geometry(point_cloud)

    while not state["b_quit"]:
        if state["b_paused"]:
            if state["step"] != 0:
                new_frame_idx = max(0, min(frame_count - 1, state["frame_idx"] + state["step"]))
                if new_frame_idx != state["frame_idx"]:
                    state["frame_idx"] = new_frame_idx
                    LoadFrame(state["frame_idx"])
                    visualizer.update_geometry(point_cloud)
                state["step"] = 0
            visualizer.poll_events()
            visualizer.update_renderer()
            time.sleep(0.01)
            continue

        if state["frame_idx"] < frame_count - 1:
            state["frame_idx"] += 1
            LoadFrame(state["frame_idx"])
            visualizer.update_geometry(point_cloud)
        elif b_loop:
            state["frame_idx"] = 0
            LoadFrame(state["frame_idx"])
            visualizer.update_geometry(point_cloud)
        else:
            state["b_paused"] = True
            print("\n播放到最后一帧，自动暂停。")

        visualizer.poll_events()
        visualizer.update_renderer()
        time.sleep(frame_interval_sec)

    visualizer.destroy_window()
    print("\n播放器退出。")


def Main() -> None:
    waymo_data_root = Path(os.environ["WAYMO_DATA_ROOT"])
    sequence_output_dir = ResolveSequenceOutputDir(waymo_data_root, SEQUENCE_NAME)
    pcd_dir = sequence_output_dir / "pcd"
    pcd_path_list = sorted(pcd_dir.glob("*.pcd"), key=BuildSortKey)
    if not pcd_path_list:
        raise RuntimeError(f"未找到 pcd 文件: {pcd_dir}")

    pose_list = None
    if B_USE_WORLD:
        pose_path = FindPoseFile(sequence_output_dir)
        if pose_path is None:
            raise FileNotFoundError(f"未找到 pose_slam_euler*.txt: {sequence_output_dir}")
        frame_id_list = [pcd_path.stem for pcd_path in pcd_path_list]
        pose_list = LoadPosesFromPoseTxt(str(pose_path), frame_id_list=frame_id_list)
        print(f"使用位姿文件: {pose_path}")

    StreamPointCloudSequence(
        pcd_path_list=pcd_path_list,
        pose_list=pose_list,
        voxel_size=VOXEL_SIZE,
        fps=FPS,
        point_size=POINT_SIZE,
        background_color=BACKGROUND_COLOR,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        b_loop=B_LOOP,
        window_name=sequence_output_dir.parent.name,
    )


if __name__ == "__main__":
    Main()
