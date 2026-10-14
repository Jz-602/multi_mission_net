import sys
import time
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d
import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
for path in [str(PROJECT_DIR), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from pcd_io import ReadCleanPcdXYZ


## 目录格式建议：
## sequence_root/
##   pcd/        # 输入 .pcd 点云，文件名按时间戳或序号排序，例如 1641977312551.pcd
##   pose.txt    # 可选；本脚本逐帧可视化不需要 pose，推理脚本需要。
##   label/      # 推理脚本输出的 .label，默认 label/memory/*.label
##   detection/  # 预留给后续检测结果。
PCD_DIR = Path(
    os.environ["WAYMO_DATA_ROOT"]
) / (
    "15832924468527961_1564_160_1584_160/output/pcd"
)                                                       # 输入点云文件夹；label 会按同名 pcd stem 查找。
MODEL_NAME = "memory"                                   # 要可视化的模型结果子目录名，例如 label/memory。
LABEL_DIR = PCD_DIR.parent / "label" / MODEL_NAME       # label 文件夹；通常保持这个默认结构即可。

LABEL_MAPPING_PATH = PROJECT_DIR / "label_mapping" / "waymo_merge_example.yaml"  # 类别名和颜色表配置。
POINT_STRIDE = 1                    # 可视化抽点间隔；1 表示显示所有点，数值越大越流畅但越稀疏。
POINT_SIZE = 2.0                    # Open3D 点大小。
WINDOW_WIDTH = 1600                 # 可视化窗口宽度。
WINDOW_HEIGHT = 900                 # 可视化窗口高度。
FRAME_INTERVAL_SEC = 0.08           # 自动播放时每帧间隔秒数。
B_LOOP_SEQUENCE = False             # True: 播放到最后一帧后循环；False: 到末尾暂停。


def LoadLabelStyle() -> Tuple[List[str], np.ndarray]:
    with LABEL_MAPPING_PATH.open("r", encoding="utf-8") as file:
        label_mapping_cfg = yaml.safe_load(file)

    class_names = list(label_mapping_cfg["CLASS_NAMES"])
    if "VISUALIZATION_COLORS" not in label_mapping_cfg:
        color_table = np.zeros((len(class_names), 3), dtype=np.float64)
        color_table[:, 0] = np.linspace(0.0, 1.0, len(class_names))
        return class_names, color_table

    color_table = np.asarray(
        [label_mapping_cfg["VISUALIZATION_COLORS"][class_name] for class_name in class_names],
        dtype=np.float64,
    )
    return class_names, color_table


def BuildSortKey(path: Path):
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem


def BuildFrameList() -> List[Tuple[Path, Path]]:
    if not PCD_DIR.is_dir():
        raise FileNotFoundError(f"PCD_DIR not found: {PCD_DIR}")
    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"LABEL_DIR not found: {LABEL_DIR}")

    frame_list: List[Tuple[Path, Path]] = []
    for pcd_path in sorted(PCD_DIR.glob("*.pcd"), key=BuildSortKey):
        label_path = LABEL_DIR / f"{pcd_path.stem}.label"
        if label_path.is_file():
            frame_list.append((pcd_path, label_path))

    if len(frame_list) == 0:
        raise RuntimeError(f"No matched pcd/label frames found: {PCD_DIR} {LABEL_DIR}")
    return frame_list


def LoadLabel(label_path: Path, point_count: int) -> np.ndarray:
    point_label = np.fromfile(str(label_path), dtype=np.uint32).astype(np.int64)
    if point_label.shape[0] != point_count:
        raise ValueError(
            f"label count mismatch: label={point_label.shape[0]}, "
            f"points={point_count}, path={label_path}"
        )
    return point_label


def BuildSemanticPointCloud(
    points_xyz_lidar: np.ndarray,
    point_label: np.ndarray,
    color_table: np.ndarray,
) -> o3d.geometry.PointCloud:
    """
    points_xyz_lidar: [N, 3]，当前帧 LiDAR/车体坐标系 xyz，单位 m。
    point_label: [N]，与 points_xyz_lidar 一一对应的模型预测语义类别 id。
    """
    k_stride = max(int(POINT_STRIDE), 1)
    points_show = points_xyz_lidar[::k_stride].astype(np.float64, copy=False)
    label_show = np.clip(point_label[::k_stride], 0, color_table.shape[0] - 1)
    colors_show = color_table[label_show]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_show)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_show)
    return point_cloud


def PrintClassHistogram(point_label: np.ndarray, class_names: List[str]) -> None:
    class_hist = np.bincount(
        np.clip(point_label, 0, len(class_names) - 1),
        minlength=len(class_names),
    )
    hist_text = ", ".join(
        f"{class_name}:{int(class_hist[class_idx])}"
        for class_idx, class_name in enumerate(class_names)
    )
    print(f"[Hist] {hist_text}", flush=True)


class FrameController:
    def __init__(self, num_frames: int) -> None:
        self.num_frames = int(num_frames)
        self.frame_idx = 0
        self.b_paused = False
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


def Main() -> None:
    class_names, color_table = LoadLabelStyle()
    frame_list = BuildFrameList()
    controller = FrameController(num_frames=len(frame_list))

    print(f"[Dataset] frames={len(frame_list)}", flush=True)
    print(f"[Dataset] pcd_dir={PCD_DIR}", flush=True)
    print(f"[Dataset] label_dir={LABEL_DIR}", flush=True)
    print("[Keys] Space pause/play, q previous frame, e next frame", flush=True)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"{PCD_DIR.parent.name} {MODEL_NAME}",
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
                pcd_path, label_path = frame_list[controller.frame_idx]
                points_xyz_lidar = ReadCleanPcdXYZ(pcd_path)
                point_label = LoadLabel(label_path, points_xyz_lidar.shape[0])
                point_cloud = BuildSemanticPointCloud(points_xyz_lidar, point_label, color_table)

                vis.clear_geometries()
                vis.add_geometry(point_cloud, reset_bounding_box=b_first_frame)
                b_first_frame = False

                print(
                    f"[Frame] {controller.frame_idx + 1}/{len(frame_list)} "
                    f"timestamp={pcd_path.stem} points={points_xyz_lidar.shape[0]}",
                    flush=True,
                )
                PrintClassHistogram(point_label, class_names)

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


if __name__ == "__main__":
    Main()
