import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
ODSS_PROJECT_DIR = REPO_ROOT / "ODSS_project"
for path in [str(ODSS_PROJECT_DIR), str(REPO_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from pcd_io import ReadCleanPcdXYZ


DEFAULT_OCC_SCENE_DIR = Path("/media/estella/Osmanthus/aa_data/FH_data/091")
DEFAULT_PCD_DIR = Path(
    "/media/estella/Osmanthus/aa_data/FH_data/FH_waymo_small/"
    "11967272535264406807_580_000_600_000/output/pcd"
)


# DEFAULT_OCC_SCENE_DIR = Path("/media/estella/Osmanthus/aa_data/FH_data/284")
# DEFAULT_PCD_DIR = Path(
#     "/media/estella/Osmanthus/aa_data/FH_data/FH_waymo_small/"
#     "16102220208346880_1420_000_1440_000/output/pcd"
# )


FREE_LABEL = 15
LEGACY_FREE_LABEL = 23
UNKNOWN_LABEL = 0
DEFAULT_POINT_SIZE = 3.0
DEFAULT_PCD_POINT_SIZE = 1.0
DEFAULT_FRAME_INTERVAL_SEC = 0.12
B_SHOW_SINGLE_FRAME_PCD = False
INVISIBLE_MASK_KEY = "origin_voxel_state"
INVISIBLE_COLOR = np.asarray([0.82, 0.82, 0.82], dtype=np.float64)

LABEL_COLORS = {
    0: [0.55, 0.55, 0.55],  # general object
    1: [0.90, 0.12, 0.10],  # vehicle
    2: [0.10, 0.55, 1.00],  # pedestrian
    3: [1.00, 0.78, 0.12],  # sign
    4: [0.65, 0.20, 0.95],  # cyclist
    5: [0.95, 0.25, 0.55],  # traffic light
    6: [0.20, 0.80, 0.85],  # pole
    7: [1.00, 0.48, 0.12],  # construction cone
    8: [0.12, 0.75, 0.35],  # bicycle
    9: [0.00, 0.48, 0.80],  # motorcycle
    10: [0.36, 0.42, 0.62],  # building
    11: [0.15, 0.68, 0.20],  # vegetation
    12: [0.42, 0.28, 0.14],  # tree trunk
    13: [0.30, 0.30, 0.30],  # road
    14: [0.72, 0.72, 0.48],  # walkable
    15: [0.04, 0.04, 0.04],  # free
    23: [0.04, 0.04, 0.04],  # legacy free in existing FH_data npz
}


def BuildSortKey(path: Path):
    stem = path.stem.replace("_04", "")
    try:
        return 0, int(stem)
    except ValueError:
        return 1, stem


def ResolveGridMeta(voxel_label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    输入 voxel_label: [X, Y, Z] uint8，Occ3D 当前帧车体/LiDAR坐标系下的体素语义标签。
    输出 voxel_origin_xyz: [3] 和 voxel_size_xyz: [3]，单位 m，均在当前帧车体/LiDAR坐标系。
    """
    if tuple(voxel_label.shape) == (200, 200, 16):
        return (
            np.asarray([-40.0, -40.0, -1.0], dtype=np.float32),
            np.asarray([0.4, 0.4, 0.4], dtype=np.float32),
        )
    if tuple(voxel_label.shape) == (1600, 1600, 64):
        return (
            np.asarray([-80.0, -80.0, -5.0], dtype=np.float32),
            np.asarray([0.1, 0.1, 0.2], dtype=np.float32),
        )
    raise ValueError(f"unsupported Occ3D voxel shape: {voxel_label.shape}")


def BuildColorTable() -> np.ndarray:
    color_table = np.zeros((256, 3), dtype=np.float64)
    color_table[:] = np.asarray([0.8, 0.8, 0.8], dtype=np.float64)
    for label, color in LABEL_COLORS.items():
        color_table[label] = np.asarray(color, dtype=np.float64)
    return color_table


def LoadOccPoints(
    occ_path: Path,
    state_filter: str,
    b_show_free: bool,
    occ_stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    输入 occ_path: Occ3D npz，包含 voxel_label/origin_voxel_state/final_voxel_state/infov/ego2global。
    输出 occ_points_xyz: [N, 3] float64，非 free 体素中心点，当前帧车体/LiDAR坐标系，单位 m；
    输出 occ_labels: [N] uint8，对应每个体素中心点的语义标签。
    输出 b_visible_mask: [N] bool，True 表示该体素在当前帧视野内，False 表示看不见区域。
    """
    occ_data = np.load(occ_path)
    voxel_label = occ_data["voxel_label"]
    voxel_origin_xyz, voxel_size_xyz = ResolveGridMeta(voxel_label)

    if b_show_free:
        b_valid_mask = np.ones(voxel_label.shape, dtype=bool)
    else:
        b_valid_mask = (voxel_label != FREE_LABEL) & (voxel_label != LEGACY_FREE_LABEL)

    if state_filter != "none":
        if state_filter not in occ_data.files:
            raise KeyError(f"state_filter={state_filter} not found in {occ_path}")
        b_valid_mask &= occ_data[state_filter].astype(bool)

    indices_xyz = np.argwhere(b_valid_mask)
    if occ_stride > 1:
        indices_xyz = indices_xyz[::occ_stride]

    if indices_xyz.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.uint8),
            np.empty((0,), dtype=bool),
        )

    occ_points_xyz = voxel_origin_xyz + (indices_xyz.astype(np.float32) + 0.5) * voxel_size_xyz
    occ_labels = voxel_label[
        indices_xyz[:, 0],
        indices_xyz[:, 1],
        indices_xyz[:, 2],
    ]
    if INVISIBLE_MASK_KEY in occ_data.files:
        b_visible_mask = occ_data[INVISIBLE_MASK_KEY][
            indices_xyz[:, 0],
            indices_xyz[:, 1],
            indices_xyz[:, 2],
        ].astype(bool)
    else:
        b_visible_mask = np.ones((indices_xyz.shape[0],), dtype=bool)
    return occ_points_xyz.astype(np.float64), occ_labels, b_visible_mask


def BuildOccGeometry(
    occ_points_xyz: np.ndarray,
    occ_labels: np.ndarray,
    b_visible_mask: np.ndarray,
    color_table: np.ndarray,
    render_mode: str,
    voxel_size: float,
):
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(occ_points_xyz)
    colors = color_table[occ_labels].copy()
    colors[~b_visible_mask] = INVISIBLE_COLOR
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    if render_mode == "points":
        return point_cloud
    return o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)


def BuildPcdGeometry(pcd_path: Path, pcd_stride: int):
    """
    输入 pcd_path: PCD 文件，字段至少包含 x/y/z。
    输出 Open3D 点云；points_xyz_lidar 为 [N, 3] float32，当前帧车体/LiDAR坐标系，单位 m。
    """
    import open3d as o3d

    points_xyz_lidar = ReadCleanPcdXYZ(pcd_path)
    points_xyz_lidar = points_xyz_lidar[::max(int(pcd_stride), 1)]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_xyz_lidar.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.full((points_xyz_lidar.shape[0], 3), 0.72, dtype=np.float64)
    )
    return point_cloud


def BuildFrameList(
    occ_scene_dir: Path,
    pcd_dir: Path,
    occ_pattern: str,
    b_show_single_frame_pcd: bool,
) -> List[Tuple[Path, Optional[Path]]]:
    if not occ_scene_dir.is_dir():
        raise FileNotFoundError(f"occ_scene_dir not found: {occ_scene_dir}")

    occ_paths = sorted(occ_scene_dir.glob(occ_pattern), key=BuildSortKey)
    if len(occ_paths) == 0:
        raise RuntimeError(f"no occ files found: {occ_scene_dir}/{occ_pattern}")

    if not b_show_single_frame_pcd:
        return [(occ_path, None) for occ_path in occ_paths]

    if not pcd_dir.is_dir():
        raise FileNotFoundError(f"pcd_dir not found: {pcd_dir}")

    pcd_paths = sorted(pcd_dir.glob("*.pcd"), key=BuildSortKey)
    if len(pcd_paths) == 0:
        raise RuntimeError(f"no pcd files found: {pcd_dir}")

    pair_count = min(len(occ_paths), len(pcd_paths))
    if len(occ_paths) != len(pcd_paths):
        print(
            f"[Warn] occ_count={len(occ_paths)} pcd_count={len(pcd_paths)}, "
            f"only pair first {pair_count} frames by sorted order",
            flush=True,
        )
    return list(zip(occ_paths[:pair_count], pcd_paths[:pair_count]))


def PrintOccHistogram(occ_labels: np.ndarray) -> None:
    if occ_labels.shape[0] == 0:
        print("[Occ] empty", flush=True)
        return
    labels, counts = np.unique(occ_labels, return_counts=True)
    hist_text = ", ".join(f"{int(label)}:{int(count)}" for label, count in zip(labels, counts))
    print(f"[Occ] labels {hist_text}", flush=True)


class FrameController:
    def __init__(self, num_frames: int, b_paused: bool, frame_interval_sec: float) -> None:
        self.num_frames = int(num_frames)
        self.frame_idx = 0
        self.b_paused = bool(b_paused)
        self.b_dirty = True
        self.frame_interval_sec = float(frame_interval_sec)
        self.last_frame_time = 0.0

    def PrevFrame(self, vis) -> bool:
        if self.frame_idx > 0:
            self.frame_idx -= 1
        self.b_paused = True
        self.b_dirty = True
        print(f"[Control] previous frame {self.frame_idx + 1}/{self.num_frames}", flush=True)
        return False

    def NextFrame(self, vis) -> bool:
        if self.frame_idx + 1 < self.num_frames:
            self.frame_idx += 1
        self.b_paused = True
        self.b_dirty = True
        print(f"[Control] next frame {self.frame_idx + 1}/{self.num_frames}", flush=True)
        return False

    def TogglePause(self, vis) -> bool:
        self.b_paused = not self.b_paused
        self.last_frame_time = time.perf_counter()
        print(f"[Control] {'pause' if self.b_paused else 'play'}", flush=True)
        return False

    def AutoStep(self) -> None:
        if self.b_paused or self.b_dirty:
            return
        if time.perf_counter() - self.last_frame_time < self.frame_interval_sec:
            return
        if self.frame_idx + 1 < self.num_frames:
            self.frame_idx += 1
            self.b_dirty = True
        else:
            self.b_paused = True


def RegisterKeys(vis, controller: FrameController) -> None:
    ## 空格: 暂停/继续；q/Q: 上一帧；e/E: 下一帧。
    vis.register_key_callback(ord("q"), controller.PrevFrame)
    vis.register_key_callback(ord("Q"), controller.PrevFrame)
    vis.register_key_callback(ord("e"), controller.NextFrame)
    vis.register_key_callback(ord("E"), controller.NextFrame)
    vis.register_key_callback(32, controller.TogglePause)


def ParseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Occ3D-Waymo occupancy GT with matched Waymo PCD.")
    parser.add_argument("--occ-scene-dir", type=Path, default=DEFAULT_OCC_SCENE_DIR)
    parser.add_argument("--pcd-dir", type=Path, default=DEFAULT_PCD_DIR)
    parser.add_argument("--occ-pattern", type=str, default="[0-9][0-9][0-9].npz")
    parser.add_argument("--state-filter", choices=["none", "origin_voxel_state", "final_voxel_state", "infov"], default="none")
    parser.add_argument("--render-mode", choices=["points", "voxels"], default="points")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--occ-stride", type=int, default=4)
    parser.add_argument("--pcd-stride", type=int, default=4)
    parser.add_argument("--point-size", type=float, default=DEFAULT_POINT_SIZE)
    parser.add_argument("--pcd-point-size", type=float, default=DEFAULT_PCD_POINT_SIZE)
    parser.add_argument("--frame-interval-sec", type=float, default=DEFAULT_FRAME_INTERVAL_SEC)
    parser.add_argument("--show-free", action="store_true")
    parser.add_argument("--hide-pcd", action="store_true")
    parser.add_argument("--play", action="store_true")
    return parser.parse_args()


def Main() -> None:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required; run with conda env multi_mission.") from exc

    args = ParseArgs()
    b_show_single_frame_pcd = bool(B_SHOW_SINGLE_FRAME_PCD) and not args.hide_pcd
    frame_list = BuildFrameList(
        args.occ_scene_dir,
        args.pcd_dir,
        args.occ_pattern,
        b_show_single_frame_pcd,
    )
    if args.max_frames > 0:
        frame_list = frame_list[: args.max_frames]
    if not 0 <= args.frame < len(frame_list):
        raise ValueError(f"--frame must be in [0, {len(frame_list) - 1}], got {args.frame}")

    color_table = BuildColorTable()
    controller = FrameController(
        num_frames=len(frame_list),
        b_paused=not args.play,
        frame_interval_sec=args.frame_interval_sec,
    )
    controller.frame_idx = args.frame

    print(f"[Dataset] occ_scene_dir={args.occ_scene_dir}", flush=True)
    print(f"[Dataset] b_show_single_frame_pcd={b_show_single_frame_pcd}", flush=True)
    if b_show_single_frame_pcd:
        print(f"[Dataset] pcd_dir={args.pcd_dir}", flush=True)
    print(f"[Dataset] frames={len(frame_list)}", flush=True)
    print("[Keys] Space pause/play, q previous frame, e next frame", flush=True)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Occ3D {args.occ_scene_dir.name}", width=1600, height=900)
    RegisterKeys(vis, controller)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option.background_color = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        render_option.point_size = float(args.point_size)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0.0, 0.0, 0.0])
    b_window_alive = True
    b_first_frame = True
    try:
        while b_window_alive:
            if controller.b_dirty:
                occ_path, pcd_path = frame_list[controller.frame_idx]
                occ_points_xyz, occ_labels, b_visible_mask = LoadOccPoints(
                    occ_path=occ_path,
                    state_filter=args.state_filter,
                    b_show_free=args.show_free,
                    occ_stride=max(int(args.occ_stride), 1),
                )
                occ_geometry = BuildOccGeometry(
                    occ_points_xyz=occ_points_xyz,
                    occ_labels=occ_labels,
                    b_visible_mask=b_visible_mask,
                    color_table=color_table,
                    render_mode=args.render_mode,
                    voxel_size=0.4,
                )

                vis.clear_geometries()
                vis.add_geometry(axis, reset_bounding_box=b_first_frame)
                if b_show_single_frame_pcd and pcd_path is not None:
                    pcd_geometry = BuildPcdGeometry(pcd_path, args.pcd_stride)
                    vis.add_geometry(pcd_geometry, reset_bounding_box=False)
                vis.add_geometry(occ_geometry, reset_bounding_box=b_first_frame)
                b_first_frame = False

                pcd_name = pcd_path.name if pcd_path is not None else "hidden"
                print(
                    f"[Frame] {controller.frame_idx + 1}/{len(frame_list)} "
                    f"occ={occ_path.name} pcd={pcd_name} occ_points={occ_points_xyz.shape[0]} "
                    f"invisible={int(np.count_nonzero(~b_visible_mask))}",
                    flush=True,
                )
                PrintOccHistogram(occ_labels)
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
