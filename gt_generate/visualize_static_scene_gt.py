import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ["WAYMO_DATA_ROOT"])
SEQUENCE_NAME = "6771922013310347577_4249_290_4269_290"

SEQUENCE_ROOT = DATA_ROOT / SEQUENCE_NAME
OUTPUT_ROOT = SEQUENCE_ROOT / "output"
PCD_DIR = OUTPUT_ROOT / "pcd"
DETECTION_INFO_PATH = OUTPUT_ROOT / "detection_info.pkl"
LABEL_MAPPING_PATH = REPO_ROOT / "Scene_Tag_Completion" / "label_mapping" / "waymo_merge_example.yaml"

TERRAIN_FRAME_GT_DIR = OUTPUT_ROOT / "label" / "ground_truth" / "frames"
BACKGROUND_OCC_DIR = OUTPUT_ROOT / "label" / "background_occ"

POINT_STRIDE = 3
TERRAIN_STRIDE = 3
PC_RANGE = np.asarray([-80.0, -80.0, -7.0, 80.0, 80.0, 5.8], dtype=np.float64)
POINT_SIZE = 2.0
WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
FRAME_INTERVAL_SEC = 0.08

sys.path.insert(0, str(REPO_ROOT))

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.io_utils import LoadPcdXyzi, ReadMat


TYPE_TO_NAME = {
    1: "VEHICLE",
    2: "PEDESTRIAN",
    4: "CYCLIST",
}
DETECTION_BOX_COLOR_RGBA = np.asarray([1.0, 0.02, 0.02, 0.38], dtype=np.float32)

CUBE_CORNERS = np.asarray(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)
CUBE_TRIANGLES = np.asarray(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ],
    dtype=np.int32,
)


def BuildSortKey(path):
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem


def LoadColorTable():
    with open(LABEL_MAPPING_PATH, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    class_names = list(cfg["CLASS_NAMES"])
    colors = np.asarray([cfg["VISUALIZATION_COLORS"][name] for name in class_names], dtype=np.float64)
    return class_names, colors


def LoadDetectionDict():
    with open(DETECTION_INFO_PATH, "rb") as fp:
        items = pickle.load(fp)
    output = {}
    for item in items:
        timestamp_ms = int(int(item["frame_timestamp_micros"]) // 1000)
        output[timestamp_ms] = item
    return output


def BuildFrameList():
    frame_list = []
    for pcd_path in sorted(PCD_DIR.glob("*.pcd"), key=BuildSortKey):
        timestamp = int(pcd_path.stem)
        terrain_dir = TERRAIN_FRAME_GT_DIR / f"{timestamp:013d}"
        if terrain_dir.exists() and BuildBackgroundOccPath(timestamp).exists():
            frame_list.append((timestamp, pcd_path, terrain_dir))
    if not frame_list:
        raise RuntimeError(
            f"No frames with terrain GT and background OCC found under "
            f"{TERRAIN_FRAME_GT_DIR} / {BACKGROUND_OCC_DIR}"
        )
    return frame_list


def BuildRawPointCloud(pcd_path):
    points = LoadPcdXyzi(str(pcd_path))[:, :3]
    range_mask = (
        (points[:, 0] >= PC_RANGE[0])
        & (points[:, 1] >= PC_RANGE[1])
        & (points[:, 2] >= PC_RANGE[2])
        & (points[:, 0] <= PC_RANGE[3])
        & (points[:, 1] <= PC_RANGE[4])
        & (points[:, 2] <= PC_RANGE[5])
    )
    points = points[range_mask]
    points = points[:: max(1, POINT_STRIDE)].astype(np.float64, copy=False)
    colors = np.full(points.shape, 0.18, dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def BuildTerrainPointCloud(terrain_dir):
    terrain_path = terrain_dir / "base" / "terrain_completion" / "terrain_z_inpainted.bin"
    meta_path = terrain_dir / "frame_meta.json"
    terrain_z = ReadMat(str(terrain_path)).astype(np.float32)
    with open(meta_path, "r", encoding="utf-8") as fp:
        meta = json.load(fp)

    rows = np.arange(0, terrain_z.shape[0], max(1, TERRAIN_STRIDE), dtype=np.int32)
    cols = np.arange(0, terrain_z.shape[1], max(1, TERRAIN_STRIDE), dtype=np.int32)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    z_values = terrain_z[grid_rows, grid_cols]
    valid_mask = np.isfinite(z_values) & (z_values != INVALID_HEIGHT)
    local_x = float(meta["forward_range_m"]) - (grid_rows.astype(np.float64) + 0.5) * float(meta["map_resolution_m"])
    local_y = float(meta["left_range_m"]) - (grid_cols.astype(np.float64) + 0.5) * float(meta["map_resolution_m"])
    points = np.column_stack([local_x[valid_mask], local_y[valid_mask], z_values[valid_mask].astype(np.float64)])
    colors = np.zeros(points.shape, dtype=np.float64)
    colors[:, 0] = 0.05
    colors[:, 1] = 0.55
    colors[:, 2] = 1.00
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def BuildBackgroundOccPath(timestamp):
    occ_path = BACKGROUND_OCC_DIR / f"{int(timestamp)}.npz"
    if occ_path.exists():
        return occ_path
    return BACKGROUND_OCC_DIR / f"{int(timestamp):013d}.npz"


def LoadBackgroundOcc(timestamp, class_colors):
    occ_path = BuildBackgroundOccPath(timestamp)
    data = np.load(occ_path, allow_pickle=True)
    if "voxel_indices" in data and "voxel_size_xyz" in data and "pc_range" in data:
        voxel_indices = data["voxel_indices"].astype(np.float64)
        voxel_size_xyz = data["voxel_size_xyz"].astype(np.float64)
        pc_range = data["pc_range"].astype(np.float64)
        voxel_origins = pc_range[:3].reshape(1, 3) + voxel_indices * voxel_size_xyz.reshape(1, 3)
    else:
        voxel_size_xyz = np.full(3, 0.2, dtype=np.float64)
        local_centers = data["local_voxel_centers_xyz"].astype(np.float64)
        voxel_origins = local_centers - 0.5 * voxel_size_xyz.reshape(1, 3)
    labels = data["labels"].astype(np.int64)
    valid_mask = labels > 0
    voxel_origins = voxel_origins[valid_mask]
    labels = labels[valid_mask]
    colors = class_colors[np.clip(labels, 0, class_colors.shape[0] - 1)]
    return voxel_origins, voxel_size_xyz, colors, labels


def BuildOccVoxelMesh(voxel_origins, voxel_size_xyz, colors):
    """构造当前帧车体/LiDAR坐标系中的静态 OCC 实体格子。"""
    voxel_num = voxel_origins.shape[0]
    vertices = (
        voxel_origins[:, None, :]
        + CUBE_CORNERS[None, :, :] * voxel_size_xyz.reshape(1, 1, 3)
    ).reshape(-1, 3)
    triangle_offsets = np.arange(voxel_num, dtype=np.int32).reshape(-1, 1, 1) * 8
    triangles = (CUBE_TRIANGLES[None, :, :] + triangle_offsets).reshape(-1, 3)
    vertex_colors = np.repeat(colors, 8, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    return mesh


def BuildBoxMesh(box):
    """构造当前帧车体/LiDAR坐标系中带真实尺寸和 yaw 的检测块。"""
    cx, cy, cz, length, width, height, yaw = [float(item) for item in box[:7]]
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation = np.asarray([[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    mesh = o3d.geometry.TriangleMesh.create_box(width=length, height=width, depth=height)
    mesh.translate((-0.5 * length, -0.5 * width, -0.5 * height))
    mesh.rotate(rotation, center=(0.0, 0.0, 0.0))
    mesh.translate((cx, cy, cz))
    mesh.compute_vertex_normals()
    return mesh


def BuildDetectionBoxes(detection_item):
    combined_mesh = o3d.geometry.TriangleMesh()
    if detection_item is None:
        return combined_mesh
    for obj in detection_item.get("objects", []):
        type_id = int(obj.get("type", 0))
        if type_id not in TYPE_TO_NAME:
            continue
        combined_mesh += BuildBoxMesh(obj["box"])
    return combined_mesh


def BuildRenderMaterials():
    point_material = o3d.visualization.rendering.MaterialRecord()
    point_material.shader = "defaultUnlit"
    point_material.point_size = float(POINT_SIZE)

    occ_material = o3d.visualization.rendering.MaterialRecord()
    occ_material.shader = "defaultLit"

    detection_material = o3d.visualization.rendering.MaterialRecord()
    detection_material.shader = "defaultLitTransparency"
    detection_material.base_color = DETECTION_BOX_COLOR_RGBA
    detection_material.base_roughness = 0.75
    return point_material, occ_material, detection_material


def ParseDisplayMode():
    if len(sys.argv) <= 1:
        return 2
    if sys.argv[1] not in {"1", "2"}:
        raise ValueError("Usage: python visualize_static_scene_gt.py [1|2], default 2=GT only, 1=current frame + GT")
    return int(sys.argv[1])


class Controller:
    def __init__(self, frame_num, display_mode):
        self.frame_num = int(frame_num)
        self.frame_idx = 0
        self.b_paused = True
        self.b_dirty = True
        self.b_show_ground = True
        self.b_show_occ = True
        self.b_show_boxes = True
        self.display_mode = int(display_mode)
        self.b_show_raw = self.display_mode == 1
        self.last_frame_time = time.perf_counter()

    def PrevFrame(self):
        self.frame_idx = max(0, self.frame_idx - 1)
        self.b_paused = True
        self.b_dirty = True

    def NextFrame(self):
        self.frame_idx = min(self.frame_num - 1, self.frame_idx + 1)
        self.b_paused = True
        self.b_dirty = True

    def TogglePause(self):
        self.b_paused = not self.b_paused
        self.last_frame_time = time.perf_counter()

    def AutoStep(self):
        if self.b_paused or self.b_dirty:
            return
        if time.perf_counter() - self.last_frame_time < FRAME_INTERVAL_SEC:
            return
        if self.frame_idx + 1 < self.frame_num:
            self.frame_idx += 1
            self.b_dirty = True
        else:
            self.b_paused = True


def RegisterKeys(scene_widget, controller):
    gui = o3d.visualization.gui

    def OnKey(event):
        if event.type != gui.KeyEvent.Type.DOWN:
            return gui.Widget.EventCallbackResult.IGNORED
        if event.key == gui.KeyName.Q:
            controller.PrevFrame()
        elif event.key == gui.KeyName.E:
            controller.NextFrame()
        elif event.key == gui.KeyName.SPACE:
            controller.TogglePause()
        else:
            return gui.Widget.EventCallbackResult.IGNORED
        return gui.Widget.EventCallbackResult.HANDLED

    scene_widget.set_on_key(OnKey)


def Main():
    display_mode = ParseDisplayMode()
    class_names, class_colors = LoadColorTable()
    detection_dict = LoadDetectionDict()
    frame_list = BuildFrameList()
    controller = Controller(len(frame_list), display_mode)

    print(f"[Dataset] frames={len(frame_list)}")
    print(f"[Dataset] background_occ_dir={BACKGROUND_OCC_DIR}")
    print(f"[Mode] {display_mode} {'current frame + GT' if display_mode == 1 else 'GT only'}")
    print("[Keys] Space play/pause, q prev, e next")

    gui = o3d.visualization.gui
    rendering = o3d.visualization.rendering
    application = gui.Application.instance
    application.initialize()
    window = application.create_window("Static scene GT demo", WINDOW_WIDTH, WINDOW_HEIGHT)
    scene_widget = gui.SceneWidget()
    scene_widget.scene = rendering.Open3DScene(window.renderer)
    scene_widget.scene.set_background(np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    window.add_child(scene_widget)
    window.set_on_layout(lambda layout_context: setattr(scene_widget, "frame", window.content_rect))
    RegisterKeys(scene_widget, controller)

    point_material, occ_material, detection_material = BuildRenderMaterials()

    b_first_frame = True

    def UpdateFrame():
        nonlocal b_first_frame
        timestamp, pcd_path, terrain_dir = frame_list[controller.frame_idx]
        detection_item = detection_dict.get(int(timestamp))
        scene = scene_widget.scene
        scene.clear_geometry()
        geometry_num = 0
        occ_voxel_num = 0

        if controller.b_show_raw:
            scene.add_geometry("raw_points", BuildRawPointCloud(pcd_path), point_material)
            geometry_num += 1
        if controller.b_show_ground:
            scene.add_geometry("ground_points", BuildTerrainPointCloud(terrain_dir), point_material)
            geometry_num += 1
        if controller.b_show_occ:
            occ_origins, occ_voxel_size_xyz, occ_colors, occ_labels = LoadBackgroundOcc(
                timestamp,
                class_colors,
            )
            scene.add_geometry(
                "static_occ",
                BuildOccVoxelMesh(occ_origins, occ_voxel_size_xyz, occ_colors),
                occ_material,
            )
            geometry_num += 1
            occ_voxel_num = occ_labels.shape[0]
        detection_mesh = BuildDetectionBoxes(detection_item)
        if controller.b_show_boxes and len(detection_mesh.vertices) > 0:
            scene.add_geometry("detection_boxes", detection_mesh, detection_material)
            geometry_num += 1

        if b_first_frame:
            bounds = scene.bounding_box
            scene_widget.setup_camera(60.0, bounds, bounds.get_center())
            b_first_frame = False
        box_num = 0 if detection_item is None else len(
            [
                obj
                for obj in detection_item.get("objects", [])
                if int(obj.get("type", 0)) in TYPE_TO_NAME
            ]
        )
        print(
            f"[Frame] {controller.frame_idx + 1}/{len(frame_list)} "
            f"timestamp={timestamp} mode={controller.display_mode} geometries={geometry_num} boxes={box_num} "
            f"occ_voxels={occ_voxel_num} "
            f"show_ground={controller.b_show_ground} show_occ={controller.b_show_occ} "
            f"show_boxes={controller.b_show_boxes} show_raw={controller.b_show_raw}",
            flush=True,
        )
        controller.b_dirty = False
        controller.last_frame_time = time.perf_counter()

    def OnTick():
        controller.AutoStep()
        if not controller.b_dirty:
            return False
        UpdateFrame()
        return True

    window.set_on_tick_event(OnTick)
    UpdateFrame()
    application.run()


if __name__ == "__main__":
    Main()
