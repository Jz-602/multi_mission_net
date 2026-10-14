"""
选择序列有效帧列表的中间一帧，绘制静态场景真值：

1. 地面真值：使用不透明点表示；
2. 静态占据：使用原始 0.2 m 实心体素，并按语义标签着色；
3. 目标检测：使用带真实尺寸和 yaw 朝向的实心长方体表示。

脚本仅打开 Open3D 交互窗口，不生成图片或输出目录。
"""

from __future__ import annotations

import json
import os
import pickle
import struct
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ["WAYMO_DATA_ROOT"])
SEQUENCE_NAME = "16102220208346880_1420_000_1440_000"

OUTPUT_ROOT = DATA_ROOT / SEQUENCE_NAME / "output"
PCD_DIR = OUTPUT_ROOT / "pcd"
DETECTION_INFO_PATH = OUTPUT_ROOT / "detection_info.pkl"
TERRAIN_FRAME_GT_DIR = OUTPUT_ROOT / "label" / "ground_truth" / "frames"
BACKGROUND_OCC_DIR = OUTPUT_ROOT / "label" / "background_occ"
LABEL_MAPPING_PATH = (
    REPO_ROOT
    / "Scene_Tag_Completion"
    / "label_mapping"
    / "waymo_merge_example.yaml"
)

# 扫描序列后选出的行人较多且靠近坐标中心的有效帧。
# 设为 None 时，自动选择有效帧列表的中间一帧。
TARGET_TIMESTAMP: int | None = 1554232889658

# 只影响最终画面裁剪，不改变真值坐标。
DISPLAY_PC_RANGE = np.array(
    [-40.0, -40.0, -5.0, 40.0, 40.0, 4.0],
    dtype=np.float64,
)

# 地面原始分辨率为 0.2 m。使用全部有效网格点，并增大点尺寸。
GROUND_STRIDE = 1
GROUND_COLOR = (0.06, 0.43, 0.72, 1.0)
OPEN3D_GROUND_POINT_SIZE = 3.0

# Waymo 检测类型，颜色均为完全不透明。
TYPE_TO_NAME = {
    1: "VEHICLE",
    2: "PEDESTRIAN",
    4: "CYCLIST",
}
DETECTION_COLORS = {
    "VEHICLE": (0.91, 0.18, 0.16, 1.0),
    "PEDESTRIAN": (1.00, 0.52, 0.08, 1.0),
    "CYCLIST": (0.66, 0.23, 0.82, 1.0),
}
CUBE_CORNERS = np.array(
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
    dtype=np.float32,
)
CUBE_TRIANGLES = np.array(
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


def read_mat(path: Path) -> np.ndarray:
    """读取项目使用的 OpenCV Mat 二进制格式。"""
    with path.open("rb") as file:
        rows, cols, cv_type = struct.unpack("<iii", file.read(12))
        depth = cv_type & 7
        channels = 1 + (cv_type >> 3)
        dtype_map = {
            0: np.uint8,
            2: np.uint16,
            4: np.int32,
            5: np.float32,
            6: np.float64,
        }
        if depth not in dtype_map:
            raise ValueError(f"不支持的 OpenCV Mat depth：{depth}")
        data = np.frombuffer(file.read(), dtype=dtype_map[depth])
    if channels > 1:
        return data.reshape(rows, cols, channels).copy()
    return data.reshape(rows, cols).copy()


def build_background_occ_path(timestamp: int) -> Path:
    path = BACKGROUND_OCC_DIR / f"{timestamp}.npz"
    if path.exists():
        return path
    return BACKGROUND_OCC_DIR / f"{timestamp:013d}.npz"


def build_frame_list() -> list[tuple[int, Path]]:
    frames = []
    pcd_paths = sorted(PCD_DIR.glob("*.pcd"), key=lambda path: int(path.stem))
    for pcd_path in pcd_paths:
        timestamp = int(pcd_path.stem)
        terrain_dir = TERRAIN_FRAME_GT_DIR / f"{timestamp:013d}"
        if terrain_dir.exists() and build_background_occ_path(timestamp).exists():
            frames.append((timestamp, terrain_dir))
    if not frames:
        raise RuntimeError("没有找到同时包含地面真值和静态占据的有效帧")
    return frames


def load_semantic_colors() -> tuple[list[str], np.ndarray]:
    with LABEL_MAPPING_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    class_names = list(config["CLASS_NAMES"])
    colors = np.asarray(
        [
            [*config["VISUALIZATION_COLORS"][name], 1.0]
            for name in class_names
        ],
        dtype=np.float64,
    )
    return class_names, colors


def load_detection_dict() -> dict[int, dict]:
    with DETECTION_INFO_PATH.open("rb") as file:
        items = pickle.load(file)
    return {
        int(int(item["frame_timestamp_micros"]) // 1000): item
        for item in items
    }


def load_ground_points(terrain_dir: Path) -> np.ndarray:
    terrain_path = (
        terrain_dir
        / "base"
        / "terrain_completion"
        / "terrain_z_inpainted.bin"
    )
    meta_path = terrain_dir / "frame_meta.json"
    terrain_z = read_mat(terrain_path).astype(np.float32)
    with meta_path.open("r", encoding="utf-8") as file:
        meta = json.load(file)

    rows = np.arange(0, terrain_z.shape[0], GROUND_STRIDE, dtype=np.int32)
    cols = np.arange(0, terrain_z.shape[1], GROUND_STRIDE, dtype=np.int32)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    z_values = terrain_z[grid_rows, grid_cols]

    # 项目常量 INVALID_HEIGHT = -123456.0。
    valid = np.isfinite(z_values) & (z_values != -123456.0)
    resolution = float(meta["map_resolution_m"])
    local_x = (
        float(meta["forward_range_m"])
        - (grid_rows.astype(np.float64) + 0.5) * resolution
    )
    local_y = (
        float(meta["left_range_m"])
        - (grid_cols.astype(np.float64) + 0.5) * resolution
    )
    points = np.column_stack(
        [
            local_x[valid],
            local_y[valid],
            z_values[valid].astype(np.float64),
        ]
    )
    return crop_points(points, DISPLAY_PC_RANGE)


def crop_points(points: np.ndarray, pc_range: np.ndarray) -> np.ndarray:
    lower = pc_range[:3]
    upper = pc_range[3:]
    inside = (points >= lower).all(axis=1) & (points < upper).all(axis=1)
    return points[inside]


def load_static_occ(
    timestamp: int,
    semantic_colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回体素原点、体素尺寸、RGBA 颜色和语义标签。"""
    data = np.load(build_background_occ_path(timestamp), allow_pickle=True)
    voxel_indices = data["voxel_indices"].astype(np.float64)
    voxel_size = data["voxel_size_xyz"].astype(np.float64)
    source_range = data["pc_range"].astype(np.float64)
    labels = data["labels"].astype(np.int64)

    origins = source_range[:3] + voxel_indices * voxel_size
    centers = origins + 0.5 * voxel_size
    lower = DISPLAY_PC_RANGE[:3]
    upper = DISPLAY_PC_RANGE[3:]
    keep = (
        (centers >= lower).all(axis=1)
        & (centers < upper).all(axis=1)
        & (labels > 0)
    )
    origins = origins[keep]
    labels = labels[keep]
    colors = semantic_colors[
        np.clip(labels, 0, len(semantic_colors) - 1)
    ]
    return origins, voxel_size, colors, labels


def load_detection_boxes(
    detection_item: dict | None,
) -> list[tuple[np.ndarray, str]]:
    output = []
    if detection_item is None:
        return output
    lower = DISPLAY_PC_RANGE[:3]
    upper = DISPLAY_PC_RANGE[3:]
    for obj in detection_item.get("objects", []):
        type_id = int(obj.get("type", 0))
        if type_id not in TYPE_TO_NAME:
            continue
        box = np.asarray(obj["box"], dtype=np.float64)
        center = box[:3]
        if (center >= lower).all() and (center < upper).all():
            output.append((box, TYPE_TO_NAME[type_id]))
    return output


def build_open3d_voxel_mesh(
    origins: np.ndarray,
    voxel_size: np.ndarray,
    colors: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    """将全部语义体素批量合并为一个可交互显示的实体网格。"""
    voxel_count = len(origins)
    vertices = (
        origins.astype(np.float64)[:, None, :]
        + CUBE_CORNERS.astype(np.float64)[None, :, :]
        * voxel_size.astype(np.float64)[None, None, :]
    ).reshape(-1, 3)
    triangle_offsets = (
        np.arange(voxel_count, dtype=np.int32)[:, None, None] * 8
    )
    triangles = (
        CUBE_TRIANGLES[None, :, :] + triangle_offsets
    ).reshape(-1, 3)
    vertex_colors = np.repeat(colors[:, :3], 8, axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.compute_vertex_normals()
    return mesh


def build_open3d_detection_mesh(
    detection_boxes: list[tuple[np.ndarray, str]],
) -> o3d.geometry.TriangleMesh:
    """构造带真实尺寸和 yaw 的实体检测长方体。"""
    combined = o3d.geometry.TriangleMesh()
    for box, class_name in detection_boxes:
        cx, cy, cz, length, width, height, yaw = [
            float(value) for value in box[:7]
        ]
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=length,
            height=width,
            depth=height,
        )
        mesh.translate((-length / 2, -width / 2, -height / 2))
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        mesh.rotate(rotation, center=(0.0, 0.0, 0.0))
        mesh.translate((cx, cy, cz))
        mesh.paint_uniform_color(DETECTION_COLORS[class_name][:3])
        mesh.compute_vertex_normals()
        combined += mesh
    return combined


def show_interactive_scene(
    ground_points: np.ndarray,
    occ_origins: np.ndarray,
    occ_voxel_size: np.ndarray,
    occ_colors: np.ndarray,
    detection_boxes: list[tuple[np.ndarray, str]],
) -> None:
    """打开 Open3D 窗口，可用鼠标自由旋转、缩放和平移。"""
    ground_cloud = o3d.geometry.PointCloud()
    ground_cloud.points = o3d.utility.Vector3dVector(ground_points)
    ground_cloud.colors = o3d.utility.Vector3dVector(
        np.tile(
            np.asarray(GROUND_COLOR[:3], dtype=np.float64),
            (len(ground_points), 1),
        )
    )
    occ_mesh = build_open3d_voxel_mesh(
        occ_origins,
        occ_voxel_size,
        occ_colors,
    )
    detection_mesh = build_open3d_detection_mesh(detection_boxes)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        window_name="Middle-frame static scene GT",
        width=1600,
        height=1000,
    )
    visualizer.add_geometry(ground_cloud)
    visualizer.add_geometry(occ_mesh)
    if len(detection_mesh.vertices):
        visualizer.add_geometry(detection_mesh)

    render_option = visualizer.get_render_option()
    if render_option is not None:
        render_option.background_color = np.array(
            [1.0, 1.0, 1.0],
            dtype=np.float64,
        )
        render_option.point_size = OPEN3D_GROUND_POINT_SIZE
        render_option.mesh_show_back_face = True
        render_option.light_on = True

    print(
        "Open3D 操作：鼠标左键旋转，滚轮缩放，"
        "Shift+左键或鼠标中键平移，关闭窗口结束。"
    )
    visualizer.run()
    visualizer.destroy_window()


def main() -> None:
    frame_list = build_frame_list()
    if TARGET_TIMESTAMP is None:
        selected_index = len(frame_list) // 2
    else:
        selected_index = next(
            (
                index
                for index, (timestamp, _) in enumerate(frame_list)
                if timestamp == TARGET_TIMESTAMP
            ),
            -1,
        )
        if selected_index < 0:
            raise ValueError(
                f"TARGET_TIMESTAMP={TARGET_TIMESTAMP} 不是有效真值帧"
            )
    timestamp, terrain_dir = frame_list[selected_index]

    class_names, semantic_colors = load_semantic_colors()
    detection_dict = load_detection_dict()
    ground_points = load_ground_points(terrain_dir)
    occ_origins, occ_voxel_size, occ_colors, occ_labels = load_static_occ(
        timestamp,
        semantic_colors,
    )
    detection_boxes = load_detection_boxes(detection_dict.get(timestamp))

    print(
        f"有效帧：{len(frame_list)}，选择第 {selected_index + 1} 帧，"
        f"timestamp={timestamp}"
    )
    print(f"地面点：{len(ground_points)}")
    print(
        f"静态占据：{len(occ_origins)}，"
        f"体素尺寸：{occ_voxel_size.tolist()}"
    )
    for label, count in zip(*np.unique(occ_labels, return_counts=True)):
        class_name = (
            class_names[int(label)]
            if 0 <= int(label) < len(class_names)
            else f"LABEL_{label}"
        )
        print(
            f"  {class_name}: {int(count)}，"
            f"颜色={semantic_colors[int(label), :3].tolist()}"
        )
    box_counts: dict[str, int] = {}
    for _, class_name in detection_boxes:
        box_counts[class_name] = box_counts.get(class_name, 0) + 1
    print(f"检测长方体：{len(detection_boxes)}，分类统计：{box_counts}")

    show_interactive_scene(
        ground_points,
        occ_origins,
        occ_voxel_size,
        occ_colors,
        detection_boxes,
    )


if __name__ == "__main__":
    main()
