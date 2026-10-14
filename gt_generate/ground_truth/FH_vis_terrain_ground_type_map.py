import os
import sys
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.io_utils import ReadMat, UpdateNamespaceFromCli
from gt_generate.ground_truth.data_paths import DEFAULT_FH_WAYMO_ROOT, DEFAULT_FH_WAYMO_SEQUENCE_NAME, BuildDefaultTerrainDir, ResolveTerrainOnlyDataPaths
from gt_generate.ground_truth.debug_view import BuildTerrainMapPoints
from gt_generate.common.open3d_debug_view import _RequireOpen3d


TERRAIN_GROUND_UNKNOWN = 0
TERRAIN_GROUND_ROAD = 1
TERRAIN_GROUND_ROAD2 = 2


GROUND_TYPE_COLOR_DICT = {
    TERRAIN_GROUND_UNKNOWN: (0.45, 0.45, 0.45),
    TERRAIN_GROUND_ROAD: (0.95, 0.00, 0.95),
    TERRAIN_GROUND_ROAD2: (1.00, 0.58, 0.95),
}


def BuildArgs():
    return Namespace(
        data_path=DEFAULT_FH_WAYMO_ROOT,
        sequence_name=DEFAULT_FH_WAYMO_SEQUENCE_NAME,
        pcd_dir="",
        semantic_label_dir="",
        pose_filename="",
        # terrain-only 输出目录，要求包含 terrain_z_inpainted.bin 和 terrain_ground_type_map.bin。
        terrain_dir="",
        # 显示 raw 直接观测标签图还是最终传播后的标签图；可选 raw/final/both。
        mode="both",
        # 点大小。
        point_size=2.0,
    )


def BuildGroundTypeCloud(o3d, terrain_map, ground_type_map):
    valid_mask = np.isfinite(terrain_map) & (terrain_map != INVALID_HEIGHT)
    points, rows, cols = BuildTerrainMapPoints(terrain_map, valid_mask=valid_mask)
    if points.shape[0] == 0:
        return o3d.geometry.PointCloud(), {}
    labels = ground_type_map[rows, cols].astype(np.uint8)
    colors = np.zeros((points.shape[0], 3), dtype=np.float64)
    for label in np.unique(labels):
        colors[labels == label] = np.asarray(GROUND_TYPE_COLOR_DICT.get(int(label), GROUND_TYPE_COLOR_DICT[TERRAIN_GROUND_UNKNOWN]), dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors, dtype=np.float64))
    counts = {int(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))}
    return cloud, counts


def BuildLegendGeometry(o3d, x0, y0, z0):
    points = np.asarray(
        [
            [x0, y0, z0],
            [x0 + 2.0, y0, z0],
            [x0 + 4.0, y0, z0],
        ],
        dtype=np.float64,
    )
    colors = np.asarray(
        [
            GROUND_TYPE_COLOR_DICT[TERRAIN_GROUND_UNKNOWN],
            GROUND_TYPE_COLOR_DICT[TERRAIN_GROUND_ROAD],
            GROUND_TYPE_COLOR_DICT[TERRAIN_GROUND_ROAD2],
        ],
        dtype=np.float64,
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def ShowGroundTypeMaps(args):
    terrain_map = ReadMat(os.path.join(args.terrain_dir, "terrain_z_inpainted.bin")).astype(np.float32)
    items = []
    if str(args.mode) in ("raw", "both"):
        raw_filename = os.path.join(args.terrain_dir, "terrain_ground_type_raw_map.bin")
        if os.path.exists(raw_filename):
            items.append(("raw", ReadMat(raw_filename).astype(np.uint8)))
        else:
            print(f"[TerrainGroundTypeVis] missing: {raw_filename}")
    if str(args.mode) in ("final", "both"):
        final_filename = os.path.join(args.terrain_dir, "terrain_ground_type_map.bin")
        if os.path.exists(final_filename):
            items.append(("final", ReadMat(final_filename).astype(np.uint8)))
        else:
            print(f"[TerrainGroundTypeVis] missing: {final_filename}")
    if not items:
        raise FileNotFoundError("No terrain ground type map found. Re-run build_waymo_terrain_only.py first.")

    o3d = _RequireOpen3d()
    visualizers = []
    try:
        for item_idx, (name, ground_type_map) in enumerate(items):
            cloud, counts = BuildGroundTypeCloud(o3d, terrain_map, ground_type_map)
            valid_points = np.asarray(cloud.points)
            if valid_points.shape[0] > 0:
                min_xyz = np.min(valid_points, axis=0)
                legend_cloud = BuildLegendGeometry(o3d, float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2]) + 2.0)
            else:
                legend_cloud = o3d.geometry.PointCloud()
            visualizer = o3d.visualization.Visualizer()
            visualizer.create_window(
                window_name=f"TerrainGroundType-{name} gray:unknown magenta:road pink:road2",
                width=960,
                height=760,
                left=40 + item_idx * 980,
                top=40,
                visible=True,
            )
            render_option = visualizer.get_render_option()
            render_option.background_color = np.asarray((0.04, 0.04, 0.04), dtype=np.float64)
            render_option.point_size = float(args.point_size)
            visualizer.add_geometry(cloud)
            visualizer.add_geometry(legend_cloud)
            visualizer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0))
            visualizers.append(visualizer)
            print(
                "[TerrainGroundTypeVis] "
                f"{name} unknown={counts.get(TERRAIN_GROUND_UNKNOWN, 0)} "
                f"road={counts.get(TERRAIN_GROUND_ROAD, 0)} "
                f"road2={counts.get(TERRAIN_GROUND_ROAD2, 0)}"
            )

        while True:
            event_flags = []
            for visualizer in visualizers:
                event_flags.append(bool(visualizer.poll_events()))
                visualizer.update_renderer()
            if not all(event_flags):
                break
    finally:
        for visualizer in visualizers:
            visualizer.destroy_window()


def Main():
    args = BuildArgs()
    UpdateNamespaceFromCli(args, sys.argv)
    if not str(args.terrain_dir).strip():
        ResolveTerrainOnlyDataPaths(args, b_require_semantic=False, b_require_pose=False)
        args.terrain_dir = BuildDefaultTerrainDir(args)
    ShowGroundTypeMaps(args)


if __name__ == "__main__":
    Main()
