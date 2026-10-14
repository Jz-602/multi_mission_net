import os

import numpy as np

from gt_generate.common.constants import INVALID_HEIGHT


def _SanitizeQtPluginEnvironment():
    for env_name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        env_value = os.environ.get(env_name, "")
        if "cv2" in env_value:
            os.environ.pop(env_name, None)


def _RequireOpen3d():
    _SanitizeQtPluginEnvironment()
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("Open3D is not installed. Install open3d before using interactive viewers.") from exc
    return o3d


def _BuildUniformPointCloud(points, color_rgb):
    o3d = _RequireOpen3d()
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.shape[0] == 0:
        return None
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.asarray(color_rgb, dtype=np.float64), (points.shape[0], 1))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _UpdatePointCloudGeometry(cloud, points, colors):
    o3d = _RequireOpen3d()
    points = np.ascontiguousarray(points, dtype=np.float64)
    colors = np.ascontiguousarray(colors, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid point array shape: {points.shape}")
    if colors.ndim != 2 or colors.shape != points.shape:
        raise ValueError(f"Invalid color array shape: {colors.shape}, expected {points.shape}")
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def _CreateVisualizerWindow(window_name, geometries, left, top, width, height, point_size=2.0):
    o3d = _RequireOpen3d()
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=window_name, width=width, height=height, left=left, top=top, visible=True)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray((0.04, 0.04, 0.04), dtype=np.float64)
    render_option.point_size = float(point_size)
    for geometry in geometries:
        if geometry is not None:
            visualizer.add_geometry(geometry)
    return visualizer


def _RunVisualizerGrid(window_specs, width=720, height=540):
    visualizers = []
    try:
        for idx, (window_name, geometries) in enumerate(window_specs):
            row_idx = idx // 2
            col_idx = idx % 2
            visualizers.append(
                _CreateVisualizerWindow(
                    window_name=window_name,
                    geometries=geometries,
                    left=40 + col_idx * (width + 24),
                    top=40 + row_idx * (height + 60),
                    width=width,
                    height=height,
                )
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
