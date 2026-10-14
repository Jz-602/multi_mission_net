import math
import numpy as np
import open3d as o3d

def YPR2RotationMatrixV2(ypr):
    """
    ypr: [yaw, pitch, roll]  (Z, Y, X)
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    eulerZ = ypr[0]  # yaw
    eulerY = ypr[1]  # pitch
    eulerX = ypr[2]  # roll

    ci = math.cos(eulerX)
    cj = math.cos(eulerY)
    ch = math.cos(eulerZ)
    si = math.sin(eulerX)
    sj = math.sin(eulerY)
    sh = math.sin(eulerZ)

    cc = ci * ch
    cs = ci * sh
    sc = si * ch
    ss = si * sh

    R = np.array([
        [cj * ch,  sj * sc - cs,  sj * cc + ss],
        [cj * sh,  sj * ss + cc,  sj * cs - sc],
        [-sj,      cj * si,       cj * ci]
    ], dtype=np.float32)

    return R

def rotate_points_by_pose(points_xyz: np.ndarray, pose6: np.ndarray):
    """
    points_xyz: (N,3)
    pose6: [x, y, z, roll, pitch, yaw]
    注意：这里按你给的代码推断 pose[3]=roll pose[4]=pitch pose[5]=yaw
    并且只做旋转，不做平移（与你 point_process 情况一致）
    """
    roll  = float(pose6[3])
    pitch = float(pose6[4])
    yaw   = float(pose6[5])

    R = YPR2RotationMatrixV2([yaw, pitch, roll])  # [yaw, pitch, roll]
    points_rot = (R @ points_xyz.T).T
    return points_rot, R

def visualize_point_cloud(points=None, labels=None, label_colors=None, ignore_labels=None,
                          point_size=2, point_cloud_list=None, show_axis=False):
    """
    可视化点云并为不同标签分配颜色，忽略某些标签，允许设置点大小，支持多个点云的显示
    :param points: n x 3 numpy array, 每行是一个点的 (x, y, z) 坐标
    :param labels: n x 1 numpy array (可选)，每个点的标签
    :param label_colors: dict (可选)，指定标签到颜色的映射，如    label_colors = {10: [1, 1, 1], 70: [0, 0, 0]}
    :param ignore_labels: list (可选)，指定需要忽略的标签，忽略这些标签的点会从可视化中去除  例如  ignore_labels1 = [1, 0]
    :param point_size: int, 可选，点的大小，默认值为2
    :param point_cloud_list: list, 可选，包含多个点云的列表，每个元素为一个点云的坐标（n x 3 array）
    :param show_axis: bool, 可选，是否显示坐标轴 (Open3D 坐标系)
    """

    from matplotlib import colors as mcolors
    import matplotlib.pyplot as plt
    import open3d as o3d
    import numpy as np

    def _distinct_colors(n: int) -> np.ndarray:
        """
        返回 n×3 的 RGB 颜色（范围 0~1）。
        n <= 20 时用 tab20，>20 时用等分 HSV，任意 n 都能稳定生成。
        """
        if n <= 0:
            return np.empty((0, 3), dtype=float)
        if n < 5:
            base = np.array([
                [0.1216, 0.4667, 0.7059],  # 蓝  #1f77b4
                [0.8392, 0.1529, 0.1569],  # 红  #d62728
                [0.1725, 0.6275, 0.1725],  # 绿  #2ca02c
                [0.5804, 0.4039, 0.7412],  # 紫  #9467bd
            ], dtype=float)
            return base[:n]
        if n <= 10:
            cmap = plt.cm.get_cmap('tab10', n)
            return cmap(np.arange(n))[:, :3]
        if n <= 20:
            cmap = plt.cm.get_cmap('tab20', n)
            return cmap(np.arange(n))[:, :3]
        h = np.linspace(0, 1, n, endpoint=False)
        s = np.full_like(h, 0.65)
        v = np.full_like(h, 0.95)
        hsv = np.stack([h, s, v], axis=1)
        rgb = mcolors.hsv_to_rgb(hsv)
        return rgb

    if show_axis:
        print("红色（X）   绿色（Y）   蓝色（Z）")

    if label_colors is not None:
        # 收集所有颜色值并转成 float 数组
        all_vals = np.array(list(label_colors.values()), dtype=float).reshape(-1)
        if all_vals.size > 0 and all_vals.max() > 1.0:
            # 说明像是 0~255 的颜色，统一除以 255
            normalized = {}
            for k, v in label_colors.items():
                arr = np.asarray(v, dtype=float) / 255.0
                # clip 一下，避免越界
                arr = np.clip(arr, 0.0, 1.0)
                normalized[k] = arr.tolist()
            label_colors = normalized
            print("检测到 label_colors 使用 0~255 颜色值，已自动归一化为 0~1。")
    # ----------------------------------------------------------------------

    # 确保points和labels是正确的形状
    if points is not None:
        points = np.asarray(points)
        if points.shape[1] != 3:
            raise ValueError("points应该是n x 3的数组，表示每个点的(x, y, z)坐标")
        points = points.reshape(-1, 3)  # 确保是n x 3的形状

    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] != points.shape[0]:
            raise ValueError("labels的数量应该与points的数量一致")
        labels = labels.reshape(-1)  # 确保是n x 1的形状

    # 功能1：处理多个点云（自动配色）
    if point_cloud_list is not None:
        if not isinstance(point_cloud_list, (list, tuple)) or len(point_cloud_list) == 0:
            raise ValueError("point_cloud_list 应为非空的列表/元组，每个元素是 n×3 的点坐标数组")

        # 准备颜色：根据列表长度自动生成
        colors = _distinct_colors(len(point_cloud_list))
        merged_pcd = o3d.geometry.PointCloud()

        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # 是否显示坐标轴
        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis)

        for i, pts in enumerate(point_cloud_list):
            pts = np.asarray(pts, dtype=float).reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            clr = np.tile(colors[i], (pts.shape[0], 1))
            pcd.colors = o3d.utility.Vector3dVector(clr)
            vis.add_geometry(pcd)
            merged_pcd += pcd

        opt = vis.get_render_option()
        opt.point_size = point_size  # 设置点大小
        vis.run()
        vis.destroy_window()
        return merged_pcd

    # 功能2：只输入点云坐标
    if labels is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis)

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        vis.run()
        vis.destroy_window()
        return pcd

    # 功能3：输入点云坐标和点云标签，随机分配颜色
    if label_colors is None and ignore_labels is None:
        unique_labels = np.unique(labels)
        point_colors = np.ones((len(points), 3))
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))[:, :3]

        print("功能2: 随机为标签分配颜色:")
        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            point_colors[mask] = colors[i]
            label_count = np.sum(mask)
            print(f"标签 {label} 被分配了颜色 {colors[i]}，共 {label_count} 个点")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis)

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        vis.run()
        vis.destroy_window()
        return pcd

    # 功能4：输入点云坐标和标签，特定标签指定颜色
    if label_colors is not None and ignore_labels is None:
        point_colors = np.ones((len(points), 3))
        print("功能3: 对特定标签指定颜色:")
        for label, color in label_colors.items():
            mask = (labels == label)
            point_colors[mask] = color
            label_count = np.sum(mask)
            print(f"标签 {label} 被指定了颜色 {color}，共 {label_count} 个点")

        remaining_labels = [label for label in np.unique(labels) if label not in label_colors]
        if remaining_labels:
            colors = plt.cm.jet(np.linspace(0, 1, len(remaining_labels)))[:, :3]
            for i, label in enumerate(remaining_labels):
                mask = (labels == label)
                point_colors[mask] = colors[i]
                label_count = np.sum(mask)
                print(f"标签 {label} 被分配了颜色 {colors[i]}，共 {label_count} 个点")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis)

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        vis.run()
        vis.destroy_window()
        return pcd

    # 功能5：输入点云坐标和标签，忽略特定标签
    if ignore_labels is not None and label_colors is None:
        mask = ~np.isin(labels, ignore_labels)
        points = points[mask]
        labels = labels[mask]

        print("功能4: 忽略特定标签:")
        print(f"忽略的标签: {ignore_labels}")

        unique_labels = np.unique(labels)
        point_colors = np.ones((len(points), 3))
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))[:, :3]

        for i, label in enumerate(unique_labels):
            mask = (labels == label)
            point_colors[mask] = colors[i]
            label_count = np.sum(mask)
            print(f"标签 {label} 被分配了颜色 {colors[i]}，共 {label_count} 个点")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis)

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        vis.run()
        vis.destroy_window()
        return pcd

    # 功能6：输入点云坐标和标签，特定标签指定颜色，忽略特定标签
    if ignore_labels is not None and label_colors is not None:
        mask = ~np.isin(labels, ignore_labels)
        points = points[mask]
        labels = labels[mask]

        print("功能5: 忽略特定标签并指定颜色:")
        print(f"忽略的标签: {ignore_labels}")

        point_colors = np.ones((len(points), 3))
        for label, color in label_colors.items():
            mask = (labels == label)
            point_colors[mask] = color
            label_count = np.sum(mask)
            print(f"标签 {label} 被指定了颜色 {color}，共 {label_count} 个点")

        remaining_labels = [label for label in np.unique(labels) if label not in label_colors]
        if remaining_labels:
            colors = plt.cm.jet(np.linspace(0, 1, len(remaining_labels)))[:, :3]
            for i, label in enumerate(remaining_labels):
                mask = (labels == label)
                point_colors[mask] = colors[i]
                label_count = np.sum(mask)
                print(f"标签 {label} 被分配了颜色 {colors[i]}，共 {label_count} 个点")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis)

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        vis.run()
        vis.destroy_window()
        return pcd



def YPR2RotationMatrixV2(ypr):
    """
    ypr: [yaw, pitch, roll]  (Z, Y, X)
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    eulerZ = ypr[0]  # yaw
    eulerY = ypr[1]  # pitch
    eulerX = ypr[2]  # roll

    ci = math.cos(eulerX)
    cj = math.cos(eulerY)
    ch = math.cos(eulerZ)
    si = math.sin(eulerX)
    sj = math.sin(eulerY)
    sh = math.sin(eulerZ)

    cc = ci * ch
    cs = ci * sh
    sc = si * ch
    ss = si * sh

    R = np.array([
        [cj * ch,  sj * sc - cs,  sj * cc + ss],
        [cj * sh,  sj * ss + cc,  sj * cs - sc],
        [-sj,      cj * si,       cj * ci]
    ], dtype=np.float32)

    return R

def build_R_from_pose(pose6):
    """
    pose6: [x, y, z, roll, pitch, yaw]
    按你给的 C++ 调用：YPR2RotationMatrixV2([yaw, pitch, roll])
    """
    roll  = float(pose6[3])
    pitch = float(pose6[4])
    yaw   = float(pose6[5])
    R = YPR2RotationMatrixV2([yaw, pitch, roll])
    return R

def rotate_forward(points_xyz: np.ndarray, pose6: np.ndarray):
    """把点按 pose 的姿态旋转（不平移）"""
    R = build_R_from_pose(pose6)
    return (R @ points_xyz.T).T, R

def rotate_back(points_xyz: np.ndarray, pose6: np.ndarray):
    """
    把“已经按 pose 旋转过的点”旋转回去（不平移）
    等价于乘 R^{-1} = R.T
    """
    R = build_R_from_pose(pose6)
    R_inv = R.T
    return (R_inv @ points_xyz.T).T, R_inv

pose = np.array([656.747833, 6637.501920, 93.085841, 0.007578, -0.011641, -2.596655], dtype=np.float32)

pcd_path_original = f'/media/gk/data/lyf/aa_data/FH_waymo/14619874262915043759_2801_090_2821_090/output/pcd/1510844671210.pcd'
pcd_original = o3d.t.io.read_point_cloud(pcd_path_original)
point_original = pcd_original.point["positions"].numpy().astype(np.float32)
point_original_rot, R = rotate_points_by_pose(point_original, pose)

pcd_path_process = f'/media/gk/data/lyf/aa_data/FH_waymo/14619874262915043759_2801_090_2821_090/output/Occ3D/1510844671210.pcd'
pcd_process = o3d.t.io.read_point_cloud(pcd_path_process)
point_process = pcd_process.point["positions"].numpy().astype(np.float32)
point_label = pcd_process.point["intensity"].numpy().astype(np.int32)
point_process_back, R_inv = rotate_back(point_process, pose)

visualize_point_cloud(point_process, point_label)

path0 = '/media/gk/data/lyf/aa_data/FH_waymo/15832924468527961_1564_160_1584_160/output/pcd/1507678826876.pcd'
path1 = '/media/gk/data/lyf/aa_data/FH_waymo/15832924468527961_1564_160_1584_160/output/SemanticMap/allframe_non_terrain_point_cloud.pcd'# 只有静态地面上的地图
path2 = '/media/gk/data/lyf/aa_data/FH_waymo/15832924468527961_1564_160_1584_160/output/SemanticMap/allframe_terrain_point_cloud.pcd'   # 地面点
path3 = '/media/gk/data/lyf/aa_data/FH_waymo/15832924468527961_1564_160_1584_160/output/SemanticMap/all_stuff_map_point_cloud.pcd'    # 静态地图无标签
path4 = '/media/gk/data/lyf/aa_data/FH_waymo/15832924468527961_1564_160_1584_160/output/SemanticMap/keyframe_object_map_point_cloud.pcd'
path5 = '/media/gk/data/lyf/aa_data/FH_waymo/15832924468527961_1564_160_1584_160/output/SemanticMap/semantic_map_point_cloud.pcd'
path6 = '/media/gk/data/lyf/aa_data/FH_waymo/15832924468527961_1564_160_1584_160/output/SemanticMap/static_semantic_map_point_cloud.pcd'
path7 = '/media/gk/data/lyf/aa_data/FH_waymo/15832924468527961_1564_160_1584_160/output/SemanticMap/stuff_map_point_cloud.pcd'


path_list = [path0, path1, path2, path3, path4, path5, path6, path7]

# for pcd_map_path in path_list:
#     print(pcd_map_path)
#     pcd_map = o3d.t.io.read_point_cloud(pcd_map_path)
#     map_point = pcd_map.point["positions"].numpy().astype(np.float32)
#
#     if "intensity" in pcd_map.point:
#         map_label = pcd_map.point["intensity"].numpy().astype(np.int32)
#         visualize_point_cloud(map_point, map_label, show_axis=True)
#     else:
#         visualize_point_cloud(map_point)

visualize_point_cloud(point_cloud_list=[point_original,point_process_back ],show_axis=True)
pass