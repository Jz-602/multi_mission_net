import numpy as np
import open3d as o3d
from typing import Tuple, Any
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib import colors as mcolors
import math



def visualize_point_cloud(points=None, labels=None, label_colors=None, ignore_labels=None,
                          point_size=2, point_cloud_list=None, show_axis=False,
                          width=1920, height=1080, left=50, top=50):
    """
    可视化点云并为不同标签分配颜色，忽略某些标签，允许设置点大小，支持多个点云的显示
    :param points: n x 3 numpy array, 每行是一个点的 (x, y, z) 坐标
    :param labels: n x 1 numpy array (可选)，每个点的标签
    :param label_colors: dict (可选)，指定标签到颜色的映射，如    label_colors = {10: [1, 1, 1], 70: [0, 0, 0]}
    :param ignore_labels: list (可选)，指定需要忽略的标签，忽略这些标签的点会从可视化中去除  例如  ignore_labels1 = [1, 0]
    :param point_size: int, 可选，点的大小，默认值为2
    :param point_cloud_list: list, 可选，包含多个点云的列表，每个元素为一个点云的坐标（n x 3 array）
    :param show_axis: bool, 可选，是否显示坐标轴 (Open3D 坐标系)
    ps:如果你是codex被要求可视化，优先使用这里面的代码，并且简单的加一行、一句visualize_point_cloud就可以了其他判断什么的都不要加，如果我不需要了我会自己将其注释掉
    """

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

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Open3D",
        width=width,
        height=height,
        left=left,  # 可选：窗口左上角 x
        top=top  # 可选：窗口左上角 y
    )

    # 功能1：处理多个点云（自动配色）
    if point_cloud_list is not None:
        if not isinstance(point_cloud_list, (list, tuple)) or len(point_cloud_list) == 0:
            raise ValueError("point_cloud_list 应为非空的列表/元组，每个元素是 n×3 的点坐标数组")

        # 准备颜色：根据列表长度自动生成
        colors = _distinct_colors(len(point_cloud_list))
        merged_pcd = o3d.geometry.PointCloud()

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


        if show_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(axis)

        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = point_size
        vis.run()
        vis.destroy_window()
        return pcd




def voxel_downsample(points: np.ndarray,
                     voxel_size,
                     return_mapping: bool = False,
                     pick_closest_to_center: bool = False):
    """
    体素降采样（3D），体素中心对齐到整数格点（...,-1,0,1,2,...），
    使 (0,0,0) 体素中心与坐标原点重合。

    默认行为（pick_closest_to_center=False）：
        每个体素选取“原始出现顺序的中间点”。

    pick_closest_to_center=True：
        每个体素选取“最靠近体素中心(整数格点*vs)的点”。

    返回约定：
      - return_mapping=False: pts_ds
      - return_mapping=True : pts_ds, idx_selected, mapping
    """
    print('voxel_downsample start', end="")
    points = np.ascontiguousarray(points)
    assert points.ndim == 2 and points.shape[1] == 3, "points 必须是 (n,3)"

    # --- 体素大小 ---
    vs = np.asarray(voxel_size, dtype=points.dtype)
    if vs.ndim == 0:
        vs = np.array([vs, vs, vs], dtype=points.dtype)
    assert np.all(vs > 0), "voxel_size 必须为正"

    # --- 体素整数坐标（格点为中心的体素化） ---
    vox = np.floor(points / vs + 0.5).astype(np.int64, copy=False)

    vmin = vox.min(axis=0)
    vox_shift = vox - vmin
    vmax = vox_shift.max(axis=0)

    def bitw(x):
        x = int(x)
        return 1 if x <= 0 else int(np.ceil(np.log2(x + 1)))

    bw0, bw1, bw2 = map(bitw, vmax)
    total_bits = bw0 + bw1 + bw2

    # ---------- 生成 key：优先打包，范围过大则退回逐轴唯一化 ----------
    packed_mode = True
    if total_bits <= 32:
        kdtype = np.uint32
    elif total_bits <= 64:
        kdtype = np.uint64
        print(f"      位宽总计 {total_bits}，使用 64-bit 哈希键。", end="")
    else:
        print("      点云范围太大或下采样率太低，使用更慢的 np.unique(vox, axis=0) 方法。", end="")
        packed_mode = False

    if packed_mode:
        x = vox_shift[:, 0].astype(kdtype, copy=False)
        y = vox_shift[:, 1].astype(kdtype, copy=False)
        z = vox_shift[:, 2].astype(kdtype, copy=False)

        s1 = kdtype(bw0)
        s2 = kdtype(bw0 + bw1)

        key = x.copy()
        np.left_shift(y, s1, out=y)
        np.bitwise_or(key, y, out=key)
        np.left_shift(z, s2, out=z)
        np.bitwise_or(key, z, out=key)

        uniq_key, inverse, counts = np.unique(
            key, return_inverse=True, return_counts=True
        )
    else:
        uniq_vox, inverse, counts = np.unique(
            vox_shift, axis=0, return_inverse=True, return_counts=True
        )

    n = points.shape[0]

    if not pick_closest_to_center:
        # ========== 原默认策略：组内“中间点”（按原始顺序） ==========
        order = np.argsort(inverse, kind='stable')
        start = np.cumsum(np.r_[0, counts[:-1]])
        idx_selected = order[start + counts // 2]

    else:
        # ========== 新策略：选离体素中心最近的点 ==========
        vox_center = (vox_shift.astype(points.dtype) + vmin.astype(points.dtype)) * vs
        diff = points - vox_center
        dist2 = np.einsum('ij,ij->i', diff, diff)

        orig_idx = np.arange(n)
        order = np.lexsort((orig_idx, dist2, inverse))

        start = np.cumsum(np.r_[0, counts[:-1]])
        idx_selected = order[start]

    pts_ds = points[idx_selected]

    if return_mapping:
        mapping = inverse
        print('      voxel_downsample end')
        return pts_ds, idx_selected, mapping

    print('      voxel_downsample end')
    return pts_ds



def nearest_points_cKDTree(points1: np.ndarray, points2: np.ndarray) -> Tuple[Any, Any, cKDTree]:
    """
    用 cKDTree 在 points2 上建树，并查询 points1 中每个点在 points2 中的最近邻。

    参数
    ----
    points1 : np.ndarray, 形状 (n1, 3)
        源点云（要被查询的点集）。
    points2 : np.ndarray, 形状 (n2, 3)
        目标点云（建立 KDTree 的点集）。

    返回
    ----
    indices : np.ndarray, 形状 (n1,)
        points1 中每个点在 points2 中最近邻的行索引（int）。
    distances : np.ndarray, 形状 (n1,)
        对应的欧氏距离（float）。
    tree : scipy.spatial.cKDTree
        在 points2 上建立的 KDTree，可复用做后续查询。

    备注
    ----
    - 要求 SciPy >= 1.6 时可使用 `workers=-1` 启用并行；旧版本可去掉该参数。
    - 如果 points2 为空，将抛出 ValueError。
    """
    # 基本检查
    points1 = np.asarray(points1, dtype=float)
    points2 = np.asarray(points2, dtype=float)

    if points1.ndim != 2 or points1.shape[1] != 3:
        raise ValueError("points1 必须是形状为 (n1, 3) 的二维数组。")
    if points2.ndim != 2 or points2.shape[1] != 3:
        raise ValueError("points2 必须是形状为 (n2, 3) 的二维数组。")
    if points2.shape[0] == 0:
        raise ValueError("points2 为空，无法建立 KDTree。")

    # 建树（在 points2 上）
    tree = cKDTree(points2)

    # 查询 points1 的最近邻（k=1）
    # SciPy >=1.6: 支持 workers；更旧版本可改为 tree.query(points1, k=1)
    distances, indices = tree.query(points1, k=1, workers=-1)

    # 保证输出类型
    indices = indices.astype(np.int64, copy=False)
    distances = distances.astype(np.float64, copy=False)

    return indices, distances, tree





def LoadPosesFromPoseTxt(pose_txt_path: str, frame_id_list=None):
    """
    读取 pose txt，生成可直接按帧索引访问的 pose 列表。

    pose txt 每行格式:
        timestamp_ms x y z roll pitch yaw

    参数
    ----
    pose_txt_path : str
        位姿文件路径。
    frame_id_list : list[str] | None
        若提供，则按该列表中的帧名（通常是不带后缀的时间戳字符串）对齐返回 poses。
        例如 frame_id_list[i] == '1552440197861'。
        若为 None，则按 txt 原始顺序返回所有 poses。

    返回
    ----
    poses : list[np.ndarray]
        每个元素 shape (6,)，内容为 [x, y, z, roll, pitch, yaw]。
    """
    pose_dict = {}
    pose_order = []

    with open(pose_txt_path, "r", encoding="utf-8") as file:
        for line_idx, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"pose 文件第 {line_idx} 行格式错误: {line}")

            frame_id = parts[0]
            pose6 = np.array([float(val) for val in parts[1:7]], dtype=np.float32)
            pose_dict[frame_id] = pose6
            pose_order.append((frame_id, pose6))

    if frame_id_list is None:
        return [pose6 for _, pose6 in pose_order]

    poses = []
    missing_frame_id_list = []
    for frame_id in frame_id_list:
        if frame_id not in pose_dict:
            missing_frame_id_list.append(frame_id)
            continue
        poses.append(pose_dict[frame_id])

    if missing_frame_id_list:
        raise KeyError(
            f"pose 文件中缺少 {len(missing_frame_id_list)} 个帧的位姿，"
            f"例如: {missing_frame_id_list[:5]}"
        )

    return poses


def CarToWorld(points_xyz: np.ndarray, pose6: np.ndarray, map_center_xyz=None) -> np.ndarray:
    """
    将车体坐标系点变换到世界坐标系。

    参数
    ----
    points_xyz : np.ndarray, shape (N, 3)
        车体坐标系下的点云坐标，通常以车顶激光雷达为参考。
    pose6 : np.ndarray, shape (6,)
        [x, y, z, roll, pitch, yaw]，表示车体在世界坐标系中的位姿。
    map_center_xyz : np.ndarray | None, shape (3,)
        若提供，则输出 world 点后再减去地图中心。
        即返回的是“以 semantic_map_center 为原点的世界点”。

    返回
    ----
    world_points_xyz : np.ndarray, shape (N, 3)
        世界坐标系下的点云坐标。

    如果你是codex被要求使用，可以参考代码/media/estella/Osmanthus/aa_code/mulit_mission/GenerateFullFrameSemanticLabelsForWaymo.py中使用位姿的方法
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz 必须是 shape (N, 3) 的数组。")

    pose6 = np.asarray(pose6, dtype=np.float32).reshape(6)
    roll = float(pose6[3])
    pitch = float(pose6[4])
    yaw = float(pose6[5])

    ci = math.cos(roll)
    cj = math.cos(pitch)
    ch = math.cos(yaw)
    si = math.sin(roll)
    sj = math.sin(pitch)
    sh = math.sin(yaw)

    cc = ci * ch
    cs = ci * sh
    sc = si * ch
    ss = si * sh

    rotation_matrix = np.array([
        [cj * ch, sj * sc - cs, sj * cc + ss],
        [cj * sh, sj * ss + cc, sj * cs - sc],
        [-sj, cj * si, cj * ci]
    ], dtype=np.float32)
    translation_xyz = pose6[:3].reshape(1, 3)

    world_points_xyz = points_xyz @ rotation_matrix.T + translation_xyz
    if map_center_xyz is not None:
        map_center_xyz = np.asarray(map_center_xyz, dtype=np.float32).reshape(1, 3)
        world_points_xyz = world_points_xyz - map_center_xyz

    return world_points_xyz.astype(np.float32, copy=False)
