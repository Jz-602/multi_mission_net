
import os
import math
import numpy as np
import open3d as o3d


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

def remap_labels(labels, mapping_dict):
    """
    使用查找表(Lookup Table)快速映射标签
    :param labels: (N,) 的 numpy 数组
    :param mapping_dict: 原始标签到新标签的字典
    :return: 映射后的 labels
    """
    labels = np.asarray(labels)

    # 1. 确定查找表的大小 (最大键值 + 1)
    max_key = max(mapping_dict.keys())

    # 2. 创建查找表 (Lookup Table)
    # 初始化为全 0 (或者你可以初始化为 np.arange(max_key + 1) 以保持未定义的标签不变)
    lut = np.zeros(max_key + 1, dtype=labels.dtype)

    # 3. 填充映射规则
    for src, dst in mapping_dict.items():
        lut[src] = dst

    # 4. 执行映射
    # 注意：为了防止 labels 中有超出字典范围的异常值导致索引报错，可以加个掩码
    # 但 Waymo 数据通常严格在 0-22 之间，直接映射是最快的
    try:
        new_labels = lut[labels]
    except IndexError:
        print("警告: 存在超出映射表范围的标签，将保留原值")
        mask = labels <= max_key
        new_labels = labels.copy()
        new_labels[mask] = lut[labels[mask]]

    return new_labels


def expand_dynamic_labels(points, dense_labels, map_dynamic, search_radius=0.5, src_label=1, target_label=6):
    """
    如果一个点同时满足 map_dynamic=1 和 dense_labels=src_label (种子点)，
    则将其周围 search_radius 范围内的、且 dense_labels 也是 src_label 的点，
    全部修改为 target_label。
    """
    from scipy.spatial import cKDTree
    import time

    t0 = time.time()

    # 1. 确定种子点 (Seeds)：既是动态(dynamic=1) 又是 车辆(dense_labels=1)
    # 注意：map_dynamic 通常是 0 或 1，dense_labels 此时应该是映射后的值
    seed_mask = (map_dynamic == 1) & (dense_labels == src_label)
    seed_indices = np.where(seed_mask)[0]
    seed_points = points[seed_mask]

    if len(seed_points) == 0:
        print("未找到满足条件的动态种子点，跳过扩散操作。")
        return dense_labels

    print(f"找到 {len(seed_points)} 个动态车辆种子点，正在搜索半径 {search_radius}m 内的邻居...")

    # 2. 构建 KDTree (对所有点构建，以便查找全局邻居)
    # 如果之前有构建过且点没变，可以复用，但这里为了独立性重新构建
    tree = cKDTree(points)

    # 3. 半径搜索 (Query Ball Point)
    # 返回的是一个列表的列表 (list of lists)，每个元素是对应种子点的邻居索引
    neighbors_list = tree.query_ball_point(seed_points, r=search_radius, workers=-1)

    # 4. 展平并去重索引
    # 将所有邻居索引合并成一个大数组
    all_neighbor_indices = np.concatenate(neighbors_list)
    unique_neighbor_indices = np.unique(all_neighbor_indices)

    # 5. 过滤条件：被修改的点本身必须是 src_label (车辆)
    # 我们只修改那些原本被标记为“车辆(1)”的点，不修改地面或其他物体
    candidate_labels = dense_labels[unique_neighbor_indices]
    valid_mask = (candidate_labels == src_label)

    final_indices_to_change = unique_neighbor_indices[valid_mask]

    # 6. 执行修改
    dense_labels[final_indices_to_change] = target_label

    t1 = time.time()
    print(f"扩散完成 (耗时 {t1 - t0:.3f}s): 将 {len(final_indices_to_change)} 个点的标签修正为 {target_label}。")

    return dense_labels

def car2world(points_xyz: np.ndarray, pose6: np.ndarray):
    """
    将点云从车坐标系转换到世界坐标系
    points_xyz: (N, 3) 的点云坐标
    pose6: [x, y, z, roll, pitch, yaw]，其中 x, y, z 是位置，roll, pitch, yaw 是旋转角度
    """
    # 定义 Yaw-Pitch-Roll 到旋转矩阵的转换
    def YPR2RotationMatrixV2(ypr):
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

    # 定义点云旋转的函数
    def rotate_points_by_pose(points_xyz: np.ndarray, pose6: np.ndarray):
        roll  = float(pose6[3])
        pitch = float(pose6[4])
        yaw   = float(pose6[5])

        R = YPR2RotationMatrixV2([yaw, pitch, roll])  # [yaw, pitch, roll]
        points_rot = (R @ points_xyz.T).T
        return points_rot, R

    # 获取位姿的旋转部分 (roll, pitch, yaw) 和位置部分 (x, y, z)
    position = pose6[:3]  # 位姿中的位置部分 [x, y, z]
    rotation = pose6[3:]  # 位姿中的旋转部分 [roll, pitch, yaw]

    # 进行旋转
    points_rot, R = rotate_points_by_pose(points_xyz, pose6)

    # 进行平移：将旋转后的点云平移到指定位置
    points_world = points_rot + position  # 对所有点进行平移

    return points_world



from scipy.spatial import cKDTree
def transfer_labels_knn(map_point, map_label, points_np, max_dist=0.2, default_label=0):
    """
    map_point: (M,3) 稀疏点云坐标
    map_label: (M,)  稀疏点云标签
    points_np: (N,3) 致密点云坐标
    max_dist:  最近邻距离阈值，超过则置 default_label
    """
    map_point = np.asarray(map_point, dtype=np.float32).reshape(-1, 3)
    map_label = np.asarray(map_label).reshape(-1)
    points_np = np.asarray(points_np, dtype=np.float32).reshape(-1, 3)

    if map_point.shape[0] != map_label.shape[0]:
        raise ValueError(f"map_point 行数 {map_point.shape[0]} != map_label 长度 {map_label.shape[0]}")

    tree = cKDTree(map_point)
    dists, idxs = tree.query(points_np, k=1, workers=-1)  # workers=-1 用满核

    out_labels = np.full(points_np.shape[0], default_label, dtype=map_label.dtype)
    ok = dists <= max_dist
    out_labels[ok] = map_label[idxs[ok]]
    return out_labels, dists, idxs



path = "/media/gk/data/lyf/aa_data/FH_waymo/1005081002024129653_5313_150_5333_150/output/SemanticMap/semantic_map_center.txt"
center = np.loadtxt(path, dtype=np.float64)

# 制造稠密点云地图

# root_path 和 pose_file_path 已经设置好
root_path = '/media/gk/data/lyf/aa_data/FH_waymo/1005081002024129653_5313_150_5333_150/output/'
pcd_sequence = [f for f in os.listdir(root_path + 'pcd') if f.endswith('.pcd')]
pcd_sequence_sorted = sorted(pcd_sequence)

pose_file_path = root_path + '/pose_slam_euler.txt'
pose_list = []
with open(pose_file_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        data = line.strip().split()  # 按空格拆分每一行的数据
        timestamp = float(data[0])  # 时间戳
        pose_data = np.array([float(x) for x in data[1:7]], dtype=np.float32)  # 位姿数据
        pose_list.append(pose_data)

all_points = []

# 逐帧读取点云数据并进行坐标转换
for i in range(len(pcd_sequence)):
    pcd_path = root_path + 'pcd/' + pcd_sequence_sorted[i]
    pcd = o3d.t.io.read_point_cloud(pcd_path)
    point = pcd.point["positions"].numpy().astype(np.float32)

    # 坐标变换
    point_world = car2world(point, pose_list[i])

    # 添加到总点集合
    all_points.append(point_world)

points_np = np.vstack(all_points)
points_np -= center






# 读取动态信息
bin_file_path = "/media/gk/data/lyf/aa_data/FH_waymo/1005081002024129653_5313_150_5333_150/output/map_cleaner/all_frame_label_counter.bin"
# 读取二进制文件
def read_bin_file(file_path):
    # 打开文件进行二进制读取
    with open(file_path, 'rb') as file:
        # 使用 numpy 读取 UINT16 数据
        data = np.fromfile(file, dtype=np.uint16)
    return data

label_counter = read_bin_file(bin_file_path)
label_counter_reshaped = label_counter.reshape(-1, 2)

pcd_object_path = f'/media/gk/data/lyf/aa_data/FH_waymo/1005081002024129653_5313_150_5333_150/output/map_cleaner/all_frame_obstacle_point_cloud_map.pcd'
pcd_map_object = o3d.t.io.read_point_cloud(pcd_object_path)
pcd_map_object_point = pcd_map_object.point["positions"].numpy().astype(np.float32)
object_dynamic_label = (label_counter_reshaped[:, 1] > 9).astype(np.uint16)
# visualize_point_cloud(pcd_map_object_point, object_dynamic_label)



# visualize_point_cloud(point_cloud_list=[points_np, pcd_map_object_point])


map_dynamic, nn_dists, nn_idxs = transfer_labels_knn(
    map_point=pcd_map_object_point,
    map_label=object_dynamic_label,
    points_np=points_np,
    max_dist=0.15,
    default_label=0
)

# visualize_point_cloud(points_np, map_dynamic)




# 读取语义地图


path5 = '/media/gk/data/lyf/aa_data/FH_waymo/1005081002024129653_5313_150_5333_150/output/SemanticMap/semantic_map_point_cloud.pcd'
pcd_map = o3d.t.io.read_point_cloud(path5)
map_point = pcd_map.point["positions"].numpy().astype(np.float32)
map_label = pcd_map.point["intensity"].numpy().astype(np.int32)
visualize_point_cloud(map_point, map_label, show_axis=True)






# 复制语义信息

dense_labels, nn_dists, nn_idxs = transfer_labels_knn(
    map_point=map_point,
    map_label=map_label,
    points_np=points_np,
    max_dist=0.2,
    default_label=0
)

# visualize_point_cloud(points_np, dense_labels)





points_np, flag, mapping = voxel_downsample(points_np, 0.1, True)
map_dynamic = map_dynamic[flag]
dense_labels = dense_labels[flag]




# 定义你的映射关系
mapping_dict = {
    0: 0,  # Undefined -> Undefined
    1: 1, 2: 1, 3: 1, 4: 1,  # Vehicles -> 1
    5: 7, 6: 7, 7: 7, 12: 7, 13: 7,  # Pedestrian/Cyclist/Motorcyclist -> 7 (VRU)
    8: 2, 9: 2, 10: 2, 16: 2,  # Pole/Sign/Traffic Light/Tree Trunk -> 2
    11: 0, 17: 0,  # Cone/Curb -> 0 (Ignore)
    14: 3,  # Building -> 3
    15: 4,  # Vegetation -> 4
    18: 5, 19: 5, 20: 5, 21: 5, 22: 5  # Road/Ground/Sidewalk -> 5
}
dense_labels = remap_labels(dense_labels, mapping_dict)

# dense_labels = expand_dynamic_labels(
#     points=points_np,
#     dense_labels=dense_labels,
#     map_dynamic=map_dynamic,
#     search_radius=0.5,
#     src_label=1,    # 原始车辆标签
#     target_label=6  # 目标标签 (动态车辆)
# )


visualize_point_cloud(points_np, dense_labels,ignore_labels=[0])
visualize_point_cloud(points_np, map_dynamic)

# 保存结果
pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(points_np.astype(np.float32))
output_pcd_path = '/media/estella/Osmanthus/aa_data/zz_others/waymo_map_points.pcd'
o3d.io.write_point_cloud(output_pcd_path, pcd_out)
print(f"点云保存成功：{output_pcd_path}")

output_dynamic_path = '/media/estella/Osmanthus/aa_data/zz_others/dynamic_labels.label'
map_dynamic.astype(np.uint16).tofile(output_dynamic_path)
print(f"map_dynamic 保存成功：{output_dynamic_path}")

output_semantic_path = '/media/estella/Osmanthus/aa_data/zz_others/semantic_labels.label'
dense_labels.astype(np.uint16).tofile(output_semantic_path)
print(f"dense_labels 保存成功：{output_semantic_path}")


# point_original_rot, R = rotate_points_by_pose(point_original, pose)