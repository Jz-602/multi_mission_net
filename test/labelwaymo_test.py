import os
import pickle

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from lyf_tools import CarToWorld
from lyf_tools import LoadPosesFromPoseTxt
from lyf_tools import visualize_point_cloud
from lyf_tools import voxel_downsample


'''
ID	英文类别	            中文类别
0	UNDEFINED	        未定义

1	CAR	                小汽车
2	TRUCK	            卡车
3	BUS	                公交车
4	OTHER_VEHICLE	    其他车辆

5	MOTORCYCLIST	    摩托车骑行者
6	BICYCLIST	        自行车骑行者

7	PEDESTRIAN	        行人

8	SIGN	            交通标志
9	TRAFFIC_LIGHT	    交通信号灯

10	POLE	            杆（路灯杆/电线杆等）
11	CONSTRUCTION_CONE	施工锥桶

12	BICYCLE	            自行车
13	MOTORCYCLE	        摩托车

14	BUILDING	        建筑物
15	VEGETATION	        植被
16	TREE_TRUNK	        树干

17	CURB	            路缘石

18	ROAD	            车道/道路
19	LANE_MARKER	        车道线
20	OTHER_GROUND	    其他地面
21	WALKABLE	        可行走区域
22	SIDEWALK	        人行道
'''
def map_step1(
    map_points,
    vis_bool=False,
    step=0.2,
    strict_std_thresh=0.05,
    medium_std_thresh=0.125,
    b_select_medium_low_layer=True,
    medium_ground_layer_th=0.1,
    b_select_rough_low_layer=True,
    rough_ground_layer_th=0.1,
):
    """
    将点云按每个格子的高度标准差分成三类。

    1) strict: std < strict_std_thresh
    2) medium: strict_std_thresh <= std < medium_std_thresh
    3) rough: std >= medium_std_thresh

    对 medium / rough 两类，可分别选择：
    - 保留该类格子中的全部点
    - 或仅保留“高出本格最低点一定范围内”的底层点

    参数:
        map_points: (N, 3) 点云
        vis_bool: 是否可视化
        step: 网格分辨率
        strict_std_thresh: strict 类高度标准差上界
        medium_std_thresh: medium 类高度标准差上界
        b_select_medium_low_layer: medium 类是否只保留底层点
        medium_ground_layer_th: medium 类底层点阈值，dz <= 该值保留
        b_select_rough_low_layer: rough 类是否只保留底层点
        rough_ground_layer_th: rough 类底层点阈值，dz <= 该值保留

    返回:
        strict_points_xyz: (N1, 3)，非常平坦格中的点
        medium_points_xyz: (N2, 3)，中间类点，是否只保留底层由参数控制
        rough_points_xyz: (N3, 3)，粗糙类点，是否只保留底层由参数控制
    """
    pts = np.asarray(map_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"map_points 必须是 shape (N, 3) 的数组，当前为 {pts.shape}")
    if strict_std_thresh > medium_std_thresh:
        raise ValueError(
            f"strict_std_thresh={strict_std_thresh} 不能大于 medium_std_thresh={medium_std_thresh}"
        )

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    grid_x = int(np.floor((x_max - x_min) / step)) + 1
    grid_y = int(np.floor((y_max - y_min) / step)) + 1

    x_idx = np.floor((x - x_min) / step).astype(np.int32)
    y_idx = np.floor((y - y_min) / step).astype(np.int32)
    np.clip(x_idx, 0, grid_x - 1, out=x_idx)
    np.clip(y_idx, 0, grid_y - 1, out=y_idx)

    flat = x_idx * grid_y + y_idx
    n_cells = grid_x * grid_y

    ## 统计每个格子的高度标准差
    cnt = np.bincount(flat, minlength=n_cells)
    sum_z = np.bincount(flat, weights=z, minlength=n_cells)
    sum_z2 = np.bincount(flat, weights=z * z, minlength=n_cells)

    with np.errstate(divide="ignore", invalid="ignore"):
        mean = sum_z / cnt
        var = (sum_z2 / cnt) - (mean * mean)
    var[var < 0] = 0.0
    std = np.sqrt(var)
    std[cnt == 0] = np.inf

    ## 统计每个格子的最低点高度，并得到每个点离本格最低点的高度差
    z_min = np.full(n_cells, np.inf, dtype=np.float32)
    np.minimum.at(z_min, flat, z)
    cell_min_z_for_pts = z_min[flat]
    dz = z - cell_min_z_for_pts

    ## 三档格子分类
    strict_cell_mask = std < strict_std_thresh
    medium_cell_mask = (std >= strict_std_thresh) & (std < medium_std_thresh)
    rough_cell_mask = std >= medium_std_thresh

    strict_point_mask = strict_cell_mask[flat]

    medium_point_mask = medium_cell_mask[flat]
    if b_select_medium_low_layer:
        medium_point_mask = medium_point_mask & (dz <= medium_ground_layer_th)

    rough_point_mask = rough_cell_mask[flat]
    if b_select_rough_low_layer:
        rough_point_mask = rough_point_mask & (dz <= rough_ground_layer_th)

    strict_points_xyz = pts[strict_point_mask]
    medium_points_xyz = pts[medium_point_mask]
    rough_points_xyz = pts[rough_point_mask]

    if vis_bool:
        point_cloud_list = []
        if strict_points_xyz.size > 0:
            point_cloud_list.append(strict_points_xyz)
        if medium_points_xyz.size > 0:
            point_cloud_list.append(medium_points_xyz)
        if rough_points_xyz.size > 0:
            point_cloud_list.append(rough_points_xyz)

        if point_cloud_list:
            visualize_point_cloud(
                point_cloud_list=point_cloud_list,
                point_size=1,
                show_axis=True
            )

    return strict_points_xyz, medium_points_xyz, rough_points_xyz



def ReadMapCenterTxt(map_center_txt_path: str) -> np.ndarray:
    """
    读取 semantic_map_center.txt。

    返回:
        map_center_xyz: (3,), float32
    """
    with open(map_center_txt_path, "r", encoding="utf-8") as file:
        line = file.readline().strip()

    parts = line.split()
    if len(parts) < 3:
        raise ValueError(f"地图中心文件格式错误: {map_center_txt_path}")

    map_center_xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
    return map_center_xyz


def ReadPCDXYZ(pcd_path: str) -> np.ndarray:
    """
    用 Open3D 读取单帧点云的 xyz。

    返回:
        points_xyz: (N, 3), float32
    """
    point_cloud = o3d.t.io.read_point_cloud(pcd_path)
    if "positions" not in point_cloud.point:
        raise ValueError(f"点云里没有 positions 字段: {pcd_path}")

    points_xyz = point_cloud.point["positions"].numpy().astype(np.float32, copy=False)
    return points_xyz


def LoadSemanticLabelsForFrame(semantic_npy_path: str, point_count: int) -> np.ndarray:
    """
    读取单帧语义标签。

    当前数据中 `.npy` 约定为 shape (N, 2)，
    每个点一行，第二列为语义标签。

    返回:
        semantic_labels: (N,), int32
    """
    semantic_data = np.load(semantic_npy_path)
    semantic_data = np.asarray(semantic_data, dtype=np.int32)

    if semantic_data.ndim != 2 or semantic_data.shape[1] < 2:
        raise ValueError(f"不支持的语义标签格式: {semantic_npy_path}, shape={semantic_data.shape}")
    if semantic_data.shape[0] != point_count:
        raise ValueError(
            f"语义标签行数与点数不一致: {semantic_npy_path}, "
            f"labels={semantic_data.shape[0]}, points={point_count}"
        )

    return semantic_data[:, 1].astype(np.int32, copy=False)


def PropagateLabelsByNearestNeighbor(
    query_points_xyz: np.ndarray,
    source_points_xyz: np.ndarray,
    source_labels: np.ndarray,
    radius_meters: float,
    unknown_label: int,
) -> np.ndarray:
    """
    对 query_points_xyz 中每个点，在 source_points_xyz 上找最近邻并做标签传播。

    输入:
        query_points_xyz: (N1, 3)
        source_points_xyz: (N2, 3)
        source_labels: (N2,)

    返回:
        propagated_labels: (N1,), int32
    """
    if source_points_xyz.shape[0] == 0:
        return np.full(query_points_xyz.shape[0], unknown_label, dtype=np.int32)

    tree = cKDTree(source_points_xyz)
    distances, indices = tree.query(query_points_xyz, k=1, workers=-1)
    propagated_labels = np.full(query_points_xyz.shape[0], unknown_label, dtype=np.int32)
    b_valid = distances <= radius_meters
    propagated_labels[b_valid] = source_labels[indices[b_valid]]
    return propagated_labels


def AssignPointsToNearestClass(
    query_points_xyz: np.ndarray,
    strict_points_xyz: np.ndarray,
    medium_points_xyz: np.ndarray,
    rough_points_xyz: np.ndarray,
    max_assign_distance_meters: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 query_points_xyz 按空间最近邻分配到 strict / medium / rough 三类之一。

    输入:
        query_points_xyz: (N, 3)
        strict_points_xyz / medium_points_xyz / rough_points_xyz:
            三类参考点云，均为 shape (Ni, 3)
        max_assign_distance_meters:
            若最近类距离超过该阈值，则该点记为未分配

    返回:
        class_ids: (N,), int32
            -1 表示未分配
             0 表示 strict
             1 表示 medium
             2 表示 rough
        nearest_distances: (N,), float32
            query 点到其最近类别参考点的距离
    """
    query_points_xyz = np.asarray(query_points_xyz, dtype=np.float32)
    if query_points_xyz.ndim != 2 or query_points_xyz.shape[1] != 3:
        raise ValueError(f"query_points_xyz 必须是 shape (N, 3) 的数组，当前为 {query_points_xyz.shape}")

    class_point_list = [strict_points_xyz, medium_points_xyz, rough_points_xyz]
    distance_matrix = np.full((query_points_xyz.shape[0], 3), np.inf, dtype=np.float32)

    for class_idx, class_points_xyz in enumerate(class_point_list):
        class_points_xyz = np.asarray(class_points_xyz, dtype=np.float32)
        if class_points_xyz.shape[0] == 0:
            continue
        class_tree = cKDTree(class_points_xyz)
        nearest_distances, _ = class_tree.query(query_points_xyz, k=1, workers=-1)
        distance_matrix[:, class_idx] = nearest_distances.astype(np.float32, copy=False)

    class_ids = np.argmin(distance_matrix, axis=1).astype(np.int32, copy=False)
    nearest_distances = distance_matrix[np.arange(query_points_xyz.shape[0]), class_ids]

    b_invalid = ~np.isfinite(nearest_distances) | (nearest_distances > max_assign_distance_meters)
    class_ids[b_invalid] = -1
    nearest_distances = nearest_distances.astype(np.float32, copy=False)

    return class_ids, nearest_distances


def BuildSequenceSemanticMap(
    sequence_dir: str,
    pcd_file_name_list,
    pose_list,
    map_center_xyz: np.ndarray,
    raw_map_voxel_size_meters: float,
    unknown_label: int,
):
    """
    构建两个世界坐标地图:
    1) raw_map_points_xyz: 所有帧点云拼起来的纯点云地图，shape (N, 3)
    2) semantic_map_points_xyz / semantic_map_labels: 有标签的稀疏语义地图
    """
    semantic_dir = os.path.join(sequence_dir, "SemanticSegmentation")
    pcd_dir = os.path.join(sequence_dir, "pcd")

    semantic_frame_id_set = {
        os.path.splitext(file_name)[0]
        for file_name in os.listdir(semantic_dir)
        if file_name.endswith(".npy")
    }  # 有单帧语义结果的帧 id 集合

    semantic_point_block_list = []  # 所有有语义标签的 world 点
    semantic_label_block_list = []  # 与 semantic_point_block_list 一一对应的标签
    raw_map_point_block_list = []  # 所有帧 world 点，用来拼纯点云地图

    ## 逐帧读取点云并变换到世界坐标
    for frame_idx, (pcd_file_name, pose6) in enumerate(zip(pcd_file_name_list, pose_list)):
        frame_id = os.path.splitext(pcd_file_name)[0]
        pcd_path = os.path.join(pcd_dir, pcd_file_name)
        car_points_xyz = ReadPCDXYZ(pcd_path)  # 当前帧原始点云，车体坐标系
        world_points_xyz = CarToWorld(car_points_xyz, pose6, map_center_xyz=map_center_xyz)  # 当前帧点云转到地图中心坐标系后的世界点
        raw_map_point_block_list.append(world_points_xyz)

        if frame_id in semantic_frame_id_set:
            semantic_npy_path = os.path.join(semantic_dir, f"{frame_id}.npy")
            point_labels = LoadSemanticLabelsForFrame(
                semantic_npy_path=semantic_npy_path,
                point_count=car_points_xyz.shape[0],
            )
            b_labeled = point_labels != unknown_label  # 当前帧中真正有语义标签的点
            if np.any(b_labeled):
                semantic_point_block_list.append(world_points_xyz[b_labeled])
                semantic_label_block_list.append(point_labels[b_labeled])

        print(f"[LoadFrame] frame={frame_idx + 1}/{len(pcd_file_name_list)} name={frame_id}")

    ## 合并为纯点云地图并下采样
    raw_map_points_xyz = np.concatenate(raw_map_point_block_list, axis=0).astype(np.float32, copy=False)  # 所有帧拼接后的纯点云地图
    raw_map_points_xyz = voxel_downsample(
        raw_map_points_xyz,
        voxel_size=raw_map_voxel_size_meters,
        return_mapping=False,
        pick_closest_to_center=True
    ).astype(np.float32, copy=False)  # 下采样后的纯点云地图

    ## 合并为稀疏语义地图
    semantic_map_points_xyz = np.concatenate(semantic_point_block_list, axis=0).astype(np.float32, copy=False)  # 稀疏语义地图点
    semantic_map_labels = np.concatenate(semantic_label_block_list, axis=0).astype(np.int32, copy=False)  # 稀疏语义地图点对应的标签

    return raw_map_points_xyz, semantic_map_points_xyz, semantic_map_labels


def LoadDetectionInfo(detection_info_path: str):
    if not os.path.exists(detection_info_path):
        return None

    with open(detection_info_path, "rb") as file:
        detection_info = pickle.load(file)

    if not isinstance(detection_info, list):
        raise TypeError(f"detection_info.pkl 不是 list，而是 {type(detection_info)}")

    type_counter = {}
    object_count_list = []
    for frame_info in detection_info:
        object_num = len(frame_info["objects"])
        object_count_list.append(object_num)
        for object_info in frame_info["objects"]:
            object_type = object_info["type"]
            if object_type not in type_counter:
                type_counter[object_type] = 0
            type_counter[object_type] += 1

    return detection_info


def GenerateSequenceSemanticLabels(
    sequence_dir: str,
    propagate_radius_meters: float = 0.15,
    raw_map_voxel_size_meters: float = 0.03,
    unknown_label: int = 0,
):
    """
    输入一个序列 output 目录，生成 `sequence_SemanticSegmentation/*.label`。

    输入:
        sequence_dir:
            例如 `/.../FH_waymo/.../output`
    """
    pcd_dir = os.path.join(sequence_dir, "pcd")
    semantic_dir = os.path.join(sequence_dir, "SemanticSegmentation")
    pose_txt_path = os.path.join(sequence_dir, "pose_slam_euler_hba_refined.txt")
    map_center_txt_path = os.path.join(sequence_dir, "SemanticMap", "semantic_map_center.txt")
    detection_info_path = os.path.join(sequence_dir, "detection_info.pkl")
    output_dir = os.path.join(sequence_dir, "sequence_SemanticSegmentation")

    if not os.path.isdir(pcd_dir):
        raise FileNotFoundError(f"未找到 pcd 目录: {pcd_dir}")
    if not os.path.isdir(semantic_dir):
        raise FileNotFoundError(f"未找到 SemanticSegmentation 目录: {semantic_dir}")
    if not os.path.isfile(pose_txt_path):
        raise FileNotFoundError(f"未找到 pose 文件: {pose_txt_path}")
    if not os.path.isfile(map_center_txt_path):
        raise FileNotFoundError(f"未找到地图中心文件: {map_center_txt_path}")

    os.makedirs(output_dir, exist_ok=True)

    ## 读取序列文件列表、位姿和地图中心
    pcd_file_name_list = sorted(
        file_name for file_name in os.listdir(pcd_dir)
        if file_name.endswith(".pcd")
    )
    frame_id_list = [os.path.splitext(file_name)[0] for file_name in pcd_file_name_list]
    pose_list = LoadPosesFromPoseTxt(pose_txt_path, frame_id_list=frame_id_list)
    map_center_xyz = ReadMapCenterTxt(map_center_txt_path)

    ## 读取检测信息摘要
    detection_info = LoadDetectionInfo(detection_info_path)


    ## 构建纯点云地图和稀疏语义地图
    raw_map_points_xyz, semantic_map_points_xyz, semantic_map_labels = BuildSequenceSemanticMap(
        sequence_dir=sequence_dir,
        pcd_file_name_list=pcd_file_name_list,
        pose_list=pose_list,
        map_center_xyz=map_center_xyz,
        raw_map_voxel_size_meters=raw_map_voxel_size_meters,
        unknown_label=unknown_label,
    )

    ## 利用方格内方差将点分成 strict / medium / rough 三类
    strict_points_xyz, medium_points_xyz, rough_points_xyz = map_step1(
        raw_map_points_xyz,
        vis_bool=False,
        step=0.3,
        strict_std_thresh=0.03,
        medium_std_thresh=0.125,
        b_select_medium_low_layer=False,
        medium_ground_layer_th=0.1,
        b_select_rough_low_layer=False,
        rough_ground_layer_th=0.1,
    )
    print(
        f"[MapStep1] strict_points={strict_points_xyz.shape[0]} "
        f"medium_points={medium_points_xyz.shape[0]} "
        f"rough_points={rough_points_xyz.shape[0]}"
    )
    visualize_point_cloud(
        point_cloud_list=[strict_points_xyz, medium_points_xyz, rough_points_xyz],
        point_size=1,
        show_axis=True
    )

    ## 基于 raw_map 三类子图的空间最近邻，把 raw_map 和 semantic_map 都分到 strict / medium / rough
    k_class_assign_radius_meters = max(propagate_radius_meters, raw_map_voxel_size_meters * 2.0)

    raw_map_class_ids, raw_map_class_distances = AssignPointsToNearestClass(
        query_points_xyz=raw_map_points_xyz,
        strict_points_xyz=strict_points_xyz,
        medium_points_xyz=medium_points_xyz,
        rough_points_xyz=rough_points_xyz,
        max_assign_distance_meters=k_class_assign_radius_meters,
    )
    print('1')
    semantic_map_class_ids, semantic_map_class_distances = AssignPointsToNearestClass(
        query_points_xyz=semantic_map_points_xyz,
        strict_points_xyz=strict_points_xyz,
        medium_points_xyz=medium_points_xyz,
        rough_points_xyz=rough_points_xyz,
        max_assign_distance_meters=k_class_assign_radius_meters,
    )



    ## 在同类之间做语义传播
    raw_map_labels = np.full(raw_map_points_xyz.shape[0], unknown_label, dtype=np.int32)

    b_raw_strict = raw_map_class_ids == 0
    b_raw_medium = raw_map_class_ids == 1
    b_raw_rough = raw_map_class_ids == 2
    b_semantic_strict = semantic_map_class_ids == 0
    b_semantic_medium = semantic_map_class_ids == 1
    b_semantic_rough = semantic_map_class_ids == 2

    raw_map_labels[b_raw_strict] = PropagateLabelsByNearestNeighbor(
        query_points_xyz=raw_map_points_xyz[b_raw_strict],
        source_points_xyz=semantic_map_points_xyz[b_semantic_strict],
        source_labels=semantic_map_labels[b_semantic_strict],
        radius_meters=propagate_radius_meters,
        unknown_label=unknown_label,
    )
    raw_map_labels[b_raw_medium] = PropagateLabelsByNearestNeighbor(
        query_points_xyz=raw_map_points_xyz[b_raw_medium],
        source_points_xyz=semantic_map_points_xyz[b_semantic_medium],
        source_labels=semantic_map_labels[b_semantic_medium],
        radius_meters=propagate_radius_meters,
        unknown_label=unknown_label,
    )
    raw_map_labels[b_raw_rough] = PropagateLabelsByNearestNeighbor(
        query_points_xyz=raw_map_points_xyz[b_raw_rough],
        source_points_xyz=semantic_map_points_xyz[b_semantic_rough],
        source_labels=semantic_map_labels[b_semantic_rough],
        radius_meters=propagate_radius_meters,
        unknown_label=unknown_label,
    )

    visualize_point_cloud(raw_map_points_xyz,raw_map_labels)

    ## 将稠密语义地图回写到每一帧
    labeled_raw_tree = cKDTree(raw_map_points_xyz)  # 在稠密语义地图点上建树

    for frame_idx, (pcd_file_name, pose6) in enumerate(zip(pcd_file_name_list, pose_list)):
        frame_id = os.path.splitext(pcd_file_name)[0]
        pcd_path = os.path.join(pcd_dir, pcd_file_name)
        car_points_xyz = ReadPCDXYZ(pcd_path)  # 当前帧原始点云，车体坐标系
        world_points_xyz = CarToWorld(car_points_xyz, pose6, map_center_xyz=map_center_xyz)  # 当前帧 world 点

        distances, indices = labeled_raw_tree.query(world_points_xyz, k=1, workers=-1)  # 当前帧每个 world 点在稠密语义地图中的最近邻
        frame_labels = np.full(car_points_xyz.shape[0], unknown_label, dtype=np.uint16)  # 当前帧最终输出标签
        b_valid = distances <= propagate_radius_meters  # 最近邻距离在传播半径内的点
        frame_labels[b_valid] = raw_map_labels[indices[b_valid]].astype(np.uint16, copy=False)

        output_path = os.path.join(output_dir, f"{frame_id}.label")
        frame_labels.tofile(output_path)

        visualize_point_cloud(car_points_xyz, frame_labels)

        labeled_count = int(np.count_nonzero(frame_labels))
        print(
            f"[WriteLabel] frame={frame_idx + 1}/{len(pcd_file_name_list)} "
            f"name={frame_id} labeled={labeled_count}/{frame_labels.shape[0]}"
        )


def Main():
    GenerateSequenceSemanticLabels(
        "/media/gk/data/lyf/aa_data/FH_waymo/9179922063516210200_157_000_177_000/output"
    )


if __name__ == "__main__":
    Main()
