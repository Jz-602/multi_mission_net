import os
import pickle
import math

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
    pose_txt_path = os.path.join(sequence_dir, "pose_slam_euler.txt")
    map_center_txt_path = os.path.join(sequence_dir, "SemanticMap", "semantic_map_center.txt")
    detection_info_path = os.path.join(sequence_dir, "detection_info.pkl")
    output_dir = os.path.join(sequence_dir, "sequence_SemanticSegmentation")

    if not os.path.isdir(pcd_dir):
        raise FileNotFoundError(f"未找到 pcd 目录: {pcd_dir}")
    if not os.path.isdir(semantic_dir):
        raise FileNotFoundError(f"未找到 SemanticSegmentation 目录: {semantic_dir}")
    if not os.path.isfile(pose_txt_path):
        raise FileNotFoundError(f"未找到 pose 文件: {pose_txt_path}")
    os.makedirs(output_dir, exist_ok=True)

    ## 读取序列文件列表、位姿和地图中心
    pcd_file_name_list = sorted(
        file_name for file_name in os.listdir(pcd_dir)
        if file_name.endswith(".pcd")
    )
    frame_id_list = [os.path.splitext(file_name)[0] for file_name in pcd_file_name_list]
    pose_list = LoadPosesFromPoseTxt(pose_txt_path, frame_id_list=frame_id_list)
    map_center_xyz = None
    if os.path.isfile(map_center_txt_path):
        map_center_xyz = ReadMapCenterTxt(map_center_txt_path)
        print(f"[MapCenter] use map center file: {map_center_txt_path}")
    else:
        print("[MapCenter] semantic_map_center.txt not found, use raw world coordinates without map center offset")

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

    visualize_point_cloud(raw_map_points_xyz)

    ## 将稀疏语义传播到稠密纯云点地图
    raw_map_labels = np.full(raw_map_points_xyz.shape[0], unknown_label, dtype=np.int32)
    if semantic_map_points_xyz.shape[0] > 0:
        semantic_tree = cKDTree(semantic_map_points_xyz)
        distances, indices = semantic_tree.query(raw_map_points_xyz, k=1, workers=-1)
        b_valid = distances <= propagate_radius_meters
        raw_map_labels[b_valid] = semantic_map_labels[indices[b_valid]]

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


def GenerateSequenceSemanticLabelsFromOcc3D(
    sequence_dir: str,
    match_radius_meters: float = 0.15,
    unknown_label: int = 0,
):
    """
    优先使用 Occ3D 单帧点云给当前帧原始点云赋标签。

    坐标系说明:
    - 输入 `car_points_xyz` 为当前帧车体坐标系点，shape (N, 3)
    - Occ3D 中的点按当前帧姿态做过旋转，但不做平移
    - 因此这里将 `car_points_xyz` 仅按 pose 的旋转矩阵前向旋转，再与 Occ3D 点做最近邻匹配
    """
    pcd_dir = os.path.join(sequence_dir, "pcd")
    pose_txt_path = os.path.join(sequence_dir, "pose_slam_euler.txt")
    output_dir = os.path.join(sequence_dir, "sequence_SemanticSegmentation")

    occ3d_dir = os.path.join(sequence_dir, "Occ3D")

    if not os.path.isdir(pcd_dir):
        raise FileNotFoundError(f"未找到 pcd 目录: {pcd_dir}")
    if not os.path.isfile(pose_txt_path):
        raise FileNotFoundError(f"未找到 pose 文件: {pose_txt_path}")

    os.makedirs(output_dir, exist_ok=True)

    pcd_file_name_list = sorted(
        file_name for file_name in os.listdir(pcd_dir)
        if file_name.endswith(".pcd")
    )
    frame_id_list = [os.path.splitext(file_name)[0] for file_name in pcd_file_name_list]
    pose_list = LoadPosesFromPoseTxt(pose_txt_path, frame_id_list=frame_id_list)

    for frame_idx, (pcd_file_name, pose6) in enumerate(zip(pcd_file_name_list, pose_list)):
        frame_id = os.path.splitext(pcd_file_name)[0]
        raw_pcd_path = os.path.join(pcd_dir, pcd_file_name)
        occ3d_pcd_path = os.path.join(occ3d_dir, f"{frame_id}.pcd")
        if not os.path.isfile(occ3d_pcd_path):
            raise FileNotFoundError(f"Occ3D 缺少当前帧点云: {occ3d_pcd_path}")

        car_points_xyz = ReadPCDXYZ(raw_pcd_path)  # 当前帧原始点云，车体坐标系，shape (N, 3)

        occ3d_point_cloud = o3d.t.io.read_point_cloud(occ3d_pcd_path)
        if "positions" not in occ3d_point_cloud.point:
            raise ValueError(f"Occ3D 点云里没有 positions 字段: {occ3d_pcd_path}")
        if "intensity" not in occ3d_point_cloud.point:
            raise ValueError(f"Occ3D 点云里没有 intensity 字段: {occ3d_pcd_path}")

        occ3d_points_xyz = occ3d_point_cloud.point["positions"].numpy().astype(np.float32, copy=False)
        occ3d_labels = occ3d_point_cloud.point["intensity"].numpy().reshape(-1).astype(np.uint16, copy=False)

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

        rotated_car_points_xyz = car_points_xyz @ rotation_matrix.T  # 只旋转，不平移
        frame_labels = np.full(car_points_xyz.shape[0], unknown_label, dtype=np.uint16)

        if occ3d_points_xyz.shape[0] > 0:
            occ3d_tree = cKDTree(occ3d_points_xyz)
            distances, indices = occ3d_tree.query(rotated_car_points_xyz, k=1, workers=-1)
            b_valid = distances <= match_radius_meters
            frame_labels[b_valid] = occ3d_labels[indices[b_valid]]

        output_path = os.path.join(output_dir, f"{frame_id}.label")
        frame_labels.tofile(output_path)

        labeled_count = int(np.count_nonzero(frame_labels))
        print(
            f"[WriteLabel][Occ3D] frame={frame_idx + 1}/{len(pcd_file_name_list)} "
            f"name={frame_id} labeled={labeled_count}/{frame_labels.shape[0]}"
        )


def Main():
    sequence_dir = "/media/estella/Osmanthus/aa_data/FH_data/FH_waymo_small/15832924468527961_1564_160_1584_160/output"
    b_has_occ3d = os.path.isdir(os.path.join(sequence_dir, "Occ3D"))

    if b_has_occ3d:
        GenerateSequenceSemanticLabelsFromOcc3D(sequence_dir)
    else:
        GenerateSequenceSemanticLabels(sequence_dir)


if __name__ == "__main__":
    Main()


'''
## 长时间不怎么动的序列
01. 5144634012371033641_920_000_940_000 | longest_low_motion_duration_sec=19.800 | longest_low_motion_frames=199 | start=1554243497822 | end=1554243517622 | mean_step_distance_m=0.0064
02. 8327447186504415549_5200_000_5220_000 | longest_low_motion_duration_sec=19.800 | longest_low_motion_frames=199 | start=1543276148562 | end=1543276168362 | mean_step_distance_m=0.0070
03. 15696964848687303249_4615_200_4635_200 | longest_low_motion_duration_sec=19.701 | longest_low_motion_frames=198 | start=1507241617619 | end=1507241637320 | mean_step_distance_m=0.0051
04. 6606076833441976341_1340_000_1360_000 | longest_low_motion_duration_sec=19.700 | longest_low_motion_frames=198 | start=1547744136737 | end=1547744156437 | mean_step_distance_m=0.0079
05. 14133920963894906769_1480_000_1500_000 | longest_low_motion_duration_sec=18.000 | longest_low_motion_frames=181 | start=1558033977547 | end=1558033995547 | mean_step_distance_m=0.0168
06. 18295766828140813622_6775_000_6795_000 | longest_low_motion_duration_sec=17.801 | longest_low_motion_frames=179 | start=1520779920580 | end=1520779938381 | mean_step_distance_m=0.0177
07. 4457475194088194008_3100_000_3120_000 | longest_low_motion_duration_sec=17.200 | longest_low_motion_frames=173 | start=1549651357289 | end=1549651374489 | mean_step_distance_m=0.0136
08. 12505030131868863688_1740_000_1760_000 | longest_low_motion_duration_sec=17.000 | longest_low_motion_frames=171 | start=1558483877522 | end=1558483894522 | mean_step_distance_m=0.0373
09. 204421859195625800_1080_000_1100_000 | longest_low_motion_duration_sec=17.000 | longest_low_motion_frames=171 | start=1557159634922 | end=1557159651922 | mean_step_distance_m=0.0348
10. 9521653920958139982_940_000_960_000 | longest_low_motion_duration_sec=15.000 | longest_low_motion_frames=151 | start=1557886617347 | end=1557886632347 | mean_step_distance_m=0.0222
11. 10023947602400723454_1120_000_1140_000 | longest_low_motion_duration_sec=14.700 | longest_low_motion_frames=148 | start=1552440200462 | end=1552440215162 | mean_step_distance_m=0.0459
12. 11113047206980595400_2560_000_2580_000 | longest_low_motion_duration_sec=10.900 | longest_low_motion_frames=110 | start=1557855055247 | end=1557855066147 | mean_step_distance_m=0.1351
13. 16087604685956889409_40_000_60_000 | longest_low_motion_duration_sec=7.100 | longest_low_motion_frames=72 | start=1553206988674 | end=1553206995774 | mean_step_distance_m=0.1767
14. 12339284075576056695_1920_000_1940_000 | longest_low_motion_duration_sec=6.799 | longest_low_motion_frames=69 | start=1557335963149 | end=1557335969948 | mean_step_distance_m=0.1622
15. 17818548625922145895_1372_430_1392_430 | longest_low_motion_duration_sec=6.400 | longest_low_motion_frames=65 | start=1513637703532 | end=1513637709932 | mean_step_distance_m=0.5037
16. 4967385055468388261_720_000_740_000 | longest_low_motion_duration_sec=5.600 | longest_low_motion_frames=57 | start=1552658071347 | end=1552658076947 | mean_step_distance_m=0.3016
17. 16578409328451172992_3780_000_3800_000 | longest_low_motion_duration_sec=5.200 | longest_low_motion_frames=53 | start=1557341347122 | end=1557341352322 | mean_step_distance_m=0.2699
18. 14098575914252965554_511_000_531_000 | longest_low_motion_duration_sec=4.699 | longest_low_motion_frames=45 | start=1521563064101 | end=1521563068800 | mean_step_distance_m=0.0958
19. 6771922013310347577_4249_290_4269_290 | longest_low_motion_duration_sec=4.500 | longest_low_motion_frames=46 | start=1511659168486 | end=1511659172986 | mean_step_distance_m=0.1346
20. 3644145307034257093_3000_400_3020_400 | longest_low_motion_duration_sec=3.700 | longest_low_motion_frames=36 | start=1511657365258 | end=1511657368958 | mean_step_distance_m=0.7335
21. 1918764220984209654_5680_000_5700_000 | longest_low_motion_duration_sec=3.500 | longest_low_motion_frames=36 | start=1541710113724 | end=1541710117224 | mean_step_distance_m=0.2915
22. 18418533015310829002_480_000_500_000 | longest_low_motion_duration_sec=2.101 | longest_low_motion_frames=22 | start=1550088116752 | end=1550088118853 | mean_step_distance_m=0.2732
23. 14106113060128637865_1200_000_1220_000 | longest_low_motion_duration_sec=2.100 | longest_low_motion_frames=22 | start=1547700372572 | end=1547700374672 | mean_step_distance_m=0.4965
24. 2618605158242502527_1860_000_1880_000 | longest_low_motion_duration_sec=2.100 | longest_low_motion_frames=22 | start=1552675802977 | end=1552675805077 | mean_step_distance_m=0.3031
'''