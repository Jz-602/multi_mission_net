import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d



"""
• 把数据增强也写成 4x4 齐次变换矩阵，然后融合进多帧对齐矩阵里。
  设：
  p_i          第 i 帧原始 lidar/ego 坐标系点，shape [N, 3]
  T_w_i        第 i 帧 ego/lidar -> world 的 pose，shape [4, 4]
  p_i_aug      第 i 帧增强后的点
  A_i          第 i 帧数据增强矩阵，shape [4, 4]
  c            当前帧
  如果没有数据增强，历史帧 i 对齐到当前帧 c 的矩阵是：
  M_i_to_c = inv(T_w_c) @ T_w_i
  也就是：
  p_i_in_current = M_i_to_c @ p_i
  如果有数据增强，并且网络输入的是增强后的点：
  p_i_aug = A_i @ p_i
  p_c_aug = A_c @ p_c
  那么历史增强点 p_i_aug 对齐到当前增强坐标系 p_c_aug 的矩阵应该是：
  M_aug_i_to_c = A_c @ inv(T_w_c) @ T_w_i @ inv(A_i)
  也就是：
  p_i_aug_in_current_aug = M_aug_i_to_c @ p_i_aug
  这个矩阵就同时融合了：
  历史帧增强的逆变换 inv(A_i)
  历史帧 ego -> world: T_w_i
  world -> 当前帧 ego: inv(T_w_c)
  当前帧增强: A_c
  所以答案是：可以让对齐矩阵融合数据增强信息，而且应该这么做。
  如果当前帧和所有历史帧使用同一个增强矩阵 A，那么：
  M_aug_i_to_c = A @ inv(T_w_c) @ T_w_i @ inv(A)
  注意不是简单地继续用：
  inv(T_w_c) @ T_w_i
  因为点云坐标系已经被增强矩阵重新定义了。
  如果某些样本不做数据增强，就令：
  A_i = I
  A_c = I
  那么自然退化回原始对齐矩阵：
  M_aug_i_to_c = inv(T_w_c) @ T_w_i
  所以 MEMORY 训练里可以统一维护一个接口：
  align_matrix_i_to_current = A_current @ inv(pose_current) @ pose_i @ inv(A_i)
  然后无论点云是否增强，网络都用这个矩阵对齐。没有增强时 A=I，有增强时 A 记录真实增强矩阵。
"""


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/estella/Osmanthus/aa_data/FH_data/FH_semantickitti")

INFO_NAME = "semantickitti_infos_validation.pkl"
SEQUENCE_NAME = "08"
START_IDX = 0
FRAME_COUNT = 10
CASE_NAME_LIST = ["rotation", "scaling", "translation", "flip"]
VOXEL_SIZE = 0.1
POINT_SIZE = 2.0
RANDOM_SEED = 7
B_RANDOM_PER_FRAME = False

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from lyf_tools import visualize_point_cloud, voxel_downsample


def LoadInfos():
    info_path = DATA_ROOT / INFO_NAME
    with open(info_path, "rb") as file:
        infos = pickle.load(file)
    return infos


def SelectContinuousInfos(infos):
    sequence_infos = [info for info in infos if info["sequence_name"] == SEQUENCE_NAME]
    sequence_infos = sorted(sequence_infos, key=lambda item: int(item["original_sample_idx"]))
    selected_infos = sequence_infos[START_IDX:START_IDX + FRAME_COUNT]
    if len(selected_infos) != FRAME_COUNT:
        raise ValueError(
            f"sequence {SEQUENCE_NAME} only has {len(sequence_infos)} frames, "
            f"cannot select START_IDX={START_IDX}, FRAME_COUNT={FRAME_COUNT}"
        )
    return selected_infos


def LoadFramePoints(info):
    ## points_xyz: shape [N, 3]，SemanticKITTI 当前帧 lidar/ego 坐标系，单位 m。
    frame_name = str(info["frame_timestamp_micros"])[:-3]
    pcd_path = DATA_ROOT / info["sequence_name"] / "output" / "pcd" / f"{frame_name}.pcd"
    point_cloud = o3d.io.read_point_cloud(str(pcd_path))
    points_xyz = np.asarray(point_cloud.points, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"invalid points shape from {pcd_path}: {points_xyz.shape}")
    return points_xyz


def LoadFrameSemanticLabel(info):
    ## semantic_label: shape [N]，与 points_xyz 一一对应；类别仍为 SemanticKITTI 原始语义标签。
    semantic_path = DATA_ROOT / info["semantic_path"]
    semantic_data = np.load(semantic_path)
    semantic_label = np.asarray(semantic_data[:, 1], dtype=np.int32)
    return semantic_label


def TransformToWorld(points_xyz, pose):
    ## pose: shape [4, 4]，当前帧 lidar/ego -> world；输出 points_world: shape [N, 3]，world 坐标系。
    pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    rotation = pose[:3, :3]
    translation = pose[:3, 3].reshape(1, 3)
    points_world = points_xyz @ rotation.T + translation
    return points_world.astype(np.float32, copy=False)


def BuildRotationMatrix(noise_rotation):
    cos_value = np.cos(noise_rotation)
    sin_value = np.sin(noise_rotation)
    rotation_matrix = np.array([
        [cos_value, -sin_value, 0.0],
        [sin_value, cos_value, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return rotation_matrix


def ApplyAugmentation(points_xyz, case_name, rng):
    ## 增强发生在每帧自己的 lidar/ego 坐标系；这里故意不同步修改 pose，用于观察对齐偏差。
    augmented_points_xyz = points_xyz.copy()

    if case_name == "rotation":
        noise_rotation = rng.uniform(-0.785, 0.785) if B_RANDOM_PER_FRAME else 0.785
        rotation_matrix = BuildRotationMatrix(noise_rotation)
        augmented_points_xyz = augmented_points_xyz @ rotation_matrix.T
        return augmented_points_xyz.astype(np.float32, copy=False), f"rotation={noise_rotation:.3f} rad"

    if case_name == "scaling":
        noise_scale = rng.uniform(0.9, 1.1) if B_RANDOM_PER_FRAME else 1.1
        augmented_points_xyz *= noise_scale
        return augmented_points_xyz.astype(np.float32, copy=False), f"scale={noise_scale:.3f}"

    if case_name == "translation":
        noise_translate = rng.normal(0.0, 0.5, 1) if B_RANDOM_PER_FRAME else np.array([0.5], dtype=np.float32)
        augmented_points_xyz += noise_translate
        return augmented_points_xyz.astype(np.float32, copy=False), f"translation=[{float(noise_translate[0]):.3f}]"

    if case_name == "flip":
        if B_RANDOM_PER_FRAME:
            b_flip_y = bool(rng.choice([False, True], p=[0.5, 0.5]))
            b_flip_x = bool(rng.choice([False, True], p=[0.5, 0.5]))
        else:
            b_flip_y = True
            b_flip_x = True

        if b_flip_y:
            augmented_points_xyz[:, 1] = -augmented_points_xyz[:, 1]
        if b_flip_x:
            augmented_points_xyz[:, 0] = -augmented_points_xyz[:, 0]
        return augmented_points_xyz.astype(np.float32, copy=False), f"flip_y={b_flip_y}, flip_x={b_flip_x}"

    raise ValueError(f"unknown case_name: {case_name}")


def DownsampleForVisual(points_xyz, semantic_label):
    if VOXEL_SIZE <= 0:
        return points_xyz, semantic_label
    points_xyz, idx_selected, _ = voxel_downsample(points_xyz, VOXEL_SIZE, return_mapping=True)
    semantic_label = semantic_label[idx_selected]
    return points_xyz, semantic_label


def BuildAlignedPointCloudList(frame_items, case_name=None):
    merged_points_list = []
    merged_label_list = []
    aug_desc_list = []

    for frame_idx, frame_item in enumerate(frame_items):
        points_xyz = frame_item["points_xyz"]
        semantic_label = frame_item["semantic_label"]
        if case_name is not None:
            seed = RANDOM_SEED + frame_idx if B_RANDOM_PER_FRAME else RANDOM_SEED
            rng = np.random.default_rng(seed)
            points_xyz, aug_desc = ApplyAugmentation(points_xyz, case_name, rng)
            aug_desc_list.append(f"{frame_item['token']}: {aug_desc}")

        ## 使用原始 pose 做同样的拼接：lidar/ego 坐标点 -> world 坐标点。
        points_world = TransformToWorld(points_xyz, frame_item["pose"])
        points_world, semantic_label = DownsampleForVisual(points_world, semantic_label)
        merged_points_list.append(points_world)
        merged_label_list.append(semantic_label)

    merged_points = np.concatenate(merged_points_list, axis=0)
    merged_label = np.concatenate(merged_label_list, axis=0)
    return merged_points, merged_label, aug_desc_list


def PrintFrameSummary(frame_items):
    print("selected frames:")
    for frame_item in frame_items:
        pose = frame_item["pose"]
        print(
            f"  {frame_item['token']} points={frame_item['points_xyz'].shape[0]} "
            f"world_t=({pose[0, 3]:.2f}, {pose[1, 3]:.2f}, {pose[2, 3]:.2f})"
        )


def VisualizeCase(frame_items, case_name):
    if case_name is None:
        print("\n## original pose alignment")
    else:
        print(f"\n## {case_name} in each local lidar frame, then align with original pose")

    points, labels, aug_desc_list = BuildAlignedPointCloudList(frame_items, case_name)
    for aug_desc in aug_desc_list:
        print("  " + aug_desc)

    visualize_point_cloud(
        points=points,
        labels=labels,
        point_size=POINT_SIZE,
        show_axis=True,
    )


def main():
    infos = LoadInfos()
    selected_infos = SelectContinuousInfos(infos)

    frame_items = []
    for info in selected_infos:
        frame_item = {
            "token": info["token"],
            "pose": np.asarray(info["pose"], dtype=np.float32),
            "points_xyz": LoadFramePoints(info),
            "semantic_label": LoadFrameSemanticLabel(info),
        }
        if frame_item["points_xyz"].shape[0] != frame_item["semantic_label"].shape[0]:
            raise ValueError(
                f"point count and semantic label count mismatch for {frame_item['token']}: "
                f"{frame_item['points_xyz'].shape[0]} vs {frame_item['semantic_label'].shape[0]}"
            )
        frame_items.append(frame_item)

    PrintFrameSummary(frame_items)
    VisualizeCase(frame_items, None)

    for case_name in CASE_NAME_LIST:
        VisualizeCase(frame_items, case_name)


if __name__ == "__main__":
    main()
