import pickle
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/media/estella/Osmanthus/aa_data/FH_data/FH_semantickitti")

INFO_NAME = "semantickitti_infos_validation.pkl"
SEQUENCE_NAME = "08"
START_IDX = 0
FRAME_COUNT = 10
CURRENT_FRAME_OFFSET = FRAME_COUNT - 1
CASE_NAME = "combined"
VOXEL_SIZE = 0.1
POINT_SIZE = 2.0
RANDOM_SEED = 7
B_SHARED_AUGMENTATION = True
B_RANDOM_PER_FRAME = False
FIXED_ROTATION = 0.785
FIXED_SCALE = 1.1
FIXED_TRANSLATION_XYZ = np.array([0.5, -0.25, 0.0], dtype=np.float32)

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


def BuildRotationMatrix(noise_rotation):
    cos_value = np.cos(noise_rotation)
    sin_value = np.sin(noise_rotation)
    rotation_matrix = np.array([
        [cos_value, -sin_value, 0.0],
        [sin_value, cos_value, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return rotation_matrix


def BuildHomogeneousMatrix(rotation_matrix=None, translation_xyz=None):
    transform_matrix = np.eye(4, dtype=np.float32)
    if rotation_matrix is not None:
        transform_matrix[:3, :3] = np.asarray(rotation_matrix, dtype=np.float32).reshape(3, 3)
    if translation_xyz is not None:
        transform_matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float32).reshape(3)
    return transform_matrix


def ApplyTransform(points_xyz, transform_matrix):
    ## points_xyz: shape [N, 3]，输入坐标系由 transform_matrix 定义；输出 shape [N, 3]。
    transform_matrix = np.asarray(transform_matrix, dtype=np.float32).reshape(4, 4)
    rotation = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3].reshape(1, 3)
    transformed_points_xyz = points_xyz @ rotation.T + translation
    return transformed_points_xyz.astype(np.float32, copy=False)


def BuildAugmentationMatrix(case_name, rng):
    ## A: shape [4, 4]，原始 lidar/ego 坐标系 -> 增强后 lidar/ego 坐标系。
    if case_name == "rotation":
        noise_rotation = rng.uniform(-0.785, 0.785) if B_RANDOM_PER_FRAME else FIXED_ROTATION
        aug_matrix = BuildHomogeneousMatrix(rotation_matrix=BuildRotationMatrix(noise_rotation))
        return aug_matrix, f"rotation={noise_rotation:.3f} rad"

    if case_name == "scaling":
        noise_scale = rng.uniform(0.5, 1.5) if B_RANDOM_PER_FRAME else FIXED_SCALE
        aug_matrix = np.diag([noise_scale, noise_scale, noise_scale, 1.0]).astype(np.float32)
        return aug_matrix, f"scale={noise_scale:.3f}"

    if case_name == "translation":
        if B_RANDOM_PER_FRAME:
            noise_translate = rng.normal(0.0, 0.5, 3).astype(np.float32)
        else:
            noise_translate = FIXED_TRANSLATION_XYZ.copy()
        aug_matrix = BuildHomogeneousMatrix(translation_xyz=noise_translate)
        return aug_matrix, f"translation={np.array2string(noise_translate, precision=3)}"

    if case_name == "flip":
        if B_RANDOM_PER_FRAME:
            b_flip_x = bool(rng.choice([False, True], p=[0.5, 0.5]))
            b_flip_y = bool(rng.choice([False, True], p=[0.5, 0.5]))
        else:
            b_flip_x = True
            b_flip_y = True
        aug_matrix = np.eye(4, dtype=np.float32)
        if b_flip_x:
            aug_matrix[0, 0] = -1.0
        if b_flip_y:
            aug_matrix[1, 1] = -1.0
        return aug_matrix, f"flip_x={b_flip_x}, flip_y={b_flip_y}"

    if case_name == "combined":
        noise_rotation = rng.uniform(-0.785, 0.785) if B_RANDOM_PER_FRAME else FIXED_ROTATION
        noise_scale = rng.uniform(0.9, 1.1) if B_RANDOM_PER_FRAME else FIXED_SCALE
        noise_translate = (
            rng.normal(0.0, 0.5, 3).astype(np.float32)
            if B_RANDOM_PER_FRAME
            else FIXED_TRANSLATION_XYZ.copy()
        )

        rotation_aug = BuildHomogeneousMatrix(rotation_matrix=BuildRotationMatrix(noise_rotation))
        scaling_aug = np.diag([noise_scale, noise_scale, noise_scale, 1.0]).astype(np.float32)
        flip_aug = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)
        translation_aug = BuildHomogeneousMatrix(translation_xyz=noise_translate)

        aug_matrix = translation_aug @ flip_aug @ scaling_aug @ rotation_aug
        aug_desc = (
            f"combined rotation={noise_rotation:.3f} rad, scale={noise_scale:.3f}, "
            f"flip_x=True, flip_y=True, translation={np.array2string(noise_translate, precision=3)}"
        )
        return aug_matrix.astype(np.float32), aug_desc

    raise ValueError(f"unknown case_name: {case_name}")


def BuildAugmentationList(frame_items):
    aug_matrix_list = []
    aug_desc_list = []

    if B_SHARED_AUGMENTATION:
        rng = np.random.default_rng(RANDOM_SEED)
        aug_matrix, aug_desc = BuildAugmentationMatrix(CASE_NAME, rng)
        for frame_item in frame_items:
            aug_matrix_list.append(aug_matrix.copy())
            aug_desc_list.append(f"{frame_item['token']}: shared {aug_desc}")
        return aug_matrix_list, aug_desc_list

    for frame_idx, frame_item in enumerate(frame_items):
        seed = RANDOM_SEED + frame_idx if B_RANDOM_PER_FRAME else RANDOM_SEED
        rng = np.random.default_rng(seed)
        aug_matrix, aug_desc = BuildAugmentationMatrix(CASE_NAME, rng)
        aug_matrix_list.append(aug_matrix)
        aug_desc_list.append(f"{frame_item['token']}: {aug_desc}")

    return aug_matrix_list, aug_desc_list


def BuildOriginalAlignMatrix(pose_source, pose_current):
    ## pose_*: shape [4, 4]，lidar/ego -> world；输出 source 原始 lidar/ego -> current 原始 lidar/ego。
    return (np.linalg.inv(pose_current) @ pose_source).astype(np.float32)


def BuildAugmentedAlignMatrix(pose_source, pose_current, aug_source, aug_current):
    """
    pose_source / pose_current: shape [4, 4]，source/current lidar/ego -> world。
    aug_source / aug_current: shape [4, 4]，原始 lidar/ego -> 增强后 lidar/ego。
    输出: source 增强后 lidar/ego -> current 增强后 lidar/ego。
    """
    align_matrix = aug_current @ np.linalg.inv(pose_current) @ pose_source @ np.linalg.inv(aug_source)
    return align_matrix.astype(np.float32)


def DownsampleForVisual(points_xyz, semantic_label):
    if VOXEL_SIZE <= 0:
        return points_xyz, semantic_label
    points_xyz, idx_selected, _ = voxel_downsample(points_xyz, VOXEL_SIZE, return_mapping=True)
    semantic_label = semantic_label[idx_selected]
    return points_xyz, semantic_label


def BuildMergedPointCloud(frame_items, aug_matrix_list, mode_name):
    current_item = frame_items[CURRENT_FRAME_OFFSET]
    pose_current = current_item["pose"]
    aug_current = aug_matrix_list[CURRENT_FRAME_OFFSET]

    merged_points_list = []
    merged_label_list = []

    for frame_idx, frame_item in enumerate(frame_items):
        points_xyz = frame_item["points_xyz"]
        semantic_label = frame_item["semantic_label"]
        pose_source = frame_item["pose"]
        aug_source = aug_matrix_list[frame_idx]

        if mode_name == "original":
            ## 原始点 p_i 对齐到当前帧原始 lidar/ego 坐标系。
            points_input_xyz = points_xyz
            align_matrix = BuildOriginalAlignMatrix(pose_source, pose_current)
        elif mode_name == "wrong_aug_with_original_pose":
            ## 错误对照：p_i_aug 已在增强坐标系，却仍当成原始坐标去套 inv(T_c) @ T_i。
            points_input_xyz = ApplyTransform(points_xyz, aug_source)
            align_matrix = BuildOriginalAlignMatrix(pose_source, pose_current)
        elif mode_name == "correct_aug":
            ## 正确方案：p_i_aug 对齐到当前帧增强后 lidar/ego 坐标系。
            points_input_xyz = ApplyTransform(points_xyz, aug_source)
            align_matrix = BuildAugmentedAlignMatrix(
                pose_source=pose_source,
                pose_current=pose_current,
                aug_source=aug_source,
                aug_current=aug_current,
            )
        else:
            raise ValueError(f"unknown mode_name: {mode_name}")

        points_aligned_xyz = ApplyTransform(points_input_xyz, align_matrix)
        points_aligned_xyz, semantic_label = DownsampleForVisual(points_aligned_xyz, semantic_label)
        merged_points_list.append(points_aligned_xyz)
        merged_label_list.append(semantic_label)

    merged_points = np.concatenate(merged_points_list, axis=0)
    merged_label = np.concatenate(merged_label_list, axis=0)
    return merged_points, merged_label


def PrintFrameSummary(frame_items):
    print("selected frames:")
    for frame_item in frame_items:
        pose = frame_item["pose"]
        print(
            f"  {frame_item['token']} points={frame_item['points_xyz'].shape[0]} "
            f"world_t=({pose[0, 3]:.2f}, {pose[1, 3]:.2f}, {pose[2, 3]:.2f})"
        )


def PrintAugmentationSummary(aug_desc_list):
    print("\naugmentation matrices:")
    for aug_desc in aug_desc_list:
        print("  " + aug_desc)


def PrintAlignmentError(frame_items, aug_matrix_list):
    current_item = frame_items[CURRENT_FRAME_OFFSET]
    pose_current = current_item["pose"]
    aug_current = aug_matrix_list[CURRENT_FRAME_OFFSET]

    print("\nalignment numeric check:")
    for frame_idx, frame_item in enumerate(frame_items):
        points_xyz = frame_item["points_xyz"]
        pose_source = frame_item["pose"]
        aug_source = aug_matrix_list[frame_idx]

        original_align = BuildOriginalAlignMatrix(pose_source, pose_current)
        aug_align = BuildAugmentedAlignMatrix(
            pose_source=pose_source,
            pose_current=pose_current,
            aug_source=aug_source,
            aug_current=aug_current,
        )

        points_aug_xyz = ApplyTransform(points_xyz, aug_source)
        points_correct_xyz = ApplyTransform(points_aug_xyz, aug_align)
        points_expected_xyz = ApplyTransform(ApplyTransform(points_xyz, original_align), aug_current)
        points_wrong_xyz = ApplyTransform(points_aug_xyz, original_align)

        correct_error = np.linalg.norm(points_correct_xyz - points_expected_xyz, axis=1)
        wrong_error = np.linalg.norm(points_wrong_xyz - points_expected_xyz, axis=1)
        print(
            f"  {frame_item['token']}: "
            f"correct_mean={correct_error.mean():.6f}, correct_max={correct_error.max():.6f}, "
            f"wrong_mean={wrong_error.mean():.3f}, wrong_max={wrong_error.max():.3f}"
        )


def VisualizeMerged(frame_items, aug_matrix_list, mode_name, title):
    print(f"\n## visualize: {title}")
    points, labels = BuildMergedPointCloud(frame_items, aug_matrix_list, mode_name)
    print(f"  merged points shape={points.shape}, labels shape={labels.shape}")
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

    aug_matrix_list, aug_desc_list = BuildAugmentationList(frame_items)

    PrintFrameSummary(frame_items)
    PrintAugmentationSummary(aug_desc_list)
    PrintAlignmentError(frame_items, aug_matrix_list)

    VisualizeMerged(
        frame_items=frame_items,
        aug_matrix_list=aug_matrix_list,
        mode_name="original",
        title="original points aligned to current raw lidar frame",
    )
    VisualizeMerged(
        frame_items=frame_items,
        aug_matrix_list=aug_matrix_list,
        mode_name="wrong_aug_with_original_pose",
        title="wrong: augmented points aligned by original pose only",
    )
    VisualizeMerged(
        frame_items=frame_items,
        aug_matrix_list=aug_matrix_list,
        mode_name="correct_aug",
        title="correct: augmented points aligned by A_c @ inv(T_c) @ T_i @ inv(A_i)",
    )


if __name__ == "__main__":
    main()
