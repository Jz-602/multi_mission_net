import pickle
import sys
import time
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [str(PROJECT_DIR), str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from LYF_train_minkunet_seg_multi import CfgFromYamlFile
from pcd_io import ReadCleanPcdXYZ
from spconv_minkunet_single_file import SpconvMinkUNet
from trainer.utils.checkpoint import load_checkpoint

try:
    from scipy.spatial.transform import Rotation as ScipyRotation
except ImportError:
    ScipyRotation = None


## 只需要改这里。
WAYMO_DATA_ROOT = Path(os.environ["WAYMO_DATA_ROOT"])
SEQUENCE_NAME_LIST: List[str] = []                    # 空列表表示全序列推理；非空时只推理指定 sequence 名。
B_FORCE_RERUN_INFERENCE = True                        # True: label 已存在也重新推理；False: label 完整时跳过。
B_VISUALIZE_AFTER_INFERENCE = False                   # True: 推理结束后打开逐帧可视化。

DEVICE = "cuda"                                       # "cuda" 或 "cpu"；写 cuda 但机器无 GPU 时会自动退回 CPU。
MAX_FRAMES = -1                                       # 每个 sequence 最多推理多少帧；-1 表示全部。
POSE_MAX_TIME_DIFF = 200                              # pcd timestamp 找不到 pose 时，允许匹配最近 pose 的最大时间差。
LOG_EVERY_NFRAMES = 10

CFG_PATH = PROJECT_DIR / "LYF_segmentation_multi.yaml"
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints_multi" / "epoch_48.pth"
INPUT_PSEUDO_LABEL_DIRNAME = "label/pseudo_label_1"   # 输入第 4 维语义先验目录。
OUTPUT_LABEL_NAME = "pseudo_label_2"                  # 输出到 output/label/pseudo_label_2/*.label。


def HandleCudaLaunch() -> None:
    if "LOCAL_RANK" in os.environ or "__internal_launch__" in sys.argv[1:]:
        return

    cuda_id_list: List[str] = []
    b_collect_following_cuda_ids = False
    for argument in sys.argv[1:]:
        if argument == "__internal_launch__":
            continue

        if argument.startswith("CUDA="):
            b_collect_following_cuda_ids = True
            first_cuda_text = argument.split("=", 1)[1].strip()
            if first_cuda_text != "":
                for cuda_id in first_cuda_text.split(","):
                    cuda_id = cuda_id.strip()
                    if cuda_id != "":
                        cuda_id_list.append(cuda_id)
            continue

        if b_collect_following_cuda_ids:
            normalized_argument = argument.replace(",", "").strip()
            if normalized_argument.isdigit():
                cuda_id = argument.strip().strip(",")
                if cuda_id != "":
                    cuda_id_list.append(cuda_id)
                continue
            b_collect_following_cuda_ids = False

    if len(cuda_id_list) == 0:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("[Launch] CUDA not specified, default to GPU 0.", flush=True)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_id_list)

    if len(cuda_id_list) == 1:
        print(f"[Launch] single GPU mode, use GPU {cuda_id_list[0]}.", flush=True)
        return

    print(f"[Launch] multi-GPU inference, use GPUs {','.join(cuda_id_list)}.", flush=True)
    launch_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(cuda_id_list)}",
        str(Path(__file__).resolve()),
        "__internal_launch__",
    ]
    raise SystemExit(subprocess.call(launch_command, env=os.environ.copy()))


@dataclass
class MultiFrameConfig:
    input_frame_count: int
    frame_interval: int
    supervised_window_offset: int


def LoadLabelMapping(cfg: EasyDict) -> Tuple[List[str], np.ndarray]:
    label_mapping_path = PROJECT_DIR / "label_mapping" / cfg.common_variables.label_mapping
    with label_mapping_path.open("r", encoding="utf-8") as yaml_file:
        label_mapping_cfg = yaml.safe_load(yaml_file)

    class_names = list(label_mapping_cfg["CLASS_NAMES"])
    if "VISUALIZATION_COLORS" in label_mapping_cfg:
        color_table = np.asarray(
            [label_mapping_cfg["VISUALIZATION_COLORS"][class_name] for class_name in class_names],
            dtype=np.float64,
        )
    else:
        color_table = np.zeros((len(class_names), 3), dtype=np.float64)
        color_table[:, 0] = np.linspace(0.0, 1.0, len(class_names))
    return class_names, color_table


def LoadConfig(cfg_path: Path) -> Tuple[EasyDict, np.ndarray, MultiFrameConfig]:
    cfg = EasyDict()
    CfgFromYamlFile(str(cfg_path), cfg)
    cfg.common_variables.ignore_index = cfg.model.ignore_index

    class_names, color_table = LoadLabelMapping(cfg)
    cfg.model.class_names = class_names
    cfg.model.num_classes = len(class_names)
    cfg.model.ignore_index = int(cfg.common_variables.ignore_index)
    cfg.model.voxel_size = list(cfg.common_variables.voxel_size)
    cfg.model.point_cloud_range = list(cfg.common_variables.pc_range)

    dataset_cfg = cfg.data.test_dataset if "test_dataset" in cfg.data else cfg.data.val_dataset
    multi_frame_cfg = MultiFrameConfig(
        input_frame_count=int(dataset_cfg.get("input_frame_count", 5)),
        frame_interval=int(dataset_cfg.get("frame_interval", 2)),
        supervised_window_offset=int(dataset_cfg.get("supervised_window_offset", 3)),
    )
    if not 0 <= multi_frame_cfg.supervised_window_offset < multi_frame_cfg.input_frame_count:
        raise ValueError(
            "supervised_window_offset must be in "
            f"[0, {multi_frame_cfg.input_frame_count - 1}], got {multi_frame_cfg.supervised_window_offset}"
        )
    return cfg, color_table, multi_frame_cfg


def BuildRotationXYZ(roll: float, pitch: float, yaw: float) -> np.ndarray:
    if ScipyRotation is not None:
        return ScipyRotation.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()

    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    rot_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_roll, -sin_roll], [0.0, sin_roll, cos_roll]],
        dtype=np.float64,
    )
    rot_y = np.array(
        [[cos_pitch, 0.0, sin_pitch], [0.0, 1.0, 0.0], [-sin_pitch, 0.0, cos_pitch]],
        dtype=np.float64,
    )
    rot_z = np.array(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def BuildPoseMatrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    输入 pose 行含义：timestamp x y z roll pitch yaw。
    输出 pose_lidar_to_world: [4, 4]，当前帧 LiDAR/车体坐标系 -> world 坐标系。
    """
    pose_lidar_to_world = np.eye(4, dtype=np.float32)
    pose_lidar_to_world[:3, :3] = BuildRotationXYZ(roll, pitch, yaw).astype(np.float32)
    pose_lidar_to_world[:3, 3] = np.asarray([x, y, z], dtype=np.float32)
    return pose_lidar_to_world


def LoadPoseDict(pose_path: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    if not pose_path.is_file():
        raise FileNotFoundError(f"pose_path not found: {pose_path}")

    pose_dict: Dict[str, np.ndarray] = {}
    timestamp_list: List[str] = []
    with pose_path.open("r", encoding="utf-8") as file_object:
        for line in file_object:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            core_parts = parts[-7:]
            timestamp = str(core_parts[0])
            try:
                x, y, z = float(core_parts[1]), float(core_parts[2]), float(core_parts[3])
                roll, pitch, yaw = float(core_parts[4]), float(core_parts[5]), float(core_parts[6])
            except ValueError:
                continue
            pose_dict[timestamp] = BuildPoseMatrix(x, y, z, roll, pitch, yaw)
            timestamp_list.append(timestamp)

    if len(pose_dict) == 0:
        raise ValueError(f"No valid pose rows found in {pose_path}")

    pose_time_array = np.asarray([int(timestamp) for timestamp in timestamp_list], dtype=np.int64)
    return pose_dict, pose_time_array, timestamp_list


def FindPoseForTimestamp(
    timestamp: str,
    pose_dict: Dict[str, np.ndarray],
    pose_time_array: np.ndarray,
    timestamp_list: List[str],
) -> Tuple[np.ndarray, str, bool, int]:
    if timestamp in pose_dict:
        return pose_dict[timestamp], timestamp, True, 0

    timestamp_value = int(timestamp)
    nearest_idx = int(np.argmin(np.abs(pose_time_array - timestamp_value)))
    matched_timestamp = timestamp_list[nearest_idx]
    time_diff = int(abs(int(matched_timestamp) - timestamp_value))
    if POSE_MAX_TIME_DIFF >= 0 and time_diff > POSE_MAX_TIME_DIFF:
        raise KeyError(
            f"pose for timestamp={timestamp} not found; nearest={matched_timestamp}, "
            f"diff={time_diff} > {POSE_MAX_TIME_DIFF}"
        )
    return pose_dict[matched_timestamp], matched_timestamp, False, time_diff


def BuildSortKey(path: Path):
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem


def ResolvePosePath(pcd_dir: Path, pose_path: Optional[Path]) -> Path:
    sequence_output_root = pcd_dir.resolve().parent
    if pose_path is not None:
        resolved_pose_path = pose_path.resolve()
        if resolved_pose_path.parent != sequence_output_root:
            raise ValueError(
                f"POSE_PATH must be in the same output dir as PCD_DIR: "
                f"pose_parent={resolved_pose_path.parent}, output_root={sequence_output_root}"
            )
        return resolved_pose_path

    for candidate_name in ["pose_slam_euler.txt", "pose.txt", "poses.txt", "poses_ch.txt"]:
        candidate_path = sequence_output_root / candidate_name
        if candidate_path.is_file():
            return candidate_path
    raise FileNotFoundError(
        f"pose file not found under {sequence_output_root}; "
        "expected pose_slam_euler.txt, pose.txt, poses.txt, or poses_ch.txt"
    )


def TransformPoints(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    points: [N, 4]，输入点所在坐标系由 transform_matrix 输入侧定义；第 4 维 pseudo_label_1 不参与坐标变换。
    transform_matrix: [4, 4]，输入 LiDAR/车体坐标系 -> target frame LiDAR/车体坐标系。
    """
    points_xyz = points[:, :3]
    rotation = transform_matrix[:3, :3]
    translation = transform_matrix[:3, 3].reshape(1, 3)
    transformed_xyz = points_xyz @ rotation.T + translation
    return np.concatenate(
        [transformed_xyz.astype(np.float32, copy=False), points[:, 3:].astype(np.float32, copy=False)],
        axis=1,
    )


class MultiFramePcdDataset(Dataset):
    """
    第二阶段全序列推理 dataset。

    每个样本以一个 target frame 为输出帧：
    - 读取窗口内每帧点云 points_xyz_lidar: [N, 3]，坐标系为该帧 LiDAR/车体坐标系。
    - 读取每帧 pseudo_label_1: [N]，uint32，点顺序与对应 PCD 一致。
    - 使用 pose 将上下文帧点云对齐到 target frame LiDAR/车体坐标系。
    - 输出 points: [sum_N, 4]，第 0:3 列为 target frame 坐标系 xyz，第 4 列为 pseudo_label_1。
    """

    def __init__(self, pcd_dir: Path, pose_path: Path, multi_frame_cfg: MultiFrameConfig) -> None:
        self.pcd_dir = pcd_dir.resolve()
        if not self.pcd_dir.is_dir():
            raise FileNotFoundError(f"PCD_DIR not found: {self.pcd_dir}")

        self.pose_path = pose_path.resolve()
        self.output_root = self.pcd_dir.parent
        self.sequence_root = self.output_root.parent
        self.sequence_name = self.sequence_root.name
        self.multi_frame_cfg = multi_frame_cfg

        pose_dict, pose_time_array, timestamp_list = LoadPoseDict(self.pose_path)
        pcd_path_list = sorted(self.pcd_dir.glob("*.pcd"), key=BuildSortKey)
        if MAX_FRAMES >= 0:
            pcd_path_list = pcd_path_list[:MAX_FRAMES]
        if len(pcd_path_list) == 0:
            raise ValueError(f"No .pcd files found in {self.pcd_dir}")

        self.infos: List[Dict[str, Any]] = []
        for frame_idx, pcd_path in enumerate(pcd_path_list):
            source_timestamp = pcd_path.stem
            pose_lidar_to_world, matched_pose_timestamp, b_pose_exact, pose_time_diff = FindPoseForTimestamp(
                timestamp=source_timestamp,
                pose_dict=pose_dict,
                pose_time_array=pose_time_array,
                timestamp_list=timestamp_list,
            )
            self.infos.append(
                {
                    "token": f"{self.sequence_name}-{source_timestamp}",
                    "sequence_name": self.sequence_name,
                    "frame_idx": frame_idx,
                    "source_timestamp": source_timestamp,
                    "pcd_path": str(pcd_path),
                    "pose_lidar_to_world": pose_lidar_to_world.astype(np.float32),
                    "matched_pose_timestamp": matched_pose_timestamp,
                    "b_pose_exact": bool(b_pose_exact),
                    "pose_time_diff": int(pose_time_diff),
                }
            )

    def __len__(self) -> int:
        return len(self.infos)

    def GetLegalSupervisedOffsetRange(self, target_frame_idx: int) -> Tuple[int, int]:
        sequence_len = len(self.infos)
        min_offset = self.multi_frame_cfg.input_frame_count - 1 - (
            sequence_len - 1 - target_frame_idx
        ) // self.multi_frame_cfg.frame_interval
        max_offset = target_frame_idx // self.multi_frame_cfg.frame_interval
        min_offset = max(0, int(min_offset))
        max_offset = min(self.multi_frame_cfg.input_frame_count - 1, int(max_offset))
        return min_offset, max_offset

    def SelectSupervisedWindowOffset(self, target_frame_idx: int) -> int:
        min_offset, max_offset = self.GetLegalSupervisedOffsetRange(target_frame_idx)
        if min_offset <= max_offset:
            return int(np.clip(self.multi_frame_cfg.supervised_window_offset, min_offset, max_offset))
        return self.multi_frame_cfg.supervised_window_offset

    def BuildFrameIndexList(self, target_frame_idx: int, supervised_window_offset: int) -> Tuple[List[int], bool]:
        frame_idx_list = []
        b_window_padded = False
        sequence_len = len(self.infos)
        for frame_offset in range(self.multi_frame_cfg.input_frame_count):
            relative_offset = (frame_offset - supervised_window_offset) * self.multi_frame_cfg.frame_interval
            raw_frame_idx = target_frame_idx + relative_offset
            frame_idx = int(np.clip(raw_frame_idx, 0, sequence_len - 1))
            if frame_idx != raw_frame_idx:
                b_window_padded = True
            frame_idx_list.append(frame_idx)
        return frame_idx_list, b_window_padded

    def LoadFramePoints(self, info: Dict[str, Any]) -> np.ndarray:
        points_xyz_lidar = ReadCleanPcdXYZ(info["pcd_path"])
        if points_xyz_lidar.ndim != 2 or points_xyz_lidar.shape[1] != 3:
            raise ValueError(f"points in {info['pcd_path']} must be [N, 3], got {points_xyz_lidar.shape}")
        if points_xyz_lidar.shape[0] == 0:
            raise ValueError(f"Empty point cloud after finite filtering: {info['pcd_path']}")

        pseudo_label_path = self.output_root / INPUT_PSEUDO_LABEL_DIRNAME / f"{info['source_timestamp']}.label"
        pseudo_label = np.fromfile(pseudo_label_path, dtype=np.uint32)
        if pseudo_label.shape[0] != points_xyz_lidar.shape[0]:
            raise ValueError(
                f"pseudo_label_1 count mismatch for {info['token']}: "
                f"points={points_xyz_lidar.shape[0]} labels={pseudo_label.shape[0]} path={pseudo_label_path}"
            )
        return np.concatenate(
            [
                points_xyz_lidar.astype(np.float32, copy=False),
                pseudo_label.astype(np.float32).reshape(-1, 1),
            ],
            axis=1,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        target_info = self.infos[idx]
        target_pose = target_info["pose_lidar_to_world"]
        world_to_target = np.linalg.inv(target_pose).astype(np.float32)
        supervised_window_offset = self.SelectSupervisedWindowOffset(idx)
        frame_idx_list, b_window_padded = self.BuildFrameIndexList(idx, supervised_window_offset)

        points_list = []
        target_start = 0
        target_end = 0
        for frame_offset, frame_idx in enumerate(frame_idx_list):
            frame_info = self.infos[frame_idx]
            frame_points = self.LoadFramePoints(frame_info)
            lidar_to_target = world_to_target @ frame_info["pose_lidar_to_world"]
            points_target = TransformPoints(frame_points, lidar_to_target)
            if frame_offset == supervised_window_offset:
                target_start = sum(points.shape[0] for points in points_list)
                target_end = target_start + points_target.shape[0]
            points_list.append(points_target)

        points = np.concatenate(points_list, axis=0).astype(np.float32, copy=False)
        return {
            "token": target_info["token"],
            "points": points,
            "target_start": np.asarray([target_start], dtype=np.int64),
            "target_end": np.asarray([target_end], dtype=np.int64),
            "target_point_count": np.asarray([target_end - target_start], dtype=np.int64),
            "frame_indices": np.asarray(frame_idx_list, dtype=np.int32),
            "target_frame_index": np.asarray([idx], dtype=np.int32),
            "supervised_window_offset": np.asarray([supervised_window_offset], dtype=np.int32),
            "b_window_padded": np.asarray([b_window_padded], dtype=np.bool_),
        }

    @staticmethod
    def CollateBatch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch_list) != 1:
            raise ValueError("MultiFramePcdDataset inference only supports batch_size=1.")
        item = batch_list[0]
        points_with_batch = np.pad(item["points"], ((0, 0), (1, 0)), mode="constant", constant_values=0)
        return {
            "token": [item["token"]],
            "points": torch.tensor(points_with_batch, dtype=torch.float32),
            "target_start": torch.tensor(item["target_start"], dtype=torch.long),
            "target_end": torch.tensor(item["target_end"], dtype=torch.long),
            "target_point_count": torch.tensor(item["target_point_count"], dtype=torch.long),
            "frame_indices": torch.tensor(item["frame_indices"], dtype=torch.int32),
            "target_frame_index": torch.tensor(item["target_frame_index"], dtype=torch.int32),
            "supervised_window_offset": torch.tensor(item["supervised_window_offset"], dtype=torch.int32),
            "b_window_padded": torch.tensor(item["b_window_padded"], dtype=torch.bool),
        }


def MoveBatchToDevice(data_batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    batch_dict: Dict[str, Any] = {}
    for key, value in data_batch.items():
        if key == "token":
            batch_dict[key] = value
        elif isinstance(value, torch.Tensor):
            batch_dict[key] = value.to(device, non_blocking=False)
        else:
            batch_dict[key] = value
    return batch_dict


def BuildModel(cfg: EasyDict, device: torch.device) -> torch.nn.Module:
    model = SpconvMinkUNet(cfg.model).to(device)
    load_checkpoint(model, str(CHECKPOINT_PATH), map_location=device, strict=True)
    model.eval()
    return model


def BuildTargetFramePrediction(batch_dict: Dict[str, Any], forward_dict: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    batch_dict["points"]: [sum_N, 5]，第 0 列 batch_idx，第 1:4 列为 target frame 坐标系 xyz。
    输出 target_point_semantic_pred: [N_target]，只与 target frame 原始 PCD 点顺序对齐；范围外点填 0。
    """
    target_start = int(batch_dict["target_start"][0].item())
    target_end = int(batch_dict["target_end"][0].item())
    target_point_count = int(batch_dict["target_point_count"][0].item())
    b_valid_mask = forward_dict["b_valid_mask"].detach().cpu().numpy().astype(bool)
    point_logits = forward_dict["point_logits"]
    valid_point_semantic_pred = point_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)

    all_point_semantic_pred = np.zeros((b_valid_mask.shape[0],), dtype=np.int32)
    all_point_semantic_pred[b_valid_mask] = valid_point_semantic_pred
    target_point_semantic_pred = all_point_semantic_pred[target_start:target_end]
    if target_point_semantic_pred.shape[0] != target_point_count:
        raise ValueError(
            f"target prediction count mismatch: pred={target_point_semantic_pred.shape[0]} target={target_point_count}"
        )
    return target_point_semantic_pred, valid_point_semantic_pred


def BuildOutputDirs(sequence_output_root: Path) -> Dict[str, Path]:
    output_dirs = {
        "label": sequence_output_root / "label" / OUTPUT_LABEL_NAME,
        "detection": sequence_output_root / "detection" / OUTPUT_LABEL_NAME,
    }
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dirs


def HasCompleteLabels(dataset: MultiFramePcdDataset, label_dir: Path) -> bool:
    label_count = sum((label_dir / f"{info['source_timestamp']}.label").is_file() for info in dataset.infos)
    b_complete = label_count == len(dataset)
    print(f"[Check] labels={label_count}/{len(dataset)} complete={b_complete}", flush=True)
    return b_complete


def WriteDetectionNote(detection_dir: Path) -> None:
    note_path = detection_dir / "README.txt"
    note_path.write_text(
        "Second-stage multi-frame semantic model only outputs point-wise segmentation labels.\n"
        "This detection directory is reserved for future detection results.\n",
        encoding="utf-8",
    )


def RunInference(
    dataset: MultiFramePcdDataset,
    cfg: EasyDict,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    output_dirs = BuildOutputDirs(dataset.output_root)
    if not B_FORCE_RERUN_INFERENCE and HasCompleteLabels(dataset, output_dirs["label"]):
        print(f"[Skip] complete labels already exist: {output_dirs['label']}", flush=True)
        return

    frame_summaries: List[Dict[str, Any]] = []
    infer_start_time = time.perf_counter()
    with torch.no_grad():
        for frame_idx in range(len(dataset)):
            info = dataset.infos[frame_idx]
            data_batch = dataset.CollateBatch([dataset[frame_idx]])
            batch_dict = MoveBatchToDevice(data_batch, device)
            forward_dict = model(batch_dict, b_return_details=True)
            target_pred, valid_pred = BuildTargetFramePrediction(batch_dict, forward_dict)

            label_path = output_dirs["label"] / f"{info['source_timestamp']}.label"
            target_pred.astype(np.uint32).tofile(label_path)

            class_hist = np.bincount(target_pred.astype(np.int64), minlength=int(cfg.model.num_classes))
            valid_class_hist = np.bincount(valid_pred.astype(np.int64), minlength=int(cfg.model.num_classes))
            frame_summaries.append(
                {
                    "token": info["token"],
                    "frame_idx": int(info["frame_idx"]),
                    "source_timestamp": info["source_timestamp"],
                    "pcd_path": info["pcd_path"],
                    "label_path": str(label_path),
                    "point_count": int(target_pred.shape[0]),
                    "class_hist": class_hist.astype(np.int64).tolist(),
                    "valid_class_hist": valid_class_hist.astype(np.int64).tolist(),
                    "frame_indices": batch_dict["frame_indices"].detach().cpu().numpy().astype(np.int32).tolist(),
                    "target_frame_index": int(batch_dict["target_frame_index"][0].item()),
                    "supervised_window_offset": int(batch_dict["supervised_window_offset"][0].item()),
                    "b_window_padded": bool(batch_dict["b_window_padded"][0].item()),
                    "matched_pose_timestamp": info["matched_pose_timestamp"],
                    "b_pose_exact": bool(info["b_pose_exact"]),
                    "pose_time_diff": int(info["pose_time_diff"]),
                }
            )

            b_log_frame = (
                frame_idx == 0
                or frame_idx + 1 == len(dataset)
                or (LOG_EVERY_NFRAMES > 0 and (frame_idx + 1) % LOG_EVERY_NFRAMES == 0)
            )
            if b_log_frame:
                print(
                    f"[Infer:{OUTPUT_LABEL_NAME}] {frame_idx + 1}/{len(dataset)} "
                    f"timestamp={info['source_timestamp']} target_points={target_pred.shape[0]} "
                    f"label={label_path}",
                    flush=True,
                )

    infer_elapsed_time = time.perf_counter() - infer_start_time
    prediction_index_path = output_dirs["label"] / "sequence_predictions.pkl"
    with prediction_index_path.open("wb") as file_object:
        pickle.dump(
            {
                "metadata": {
                    "model_name": OUTPUT_LABEL_NAME,
                    "model_type": "multi_frame",
                    "cfg": str(CFG_PATH),
                    "checkpoint": str(CHECKPOINT_PATH),
                    "pcd_dir": str(dataset.pcd_dir),
                    "pose_path": str(dataset.pose_path),
                    "input_pseudo_label_dirname": INPUT_PSEUDO_LABEL_DIRNAME,
                    "label_dir": str(output_dirs["label"]),
                    "detection_dir": str(output_dirs["detection"]),
                    "input_frame_count": int(dataset.multi_frame_cfg.input_frame_count),
                    "frame_interval": int(dataset.multi_frame_cfg.frame_interval),
                    "supervised_window_offset": int(dataset.multi_frame_cfg.supervised_window_offset),
                    "num_frames": len(frame_summaries),
                    "label_note": "Each .label is uint32 and aligned with the cleaned target PCD point order; out-of-range points are 0.",
                    "infer_elapsed_time_sec": float(infer_elapsed_time),
                    "infer_fps": float(len(frame_summaries) / max(infer_elapsed_time, 1e-6)),
                },
                "frames": frame_summaries,
            },
            file_object,
        )
    WriteDetectionNote(output_dirs["detection"])
    print(
        f"[Done:{OUTPUT_LABEL_NAME}] frames={len(frame_summaries)} "
        f"time={infer_elapsed_time:.2f}s labels={output_dirs['label']} detection={output_dirs['detection']}",
        flush=True,
    )


def FindSequencePcdDirs(data_root: Path) -> List[Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"WAYMO_DATA_ROOT not found: {data_root}")

    if len(SEQUENCE_NAME_LIST) > 0:
        pcd_dir_list = [data_root / sequence_name / "output" / "pcd" for sequence_name in SEQUENCE_NAME_LIST]
    else:
        pcd_dir_list = sorted(data_root.glob("*/output/pcd"), key=lambda path: path.parent.parent.name)

    valid_pcd_dir_list = [pcd_dir for pcd_dir in pcd_dir_list if pcd_dir.is_dir()]
    if len(valid_pcd_dir_list) == 0:
        raise FileNotFoundError(f"No sequence pcd dirs found under {data_root}")
    return valid_pcd_dir_list


def Main() -> None:
    cfg, _, multi_frame_cfg = LoadConfig(CFG_PATH)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    b_distributed_inference = world_size > 1

    if DEVICE == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        if b_distributed_inference:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")

    if DEVICE == "cuda" and device.type == "cpu":
        print("[Infer] CUDA is not available, fallback to CPU.", flush=True)

    model = BuildModel(cfg, device)
    pcd_dir_list = FindSequencePcdDirs(WAYMO_DATA_ROOT.resolve())
    rank_pcd_dir_list = pcd_dir_list[rank::world_size]
    print(
        f"[Sequences] rank={rank}/{world_size} local_rank={local_rank} device={device} "
        f"model={OUTPUT_LABEL_NAME} total={len(pcd_dir_list)} assigned={len(rank_pcd_dir_list)} "
        f"root={WAYMO_DATA_ROOT.resolve()} input_frames={multi_frame_cfg.input_frame_count} "
        f"interval={multi_frame_cfg.frame_interval} offset={multi_frame_cfg.supervised_window_offset}",
        flush=True,
    )

    for sequence_idx, pcd_dir in enumerate(rank_pcd_dir_list):
        pose_path = ResolvePosePath(pcd_dir.resolve(), None)
        dataset = MultiFramePcdDataset(
            pcd_dir=pcd_dir,
            pose_path=pose_path,
            multi_frame_cfg=multi_frame_cfg,
        )
        print(
            f"[Dataset] rank={rank} {sequence_idx + 1}/{len(rank_pcd_dir_list)} model={OUTPUT_LABEL_NAME} "
            f"frames={len(dataset)} pcd_dir={dataset.pcd_dir} pose={dataset.pose_path}",
            flush=True,
        )

        RunInference(dataset, cfg, model, device)

        if B_VISUALIZE_AFTER_INFERENCE:
            if b_distributed_inference:
                print("[Visualize] skip visualization in multi-GPU inference.", flush=True)
                continue
            import visualize_label as visualize_script

            visualize_script.PCD_DIR = dataset.pcd_dir
            visualize_script.MODEL_NAME = OUTPUT_LABEL_NAME
            visualize_script.LABEL_DIR = dataset.output_root / "label" / OUTPUT_LABEL_NAME
            visualize_script.Main()


if __name__ == "__main__":
    HandleCudaLaunch()
    Main()
