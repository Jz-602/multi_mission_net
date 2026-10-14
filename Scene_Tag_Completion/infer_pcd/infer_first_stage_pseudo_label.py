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
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [str(PROJECT_DIR), str(REPO_ROOT), str(SCRIPT_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from LYF_train_minkunet_seg_xyzl import CfgFromYamlFile
from WaymoDatasetV2 import WaymoDataset
from pcd_io import ReadCleanPcdXYZ
from spconv_minkunet_memory import SpconvMinkUNet as MemorySpconvMinkUNet
from spconv_minkunet_single_file import SpconvMinkUNet as SingleSpconvMinkUNet
from trainer.utils.checkpoint import load_checkpoint

try:
    from scipy.spatial.transform import Rotation as ScipyRotation
except ImportError:
    ScipyRotation = None


## 只需要改这里。
# 数据目录格式建议：
# WAYMO_DATA_ROOT/
#   sequence_name/
#     output/
#       pcd/        # 输入 .pcd 点云，按文件名排序逐帧推理。
#       pose_slam_euler.txt
#       label/      # 本脚本输出 label/pseudo_label_1/*.label。
#       detection/  # 预留检测输出；当前 MEMORY_waymo 只做语义分割，会写一个说明文件。
#
## pose 文件格式：
#   每行至少 7 列，脚本读取最后 7 列：
#   timestamp x y z roll pitch yaw
#   timestamp 要与 pcd 文件名 stem 对齐，例如 1641977312551.pcd 对应 timestamp=1641977312551。
#   x/y/z 是 world 坐标系下的 LiDAR/车体位移，单位 m。
#   roll/pitch/yaw 是弧度，按 xyz 欧拉角构造当前帧 LiDAR/车体坐标系 -> world 的 pose。

WAYMO_DATA_ROOT = Path(os.environ["WAYMO_DATA_ROOT"])  # Waymo 数据根目录；脚本会遍历其中所有 */output/pcd。
SEQUENCE_NAME_LIST: List[str] = []                     # 空列表表示全序列推理；非空时只推理指定 sequence 名。

RUN_MODEL = "single"                                # 推理模型："memory" 使用多帧 memory 模型；"single" 使用单帧模型。
B_FORCE_RERUN_INFERENCE = True                      # True: 不管 label 是否已存在，都重新推理一遍；False: label 完整时跳过推理。
B_VISUALIZE_AFTER_INFERENCE = False                 # True: 推理结束后自动打开逐帧语义可视化；False: 只保存结果不显示。

DEVICE = "cuda"                                     # "cuda" 或 "cpu"；写 cuda 但机器无 GPU 时会自动退回 CPU。
MAX_FRAMES = -1                                     # 每个 sequence 最多推理多少帧；-1 表示读取该 sequence 下所有 .pcd。
POSE_MAX_TIME_DIFF = 200                            # pcd timestamp 找不到完全匹配 pose 时，允许匹配最近 pose 的最大时间差；<0 表示不限制。
NUM_WORKERS = 0                                     # DataLoader worker 数；PCD 顺序推理通常 0 最稳。
LOG_EVERY_NFRAMES = 10                              # 每隔多少帧打印一次推理进度；0 表示只打印首尾。

SINGLE_CFG_PATH = PROJECT_DIR / "LYF_segmentation_xyzl.yaml"  # 单帧 xyzl 模型配置。
SINGLE_CHECKPOINT_PATH = PROJECT_DIR / "checkpoints_xyzl" / "epoch_60.pth"  # 单帧 xyzl 模型权重。
MEMORY_CFG_PATH = PROJECT_DIR / "LYF_segmentation_multi.yaml"  # memory 模型配置。
MEMORY_CHECKPOINT_PATH = PROJECT_DIR / "checkpoints_multi" / "epoch_46.pth"  # memory 模型权重。
SEMANTIC_PRIOR_DIRNAME = "sequence_SemanticPrior"    # xyzl 第 4 维输入特征目录，label 为 uint16 训练标签空间。


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
class ModelSpec:
    name: str
    model_type: str
    cfg_path: Path
    checkpoint_path: Path


def BuildModelSpec() -> ModelSpec:
    if RUN_MODEL == "single":
        return ModelSpec(
            name="pseudo_label_1",
            model_type="single",
            cfg_path=SINGLE_CFG_PATH,
            checkpoint_path=SINGLE_CHECKPOINT_PATH,
        )
    if RUN_MODEL == "memory":
        return ModelSpec(
            name="memory",
            model_type="memory",
            cfg_path=MEMORY_CFG_PATH,
            checkpoint_path=MEMORY_CHECKPOINT_PATH,
        )
    raise ValueError(f'RUN_MODEL must be "memory" or "single", got {RUN_MODEL}')


def ResolvePosePath(pcd_dir: Path, pose_path: Optional[Path]) -> Path:
    sequence_root = pcd_dir.resolve().parent
    if pose_path is not None:
        resolved_pose_path = pose_path.resolve()
        if resolved_pose_path.parent != sequence_root:
            raise ValueError(
                f"POSE_PATH must be in the same sequence_root as PCD_DIR: "
                f"pose_parent={resolved_pose_path.parent}, sequence_root={sequence_root}"
            )
        return resolved_pose_path

    for candidate_name in ["pose_slam_euler.txt", "pose.txt", "poses.txt", "poses_ch.txt"]:
        candidate_path = sequence_root / candidate_name
        if candidate_path.is_file():
            return candidate_path
    raise FileNotFoundError(
        f"pose file not found under {sequence_root}; expected pose_slam_euler.txt, pose.txt, poses.txt, or poses_ch.txt"
    )


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


def LoadConfig(cfg_path: Path) -> Tuple[EasyDict, np.ndarray]:
    cfg = EasyDict()
    CfgFromYamlFile(str(cfg_path), cfg)
    cfg.common_variables.ignore_index = cfg.model.ignore_index

    class_names, color_table = LoadLabelMapping(cfg)
    cfg.model.class_names = class_names
    cfg.model.num_classes = len(class_names)
    cfg.model.ignore_index = int(cfg.common_variables.ignore_index)
    cfg.model.voxel_size = list(cfg.common_variables.voxel_size)
    cfg.model.point_cloud_range = list(cfg.common_variables.pc_range)
    return cfg, color_table


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


class CustomPcdPoseDataset(Dataset):
    """
    自建 PCD 序列 dataset。

    __getitem__ 输出：
    - points: [N, 3]，当前帧 LiDAR/车体坐标系 xyz，单位 m。
    - pose: [4, 4]，当前帧 LiDAR/车体坐标系 -> world。
    - pose_to_prev: [4, 4]，上一帧 LiDAR/车体坐标系 -> 当前帧 LiDAR/车体坐标系。
    """

    def __init__(self, pcd_dir: Path, pose_path: Path) -> None:
        self.pcd_dir = pcd_dir.resolve()
        if not self.pcd_dir.is_dir():
            raise FileNotFoundError(f"PCD_DIR not found: {self.pcd_dir}")

        self.pose_path = pose_path.resolve()
        self.sequence_root = self.pcd_dir.parent
        self.sequence_name = self.sequence_root.name
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
            if frame_idx == 0:
                pose_to_prev = np.eye(4, dtype=np.float32)
            else:
                prev_pose_lidar_to_world = self.infos[-1]["pose_lidar_to_world"]
                pose_to_prev = np.linalg.inv(pose_lidar_to_world) @ prev_pose_lidar_to_world
                pose_to_prev = pose_to_prev.astype(np.float32)

            self.infos.append(
                {
                    "token": f"{self.sequence_name}-{source_timestamp}",
                    "sequence_name": self.sequence_name,
                    "frame_idx": frame_idx,
                    "frame_timestamp_micros": int(source_timestamp),
                    "source_timestamp": source_timestamp,
                    "pcd_path": str(pcd_path),
                    "pose_lidar_to_world": pose_lidar_to_world.astype(np.float32),
                    "pose_to_prev": pose_to_prev,
                    "matched_pose_timestamp": matched_pose_timestamp,
                    "b_pose_exact": bool(b_pose_exact),
                    "pose_time_diff": int(pose_time_diff),
                }
            )

    def __len__(self) -> int:
        return len(self.infos)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.infos[idx]
        points_xyz_lidar = ReadCleanPcdXYZ(info["pcd_path"])
        if points_xyz_lidar.ndim != 2 or points_xyz_lidar.shape[1] != 3:
            raise ValueError(
                f"points in {info['pcd_path']} must be [N, 3], got {points_xyz_lidar.shape}"
            )
        if points_xyz_lidar.shape[0] == 0:
            raise ValueError(f"Empty point cloud after finite filtering: {info['pcd_path']}")

        if RUN_MODEL == "single":
            semantic_prior_path = self.sequence_root / SEMANTIC_PRIOR_DIRNAME / f"{info['source_timestamp']}.label"
            semantic_prior_label = np.fromfile(semantic_prior_path, dtype=np.uint16)
            if semantic_prior_label.shape[0] != points_xyz_lidar.shape[0]:
                raise ValueError(
                    f"semantic prior count mismatch for {info['token']}: "
                    f"points={points_xyz_lidar.shape[0]} prior={semantic_prior_label.shape[0]} "
                    f"path={semantic_prior_path}"
                )
            ## points: [N, 4]，当前帧 LiDAR/车体坐标系 xyz + 传播语义先验 label。
            points_xyz_lidar = np.concatenate(
                [
                    points_xyz_lidar.astype(np.float32, copy=False),
                    semantic_prior_label.astype(np.float32).reshape(-1, 1),
                ],
                axis=1,
            )

        return {
            "token": info["token"],
            "points": points_xyz_lidar,
            "pose": info["pose_lidar_to_world"],
            "pose_to_prev": info["pose_to_prev"],
            "frame_idx": np.asarray([info["frame_idx"]], dtype=np.int32),
        }

    @staticmethod
    def CollateBatch(batch_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        return WaymoDataset.CollateBatch(batch_list)


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


def DetachMemory(memory: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if memory is None:
        return None
    memory_detached: Dict[str, Any] = {}
    for key, value in memory.items():
        memory_detached[key] = value.detach() if isinstance(value, torch.Tensor) else value
    return memory_detached


def BuildModel(spec: ModelSpec, cfg: EasyDict, device: torch.device):
    if spec.model_type == "single":
        model = SingleSpconvMinkUNet(cfg.model).to(device)
    elif spec.model_type == "memory":
        model = MemorySpconvMinkUNet(cfg.model).to(device)
    else:
        raise ValueError(f"Unsupported model_type={spec.model_type}")

    load_checkpoint(model, str(spec.checkpoint_path), map_location=device, strict=True)
    model.eval()
    return model


def BuildFullFramePrediction(
    batch_dict: Dict[str, Any],
    forward_dict: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    batch_dict["points"]: [N, 4]，第 0 列 batch_idx，第 1:4 列为当前帧 LiDAR/车体坐标系 xyz。
    输出 point_semantic_pred: [N]，与原始 points 对齐；训练范围外点填 0。
    """
    points_xyz_lidar = batch_dict["points"][:, 1:4].detach().cpu().numpy().astype(np.float32)
    b_valid_mask = forward_dict["b_valid_mask"].detach().cpu().numpy().astype(bool)
    point_logits = forward_dict["point_logits"]
    valid_point_semantic_pred = point_logits.argmax(dim=1).detach().cpu().numpy().astype(np.int32)

    point_semantic_pred = np.zeros((points_xyz_lidar.shape[0],), dtype=np.int32)
    point_semantic_pred[b_valid_mask] = valid_point_semantic_pred
    return points_xyz_lidar, point_semantic_pred, valid_point_semantic_pred


def BuildOutputDirs(sequence_root: Path, spec: ModelSpec) -> Dict[str, Path]:
    output_dirs = {
        "label": sequence_root / "label" / spec.name,
        "detection": sequence_root / "detection" / spec.name,
    }
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dirs


def HasCompleteLabels(dataset: CustomPcdPoseDataset, label_dir: Path) -> bool:
    label_count = sum((label_dir / f"{info['source_timestamp']}.label").is_file() for info in dataset.infos)
    b_complete = label_count == len(dataset)
    print(f"[Check] labels={label_count}/{len(dataset)} complete={b_complete}", flush=True)
    return b_complete


def WriteDetectionNote(detection_dir: Path) -> None:
    note_path = detection_dir / "README.txt"
    note_path.write_text(
        "MEMORY_waymo semantic model only outputs point-wise segmentation labels.\n"
        "This detection directory is reserved for future detection results.\n",
        encoding="utf-8",
    )


def RunInference(
    dataset: CustomPcdPoseDataset,
    spec: ModelSpec,
    cfg: EasyDict,
    model: torch.nn.Module,
    device: torch.device,
) -> None:
    output_dirs = BuildOutputDirs(dataset.sequence_root, spec)
    if not B_FORCE_RERUN_INFERENCE and HasCompleteLabels(dataset, output_dirs["label"]):
        print(f"[Skip] complete labels already exist: {output_dirs['label']}", flush=True)
        return

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=dataset.CollateBatch,
        pin_memory=False,
    )

    memory = None
    frame_summaries: List[Dict[str, Any]] = []
    infer_start_time = time.perf_counter()
    with torch.no_grad():
        for frame_idx, data_batch in enumerate(dataloader):
            info = dataset.infos[frame_idx]
            batch_dict = MoveBatchToDevice(data_batch, device)

            if spec.model_type == "memory":
                forward_dict = model(batch_dict, memory=memory, b_return_details=True)
                memory = DetachMemory(forward_dict["memory"])
            else:
                forward_dict = model(batch_dict, b_return_details=True)

            points_xyz_lidar, point_semantic_pred, valid_pred = BuildFullFramePrediction(
                batch_dict=batch_dict,
                forward_dict=forward_dict,
            )
            label_path = output_dirs["label"] / f"{info['source_timestamp']}.label"
            point_semantic_pred.astype(np.uint32).tofile(label_path)

            class_hist = np.bincount(
                point_semantic_pred.astype(np.int64),
                minlength=int(cfg.model.num_classes),
            )
            valid_class_hist = np.bincount(
                valid_pred.astype(np.int64),
                minlength=int(cfg.model.num_classes),
            )
            frame_summaries.append(
                {
                    "token": info["token"],
                    "frame_idx": int(info["frame_idx"]),
                    "source_timestamp": info["source_timestamp"],
                    "pcd_path": info["pcd_path"],
                    "label_path": str(label_path),
                    "point_count": int(points_xyz_lidar.shape[0]),
                    "class_hist": class_hist.astype(np.int64).tolist(),
                    "valid_class_hist": valid_class_hist.astype(np.int64).tolist(),
                    "pose_lidar_to_world": info["pose_lidar_to_world"],
                    "pose_to_prev": info["pose_to_prev"],
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
                    f"[Infer:{spec.name}] {frame_idx + 1}/{len(dataset)} "
                    f"timestamp={info['source_timestamp']} points={points_xyz_lidar.shape[0]} "
                    f"label={label_path}",
                    flush=True,
                )

    infer_elapsed_time = time.perf_counter() - infer_start_time
    prediction_index_path = output_dirs["label"] / "sequence_predictions.pkl"
    with prediction_index_path.open("wb") as file_object:
        pickle.dump(
            {
                "metadata": {
                    "model_name": spec.name,
                    "model_type": spec.model_type,
                    "cfg": str(spec.cfg_path),
                    "checkpoint": str(spec.checkpoint_path),
                    "pcd_dir": str(dataset.pcd_dir),
                    "pose_path": str(dataset.pose_path),
                    "label_dir": str(output_dirs["label"]),
                    "detection_dir": str(output_dirs["detection"]),
                    "num_frames": len(frame_summaries),
                    "label_note": "Each .label is uint32 and aligned with the cleaned PCD point order; out-of-range points are 0.",
                    "infer_elapsed_time_sec": float(infer_elapsed_time),
                    "infer_fps": float(len(frame_summaries) / max(infer_elapsed_time, 1e-6)),
                },
                "frames": frame_summaries,
            },
            file_object,
        )
    WriteDetectionNote(output_dirs["detection"])
    print(
        f"[Done:{spec.name}] frames={len(frame_summaries)} "
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
    spec = BuildModelSpec()
    cfg, _ = LoadConfig(spec.cfg_path)
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

    model = BuildModel(spec, cfg, device)
    pcd_dir_list = FindSequencePcdDirs(WAYMO_DATA_ROOT.resolve())
    rank_pcd_dir_list = pcd_dir_list[rank::world_size]
    print(
        f"[Sequences] rank={rank}/{world_size} local_rank={local_rank} device={device} "
        f"model={spec.name} total={len(pcd_dir_list)} assigned={len(rank_pcd_dir_list)} "
        f"root={WAYMO_DATA_ROOT.resolve()}",
        flush=True,
    )

    for sequence_idx, pcd_dir in enumerate(rank_pcd_dir_list):
        pose_path = ResolvePosePath(pcd_dir.resolve(), None)
        dataset = CustomPcdPoseDataset(pcd_dir=pcd_dir, pose_path=pose_path)
        print(
            f"[Dataset] rank={rank} {sequence_idx + 1}/{len(rank_pcd_dir_list)} model={spec.name} "
            f"frames={len(dataset)} pcd_dir={dataset.pcd_dir} pose={dataset.pose_path}",
            flush=True,
        )

        RunInference(dataset, spec, cfg, model, device)

        if B_VISUALIZE_AFTER_INFERENCE:
            if b_distributed_inference:
                print("[Visualize] skip visualization in multi-GPU inference.", flush=True)
                continue
            import visualize_label as visualize_script

            visualize_script.PCD_DIR = dataset.pcd_dir
            visualize_script.MODEL_NAME = spec.name
            visualize_script.LABEL_DIR = dataset.sequence_root / "label" / spec.name
            visualize_script.Main()


if __name__ == "__main__":
    HandleCudaLaunch()
    Main()
