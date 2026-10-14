import pickle
import sys
import time
import os
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

from LYF_train_ODSS import cfg_from_yaml_file
from WaymoDatasetV2 import WAYMO_INT_TO_TYPE_DICT, WaymoDataset, label_to_type
from multi_mission_model import MultiMissionModel
from pcd_io import ReadCleanPcdXYZ
from trainer.utils.checkpoint import load_checkpoint

try:
    from scipy.spatial.transform import Rotation as ScipyRotation
except ImportError:
    ScipyRotation = None


## 只需要改这里。
# 目录格式建议：
# sequence_root/
#   pcd/        # 输入 .pcd 点云，按文件名排序逐帧推理。
#   pose.txt    # 位姿文件；POSE_PATH 和 PCD_DIR 必须在同一个 sequence_root 下面。
#   label/      # 本脚本输出 label/<MODEL_NAME>/*.label。
#   detection/  # 本脚本输出 detection/<MODEL_NAME>/*.pkl。
#
## pose 文件格式：
#   每行至少 7 列，脚本读取最后 7 列：
#   timestamp x y z roll pitch yaw
#   timestamp 要与 pcd 文件名 stem 对齐，例如 1641977312551.pcd 对应 timestamp=1641977312551。
#   x/y/z 是 world 坐标系下的 LiDAR/车体位移，单位 m。
#   roll/pitch/yaw 是弧度，按 xyz 欧拉角构造当前帧 LiDAR/车体坐标系 -> world 的 pose。
#   ODSS 单帧模型推理不使用 pose，但脚本会读取并写入 metadata，方便后续多帧或复查。

PCD_DIR = Path(
    os.environ["WAYMO_DATA_ROOT"]
) / (
    "15832924468527961_1564_160_1584_160/output/pcd"
)                                                   # 输入点云文件夹；要求 .pcd 文件名 stem 能对应 pose timestamp。
POSE_PATH = PCD_DIR.parent / "pose_slam_euler.txt"  # 位姿文件；必须和 pcd/ 同级，也就是放在 PCD_DIR.parent 下面。

B_FORCE_RERUN_INFERENCE = True       # True: 不管 label/detection 是否已存在，都重新推理一遍；False: 输出完整时跳过推理。
B_VISUALIZE_AFTER_INFERENCE = True   # True: 推理结束后自动打开逐帧语义可视化；False: 只保存结果不显示。

DEVICE = "cuda"                      # "cuda" 或 "cpu"；写 cuda 但机器无 GPU 时会自动退回 CPU。
MAX_FRAMES = -1                      # 最多推理多少帧；-1 表示读取 PCD_DIR 下所有 .pcd。
POSE_MAX_TIME_DIFF = 200             # pcd timestamp 找不到完全匹配 pose 时，允许匹配最近 pose 的最大时间差；<0 表示不限制。
NUM_WORKERS = 0                      # DataLoader worker 数；PCD 顺序推理通常 0 最稳。
LOG_EVERY_NFRAMES = 10               # 每隔多少帧打印一次推理进度；0 表示只打印首尾。

CFG_PATH = PROJECT_DIR / "LYF_ODSS_config.yaml"                 # ODSS 多任务模型配置。
CHECKPOINT_PATH = PROJECT_DIR / "checkpoints" / "epoch_48.pth"  # ODSS 多任务模型权重。
MODEL_NAME = "ODSS"                                             # 输出子目录名：label/<MODEL_NAME> 和 detection/<MODEL_NAME>。
SCORE_THRESHOLD_FOR_LOG = 0.3                                   # 仅用于日志统计检测框数量，不影响保存结果。


@dataclass
class ModelSpec:
    name: str
    cfg_path: Path
    checkpoint_path: Path


def BuildModelSpec() -> ModelSpec:
    return ModelSpec(
        name=MODEL_NAME,
        cfg_path=CFG_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )


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

    for candidate_name in ["pose.txt", "poses.txt", "poses_ch.txt", "pose_slam_euler.txt"]:
        candidate_path = sequence_root / candidate_name
        if candidate_path.is_file():
            return candidate_path
    raise FileNotFoundError(
        f"pose file not found under {sequence_root}; expected pose.txt, poses.txt, poses_ch.txt, or pose_slam_euler.txt"
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
    cfg_from_yaml_file(str(cfg_path), cfg)
    cfg.common_variables.ignore_index = cfg.model.ignore_index

    class_names, color_table = LoadLabelMapping(cfg)
    cfg.model.class_names = class_names
    cfg.model.num_classes = len(class_names)
    cfg.model.ignore_index = int(cfg.common_variables.ignore_index)
    cfg.model.tasks = [list(task) for task in cfg.common_variables.tasks]
    cfg.model.out_size_factor = list(cfg.common_variables.out_size_factor)
    cfg.model.voxel_size = list(cfg.common_variables.voxel_size)
    cfg.model.point_cloud_range = list(cfg.common_variables.pc_range)
    cfg.model.semantic_color_table = color_table.tolist()
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
            self.infos.append(
                {
                    "token": f"{self.sequence_name}-{source_timestamp}",
                    "sequence_name": self.sequence_name,
                    "frame_idx": frame_idx,
                    "frame_timestamp_micros": int(source_timestamp),
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        info = self.infos[idx]
        points_xyz_lidar = ReadCleanPcdXYZ(info["pcd_path"])
        if points_xyz_lidar.ndim != 2 or points_xyz_lidar.shape[1] != 3:
            raise ValueError(
                f"points in {info['pcd_path']} must be [N, 3], got {points_xyz_lidar.shape}"
            )
        if points_xyz_lidar.shape[0] == 0:
            raise ValueError(f"Empty point cloud after finite filtering: {info['pcd_path']}")

        return {
            "token": info["token"],
            "points": points_xyz_lidar,
            "pose": info["pose_lidar_to_world"],
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
        elif isinstance(value, list):
            batch_dict[key] = [
                item.to(device, non_blocking=False) if isinstance(item, torch.Tensor) else item
                for item in value
            ]
        elif isinstance(value, torch.Tensor):
            batch_dict[key] = value.to(device, non_blocking=False)
        else:
            batch_dict[key] = value
    return batch_dict


def BuildModel(spec: ModelSpec, cfg: EasyDict, device: torch.device) -> MultiMissionModel:
    model = MultiMissionModel(cfg.model).to(device)
    load_checkpoint(model, str(spec.checkpoint_path), map_location=device, strict=True)
    model.eval()
    return model


def BuildFullFrameSemanticPred(
    cfg: EasyDict,
    batch_dict: Dict[str, Any],
    token: str,
    batch_result: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    batch_dict["points"]: [N, 4]，第 0 列 batch_idx，第 1:4 列为当前帧 LiDAR/车体坐标系 xyz。
    输出 point_semantic_pred: [N]，与清洗后的 PCD 点顺序对齐；训练范围外点填 0。
    """
    points_xyz_tensor = batch_dict["points"][:, 1:4]
    points_xyz_lidar = points_xyz_tensor.detach().cpu().numpy().astype(np.float32)

    pc_range = points_xyz_tensor.new_tensor(cfg.model.point_cloud_range)
    pc_min_xyz = pc_range[:3]
    pc_max_xyz = pc_range[3:]
    b_valid_mask_tensor = ((points_xyz_tensor >= pc_min_xyz) & (points_xyz_tensor < pc_max_xyz)).all(dim=1)
    b_valid_mask = b_valid_mask_tensor.detach().cpu().numpy().astype(bool)

    valid_point_semantic_pred = (
        batch_result[token]["seg"]["point_pred"].detach().cpu().numpy().astype(np.int32)
    )
    if valid_point_semantic_pred.shape[0] != int(np.sum(b_valid_mask)):
        raise ValueError(
            f"ODSS segmentation count mismatch: pred={valid_point_semantic_pred.shape[0]}, "
            f"valid_points={int(np.sum(b_valid_mask))}, token={token}"
        )

    point_semantic_pred = np.zeros((points_xyz_lidar.shape[0],), dtype=np.int32)
    point_semantic_pred[b_valid_mask] = valid_point_semantic_pred
    return points_xyz_lidar, point_semantic_pred, valid_point_semantic_pred


def BuildDetectionResult(token: str, batch_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    输出检测结果：
    - boxes_lidar: [M, 7] = [x, y, z, l, w, h, yaw]，当前帧 LiDAR/车体坐标系。
    - scores: [M]，检测置信度。
    - label_preds: [M]，检测类别 id。
    """
    det_result = batch_result[token].get("det", {})
    boxes_lidar = det_result.get(
        "box3d_lidar",
        torch.zeros((0, 7), dtype=torch.float32),
    ).detach().cpu().numpy().astype(np.float32)
    scores = det_result.get(
        "scores",
        torch.zeros((0,), dtype=torch.float32),
    ).detach().cpu().numpy().astype(np.float32)
    label_preds = det_result.get(
        "label_preds",
        torch.zeros((0,), dtype=torch.long),
    ).detach().cpu().numpy().astype(np.int64)

    label_names = []
    for label in label_preds:
        try:
            label_names.append(WAYMO_INT_TO_TYPE_DICT[label_to_type(int(label))])
        except KeyError:
            label_names.append(str(int(label)))

    return {
        "boxes_lidar": boxes_lidar,
        "scores": scores,
        "label_preds": label_preds,
        "label_names": label_names,
    }


def BuildOutputDirs(sequence_root: Path, spec: ModelSpec) -> Dict[str, Path]:
    output_dirs = {
        "label": sequence_root / "label" / spec.name,
        "detection": sequence_root / "detection" / spec.name,
    }
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dirs


def HasCompleteOutputs(dataset: CustomPcdPoseDataset, output_dirs: Dict[str, Path]) -> bool:
    label_count = sum((output_dirs["label"] / f"{info['source_timestamp']}.label").is_file() for info in dataset.infos)
    detection_count = sum((output_dirs["detection"] / f"{info['source_timestamp']}.pkl").is_file() for info in dataset.infos)
    b_complete = label_count == len(dataset) and detection_count == len(dataset)
    print(
        f"[Check] labels={label_count}/{len(dataset)} detections={detection_count}/{len(dataset)} complete={b_complete}",
        flush=True,
    )
    return b_complete


def SaveDetectionFrame(detection_path: Path, detection_result: Dict[str, Any], info: Dict[str, Any]) -> None:
    with detection_path.open("wb") as file_object:
        pickle.dump(
            {
                "token": info["token"],
                "sequence_name": info["sequence_name"],
                "frame_idx": int(info["frame_idx"]),
                "frame_timestamp_micros": int(info["frame_timestamp_micros"]),
                "source_timestamp": info["source_timestamp"],
                "boxes_lidar": detection_result["boxes_lidar"],
                "scores": detection_result["scores"],
                "label_preds": detection_result["label_preds"],
                "label_names": detection_result["label_names"],
                "box_note": (
                    "boxes_lidar shape [M, 7] = [x, y, z, l, w, h, yaw], "
                    "current-frame LiDAR/vehicle coordinates."
                ),
            },
            file_object,
        )


def RunInference(dataset: CustomPcdPoseDataset, spec: ModelSpec) -> None:
    output_dirs = BuildOutputDirs(dataset.sequence_root, spec)
    if not B_FORCE_RERUN_INFERENCE and HasCompleteOutputs(dataset, output_dirs):
        print(f"[Skip] complete outputs already exist: {output_dirs['label']} {output_dirs['detection']}", flush=True)
        return

    cfg, _ = LoadConfig(spec.cfg_path)
    device = torch.device(DEVICE if DEVICE == "cpu" or torch.cuda.is_available() else "cpu")
    if DEVICE == "cuda" and device.type == "cpu":
        print("[Infer] CUDA is not available, fallback to CPU.", flush=True)

    model = BuildModel(spec, cfg, device)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=dataset.CollateBatch,
        pin_memory=False,
    )

    frame_summaries: List[Dict[str, Any]] = []
    infer_start_time = time.perf_counter()
    with torch.no_grad():
        for frame_idx, data_batch in enumerate(dataloader):
            info = dataset.infos[frame_idx]
            batch_dict = MoveBatchToDevice(data_batch, device)
            batch_result = model(batch_dict)
            token = batch_dict["token"][0]

            points_xyz_lidar, point_semantic_pred, valid_pred = BuildFullFrameSemanticPred(
                cfg=cfg,
                batch_dict=batch_dict,
                token=token,
                batch_result=batch_result,
            )
            detection_result = BuildDetectionResult(token, batch_result)

            label_path = output_dirs["label"] / f"{info['source_timestamp']}.label"
            detection_path = output_dirs["detection"] / f"{info['source_timestamp']}.pkl"
            point_semantic_pred.astype(np.uint32).tofile(label_path)
            SaveDetectionFrame(detection_path, detection_result, info)

            class_hist = np.bincount(
                point_semantic_pred.astype(np.int64),
                minlength=int(cfg.model.num_classes),
            )
            valid_class_hist = np.bincount(
                valid_pred.astype(np.int64),
                minlength=int(cfg.model.num_classes),
            )
            b_score_mask = detection_result["scores"] >= float(SCORE_THRESHOLD_FOR_LOG)
            frame_summaries.append(
                {
                    "token": info["token"],
                    "frame_idx": int(info["frame_idx"]),
                    "source_timestamp": info["source_timestamp"],
                    "pcd_path": info["pcd_path"],
                    "label_path": str(label_path),
                    "detection_path": str(detection_path),
                    "point_count": int(points_xyz_lidar.shape[0]),
                    "valid_point_count": int(valid_pred.shape[0]),
                    "class_hist": class_hist.astype(np.int64).tolist(),
                    "valid_class_hist": valid_class_hist.astype(np.int64).tolist(),
                    "num_boxes": int(detection_result["scores"].shape[0]),
                    "num_boxes_log_threshold": int(np.sum(b_score_mask)),
                    "pose_lidar_to_world": info["pose_lidar_to_world"],
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
                    f"valid={valid_pred.shape[0]} boxes={detection_result['scores'].shape[0]} "
                    f"label={label_path}",
                    flush=True,
                )

    infer_elapsed_time = time.perf_counter() - infer_start_time
    prediction_index_path = output_dirs["detection"] / "sequence_predictions.pkl"
    with prediction_index_path.open("wb") as file_object:
        pickle.dump(
            {
                "metadata": {
                    "model_name": spec.name,
                    "cfg": str(spec.cfg_path),
                    "checkpoint": str(spec.checkpoint_path),
                    "pcd_dir": str(dataset.pcd_dir),
                    "pose_path": str(dataset.pose_path),
                    "label_dir": str(output_dirs["label"]),
                    "detection_dir": str(output_dirs["detection"]),
                    "num_frames": len(frame_summaries),
                    "point_cloud_range": list(cfg.model.point_cloud_range),
                    "class_names": list(cfg.model.class_names),
                    "label_note": "Each .label is uint32 and aligned with the cleaned PCD point order; out-of-range points are 0.",
                    "detection_note": "Each per-frame .pkl stores boxes_lidar [M, 7], scores [M], label_preds [M].",
                    "infer_elapsed_time_sec": float(infer_elapsed_time),
                    "infer_fps": float(len(frame_summaries) / max(infer_elapsed_time, 1e-6)),
                },
                "frames": frame_summaries,
            },
            file_object,
        )
    print(
        f"[Done:{spec.name}] frames={len(frame_summaries)} "
        f"time={infer_elapsed_time:.2f}s labels={output_dirs['label']} detection={output_dirs['detection']}",
        flush=True,
    )


def Main() -> None:
    spec = BuildModelSpec()
    pcd_dir = PCD_DIR.resolve()
    pose_path = ResolvePosePath(pcd_dir, POSE_PATH)
    dataset = CustomPcdPoseDataset(pcd_dir=pcd_dir, pose_path=pose_path)
    print(
        f"[Dataset] model={spec.name} frames={len(dataset)} pcd_dir={dataset.pcd_dir} pose={dataset.pose_path}",
        flush=True,
    )

    RunInference(dataset, spec)

    if B_VISUALIZE_AFTER_INFERENCE:
        import visualize_label as visualize_script

        visualize_script.PCD_DIR = dataset.pcd_dir
        visualize_script.MODEL_NAME = spec.name
        visualize_script.LABEL_DIR = dataset.sequence_root / "label" / spec.name
        visualize_script.Main()


if __name__ == "__main__":
    Main()
