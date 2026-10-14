import os
import pickle
from dataclasses import dataclass

import numpy as np
import torch

import box_np_ops
from WaymoDatasetV2 import BaseDataset, WaymoDataset


@dataclass
class ClipAugmentationParams:
    """
    一个 clip 共享的数据增强参数。

    aug_matrix: shape [4, 4]，原始 LiDAR/车体坐标系 -> 增强后 LiDAR/车体坐标系。
    """
    rotation: float = 0.0
    scale: float = 1.0
    translation: np.ndarray = None
    b_flip_x: bool = False
    b_flip_y: bool = False
    aug_matrix: np.ndarray = None


class WaymoDatasetMulti(WaymoDataset):
    """
    ODSS 多任务 memory 训练 dataset。

    每个样本是一段 clip，最后一帧为有监督 target frame：
    - history frames: points / pose，用于 warmup memory，不生成检测或语义 loss。
    - target frame: points / semantic_label / gt_boxes / pose，用于生成语义与检测真值。
    - points: shape [N, 3]，增强后当前帧 LiDAR/车体坐标系 xyz，单位 m。
    - gt_boxes: shape [M, 7]，仅 target frame 存在，增强后当前帧 LiDAR/车体坐标系
      [x, y, z, l, w, h, yaw]。
    - pose_to_prev: shape [4, 4]，上一帧增强后 LiDAR/车体坐标系 -> 当前帧增强后
      LiDAR/车体坐标系。
    """

    def __init__(self, cfg, common_variables):
        BaseDataset.__init__(
            self,
            cfg,
            common_variables,
            use_gt_sampling=bool(cfg.get("b_use_gt_sampling", False)),
        )
        self.tasks = common_variables.tasks
        self.cfg = cfg
        self._cfg_keys = cfg.keys()
        self._cfg_augmentation_items = (
            list(cfg.augmentation.items())
            if "augmentation" in self._cfg_keys and cfg.augmentation is not None
            else []
        )

        ## 多帧样本长度设置：一段 sample = warmup_len 个历史帧 + 1 个监督 target frame。
        self.warmup_len = int(cfg.get("warmup_len", cfg.get("WARMUP_LEN", 9)))
        self.train_len = 1
        self.seq_len = self.warmup_len + self.train_len
        self.b_shared_clip_augmentation = bool(cfg.get("b_shared_clip_augmentation", True))
        self.b_random_history_interval = bool(
            cfg.get("b_random_history_interval", len(self._cfg_augmentation_items) > 0)
        )
        self.history_interval_min = int(cfg.get("history_interval_min", 1))
        self.history_interval_max = int(cfg.get("history_interval_max", 5))
        self.eval_history_interval = int(cfg.get("eval_history_interval", 3))

        if self.seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {self.seq_len}")
        if self.history_interval_min <= 0 or self.history_interval_max < self.history_interval_min:
            raise ValueError(
                f"history interval range error: "
                f"{self.history_interval_min} to {self.history_interval_max}"
            )
        if self.eval_history_interval <= 0:
            raise ValueError(f"eval_history_interval must be positive, got {self.eval_history_interval}")

        self.full_info_path = self.GetFullInfoPath()
        self.full_infos = self.LoadFullInfos()
        self.full_sequence_info_dict, self.full_timestamp_index_dict = self.BuildFullSequenceInfoDict()
        self.anchor_list = self.BuildAnchorList()
        if len(self.anchor_list) == 0:
            raise RuntimeError(f"no valid semantic anchors built from {self._info_path}")

    def __len__(self):
        return len(self.anchor_list)

    def GetFullInfoPath(self):
        if "full_info_path" in self._cfg_keys and self.cfg.full_info_path is not None:
            return str(self.cfg.full_info_path)

        info_path = str(self._info_path)
        if "training" in info_path:
            return "waymo_infos_training.pkl"
        if "validation" in info_path:
            return "waymo_infos_validation.pkl"
        return info_path.replace("LYF_", "").replace("_semantic", "")

    def LoadFullInfos(self):
        full_info_filename = os.path.join(self._root_path, self.full_info_path)
        with open(full_info_filename, "rb") as file:
            return pickle.load(file)

    def BuildFullSequenceInfoDict(self):
        sequence_info_dict = {}
        timestamp_index_dict = {}
        for info in self.full_infos:
            sequence_name = str(info["sequence_name"])
            if sequence_name not in sequence_info_dict:
                sequence_info_dict[sequence_name] = []
            sequence_info_dict[sequence_name].append(info)

        for sequence_name, sequence_infos in sequence_info_dict.items():
            sequence_infos.sort(key=self.GetFrameOrderValue)
            timestamp_index_dict[sequence_name] = {}
            for frame_idx, info in enumerate(sequence_infos):
                timestamp_millis = str(info["frame_timestamp_micros"])[:-3]
                timestamp_index_dict[sequence_name][timestamp_millis] = frame_idx

        return sequence_info_dict, timestamp_index_dict

    def BuildAnchorList(self):
        anchor_list = []
        for anchor_info in self.infos:
            sequence_name = str(anchor_info["sequence_name"])
            target_full_idx = self.GetTargetFullIndex(anchor_info)
            anchor_list.append({
                "sequence_name": sequence_name,
                "target_full_idx": target_full_idx,
                "anchor_info": anchor_info,
            })
        return anchor_list

    def GetTargetFullIndex(self, anchor_info):
        sequence_name = str(anchor_info["sequence_name"])
        if sequence_name not in self.full_sequence_info_dict:
            raise KeyError(f"sequence {sequence_name} not found in {self.full_info_path}")

        sequence_infos = self.full_sequence_info_dict[sequence_name]
        if "original_sample_idx" in anchor_info:
            target_full_idx = int(anchor_info["original_sample_idx"])
            if 0 <= target_full_idx < len(sequence_infos):
                return target_full_idx

        timestamp_millis = str(anchor_info["frame_timestamp_micros"])[:-3]
        timestamp_index_dict = self.full_timestamp_index_dict[sequence_name]
        if timestamp_millis not in timestamp_index_dict:
            raise KeyError(
                f"semantic anchor {anchor_info['token']} not found in full infos by timestamp {timestamp_millis}"
            )
        return int(timestamp_index_dict[timestamp_millis])

    def GetFrameOrderValue(self, info):
        if "original_sample_idx" in info:
            return int(info["original_sample_idx"])
        if "sample_idx" in info:
            return int(info["sample_idx"])
        return int(str(info["frame_timestamp_micros"])[:-3])

    def SelectHistoryInterval(self, target_full_idx):
        if self.warmup_len <= 0:
            return 1, False

        if self.b_random_history_interval:
            legal_interval_list = [
                interval
                for interval in range(self.history_interval_min, self.history_interval_max + 1)
                if target_full_idx - self.warmup_len * interval >= 0
            ]
            if len(legal_interval_list) > 0:
                return int(np.random.choice(legal_interval_list)), False
            return 1, True

        for history_interval in range(self.eval_history_interval, 0, -1):
            if target_full_idx - self.warmup_len * history_interval >= 0:
                return int(history_interval), False
        return 1, True

    def BuildFrameIndexList(self, target_full_idx, history_interval):
        frame_idx_list = []
        b_history_padded = False
        for history_offset in range(self.warmup_len, 0, -1):
            frame_idx = target_full_idx - history_offset * history_interval
            if frame_idx < 0:
                frame_idx = 0
                b_history_padded = True
            frame_idx_list.append(int(frame_idx))
        frame_idx_list.append(int(target_full_idx))
        return frame_idx_list, b_history_padded

    def BuildTargetInfo(self, full_info, anchor_info):
        target_info = dict(full_info)
        target_info.update(anchor_info)
        return target_info

    def LoadSingleFrame(self, info, b_load_target_labels):
        """
        points: shape [N, 3]，原始当前帧 LiDAR/车体坐标系 xyz，单位 m。
        semantic_label: shape [N]，仅 target frame 存在，尚未增强但与 points 顺序一致。
        pose: shape [4, 4]，当前帧 LiDAR/车体坐标系 -> world 坐标系。
        """
        res = {"token": info["token"]}
        if self.loading_pipelines is not None:
            for loading_pipeline in self.loading_pipelines:
                if (not b_load_target_labels) and loading_pipeline in ["load_semantic", "load_box3d"]:
                    continue
                res = getattr(self, loading_pipeline)(res, info)

        if b_load_target_labels:
            res["semantic_label"] = self.label_remap_lut[res["semantic_label"]]
            if res["points"].shape[0] != res["semantic_label"].shape[0]:
                raise ValueError(
                    f"point count and semantic label count mismatch for {info['token']}: "
                    f"{res['points'].shape[0]} vs {res['semantic_label'].shape[0]}"
                )

        res["points"] = res["points"][:, :3].astype(np.float32, copy=False)
        res["pose"] = np.asarray(info["pose"], dtype=np.float32).reshape(4, 4)
        res["frame_idx"] = np.array([self.GetFrameOrderValue(info)], dtype=np.int32)
        return res

    def SampleClipAugmentationParams(self):
        params = ClipAugmentationParams(
            translation=np.zeros(3, dtype=np.float32),
            aug_matrix=np.eye(4, dtype=np.float32),
        )

        if len(self._cfg_augmentation_items) == 0:
            return params

        aug_matrix = np.eye(4, dtype=np.float32)
        for _, aug_cfg in self._cfg_augmentation_items:
            target_name = str(aug_cfg["_target_"])
            stage_matrix = np.eye(4, dtype=np.float32)

            if target_name.endswith("Rotation"):
                params.rotation = float(np.random.uniform(float(aug_cfg["rotation"][0]), float(aug_cfg["rotation"][1])))
                cos_value = np.cos(params.rotation)
                sin_value = np.sin(params.rotation)
                stage_matrix[:3, :3] = np.array([
                    [cos_value, -sin_value, 0.0],
                    [sin_value, cos_value, 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float32)
            elif target_name.endswith("Scaling"):
                params.scale = float(np.random.uniform(float(aug_cfg["scale"][0]), float(aug_cfg["scale"][1])))
                stage_matrix = np.diag([params.scale, params.scale, params.scale, 1.0]).astype(np.float32)
            elif target_name.endswith("Translation"):
                noise_translate = np.random.normal(0.0, float(aug_cfg["noise"]), 1).astype(np.float32)
                params.translation = np.repeat(noise_translate, 3).astype(np.float32)
                stage_matrix[:3, 3] = params.translation
            elif target_name.endswith("Flip"):
                flip_prob = aug_cfg["flip_prob"]
                if float(flip_prob[0]) > 0.0:
                    params.b_flip_x = bool(np.random.choice([False, True], p=[1.0 - float(flip_prob[0]), float(flip_prob[0])]))
                    if params.b_flip_x:
                        stage_matrix[1, 1] = -1.0
                if float(flip_prob[1]) > 0.0:
                    params.b_flip_y = bool(np.random.choice([False, True], p=[1.0 - float(flip_prob[1]), float(flip_prob[1])]))
                    if params.b_flip_y:
                        stage_matrix[0, 0] = -1.0
            else:
                raise ValueError(f"unsupported augmentation target: {target_name}")

            aug_matrix = stage_matrix @ aug_matrix

        params.aug_matrix = aug_matrix.astype(np.float32)
        return params

    def BuildClipAugmentationList(self):
        if self.b_shared_clip_augmentation:
            params = self.SampleClipAugmentationParams()
            return [params for _ in range(self.seq_len)]
        return [self.SampleClipAugmentationParams() for _ in range(self.seq_len)]

    def ApplyTransform(self, points_xyz, transform_matrix):
        """
        points_xyz: shape [N, 3]，输入点所在坐标系由 transform_matrix 的输入侧定义。
        transform_matrix: shape [4, 4]，输入坐标系 -> 输出坐标系。
        """
        rotation = transform_matrix[:3, :3]
        translation = transform_matrix[:3, 3].reshape(1, 3)
        return (points_xyz @ rotation.T + translation).astype(np.float32, copy=False)

    def ApplyBoxAugmentation(self, gt_boxes, params):
        """
        gt_boxes: shape [M, 7]，原始当前帧 LiDAR/车体坐标系 [x, y, z, l, w, h, yaw]。
        返回增强后当前帧 LiDAR/车体坐标系 gt_boxes。
        """
        if gt_boxes.shape[0] == 0:
            return gt_boxes.astype(np.float32, copy=False)

        boxes = gt_boxes.astype(np.float32, copy=True)
        mask = np.isnan(boxes)
        boxes[mask] = 0

        if abs(params.rotation) > 0.0:
            boxes = box_np_ops.rotate(boxes, params.rotation)
        if abs(params.scale - 1.0) > 0.0:
            boxes = box_np_ops.scaling(boxes, params.scale)
        if np.any(np.abs(params.translation) > 0.0):
            boxes = box_np_ops.translate(boxes, params.translation)
        if params.b_flip_x:
            boxes = box_np_ops.flip(boxes, axis="x")
        if params.b_flip_y:
            boxes = box_np_ops.flip(boxes, axis="y")

        boxes[mask] = np.nan
        return boxes.astype(np.float32, copy=False)

    def ApplyTargetLabels(self, frame_ret, frame_data, aug_params):
        frame_ret["semantic_label"] = frame_data["semantic_label"].astype(np.int64, copy=False)

        if "annotations" in frame_data:
            annotations = {
                "gt_names": frame_data["annotations"]["gt_names"].copy(),
                "gt_boxes": self.ApplyBoxAugmentation(
                    frame_data["annotations"]["gt_boxes"],
                    aug_params,
                ),
            }
            frame_ret["annotations"] = annotations

        if self.prepare_label is not None:
            for _, prepare_label in self.prepare_label.items():
                frame_ret = prepare_label(frame_ret)

        if "annotations" in frame_ret:
            del frame_ret["annotations"]
        return frame_ret

    def __getitem__(self, idx):
        anchor_item = self.anchor_list[idx]
        sequence_infos = self.full_sequence_info_dict[anchor_item["sequence_name"]]
        target_full_idx = int(anchor_item["target_full_idx"])
        history_interval, b_interval_padded = self.SelectHistoryInterval(target_full_idx)
        frame_idx_list, b_history_padded = self.BuildFrameIndexList(target_full_idx, history_interval)
        b_history_padded = bool(b_history_padded or b_interval_padded)

        clip_infos = [sequence_infos[frame_idx] for frame_idx in frame_idx_list[:-1]]
        target_info = self.BuildTargetInfo(sequence_infos[target_full_idx], anchor_item["anchor_info"])
        clip_infos.append(target_info)

        frame_data_list = []
        for frame_offset, frame_info in enumerate(clip_infos):
            b_load_target_labels = frame_offset == self.warmup_len
            frame_data_list.append(self.LoadSingleFrame(frame_info, b_load_target_labels=b_load_target_labels))

        aug_params_list = self.BuildClipAugmentationList()

        frame_ret_list = []
        pose_seq = []
        aug_matrix_seq = []
        pose_to_prev_seq = []
        frame_idx_seq = []
        token_seq = []

        for frame_offset, frame_data in enumerate(frame_data_list):
            pose_cur = frame_data["pose"]
            aug_params = aug_params_list[frame_offset]
            aug_cur = aug_params.aug_matrix

            if frame_offset == 0:
                pose_to_prev = np.eye(4, dtype=np.float32)
            else:
                pose_prev = frame_data_list[frame_offset - 1]["pose"]
                aug_prev = aug_params_list[frame_offset - 1].aug_matrix
                pose_to_prev = aug_cur @ np.linalg.inv(pose_cur) @ pose_prev @ np.linalg.inv(aug_prev)
                pose_to_prev = pose_to_prev.astype(np.float32)

            points_aug = self.ApplyTransform(frame_data["points"], aug_cur)
            frame_ret = {
                "token": frame_data["token"],
                "points": points_aug,
                "pose": pose_cur.astype(np.float32, copy=False),
                "aug_matrix": aug_cur.astype(np.float32, copy=False),
                "pose_to_prev": pose_to_prev.astype(np.float32, copy=False),
                "frame_idx": frame_data["frame_idx"],
            }
            if "semantic_label" in frame_data:
                frame_ret = self.ApplyTargetLabels(frame_ret, frame_data, aug_params)

            frame_ret_list.append(frame_ret)
            pose_seq.append(pose_cur.astype(np.float32, copy=False))
            aug_matrix_seq.append(aug_cur.astype(np.float32, copy=False))
            pose_to_prev_seq.append(pose_to_prev.astype(np.float32, copy=False))
            frame_idx_seq.append(int(frame_data["frame_idx"][0]))
            token_seq.append(frame_data["token"])

        return {
            "frames": frame_ret_list,
            "sequence_name": anchor_item["sequence_name"],
            "clip_start_idx": np.int32(frame_idx_list[0]),
            "target_frame_index": np.int32(target_full_idx),
            "frame_indices": np.asarray(frame_idx_seq, dtype=np.int32),
            "full_frame_indices": np.asarray(frame_idx_list, dtype=np.int32),
            "token_seq": token_seq,
            "pose_seq": np.stack(pose_seq, axis=0).astype(np.float32),
            "aug_matrix_seq": np.stack(aug_matrix_seq, axis=0).astype(np.float32),
            "pose_to_prev_seq": np.stack(pose_to_prev_seq, axis=0).astype(np.float32),
            "history_interval": np.int32(history_interval),
            "b_history_padded": np.bool_(b_history_padded),
            "warmup_len": np.int32(self.warmup_len),
            "train_len": np.int32(self.train_len),
        }

    @staticmethod
    def CollateOneFrame(frame_inputs):
        return BaseDataset.CollateBatch(frame_inputs)

    @staticmethod
    def CollateBatch(batch_list):
        seq_len = len(batch_list[0]["frames"])
        frames_collated = []
        for frame_offset in range(seq_len):
            frame_inputs = [sample["frames"][frame_offset] for sample in batch_list]
            frames_collated.append(WaymoDatasetMulti.CollateOneFrame(frame_inputs))

        return {
            "frames": frames_collated,
            "sequence_name": [sample["sequence_name"] for sample in batch_list],
            "clip_start_idx": torch.tensor(
                [sample["clip_start_idx"] for sample in batch_list],
                dtype=torch.int32,
            ),
            "target_frame_index": torch.tensor(
                [sample["target_frame_index"] for sample in batch_list],
                dtype=torch.int32,
            ),
            "frame_indices": torch.stack(
                [torch.from_numpy(sample["frame_indices"]).int() for sample in batch_list],
                dim=0,
            ),
            "full_frame_indices": torch.stack(
                [torch.from_numpy(sample["full_frame_indices"]).int() for sample in batch_list],
                dim=0,
            ),
            "token_seq": [sample["token_seq"] for sample in batch_list],
            "pose_seq": torch.stack(
                [torch.from_numpy(sample["pose_seq"]).float() for sample in batch_list],
                dim=0,
            ),
            "aug_matrix_seq": torch.stack(
                [torch.from_numpy(sample["aug_matrix_seq"]).float() for sample in batch_list],
                dim=0,
            ),
            "pose_to_prev_seq": torch.stack(
                [torch.from_numpy(sample["pose_to_prev_seq"]).float() for sample in batch_list],
                dim=0,
            ),
            "history_interval": torch.tensor(
                [sample["history_interval"] for sample in batch_list],
                dtype=torch.int32,
            ),
            "b_history_padded": torch.tensor(
                [sample["b_history_padded"] for sample in batch_list],
                dtype=torch.bool,
            ),
            "warmup_len": int(batch_list[0]["warmup_len"]),
            "train_len": int(batch_list[0]["train_len"]),
        }
