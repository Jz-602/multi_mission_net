import os
import pickle

import numpy as np

from WaymoDatasetV2 import BaseDataset, WaymoDataset


class WaymoDatasetMulti(WaymoDataset):
    """
    多帧拼接语义补全 dataset。

    每个样本以一个有人工语义真值的 target frame 为监督帧：
    - 输入点云: shape [sum_N, 4]，所有上下文帧点云都对齐到 target frame LiDAR/车体坐标系，
      第 0:3 列为 target frame 坐标系 xyz，第 4 列为该点所属原始帧的 pseudo_label_1。
    - 监督标签: shape [sum_N]，只有 target frame 自身点为人工语义真值，其它上下文帧点填 ignore_index。
    - pose: 每帧 info['pose'] 表示该帧 LiDAR/车体坐标系 -> 世界坐标系。
    """

    def __init__(self, cfg, common_variables):
        super(WaymoDatasetMulti, self).__init__(cfg, common_variables)

        self.cfg = cfg
        self._cfg_keys = cfg.keys()
        self.input_frame_count = int(cfg.get("input_frame_count", 7))
        self.frame_interval = int(cfg.get("frame_interval", 2))
        self.supervised_window_offset = int(
            cfg.get("supervised_window_offset", self.input_frame_count // 2)
        )
        self.b_random_supervised_position = bool(cfg.get("b_random_supervised_position", False))
        self.pseudo_label_dirname = str(cfg.get("pseudo_label_dirname", "label/pseudo_label_1"))
        self.pseudo_label_dtype = np.dtype(str(cfg.get("pseudo_label_dtype", "uint32")))

        if self.input_frame_count <= 0:
            raise ValueError(f"input_frame_count must be positive, got {self.input_frame_count}")
        if self.frame_interval <= 0:
            raise ValueError(f"frame_interval must be positive, got {self.frame_interval}")
        if not 0 <= self.supervised_window_offset < self.input_frame_count:
            raise ValueError(
                f"supervised_window_offset must be in [0, {self.input_frame_count - 1}], "
                f"got {self.supervised_window_offset}"
            )

        if self.loading_pipelines is not None and "load_pseudo_label" in self.loading_pipelines:
            self.point_feature_dim = 4

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
            return self.cfg.full_info_path

        info_path = str(self._info_path)
        if "training" in info_path:
            return "waymo_infos_training_point_only.pkl"
        if "validation" in info_path:
            return "waymo_infos_validation_point_only.pkl"
        return info_path.replace("LYF_", "").replace("_semantic", "")

    def LoadFullInfos(self):
        full_info_path_list = self.full_info_path
        if isinstance(full_info_path_list, str):
            full_info_path_list = [full_info_path_list]

        full_infos = []
        for full_info_path in full_info_path_list:
            full_info_filename = os.path.join(self._root_path, full_info_path)
            with open(full_info_filename, "rb") as file:
                full_infos.extend(pickle.load(file))
        return full_infos

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
            anchor_list.append(
                {
                    "sequence_name": sequence_name,
                    "target_full_idx": target_full_idx,
                    "anchor_info": anchor_info,
                }
            )
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

    def GetLegalSupervisedOffsetRange(self, target_full_idx, sequence_len):
        min_offset = self.input_frame_count - 1 - (sequence_len - 1 - target_full_idx) // self.frame_interval
        max_offset = target_full_idx // self.frame_interval
        min_offset = max(0, int(min_offset))
        max_offset = min(self.input_frame_count - 1, int(max_offset))
        return min_offset, max_offset

    def SelectSupervisedWindowOffset(self, target_full_idx, sequence_len):
        min_offset, max_offset = self.GetLegalSupervisedOffsetRange(target_full_idx, sequence_len)
        if min_offset <= max_offset:
            if self.b_random_supervised_position:
                return int(np.random.randint(min_offset, max_offset + 1))
            return int(np.clip(self.supervised_window_offset, min_offset, max_offset))
        return self.supervised_window_offset

    def BuildFrameIndexList(self, target_full_idx, sequence_len, supervised_window_offset):
        frame_idx_list = []
        b_window_padded = False
        for frame_offset in range(self.input_frame_count):
            relative_offset = (frame_offset - supervised_window_offset) * self.frame_interval
            raw_frame_idx = target_full_idx + relative_offset
            frame_idx = int(np.clip(raw_frame_idx, 0, sequence_len - 1))
            if frame_idx != raw_frame_idx:
                b_window_padded = True
            frame_idx_list.append(frame_idx)
        return frame_idx_list, b_window_padded

    def BuildTargetInfo(self, full_info, anchor_info):
        target_info = dict(full_info)
        target_info.update(anchor_info)
        return target_info

    def load_pseudo_label(self, res, info):
        timestamp_millis = str(info["frame_timestamp_micros"])[:-3]
        pseudo_label_filename = os.path.join(
            self._root_path,
            info["sequence_name"],
            "output",
            self.pseudo_label_dirname,
            timestamp_millis + ".label",
        )

        pseudo_label = np.fromfile(pseudo_label_filename, dtype=self.pseudo_label_dtype)
        if "points" in res and res["points"].shape[0] != pseudo_label.shape[0]:
            raise ValueError(
                f"point count and pseudo label count mismatch for {info['token']}: "
                f"{res['points'].shape[0]} vs {pseudo_label.shape[0]}, path={pseudo_label_filename}"
            )

        pseudo_label_feature = pseudo_label.astype(np.float32).reshape(-1, 1)
        res["points"] = np.concatenate(
            [res["points"][:, :3].astype(np.float32, copy=False), pseudo_label_feature],
            axis=1,
        )
        return res

    def LoadSingleFrame(self, info, b_load_semantic):
        """
        读取单帧数据。

        points: shape [N, 4]，当前帧 LiDAR/车体坐标系 xyz + pseudo_label_1。
        semantic_label: shape [N]，仅 target frame 读取，原始 Waymo 标签会映射到训练标签空间。
        pose: shape [4, 4]，当前帧 LiDAR/车体坐标系 -> 世界坐标系。
        """
        res = {"token": info["token"]}

        if self.loading_pipelines is not None:
            for loading_pipeline in self.loading_pipelines:
                if (not b_load_semantic) and loading_pipeline == "load_semantic":
                    continue
                res = getattr(self, loading_pipeline)(res, info)

        if b_load_semantic:
            res["semantic_label"] = self.label_remap_lut[res["semantic_label"]]
        res["points"] = self.FormatPointFeatures(res["points"])
        res["pose"] = np.asarray(info["pose"], dtype=np.float32).reshape(4, 4)
        res["frame_idx"] = np.array([self.GetFrameOrderValue(info)], dtype=np.int32)

        if b_load_semantic and res["points"].shape[0] != res["semantic_label"].shape[0]:
            raise ValueError(
                f"point count and semantic label count mismatch for {info['token']}: "
                f"{res['points'].shape[0]} vs {res['semantic_label'].shape[0]}"
            )
        return res

    def ApplyTransform(self, points, transform_matrix):
        """
        points: shape [N, 4]，输入点所在坐标系由 transform_matrix 输入侧定义；
        第 4 列 pseudo label 不参与坐标变换。
        transform_matrix: shape [4, 4]，输入 LiDAR/车体坐标系 -> target frame LiDAR/车体坐标系。
        """
        points_xyz = points[:, :3]
        rotation = transform_matrix[:3, :3]
        translation = transform_matrix[:3, 3].reshape(1, 3)
        transformed_points_xyz = points_xyz @ rotation.T + translation
        return np.concatenate(
            [transformed_points_xyz, points[:, 3:].astype(np.float32, copy=False)],
            axis=1,
        ).astype(np.float32, copy=False)

    def __getitem__(self, idx):
        anchor_item = self.anchor_list[idx]
        sequence_infos = self.full_sequence_info_dict[anchor_item["sequence_name"]]
        sequence_len = len(sequence_infos)
        target_full_idx = int(anchor_item["target_full_idx"])
        supervised_window_offset = self.SelectSupervisedWindowOffset(target_full_idx, sequence_len)
        frame_idx_list, b_window_padded = self.BuildFrameIndexList(
            target_full_idx,
            sequence_len,
            supervised_window_offset,
        )

        clip_infos = [sequence_infos[frame_idx] for frame_idx in frame_idx_list]
        target_info = self.BuildTargetInfo(
            sequence_infos[target_full_idx],
            anchor_item["anchor_info"],
        )
        clip_infos[supervised_window_offset] = target_info

        target_pose = np.asarray(target_info["pose"], dtype=np.float32).reshape(4, 4)
        world_to_target = np.linalg.inv(target_pose).astype(np.float32)

        points_list = []
        semantic_label_list = []
        for frame_offset, frame_info in enumerate(clip_infos):
            b_load_semantic = frame_offset == supervised_window_offset
            frame_data = self.LoadSingleFrame(frame_info, b_load_semantic=b_load_semantic)
            lidar_to_target = world_to_target @ frame_data["pose"]
            points_target = self.ApplyTransform(frame_data["points"], lidar_to_target)
            points_list.append(points_target)

            if b_load_semantic:
                semantic_label_list.append(frame_data["semantic_label"].astype(np.int64, copy=False))
            else:
                semantic_label_list.append(
                    np.full(points_target.shape[0], int(self.ignore_index), dtype=np.int64)
                )

        return {
            "token": target_info["token"],
            "points": np.concatenate(points_list, axis=0).astype(np.float32, copy=False),
            "semantic_label": np.concatenate(semantic_label_list, axis=0).astype(np.int64, copy=False),
            "frame_indices": np.asarray(frame_idx_list, dtype=np.int32),
            "target_frame_index": np.int32(target_full_idx),
            "supervised_window_offset": np.int32(supervised_window_offset),
            "b_window_padded": np.bool_(b_window_padded),
        }

    @staticmethod
    def CollateBatch(batch_list):
        return BaseDataset.CollateBatch(batch_list)
