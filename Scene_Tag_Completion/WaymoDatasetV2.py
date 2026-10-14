import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from lyf_tools import visualize_point_cloud

import os
import pickle
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset
import inspect
import box_np_ops

import Augmentation
import Sampler

from pcd_io import ReadPcdXYZ

TYPE_LIST = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
SEMANTIC_PRIOR_DIRNAME = "sequence_SemanticPrior"

class BaseDataset(Dataset):

    def __init__(self, cfg, common_variables,
        evaluations = None):
        self.cfg = cfg
        self._info_path = cfg.info_path
        self._root_path = Path(common_variables.root_path)
        self.loading_pipelines = cfg.loading_pipelines
        self.b_load_semantic_prior = (
            self.loading_pipelines is not None
            and "load_semantic_prior" in self.loading_pipelines
        )
        self.point_feature_dim = 4 if self.b_load_semantic_prior else 3
        self.semantic_prior_dropout_max_ratio = 0.0
        if "semantic_prior_dropout" in cfg.keys() and cfg.semantic_prior_dropout is not None:
            self.semantic_prior_dropout_max_ratio = float(cfg.semantic_prior_dropout.get("max_drop_ratio", 0.0))

        ## 标签转化初始化
        self.ignore_index = int(common_variables.ignore_index)

        label_mapping_path = Path(__file__).resolve().parent / 'label_mapping' / common_variables.label_mapping
        with open(label_mapping_path, 'r', encoding='utf-8') as yaml_file:
            label_mapping_cfg = yaml.safe_load(yaml_file)

        self.class_names = list(label_mapping_cfg['CLASS_NAMES'])
        learning_map = label_mapping_cfg['learning_map']
        max_src_label = max(int(src_label) for src_label in learning_map.keys())
        self.label_remap_lut = np.full((max_src_label + 10,), self.ignore_index, dtype=np.int32)
        self.label_remap_lut[list(learning_map.keys())] = list(learning_map.values())

        ## 数据增强流水线
        self.augmentations = {}
        if 'augmentation' in cfg.keys():
            if cfg.augmentation is not None:
                for name,value in cfg.augmentation.items():
                    arguments = {}
                    func_name = None
                    for sub_name, sub_value in value.items():
                        if sub_name == '_target_':
                            func_name = sub_value
                        else:
                            arguments.update({sub_name:sub_value})
                    self.augmentations[name] = eval(func_name)(**arguments)

        ## 数据库采样器：用当前帧 lidar 坐标系下的 gt_boxes 做 BEV 碰撞过滤，再把采样点加入当前帧。
        self.sampler = None
        self.sampler_use_until_epoch_ratio = float(cfg.get("sampler_use_until_epoch_ratio", 1.0))
        if 'sampler' in cfg.keys():
            if cfg.sampler is not None:
                arguments = {}
                for sub_name, sub_value in cfg.sampler.items():
                    arguments.update({sub_name: sub_value})

                required_arguments = inspect.getfullargspec(Sampler.DataBaseSamplerV2)
                for argument_name in required_arguments.args:
                    if (not argument_name in arguments) and (argument_name in common_variables):
                        arguments[argument_name] = common_variables[argument_name]

                self.sampler = Sampler.DataBaseSamplerV2(**arguments)

        self.current_epoch = 0
        self.max_epochs = 1

        self.load_infos()
        self.infos = self.FilterInfosForMultiFrameEval(self.infos)
        self.sample_fraction = float(cfg.sample_fraction)
        self.sample_seed = int(cfg.sample_seed)
        self.sample_source_infos = list(self.infos)
        sample_count = max(1, int(np.ceil(len(self.sample_source_infos) * self.sample_fraction)))
        rng = np.random.default_rng(self.sample_seed)
        sampled_indices = np.sort(
            rng.choice(len(self.sample_source_infos), size=sample_count, replace=False)
        )
        self.infos = [self.sample_source_infos[int(sample_idx)] for sample_idx in sampled_indices]


    def __len__(self):
        return len(self.infos)

    def SetCurrentEpoch(self, current_epoch, max_epochs):
        self.current_epoch = int(current_epoch)
        self.max_epochs = max(1, int(max_epochs))

    def IsSamplerEnabled(self):
        if self.sampler is None:
            return False
        return self.current_epoch < self.max_epochs * self.sampler_use_until_epoch_ratio

    def load_infos(self):
        info_path_list = self._info_path
        if isinstance(info_path_list, str):
            info_path_list = [info_path_list]

        self.infos = []
        for info_path in info_path_list:
            with open(os.path.join(self._root_path, info_path), "rb") as f:
                self.infos.extend(pickle.load(f))

    def FilterInfosForMultiFrameEval(self, infos):
        if "multi_frame_eval_filter" not in self.cfg.keys() or self.cfg.multi_frame_eval_filter is None:
            return infos

        filter_cfg = self.cfg.multi_frame_eval_filter
        warmup_len = int(filter_cfg.get("warmup_len", 14))
        train_len = int(filter_cfg.get("train_len", 1))
        clip_stride = int(filter_cfg.get("clip_stride", 10))
        seq_len = warmup_len + train_len
        frame_offset_list = list(range(0, seq_len * 3, 3))
        clip_span = frame_offset_list[-1] + 1

        sequence_info_dict = {}
        for info in infos:
            sequence_name = str(info["sequence_name"])
            if sequence_name not in sequence_info_dict:
                sequence_info_dict[sequence_name] = []
            sequence_info_dict[sequence_name].append(info)

        filtered_infos = []
        for _, sequence_infos in sequence_info_dict.items():
            sequence_infos.sort(key=self.GetFrameOrderValue)
            frame_count = len(sequence_infos)
            if frame_count < clip_span:
                continue

            eval_start = min(warmup_len, seq_len)
            eval_end = min(seq_len, eval_start + train_len)
            if eval_end <= eval_start:
                eval_start = max(0, seq_len - 1)
                eval_end = seq_len

            for start_idx in range(0, frame_count - clip_span + 1, clip_stride):
                for frame_offset in frame_offset_list[eval_start:eval_end]:
                    filtered_infos.append(sequence_infos[start_idx + frame_offset])

        return filtered_infos

    def GetFrameOrderValue(self, info):
        if "original_sample_idx" in info:
            return int(info["original_sample_idx"])
        if "sample_idx" in info:
            return int(info["sample_idx"])
        return int(str(info["frame_timestamp_micros"])[:-3])

    def FormatPointFeatures(self, points):
        """
        points: shape [N, 3] 或 [N, 4]，当前帧 LiDAR/车体坐标系；第 0:3 列为 xyz，
        若存在第 4 列则为传播语义先验 label 的 float 值。
        """
        return points[:, :self.point_feature_dim].astype(np.float32, copy=False)

    def ApplySemanticPriorDropout(self, points):
        """
        points: shape [N, 4]，当前帧 LiDAR/车体坐标系 xyz + 传播语义先验 label。
        随机选取 0 到 max_drop_ratio 比例的点，把第 4 列 prior label 置 0。
        """
        if (not self.b_load_semantic_prior) or self.semantic_prior_dropout_max_ratio <= 0.0:
            return points
        if points.shape[1] < 4 or points.shape[0] == 0:
            return points

        drop_ratio = np.random.uniform(0.0, self.semantic_prior_dropout_max_ratio)
        if drop_ratio <= 0.0:
            return points

        b_drop = np.random.rand(points.shape[0]) < drop_ratio
        points = points.copy()
        points[b_drop, 3] = 0.0
        return points

    @staticmethod
    def CollateBatch(batch_list):
        example_merged = {}
        for example in batch_list:
            for key, value in example.items():
                if key not in example_merged:
                    example_merged[key] = []
                example_merged[key].append(value)

        ret = {}
        for key, elems in example_merged.items():
            if key == "token":
                ret[key] = elems
            elif 'point' in key:
                coors = []
                for i, coor in enumerate(elems):
                    coor_pad = np.pad(
                        coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                    )
                    coors.append(coor_pad)
                ret[key] = torch.tensor(np.concatenate(coors, axis=0))
            elif key == 'semantic_label':
                ret[key] = torch.tensor(np.concatenate(elems, axis=0)).long()
            elif key in ['objects', 'annos']:
                ret[key] = elems
            elif isinstance(elems[0], list):
                merged_items = {}
                res = []
                for elem in elems:
                    for idx, ele in enumerate(elem):
                        if str(idx) not in merged_items:
                            merged_items[str(idx)] = []
                        merged_items[str(idx)].append(torch.tensor(ele))
                for _, vv in merged_items.items():
                    res.append(torch.stack(vv))
                ret[key] = res
            else:
                ret[key] = torch.tensor(np.stack(elems, axis=0)).float()

        return ret

    def __getitem__(self, idx):

        info = self.infos[idx]
        res = {"token": info["token"]}

        if self.loading_pipelines is not None:
            for lp in self.loading_pipelines:
                res = getattr(self, lp)(res, info)
        # visualize_point_cloud(points=res['points'], labels=res['semantic_label'])

        if self.IsSamplerEnabled():
            sampled_dict = self.sampler.SampleAll(
                res['annotations']['gt_boxes'],
                res['annotations']['gt_names'],
            )
            if sampled_dict is not None:
                sampled_gt_boxes = sampled_dict['gt_boxes']
                ## sampled_point_indices: [N, K]，原始点 N 是否落入 K 个采样框；点和框都在当前帧 lidar 坐标系。
                sampled_point_indices = box_np_ops.points_in_rbbox(res['points'], sampled_gt_boxes)
                point_keep_mask = np.logical_not(sampled_point_indices.any(-1))
                res['points'] = res['points'][point_keep_mask]
                res['semantic_label'] = res['semantic_label'][point_keep_mask]

                res['annotations']['gt_names'] = np.concatenate(
                    [res['annotations']['gt_names'], sampled_dict['gt_names']],
                    axis=0,
                )
                res['annotations']['gt_boxes'] = np.concatenate(
                    [res['annotations']['gt_boxes'], sampled_gt_boxes],
                    axis=0,
                )
                sampled_points = sampled_dict['points']
                if res['points'].shape[1] > sampled_points.shape[1]:
                    pad_channels = res['points'].shape[1] - sampled_points.shape[1]
                    sampled_prior_pad = np.zeros(
                        (sampled_points.shape[0], pad_channels),
                        dtype=sampled_points.dtype,
                    )
                    sampled_points = np.concatenate([sampled_points, sampled_prior_pad], axis=1)
                res['points'] = np.concatenate([sampled_points[:, :res['points'].shape[1]], res['points']], axis=0)
                res['semantic_label'] = np.concatenate(
                    [sampled_dict['semantic_label'], res['semantic_label']],
                    axis=0,
                )

        ## 标签转化
        ## semantic_label: [N]，与 points [N, 3] 一一对应，坐标系为当前帧 lidar；这里统一转到训练标签空间
        res['semantic_label'] = self.label_remap_lut[res['semantic_label']]
        if self.augmentations is not None:
            for aug in self.augmentations.values():
                res = aug(res)

        res['points'] = self.FormatPointFeatures(res['points'])
        res['points'] = self.ApplySemanticPriorDropout(res['points'])
        if 'annotations' in res:
            del res['annotations']

        # visualize_point_cloud(points=res['points'], labels=res['semantic_label'])
        return res

class WaymoDataset(BaseDataset):

    def __init__(self, cfg, common_variables):
        super(WaymoDataset, self).__init__(cfg, common_variables)
        self.tasks = common_variables.tasks


    # def read_file(self, path, timestamp=0):
    #     points = np.fromfile(os.path.join(
    #         self._root_path, path), dtype=np.float32).reshape(-1, 6)
    #     # x, y, z, intensity, (remove elongation, exclude nlz points)
    #     points = points[points[:, -1] == -1, :4]
    #     timelist = timestamp * np.ones((points.shape[0], 1)).astype(np.float32)
    #     return np.concatenate((points, timelist), axis=1)

    # def read_file(self, path, timestamp=0):
    #     points = np.fromfile(os.path.join(
    #         self._root_path, path), dtype=np.float32).reshape(-1, 4)
    #     # x, y, z, intensity, (remove elongation, exclude nlz points)
    #     return points
    #
    # def load_pointcloud(self, res, info):
    #     lidar_path = 'lidar_point/' + info['token'] + '.bin'
    #     points = self.read_file(lidar_path)
    #     res["points"] = points.astype(np.float32)
    #     return res

    def FH_LoadPointCloud(self, res, info):
        lidar_filename = os.path.join(self._root_path, info['sequence_name'] + '/output/pcd/' + str(info['frame_timestamp_micros'])[:-3] + '.pcd')
        ## points: [N, 3]，当前帧 LiDAR/车体坐标系 xyz，单位 m。
        res["points"] = ReadPcdXYZ(lidar_filename)
        return res

    def load_semantic_prior(self, res, info):
        timestamp_millis = str(info['frame_timestamp_micros'])[:-3]
        semantic_prior_filename = os.path.join(
            self._root_path,
            info['sequence_name'],
            'output',
            SEMANTIC_PRIOR_DIRNAME,
            timestamp_millis + '.label',
        )

        ## semantic_prior_label: [N]，与当前帧 points 顺序一致；0 表示 unknown / no prior。
        semantic_prior_label = np.fromfile(semantic_prior_filename, dtype=np.uint16)
        if 'points' in res and res['points'].shape[0] != semantic_prior_label.shape[0]:
            raise ValueError(
                f"point count and semantic prior count mismatch for {info['token']}: "
                f"{res['points'].shape[0]} vs {semantic_prior_label.shape[0]}"
            )

        semantic_prior_feature = semantic_prior_label.astype(np.float32).reshape(-1, 1)
        res['points'] = np.concatenate(
            [res['points'][:, :3].astype(np.float32, copy=False), semantic_prior_feature],
            axis=1,
        )
        return res

    def load_box3d(self, res, info):
        annos = info['objects']
        num_points_in_gt = np.array([ann['num_points_in_top_lidar'] for ann in annos])
        mask_not_zero = (num_points_in_gt > 0).reshape(-1)

        ## gt_boxes: [K, 7]，当前帧 LiDAR/车体坐标系 3D 框，格式 [x, y, z, l, w, h, yaw]。
        gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 7)
        gt_names = np.array([TYPE_LIST[ann['type']] for ann in annos])

        res['annotations'] = {
            'gt_boxes': gt_boxes[mask_not_zero, :].astype(np.float32).copy(),
            'gt_names': gt_names[mask_not_zero].copy(),
        }
        return res

    def load_semantic(self, res, info):
        if 'semantic_path' in info:
            semantic_filename = os.path.join(self._root_path, info['semantic_path'])
        else:
            semantic_filename = os.path.join(
                self._root_path,
                info['sequence_name'],
                'output/SemanticSegmentation',
                str(info['frame_timestamp_micros'])[:-3] + '.npy',
            )

        semantic_data = np.load(semantic_filename)
        semantic_data = np.asarray(semantic_data, dtype=np.int32)
        semantic_label = semantic_data[:, 1].astype(np.int32, copy=False)

        if 'points' in res and res['points'].shape[0] != semantic_label.shape[0]:
            raise ValueError(
                f"point count and semantic label count mismatch for {info['token']}: "
                f"{res['points'].shape[0]} vs {semantic_label.shape[0]}"
            )

        res['semantic_label'] = semantic_label.copy()
        return res
