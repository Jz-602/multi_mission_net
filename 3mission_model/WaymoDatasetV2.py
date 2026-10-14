
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

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
import box_np_ops

import Augmentation
import Assign
import Sampler
from pcd_io import ReadPcdXYZ
from three_task_labels import LoadOCCLabelData, LoadTerrainLabelData, ValidateRange

import inspect

import open3d as o3d

TYPE_LIST = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
THREE_TASK_LABEL_FIELDS = (
    "terrain_height",
    "terrain_valid_mask",
    "terrain_ignore_mask",
    "terrain_train_weight",
    "terrain_inpaint_distance",
    "terrain_resolution",
    "terrain_xy_range",
    "occ_voxel_indices",
    "occ_labels",
    "occ_voxel_size_xyz",
    "occ_pc_range",
    "occ_grid_shape",
)

class BaseDataset(Dataset):

    def __init__(self, cfg, common_variables,
        evaluations = None,
        # create_database = False,
        use_gt_sampling = True):
        self._info_path = cfg.info_path
        self._root_path = Path(common_variables.root_path)
        self.loading_pipelines = cfg.loading_pipelines
        self.b_use_time_feature = bool(cfg.get("b_use_time_feature", False))
        self.b_use_completed_detection_labels = bool(cfg.get("b_use_completed_detection_labels", False))
        self.model_pc_range = ValidateRange(common_variables.pc_range, "dataset configuration").astype(np.float32)
        self.terrain_label_path = str(
            cfg.get("terrain_label_path", "output/label/ground_truth/frames")
        )
        self.occ_label_path = str(cfg.get("occ_label_path", "output/label/background_occ"))
        if Path(self.terrain_label_path).is_absolute() or Path(self.occ_label_path).is_absolute():
            raise ValueError("terrain_label_path and occ_label_path must be relative to each sequence")

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
        ## 标签预处理/生成模块
        self.prepare_label = {}
        if 'prepare_label' in cfg.keys():
            if cfg.prepare_label is not None:
                for name,value in cfg.prepare_label.items():
                    arguments = {}
                    func_name = None
                    for sub_name, sub_value in value.items():
                        if sub_name == '_target_':
                            func_name = sub_value
                        else:
                            arguments.update({sub_name:sub_value})


                    required_arguments = inspect.getfullargspec(eval(func_name))
                    for argument_name in required_arguments.args:
                        if (not argument_name in arguments) and (argument_name in common_variables):
                            arguments[argument_name] = common_variables[argument_name]


                    self.prepare_label[name] = eval(func_name)(**arguments)
        ## 数据库采样器（GT Sampling）
        self.sampler = None
        self.sampler_use_until_epoch_ratio = float(cfg.get("sampler_use_until_epoch_ratio", 1.0))
        if 'sampler' in cfg.keys():
            if use_gt_sampling and cfg.sampler is not None:
                arguments = {}
                for sub_name, sub_value in cfg.sampler.items():
                    arguments.update({sub_name: sub_value})

                required_arguments = inspect.getfullargspec(Sampler.DataBaseSamplerV2)
                for argument_name in required_arguments.args:
                    if (not argument_name in arguments) and (argument_name in common_variables):
                        arguments[argument_name] = common_variables[argument_name]

                self.sampler = Sampler.DataBaseSamplerV2(**arguments)

        label_pipeline_names = set(self.loading_pipelines or [])
        b_load_terrain = "LoadTerrainLabels" in label_pipeline_names
        b_load_occ = "LoadOCCLabels" in label_pipeline_names
        self.b_load_three_task_labels = b_load_terrain and b_load_occ
        if b_load_terrain != b_load_occ:
            raise ValueError("LoadTerrainLabels and LoadOCCLabels must be enabled together")
        if b_load_terrain and (len(self.augmentations) > 0 or self.sampler is not None):
            raise ValueError("three-task terrain/OCC labels require augmentation: null and sampler: null")

        self.current_epoch = 0
        self.max_epochs = 1
        self.load_infos()
        self.sample_fraction = float(cfg.get("sample_fraction", 1.0))
        self.sample_seed = int(cfg.get("sample_seed", 0))
        self.sample_source_infos = list(self.infos)
        self.b_dynamic_epoch_sampling = bool(cfg.get("b_dynamic_epoch_sampling", False))
        if self.b_dynamic_epoch_sampling:
            if not 0 < self.sample_fraction <= 1:
                raise ValueError("dynamic sample_fraction must be in (0, 1]")
            # infos keeps the full target pool; sampler selects indices, temporal anchors/history stay intact.
        else:
            sample_count = max(1, int(np.ceil(len(self.sample_source_infos) * self.sample_fraction)))
            rng = np.random.default_rng(self.sample_seed)
            sampled_indices = np.sort(
                rng.choice(len(self.sample_source_infos), size=sample_count, replace=False)
            )
            self.infos = [self.sample_source_infos[int(sample_idx)] for sample_idx in sampled_indices]



    def __len__(self):
        return len(self.infos)

    def load_infos(self):
        with open(os.path.join(self._root_path, self._info_path), "rb") as f:
            self.infos = pickle.load(f)

    def SetCurrentEpoch(self, current_epoch, max_epochs):
        self.current_epoch = int(current_epoch)
        self.max_epochs = max(1, int(max_epochs))

    def IsSamplerEnabled(self):
        if self.sampler is None:
            return False
        return self.current_epoch < self.max_epochs * self.sampler_use_until_epoch_ratio

    def CheckThreeTaskLabels(self, res, token):
        missing_fields = [field_name for field_name in THREE_TASK_LABEL_FIELDS if field_name not in res]
        if missing_fields:
            raise KeyError(f"target frame {token} is missing three-task labels: {missing_fields}")

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
            elif key in ["occ_voxel_indices", "occ_labels"]:
                ret[key] = [torch.from_numpy(elem).long() for elem in elems]
            elif key in ("terrain_valid_mask", "terrain_ignore_mask"):
                ret[key] = torch.stack([torch.from_numpy(elem).bool() for elem in elems], dim=0)
            elif key == "occ_grid_shape":
                ret[key] = torch.stack([torch.from_numpy(elem).long() for elem in elems], dim=0)
            elif key in [
                "terrain_height",
                "terrain_train_weight",
                "terrain_inpaint_distance",
                "terrain_resolution",
                "terrain_xy_range",
                "occ_voxel_size_xyz",
                "occ_pc_range",
            ]:
                ret[key] = torch.stack([torch.from_numpy(elem).float() for elem in elems], dim=0)
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


    # def load_box3d(self, res, info):
    #     res["annotations"] = {
    #         'gt_boxes': info["gt_boxes"].astype(np.float32).copy(),
    #         'gt_names': np.array(info["gt_names"]).reshape(-1).copy(),
    #     }
    #
    #     return res

    def __getitem__(self, idx):

        info = self.infos[idx]
        res = {"token": info["token"]}

        if self.loading_pipelines is not None:
            for lp in self.loading_pipelines:
                res = getattr(self, lp)(res, info)
        if self.b_load_three_task_labels:
            self.CheckThreeTaskLabels(res, info["token"])
        # visualize_point_cloud(points=res['points'], labels=res['semantic_label'])
        if self.IsSamplerEnabled():
            sampled_dict = self.sampler.sample_all(
                res['annotations']['gt_boxes'],
                res["annotations"]['gt_names']
            )
            if sampled_dict is not None:
                sampled_gt_names = sampled_dict["gt_names"]
                sampled_gt_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_semantic_label = sampled_dict["semantic_label"]
                # sampled_gt_masks = sampled_dict["gt_masks"]
                res['annotations']["gt_names"] = np.concatenate(
                    [res['annotations']["gt_names"], sampled_gt_names], axis=0
                )
                res['annotations']["gt_boxes"] = np.concatenate(
                    [res['annotations']["gt_boxes"], sampled_gt_boxes]
                )

                # remove points in sampled gt boxes
                # sampled_point_indices = box_np_ops.points_in_rbbox(
                #     res['points'], sampled_gt_boxes[sampled_gt_masks])
                sampled_point_indices = box_np_ops.points_in_rbbox(    # 删除“被新物体占据的原始点”
                    res['points'], sampled_gt_boxes)
                point_keep_mask = np.logical_not(sampled_point_indices.any(-1))
                res['points'] = res['points'][point_keep_mask]
                res['semantic_label'] = res['semantic_label'][point_keep_mask]

                res['points'] = np.concatenate(
                    [sampled_points, res['points']], axis=0)
                res['semantic_label'] = np.concatenate(
                    [sampled_semantic_label, res['semantic_label']], axis=0)

        ## 标签转化
        ## semantic_label: [N]，与 points [N, 3] 一一对应，坐标系为当前帧 lidar；这里统一转到训练标签空间
        if 'semantic_label' in res:
            res['semantic_label'] = self.label_remap_lut[res['semantic_label']]
        if self.augmentations is not None:
            for aug in self.augmentations.values():
                res = aug(res)

        if self.prepare_label is not None:
            for _, pl in self.prepare_label.items():
            # for pl in self.prepare_label.values():
                res = pl(res)

        if 1:
            if 'annotations' in res:
                del res['annotations']

        res['points'] = res['points'][:,:3] # only use xyz
        if self.b_use_time_feature:
            res['points'] = self.AppendTimeFeature(
                res['points'], info['frame_timestamp_micros'], info['frame_timestamp_micros']
            )

        # visualize_point_cloud(points=res['points'], labels=res['semantic_label'])
        return res

    @staticmethod
    def AppendTimeFeature(points_xyz, frame_timestamp_micros, target_timestamp_micros):
        """[N,3] 当前帧车体系 XYZ(m) -> [N,4] XYZT，T 是相对目标帧的秒数。

        微秒时间戳先做整数差，避免绝对时间转 float32 丢失帧间时间精度。
        返回新数组，不修改共享点云；位姿/增强仅作用于前三列。
        """
        delta_seconds = (int(frame_timestamp_micros) - int(target_timestamp_micros)) / 1_000_000.0
        time_feature = np.full((len(points_xyz), 1), delta_seconds, dtype=np.float32)
        return np.concatenate((points_xyz.astype(np.float32, copy=False), time_feature), axis=1)


###################################################################

def label_to_type(label):
    if label <= 1:
        return int(label) + 1
    else:
        return 4

WAYMO_INT_TO_TYPE_DICT = {0:'unknown', 1:'vehicle', 2:'pedestrian', 3:'sign', 4:'cyclist'}
WAYMO_TYPE_TO_INT_DICT = {'unknown':0, 'vehicle':1, 'pedestrian':2, 'sign':3, 'cyclist':4}

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

    def load_box3d(self, res, info):
        annos = info['objects']  # 当前帧中的所有目标标注列表，每个元素对应一个目标
        num_points_in_gt = np.array([ann['num_points_in_top_lidar'] for ann in annos])  # 每个目标框内落入 top lidar 的点数
        mask_not_zero = (num_points_in_gt > 0).reshape(-1)  # 过滤掩码，只保留至少包含 1 个点的目标
        if self.b_use_completed_detection_labels:
            ## 补框可能因遮挡没有当前帧点；仅显式补框可绕过零点过滤，原始框规则不变。
            b_supplemental = np.asarray([ann.get('b_is_supplemental', False) for ann in annos], dtype=bool)
            mask_not_zero |= b_supplemental

        gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 7)  # 所有目标的 3D 框，格式为 [x, y, z, l, w, h, yaw]
        gt_names = np.array([TYPE_LIST[ann['type']] for ann in annos])  # 所有目标的类别名，如 vehicle、pedestrian、cyclist

        gt_boxes = gt_boxes[mask_not_zero, :]  # 过滤后保留下来的有效 3D 框
        gt_names = gt_names[mask_not_zero]  # 与有效 3D 框一一对应的类别名

        res["annotations"] = {  # 写入当前样本的检测标注
            "gt_boxes": gt_boxes.astype(np.float32).copy(),  # float32 类型的 GT 3D 框，供后续增强和标签编码使用
            "gt_names": gt_names.copy()}  # GT 3D 框对应的类别名称数组

        return res  # 返回补充完 annotations 的样本字典

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

    def LoadTerrainLabels(self, res, info):
        """
        返回当前帧 LiDAR/车体坐标系地形图。数组行对应 x 递减，列对应 y 递减。
        """
        sequence_name = str(info["sequence_name"])
        timestamp_ms = int(info["frame_timestamp_micros"]) // 1000
        terrain_data = LoadTerrainLabelData(
            str(self._root_path),
            sequence_name,
            timestamp_ms,
            self.terrain_label_path,
            self.model_pc_range,
        )
        res.update(terrain_data)
        return res

    def LoadOCCLabels(self, res, info):
        """
        返回当前帧 LiDAR/车体坐标系稀疏 OCC；坐标 shape [M, 3]，顺序为 XYZ。
        """
        sequence_name = str(info["sequence_name"])
        timestamp_ms = int(info["frame_timestamp_micros"]) // 1000
        occ_data = LoadOCCLabelData(
            str(self._root_path),
            sequence_name,
            timestamp_ms,
            self.occ_label_path,
            self.model_pc_range,
        )
        res.update(occ_data)
        return res

    def evaluation(self, detections, output_dir=None):
        for token in detections:
            detections[token]["box3d_lidar"] = detections[token]["box3d_lidar"].detach(
            ).cpu().numpy()
            detections[token]["scores"] = detections[token]["scores"].detach(
            ).cpu().numpy()
            detections[token]["label_preds"] = detections[token]["label_preds"].detach(
            ).cpu().numpy()

        objects = metrics_pb2.Objects()
        for pred in detections:
            pred = detections[pred]
            pred_boxes = pred['box3d_lidar']
            pred_label = pred['label_preds']
            pred_score = pred['scores']

            for i in range(pred_boxes.shape[0]):
                det = pred_boxes[i]
                o = metrics_pb2.Object()
                o.context_name = pred['token'].split('-')[0]
                o.frame_timestamp_micros = int(pred['token'].split('-')[1])
                box = label_pb2.Label.Box()
                box.center_x = det[0]
                box.center_y = det[1]
                box.center_z = det[2]
                box.length = det[3]
                box.width = det[4]
                box.height = det[5]
                box.heading = det[-1]
                o.object.box.CopyFrom(box)
                o.score = pred_score[i]
                o.object.type = label_to_type(pred_label[i])
                objects.objects.append(o)

        f = open(os.path.join(output_dir, 'waymo_preds.bin'), 'wb')
        f.write(objects.SerializeToString())
        f.close()

        print("use waymo devkit tool for evaluation")

        return {}


    def FH_evaluation(self, detections, output_dir=None):
        for token in detections:
            detections[token]["box3d_lidar"] = detections[token]["box3d_lidar"].detach(
            ).cpu().numpy()
            detections[token]["scores"] = detections[token]["scores"].detach(
            ).cpu().numpy()
            detections[token]["label_preds"] = detections[token]["label_preds"].detach(
            ).cpu().numpy()

        pred_infos = []
        for token in detections:
            pred_info = {}
            pred_info['name'] = [WAYMO_INT_TO_TYPE_DICT[label_to_type(label)] for label in detections[token]["label_preds"]]
            pred_info['score'] = detections[token]["scores"]
            pred_info['boxes_lidar'] = detections[token]["box3d_lidar"]
            pred_info['sequence_name'] = token.split('-')[0]
            pred_info['frame_timestamp'] = int(token.split('-')[1])
            pred_infos.append(pred_info)

        with open(os.path.join(output_dir, 'waymo_preds.pkl'), 'wb') as f:
            pickle.dump(pred_infos, f)

        objects = metrics_pb2.Objects()
        for pred in detections:
            pred = detections[pred]
            pred_boxes = pred['box3d_lidar']
            pred_label = pred['label_preds']
            pred_score = pred['scores']

            for i in range(pred_boxes.shape[0]):
                det = pred_boxes[i]
                o = metrics_pb2.Object()
                o.context_name = pred['token'].split('-')[0]
                o.frame_timestamp_micros = int(pred['token'].split('-')[1])
                box = label_pb2.Label.Box()
                box.center_x = det[0]
                box.center_y = det[1]
                box.center_z = det[2]
                box.length = det[3]
                box.width = det[4]
                box.height = det[5]
                box.heading = det[-1]
                o.object.box.CopyFrom(box)
                o.score = pred_score[i]
                o.object.type = label_to_type(pred_label[i])
                objects.objects.append(o)

        f = open(os.path.join(output_dir, 'waymo_preds.bin'), 'wb')
        f.write(objects.SerializeToString())
        f.close()
        print("use waymo devkit tool for evaluation")

        return {}
