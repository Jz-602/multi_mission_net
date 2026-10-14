
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
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

import inspect

import open3d as o3d

TYPE_LIST = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

class BaseDataset(Dataset):

    def __init__(self, cfg, common_variables,
        evaluations = None,
        # create_database = False,
        use_gt_sampling = True):
        self._info_path = cfg.info_path
        self._root_path = Path(common_variables.root_path)
        self.loading_pipelines = cfg.loading_pipelines

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

        self.load_infos()



    def __len__(self):
        return len(self.infos)

    def load_infos(self):
        with open(os.path.join(self._root_path, self._info_path), "rb") as f:
            self.infos = pickle.load(f)

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
        # visualize_point_cloud(points=res['points'], labels=res['semantic_label'])
        if self.sampler is not None:
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

        # visualize_point_cloud(points=res['points'], labels=res['semantic_label'])
        return res


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
        point_cloud = o3d.io.read_point_cloud(lidar_filename)
        res["points"] = np.asarray(point_cloud.points, dtype=np.float32)
        return res

    def load_box3d(self, res, info):
        annos = info['objects']  # 当前帧中的所有目标标注列表，每个元素对应一个目标
        num_points_in_gt = np.array([ann['num_points_in_top_lidar'] for ann in annos])  # 每个目标框内落入 top lidar 的点数
        mask_not_zero = (num_points_in_gt > 0).reshape(-1)  # 过滤掩码，只保留至少包含 1 个点的目标

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
