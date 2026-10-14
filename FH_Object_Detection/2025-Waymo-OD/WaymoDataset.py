

import pickle
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

import box_np_ops

from torch.utils.data import Dataset

import open3d as o3d

TYPE_LIST = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

class BaseDataset(Dataset):
    def __init__(
            self,
            root_path,
            info_path,
            sampler=None,
            loading_pipelines=None,
            augmentation=None,
            prepare_label=None,
            evaluations=None,
            create_database=False,
            use_gt_sampling=True,):

        self._info_path = info_path
        self._root_path = Path(root_path)
        self.loading_pipelines = loading_pipelines
        self.augmentations = augmentation
        self.prepare_label = prepare_label
        self.evaluations = evaluations
        self.create_database = create_database
        self.use_gt_sampling = use_gt_sampling

        with open(os.path.join(self._root_path, self._info_path), "rb") as f:
            self.infos = pickle.load(f)

        if use_gt_sampling and sampler is not None:
            self.sampler = sampler()
        else:
            self.sampler = None

    def __len__(self):
        return len(self.infos)


    def __getitem__(self, idx):
        info = self.infos[idx]
        res = {"token": info["token"]}

        if self.loading_pipelines is not None:
            for lp in self.loading_pipelines:
                res = getattr(self, lp)(res, info)

        if 'annotations' in res and (not self.create_database):
            del res['annotations']

        return res
#########################################################################################

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

def label_to_type(label):
    if label <= 1:
        return int(label) + 1
    else:
        return 4

class WaymoDataset(BaseDataset):
    def __init__(self,
                 info_path,
                 root_path,
                 sampler=None,
                 loading_pipelines=None,
                 augmentation=None,
                 prepare_label=None,
                 evaluations=None,
                 create_database=False,
                 use_gt_sampling=True):

        super(WaymoDataset, self).__init__(
            root_path, info_path, sampler, loading_pipelines, augmentation, prepare_label, evaluations, create_database,
            use_gt_sampling=use_gt_sampling)


    def FH_LoadPointCloud(self, res, info):
        lidar_filename = os.path.join(self._root_path, info['sequence_name'] + '/output/pcd/' + str(info['frame_timestamp_micros'])[:-3] + '.pcd')
        point_cloud = o3d.io.read_point_cloud(lidar_filename)
        res["points"] = np.asarray(point_cloud.points, dtype=np.float32)
        return res

    def load_box3d(self, res, info):
        annos = info['objects']
        num_points_in_gt = np.array([ann['num_points_in_top_lidar'] for ann in annos])  # originally, it should be ann['num_points_in_all_lidar']
        mask_not_zero = (num_points_in_gt > 0).reshape(-1)

        difficulty_level = np.array([ann['difficulty_level'][0] for ann in annos])

        gt_boxes = np.array([ann['box'] for ann in annos]).reshape(-1, 7)
        gt_names = np.array([TYPE_LIST[ann['type']] for ann in annos])

        gt_boxes = gt_boxes[mask_not_zero, :]
        gt_names = gt_names[mask_not_zero]

        difficulty_level = difficulty_level[mask_not_zero]

        res["annotations"] = {
            "gt_boxes": gt_boxes.astype(np.float32).copy(),
            "gt_names": gt_names.copy(),
            "difficulty_level": difficulty_level.copy()}

        return res


