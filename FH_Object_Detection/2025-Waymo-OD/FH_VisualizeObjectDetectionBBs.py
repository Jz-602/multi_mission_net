


# import pickle
# from pathlib import Path
# import os
import numpy as np
from tqdm import tqdm

# import box_np_ops
#
# from torch.utils.data import Dataset

import open3d as o3d

from WaymoDataset import WaymoDataset

from vis import VisualizePCandBB, Box, CreateO3dboxFromBoxCorners

from o3d_view import MyVisualizer

WAYMO_INT_TO_TYPE_DICT = {0:'unknown', 1:'vehicle', 2:'pedestrian', 3:'sign', 4:'cyclist'}
WAYMO_TYPE_TO_INT_DICT = {'unknown':0, 'vehicle':1, 'pedestrian':2, 'sign':3, 'cyclist':4}


import math

if __name__ == "__main__":

    data_path = '/home/hao/dataset/waymo/FH'
    info_path='waymo_infos_train.pkl'

    dataset = WaymoDataset(
        loading_pipelines= ['FH_LoadPointCloud', 'load_box3d'],
        root_path=data_path,
        info_path=info_path,
        create_database=True,
    )


    o3d_vis = MyVisualizer()
    opt = o3d_vis.vis.get_render_option()
    # opt.background_color = np.asarray([80 / 255, 90 / 255, 110 / 255])
    opt.background_color = np.asarray([150 / 255, 150 / 255, 150 / 255])
    opt.point_size = 2.0


    for index in tqdm(range(len(dataset))):
        image_idx = index
        sensor_data = dataset[index]

        annos = sensor_data['annotations']
        gt_boxes = annos['gt_boxes']
        names = annos['gt_names']
        difficulty_level = annos['difficulty_level']

        labels = np.array([WAYMO_TYPE_TO_INT_DICT[name] for name in names])

        if 0:
            VisualizePCandBB(sensor_data['points'], gt_boxes, labels)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sensor_data['points'])

            o3d_boxes = []
            for i in range(gt_boxes.shape[0]):
                BB = Box(gt_boxes[i, :3], gt_boxes[i, 3:6], gt_boxes[i, 6])
                if not math.isnan(difficulty_level[i]):
                    labels[i] = 0
                o3d_box = CreateO3dboxFromBoxCorners(BB.Corners().T, labels[i])
                o3d_boxes.append(o3d_box)

            o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)] + o3d_boxes)