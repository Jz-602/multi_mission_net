import pickle

from thop import profile
import yaml
from easydict import EasyDict

from collections import defaultdict

import torch
import numpy as np

import random

from WaymoDatasetV2 import WaymoDataset

from LYF_all_Model import LYF_all_Model
from pathlib import Path
import os

def merge_new_config(config, new_config):
    if '_BASE_CONFIG_' in new_config:
        with open(new_config['_BASE_CONFIG_'], 'r') as f:
            try:
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_config = yaml.load(f)
        config.update(EasyDict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)

    return config

def cfg_from_yaml_file(cfg_file, config):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

        merge_new_config(config=config, new_config=new_config)

    return config


def example_to_device(example, device, non_blocking=False):
    example_torch = {}
    for k, v in example.items():
        if k in ['token']:
            example_torch[k] = v
        elif isinstance(v, list):
            example_torch[k] = [
                res.to(device, non_blocking=non_blocking) for res in v]
        else:
            example_torch[k] = v.to(device, non_blocking=non_blocking)

    return example_torch



from trainer.utils.checkpoint import load_checkpoint

# from Visualizer import VisualizePCandBB
# from VisualizerBEV import VisualizePCandBB

from vis import VisualizePCandBB, Box, CreateO3dboxFromBoxCorners


# from VisualizerNonBlocking import NonBlockingVisualizer
# vis = NonBlockingVisualizer()

from tqdm import tqdm

from o3d_view import MyVisualizer

import open3d as o3d

import glob

import time

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    cfg = EasyDict()
    cfg_file = 'FH_waymo_det_pp18_aspp_iou_car_sp.yaml'
    cfg_from_yaml_file(cfg_file, cfg)


    # dataset = WaymoDataset(cfg.data.test_dataset, cfg.common_variables)

    model = LYF_all_Model(cfg.model, cfg.common_variables)
    model = model.cuda()



    load_checkpoint(model, './checkpoints/epoch_48.pth', "cpu", False)

    model.eval()

    print(sum([param.nelement() for param in model.parameters()]))
    if 1:
        input = {}
        input['points'] = torch.zeros((100000,4)).to("cuda:0")
        flops, num_of_params = profile(model, inputs=(input,))
        print(flops)
        print(num_of_params)



    # o3d_vis = MyVisualizer()
    o3d_vis = MyVisualizer(view_file = 'logs/imgs/ScreenView_2025-11-11-10-15-43.json')
    opt = o3d_vis.vis.get_render_option()
    opt.background_color = np.asarray([150 / 255, 150 / 255, 150 / 255])
    opt.point_size = 2.0

    os.makedirs('our_results', exist_ok=True)

    results = {}

    # dir_name = '/home/hao/dataset/2025/2025-04-17/2025-04-17-15-03-27-car+bycicle'
    # dir_name = '/home/hao/dataset/2025/2025-04-17/2025-04-17-14-43-11-person1'

    dir_name = '/home/hao/dataset/2025-F2/output_body_tracker'


    filenames = list(glob.glob(os.path.join(dir_name, 'output/pcd', '*.pcd')))
    filenames.sort()

    data = {}
    data_batch = {}
    for file_idx in range(len(filenames)):
        point_cloud = o3d.io.read_point_cloud(filenames[file_idx])
        points = np.asarray(point_cloud.points, dtype=np.float32)
        padded_points = np.pad(points, ((0, 0), (1, 0)), mode="constant", constant_values=0)

        data_batch['points'] = torch.tensor(padded_points)
        data_batch['token'] = [f'random_sequence-{file_idx}']

        data['token'] = data_batch['token']
        data['points'] = points

        data_batch = example_to_device(data_batch, torch.cuda.current_device(), non_blocking=False)

        start = time.time()
        res = model(data_batch)
        end = time.time()
        print(end-start)

        for res_value in res.values():
            bbs = res_value['box3d_lidar'].numpy()
            label_preds = res_value['label_preds'].numpy()
            scores = res_value['scores'].numpy()
            mask = scores > 0.5
            bbs = bbs[mask, :]
            label_preds = label_preds[mask]
            scores = scores[mask]

            # VisualizePCandBB(data['points'], bbs[:, [0, 1, 2, 3, 4, 5, -1]], label_preds)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(data['points'])
            o3d_boxes = []
            for i in range(bbs.shape[0]):
                BB = Box(bbs[i, :3], bbs[i, 3:6], bbs[i, 6])
                o3d_box = CreateO3dboxFromBoxCorners(BB.Corners().T, label_preds[i])
                o3d_boxes.append(o3d_box)

            if 0:
                o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)] + o3d_boxes)
            else:
                # o3d_vis.block_vis = False
                # o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)] + o3d_boxes)
                # o3d_vis._save_screen(o3d_vis.vis)
                o3d_vis.show_and_save([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)] + o3d_boxes, file_idx)
