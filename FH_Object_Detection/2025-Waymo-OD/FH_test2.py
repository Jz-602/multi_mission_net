import pickle

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

def collate(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
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
        elif isinstance(elems[0], list):
            ret[key] = defaultdict(list)
            res = []
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele))
            for kk, vv in ret[key].items():
                res.append(torch.stack(vv))
            ret[key] = res
        else:
            ret[key] = torch.tensor(np.stack(elems, axis=0)).float()

    return ret



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

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    cfg = EasyDict()
    cfg_file = 'FH_waymo_det_pp18_aspp_iou_car_sp.yaml'
    cfg_from_yaml_file(cfg_file, cfg)


    dataset = WaymoDataset(cfg.data.test_dataset, cfg.common_variables)

    model = LYF_all_Model(cfg.model, cfg.common_variables)
    model = model.cuda()



    load_checkpoint(model, './checkpoints/epoch_48.pth', "cpu", False)

    model.eval()

    print(sum([param.nelement() for param in model.parameters()]))

    o3d_vis = MyVisualizer()
    opt = o3d_vis.vis.get_render_option()
    opt.background_color = np.asarray([150 / 255, 150 / 255, 150 / 255])
    opt.point_size = 2.0

    os.makedirs('det_results', exist_ok=True)

    results = {}

    for i in tqdm(range(0, dataset.__len__(), 10)):
        data = dataset[i]

        if 0:   # show the labeling ground-truth
            label_preds = np.zeros(data['annotations']['gt_boxes'].shape[0])
            for iter in range(len(label_preds)):
                name = data['annotations']['gt_names'][iter]
                if name == 'vehicle':
                    label_preds[iter] = 0
                elif name == 'pedestrain':
                    label_preds[iter] = 1
                elif name == 'cyclist':
                    label_preds[iter] = 2
                else:
                    label_preds[iter] = 3
            VisualizePCandBB(data['points'], data['annotations']['gt_boxes'][:, [0, 1, 2, 3, 4, 5, -1]], label_preds)

        if 1:
            data_batch = collate([data])
            data_batch = example_to_device(data_batch, torch.cuda.current_device(), non_blocking=False)
            res = model(data_batch)

            if cfg.data.test_dataset.info_path == 'waymo_infos_val.pkl':
                with open(f'det_results/val_{i}.pkl', 'wb') as f:
                    pickle.dump(res, f)
            elif cfg.data.test_dataset.info_path == 'waymo_infos_train.pkl':
                with open(f'det_results/train_{i}.pkl', 'wb') as f:
                    pickle.dump(res, f)
        else:
            if cfg.data.test_dataset.info_path == 'waymo_infos_val.pkl':
                with open(f'det_results/val_{i}.pkl', 'rb') as f:
                    res = pickle.load(f)
            elif cfg.data.test_dataset.info_path == 'waymo_infos_train.pkl':
                with open(f'det_results/train_{i}.pkl', 'rb') as f:
                    res = pickle.load(f)


        results.update(res)


        if 0:
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
                o3d_vis.update([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)] + o3d_boxes)




    output_dir = Path("results") / "epoch_36"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dict = dataset.FH_evaluation(results, output_dir)
