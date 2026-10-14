


import os
import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.v2.perception import lidar as _v2_lidar
from waymo_open_dataset.v2.perception import pose as _v2_pose
from waymo_open_dataset.v2 import column_types


import open3d as o3d


import numpy as np


import glob
import os
from tqdm import tqdm

import pickle
import copy

import matplotlib.pyplot as plt


def read(dataset_dir, context_name, tag: str) -> dd.DataFrame:
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  return dd.read_parquet(paths)

WAYMO_INT_TO_TYPE_DICT = {0:'unknown', 1:'vehicle', 2:'pedestrian', 3:'sign', 4:'cyclist'}
WAYMO_TYPE_TO_INT_DICT = {'unknown':0, 'vehicle':1, 'pedestrian':2, 'sign':3, 'cyclist':4}

if __name__ == "__main__":

    random.seed(0)
    np.random.seed(0)

    root_path = '/media/hao/MyPSSD/waymo-v2'


    #### the following code is to make a subset of the large waymo_infos_train.pkl!

    data = pickle.load(open('/home/hao/dataset/waymo/FH/waymo_infos_validation.pkl', 'rb'))
    data = data[0:-1:100]
    pickle.dump(data, open('/home/hao/dataset/waymo/FH/waymo_infos_val_subset.pkl', 'wb'))

    data = pickle.load(open('/home/hao/dataset/waymo/FH/waymo_infos_train.pkl', 'rb'))
    data = data[0:-1:10]
    pickle.dump(data, open('/home/hao/dataset/waymo/FH/waymo_infos_train_subset.pkl', 'wb'))




    #### the following code is to find a sequence that have cyclist!
    if 0:
        for split in ['train']:
            dataset_dir = os.path.join(root_path, split)

            scenes = list(glob.glob(os.path.join(dataset_dir, 'lidar/', '*.parquet')))

            for i in tqdm(range(len(scenes))):
                scene = scenes[i]
                context_name = scene.split('/')[-1].split('.')[0]

                lidar_box_df = read(dataset_dir, context_name, 'lidar_box')
                lidar_box_df = (
                    lidar_box_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros'])
                        .agg(list)
                        .reset_index()
                )

                box_types_all = []
                for index, row in lidar_box_df.iterrows():
                    lidar_box = v2.LiDARBoxComponent.from_dict(row)
                    # box_types = [type for type in lidar_box.type]
                    box_types_all += lidar_box.type

                b_have_cyclist = np.array(box_types_all) == 4

                if b_have_cyclist.any():
                    print(context_name)








    #### the following code is to examine all the calibration files in the whole waymo dataset
    if 0:
        debug_file_name = 'FH_test0_data.pickle'
        if not os.path.exists(debug_file_name):

            calibration_all = np.zeros((1000,16))
            global_iter = 0

            for split in ['train', 'validation']:
                dataset_dir = os.path.join(root_path, split)

                scenes = list(glob.glob(os.path.join(dataset_dir, 'lidar/', '*.parquet')))

                for i in tqdm(range(len(scenes))):
                    scene = scenes[i]

                    context_name = scene.split('/')[-1].split('.')[0]

                    lidar_calibration_df = read(dataset_dir, context_name, 'lidar_calibration')
                    lidar_calibrations = {}
                    for j, row in lidar_calibration_df.iterrows():
                        lidar_calibration_data = v2.LiDARCalibrationComponent.from_dict(row)
                        lidar_calibrations[lidar_calibration_data.key.laser_name] = lidar_calibration_data

                    top_lidar_extrinsic = lidar_calibrations[dataset_pb2.LaserName.TOP].extrinsic.transform

                    calibration_all[global_iter,:] = top_lidar_extrinsic
                    global_iter += 1

            pickle.dump({"calibration_all": calibration_all},
                        open(debug_file_name, 'wb'))
        else:
            data = pickle.load(open(debug_file_name, 'rb'))
            calibration_all = data['calibration_all']



        print(calibration_all.min(axis=0))
        print(calibration_all.max(axis=0))

