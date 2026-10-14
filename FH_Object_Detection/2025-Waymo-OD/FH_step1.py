
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
# from waymo_open_dataset.v2.perception.utils import lidar_utils
import FH_lidar_utils as lidar_utils
# in the original version, convert_range_image_to_point_cloud will return [x, y, z, range, intensity, elongation]
# in my implementation, convert_range_image_to_point_cloud will return [x, y, z, intensity, elongation, nlz]. Then I can use the nlz flag to remove those points in the nlz


import open3d as o3d


import numpy as np


import glob
import os
from tqdm import tqdm

import pickle
import copy

# from CTools.CVisualizer import CVisualizer
# from CTools.CFileProcesser import CFileProcesser

from CoordTransformation import RotationMatrix2YPRV2


TYPE_LIST = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']


def read(dataset_dir, context_name, tag: str) -> dd.DataFrame:
  paths = tf.io.gfile.glob(f'{dataset_dir}/{tag}/{context_name}.parquet')
  return dd.read_parquet(paths)



def extract_objects(lidar_boxes, pose, top_lidar_extrinsic):
    objects = []

    for i in range(len(lidar_boxes.type)):
        if 0:
            ######################## the original version ##########################
            category_label = lidar_boxes.type[i]
            box = lidar_boxes.box

            num_lidar_points_in_box = lidar_boxes.num_lidar_points_in_box[i]
            speed = np.array(
                [lidar_boxes.speed.x[i], lidar_boxes.speed.y[i]]).reshape([2, 1])

            speed = pose[:2, :2].T @ speed  # transform from world coordinate to vehicle coordinate???

            objects.append({
                'id': lidar_boxes.key.laser_object_id[i],
                'label': TYPE_LIST[category_label],
                'box': np.array([box.center.x[i], box.center.y[i], box.center.z[i],
                                box.size.x[i], box.size.y[i], box.size.z[i],
                                speed[0][0], speed[1][0],
                                box.heading[i]],
                                dtype=np.float32),
                'num_points': num_lidar_points_in_box})
        else:
            ######################## my modified version ##########################
            objects.append({
                'id': lidar_boxes.key.laser_object_id[i],
                'type': lidar_boxes.type[i],
                # 'label': TYPE_LIST[lidar_boxes.type[i]],
                'difficulty_level': np.array([lidar_boxes.difficulty_level.detection[i], lidar_boxes.difficulty_level.tracking[i]]),
                'num_points_in_top_lidar': lidar_boxes.num_top_lidar_points_in_box[i],
                'num_points_in_all_lidar': lidar_boxes.num_lidar_points_in_box[i],
                'speed': np.array([lidar_boxes.speed.x[i], lidar_boxes.speed.y[i], lidar_boxes.speed.z[i]]),
                'box': np.array([lidar_boxes.box.center.x[i] - top_lidar_extrinsic[3], lidar_boxes.box.center.y[i] - top_lidar_extrinsic[7],
                                 lidar_boxes.box.center.z[i] - top_lidar_extrinsic[11],
                                lidar_boxes.box.size.x[i], lidar_boxes.box.size.y[i], lidar_boxes.box.size.z[i],
                                lidar_boxes.box.heading[i]],
                                dtype=np.float32)})

    return objects



def convert(scene, dataset_dir, save_root, b_save_lidar_points):

    context_name = scene.split('/')[-1].split('.')[0]

    save_root = os.path.join(save_root, context_name)
    os.makedirs(save_root, exist_ok=True)
    top_lidar_save_path = os.path.join(save_root, 'output/pcd')
    os.makedirs(top_lidar_save_path, exist_ok=True)
    other_lidar_save_path = os.path.join(save_root, 'output/pcd2')
    os.makedirs(other_lidar_save_path, exist_ok=True)

    lidar_df = read(dataset_dir, context_name, 'lidar')
    lidar_pose_df = read(dataset_dir, context_name, 'lidar_pose')

    lidar_calibration_df = read(dataset_dir, context_name, 'lidar_calibration')
    lidar_calibrations = {}
    for i, row in lidar_calibration_df.iterrows():
        lidar_calibration_data = v2.LiDARCalibrationComponent.from_dict(row)
        lidar_calibrations[lidar_calibration_data.key.laser_name] = lidar_calibration_data

    vehicle_pose_df = read(dataset_dir, context_name, 'vehicle_pose')

    lidar_box_df = read(dataset_dir, context_name, 'lidar_box')
    lidar_box_df = (
        lidar_box_df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros'])
            .agg(list)
            .reset_index()
    )
#################################################################################################
    df = v2.merge(lidar_df, lidar_pose_df, right_nullable = True)
    df = v2.merge(df, vehicle_pose_df)
    df = (
        df.groupby(['key.segment_context_name', 'key.frame_timestamp_micros'])
            .agg(list)
            .reset_index()
    )

    df = v2.merge(df, lidar_box_df)



    all_frames = []
    frame_id = 0
    for index, row in df.iterrows():


        vehicle_pose_data = v2.VehiclePoseComponent.from_dict(row)
        lidar_box = v2.LiDARBoxComponent.from_dict(row)


        vehicle_pose_data = v2.VehiclePoseComponent(key=vehicle_pose_data.key, world_from_vehicle=column_types.Transform(vehicle_pose_data.world_from_vehicle.transform[0]))

        #### transform the lidar points from vehicle coordinate to toplidar coordinate. the lidar points will be transfromed from lidar-coordinate to vehicle coordinate
        # with calibration.extrinsics, then from vehicle coordinate to world coordinate with pose_range_image, then from world back to vehicle (or toplidar coordinate) with vehicle_pose_data
        top_lidar_extrinsic = lidar_calibrations[dataset_pb2.LaserName.TOP].extrinsic.transform
        Tr_vehicle_toplidar = np.array([(1,0,0,top_lidar_extrinsic[3]),(0,1,0,top_lidar_extrinsic[7]),(0,0,1,top_lidar_extrinsic[11]),(0,0,0,1)])
        Tr_world_vehicle = np.reshape(np.array(vehicle_pose_data.world_from_vehicle.transform), [4, 4])
        Tr_world_toplidar = Tr_world_vehicle @ Tr_vehicle_toplidar
        vehicle_pose_data = v2.VehiclePoseComponent(key=vehicle_pose_data.key,
                                                    world_from_vehicle=column_types.Transform(Tr_world_toplidar.ravel()))

        if b_save_lidar_points:
            lidar_data = v2.LiDARComponent.from_dict(row)
            lidar_pose_data = v2.LiDARPoseComponent.from_dict(row)
            other_lidar_points = None
            top_lidar_points = None
            pose_range_image = _v2_lidar.PoseRangeImage(values=lidar_pose_data.range_image_return1.values[0],
                                                        shape=lidar_pose_data.range_image_return1.shape[0])

            for i in range(len(lidar_data.key.laser_name)):
                if lidar_data.key.laser_name[i] == dataset_pb2.LaserName.TOP:
                    top_lidar_points = lidar_utils.convert_range_image_to_point_cloud(_v2_lidar.RangeImage(values=lidar_data.range_image_return1.values[i],
                                                             shape=lidar_data.range_image_return1.shape[i]),
                                                                                   lidar_calibrations[lidar_data.key.laser_name[i]],
                                                                                   pose_range_image,
                                                                                   vehicle_pose_data,
                                                                                   True)
                    points2 = lidar_utils.convert_range_image_to_point_cloud(_v2_lidar.RangeImage(values=lidar_data.range_image_return2.values[i],
                                                             shape=lidar_data.range_image_return2.shape[i]),
                                                                                   lidar_calibrations[lidar_data.key.laser_name[i]],
                                                                                   pose_range_image,
                                                                                   vehicle_pose_data,
                                                                                   True)

                    top_lidar_points = np.concatenate(tuple([top_lidar_points, points2]), axis=0)
                    top_lidar_points = top_lidar_points[top_lidar_points[:, -1] == -1, :3]  # remove points in the no-label zone
                    # top_lidar_points = top_lidar_points[top_lidar_points[:, -1] == -1, :4]  # intensity will be saved
                else:
                    points = lidar_utils.convert_range_image_to_point_cloud(
                        _v2_lidar.RangeImage(values=lidar_data.range_image_return1.values[i],
                                             shape=lidar_data.range_image_return1.shape[i]),
                        lidar_calibrations[lidar_data.key.laser_name[i]],
                        None, None, True)
                    points2 = lidar_utils.convert_range_image_to_point_cloud(
                        _v2_lidar.RangeImage(values=lidar_data.range_image_return2.values[i],
                                             shape=lidar_data.range_image_return2.shape[i]),
                        lidar_calibrations[lidar_data.key.laser_name[i]],
                        None, None, True)

                    points = np.concatenate(tuple([points, points2]), axis=0)
                    points = points[points[:, -1] == -1, :3]  # remove points in the no-label zone
                    # points = points[points[:, -1] == -1, :4]  # intensity will be saved

                    if other_lidar_points is None:
                        other_lidar_points = points
                    else:
                        other_lidar_points = np.concatenate(tuple([points, other_lidar_points]), axis=0)

            # transform from vehicle coordinate to toplidar coordinate. as rotation is identity matrix, only need to perform the inverse translation
            other_lidar_points[:,:3] -= np.array([(top_lidar_extrinsic[3],top_lidar_extrinsic[7],top_lidar_extrinsic[11])])


            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(top_lidar_points)
            o3d.io.write_point_cloud(os.path.join(top_lidar_save_path, str(lidar_box.key.frame_timestamp_micros)[:-3] + '.pcd'), new_pcd)

            new_pcd.points = o3d.utility.Vector3dVector(other_lidar_points)
            o3d.io.write_point_cloud(os.path.join(other_lidar_save_path, str(lidar_box.key.frame_timestamp_micros)[:-3] + '.pcd'), new_pcd)


        token = '-'.join((context_name, str(lidar_box.key.frame_timestamp_micros)))

        pose = np.reshape(np.array(vehicle_pose_data.world_from_vehicle.transform), [4, 4])
        objects = extract_objects(lidar_box, pose, top_lidar_extrinsic)

        info = {
            'token': token,
            'pose': pose,
            'sample_idx': frame_id,
            'sequence_name': context_name,
            'frame_timestamp_micros': lidar_box.key.frame_timestamp_micros,
            'objects': objects,
        }

        all_frames.append(copy.deepcopy(info))


        frame_id += 1

    with open(os.path.join(save_root, 'output/detection_info.pkl'), 'wb') as f:
        pickle.dump(all_frames, f)

    return all_frames

if __name__ == '__main__':

    random.seed(0)
    np.random.seed(0)

    root_path = '/home/hao/dataset/waymo'
    save_path = '/home/hao/dataset/waymo/FH'
    os.makedirs(save_path, exist_ok=True)

    b_save_lidar_points = True

    for split in ['train', 'val']:
    # for split in ['val']:
        dataset_dir = os.path.join(root_path, split)

        scenes = list(glob.glob(os.path.join(dataset_dir, 'lidar/', '*.parquet')))
        scenes.sort()

        if 1:
            all_infos = []
            for i in tqdm(range(len(scenes))):
            # for i in tqdm(range(3)):
                all_infos += convert(scenes[i], dataset_dir, save_path, b_save_lidar_points)

            with open(os.path.join(save_path, 'waymo_infos_' + split + '.pkl'), 'wb') as f:
                pickle.dump(all_infos, f)

        if 1:
            for i in tqdm(range(len(scenes))):
                context_name = scenes[i].split('/')[-1].split('.')[0]
                save_root = os.path.join(save_path, context_name)
                pickle_filename = os.path.join(save_root, 'output/detection_info.pkl')
                data = pickle.load(open(pickle_filename, 'rb'))

                with open(os.path.join(save_root, 'output/pose_slam_euler_waymo.txt'), "w") as f:
                    for j in range(len(data)):
                        trans_mat = data[j]['pose']
                        ypr = RotationMatrix2YPRV2(trans_mat[:3,:3])
                        f.write(f"{str(data[j]['frame_timestamp_micros'])[:-3]} {trans_mat[0,3]} {trans_mat[1,3]} {trans_mat[2,3]} {ypr[2]} {ypr[1]} {ypr[0]}\n")

                os.system(f"cp {save_root}/output/pose_slam_euler_waymo.txt {save_root}/output/pose_slam_euler.txt")


