import numpy as np
import open3d as o3d
import math
import os
# from CTools.CVisualizer import CVisualizer
# vis = CVisualizer()

class CDataLoaderAX7:
    def load_pcd_xyz_wo_nan(self,pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file,remove_nan_points=True)
        points = np.asarray(pcd.points,dtype=np.float32)
        return points

    def load_pcd_xyz_with_nan(self,pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points,dtype=np.float32)
        return points

    def load_pcd_xyzi(self,pcd_file):
        pcd = o3d.t.io.read_point_cloud(pcd_file)
        points = pcd.point["positions"].numpy()
        intensity = pcd.point["intensity"].numpy()
        return points,intensity

    def load_pcd_xyzrgb(self,pcd_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        points = np.asarray(pcd.points,dtype=np.float32)
        colors = np.asarray(pcd.colors,dtype=np.float32)
        return points,colors

    def load_bin_xyzi(self,bin_file):
        points_bin = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        return points_bin

    def load_bin_xyzi_crop(self,bin_file,roi):
        points_bin = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        print("original points number(without remissions): ", points_bin.shape)
        x = points_bin[:, 0]
        y = points_bin[:, 1]
        z = points_bin[:, 2]
        dist_to_origin = x * x + y * y
        dist_to_origin_ind = np.where(dist_to_origin > roi * roi)
        print("outlier number: ", dist_to_origin_ind[0].shape)
        points_bin[dist_to_origin_ind, 0:3] = 0
        return points_bin

    def load_bin_label(self,label_file):
        label = np.fromfile(label_file, dtype=np.uint32)
        return label

    def load_bin_semlabel(self,label_file):
        label = np.fromfile(label_file, dtype=np.uint32)
        semlabel = label & 0xFFFF
        return semlabel

    def load_calc_slam_pose(self,pose_file):
        pose_all = np.loadtxt(pose_file)
        pose_all.reshape((-1, 7))
        timestamp = pose_all[:, 0]
        slam_pose = pose_all[:, 1:7]
        XYZ = pose_all[:,1:4]
        YPR = pose_all[:,4:7]
        return timestamp,slam_pose,XYZ,YPR

    def load_mapper_pose(self,pose_file):
        pose_all = np.loadtxt(pose_file)
        pose_all.reshape((-1, 13))
        timestamp = pose_all[:, 0]
        slam_pose = pose_all[:, 1:7]
        gps_pose = pose_all[:, 7:13]
        XYZ = pose_all[:,1:4]
        R = pose_all[:, 4]
        P = pose_all[:, 5]
        Y = pose_all[:, 6] - math.pi / 2
        YPR = np.column_stack((Y, P, R))
        return timestamp,slam_pose,gps_pose,XYZ,YPR

    def transform_to_kitti_coord(self,pcd_file):
        points_ax7 = self.load_pcd_xyz_wo_nan(pcd_file)
        x = -points_ax7[:,1]
        y = points_ax7[:,0]
        z = points_ax7[:,2]
        points_kitti = np.column_stack((x,y,z))
        return points_ax7,points_kitti

    def pcd_to_bin(self,pcd_file,bin_file,bool_output=False):
        # nx3 -> (nx4)x1
        pcd_points = self.load_pcd_xyz_wo_nan(pcd_file)
        ones = np.ones((pcd_points.shape[0]),dtype=np.float32)
        bin_points = np.column_stack((pcd_points,ones))
        bin_points = bin_points.reshape(-1)
        if bool_output:
            bin_points.tofile(bin_file)
        return bin_points

    def bin_to_pcd(self,bin_file,pcd_file,bool_output=False):
        bin_points = np.fromfile(bin_file,dtype=np.float32).reshape((-1,4))
        pcd_points = bin_points[:,0:3]

        if bool_output:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            o3d.io.write_point_cloud(pcd_file, pcd)
        return pcd_points




if __name__ == '__main__':
    loader = CDataLoaderAX7()
    pcd_file = '/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/2022-01-12-16-48-31/LidarData/1641977312551.pcd'
    pcd_rgb_file = '/home/alv-2/datasets/ax7_with_label/velodyne/1641977312551.pcd'

    points_with_nan = loader.load_pcd_xyz_with_nan(pcd_file)
    points_wo_nan = loader.load_pcd_xyz_wo_nan(pcd_file)
    points,intensity = loader.load_pcd_xyzi(pcd_file)
    points,colors = loader.load_pcd_xyzrgb(pcd_rgb_file)

    AX7_Jan_dir = "/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/2022-01-12-16-48-31"
    AX7_May_dir = "/media/alv-2/Rambo/Data/2022-AX7/2022-05-23"

    lidar_spec_ini = os.path.join(AX7_Jan_dir, "params/lidar_specification2.ini")

    slam_file = '/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/2022-01-12-16-48-31/calculated_slam_all_pose.txt'
    timestamp,slam_pose,XYZ,YPR = loader.load_calc_slam_pose(slam_file)

    mapper_file = '/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/2022-01-12-16-48-31/meta_data/mapper.txt'
    timestamp, slam_pose, gps_pose,XYZ,YPR = loader.load_mapper_pose(mapper_file)

    points_ax7,points_kitti = loader.transform_to_kitti_coord(pcd_file)
    # vis.vis_two_Cloud(points_ax7,points_kitti)

    bin_file = "./test.bin"
    bin_points = loader.pcd_to_bin(pcd_file,bin_file)
    print(1)
