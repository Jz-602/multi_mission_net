import numpy as np
import math
from CTools.CFileProcesser import CFileProcesser
from CTools.CMapperAX7 import CMapperAX7

class CPoseProcesserAX7(CFileProcesser):
    """
    1. AX7 Mapper
        calculated_all_slam_pose.txt
        mapper.txt
    2. KITTI Mapper
    3. AX7 -> KITTI
    4. KITTI -> AX7
    """

    def ypr_2_rot_calc_slam(self,theta, format='radian'):
        """
        :param theta: calc_all_slam_pose.txt中对应的YPR弧度
        :param format: 指定的角度单位
        :return: YPR对应的旋转矩阵
        """

        if format == 'degree':
            theta = [i * math.pi / 180.0 for i in theta]

        Y,P,R = theta[0],theta[1],theta[2]

        crz,srz = math.cos(Y),math.sin(Y)
        crx,srx = math.cos(P),math.sin(P)
        cry,sry = math.cos(R),math.sin(R)

        R_y = np.array([[cry, 0, sry],
                        [0, 1, 0],
                        [-sry, 0, cry]
                        ])

        R_x = np.array([[1, 0, 0],
                        [0, crx, -srx],
                        [0, srx, crx]
                        ])

        R_z = np.array([[crz, -srz, 0],
                        [srz, crz, 0],
                        [0, 0, 1]
                        ])

        R_w_l = R_z @ R_x @ R_y
        return R_w_l

    def read_calc_slam_pose_to_RT(self,all_pose_file,map_center_file):
        """
        :param all_pose_file: calculated_all_slam_pose.txt 位姿文件
        :param map_center_file: mapper_center.txt 期望的地图中心
        :return: Rs,Ts 每一帧对应的Rt矩阵
        """
        pose_all = np.loadtxt(all_pose_file)
        center = np.loadtxt(map_center_file)
        timestamp = pose_all[:, 0]
        slam_pose = pose_all[:, 1:7]

        Rs = []
        ts = []
        for iter in range(slam_pose.shape[0]):
            pose = slam_pose[iter, :]

            YPR = [pose[3], pose[4], pose[5]]
            R_w_l = self.ypr_2_rot_calc_slam(YPR, format="radian")
            Rs.append(R_w_l)

            t_w_l = [pose[0], pose[1], pose[2]] - center
            ts.append(t_w_l)
        return Rs, ts

    def read_mapper_pose_to_RT(self,mapper_pose_file,map_center_file):
        """
        :param mapper_pose_file: mapper.txt 位姿文件
        :param map_center_file: mapper_center.txt 期望的地图中心
        :return: Rs,Ts 每一帧对应的Rt矩阵
        """
        pose_all = np.loadtxt(mapper_pose_file)
        center = np.loadtxt(map_center_file)
        timestamp = pose_all[:, 0]
        slam_pose = pose_all[:, 1:7]

        Rs = []
        ts = []
        for iter in range(slam_pose.shape[0]):
            pose = slam_pose[iter, :]

            YPR = [pose[5] - math.pi/2, pose[4], pose[3]]
            R_w_l = self.ypr_2_rot_calc_slam(YPR, format="radian")
            Rs.append(R_w_l)

            t_w_l = [pose[0], pose[1], pose[2]] - center
            ts.append(t_w_l)
        return Rs, ts


if __name__ == '__main__':
    pos = CPoseProcesserAX7()
    mapperAX7 = CMapperAX7()

    # print(pos.__bases__)

    pose_file = "/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/" \
                "2022-01-12-16-48-31/calculated_slam_all_pose.txt"
    map_center_file = "/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/" \
                      "2022-01-12-16-48-31/meta_data/mapper_center.txt"
    scan_file_path = "/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/2022-01-12-16-48-31/LidarData"

    Rs,ts = pos.read_calc_slam_pose_to_RT(pose_file,map_center_file)
    scan_files = pos.get_filenames(scan_file_path)
    start_frame = 400
    end_frame = 410
    map = mapperAX7.mapper_with_RT_xyz(scan_files,Rs,ts,start_frame,end_frame,bool_vis=True)



    print(1)


