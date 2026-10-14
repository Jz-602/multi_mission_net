import math

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from CTools.CPoseProcesserKITTI import CPoseProcesserKITTI
from CTools.CVisualizer import CVisualizer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
vis = CVisualizer()
pose_processer = CPoseProcesserKITTI()

class CTrajectory():

    def plot_traj_2d(self,trajectory, interval):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], 'r-', lw=2)
        ax.set_xlim([np.min(trajectory[:, 0]), np.max(trajectory[:, 0])])
        ax.set_ylim([np.min(trajectory[:, 1]), np.max(trajectory[:, 1])])

        def update_line(num, trajectory, line):
            line.set_data(trajectory[:num, 0], trajectory[:num, 1])
            return line,

        def animate():
            inter = interval
            num_frames = len(trajectory)
            ani = animation.FuncAnimation(fig, update_line, frames=num_frames, fargs=(trajectory, line),
                                          interval=inter, blit=True, repeat=False)
            plt.show()

        animate()

    def plot_traj_3d(self,trajectory, interval):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        line, = ax.plot([], [], [], 'r-', lw=2)
        ax.set_xlim3d(np.min(trajectory[:, 0]), np.max(trajectory[:, 0]))
        ax.set_ylim3d(np.min(trajectory[:, 1]), np.max(trajectory[:, 1]))
        ax.set_zlim3d(np.min(trajectory[:, 2]), np.max(trajectory[:, 2]))

        def update_line(num, trajectory, line):
            line.set_data(trajectory[:num, 0], trajectory[:num, 1])
            line.set_3d_properties(trajectory[:num, 2])
            return line,

        def animate():
            inter = interval
            num_frames = len(trajectory)
            ani = animation.FuncAnimation(fig, update_line, frames=num_frames, fargs=(trajectory, line),
                                          interval=inter, blit=True, repeat=False)
            plt.show()
        animate()

    @staticmethod
    def extract_pose_center(poses_lidar_homo):
        start_frame = 0
        if type(poses_lidar_homo) == list:
            end_frame = len(poses_lidar_homo)
        elif type(poses_lidar_homo) == np.ndarray:
            end_frame = poses_lidar_homo.shape[0]

        translation = []
        for i in range(start_frame, end_frame):
            print("%d / %d" % (i, end_frame))
            T = poses_lidar_homo[i]
            origin = np.array([0, 0, 0], dtype=np.float32)
            origin_transform = T[0:3, 0:3] @ origin + T[0:3, 3]
            translation.append(origin_transform)
        trajectory = np.array(translation)
        return trajectory

    @staticmethod
    def convert_Rt_to_T(R_sequence, t_sequence):
        T_sequence = []
        for i in range(len(R_sequence)):
            T = np.eye(4)
            T[:3, :3] = R_sequence[i]
            T[:3, 3] = t_sequence[i]
            T_sequence.append(T)
        return T_sequence


    def aug_pose_by_trans_left_right(self,poses_lidar_homo,trans_dist,bool_vis=False):
        start_frame = 0
        if type(poses_lidar_homo) == list:
            end_frame = len(poses_lidar_homo)
        elif type(poses_lidar_homo) == np.ndarray:
            end_frame = poses_lidar_homo.shape[0]
        translation = []
        aug_right = []
        aug_left = []
        for i in range(start_frame, end_frame):
            # print("%d / %d" % (i,end_frame))
            T = poses_lidar_homo[i]
            origin = np.array([0, 0, 0], dtype=np.float32)
            origin_transform = T[0:3, 0:3] @ origin + T[0:3, 3]
            x_right = origin_transform + np.array([trans_dist / 2, 0, 0], dtype=np.float32)
            x_left = origin_transform - np.array([trans_dist / 2, 0, 0], dtype=np.float32)
            aug_right.append(x_right)
            aug_left.append(x_left)
            translation.append(origin_transform)

        trajectory_all = translation + aug_left + aug_right
        trajectory = np.array(translation)
        aug_right = np.array(aug_right)
        aug_left = np.array(aug_left)
        trajectory_all = np.array(trajectory_all)
        if bool_vis:
            self.plot_traj_2d(trajectory_all,interval=1)
        return trajectory,aug_right,aug_left,trajectory_all


    def extract_keyframes_by_interval(self,poses_lidar_homo,interval):
        translation = self.extract_pose_center(poses_lidar_homo)
        translation_prev = translation[0:-1]
        translation_next = translation[1:]
        diff = translation_next - translation_prev
        dist_between_frames = np.sqrt(diff[:,0] ** 2 + diff[:,1] ** 2 + diff[:,2] ** 2)
        dist_to_first_frame = np.sqrt(translation_next[:,0] ** 2 + translation_next[:,1] ** 2 + translation_next[:,2] ** 2)
        return dist_between_frames,dist_to_first_frame


if __name__ == '__main__':
    traj = CTrajectory()
    seq_id = '07'
    pose_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/' +seq_id+ '/poses.txt'
    calib_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/' +seq_id+ '/calib.txt'
    poses_lidar_homo = pose_processer.read_pose_calib_to_Ts(pose_file ,calib_file)
    translation = traj.extract_pose_center(poses_lidar_homo)
    # traj.plot_traj_3d(translation,interval=1)


    # from CTools.CPoseProcesserAX7 import CPoseProcesserAX7
    # pose_file = "/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/" \
    #             "2022-01-12-16-48-31/meta_data/mapper.txt"
    # map_center_file = "/media/alv-2/Rambo/Data/2022-AX7/2022-01-12/" \
    #                   "2022-01-12-16-48-31/meta_data/mapper_center.txt"
    # pos = CPoseProcesserAX7()
    # Rs, ts = pos.read_mapper_pose_to_RT(pose_file, map_center_file)
    # poses_lidar_homo = traj.convert_Rt_to_T(Rs, ts)
    # poses_lidar_homo = np.asarray(poses_lidar_homo)
    # translation = traj.extract_pose_center(poses_lidar_homo)
    # # traj.plot_traj_2d(translation,interval=1)
    #
    # trajectory,aug_right,aug_left,trajectory_all = traj.aug_pose_by_trans_left_right(poses_lidar_homo=poses_lidar_homo,trans_dist=10,bool_vis=False)

    dist_between_frames,dist_to_first_frame = traj.extract_keyframes_by_interval(poses_lidar_homo,interval=5)
    threshold = 0.5
    keyframes_ind = np.argwhere(dist_between_frames > threshold).squeeze(1)
    poses_lidar_homo_keyframe = poses_lidar_homo[keyframes_ind]
    print(1)
    print(1)
