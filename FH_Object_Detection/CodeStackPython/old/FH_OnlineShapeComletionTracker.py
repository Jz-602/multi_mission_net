

from pyquaternion import Quaternion
import numpy as np
import os
from tqdm import tqdm
import torch
import imageio

from NN_ShapeCompletion import Model

# import pandas as pd

import open3d as o3d
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pylab as pylab


import math

import struct

import cv2


def YPR2RotationMatrix(ypr):
    # our coordinate: x->right->pitch, y->forward->roll, z->up->yaw
    Y, P, R = ypr[0], ypr[1], ypr[2]

    crz, srz = math.cos(Y), math.sin(Y)
    crx, srx = math.cos(P), math.sin(P)
    cry, sry = math.cos(R), math.sin(R)

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

def R2Ang_3D_Our(R):
    # Z1X2Y3, applicable for our coordinate (X->right, Y->forward, Z->upward)
    #    angle[0] = yaw; angle[1] = pitch; angle[2] = roll
    angle = np.empty((3,1))
    angle[1] = math.asin(R[2,1]);
    cp = math.cos(angle[1]);
    angle[2] = -math.atan2((R[2,0]/cp),(R[2,2]/cp));
    angle[0] = -math.atan2((R[0,1]/cp),(R[1,1]/cp));
    return angle

def Rot_z(angle):
    crz, srz = math.cos(angle), math.sin(angle)
    R_z = np.array([[crz, -srz, 0],
                    [srz, crz, 0],
                    [0, 0, 1]
                    ])
    return R_z




class Box:
    def __init__(self, center, size, orientation):
        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation

    def RotationMatrix(self):
        return self.orientation.rotation_matrix

    def Translate(self, x):
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        :return: <None>.
        """
        self.center += x

    def Rotate(self, quaternion: Quaternion):
        """
        Rotates box.
        :param quaternion: <Quaternion>. Rotation to apply.
        :return: <None>.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation


    def Corners(self, wlh_factor: float=1.0):
        """
        Returns the bounding box corners.
        :param wlh_factor: <float>. Multiply w, l, h by a factor to inflate or deflate the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, l, h = self.wlh * wlh_factor

        # # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        # x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        # y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        # z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])

        # 3D bounding box corners. (Ours: x points right, y forward, z up.)
        x_corners = w / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        y_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])

        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners



class OurRS128TrackerTestDataset():
    def __init__(self, param_file):
        with open(param_file, 'r') as f:
            for line in f.readlines():
                values = line.split()
                if len(values) == 0:
                    continue
                # elif values[0] == 'start_frame':
                #     self.start_frame = values[1]
                elif values[0] == 'm_LoadPath':
                    self.LiDARDataPath = values[1]
                elif values[0] == 'start_frame':
                    self.start_frame = int(values[1])
                elif values[0] == 'vehicle_pose':
                    self.vehicle_pose = np.array(values[1:], dtype=np.float32)
                elif values[0] == 'vehicle_wlh':
                    self.vehicle_wlh = np.array(values[1:], dtype=np.float32)

        data = np.loadtxt(os.path.join(self.LiDARDataPath, 'calculated_slam_all_pose.txt'), dtype=np.float64).reshape(-1, 7)
        self.allframe_timestamp_DB = data[:,0].astype(int)
        self.allframe_pose_DB = data[:,1:]


    def NumOfFrames(self):
        return len(self.allframe_timestamp_DB)


    def __getitem__(self, index):
        if index >= self.NumOfFrames() or index < self.start_frame:
            return False

        lidar_filename = os.path.join(self.LiDARDataPath, 'LidarData', f'{self.allframe_timestamp_DB[index]}.pcd')
        lidar_pcd = o3d.io.read_point_cloud(lidar_filename)

        points = np.asarray(lidar_pcd.points, dtype=np.float32).T

        if index == self.start_frame:
            R_l_v = YPR2RotationMatrix(np.array(self.vehicle_pose[3:]*math.pi/180.0))
            t_l_v = np.array([self.vehicle_pose[0], self.vehicle_pose[1], self.vehicle_pose[2]])
            BB = Box(t_l_v, self.vehicle_wlh, Quaternion(matrix = R_l_v))
            return PointCloud(points), BB
        else:
            return PointCloud(points), None


def CreateO3dboxFromBoxCorners(corners, color=None):
	'''
	corners: 8 corners(x, y, z)
	corners: array = 8*3
    # x->forward, y->left, z->up
	#         0 -------- 1
	#        /|         /|
	#       4 -------- 5 .
	#       | |        | |
	#       . 3 -------- 2
	#       |/         |/
	#       7 -------- 6
	'''
	lines = [[0, 1], [1, 2], [2, 3], [3, 0],
				[4, 5], [5, 6], [6, 7], [7, 4],
				[0, 4], [1, 5], [2, 6], [3, 7],
             [5, 7], [4, 6]]
	if color == None:
		color = [1, 0, 0] # red
	colors = [color for i in range(len(lines))]
	line_set = o3d.geometry.LineSet()
	line_set.points = o3d.utility.Vector3dVector(corners)
	line_set.lines = o3d.utility.Vector2iVector(lines)
	line_set.colors = o3d.utility.Vector3dVector(colors)

	return line_set


def VisualizePCandBB(PC, BB):
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window()
    vis2.get_render_option().point_size = 2  # set points size

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis2.add_geometry(mesh_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(PC.points.T)
    vis2.add_geometry(pcd)

    o3d_box = CreateO3dboxFromBoxCorners(BB.Corners().T)
    vis2.add_geometry(o3d_box)

    vis2.run()




class CBEVer():
    def __init__(self,xyz,img_size = [640,640],bound=[-25,25,-25,25,-2.73,1.27]):
        self.bound = bound
        self.xyz = xyz
        self.img_size = img_size
        self.resolution = (self.bound[1] - self.bound[0]) / self.img_size[0]
    def extract_inlier(self):
        minX,maxX,minY,maxY,minZ,maxZ = self.bound
        mask = np.where(
            (self.xyz[:, 0] >= minX) &
            (self.xyz[:, 0] < maxX) &
            (self.xyz[:, 1] >= minY) &
            (self.xyz[:, 1] < maxY) &
            (self.xyz[:, 2] >= minZ) &
            (self.xyz[:, 2] < maxZ)
        )
        self.xyz = self.xyz[mask]
        return mask

    def xyz_to_bev(self):
        xyz = np.copy(self.xyz)
        xyz[:, 0] = np.int_(np.floor((xyz[:, 0] - self.bound[0])  / self.resolution))
        xyz[:, 1] = self.img_size[1] - 1 - np.int_(np.floor((xyz[:, 1] - self.bound[2]) / self.resolution))

        _, indices = np.unique(xyz[:, 0:2], axis=0, return_index=True)
        xyz_frac = xyz[indices]
        # indices = np.lexsort((-xyz[:, 2], xyz[:, 1], xyz[:, 0]))
        # xyz = xyz[indices]

        height_map = np.zeros((self.img_size[0], self.img_size[1]))
        max_height = float(np.abs(self.bound[5] - self.bound[4]))
        height_map[np.int_(xyz_frac[:, 1]), np.int_(xyz_frac[:, 0])] = xyz_frac[:, 2] / max_height
        # height_map = plt.get_cmap('Blues')(height_map)[:, :, :3]
        height_map = (255 * height_map).astype(np.uint8)
        return height_map

    def XY2UV(self, x, y):
        u = (x - self.bound[0]) / self.resolution
        v = self.img_size[1] - 1 - (y - self.bound[2]) / self.resolution
        return u, v


def VisualizePCandBBin2D(PC, BB):
    if 0:
        plt.figure()
        plt.axis([-30,30,-30,30])
        plt.scatter(PC.points[0,:], PC.points[1,:])

        # plt.Rectangle(xy=(0.2, 0.3), width=1.2, height=0.8, color='r',
        #               fill=False, linewidth=9)

        rect1 = plt.Rectangle(xy=(BB.center[0], BB.center[1]), width=BB.wlh[0], height=BB.wlh[1], color='r',
                      fill=False, linewidth=6)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(rect1)

        plt.axis('equal')
        plt.show()
    else:

        bever = CBEVer(PC.points.T, bound=[-30, 30, -30, 30, -2.73, 1.27])
        bever.extract_inlier()
        height_map = bever.xyz_to_bev()

        colored_height_map = np.zeros((height_map.shape[0], height_map.shape[1], 3), dtype='uint8')
        colored_height_map[:,:,0] = height_map
        colored_height_map[:,:,1] = height_map
        colored_height_map[:,:,2] = height_map


        u,v = bever.XY2UV(BB.center[0], BB.center[1])

        angle = R2Ang_3D_Our(BB.RotationMatrix())
        yaw = angle[0]


        rect = ((u, v), (BB.wlh[0]/bever.resolution, BB.wlh[1]/bever.resolution), int(-yaw*180.0/math.pi))   # in rotatedrect, angle is positive clockwise
        box = cv2.boxPoints(rect)  # cv2.cv.BoxPoints(rect) for OpenCV <3.x
        box = np.int0(box)
        cv2.drawContours(colored_height_map, [box], 0, (0, 0, 255), 2)

        cv2.imshow("bev_map", colored_height_map)
        print(yaw)
        print(int(-yaw*180.0/math.pi))
        cv2.waitKey(-1)




class VechicleModelDatabase:
    def __init__(self, model_filename):
        file_size = int(os.path.getsize(model_filename)/4)
        with open(model_filename, 'rb') as file:
            data = struct.unpack('f'*file_size, file.read(file_size*4))
            idx = 0
            self.range_list = []
            self.yaw_list = []
            self.pts_list = []
            while idx<len(data):
                range = data[idx]
                idx += 1
                yaw = data[idx]
                idx += 1
                num_of_pts = int(data[idx])
                idx += 1
                pts = np.array(data[idx:idx+num_of_pts*3])
                pts = pts.reshape(num_of_pts,3,order='F').T   # F: Fortran, same as MATLAB, column major
                idx += num_of_pts*3

                self.range_list.append(range)
                self.yaw_list.append(yaw)
                self.pts_list.append(pts)

        self.unique_range = np.asarray(list(set(self.range_list)))
        self.unique_yaw = np.asarray(list(set(self.yaw_list)))

    def GetModel(self, t_l_v, R_l_v):       # vehicle coord (box coord) to LiDAR coord
        idx = self.GetIdx(t_l_v, R_l_v)
        return self.pts_list[idx]

    def GetModelCode(self, t_l_v, R_l_v):
        idx = self.GetIdx(t_l_v, R_l_v)
        return self.model_code_all[:,idx]

    def GetIdx(self, t_l_v, R_l_v):
        xy_range = math.sqrt(t_l_v[0]*t_l_v[0] + t_l_v[1]*t_l_v[1])
        idx = np.abs(self.unique_range - xy_range).argmin()
        nearest_range = self.unique_range[idx]

        alpha = math.atan2(t_l_v[0], t_l_v[1]);
        R_ln_lo = Rot_z(alpha); # LiDAR-old coord to LiDAR-new coord
        R_ln_v = R_ln_lo @ R_l_v;
        angle = R2Ang_3D_Our(R_ln_v)
        yaw = angle[0] if angle[0] >= 0 else angle[0] + 2*math.pi
        idx = np.abs(self.unique_yaw - yaw).argmin()
        nearest_yaw = self.unique_yaw[idx]

        bool_array = np.logical_and(self.range_list == nearest_range, self.yaw_list == nearest_yaw)
        idx = np.squeeze(np.where(bool_array)[0])
        return idx

    def ComputeLatentCode(self, NN_model):
        self.model_code_all = np.empty((NN_model.bneck_size, len(self.pts_list)))
        for i in range(len(self.pts_list)):
            PC = PointCloud(self.pts_list[i])
            regularized_points = RegularizePC(PC, NN_model).cuda()
            model_code = NN_model.AE.encode(regularized_points)
            self.model_code_all[:,i] = np.squeeze(model_code.detach().cpu().numpy().T)



class PointCloud:

    def __init__(self, points):
        """
        Class for manipulating and viewing point clouds.
        :param points: <np.float: 4, n>. Input point cloud matrix.
        """
        self.points = points
        if self.points.shape[0] > 3:
            self.points = self.points[0:3, :]

    def Translate(self, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        :return: <None>.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def Rotate(self, rot_matrix):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        :return: <None>.
        """
        self.points[:3, :] = np.dot(rot_matrix, self.points[:3, :])

    def Normalize(self, wlh):
        normalizer = [wlh[1], wlh[0], wlh[2]]
        self.points = self.points / np.atleast_2d(normalizer).T


def CropPC(PC, box, offset=0, scale=1.0):

    maxi = np.max(box.Corners(scale), 1) + offset
    mini = np.min(box.Corners(scale), 1) - offset

    x_filt_max = PC.points[0, :] < maxi[0]
    x_filt_min = PC.points[0, :] > mini[0]
    y_filt_max = PC.points[1, :] < maxi[1]
    y_filt_min = PC.points[1, :] > mini[1]
    z_filt_max = PC.points[2, :] < maxi[2]
    z_filt_min = PC.points[2, :] > mini[2]

    close = np.logical_and(x_filt_min, x_filt_max)
    close = np.logical_and(close, y_filt_min)
    close = np.logical_and(close, y_filt_max)
    close = np.logical_and(close, z_filt_min)
    close = np.logical_and(close, z_filt_max)

    new_PC = PointCloud(PC.points[:, close])
    return new_PC


def CropAndCenterPC(PC, box, offset=0, scale=1.0, normalize=False):

    new_PC = CropPC(PC, box, offset=2 * offset, scale=4 * scale)

    new_box = copy.deepcopy(box)

    rot_mat = np.transpose(new_box.RotationMatrix())
    trans = -new_box.center

    # align data
    new_PC.Translate(trans)
    new_box.Translate(trans)
    new_PC.Rotate((rot_mat))
    new_box.Rotate(Quaternion(matrix=(rot_mat)))

    # crop around box
    new_PC = CropPC(new_PC, new_box, offset=offset, scale=scale)

    if normalize:
        new_PC.normalize(box.wlh)
    return new_PC


def RegularizePC(PC, model):
    PC = np.array(PC.points, dtype=np.float32)
    if np.shape(PC)[1] > 2:
        if PC.shape[0] > 3:
            PC = PC[0:3, :]
        if PC.shape[1] != model.input_size:
            new_pts_idx = np.random.randint(
                low=0, high=PC.shape[1], size=model.input_size, dtype=np.int64)
            PC = PC[:, new_pts_idx]
        PC = PC.reshape(1, 3, model.input_size)

    else:
        PC = np.zeros((1, 3, model.input_size))

    return torch.from_numpy(PC).float()


def FH_Subplots3(list_of_data):
    if len(list_of_data) == 3 or len(list_of_data) == 2:
        fig = plt.figure(figsize=plt.figaspect(0.2))
        for i in range(1,len(list_of_data)+1):
            ax = fig.add_subplot(1, len(list_of_data), i, projection='3d')
            data = list_of_data[i-1]
            ax.scatter(data[0,:], data[1,:], data[2,:], marker='o')
            plt.axis('equal')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        plt.show()


def GenerateBoxesFromSearchSpace(box, search_space=[[0, 0, 0]]):
    # Geenrate more candidate boxes based on prior and search space
    # Input : Prior position, search space and search size
    # Output : List of boxes

    candidate_boxes = []

    for offset in search_space:

        rot_quat = box.orientation
        trans = np.array(box.center)

        new_box = copy.deepcopy(box)

        new_box.Translate(-trans)
        new_box.Rotate(rot_quat.inverse)

        # REMOVE TRANSfORM
        if len(offset) == 3:
            new_box.Rotate(
                Quaternion(axis=[0, 0, 1], angle=offset[2] * np.pi / 180))
        new_box.Translate(np.array([offset[0], offset[1], 0]))

        # APPLY PREVIOUS TRANSFORMATION
        new_box.Rotate(rot_quat)
        new_box.Translate(trans)

        candidate_boxes.append(new_box)

    return candidate_boxes




if __name__ == '__main__':

    torch.manual_seed(1155)
    np.random.seed(0)

    model_database = VechicleModelDatabase('/home/hao/work/2023-CodeStack-DATMO/matlab/vehicle_template_2024.bin')
    path_results = './results_ours2/'
    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    scale_BB = 1.25

    chkpt_file = "./models/Ours/model_epoch50.pth.tar"

    # chkpt_file = '/media/hao/TuZhiming/models/Ours/model_epoch1.pth.tar'
    NN_model = Model(128, chkpt_file=chkpt_file, AE_chkpt_file="").cuda()
    NN_model.eval()

    model_database.ComputeLatentCode(NN_model)



    param_filename = '/home/hao/work/2023-CodeStack-DATMO/bin/parameters/DATMO.ini'

    dataset = OurRS128TrackerTestDataset(param_filename)

    x_space = np.linspace(-2, 2, 5)  # min, max, number of sample
    y_space = np.linspace(-2, 2, 5)  # min, max, number of sample
    a_space = np.linspace(-5, 5, 7)  # min, max, number of sample
    X, Y, A = np.meshgrid(x_space, y_space, a_space)  # create mesh grid
    search_space = np.array([X.flatten(), Y.flatten(), A.flatten()]).T


    last_BB = []
    for i in range(dataset.start_frame, dataset.NumOfFrames()):
        this_PC, this_BB = dataset[i]

        if i != dataset.start_frame:
            this_BB = last_BB

        ref_BB = this_BB

        if 1:
            VisualizePCandBBin2D(this_PC, this_BB)

        if 0:
            VisualizePCandBB(this_PC, ref_BB)

        if 0:
            candidate_PC = CropAndCenterPC(this_PC, this_BB, offset=0, scale=scale_BB,
                                           normalize=False)
            model_pts = model_database.GetModel(this_BB.center, this_BB.RotationMatrix())
            FH_Subplots3([candidate_PC.points, model_pts])


        model_code = model_database.GetModelCode(ref_BB.center, ref_BB.RotationMatrix())
        model_code = torch.Tensor(model_code.reshape(1,-1)).cuda()

        candidate_BBs = GenerateBoxesFromSearchSpace(ref_BB, search_space=search_space)
        candidate_PCs = [
            CropAndCenterPC(this_PC, box, offset=0, scale=scale_BB, normalize=False)
            for box in candidate_BBs
        ]
        candidate_PCs_reg = [RegularizePC(PC, NN_model) for PC in candidate_PCs]   # duplicate or remove points to fit the input of the network
        candidate_PCs_torch = torch.cat(candidate_PCs_reg, dim=0).cuda()

        output = NN_model.FastForward(candidate_PCs_torch, model_code)

        # store scores for all candidates
        scores = output.detach().cpu().numpy()
        idx = np.argmax(scores)  # select index of higest score
        box = candidate_BBs[idx]  # select box with highest score


        last_BB = copy.deepcopy(box)


        if 0:
            candidate_PC = CropAndCenterPC(this_PC, this_BB, offset=0, scale=scale_BB,
                                           normalize=False)
            model_pts = model_database.GetModel(this_BB.center, this_BB.RotationMatrix())

            best_PC = CropAndCenterPC(this_PC, box, offset=0, scale=scale_BB,
                                           normalize=False)

            FH_Subplots3([candidate_PC.points, model_pts, best_PC.points])

        ############################################ the following codes are all for visualization ####################################
        if 0:
            # keep highest score for angle space
            scores0 = scores[1::3]  # angle = 0
            scoresp10 = scores[2::3]  # angle = 10
            scoresm10 = scores[0::3]  # angle = -10
            scores = np.max(np.stack([scores0, scoresp10, scoresm10]), axis=0)

            X, Y = np.meshgrid(x_space, y_space)
            Z = scores.reshape(len(x_space), len(y_space))

            view_PC = CropAndCenterPC(this_PC, this_BB, offset=5)
            view_BB = Box([0, 0, 0], this_BB.wlh, Quaternion())

            # Crop point clouds around GT
            x_filt_max = view_PC.points[0, :] < 5
            x_filt_min = view_PC.points[0, :] > -5
            y_filt_max = view_PC.points[1, :] < 5
            y_filt_min = view_PC.points[1, :] > -5
            z_filt_max = view_PC.points[2, :] < 2
            z_filt_min = view_PC.points[2, :] > -1
            close = np.logical_and(x_filt_min, x_filt_max)
            close = np.logical_and(close, y_filt_min)
            close = np.logical_and(close, y_filt_max)
            close = np.logical_and(close, z_filt_min)
            close = np.logical_and(close, z_filt_max)
            view_PC = PointCloud(view_PC.points[:, close])

            # Plotting parameters
            params = {
                'legend.fontsize': 'x-large',
                'figure.figsize': (15, 5),
                'axes.labelsize': 'x-large',
                'axes.titlesize': 'x-large',
                'xtick.labelsize': 'x-large',
                'ytick.labelsize': 'x-large',
                'figure.max_open_warning': 1000
            }
            pylab.rcParams.update(params)

            # Create figure for TRACKING
            fig = plt.figure(figsize=(15, 10), facecolor="white")
            # Create axis in 3D
            # ax = fig.gca(projection='3d')
            ax = plt.axes(projection='3d')

            # Scatter plot the cropped point cloud
            ratio = 1
            ax.scatter(
                view_PC.points[0, ::ratio],
                view_PC.points[1, ::ratio],
                view_PC.points[2, ::ratio] / 2 - 1,
                s=3,
                c=view_PC.points[2, ::ratio])

            # point order to draw a full Box
            order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]

            # Plot GT box
            ax.plot(
                view_BB.Corners()[0, order],
                view_BB.Corners()[1, order],
                view_BB.Corners()[2, order] / 2 - 1,
                color="red",
                alpha=0.5,
                linewidth=5,
                linestyle=":")

            # center the box to plot the GT
            box.Translate(-this_BB.center)
            box.Rotate(this_BB.orientation.inverse)

            # Plot best results
            ax.plot(
                box.Corners()[0, order],
                box.Corners()[1, order],
                box.Corners()[2, order] / 2 - 1,
                color="blue",
                alpha=0.5,
                linewidth=2,
                linestyle=":")

            # point order to draw the visible part of the box
            # order = [3, 0, 1, 2, 3, 7, 4, 0, 4, 5, 1]
            order = [6, 7, 4, 5, 6, 2, 1, 5, 1, 0, 4]

            ax.plot(
                view_BB.Corners()[0, order],
                view_BB.Corners()[1, order],
                view_BB.Corners()[2, order] / 2 - 1,
                color="red",
                linewidth=5)

            ax.plot(
                box.Corners()[0, order],
                box.Corners()[1, order],
                box.Corners()[2, order] / 2 - 1,
                color="blue",
                linewidth=2)

            # Plot results for all cadidate as a surface
            surf = ax.plot_surface(
                X,
                Y,
                Z,
                cmap=cm.coolwarm,
                rstride=1,
                cstride=1,
                linewidth=0.3,
                edgecolors=[0, 0, 0, 0.8],
                antialiased=True,
                vmin=0,
                vmax=1)

            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)

            ax.xaxis.set_pane_color((0.9, 0.9, 0.9, 0.2))
            ax.yaxis.set_pane_color((0.9, 0.9, 0.9, 0.2))
            ax.zaxis.set_pane_color((0.9, 0.9, 0.9, 0.2))

            ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

            ax.set_xticks(x_space[::2])
            ax.set_yticks(y_space[::2])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.view_init(20, -140)

            plt.tight_layout()
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-1.5, 1)

            # Save figure as Tracking results
            os.makedirs(
                os.path.join(path_results, "Tracking"),
                exist_ok=True)
            plt.savefig(
                os.path.join(path_results, "Tracking", f"{i}_Ours.png"),
                format='png',
                dpi=100)