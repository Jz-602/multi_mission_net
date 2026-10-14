import os
import numpy as np
import open3d as o3d
import yaml

from pyquaternion import Quaternion

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###########################################################################################
################################# matlab plot function ####################################
###########################################################################################


def plot(*args):
    if len(args) == 1:
        plt.plot(range(len(args[0])), args[0])  # 等效于 MATLAB 的 plot(x)
        plt.show(block='True')
    elif len(args) == 2:
        plt.plot(args[0], args[1], 'r.')  # 与 MATLAB 的 plot(x, y) 一致
        plt.xlabel('x');
        plt.ylabel('y')
        plt.show(block='True')
    elif len(args) == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(args[0], args[1], args[2], 'r.')  # 等效于 MATLAB 的 plot3(x, y, z)
        ax.set_xlabel('X');
        ax.set_ylabel('Y');
        ax.set_zlabel('Z')
        plt.show(block='True')

###########################################################################################
################################# visualize point cloud ###################################
###########################################################################################



def run(pc_xyz,point_size=3,color_list = None,background_color = [0,0,0]):
    if pc_xyz.shape[0]==3 and pc_xyz.shape[1]!=3:
        pc_xyz = pc_xyz.T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    if color_list is not None:
        pcd.paint_uniform_color(color_list)

    vis_ = o3d.visualization.Visualizer()
    vis_.create_window()
    vis_.add_geometry(pcd)
    render_options = vis_.get_render_option()
    render_options.show_coordinate_frame = True
    render_options.point_size = point_size
    render_options.background_color = np.array(background_color)
    vis_.run()


def VisualizePointCloud(pc_xyz,point_size=3,color_list = None,background_color = [0,0,0]):
    if pc_xyz.shape[0]==3 and pc_xyz.shape[1]!=3:
        pc_xyz = pc_xyz.T

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    if color_list is not None:
        pcd.paint_uniform_color(color_list)

    vis_ = o3d.visualization.Visualizer()
    vis_.create_window()
    vis_.add_geometry(pcd)
    render_options = vis_.get_render_option()
    render_options.show_coordinate_frame = True
    render_options.point_size = point_size
    render_options.background_color = np.array(background_color)
    vis_.run()

###########################################################################################
############################# visualize semantic labels ###################################
###########################################################################################

def show_semantic_point_cloud(pc_xyz, semantic_label, color_dict):
    # pc_xyz: N*3;
    # semantic_label: N*1
    if pc_xyz.shape[0] != len(semantic_label) or pc_xyz.shape[1] != 3:
       raise ValueError("The shape of pc_xyz should be N*3，semantic_label should be N*1")

    unique_labels = np.unique(semantic_label)
    point_colors = np.ones((len(semantic_label), 3))


    for i, label in enumerate(unique_labels):
        mask = (semantic_label == label)
        color = color_dict[label]
        point_colors[mask] = [color[2]/255.0, color[1]/255.0, color[0]/255.0]  # bgr2rgb

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    opt = vis.get_render_option()
    opt.point_size = 3  # 设置点的大小
    # opt.show_coordinate_frame = True
    opt.background_color = np.array([0,0,0])
    vis.run()
    vis.destroy_window()
    return


def show_instance_point_cloud(pc_xyz, instance_label):
    # pc_xyz: N*3;
    # instance_label: N*1
    if pc_xyz.shape[0] != len(instance_label) or pc_xyz.shape[1] != 3:
       raise ValueError("The shape of pc_xyz should be N*3，instance_label should be N*1")

    unique_labels = np.unique(instance_label)
    point_colors = np.ones((len(instance_label), 3))

    for i, label in enumerate(unique_labels):
        if label != -1:
            mask = (instance_label == label)
            point_colors[mask] = np.random.rand(3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    opt = vis.get_render_option()
    opt.point_size = 3  # 设置点的大小
    # opt.show_coordinate_frame = True

    opt.background_color = np.array([0,0,0])
    vis.run()
    vis.destroy_window()
    return
    
    
    
###########################################################################################
############################# visualize 3D bounding boxes #################################
###########################################################################################
class Box:
    def __init__(self, center, size, orientation):
        # center stores t_L_V, orientation stores R_L_V     (the transformation from global LiDAR coord to local Vehicle coord)
        # both LiDAR coord and Vehicle coord follows: x->forward, y->left, z->up
        # for size, it actually stores dx, dy and dz, correspongding to vehicle length, vehicle width and vehicle height!
        self.center = np.array(center)
        self.wlh = np.array(size)       # using the name of wlh is a bit misleading!
        if type(orientation) == Quaternion:
            self.orientation = orientation
        else:
            self.orientation = Quaternion(axis=[0, 0, 1], angle=orientation)

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

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = w / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])   # using the name of w, l and h is misleading. w is actually dx (vehicle length), h is actually dy (vehicle width)
        y_corners = l / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])

        # # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        # x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        # y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        # z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])

        # # 3D bounding box corners. (Ours: x points right, y forward, z up.)
        # x_corners = w / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        # y_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        # z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])

        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

def CreateO3dboxFromBoxCorners(corners, label=0):
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

    if label == 0:
        color = [1, 0, 1]
    elif label == 1:
        color = [1, 0, 0]
    elif label == 2:
        color = [0, 1, 0]
    elif label == 3:
        color = [0, 1, 1]
    elif label == 4:
        color = [1, 1, 0]
    else :
        color = [0.5, 0.5, 0.5]

    colors = [color for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def VisualizePCandBB(pc_xyz, bbs, labels):
    # bbs: N * 7
    # labels: N * 1
    if pc_xyz.shape[0]==3 and pc_xyz.shape[1]!=3:
        pc_xyz = pc_xyz.T

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3  # set points size

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    vis.add_geometry(pcd)

    for i in range(bbs.shape[0]):
        BB = Box(bbs[i,:3], bbs[i,3:6], bbs[i,6])
        o3d_box = CreateO3dboxFromBoxCorners(BB.Corners().T, labels[i])
        vis.add_geometry(o3d_box)

    vis.run()

###########################################################################################
###########################################################################################
###########################################################################################