

from pyquaternion import Quaternion
import numpy as np
import os
from tqdm import tqdm
import torch
import imageio

from NN_ShapeCompletion import Model

import pandas as pd

import open3d as o3d
import copy

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pylab as pylab

class KITTITrackingSequence():

    def __init__(self, path, sequence_id):
        self.folder_path = path
        self.sequence_id = sequence_id

        self.calib_data = self.ReadCalibFile(os.path.join(self.folder_path, 'calib', "{:04d}.txt".format(sequence_id)))
        self.labels = self.ReadLabels(os.path.join(self.folder_path, 'label_02', "{:04d}.txt".format(sequence_id)))


    def ReadLiDARFrame(self, frame_idx, b_transform_to_cam_coord = True):
        velodyne_path = os.path.join(self.folder_path, 'velodyne', "{:04d}".format(self.sequence_id), f'{frame_idx:06}.bin')
        point_cloud = np.fromfile(velodyne_path, dtype=np.float32).reshape(-1, 4).T
        point_cloud = point_cloud[:3, :]

        if b_transform_to_cam_coord:
            transf_mat = np.vstack((self.calib_data["Tr_velo_cam"], np.array([0, 0, 0, 1])))
            point_cloud = transf_mat.dot(
                np.vstack((point_cloud, np.ones(point_cloud.shape[1]))))[:3, :]

        return point_cloud

    def ReadCalibFile(self, calib_filename):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(calib_filename, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data

    def ReadLabels(self, label_filename):
        labels = pd.read_csv(
            label_filename,
            sep=' ',
            names=[
                "frame", "track_id", "type", "truncated", "occluded",
                "alpha", "bbox_left", "bbox_top", "bbox_right",
                "bbox_bottom", "height", "width", "length", "x", "y", "z",
                "rotation_y"
            ])

        return labels



class Box:
    def __init__(self, center, size, orientation, label=np.nan):

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label

        self.rotation_matrix = self.orientation.rotation_matrix

        # self.corners = self.Corners()


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
        x_corners = l / 2 * np.array([1,  1,  1,  1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1,  1,  1, -1, -1,  1])
        z_corners = h / 2 * np.array([1,  1, -1, -1,  1,  1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners




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


class KITTITrackingDatasetV2():

    def __init__(self, path, sequence_ids, catogery_name = "Car"):
        self.sequences = {}
        for sequence_id in sequence_ids:
            self.sequences[f'{sequence_id}'] = KITTITrackingSequence(path, sequence_id)

        self.list_of_tracklets = []
        for sequence_id, sequence in self.sequences.items():
            labels = sequence.labels[sequence.labels["type"] == catogery_name]
            labels.insert(loc=0, column="scene", value=sequence_id)
            for track_id in labels.track_id.unique():
                df_tracklet = labels[labels["track_id"] == track_id]
                df_tracklet = df_tracklet.reset_index(drop=True)
                tracklet_anno = [anno for index, anno in df_tracklet.iterrows()]
                self.list_of_tracklets.append(tracklet_anno)

    def GetBBandPC(self, anno):
        sequence_id = anno["scene"]
        frame_id = anno["frame"]
        points = self.sequences[sequence_id].ReadLiDARFrame(frame_id)
        BB = self.Anno2Box(anno)
        return PointCloud(points), BB

    def Anno2Box(self, anno):
        center = [anno["x"], anno["y"] - anno["height"] / 2, anno["z"]]
        size = [anno["width"], anno["length"], anno["height"]]
        orientation = Quaternion(
            axis=[0, 1, 0], radians=anno["rotation_y"]) * Quaternion(
            axis=[1, 0, 0], radians=np.pi / 2)
        BB = Box(center, size, orientation)
        return BB

    def __getitem__(self, index):
        list_of_anno = self.list_of_tracklets[index]
        PCs = []
        BBs = []
        for anno in list_of_anno:
            this_PC, this_BB = self.GetBBandPC(anno)
            PCs.append(this_PC)
            BBs.append(this_BB)
        return PCs, BBs, list_of_anno


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


def VisualizePCsandBBs(PCs, BBs):
    vis2 = o3d.visualization.VisualizerWithKeyCallback()
    vis2.create_window()
    vis2.get_render_option().point_size = 2  # set points size

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis2.add_geometry(mesh_frame)

    global g_idx
    g_idx = 0

    global b_first_run
    b_first_run = True

    pcd = o3d.geometry.PointCloud()

    def key_next_frame_callback(vis2):
        global g_idx
        global b_first_run

        if g_idx >= len(PCs):
            return True

        if not b_first_run:
            previous_view = vis2.get_view_control().convert_to_pinhole_camera_parameters()

        points = PCs[g_idx].points
        pcd.points = o3d.utility.Vector3dVector(points.T)

        if 0:
            if g_idx == 0:
                vis2.add_geometry(pcd)
            else:
                vis2.update_geometry(pcd)
        else:
            vis2.clear_geometries()
            if b_first_run:
                vis2.add_geometry(pcd)
            else:
                vis2.add_geometry(pcd, False)

            BB = BBs[g_idx]
            o3d_box = CreateO3dboxFromBoxCorners(BB.Corners().T)

            if b_first_run:
                vis2.add_geometry(o3d_box)
            else:
                vis2.add_geometry(o3d_box, False)

        vis2.poll_events()
        vis2.update_renderer()

        if not b_first_run:
            current_view = vis2.get_view_control()
            current_view.convert_from_pinhole_camera_parameters(previous_view, True)

        if b_first_run:
            b_first_run = False

        g_idx = g_idx + 1
        return True

    vis2.register_key_callback(ord('E'), key_next_frame_callback)

    key_next_frame_callback(vis2)  # draw first frame
    vis2.run()



def VisualizePCandMultipleBBs(PC, BBs):
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window()
    vis2.get_render_option().point_size = 2  # set points size

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis2.add_geometry(mesh_frame)

    pcd = o3d.geometry.PointCloud()
    points = PC.points
    pcd.points = o3d.utility.Vector3dVector(points.T)
    vis2.add_geometry(pcd)

    for BB in BBs:
        o3d_box = CreateO3dboxFromBoxCorners(BB.Corners().T)
        vis2.add_geometry(o3d_box)

    vis2.run()


def GenerateBoxesFromSearchSpace(box, search_space=[[0, 0, 0]]):
    # Geenrate more candidate boxes based on prior and search space
    # Input : Prior position, search space and search size
    # Output : List of boxes

    candidate_boxes = []

    for offset in search_space:

        # rot_quat = Quaternion(matrix=box.rotation_matrix)
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

    rot_mat = np.transpose(new_box.rotation_matrix)
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


def GetModel(PCs, boxes, offset=0, scale=1.0, normalize=False):

    if len(PCs) == 0:
        return PointCloud(np.ones((3, 0)))
    points = np.ones((PCs[0].points.shape[0], 0))
    # print(points)

    for PC, box in zip(PCs, boxes):
        cropped_PC = CropAndCenterPC(
            PC, box, offset=offset, scale=scale, normalize=normalize)
        # try:
        if cropped_PC.points.shape[1] > 0:
            points = np.concatenate([points, cropped_PC.points], axis=1)

    PC = PointCloud(points)

    return PC

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

if __name__ == '__main__':

    sequence_id = [19, 20] # 21
    track = 29

    path_KITTI = '/home/hao/dataset/KITTI_tracking/training'
    path_results = './results2/'

    scale_BB = 1.25

    model_name = "Ours"

    # Enforce reproducibility in torch
    torch.manual_seed(1155)

    np.random.seed(0)



    AE_chkpt_file = ""
    chkpt_file = "./models/ShapeCompletion/model.pth.tar"
    model = Model(128, chkpt_file=chkpt_file, AE_chkpt_file=AE_chkpt_file).cuda()
    model.eval()

    # Create Dataset
    # dataset = SiameseTest(
    #     model=model, path=path_KITTI, split="Test", scale_BB=1.25)

    dataset = KITTITrackingDatasetV2(path_KITTI, sequence_id)

    # Create search space
    # x_space = np.linspace(-4, 4, 9)  # min, max, number of sample
    # y_space = np.linspace(-4, 4, 9)  # min, max, number of sample
    x_space = np.linspace(-4, 4, 7)  # min, max, number of sample
    y_space = np.linspace(-4, 4, 7)  # min, max, number of sample
    a_space = np.linspace(-10, 10, 3)  # min, max, number of sample
    X, Y, A = np.meshgrid(x_space, y_space, a_space)  # create mesh grid
    search_space = np.array([X.flatten(), Y.flatten(), A.flatten()]).T



    # Get List of PCs and BBs from dataset
    PCs, BBs, list_of_anno = dataset[track]

    if 0:
        VisualizePCsandBBs(PCs, BBs)


    # Loop over Point Cloud and Bounding Boxes
    for i in tqdm(range(15, len(PCs))):

        this_PC = PCs[i]
        this_BB = BBs[i]
        ref_BB = BBs[i]

        # create candidate PCs and BBs
        candidate_BBs = GenerateBoxesFromSearchSpace(ref_BB, search_space=search_space)

        if 0:
            VisualizePCandMultipleBBs(this_PC, candidate_BBs)

        candidate_PCs = [
            CropAndCenterPC(
                this_PC,
                box,
                offset=0,
                scale=scale_BB,
                normalize="PreTrained" in model_name)
            for box in candidate_BBs
        ]
        candidate_PCs_reg = [RegularizePC(PC, model) for PC in candidate_PCs]   # duplicate or remove points to fit the input of the network
        candidate_PCs_torch = torch.cat(candidate_PCs_reg, dim=0).cuda()

        # create model PC
        model_PC = GetModel(
            PCs[:i],
            BBs[:i],
            offset=0,
            scale=scale_BB,
            normalize="PreTrained" in model_name)


        if 0:
            repeat_shape = np.ones(len(candidate_PCs_torch.shape), dtype=np.int32)
            repeat_shape[0] = len(candidate_PCs)
            model_PC_encoded = RegularizePC(model_PC, model).repeat(tuple(repeat_shape)).cuda()

            # infer model
            output, model_PC_decoded = model(candidate_PCs_torch, model_PC_encoded)

            if 0:
                FH_Subplots3([candidate_PCs_torch.detach().cpu().numpy()[0],
                              model_PC_encoded.detach().cpu().numpy()[0],
                              model_PC_decoded.detach().cpu().numpy()[0]])

            model_PC_decoded = PointCloud(
                model_PC_decoded.detach().cpu().numpy()[0])
            if "PreTrained" in model_name:
                normalizer = [ref_BB.wlh[1], ref_BB.wlh[0], ref_BB.wlh[2]]
                model_PC_decoded.points = model_PC_decoded.points * np.atleast_2d(
                    normalizer).T

        else:
            model_PC_encoded2 = RegularizePC(model_PC, model).cuda()
            model_code = model.AE.encode(model_PC_encoded2)
            output = model.FastForward(candidate_PCs_torch, model_code)


        # store scores for all candidates
        scores = output.detach().cpu().numpy()
        idx = np.argmax(scores)  # select index of higest score
        box = candidate_BBs[idx]  # select box with highest score




############################################ the following codes are all for visualization ####################################

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
            os.path.join(path_results, f"{track:04.0f}", "Tracking"),
            exist_ok=True)
        plt.savefig(
            os.path.join(path_results, f"{track:04.0f}",
                         "Tracking", f"{i}_{model_name}.png"),
            format='png',
            dpi=100)




        if 0:
            # Create figure for RECONSTRUCTION
            fig = plt.figure(figsize=(9, 6), facecolor="white")
            # Create axis in 3D
            ax = plt.axes(projection='3d')

            # select 2K point to visualize
            sample = np.random.randint(
                low=0,
                high=model_PC_decoded.points.shape[1],
                size=2048,
                dtype=np.int64)
            # Scatter plot the point cloud
            ax.scatter(
                model_PC_decoded.points[0, sample],
                model_PC_decoded.points[1, sample],
                model_PC_decoded.points[2, sample],
                s=3,
                c=model_PC_decoded.points[2, sample])

            # Plot the car BB
            order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]
            ax.plot(
                view_BB.Corners()[0, order],
                view_BB.Corners()[1, order],
                view_BB.Corners()[2, order],
                color="red",
                alpha=0.5,
                linewidth=3,
                linestyle=":")
            order = [6, 7, 4, 5, 6, 2, 1, 5, 1, 0, 4]
            ax.plot(
                view_BB.Corners()[0, order],
                view_BB.Corners()[1, order],
                view_BB.Corners()[2, order],
                color="red",
                linewidth=3)

            # setup axis
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_xticklabels([], fontsize=10)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels([], fontsize=10)
            ax.set_zticks([-1, 0, 1])
            ax.set_zticklabels([], fontsize=10)
            ax.view_init(20, -140)
            plt.tight_layout()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1.5, 1.5)

            # Save figure as Decoded Model results
            os.makedirs(
                os.path.join(path_results, f"{track:04.0f}",
                             "Reconstruction"),
                exist_ok=True)
            plt.savefig(
                os.path.join(path_results, f"{track:04.0f}",
                             "Reconstruction", f"{i}_{model_name}.png"),
                format='png',
                dpi=100)

        if 0:
            # Create figure for MODEL PC
            fig = plt.figure(figsize=(9, 6), facecolor="white")
            ax = plt.axes(projection='3d')

            if "PreTrained" in model_name:
                model_PC = GetModel(
                    PCs[:i],
                    BBs[:i],
                    offset=dataset.offset_BB,
                    scale=dataset.scale_BB)

            # sample 2K Points
            sample = np.random.randint(
                low=0, high=model_PC.points.shape[1], size=2048, dtype=np.int64)
            # Scatter plot the Point cloud
            ax.scatter(
                model_PC.points[0, sample],
                model_PC.points[1, sample],
                model_PC.points[2, sample],
                s=3,
                c=model_PC.points[2, sample])

            # Plot the Bounding Box
            order = [0, 4, 0, 1, 5, 1, 2, 6, 2, 3, 7, 3, 0, 4, 5, 6, 7, 4]
            ax.plot(
                view_BB.Corners()[0, order],
                view_BB.Corners()[1, order],
                view_BB.Corners()[2, order],
                color="red",
                alpha=0.5,
                linewidth=3,
                linestyle=":")
            order = [6, 7, 4, 5, 6, 2, 1, 5, 1, 0, 4]
            ax.plot(
                view_BB.Corners()[0, order],
                view_BB.Corners()[1, order],
                view_BB.Corners()[2, order],
                color="red",
                linewidth=3)

            # setup axis
            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_xticklabels([], fontsize=10)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels([], fontsize=10)
            ax.set_zticks([-1, 0, 1])
            ax.set_zticklabels([], fontsize=10)
            ax.view_init(20, -140)
            plt.tight_layout()
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_zlim(-1.5, 1.5)

            # Save figure as Model results
            os.makedirs(
                os.path.join(path_results, f"{track:04.0f}", "Model"),
                exist_ok=True)
            plt.savefig(
                os.path.join(path_results, f"{track:04.0f}", "Model",
                             f"{i}_{model_name}.png"),
                format='png',
                dpi=100)

    # create GIF Tracking
    images = []
    for i in tqdm(range(1, len(PCs))):
        image_path = os.path.join(path_results, f'{track:04.0f}',
                                  "Tracking", f'{i}_{model_name}.png')
        images.append(imageio.imread(image_path))

    image_path = os.path.join(path_results,
                              f'{track:04.0f}_{model_name}_Tracking.gif')
    imageio.mimsave(image_path, images)

    # create GIF GT Car
    images = []
    for i in tqdm(range(1, len(PCs))):
        image_path = os.path.join(path_results, f'{track:04.0f}',
                                  "Model", f'{i}_{model_name}.png')
        images.append(imageio.imread(image_path))

    image_path = os.path.join(
        path_results, f'{track:04.0f}_{model_name}_Model.gif')
    imageio.mimsave(image_path, images)

    # create GIF Reconstructed Car
    images = []
    for i in tqdm(range(1, len(PCs))):
        image_path = os.path.join(path_results, f'{track:04.0f}',
                                  "Reconstruction",
                                  f'{i}_{model_name}.png')
        images.append(imageio.imread(image_path))

    image_path = os.path.join(
        path_results,
        f'{track:04.0f}_{model_name}_Reconstruction.gif')
    imageio.mimsave(image_path, images)

