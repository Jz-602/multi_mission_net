import os
import numpy as np
import open3d as o3d
import yaml

import pandas as pd
from pyquaternion import Quaternion
import vis



class Box:
    def __init__(self, center, size, orientation, label=np.nan):

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label

        self.corners = self.Corners()
        self.bottom_corners = self.BottomCorners()

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

    def BottomCorners(self):
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners[:, [2, 3, 7, 6]]



class KITTITrackingDataset():

    def __init__(self, path, sequence_id):
        self.folder_path = path
        self.sequence_id = sequence_id
        self.num_of_frames = len(os.listdir(os.path.join(self.folder_path, 'velodyne', "{:04d}".format(self.sequence_id))))

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



def CreateO3dboxFromBoxCorners(corners, color=None):
	'''
	corners: 8 corners(x, y, z)
	corners: array = 8*3
	#         7 -------- 6
	#        /|         /|
	#       4 -------- 5 .
	#       | |        | |
	#       . 3 -------- 2
	#       |/         |/
	#       0 -------- 1

	#         0 -------- 1
	#        /|         /|
	#       4 -------- 5 .
	#       | |        | |
	#       . 3 -------- 2
	#       |/         |/
	#       7 -------- 6
	'''
	# 12 lines in a box
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



def GetO3dBoxes(framelabels):
    o3d_boxes = []
    for index, anno in framelabels.iterrows():
        if anno["type"] in ["Car"]:
            center = [anno["x"], anno["y"] - anno["height"] / 2, anno["z"]]
            size = [anno["width"], anno["length"], anno["height"]]
            orientation = Quaternion(
                axis=[0, 1, 0], radians=anno["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
            BB = Box(center, size, orientation)


            o3d_box = CreateO3dboxFromBoxCorners(BB.corners.T)

            o3d_boxes.append(o3d_box)

    return o3d_boxes


if __name__ == '__main__':
    path = '/home/hao/dataset/KITTI_tracking/training'
    sequence_id = 0
    dataset = KITTITrackingDataset(path, sequence_id)

    pcd = o3d.geometry.PointCloud()

    if 0:
        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()
        # vis_.add_geometry(point_cloud)



        for i in range(dataset.num_of_frames):
            point_cloud = dataset.ReadLiDARFrame(i)

            pcd.points = o3d.utility.Vector3dVector(point_cloud.T)

            if 0:       # the following two approaches are both fine
                if i==0:
                    vis_.add_geometry(pcd)
                else:
                    vis_.update_geometry(pcd)
            else:
                vis_.clear_geometries()
                vis_.add_geometry(pcd)

            vis_.poll_events()
            vis_.update_renderer()

            print(i)

        vis_.destroy_window()




    if 1:
        vis2 = o3d.visualization.VisualizerWithKeyCallback()
        vis2.create_window()
        vis2.get_render_option().point_size = 2  # set points size


        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        vis2.add_geometry(mesh_frame)

        global g_idx
        g_idx = 109

        global b_first_run
        b_first_run = True
        def key_next_frame_callback(vis2):
            global g_idx
            global b_first_run

            if not b_first_run:
                previous_view = vis2.get_view_control().convert_to_pinhole_camera_parameters()

            point_cloud = dataset.ReadLiDARFrame(g_idx)
            pcd.points = o3d.utility.Vector3dVector(point_cloud.T)

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

                frame_labels = dataset.labels[dataset.labels["frame"] == g_idx]

                o3d_boxes = GetO3dBoxes(frame_labels)
                if b_first_run:
                    for o3d_box in o3d_boxes:
                        vis2.add_geometry(o3d_box)
                else:
                    for o3d_box in o3d_boxes:
                        vis2.add_geometry(o3d_box, False)






            vis2.poll_events()
            vis2.update_renderer()

            if not b_first_run:
                current_view = vis2.get_view_control()
                current_view.convert_from_pinhole_camera_parameters(previous_view, True)

            if b_first_run:
                b_first_run = False

            g_idx = g_idx+1
            return True

        vis2.register_key_callback(ord('E'), key_next_frame_callback)

        key_next_frame_callback(vis2)   # draw first frame
        vis2.run()



    if 0:
        for i in range(dataset.num_of_frames):
            point_cloud = dataset.ReadLiDARFrame(i)


            frame_labels = dataset.labels[dataset.labels["frame"] == i]
            vis.run(point_cloud)

