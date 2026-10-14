import os
import numpy as np
import open3d as o3d
import yaml
from CTools.CDataLoaderKITTI import CDataLoaderKITTI
from CTools.CDataLoaderAX7 import CDataLoaderAX7
loader = CDataLoaderKITTI()

class CVisualizer(CDataLoaderAX7):

    def vis_cloud(self,pc_xyz,point_size=1,color_list = None,background_color = [0,0,0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        if color_list is not None:
            pcd.paint_uniform_color(color_list)

        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()
        vis_.add_geometry(pcd)
        render_options = vis_.get_render_option()
        render_options.point_size = point_size
        render_options.background_color = np.array(background_color)
        vis_.run()

    def vis_cloud_with_label(self,pc_xyz,colors,voxDownsize=None,bg_color = [0,0,0]):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        print("before down sampled:", np.asarray(pcd.points).shape)
        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()

        if voxDownsize is None:
            pass
            print("after down sampled:", np.asarray(pcd.points).shape)
            vis_.add_geometry(pcd)
        else:
            downpcd = pcd.voxel_down_sample(voxel_size=voxDownsize)
            print("after down sampled:",np.asarray(downpcd.points).shape)
            vis_.add_geometry(downpcd)


        render_options = vis_.get_render_option()
        render_options.point_size = 2
        render_options.background_color = np.array(bg_color)
        vis_.run()

    def vis_cloud_with_label_white_background(self,pc_xyz,colors,voxDownsize=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        print("before down sampled:", np.asarray(pcd.points).shape)
        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()

        if voxDownsize is None:
            pass
            print("after down sampled:", np.asarray(pcd.points).shape)
            vis_.add_geometry(pcd)
        else:
            downpcd = pcd.voxel_down_sample(voxel_size=voxDownsize)
            print("after down sampled:",np.asarray(downpcd.points).shape)
            vis_.add_geometry(downpcd)


        render_options = vis_.get_render_option()
        render_options.point_size = 2
        render_options.background_color = np.array([1, 1, 1])
        vis_.run()


    def vis_two_cloud(self,pc1,pc2,color_pc1 = [1, 1, 1],color_pc2 = [1, 0, 0]):

        cloud1 = o3d.geometry.PointCloud()
        cloud1.points = o3d.utility.Vector3dVector(pc1)
        cloud1.paint_uniform_color(color_pc1)

        cloud2 = o3d.geometry.PointCloud()
        cloud2.points = o3d.utility.Vector3dVector(pc2)
        cloud2.paint_uniform_color(color_pc2)

        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()
        vis_.add_geometry(cloud1)
        vis_.add_geometry(cloud2)
        render_options = vis_.get_render_option()
        render_options.point_size = 1
        render_options.background_color = np.array([0, 0, 0])
        vis_.run()

    def vis_three_cloud(self,pc1,pc2,pc3,color_pc1,color_pc2,color_pc3,bg_color):


        cloud1 = o3d.geometry.PointCloud()
        cloud1.points = o3d.utility.Vector3dVector(pc1)
        cloud1.paint_uniform_color(color_pc1)

        cloud2 = o3d.geometry.PointCloud()
        cloud2.points = o3d.utility.Vector3dVector(pc2)
        cloud2.paint_uniform_color(color_pc2)

        cloud3 = o3d.geometry.PointCloud()
        cloud3.points = o3d.utility.Vector3dVector(pc3)
        cloud3.paint_uniform_color(color_pc3)

        vis_ = o3d.visualization.Visualizer()
        vis_.create_window()
        vis_.add_geometry(cloud1)
        vis_.add_geometry(cloud2)
        vis_.add_geometry(cloud3)
        render_options = vis_.get_render_option()
        render_options.point_size = 1
        render_options.background_color = np.array(bg_color)
        vis_.run()


    def vis_cloud_with_normal(self,pc_xyz,pt_color,search_radius,search_max_nn,voxDownsize=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        print("before down sampled:", np.asarray(pcd.points).shape)
        pcd.paint_uniform_color(pt_color)
        if voxDownsize is None:
            pass
            print("after down sampled:", np.asarray(pcd.points).shape)
        else:
            pcd = pcd.voxel_down_sample(voxel_size=voxDownsize)
            print("after down sampled:",np.asarray(pcd.points).shape)

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius,max_nn=search_max_nn))
        normals = np.array(pcd.normals,dtype=np.float32)
        o3d.visualization.draw_geometries([pcd],window_name="point cloud with normal",
                                          point_show_normal=True)
        return normals

    #--------------------------------------------------------------------------------

    def vis_multicloud_bin_wo_label(self,scan_names,down_size):
        g_idx = 0
        vis = o3d.visualization.VisualizerWithKeyCallback()

        def show_cloud(vis):
            nonlocal g_idx
            print(g_idx, '/', len(scan_names))
            points = self.load_bin_xyzi(scan_names[g_idx])[:,0:3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            downpcd = pcd.voxel_down_sample(voxel_size=down_size)
            vis.clear_geometries()
            vis.add_geometry(downpcd)
            vis.update_renderer()
            vis.poll_events()

        def key_forward_callback(vis):
            nonlocal g_idx
            g_idx += 1
            if g_idx >= len(scan_names):
                g_idx = len(scan_names) - 1
            show_cloud(vis)
            return True

        def key_backward_callback(vis):
            nonlocal g_idx
            g_idx -= 1
            if g_idx < 0:
                g_idx = 0
            show_cloud(vis)
            return True

        vis.create_window()
        render_options = vis.get_render_option()
        render_options.point_size = 1
        render_options.background_color = np.array([0, 0, 0])
        vis.register_key_callback(ord('N'), key_forward_callback)
        vis.register_key_callback(ord('B'), key_backward_callback)
        vis.run()

    def vis_multicloud_with_label(self,scan_names, label_names, sem_color_lut):
        g_idx = 0
        vis = o3d.visualization.VisualizerWithKeyCallback()

        def show_cloud(vis):
            nonlocal g_idx
            print(g_idx, '/', len(scan_names))
            points = self.load_bin_xyzi(scan_names[g_idx])[:,0:3]
            # points = open_scan_3(scan_names[g_idx])
            sem_label = self.load_bin_semlabel(label_names[g_idx])
            sem_label_color = loader.calc_sem_label_color(sem_color_lut, sem_label)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(sem_label_color)
            vis.clear_geometries()
            vis.add_geometry(pcd)

            vis.update_renderer()
            vis.poll_events()

        def key_forward_callback(vis):
            nonlocal g_idx
            g_idx += 1
            if g_idx >= len(scan_names):
                g_idx = len(scan_names) - 1
            show_cloud(vis)
            return True

        def key_backward_callback(vis):
            nonlocal g_idx
            g_idx -= 1
            if g_idx < 0:
                g_idx = 0
            show_cloud(vis)
            return True

        vis.create_window()
        render_options = vis.get_render_option()
        render_options.point_size = 1
        render_options.background_color = np.array([0, 0, 0])
        vis.register_key_callback(ord('N'), key_forward_callback)
        vis.register_key_callback(ord('B'), key_backward_callback)
        vis.run()

    def visMultiCloud(self,scan_names, down_size):
        g_idx = 0
        vis = o3d.visualization.VisualizerWithKeyCallback()

        def show_cloud(vis):
            nonlocal g_idx
            print(g_idx, '/', len(scan_names))
            points, remissions = self.open_scan(scan_names[g_idx])

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            downpcd = pcd.voxel_down_sample(voxel_size=down_size)
            # pcd.voxel_down_sample(voxel_size=0.8)
            vis.clear_geometries()
            vis.add_geometry(downpcd)
            vis.update_renderer()
            vis.poll_events()

        def key_forward_callback(vis):
            nonlocal g_idx
            g_idx += 1
            if g_idx >= len(scan_names):
                g_idx = len(scan_names) - 1
            show_cloud(vis)
            return True

        def key_backward_callback(vis):
            nonlocal g_idx
            g_idx -= 1
            if g_idx < 0:
                g_idx = 0
            show_cloud(vis)
            return True

        vis.create_window()
        render_options = vis.get_render_option()
        render_options.point_size = 1
        render_options.background_color = np.array([0, 0, 0])
        vis.register_key_callback(ord('N'), key_forward_callback)
        vis.register_key_callback(ord('B'), key_backward_callback)
        vis.run()

    def visMultiCloud_with_view(self,scan_names, label_names, sem_color_lut, label_dict, down_size, viewpointjson):
        g_idx = 0
        vis = o3d.visualization.VisualizerWithKeyCallback()

        def show_cloud(vis):
            nonlocal g_idx
            print(g_idx, '/', len(scan_names))
            points, remissions = self.open_scan(scan_names[g_idx])
            # points = open_scan_AX7(scan_names[g_idx])
            label = self.open_label(label_names[g_idx])

            label_unique = np.unique(label)
            label_value = []
            for l in label_unique:
                value = label_dict[l]
                label_value.append(value)
            print("label contains:", label_unique, '\n', label_value)

            sem_label = label & 0xFFFF
            inst_label = label >> 16
            sem_label_color = self.calc_sem_label_color(sem_color_lut, sem_label)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(sem_label_color)
            downpcd = pcd.voxel_down_sample(voxel_size=down_size)
            # pcd.voxel_down_sample(voxel_size=0.8)

            vis.clear_geometries()
            vis.add_geometry(downpcd)

            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters(viewpointjson)
            ctr.convert_from_pinhole_camera_parameters(param)

            vis.update_renderer()
            vis.poll_events()

        def key_forward_callback(vis):
            nonlocal g_idx
            g_idx += 1
            if g_idx >= len(scan_names):
                g_idx = len(scan_names) - 1
            show_cloud(vis)
            return True

        def key_backward_callback(vis):
            nonlocal g_idx
            g_idx -= 1
            if g_idx < 0:
                g_idx = 0
            show_cloud(vis)
            return True

        vis.create_window(width=3840, height=2160)
        render_options = vis.get_render_option()
        render_options.point_size = 1.5
        render_options.background_color = np.array([0, 0, 0])

        vis.register_key_callback(ord('N'), key_forward_callback)
        vis.register_key_callback(ord('B'), key_backward_callback)
        vis.run()

    def compare_pcd_and_bin(self,pcd_file,bin_file):
        pcd = o3d.io.read_point_cloud(pcd_file)
        pcd_points = np.asarray(pcd.points)
        pcd_colors = np.asarray(pcd.colors)
        pcd_labels = pcd_colors * 255
        sem_color_lut = self.load_yaml()


        bin_points,bin_remissions = self.open_scan(bin_file)

        self.vis_two_Cloud(bin_points,)

if __name__ == '__main__':
    bin_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/00/velodyne/000000.bin'
    label_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/00/labels/000000.label'
    labels_dict, color_dict, color_dict_inv, contents_dict, learning_map, learning_map_inv, sem_color_lut = loader.load_kitti_yaml_original()
    points, remissions = loader.load_bin_xyzi(bin_file)
    semlabel = loader.load_bin_semlabel(label_file)
    semlabel_color = loader.calc_sem_label_color(sem_color_lut, semlabel)

    bin_file2 = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/00/velodyne/000001.bin'
    label_file2 = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/00/labels/000001.label'
    points2, remissions2 = loader.load_bin_xyzi(bin_file2)

    vis = CVisualizer()
    vis.vis_cloud(points,point_size=1,background_color=[1,1,1])
    vis.vis_cloud_with_label(points,semlabel_color)
    vis.vis_two_cloud(points,points2,color_pc1=[1,0,1],color_pc2=[0,0.5,0.5])
    # vis.vis_cloud(points,point_size=2)
    # vis.vis_cloud(points, point_size=3)
    # vis.vis_cloud(points, point_size=3,color_list=[1,0,0])
