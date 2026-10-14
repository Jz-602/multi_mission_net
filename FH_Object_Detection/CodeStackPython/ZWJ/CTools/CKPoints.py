import numpy as np
import open3d as o3d
from CTools.CVisualizer import CVisualizer
from CTools.CDataLoaderKITTI import CDataLoaderKITTI

vis = CVisualizer()
loader = CDataLoaderKITTI()
labels_dict,color_dict,color_dict_inv,contents_dict,learning_map,learning_map_inv,sem_color_lut,learning_map_8,remap_lut_8 = loader.load_kitti_yaml_randla()

bool_vis = True
num_classes = 3


class SemanticKITTIKPoints():
    def __init__(self,map_path,label_path,downsize,nearest_k,target_samples,train_class = None,bool_vis = False):
        self.name = "SemanticKITTIKPoints"
        self.map_path = map_path
        self.label_path = label_path
        self.downsize = downsize
        self.nearest_k = nearest_k

        self.ignored_labels = np.sort([0])
        self.num_classes = num_classes
        self.target_samples = target_samples

        # load map
        print("loading map & downsizing...",self.map_path)
        map_o3d = o3d.t.io.read_point_cloud(map_path)
        print("before downsize:",np.asarray(map_o3d.point["positions"].numpy().shape))
        ds_map = map_o3d.voxel_down_sample(voxel_size=downsize)
        self.ds_map_points = ds_map.point["positions"].numpy()
        print("after downsize:", self.ds_map_points.shape)
        self.ds_map_labels = ds_map.point["intensity"].numpy().astype(np.uint32).reshape(-1)
        self.ds_map_sem_labels = self.ds_map_labels & 0xFFFF
        self.ds_map_inst_labels = self.ds_map_labels >> 16
        if bool_vis:
            vis.vis_cloud(self.ds_map_points)

        # removing moving vehicle
        print("removing dynamics...")
        moving_ind = np.argwhere(self.ds_map_sem_labels > 250).squeeze(1)
        self.ds_map_points,self.ds_map_labels,self.ds_map_sem_labels,self.ds_map_inst_labels = self.removing_target_label(moving_ind)
        print("after removing dynamics:",self.ds_map_points.shape)
        if bool_vis:
            vis.vis_cloud(self.ds_map_points)


        #remap to 8 classes
        print("remapping to 8 classes...")
        # sem_label_ori = copy.deepcopy(self.ds_map_sem_labels)
        self.ds_map_sem_labels = remap_lut_8[self.ds_map_sem_labels]

        # removing road
        print("removing road...")
        road_ind = np.argwhere(self.ds_map_sem_labels == 6).squeeze(1)
        self.ds_map_points,self.ds_map_labels,self.ds_map_sem_labels,self.ds_map_inst_labels = self.removing_target_label(road_ind)
        print("after removing road:", self.ds_map_points.shape)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.ds_map_points)
        print("output...")
        o3d.io.write_point_cloud("/media/alv-2/Rambo/Data/KITTI-Map/map_wo_dynamic&road_xyz/00.pcd",pcd)

        if bool_vis:
            vis.vis_cloud(self.ds_map_points)

        # car-noncar
        print("changing labels: car-noncar")
        car_idx = np.argwhere(self.ds_map_sem_labels == 1).squeeze(1)
        unlabeled_idx = np.argwhere(self.ds_map_sem_labels == 0).squeeze(1)
        others_idx = np.argwhere((self.ds_map_sem_labels != 0) & (self.ds_map_sem_labels != 1)).squeeze(1)
        bool_sum = (car_idx.shape[0] + unlabeled_idx.shape[0] + others_idx.shape[0] == self.ds_map_sem_labels.shape[0])

        self.ds_map_sem_labels[unlabeled_idx] = 0
        self.ds_map_sem_labels[car_idx] = 1
        self.ds_map_sem_labels[others_idx] = 2
        self.ds_map_sem_labels = self.ds_map_sem_labels.astype(np.int8) # !!!
        self.ds_map_colors = np.zeros_like(self.ds_map_points,dtype=np.float32)
        self.ds_map_colors[unlabeled_idx] = np.array([0,0,0]) / 255
        self.ds_map_colors[car_idx] = np.array([245, 150, 100]) / 255
        self.ds_map_colors[others_idx] = np.array([255, 255, 255]) / 255
        if bool_vis:
            vis.vis_cloud_with_label(self.ds_map_points,self.ds_map_colors)


        # build kd tree
        print("building tree...")
        self.ds_map = o3d.geometry.PointCloud()
        self.ds_map.points = o3d.utility.Vector3dVector(self.ds_map_points)
        self.ds_map.colors = o3d.utility.Vector3dVector(self.ds_map_colors)
        self.ds_map_kdtree = o3d.geometry.KDTreeFlann(self.ds_map)

        # assign possibility
        print("assigning possibility...")
        self.possibility = np.random.rand(self.ds_map_points.shape[0]) * 1e-3
        self.min_possibility = np.min(self.possibility)

    def removing_target_label(self,target_label_ind):
        self.ds_map_points = np.delete(self.ds_map_points,target_label_ind,0)
        self.ds_map_labels = np.delete(self.ds_map_labels,target_label_ind)
        self.ds_map_sem_labels = np.delete(self.ds_map_sem_labels, target_label_ind)
        self.ds_map_inst_labels = np.delete(self.ds_map_inst_labels, target_label_ind)
        return self.ds_map_points,self.ds_map_labels,self.ds_map_sem_labels,self.ds_map_inst_labels

