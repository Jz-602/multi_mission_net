import numpy as np
import open3d as o3d
import math
import yaml

class CDataLoaderKITTI:
    @staticmethod
    def load_bin_xyzi(bin_file):
        scan = np.fromfile(bin_file, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]
        remissions = scan[:, 3]
        return points, remissions

    @staticmethod
    def load_bin_label(label_file):
        label = np.fromfile(label_file, dtype=np.uint32)
        # label = label.reshape((-1))
        return label

    @staticmethod
    def load_bin_semlabel(label_file):
        label = np.fromfile(label_file, dtype=np.uint32)
        sem_label = label & 0xFFFF
        # label = label.reshape((-1))
        return sem_label

    @staticmethod
    def load_pcd_xyzi(pcd_file):
        pcd = o3d.t.io.read_point_cloud(pcd_file)
        pcd_intensity = pcd.point["intensity"]
        pcd_points = pcd.point["positions"]
        pcd_intensity = pcd_intensity[:, :].numpy().squeeze(1).astype(np.uint32)
        pcd_points = pcd_points[:, :].numpy().astype(np.float32)
        return pcd_points,pcd_intensity


    @staticmethod
    def calc_label_color_lut(color_dict):
        max_sem_key = 0
        for key, data in color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)  # [max_sem_key+100,3]的nparray
        for key, value in color_dict.items():
            sem_color_lut[key] = np.array(value, np.float32) / 255.0  # 转换到[0,1]范围内
        return sem_color_lut

    @staticmethod
    def calc_sem_label_color(sem_color_lut, sem_label):
        sem_label_color = sem_color_lut[sem_label]
        sem_label_color = sem_label_color.reshape((-1, 3))
        return sem_label_color

    def load_kitti_yaml_original(self,config_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/config/semantic-kitti.yaml'):
        CFG = yaml.safe_load(open(config_file, 'r'))
        labels_dict = CFG["labels"]
        color_dict = CFG["color_map"]
        color_dict_inv = CFG["color_map_inv"]
        contents_dict = CFG["content"]
        learning_map = CFG["learning_map"]
        learning_map_inv = CFG["learning_map_inv"]
        sem_color_lut = self.calc_label_color_lut(color_dict)

        return labels_dict, color_dict, color_dict_inv, contents_dict, learning_map, learning_map_inv, sem_color_lut

    def load_kitti_yaml_randla(self):
        config = '/media/alv-2/Rambo/Data/KITTI-Semantic/config/semantic-kitti-randla.yaml'
        CFG = yaml.safe_load(open(config, 'r'))
        labels_dict = CFG["labels"]
        color_dict = CFG["color_map"]
        color_dict_inv = CFG["color_map_inv"]
        contents_dict = CFG["content"]
        learning_map = CFG["learning_map"]
        learning_map_inv = CFG["learning_map_inv"]
        sem_color_lut = self.calc_label_color_lut(color_dict)
        learning_map_8 = CFG["learning_map_8"]
        remap_lut_8 = self.remap_dict_to_lut(learning_map_8)
        return labels_dict, color_dict, color_dict_inv, contents_dict, learning_map, learning_map_inv, sem_color_lut, learning_map_8, remap_lut_8

    @staticmethod
    def remap_dict_to_lut(remap_dict):
        max_key_remap = max(remap_dict.keys())
        remap_lut = np.zeros((max_key_remap + 100), dtype=np.int32)
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
        return remap_lut

    @staticmethod
    def semlabel_unique_print(semlabel, label_dict):
        label_unique = np.unique(semlabel)
        print(label_unique)
        label_value = []
        for l in label_unique:
            value = label_dict[l]
            # print(value)
            label_value.append(value)
        print("label contains:", label_unique, '\n', label_value)

    def semlabel_trans(self,label_src,map_dict_src_to_dst,bool_print=False):

        remap_lut = self.remap_dict_to_lut(map_dict_src_to_dst)
        label_dst = remap_lut[label_src]
        if bool_print:
            print("label before:", np.unique(label_src))
            print("label after:", np.unique(label_dst))
        return label_dst


if __name__ == '__main__':
    from CVisualizer import CVisualizer
    vis = CVisualizer()

    loader = CDataLoaderKITTI()
    labels_dict, color_dict, color_dict_inv, contents_dict, learning_map, learning_map_inv, sem_color_lut = loader.load_kitti_yaml_original()

    bin_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/00/velodyne/000000.bin'
    label_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/sequences/00/labels/000000.label'

    points, remissions = loader.load_bin_xyzi(bin_file)
    label = loader.load_bin_label(label_file)
    semlabel = loader.load_bin_semlabel(label_file)
    semlabel_color = loader.calc_sem_label_color(sem_color_lut,semlabel)
    # vis.vis_cloud_with_label(points,semlabel_color)
    # vis.vis_cloud(points)

    remap_lut = loader.remap_dict_to_lut(learning_map)
    loader.semlabel_unique_print(semlabel,labels_dict)
    label_dst = loader.semlabel_trans(semlabel,learning_map,bool_print=True)

    print(1)