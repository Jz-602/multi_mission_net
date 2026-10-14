import yaml
import numpy as np

class CRemapper():
    @staticmethod
    def remaplut_invlut_colorlut(yaml_file = '/media/alv-2/Rambo/Data/KITTI-Semantic/config/semantic-kitti-new.yaml'):

        DATA = yaml.safe_load(open(yaml_file, 'r'))

        class_remap = DATA["learning_map"]
        class_inv_remap = DATA["learning_map_inv"]
        color_dict = DATA["color_map"]

        maxkey = max(class_remap.keys())
        remap_lut = np.zeros((maxkey + 10), dtype=np.int32)
        remap_lut[list(class_remap.keys())] = list(class_remap.values())

        maxkey = max(class_inv_remap.keys())
        inv_lut = np.zeros((maxkey + 10), dtype=np.int32)
        inv_lut[list(class_inv_remap.keys())] = list(class_inv_remap.values())

        max_sem_key = 0
        for key, data in color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)  # [max_sem_key+100,3]的nparray
        for key, value in color_dict.items():
            sem_color_lut[key] = np.array(value, np.float32) / 255.0  # 转换到[0,1]范围内

        return remap_lut,inv_lut,sem_color_lut



