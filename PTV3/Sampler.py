import copy
import pathlib
import pickle

import box_np_ops
import numpy as np
import yaml


class BatchSampler:
    def __init__(self, sampled_list, name=None, shuffle=True):
        self.sampled_list = sampled_list
        self.indices = np.arange(len(sampled_list))
        self.name = name
        self.b_shuffle = shuffle
        self.idx = 0
        self.example_num = len(sampled_list)
        if self.b_shuffle:
            np.random.shuffle(self.indices)

    def Sample(self, sample_num):
        if self.idx + sample_num >= self.example_num:
            sampled_indices = self.indices[self.idx:].copy()
            self.Reset()
        else:
            sampled_indices = self.indices[self.idx:self.idx + sample_num]
            self.idx += sample_num
        return [self.sampled_list[sampled_idx] for sampled_idx in sampled_indices]

    def Reset(self):
        if self.b_shuffle:
            np.random.shuffle(self.indices)
        self.idx = 0


class DBFilterByMinNumPoint:
    def __init__(self, min_gt_point_dict):
        self.min_gt_point_dict = min_gt_point_dict

    def __call__(self, db_infos):
        for name, min_num in self.min_gt_point_dict.items():
            if min_num <= 0 or name not in db_infos:
                continue
            filtered_infos = []
            for info in db_infos[name]:
                if info["num_points_in_gt"] >= min_num:
                    filtered_infos.append(info)
            db_infos[name] = filtered_infos
        return db_infos


class DataBaseSamplerV2:
    def __init__(
        self,
        root_path,
        dbinfo_path,
        groups,
        min_gt_point_dict,
        label_mapping,
    ):
        self.root_path = pathlib.Path(root_path)
        db_infos = self.LoadDatabaseInfos(dbinfo_path)

        self.sampler_semantic_label = self.LoadSamplerSemanticLabel(label_mapping)
        self.db_infos = DBFilterByMinNumPoint(min_gt_point_dict)(db_infos)

        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in groups:
            group_names = list(group_info.keys())
            self.sample_classes += group_names
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for class_name, class_db_infos in self.db_infos.items():
            self.sampler_dict[class_name] = BatchSampler(class_db_infos, class_name)

    def LoadDatabaseInfos(self, dbinfo_path):
        dbinfo_paths = [dbinfo_path] if isinstance(dbinfo_path, str) else list(dbinfo_path)
        merged_db_infos = {}
        for cur_dbinfo_path in dbinfo_paths:
            with open(self.root_path / cur_dbinfo_path, "rb") as file:
                db_infos = pickle.load(file)
            for class_name, class_db_infos in db_infos.items():
                merged_db_infos.setdefault(class_name, []).extend(class_db_infos)
        return merged_db_infos

    def LoadSamplerSemanticLabel(self, label_mapping):
        label_mapping_path = pathlib.Path(__file__).resolve().parent / "label_mapping" / label_mapping
        with open(label_mapping_path, "r", encoding="utf-8") as file:
            label_mapping_cfg = yaml.safe_load(file)
        return {
            str(class_name): int(semantic_label)
            for class_name, semantic_label in label_mapping_cfg["sampler_semantic_label"].items()
        }

    def SampleAll(self, gt_boxes, gt_names):
        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for class_name, max_sample_num in zip(self.sample_classes, self.sample_max_nums):
            if class_name not in self.sampler_dict:
                continue
            sampled_num = int(max_sample_num - np.sum([name == class_name for name in gt_names]))
            if sampled_num <= 0:
                continue

            sampled_cls = self.SampleClassFH(class_name, sampled_num, avoid_coll_boxes)
            sampled += sampled_cls
            if len(sampled_cls) == 0:
                continue

            sampled_gt_box = np.stack([sample_info["box3d_lidar"] for sample_info in sampled_cls], axis=0)
            sampled_gt_boxes.append(sampled_gt_box)
            avoid_coll_boxes = np.concatenate([avoid_coll_boxes, sampled_gt_box], axis=0)

        if len(sampled) == 0:
            return None

        ## sampled_points: [M, 3]，当前帧 LiDAR/车体坐标系 xyz；db 内点是以目标框中心为原点的局部坐标，这里平移到采样框中心。
        sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
        sampled_points_list = []
        sampled_semantic_label_list = []
        for info in sampled:
            sampled_points = np.fromfile(self.root_path / info["path"], dtype=np.float32).reshape(-1, 3)
            if "semantic_label_path" in info:
                sampled_semantic_label = np.load(self.root_path / info["semantic_label_path"])
                sampled_semantic_label = np.asarray(sampled_semantic_label, dtype=np.int32)
                if sampled_semantic_label.shape[0] != sampled_points.shape[0]:
                    raise ValueError(
                        f"sample point count and label count mismatch for {info['path']}: "
                        f"{sampled_points.shape[0]} vs {sampled_semantic_label.shape[0]}"
                    )
            else:
                sampled_semantic_label = np.full(
                    (sampled_points.shape[0],),
                    self.sampler_semantic_label[info["name"]],
                    dtype=np.int32,
                )
            if "rot_transform" in info:
                sampled_points[:, :3] = box_np_ops.yaw_rotation(
                    sampled_points[:, :3],
                    info["rot_transform"],
                    axis=2,
                )
            sampled_points[:, :3] += info["box3d_lidar"][:3]
            sampled_points_list.append(sampled_points)
            sampled_semantic_label_list.append(sampled_semantic_label)

        return {
            "gt_names": np.array([sample_info["name"] for sample_info in sampled]),
            "gt_boxes": sampled_gt_boxes.astype(np.float32),
            "points": np.concatenate(sampled_points_list, axis=0).astype(np.float32),
            "semantic_label": np.concatenate(sampled_semantic_label_list, axis=0).astype(np.int32),
        }

    def SampleClassFH(self, name, sample_num, gt_boxes):
        sampled = copy.deepcopy(self.sampler_dict[name].Sample(sample_num))
        if len(sampled) == 0:
            return []

        sampled_boxes = np.stack([info["box3d_lidar"] for info in sampled], axis=0)
        sampled_boxes_bv = box_np_ops.center_to_corner_box2d(sampled_boxes[:, [0, 1, 3, 4, -1]])

        if gt_boxes.shape[0] > 0:
            gt_boxes_bv = box_np_ops.center_to_corner_box2d(gt_boxes[:, [0, 1, 3, 4, -1]])
            total_bv = np.concatenate([gt_boxes_bv, sampled_boxes_bv], axis=0)
            num_gt = gt_boxes.shape[0]
        else:
            total_bv = sampled_boxes_bv
            num_gt = 0

        coll_mat = box_np_ops.box_collision_test(total_bv, total_bv)
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for box_idx in range(num_gt, num_gt + len(sampled)):
            if coll_mat[box_idx].any():
                coll_mat[box_idx] = False
                coll_mat[:, box_idx] = False
            else:
                valid_samples.append(sampled[box_idx - num_gt])
        return valid_samples
