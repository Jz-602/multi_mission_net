
import pickle
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

import box_np_ops

from torch.utils.data import Dataset

import open3d as o3d

from WaymoDataset import WaymoDataset


if __name__ == "__main__":


    data_path = '/media/gk/data/lyf/aa_data/FH_waymo'

    info_path='waymo_infos_training.pkl'


    used_classes=['vehicle', 'pedestrian', 'cyclist']
    relative_path=True



    dataset = WaymoDataset(
        loading_pipelines= ['FH_LoadPointCloud', 'load_box3d'],
        root_path=data_path,
        info_path=info_path,
        create_database=True,
    )

    root_path = Path(data_path)

    db_path = root_path / "gt_database"
    dbinfo_path = root_path / "dbinfos_train.pkl"

    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    # group_counter = 0

    for index in tqdm(range(len(dataset))):
        image_idx = index
        sensor_data = dataset[index]
        if 'token' in sensor_data:
            image_idx = sensor_data['token']
        points = sensor_data['points']

        annos = sensor_data['annotations']
        gt_boxes = annos['gt_boxes']
        names = annos['gt_names']
        if gt_boxes.shape[0] == 0:
            continue

        # if index % 4 != 0:
        #     mask = (names == 'vehicle')
        #     mask = np.logical_not(mask)
        #     names = names[mask]
        #     gt_boxes = gt_boxes[mask]
        # if index % 2 != 0:
        #     mask = (names == 'pedestrian')
        #     mask = np.logical_not(mask)
        #     names = names[mask]
        #     gt_boxes = gt_boxes[mask]

        # group_dict = {}
        # group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)
        # if "group_ids" in annos:
        #     group_ids = annos["group_ids"]
        # else:
        #     group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)
        # difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        # if "difficulty" in annos:
        #     difficulty = annos["difficulty"]
        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            continue
        point_indices = box_np_ops.points_in_rbbox(
            points, gt_boxes[:, [0, 1, 2, 3, 4, 5, -1]])
        for i in range(num_obj):
            if (used_classes is None) or names[i] in used_classes:
                filename = f"{image_idx}_{names[i]}_{i}.bin"
                dirpath = os.path.join(str(db_path), names[i])
                os.makedirs(dirpath, exist_ok=True)
                filepath = os.path.join(str(db_path), names[i], filename)
                gt_points = points[point_indices[:, i]]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    try:
                        gt_points.tofile(f)
                    except:
                        print("process {} files".format(index))
                        break
            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = os.path.join(
                        db_path.stem, names[i], filename)
                else:
                    db_dump_path = str(filepath)
                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    # "difficulty": difficulty[i],
                    # "group_id": -1,
                    # "bbox": bboxes[i],
                }
                # local_group_id = group_ids[i]
                # if local_group_id not in group_dict:
                #     group_dict[local_group_id] = group_counter
                #     group_counter += 1
                # db_info["group_id"] = group_dict[local_group_id]
                # if "score" in annos:
                #     db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, 'wb') as f:
        pickle.dump(all_db_infos, f)
