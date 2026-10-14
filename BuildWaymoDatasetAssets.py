import pickle
import shutil
import sys
import os
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
MEMORY_WAYMO_DIR = CURRENT_DIR / "MEMORY_waymo"
if str(MEMORY_WAYMO_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_WAYMO_DIR))

import box_np_ops
import numpy as np
from tqdm import tqdm

from pcd_io import ReadPcdXYZ


## 先用 conda 设置 WAYMO_DATA_ROOT，然后运行：
## conda run -n torch_OD python BuildWaymoDatasetAssets.py
ROOT_PATH = Path(os.environ["WAYMO_DATA_ROOT"])

TYPE_LIST = ["unknown", "vehicle", "pedestrian", "sign", "cyclist"]
USED_CLASSES = {"vehicle", "pedestrian", "cyclist", "sign"}

FULL_INFO_FILES = {
    "training": "waymo_infos_training.pkl",
    "validation": "waymo_infos_validation.pkl",
}
SEMANTIC_INFO_FILES = {
    "training": "LYF_waymo_infos_training_semantic.pkl",
    "validation": "LYF_waymo_infos_validation_semantic.pkl",
}
POINT_INFO_FILES = {
    "training": "waymo_infos_training_point_only.pkl",
    "validation": "waymo_infos_validation_point_only.pkl",
}

DBINFO_FILES = {
    "semantic": "LYF_waymo_dbinfos_train_sample_semantic.pkl",
    "no_semantic": "LYF_waymo_dbinfos_train_sample_no_semantic.pkl",
}
DB_DIRNAMES = {
    "semantic": "LYF_waymo_gt_database_sample_semantic",
    "no_semantic": "LYF_waymo_gt_database_sample_no_semantic",
}

## sampler database 生成策略：
## 每个类别独立设置保留比例；1.0 表示全部保留，0.5 表示保留一半。
SEMANTIC_CLASS_KEEP_RATIO = {
    "vehicle": 1.0,
    "pedestrian": 1.0,
    "cyclist": 1.0,
    "sign": 1.0,
}
NO_SEMANTIC_CLASS_KEEP_RATIO = {
    "vehicle": 0.11111,
    "pedestrian": 0.22222,
    "cyclist": 0.33333,
    "sign": 0.22222,
}
MIN_DB_POINTS = 10
B_CLEAN_DB = True
NO_SEMANTIC_SAMPLE_LABEL = 100

POINT_INFO_KEYS = [
    "token",
    "sequence_name",
    "frame_timestamp_micros",
    "sample_idx",
    "original_sample_idx",
    "pose",
    "semantic_path",
]


def LoadPickle(pkl_path):
    with open(pkl_path, "rb") as file:
        return pickle.load(file)


def SavePickle(data, pkl_path):
    with open(pkl_path, "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def GetTimestampMillis(info):
    return str(info["frame_timestamp_micros"])[:-3]


def GetSemanticRelativePath(info):
    return (
        Path(info["sequence_name"])
        / "output"
        / "SemanticSegmentation"
        / f"{GetTimestampMillis(info)}.npy"
    )


def GetPcdPath(root_path, info):
    return (
        root_path
        / info["sequence_name"]
        / "output"
        / "pcd"
        / f"{GetTimestampMillis(info)}.pcd"
    )


def LoadSemanticLabel(root_path, info):
    semantic_data = np.load(root_path / GetSemanticRelativePath(info))
    semantic_data = np.asarray(semantic_data, dtype=np.int32)
    return semantic_data[:, 1].astype(np.int32, copy=False)


def BuildPointInfo(info, semantic_relative_path=None):
    ## point-only info: 只保留点云帧索引、位姿和可选语义路径；不写入目标检测 objects。
    point_info = {key: info[key] for key in POINT_INFO_KEYS if key in info}
    if semantic_relative_path is not None:
        point_info["semantic_path"] = semantic_relative_path.as_posix()
    return point_info


def BuildInfoLists(root_path, split_name, source_infos):
    semantic_infos = []
    point_infos = []
    missing_semantic_count = 0

    for info in tqdm(source_infos, desc=f"Build {split_name} infos"):
        semantic_relative_path = GetSemanticRelativePath(info)
        b_has_semantic = (root_path / semantic_relative_path).exists()

        point_infos.append(
            BuildPointInfo(
                info,
                semantic_relative_path=semantic_relative_path if b_has_semantic else None,
            )
        )

        if not b_has_semantic:
            missing_semantic_count += 1
            continue

        semantic_info = dict(info)
        semantic_info["original_sample_idx"] = info.get("sample_idx", None)
        semantic_info["sample_idx"] = len(semantic_infos)
        semantic_info["semantic_path"] = semantic_relative_path.as_posix()
        semantic_infos.append(semantic_info)

    return semantic_infos, point_infos, missing_semantic_count


def ExtractDetectionAnnotations(info, min_db_points):
    annos = info.get("objects", [])
    if len(annos) == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype="<U1")

    num_points_in_gt = np.array([ann["num_points_in_top_lidar"] for ann in annos], dtype=np.int32)
    mask_keep = (num_points_in_gt >= int(min_db_points)).reshape(-1)

    gt_boxes = np.array([ann["box"] for ann in annos], dtype=np.float32).reshape(-1, 7)
    gt_names = np.array([TYPE_LIST[ann["type"]] for ann in annos])

    return gt_boxes[mask_keep, :], gt_names[mask_keep]


def AppendDatabaseEntries(
    all_db_infos,
    class_seen_counter,
    db_root_path,
    db_dirname,
    info,
    points,
    semantic_labels,
    min_db_points,
    class_keep_ratio,
):
    """
    points: shape [N, 3]，当前帧 lidar/ego 坐标系 xyz，单位 m。
    semantic_labels: shape [N] 或 None，与 points 一一对应；若为 None，则写入忽略标签 100。
    min_db_points: 生成 database 前用目标框原始 num_points_in_top_lidar 过滤，减少低质量小目标。
    """
    if semantic_labels is not None and semantic_labels.shape[0] != points.shape[0]:
        raise ValueError(
            f"point count and semantic label count mismatch for {info['token']}: "
            f"{points.shape[0]} vs {semantic_labels.shape[0]}"
        )

    gt_boxes, gt_names = ExtractDetectionAnnotations(info, min_db_points)
    if gt_boxes.shape[0] == 0:
        return

    point_indices = box_np_ops.points_in_rbbox(points, gt_boxes[:, [0, 1, 2, 3, 4, 5, -1]])
    image_idx = info["token"]

    for gt_index in range(gt_boxes.shape[0]):
        gt_name = gt_names[gt_index]
        if gt_name not in USED_CLASSES:
            continue

        gt_mask = point_indices[:, gt_index]
        gt_points = points[gt_mask].copy()
        if gt_points.shape[0] == 0:
            continue
        if semantic_labels is None:
            gt_semantic_labels = np.full(
                (gt_points.shape[0],),
                NO_SEMANTIC_SAMPLE_LABEL,
                dtype=np.int32,
            )
        else:
            gt_semantic_labels = semantic_labels[gt_mask].astype(np.int32, copy=True)

        class_seen_counter[gt_name] = class_seen_counter.get(gt_name, 0) + 1
        keep_ratio = float(class_keep_ratio.get(gt_name, 1.0))
        previous_keep_count = int((class_seen_counter[gt_name] - 1) * keep_ratio)
        current_keep_count = int(class_seen_counter[gt_name] * keep_ratio)
        if current_keep_count <= previous_keep_count:
            continue

        ## gt_points: shape [M, 3]，由当前帧 lidar/ego 坐标系转为目标框中心局部坐标，供 sampler 插回当前帧。
        gt_points[:, :3] -= gt_boxes[gt_index, :3]

        class_dir = db_root_path / gt_name
        class_dir.mkdir(parents=True, exist_ok=True)

        base_filename = f"{image_idx}_{gt_name}_{gt_index}"
        point_path = class_dir / f"{base_filename}.bin"
        label_path = class_dir / f"{base_filename}.label.npy"

        gt_points.astype(np.float32).tofile(point_path)
        np.save(label_path, gt_semantic_labels)

        db_info = {
            "name": gt_name,
            "path": (Path(db_dirname) / gt_name / point_path.name).as_posix(),
            "semantic_label_path": (Path(db_dirname) / gt_name / label_path.name).as_posix(),
            "box3d_lidar": gt_boxes[gt_index].copy(),
            "num_points_in_gt": int(gt_points.shape[0]),
        }
        all_db_infos.setdefault(gt_name, []).append(db_info)


def BuildDatabaseInfos(
    root_path,
    train_infos,
    dbinfo_filename,
    db_dirname,
    b_clean_db,
    min_db_points,
    class_keep_ratio,
    b_use_semantic_frames,
):
    db_root_path = root_path / db_dirname
    if b_clean_db and db_root_path.exists():
        shutil.rmtree(db_root_path)
    db_root_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    class_seen_counter = {}
    for info in tqdm(train_infos, desc="Build training GT database"):
        b_has_semantic = (root_path / GetSemanticRelativePath(info)).exists()
        if b_use_semantic_frames != b_has_semantic:
            continue

        pcd_path = GetPcdPath(root_path, info)
        if not pcd_path.exists():
            continue

        points = ReadPcdXYZ(pcd_path)
        semantic_labels = LoadSemanticLabel(root_path, info) if b_has_semantic else None
        AppendDatabaseEntries(
            all_db_infos=all_db_infos,
            class_seen_counter=class_seen_counter,
            db_root_path=db_root_path,
            db_dirname=db_dirname,
            info=info,
            points=points,
            semantic_labels=semantic_labels,
            min_db_points=min_db_points,
            class_keep_ratio=class_keep_ratio,
        )

    SavePickle(all_db_infos, root_path / dbinfo_filename)
    return all_db_infos, class_seen_counter


def PrintInfoSummary(name, info_list):
    if len(info_list) == 0:
        print(f"{name}: 0 frames")
        return
    first_info = info_list[0]
    print(
        f"{name}: frames={len(info_list)}, "
        f"objects={'objects' in first_info}, "
        f"semantic_path={'semantic_path' in first_info}, "
        f"keys={sorted(first_info.keys())}"
    )


def Main():
    source_info_dict = {}
    for split_name, info_filename in FULL_INFO_FILES.items():
        source_infos = LoadPickle(ROOT_PATH / info_filename)
        source_info_dict[split_name] = source_infos

        semantic_infos, point_infos, missing_semantic_count = BuildInfoLists(
            root_path=ROOT_PATH,
            split_name=split_name,
            source_infos=source_infos,
        )

        SavePickle(semantic_infos, ROOT_PATH / SEMANTIC_INFO_FILES[split_name])
        SavePickle(point_infos, ROOT_PATH / POINT_INFO_FILES[split_name])

        print(f"[{split_name}] source frames: {len(source_infos)}")
        print(f"[{split_name}] missing semantic frames: {missing_semantic_count}")
        PrintInfoSummary(SEMANTIC_INFO_FILES[split_name], semantic_infos)
        PrintInfoSummary(POINT_INFO_FILES[split_name], point_infos)

    database_build_cfgs = [
        ("semantic", True, SEMANTIC_CLASS_KEEP_RATIO),
        ("no_semantic", False, NO_SEMANTIC_CLASS_KEEP_RATIO),
    ]
    for database_name, b_use_semantic_frames, class_keep_ratio in database_build_cfgs:
        all_db_infos, class_seen_counter = BuildDatabaseInfos(
            root_path=ROOT_PATH,
            train_infos=source_info_dict["training"],
            dbinfo_filename=DBINFO_FILES[database_name],
            db_dirname=DB_DIRNAMES[database_name],
            b_clean_db=B_CLEAN_DB,
            min_db_points=MIN_DB_POINTS,
            class_keep_ratio=class_keep_ratio,
            b_use_semantic_frames=b_use_semantic_frames,
        )
        print(f"{DBINFO_FILES[database_name]}:")
        print(f"  db_dirname: {DB_DIRNAMES[database_name]}")
        print(f"  class_keep_ratio: {class_keep_ratio}")
        for class_name in sorted(all_db_infos.keys()):
            source_count = class_seen_counter.get(class_name, len(all_db_infos[class_name]))
            print(f"  {class_name}: {len(all_db_infos[class_name])} / {source_count}")


if __name__ == "__main__":

    
    Main()
