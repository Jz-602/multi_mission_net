import argparse
import pickle
from pathlib import Path

import box_np_ops
import numpy as np
import open3d as o3d
from tqdm import tqdm


TYPE_LIST = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

DEFAULT_ROOT_PATH = '/media/gk/data/lyf/aa_data/FH_waymo'
DEFAULT_SOURCE_INFO_FILENAME = 'waymo_infos_training.pkl'
DEFAULT_OUTPUT_INFO_FILENAME = 'LYF_waymo_infos_training_semantic.pkl'
DEFAULT_OUTPUT_DBINFO_FILENAME = 'LYF_dbinfos_train_semantic.pkl'
DEFAULT_OUTPUT_DB_DIRNAME = 'LYF_gt_database_semantic'
DEFAULT_VAL_SOURCE_INFO_FILENAME = 'waymo_infos_validation.pkl'
DEFAULT_VAL_OUTPUT_INFO_FILENAME = 'LYF_waymo_infos_validation_semantic.pkl'
DEFAULT_USED_CLASSES = ['vehicle', 'pedestrian', 'cyclist']


def ParseArguments():
    parser = argparse.ArgumentParser(
        description='Filter frames without semantic labels and build a semantic-aware GT database.'
    )
    parser.add_argument('--root_path', default=DEFAULT_ROOT_PATH, help='Waymo FH data root path.')
    parser.add_argument('--source_info_filename', default=DEFAULT_SOURCE_INFO_FILENAME, help='Input frame-wise info pkl filename.')
    parser.add_argument('--output_info_filename', default=DEFAULT_OUTPUT_INFO_FILENAME, help='Output filtered frame-wise info pkl filename.')
    parser.add_argument('--output_dbinfo_filename', default=DEFAULT_OUTPUT_DBINFO_FILENAME, help='Output GT database info pkl filename.')
    parser.add_argument('--output_db_dirname', default=DEFAULT_OUTPUT_DB_DIRNAME, help='Output GT database directory name.')
    return parser.parse_args()


def LoadFramePointCloudAndSemantic(root_path, info):
    timestamp_ms = str(info['frame_timestamp_micros'])[:-3]
    point_cloud_relative_path = Path(info['sequence_name']) / 'output' / 'pcd' / f'{timestamp_ms}.pcd'
    semantic_relative_path = Path(info['sequence_name']) / 'output' / 'SemanticSegmentation' / f'{timestamp_ms}.npy'

    point_cloud_path = root_path / point_cloud_relative_path
    semantic_path = root_path / semantic_relative_path

    if not point_cloud_path.exists():
        return None

    if not semantic_path.exists():
        return None

    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    points = np.asarray(point_cloud.points, dtype=np.float32)

    semantic_data = np.load(semantic_path)
    semantic_data = np.asarray(semantic_data, dtype=np.int32)
    semantic_label = semantic_data[:, 1].astype(np.int32, copy=False)

    if points.shape[0] != semantic_label.shape[0]:
        raise ValueError(
            f'Point count and semantic label count mismatch for {info["token"]}: '
            f'{points.shape[0]} vs {semantic_label.shape[0]}'
        )

    return {
        'points': points,
        'semantic_label': semantic_label,
        'semantic_path': semantic_relative_path.as_posix(),
    }


def ExtractDetectionAnnotations(info):
    annos = info['objects']
    num_points_in_gt = np.array([ann['num_points_in_top_lidar'] for ann in annos], dtype=np.int32)
    mask_not_zero = (num_points_in_gt > 0).reshape(-1)

    gt_boxes = np.array([ann['box'] for ann in annos], dtype=np.float32).reshape(-1, 7)
    gt_names = np.array([TYPE_LIST[ann['type']] for ann in annos])

    gt_boxes = gt_boxes[mask_not_zero, :]
    gt_names = gt_names[mask_not_zero]

    return gt_boxes, gt_names


def AppendDatabaseEntries(
    all_db_infos,
    db_root_path,
    db_dirname,
    info,
    points,
    semantic_label,
    used_classes,
):
    gt_boxes, gt_names = ExtractDetectionAnnotations(info)
    if gt_boxes.shape[0] == 0:
        return

    point_indices = box_np_ops.points_in_rbbox(points, gt_boxes[:, [0, 1, 2, 3, 4, 5, -1]])
    image_idx = info['token']

    for gt_index in range(gt_boxes.shape[0]):
        gt_name = gt_names[gt_index]
        if gt_name not in used_classes:
            continue

        gt_mask = point_indices[:, gt_index]
        gt_points = points[gt_mask].copy()
        gt_semantic_label = semantic_label[gt_mask].copy()

        if gt_points.shape[0] == 0:
            continue

        gt_points[:, :3] -= gt_boxes[gt_index, :3]

        class_dir = db_root_path / gt_name
        class_dir.mkdir(parents=True, exist_ok=True)

        base_filename = f'{image_idx}_{gt_name}_{gt_index}'
        point_filename = f'{base_filename}.bin'
        semantic_filename = f'{base_filename}_semantic.npy'

        point_path = class_dir / point_filename
        semantic_path = class_dir / semantic_filename

        gt_points.astype(np.float32).tofile(point_path)
        np.save(semantic_path, gt_semantic_label.astype(np.int32))

        db_info = {
            'name': gt_name,
            'path': str(Path(db_dirname) / gt_name / point_filename),
            'semantic_path': str(Path(db_dirname) / gt_name / semantic_filename),
            'image_idx': image_idx,
            'gt_idx': gt_index,
            'box3d_lidar': gt_boxes[gt_index].copy(),
            'num_points_in_gt': int(gt_points.shape[0]),
        }

        if gt_name in all_db_infos:
            all_db_infos[gt_name].append(db_info)
        else:
            all_db_infos[gt_name] = [db_info]


def Main():
    args = ParseArguments()

    root_path = Path(args.root_path)
    source_info_path = root_path / args.source_info_filename
    output_info_path = root_path / args.output_info_filename
    output_dbinfo_path = root_path / args.output_dbinfo_filename
    output_db_root_path = root_path / args.output_db_dirname
    val_source_info_path = root_path / DEFAULT_VAL_SOURCE_INFO_FILENAME
    val_output_info_path = root_path / DEFAULT_VAL_OUTPUT_INFO_FILENAME
    used_classes = set(DEFAULT_USED_CLASSES)

    ## 处理 training：筛掉无语义帧，并构建语义增强库
    with open(source_info_path, 'rb') as f:
        source_infos = pickle.load(f)

    filtered_infos = []
    all_db_infos = {}
    num_missing_semantic_frames = 0

    for info in tqdm(source_infos, desc='Building semantic-aware infos and database'):
        frame_data = LoadFramePointCloudAndSemantic(root_path, info)
        if frame_data is None:
            num_missing_semantic_frames += 1
            continue

        new_info = dict(info)
        new_info['original_sample_idx'] = info.get('sample_idx', None)
        new_info['sample_idx'] = len(filtered_infos)
        new_info['semantic_path'] = frame_data['semantic_path']
        filtered_infos.append(new_info)

        AppendDatabaseEntries(
            all_db_infos=all_db_infos,
            db_root_path=output_db_root_path,
            db_dirname=args.output_db_dirname,
            info=new_info,
            points=frame_data['points'],
            semantic_label=frame_data['semantic_label'],
            used_classes=used_classes,
        )

    with open(output_info_path, 'wb') as f:
        pickle.dump(filtered_infos, f)

    with open(output_dbinfo_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

    print(f'[training] source frame count: {len(source_infos)}')
    print(f'[training] filtered frame count: {len(filtered_infos)}')
    print(f'[training] missing semantic frame count: {num_missing_semantic_frames}')
    print(f'[training] filtered info path: {output_info_path}')
    print(f'[training] database info path: {output_dbinfo_path}')
    print(f'[training] database root path: {output_db_root_path}')
    for class_name, db_infos in all_db_infos.items():
        print(f'[training] loaded {len(db_infos)} semantic database infos for class {class_name}')

    ## 处理 validation：只筛掉无语义帧，并补 semantic_path
    with open(val_source_info_path, 'rb') as f:
        val_source_infos = pickle.load(f)

    val_filtered_infos = []
    val_num_missing_semantic_frames = 0

    for info in tqdm(val_source_infos, desc='Building semantic-aware validation infos'):
        frame_data = LoadFramePointCloudAndSemantic(root_path, info)
        if frame_data is None:
            val_num_missing_semantic_frames += 1
            continue

        new_info = dict(info)
        new_info['original_sample_idx'] = info.get('sample_idx', None)
        new_info['sample_idx'] = len(val_filtered_infos)
        new_info['semantic_path'] = frame_data['semantic_path']
        val_filtered_infos.append(new_info)

    with open(val_output_info_path, 'wb') as f:
        pickle.dump(val_filtered_infos, f)

    print(f'[validation] source frame count: {len(val_source_infos)}')
    print(f'[validation] filtered frame count: {len(val_filtered_infos)}')
    print(f'[validation] missing semantic frame count: {val_num_missing_semantic_frames}')
    print(f'[validation] filtered info path: {val_output_info_path}')


if __name__ == '__main__':
    Main()
