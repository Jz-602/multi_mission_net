"""复用原补框算法，按完整原序列计算，仅导出既定监督目标的新检测真值。

原始数据只读；输出分序列原子保存并可恢复，最终 infos 保留原目标顺序及所有原对象。
补框点数来自当前帧 TOP/其他 LiDAR PCD（已去 NLZ），不是锚点点数。补框难度固定 L2。
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / '3mission_model'))
from gt_generate.object_box.complete_detection_boxes import BuildParser, CompleteDetectionBoxes, TerrainSupport
from pcd_io import ReadPcdXYZ

SOURCE_FILES = ('output/detection_info.pkl', 'output/label/ground_truth/terrain_z_inpainted.bin',
                'output/label/ground_truth/terrain_ground_type_map.bin',
                'output/label/ground_truth/map_info.txt', 'output/label/ground_truth/terrain_only_summary.json')


def SHA(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def TokenSHA(tokens):
    return hashlib.sha256('\n'.join(tokens).encode()).hexdigest()


def Save(path, value, b_pickle=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('wb' if b_pickle else 'w') as file:
        if b_pickle:
            pickle.dump(value, file, protocol=4)
        else:
            json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def Equal(first, second):
    if isinstance(first, np.ndarray):
        return (isinstance(second, np.ndarray) and first.dtype == second.dtype
                and np.array_equal(first, second, equal_nan=True))
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(Equal(first[key], second[key]) for key in first)
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(Equal(a, b) for a, b in zip(first, second))
    return first == second


def CountPoints(points, boxes):
    """points [N,3] 与 boxes [K,7] 同处当前 TOP LiDAR 坐标系，框中心+yaw格式。

    先以包围球查询，再将相对坐标逆 yaw 旋转入框局部系，按闭区间半尺寸计数。
    """
    tree = cKDTree(points)
    counts = []
    for box in boxes:
        nearby = tree.query_ball_point(box[:3], np.linalg.norm(box[3:6]) * 0.5 + 1e-9)
        delta = points[nearby].astype(np.float64) - box[:3]
        c, s = np.cos(box[6]), np.sin(box[6])
        local = delta @ np.asarray([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        counts.append(int(np.count_nonzero(np.all(np.abs(local) <= box[3:6] * 0.5, axis=1))))
    return counts


def ExportSequence(task):
    data_root, output_dir, sequence, tokens, parameters, binding_sha = task
    start = time.monotonic()
    sequence_root = Path(data_root) / sequence
    output = Path(output_dir) / 'sequences' / f'{sequence}.pkl'
    receipt_path = output.with_suffix('.json')
    source_hashes = {name: SHA(sequence_root / name) for name in SOURCE_FILES}
    if receipt_path.exists() and output.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt['binding_sha256'] != binding_sha or receipt['source_sha256'] != source_hashes
                or receipt['target_token_sha256'] != TokenSHA(tokens)):
            raise ValueError(f'已有输出绑定不一致，必须使用新输出目录：{sequence}')
        if SHA(output) != receipt['output_sha256']:
            raise ValueError(f'已有输出损坏：{output}')
        receipt['b_reused'] = True
        return receipt
    with (sequence_root / SOURCE_FILES[0]).open('rb') as file:
        original_frames = pickle.load(file)
    original_bytes = pickle.dumps(original_frames, protocol=4)
    terrain = TerrainSupport(sequence_root / 'output/label/ground_truth')
    frames, supplements, events = CompleteDetectionBoxes(original_frames, terrain, **parameters)
    if pickle.dumps(original_frames, protocol=4) != original_bytes:
        raise AssertionError('补框函数修改了原始帧')
    frame_map = {frame['token']: (idx, frame) for idx, frame in enumerate(frames)}
    if len(frame_map) != len(frames):
        raise ValueError(f'原帧 token 不唯一：{sequence}')
    result_frames, pcd_sources = [], []
    count_all, count_zero, class_counts = 0, 0, {}
    for token in tokens:
        frame_idx, frame = frame_map[token]
        objects = list(frame['objects'])
        generated = supplements[frame_idx]
        if generated:
            timestamp = str(frame['frame_timestamp_micros'])[:-3]
            paths = [sequence_root / 'output' / folder / f'{timestamp}.pcd' for folder in ('pcd', 'pcd2')]
            boxes = np.stack([obj['box'] for obj in generated])
            if not np.all(np.isfinite(boxes)) or np.any(boxes[:, 3:6] <= 0):
                raise ValueError(f'无效补框：{token}')
            top_counts = CountPoints(ReadPcdXYZ(paths[0]), boxes)
            other_counts = CountPoints(ReadPcdXYZ(paths[1]), boxes)
            pcd_sources.extend({'path': str(path.relative_to(data_root)), 'sha256': SHA(path)} for path in paths)
            original_ids = {str(obj['id']) for obj in objects}
            for obj, top_count, other_count in zip(generated, top_counts, other_counts):
                if str(obj['id']) in original_ids:
                    raise AssertionError(f'同一帧重复原始目标ID：{token}')
                original_ids.add(str(obj['id']))
                event = events[obj['event_index']]
                anchor_frame = frames[event['anchor_frame']]
                anchor = next(item for item in anchor_frame['objects'] if str(item['id']) == str(obj['id']))
                # 世界系速度 [3] 与原模型同一匀加速运动方程；box 保留原算法 float64。
                speed = np.asarray(anchor['speed']) + event['acceleration_world'] * obj['elapsed_s']
                supplemental = dict(obj)
                supplemental.update(b_is_supplemental=True, difficulty_level=np.asarray([2., 2.]),
                                    num_points_in_top_lidar=top_count,
                                    num_points_in_all_lidar=top_count + other_count, speed=speed,
                                    source_metadata={'generator': 'CompleteDetectionBoxes',
                                                     'binding_sha256': binding_sha,
                                                     'anchor_token': anchor_frame['token'],
                                                     'point_count_source': 'current_frame_pcd_and_pcd2_after_nlz_removal',
                                                     'difficulty_policy': 'supplemental_level_2'})
                objects.append(supplemental)
                count_all += 1
                count_zero += top_count == 0
                key = str(int(obj['type']))
                class_counts[key] = class_counts.get(key, 0) + 1
        result = dict(frame)
        result['objects'] = objects
        result_frames.append(result)
    payload = {'frames': result_frames, 'events': events, 'binding_sha256': binding_sha,
               'point_cloud_sources': pcd_sources}
    Save(output, payload, b_pickle=True)
    receipt = {'sequence': sequence, 'binding_sha256': binding_sha, 'source_sha256': source_hashes,
               'target_token_sha256': TokenSHA(tokens), 'target_count': len(tokens),
               'full_input_frame_count': len(frames), 'all_frame_supplement_count': sum(map(len, supplements)),
               'supplement_count': count_all, 'supplement_zero_top_points': count_zero,
               'supplement_class_counts': class_counts, 'output_sha256': SHA(output),
               'output_bytes': output.stat().st_size, 'elapsed_seconds': time.monotonic() - start,
               'b_reused': False}
    Save(receipt_path, receipt)
    return receipt


def Main():
    parser = BuildParser(__doc__)
    parser.set_defaults(sequence=None)
    parser.add_argument('--output-dir', default='detection_completed_20261007', help='相对 data-root 或绝对路径')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit-sequences', type=int, default=0, help='仅试跑，非零时不组装完整 infos')
    args = parser.parse_args()
    if not args.data_root:
        raise ValueError('请设置 WAYMO_DATA_ROOT')
    data_root = Path(args.data_root).resolve()
    output_dir = data_root / args.output_dir
    parameters = {key: getattr(args, key) for key in ('max_distance_m', 'motion_model', 'history_frames',
                                                    'max_acceleration_mps2', 'max_anchor_distance_m')}
    source_paths = {split: data_root / f'LYF_waymo_infos_{split}_semantic.pkl' for split in ('training', 'validation')}
    source_infos, tasks, split_sequences = {}, {}, {}
    for split, path in source_paths.items():
        with path.open('rb') as file:
            infos = pickle.load(file)
        source_infos[split] = infos
        tokens = [info['token'] for info in infos]
        if len(set(tokens)) != len(tokens):
            raise ValueError(f'重复token：{split}')
        split_sequences[split] = sorted({info['sequence_name'] for info in infos})
        for info in infos:
            tasks.setdefault(info['sequence_name'], []).append(info['token'])
    if set(split_sequences['training']) & set(split_sequences['validation']):
        raise ValueError('训练验证序列交叉')
    source_scripts = [Path(__file__).resolve(), Path(__file__).with_name('complete_detection_boxes.py'),
                      PROJECT_ROOT / 'gt_generate/common/io_utils.py', PROJECT_ROOT / 'gt_generate/common/constants.py',
                      PROJECT_ROOT / '3mission_model/pcd_io.py']
    binding = {'schema_version': 1, 'parameters': parameters,
               'source_scripts': {str(path.relative_to(PROJECT_ROOT)): SHA(path) for path in source_scripts},
               'source_infos': {split: {'path': str(path.relative_to(data_root)), 'sha256': SHA(path),
                                       'target_count': len(source_infos[split]),
                                       'token_sha256': TokenSHA([item['token'] for item in source_infos[split]]),
                                       'sequence_count': len(split_sequences[split])} for split, path in source_paths.items()},
               'point_count_policy': 'actual_current_pcd_TOP_plus_pcd2_others_after_nlz_removal',
               'difficulty_policy': 'all_supplemental_level_2_original_unchanged',
               'b_keep_zero_point_supplements': True}
    binding_sha = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    binding_path = output_dir / 'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError('输出目录已绑定其他代码/输入/参数，请指定新目录')
    missing = [str(data_root / sequence / name) for sequence in tasks for name in SOURCE_FILES
               if not (data_root / sequence / name).is_file()]
    if missing:
        raise FileNotFoundError(f'缺少完整序列输入：{missing[:10]}，共{len(missing)}个')
    if shutil.disk_usage(data_root).free < 5 * 2**30:
        raise RuntimeError('可用空间不足5GiB')
    Save(binding_path, binding)
    selected = [sequence for sequence in tasks if args.sequence is None or sequence == args.sequence]
    if args.sequence and not selected:
        raise ValueError(f'序列不在候选池：{args.sequence}')
    if args.limit_sequences:
        selected = selected[:args.limit_sequences]
    work = [(str(data_root), str(output_dir), sequence, tasks[sequence], parameters, binding_sha) for sequence in selected]
    started = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, receipt in enumerate(pool.map(ExportSequence, work), 1):
            print(json.dumps({'completed': index, 'total': len(work), **receipt}, ensure_ascii=False), flush=True)
    if args.sequence or args.limit_sequences:
        print(json.dumps({'b_complete': False, 'b_trial_complete': True, 'output_dir': str(output_dir),
                          'elapsed_seconds': time.time() - started}), flush=True)
        return
    # 原 infos 的非 objects 字段逐值保留；原 objects 与序列检测源一致后才追加补框。
    by_token = {}
    receipts = []
    for sequence in tasks:
        path = output_dir / 'sequences' / f'{sequence}.pkl'
        with path.open('rb') as file:
            payload = pickle.load(file)
        by_token.update({frame['token']: frame for frame in payload['frames']})
        receipts.append(json.loads(path.with_suffix('.json').read_text()))
    info_files = {}
    for split, infos in source_infos.items():
        for info in infos:
            generated_objects = by_token[info['token']]['objects']
            original_count = len(info['objects'])
            if not Equal(info['objects'], generated_objects[:original_count]):
                raise AssertionError(f'原始 infos 与序列 objects 不一致：{info["token"]}')
            info['objects'] = info['objects'] + generated_objects[original_count:]
        path = output_dir / f'infos_{split}.pkl'
        Save(path, infos, b_pickle=True)
        info_files[split] = {'path': str(path.relative_to(data_root)), 'sha256': SHA(path),
                             'target_count': len(infos), 'token_sha256': TokenSHA([item['token'] for item in infos]),
                             'sequence_count': len(split_sequences[split])}
    manifest = {'b_complete': True, 'binding_sha256': binding_sha, 'binding': binding,
                'info_files': info_files, 'sequence_receipts': receipts,
                'supplement_count': sum(item['supplement_count'] for item in receipts),
                'supplement_zero_top_points': sum(item['supplement_zero_top_points'] for item in receipts),
                'elapsed_seconds_this_invocation': time.time() - started,
                'completed_unix_time': time.time(), 'free_bytes_after': shutil.disk_usage(data_root).free}
    Save(output_dir / 'manifest.json', manifest)
    print(json.dumps({'b_complete': True, 'manifest': str(output_dir / 'manifest.json'),
                      'supplement_count': manifest['supplement_count']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    Main()
