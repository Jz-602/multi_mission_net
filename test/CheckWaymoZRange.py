from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml


def LoadConfig(config_path: Path) -> dict:
    with config_path.open('r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def LoadInfos(info_path: Path) -> list[dict]:
    with info_path.open('rb') as file:
        return pickle.load(file)


def BuildPointCloudPath(root_path: Path, info: dict) -> Path:
    timestamp_ms = str(info['frame_timestamp_micros'])[:-3]
    return root_path / info['sequence_name'] / 'output' / 'pcd' / f'{timestamp_ms}.pcd'


def LoadPointsXYZ(point_cloud_path: Path) -> np.ndarray:
    ## 读取单帧点云。
    ## 输出 points_xyz: [N, 3]，每行是 [x, y, z]，坐标系为 top-lidar / LiDAR 车体坐标系。
    point_cloud = o3d.io.read_point_cloud(str(point_cloud_path))
    points_xyz = np.asarray(point_cloud.points, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(
            f'points_xyz must be [N, 3] in top-lidar frame, but got {points_xyz.shape} from {point_cloud_path}.'
        )
    return points_xyz


def CollectZStatistics(config_path: Path, split: str, num_frames: int) -> dict:
    config = LoadConfig(config_path)
    root_path = Path(config['common_variables']['root_path'])
    dataset_key = f'{split}_dataset'
    info_filename = config['data'][dataset_key]['info_path']
    info_path = root_path / info_filename
    infos = LoadInfos(info_path)

    selected_infos = infos[:num_frames]
    frame_stats = []
    global_z_values = []
    missing_paths = []

    for frame_index, info in enumerate(selected_infos):
        point_cloud_path = BuildPointCloudPath(root_path, info)
        if not point_cloud_path.exists():
            missing_paths.append(str(point_cloud_path))
            continue

        points_xyz = LoadPointsXYZ(point_cloud_path)
        if points_xyz.shape[0] == 0:
            frame_stats.append({
                'frame_index': frame_index,
                'token': info['token'],
                'num_points': 0,
                'z_min': None,
                'z_max': None,
                'z_mean': None,
                'z_std': None,
            })
            continue

        z_values = points_xyz[:, 2]
        global_z_values.append(z_values)
        frame_stats.append({
            'frame_index': frame_index,
            'token': info['token'],
            'num_points': int(points_xyz.shape[0]),
            'z_min': float(np.min(z_values)),
            'z_max': float(np.max(z_values)),
            'z_mean': float(np.mean(z_values)),
            'z_std': float(np.std(z_values)),
        })

    if not global_z_values:
        raise RuntimeError('No valid point clouds were loaded from the selected frames.')

    merged_z_values = np.concatenate(global_z_values, axis=0)
    k_z_bin_size_m = 1.0
    z_bin_start = float(np.floor(np.min(merged_z_values) / k_z_bin_size_m) * k_z_bin_size_m)
    z_bin_end = float(np.ceil(np.max(merged_z_values) / k_z_bin_size_m) * k_z_bin_size_m)
    z_bin_edges = np.arange(z_bin_start, z_bin_end + k_z_bin_size_m, k_z_bin_size_m, dtype=np.float32)
    z_bin_counts, _ = np.histogram(merged_z_values, bins=z_bin_edges)

    z_bin_statistics = []
    total_points = int(merged_z_values.shape[0])
    for bin_index, point_count in enumerate(z_bin_counts):
        z_left = float(z_bin_edges[bin_index])
        z_right = float(z_bin_edges[bin_index + 1])
        point_ratio = float(point_count / total_points)
        z_bin_statistics.append({
            'z_left': z_left,
            'z_right': z_right,
            'num_points': int(point_count),
            'point_ratio': point_ratio,
            'point_percentage': point_ratio * 100.0,
        })

    return {
        'split': split,
        'root_path': str(root_path),
        'info_path': str(info_path),
        'requested_frames': num_frames,
        'loaded_frames': len(frame_stats),
        'missing_frames': len(missing_paths),
        'total_points': total_points,
        'z_min': float(np.min(merged_z_values)),
        'z_max': float(np.max(merged_z_values)),
        'z_mean': float(np.mean(merged_z_values)),
        'z_std': float(np.std(merged_z_values)),
        'z_p01': float(np.percentile(merged_z_values, 1.0)),
        'z_p05': float(np.percentile(merged_z_values, 5.0)),
        'z_p50': float(np.percentile(merged_z_values, 50.0)),
        'z_p95': float(np.percentile(merged_z_values, 95.0)),
        'z_p99': float(np.percentile(merged_z_values, 99.0)),
        'z_bin_size_m': k_z_bin_size_m,
        'z_bin_statistics': z_bin_statistics,
        'frame_stats': frame_stats,
        'missing_paths': missing_paths,
    }


def PrintStatistics(statistics: dict) -> None:
    k_preview_frames = 5
    print('=== Waymo Point Cloud Z Range Summary ===')
    print(f"split: {statistics['split']}")
    print(f"root_path: {statistics['root_path']}")
    print(f"info_path: {statistics['info_path']}")
    print(f"requested_frames: {statistics['requested_frames']}")
    print(f"loaded_frames: {statistics['loaded_frames']}")
    print(f"missing_frames: {statistics['missing_frames']}")
    print(f"total_points: {statistics['total_points']}")
    print(f"global_z_min: {statistics['z_min']:.6f}")
    print(f"global_z_max: {statistics['z_max']:.6f}")
    print(f"global_z_mean: {statistics['z_mean']:.6f}")
    print(f"global_z_std: {statistics['z_std']:.6f}")
    print(f"global_z_p01: {statistics['z_p01']:.6f}")
    print(f"global_z_p05: {statistics['z_p05']:.6f}")
    print(f"global_z_p50: {statistics['z_p50']:.6f}")
    print(f"global_z_p95: {statistics['z_p95']:.6f}")
    print(f"global_z_p99: {statistics['z_p99']:.6f}")
    print('')
    print(f'=== Z Bin Distribution ({statistics["z_bin_size_m"]:.1f}m per bin) ===')
    for z_bin_stat in statistics['z_bin_statistics']:
        print(
            'z_range=[{z_left:.1f}, {z_right:.1f}) num_points={num_points} '
            'point_percentage={point_percentage:.4f}%'.format(**z_bin_stat)
        )
    print('')
    print(f'=== First {min(k_preview_frames, len(statistics["frame_stats"]))} Frame Stats ===')
    for frame_stat in statistics['frame_stats'][:k_preview_frames]:
        print(
            'frame_index={frame_index} token={token} num_points={num_points} '
            'z_min={z_min:.6f} z_max={z_max:.6f} z_mean={z_mean:.6f} z_std={z_std:.6f}'.format(
                **frame_stat
            )
        )

    if statistics['missing_paths']:
        print('')
        print('=== Missing Point Clouds ===')
        for missing_path in statistics['missing_paths'][:k_preview_frames]:
            print(missing_path)


def ParseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Check z-axis statistics of Waymo point clouds from the current dataset config.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('/media/estella/Osmanthus/aa_code/mulit_mission/ODSS_project/LYF_ODSS_config.yaml'),
        help='YAML config path. The script reads common_variables.root_path and data.<split>_dataset.info_path.',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split key used to pick data.<split>_dataset.info_path.',
    )
    parser.add_argument(
        '--num_frames',
        type=int,
        default=2000,
        help='Number of frames to scan from the beginning of the selected info file.',
    )
    return parser.parse_args()


def Main() -> None:
    args = ParseArgs()
    statistics = CollectZStatistics(
        config_path=args.config,
        split=args.split,
        num_frames=args.num_frames,
    )
    PrintStatistics(statistics)


if __name__ == '__main__':
    Main()
