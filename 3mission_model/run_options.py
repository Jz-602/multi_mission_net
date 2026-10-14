"""训练入口共用的运行参数与可复现配置记录。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import yaml


HASH_CHUNK_SIZE = 8 * 1024 * 1024


def HashFile(filename):
    file_hash = hashlib.sha256()
    with open(filename, 'rb') as stream:
        while True:
            chunk = stream.read(HASH_CHUNK_SIZE)
            if not chunk:
                break
            file_hash.update(chunk)
    return file_hash.hexdigest()


def DescribeFile(filename):
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Reproducibility input does not exist: {path}')
    return {
        'path': str(path),
        'size_bytes': path.stat().st_size,
        'sha256': HashFile(path),
    }


def ResolveConfigArtifact(value, config_filename):
    artifact_path = Path(os.path.expandvars(str(value))).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = Path(config_filename).expanduser().resolve().parent / artifact_path
    return artifact_path.resolve()


def CopyFileAtomic(source_filename, destination_filename):
    source_path = Path(source_filename).resolve()
    destination_path = Path(destination_filename).resolve()
    if source_path == destination_path:
        return
    temporary_path = destination_path.with_name(destination_path.name + '.tmp')
    shutil.copyfile(source_path, temporary_path)
    os.replace(temporary_path, destination_path)


def WriteJsonAtomic(filename, value):
    path = Path(filename)
    temporary_path = path.with_name(path.name + '.tmp')
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    os.replace(temporary_path, path)


def LoadLabelManifest(filename):
    with open(filename, 'r', encoding='utf-8') as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or not isinstance(manifest.get('splits'), dict):
        raise TypeError(f'Label manifest has no split mapping: {filename}')
    return manifest


def CheckCompletedDetectionManifest(cfg, config_filename):
    """新检测版本单独绑定infos内容；原地形/OCC清单仍只证明空间标签。"""
    b_completed = any(data and data.get('b_use_completed_detection_labels', False)
                      for data in cfg['data'].values())
    if not b_completed:
        return None
    filename = cfg.get('completed_detection_manifest')
    if not filename:
        raise ValueError('Completed detection labels require an explicit content manifest')
    manifest_path = ResolveConfigArtifact(filename, config_filename)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('b_complete') is not True:
        raise ValueError('Completed detection export has not passed full generation')
    data_root = Path(os.environ['WAYMO_DATA_ROOT']).resolve()
    infos = {}
    for split, export_split in (('train', 'training'), ('val', 'validation')):
        data = cfg['data'].get(f'{split}_dataset')
        if not data:
            continue
        if data.get('b_use_completed_detection_labels') is not True:
            raise ValueError('Both train and validation must use completed detection labels')
        info_path = Path(os.path.expandvars(str(data['info_path']))).expanduser()
        if not info_path.is_absolute():
            info_path = data_root / info_path
        info = DescribeFile(info_path)
        expected = manifest['info_files'][export_split]
        if (info_path.resolve() != (data_root / expected['path']).resolve()
                or info['sha256'] != expected['sha256']):
            raise ValueError(f'Completed detection {split} infos path/content changed')
        infos[split] = info
    return {'manifest': DescribeFile(manifest_path), 'infos': infos,
            'binding_sha256': manifest['binding_sha256']}


def ParseRunOptions(default_config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(default_config))
    parser.add_argument('--output', default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', default=None)
    group.add_argument('--init-checkpoint', default=None)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int, default=None,
                        help='Save and stop at this epoch without changing the configured scheduler budget')
    args, remaining = parser.parse_known_args()
    unknown = [v for v in remaining if v != '__internal_launch__' and not v.startswith('CUDA=') and not v.replace(',', '').isdigit()]
    if unknown:
        parser.error(f'Unknown options: {unknown}')
    if not os.environ.get('WAYMO_DATA_ROOT'):
        parser.error('Set WAYMO_DATA_ROOT to the original three-task dataset first.')
    return args


def CheckRunConfig(cfg):
    import torch
    import spconv.constants
    b_tf32 = bool(cfg.get('allow_tf32', False))
    torch.backends.cuda.matmul.allow_tf32 = b_tf32
    torch.backends.cudnn.allow_tf32 = b_tf32
    spconv.constants.SPCONV_ALLOW_TF32 = b_tf32

    if cfg.get('sync_batchnorm', False):
        raise ValueError('Three-task conditional sparse branches require sync_batchnorm: false')
    for name, data in cfg.data.items():
        if data and (data.get('augmentation') is not None or data.get('sampler') is not None or data.get('b_use_gt_sampling', False)):
            raise ValueError(f'{name}: all augmentation must remain disabled')
    if list(cfg.common_variables.voxel_size) != [.05, .05, .05]:
        raise ValueError('Current label contract requires 0.05m input and 0.2m output')


def LimitDataset(dataset, cfg, split):
    run = cfg.get('run', {})
    count = run.get(f'{split}_samples')
    offset = int(run.get(f'{split}_offset', 0))
    if count is not None:
        dataset.infos = dataset.infos[offset:offset + int(count)]
        if not dataset.infos:
            raise ValueError(f'Empty {split} subset')
        if hasattr(dataset, 'BuildAnchorList'):
            dataset.anchor_list = dataset.BuildAnchorList()
            if len(dataset.anchor_list) != len(dataset.infos):
                raise ValueError(f'{split}: some selected targets have no temporal anchor')


def SaveRunManifest(output_dir, cfg, args, train_dataset, val_dataset):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    plain_cfg = json.loads(json.dumps(cfg))
    (path / 'config.yaml').write_text(yaml.safe_dump(plain_cfg, sort_keys=False, allow_unicode=True))
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent, text=True).strip()
    manifest = {'git_commit': git_sha, 'arguments': vars(args), 'data_root': os.environ['WAYMO_DATA_ROOT'],
                'random_seed': cfg.get('random_seed', 0), 'augmentation': False, 'splits': {},
                'source_sha256': {str(file.relative_to(Path(__file__).parent)): hashlib.sha256(file.read_bytes()).hexdigest()
                                  for file in sorted(Path(__file__).parent.rglob('*'))
                                  if file.suffix in ('.py', '.yaml') and not any(part in ('log', 'build', '__pycache__') for part in file.parts)},
                'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain', '--', '3mission_model'], cwd=Path(__file__).parent.parent, text=True).strip())}
    manifest['checkpoint_provenance'] = {}
    for checkpoint_kind in ('init_checkpoint', 'resume'):
        checkpoint_filename = getattr(args, checkpoint_kind, None)
        if checkpoint_filename:
            manifest['checkpoint_provenance'][checkpoint_kind] = DescribeFile(checkpoint_filename)

    manifest['label_provenance'] = {
        'source_version_uncertain': True,
        'source_version_note': (
            'Content hashes can identify existing labels, but current frame metadata does not '
            'record the terrain exporter commit.'
        ),
        'content_manifest': None,
        'audit_report': None,
    }
    label_manifest_data = None
    completed_detection = CheckCompletedDetectionManifest(cfg, args.config)
    if completed_detection is not None:
        copied_path = path / 'completed_detection_manifest.json'
        CopyFileAtomic(completed_detection['manifest']['path'], copied_path)
        completed_detection['manifest']['copied_path'] = str(copied_path.resolve())
        manifest['label_provenance']['completed_detection'] = completed_detection
        manifest['input_time_feature'] = {
            'b_enabled': bool(cfg.data.train_dataset.get('b_use_time_feature', False)),
            'reference': 'final target frame timestamp', 'unit': 'seconds',
            'rule': '(frame_timestamp_micros - target_timestamp_micros) / 1e6',
            'model_in_channels': int(cfg.model.in_channels),
        }
    label_manifest_filename = cfg.get('label_manifest')
    if label_manifest_filename:
        label_manifest_path = ResolveConfigArtifact(label_manifest_filename, args.config)
        label_manifest_description = DescribeFile(label_manifest_path)
        label_manifest_data = LoadLabelManifest(label_manifest_path)
        label_manifest_description['b_partial_manifest'] = bool(
            label_manifest_data.get('b_partial_manifest', False)
        )
        copied_manifest_path = path / 'label_manifest.json'
        CopyFileAtomic(label_manifest_path, copied_manifest_path)
        label_manifest_description['copied_path'] = str(copied_manifest_path.resolve())
        manifest['label_provenance']['content_manifest'] = label_manifest_description
    label_audit_filename = cfg.get('label_audit_report')
    if label_audit_filename:
        label_audit_path = ResolveConfigArtifact(label_audit_filename, args.config)
        label_audit_description = DescribeFile(label_audit_path)
        copied_audit_path = path / 'label_audit_report.json'
        CopyFileAtomic(label_audit_path, copied_audit_path)
        label_audit_description['copied_path'] = str(copied_audit_path.resolve())
        manifest['label_provenance']['audit_report'] = label_audit_description
    for name, dataset in [('train', train_dataset), ('val', val_dataset)]:
        if dataset is None:
            continue
        tokens = [entry['token'] for entry in dataset.infos]
        WriteJsonAtomic(path / f'{name}_tokens.json', tokens)
        token_sha256 = hashlib.sha256('\n'.join(tokens).encode()).hexdigest()
        manifest['splits'][name] = {'count': len(tokens), 'sha256': token_sha256}
        if getattr(dataset, 'b_dynamic_epoch_sampling', False):
            if name != 'train':
                raise ValueError('Dynamic epoch sampling is only valid for the training pool')
            manifest['epoch_sampling'] = {'manifest_scope':'full_candidate_pool', 'candidate_pool_count':len(tokens),
                'candidate_pool_sha256':token_sha256, 'epoch_sample_fraction':dataset.sample_fraction, 'seed':dataset.sample_seed,
                'selection_rule':'torch.randperm(seed + zero_based_epoch), prefix ceil(pool*fraction), without replacement',
                'epoch_records':'epoch_sampling/epoch_NNN.json', 'b_train_tokens_are_candidate_pool':True}
        if label_manifest_data is not None:
            label_split = label_manifest_data['splits'].get(name)
            if not isinstance(label_split, dict):
                raise ValueError(f'Label manifest does not contain the configured {name} split')
            if label_split.get('target_count') != len(tokens) or label_split.get('token_sha256') != token_sha256:
                raise ValueError(
                    f'Label manifest {name} targets do not match this run: '
                    f"manifest count/hash={label_split.get('target_count')}/{label_split.get('token_sha256')}, "
                    f'run count/hash={len(tokens)}/{token_sha256}'
                )
    WriteJsonAtomic(path / 'manifest.json', manifest)
    with (path / 'run_history.jsonl').open('a') as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False) + '\n')
