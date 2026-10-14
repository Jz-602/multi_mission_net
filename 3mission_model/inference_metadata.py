"""Prepare/verify input-only inference indexes without reading detection GT infos.

Pose is [4,4], target LiDAR/vehicle -> world; timestamps are original microseconds.
sample_idx is the original full-sequence frame order, not a supervised-row index.
Only the audited point-only pickle and ordered target-token JSON are inputs.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from run_options import DescribeFile, HashFile, WriteJsonAtomic
from thesis_v1_gate import EXPECTED_TARGETS, EXPECTED_TOKEN_SHA256

INPUT_FIELDS = frozenset(('token', 'sequence_name', 'frame_timestamp_micros', 'sample_idx', 'pose'))
SOURCE_FIELDS = INPUT_FIELDS | {'semantic_path'}
PROTOCOL = 'waymo_input_only_validation_v1'


def TokenHash(tokens):
    return hashlib.sha256('\n'.join(tokens).encode()).hexdigest()


def CheckRows(rows, b_source=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Input metadata must be a nonempty list')
    tokens = set()
    sequence_orders = set()
    allowed = SOURCE_FIELDS if b_source else INPUT_FIELDS
    for row in rows:
        if not isinstance(row, dict) or not INPUT_FIELDS <= row.keys() or not row.keys() <= allowed:
            raise ValueError('Metadata has missing input fields or non-input/GT fields; no GT stripping is allowed')
        if 'semantic_path' in row and row['semantic_path'] is not None and not isinstance(row['semantic_path'], str):
            raise ValueError('semantic_path may only be an unused filename reference, not label data')
        token, sequence, stamp, order = (row[key] for key in ('token', 'sequence_name', 'frame_timestamp_micros', 'sample_idx'))
        if (not isinstance(token, str) or not isinstance(sequence, str)
                or isinstance(stamp, bool) or not isinstance(stamp, (int, np.integer))
                or isinstance(order, bool) or not isinstance(order, (int, np.integer))
                or order < 0 or token != f'{sequence}-{stamp}'):
            raise ValueError('Invalid token/sequence/timestamp/original frame order')
        if token in tokens or (sequence, int(order)) in sequence_orders:
            raise ValueError('Duplicate input token or sequence frame order')
        pose = np.asarray(row['pose'])
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError('Pose must be finite [4,4] vehicle-to-world')
        tokens.add(token)
        sequence_orders.add((sequence, int(order)))
    return tokens


def CheckTargetTokens(tokens, expected_count, expected_hash):
    if (not isinstance(tokens, list) or len(tokens) != expected_count
            or any(not isinstance(token, str) for token in tokens)
            or len(set(tokens)) != expected_count or TokenHash(tokens) != expected_hash):
        raise ValueError('Ordered unique target tokens differ from the frozen validation list')


def CheckTargetLinks(targets, history):
    lookup = {row['token']: row for row in history}
    for target in targets:
        source = lookup.get(target['token'])
        if source is None:
            raise ValueError('Target missing from point-only history')
        for key in INPUT_FIELDS:
            if not np.array_equal(target[key], source[key]):
                raise ValueError(f'Target/history input mismatch: {key}')


def PrepareInferenceMetadata(point_only_path, target_tokens_path, output,
                             expected_count=EXPECTED_TARGETS, expected_hash=EXPECTED_TOKEN_SHA256):
    point_only_path, target_tokens_path, output = map(Path, (point_only_path, target_tokens_path, output))
    if not point_only_path.name.endswith('_point_only.pkl'):
        raise ValueError('Source must be an explicitly named audited *_point_only.pkl, never a detection GT infos file')
    tokens = json.loads(target_tokens_path.read_text())
    CheckTargetTokens(tokens, expected_count, expected_hash)
    with point_only_path.open('rb') as stream:
        source_rows = pickle.load(stream)
    CheckRows(source_rows, b_source=True)  # Reject annotations, do not strip GT.
    history = [{key: row[key] for key in sorted(INPUT_FIELDS)} for row in source_rows]
    lookup = {row['token']: row for row in history}
    missing = set(tokens) - lookup.keys()
    if missing:
        raise ValueError(f'{len(missing)} targets absent from point-only metadata')
    targets = [dict(lookup[token]) for token in tokens]
    CheckRows(targets)
    CheckTargetLinks(targets, history)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (('target_infos.pkl', targets), ('history_infos.pkl', history)):
        with (output / name).open('xb') as stream:
            pickle.dump(rows, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        'schema_version': 1, 'protocol': PROTOCOL, 'split': 'val',
        'target_count': len(targets), 'history_count': len(history),
        'sequence_count': len({row['sequence_name'] for row in targets}),
        'target_token_sha256': TokenHash(tokens), 'history_token_sha256': TokenHash([row['token'] for row in history]),
        'source_point_only': DescribeFile(point_only_path), 'source_target_tokens': DescribeFile(target_tokens_path),
        'files': {name: {'path': name, 'sha256': HashFile(output / name), 'size_bytes': (output / name).stat().st_size}
                  for name in ('target_infos.pkl', 'history_infos.pkl')},
        'fields': sorted(INPUT_FIELDS), 'pose_definition': '[4,4] vehicle/LiDAR -> world, unchanged values/dtype',
        'sample_idx_definition': 'original full-sequence frame order copied from point-only source; supervised-row renumbering is not used',
        'b_gt_infos_read': False, 'b_gt_label_files_read': False,
        'removed_source_metadata_reference': 'semantic_path string only; semantic file content never read',
        'scope': 'Input-only metadata derivation, no points or label arrays loaded; output paths are relative to this manifest directory.',
    }
    WriteJsonAtomic(output / 'manifest.json', manifest)
    return manifest


def VerifyInferenceMetadata(manifest_path, expected_count=EXPECTED_TARGETS, expected_hash=EXPECTED_TOKEN_SHA256):
    """Open only generated input metadata; source provenance paths are never opened."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('protocol') != PROTOCOL or manifest.get('split') != 'val'
            or manifest.get('b_gt_infos_read') is not False or manifest.get('b_gt_label_files_read') is not False
            or manifest.get('fields') != sorted(INPUT_FIELDS)):
        raise ValueError('Invalid input-only metadata protocol')
    rows_by_name, paths = {}, {}
    for name in ('target_infos.pkl', 'history_infos.pkl'):
        entry = manifest['files'][name]
        if entry.get('path') != name:
            raise ValueError('Metadata file must be the named sibling of its manifest')
        path = manifest_path.parent / name
        if path.stat().st_size != entry['size_bytes'] or HashFile(path) != entry['sha256']:
            raise ValueError(f'Input-only metadata content binding mismatch: {name}')
        with path.open('rb') as stream:
            rows_by_name[name] = pickle.load(stream)
        CheckRows(rows_by_name[name])
        paths[name] = str(path)
    targets, history = rows_by_name['target_infos.pkl'], rows_by_name['history_infos.pkl']
    tokens = [row['token'] for row in targets]
    CheckTargetTokens(tokens, expected_count, expected_hash)
    if (manifest['target_count'] != len(targets) or manifest['history_count'] != len(history)
            or manifest['target_token_sha256'] != TokenHash(tokens)
            or manifest['history_token_sha256'] != TokenHash([row['token'] for row in history])
            or manifest['sequence_count'] != len({row['sequence_name'] for row in targets})):
        raise ValueError('Metadata manifest counts/hash mismatch')
    CheckTargetLinks(targets, history)
    return {'manifest': DescribeFile(manifest_path), 'protocol': PROTOCOL,
            'target_count': len(targets), 'history_count': len(history),
            'target_token_sha256': TokenHash(tokens), 'info_path': paths['target_infos.pkl'],
            'full_info_path': paths['history_infos.pkl'], 'b_strict_input_only_metadata': True}


def ConfigureInferenceMetadata(data_cfg, manifest_path, split):
    """Mutate only benchmark's private data config; never change training config."""
    if split != 'val':
        raise ValueError('This prepared index binds the complete fixed validation split')
    if list(data_cfg.loading_pipelines) != ['FH_LoadPointCloud']:
        raise ValueError('Strict input-only inference allows only FH_LoadPointCloud loading pipeline')
    if data_cfg.get('prepare_label') or data_cfg.get('augmentation') or data_cfg.get('sampler') or data_cfg.get('b_use_gt_sampling', False):
        raise ValueError('Labels, augmentation and GT sampling must be disabled before input-only inference')
    binding = VerifyInferenceMetadata(manifest_path)
    data_cfg.info_path = binding['info_path']
    data_cfg.full_info_path = binding['full_info_path']
    data_cfg.b_use_completed_detection_labels = False
    data_cfg.sample_fraction = 1.0
    data_cfg.b_dynamic_epoch_sampling = False
    return binding
