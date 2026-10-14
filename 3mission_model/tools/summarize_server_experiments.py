"""只读汇总一个run-root的真实日志；进度图和统计不代替训练/标签验收。"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_server_experiments import STAGES, VALIDATION_EPOCHS


TASKS = ('det', 'terrain', 'occ')
NOTES = (
    'Observed logs only; this report is not a training or label acceptance audit.',
    'Final comparison requires epoch 48; latest partial results are not final or best-selected.',
    'Epochs are local to each stage; initialized stages include prior training. Compare total budgets using the initialization records.',
    'Root-on OCC raw loss includes root BCE; root-off removes BCE, so total OCC losses differ in definition.',
    'Training means use rank-0 logged updates; with DDP they are not an all-rank dataset mean. Validation uses full-set metrics.',
)


def ReadRecords(path, snapshots):
    if not path.exists():
        return
    with path.open('rb') as stream:
        remaining = os.fstat(stream.fileno()).st_size
        status = {'size_bytes': remaining, 'b_ignored_incomplete_tail': False}
        snapshots[str(path)] = status
        line_number = 0
        while remaining:
            line = stream.readline(remaining)
            if not line:
                raise RuntimeError(f'日志读取时被截断：{path}')
            remaining -= len(line)
            line_number += 1
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                if remaining == 0 and not line.endswith(b'\n'):
                    status['b_ignored_incomplete_tail'] = True
                    return
                raise ValueError(f'日志JSON损坏：{path}:{line_number}') from error


def SummarizeStage(output, world_size, snapshots):
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    counts = {split: manifest.get('splits', {}).get(split, {}).get('count') for split in ('train', 'val')}
    epoch_sampling = manifest.get('epoch_sampling', {})
    epoch_train_count = (math.ceil(counts['train'] * epoch_sampling.get('epoch_sample_fraction', 1.0))
                         if counts['train'] else None)
    expected_updates = math.ceil(epoch_train_count / world_size) if epoch_train_count and world_size else None
    epochs = {}
    for record in ReadRecords(output / 'train_metrics.jsonl', snapshots):
        # train_metrics epoch为从0起的训练轮索引；验证epoch为完成的1起轮次。
        epoch = int(record['epoch']) + 1
        entry = epochs.setdefault(epoch, {'epoch': epoch, 'updates': 0, 'iterations': set(),
                                        'loss_sum': dict.fromkeys(TASKS, 0.0)})
        losses = {task['task'][0]: task['loss'] for task in record['tasks']}
        for task in TASKS:
            if not math.isfinite(losses[task]):
                raise ValueError(f'非有限原始loss：{output}: epoch={epoch}, task={task}')
            entry['loss_sum'][task] += losses[task]
        entry['updates'] += 1
        entry['iterations'].add(record['iter'])
    training = []
    for epoch, entry in sorted(epochs.items()):
        iterations = entry.pop('iterations')
        entry['first_iter'], entry['last_iter'] = min(iterations), max(iterations)
        entry['unique_updates'] = len(iterations)
        entry['expected_updates'] = expected_updates
        entry['b_iterations_contiguous'] = len(iterations) == entry['last_iter'] - entry['first_iter'] + 1
        entry['b_full_epoch_observed'] = (expected_updates is not None and entry['updates'] == expected_updates
                                          and len(iterations) == expected_updates and entry['b_iterations_contiguous'])
        entry['raw_loss_mean'] = {task: value / entry['updates'] for task, value in entry.pop('loss_sum').items()}
        training.append(entry)
    validation = list(ReadRecords(output / 'validation_metrics.jsonl', snapshots))
    validation_epochs = [record['epoch'] for record in validation]
    if len(set(validation_epochs)) != len(validation_epochs):
        raise ValueError(f'验证epoch重复，禁止覆盖：{output}: {validation_epochs}')
    validation.sort(key=lambda record: record['epoch'])
    final = next((record for record in validation if record['epoch'] == 48), None)
    return {'b_directory_present': output.is_dir(), 'data_counts': counts,
            'epoch_train_count': epoch_train_count, 'epoch_sampling': epoch_sampling,
            'expected_updates_per_epoch': expected_updates, 'training_epochs': training,
            'validation_history': validation,
            'validation_coverage': [
                {'epoch': record['epoch'],
                 'frame_counts': {task: record['metrics'].get(task, {}).get('frame_count') for task in ('terrain', 'occ')},
                 'b_full_count_observed': counts['val'] is not None and all(
                     record['metrics'].get(task, {}).get('frame_count') == counts['val'] for task in ('terrain', 'occ'))}
                for record in validation],
            'missing_validation_epochs': sorted(set(VALIDATION_EPOCHS) - set(validation_epochs)),
            'b_all_48_training_epochs_observed': [entry['epoch'] for entry in training if entry['b_full_epoch_observed']] == list(range(1, 49)),
            'epoch48_final': final}


def PlotReports(report, output):
    plots = (
        ('validation_curves.png', 'Validation (observed epochs)', (
            ('det', 'map_l2', 'Detection L2 mAP (%)', 100), ('det', 'maph_l2', 'Detection L2 mAPH (%)', 100),
            ('terrain', 'mae_m', 'Terrain weighted MAE (m)', 1), ('terrain', 'rmse_m', 'Terrain weighted RMSE (m)', 1),
            ('occ', 'occupancy_iou', 'OCC occupancy IoU (%)', 100), ('occ', 'semantic_miou', 'OCC semantic mIoU (%)', 100))),
        ('root_curves.png', 'Root retention and reachable GT (validation)', (
            ('occ', 'occupied_root_recall', 'Occupied root recall (%)', 100),
            ('occ', 'deletion_fraction', 'Root deletion (%)', 100),
            ('occ', 'deletion_precision', 'Deletion precision (%)', 100),
            ('occ', 'candidate_gt_coverage', 'Candidate GT coverage (%)', 100))),
        ('raw_loss_curves.png', 'Raw task loss means of rank-0 updates; x marks incomplete epochs', (
            ('det', '', 'Detection raw loss', 1), ('terrain', '', 'Terrain raw loss', 1), ('occ', '', 'OCC raw loss', 1))),
    )
    b_partial = any(stage['b_scheduled'] and (stage['epoch48_final'] is None or not stage['b_all_48_training_epochs_observed']
                    or stage['missing_validation_epochs'] or any(not entry['b_full_count_observed'] for entry in stage['validation_coverage']))
                    for stage in report['stages'].values())
    for filename, title, panels in plots:
        columns = 2 if len(panels) != 3 else 3
        rows = math.ceil(len(panels) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 3.6 * rows), squeeze=False)
        for axis, (task, metric, label, scale) in zip(axes.flat, panels):
            for stage_name, stage in report['stages'].items():
                if metric:
                    points = [(record['epoch'], record['metrics'].get(task, {}).get(metric)) for record in stage['validation_history']]
                    points = [(epoch, value * scale) for epoch, value in points if value is not None]
                else:
                    points = [(entry['epoch'], entry['raw_loss_mean'][task]) for entry in stage['training_epochs']]
                if not points:
                    continue
                line, = axis.plot(*zip(*points), marker='o', markersize=3, label=stage_name)
                if not metric:
                    partial = [(entry['epoch'], entry['raw_loss_mean'][task]) for entry in stage['training_epochs'] if not entry['b_full_epoch_observed']]
                    if partial:
                        axis.scatter(*zip(*partial), marker='x', s=85, color=line.get_color(), zorder=3)
            axis.set(title=label, xlabel='Stage epoch (final = 48)')
            axis.axvline(48, color='gray', linestyle=':', linewidth=1)
            axis.set_xlim(0, 49)
            axis.grid(alpha=0.25)
            if axis.lines and axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(panels):]:
            axis.set_visible(False)
        figure.suptitle(f'{Path(report["run_root"]).name}: {"PARTIAL" if b_partial else "epoch48 observed"}\n{title}')
        figure.text(0.01, 0.015, 'Observed logs; final=48 stage epochs. Initialized stages include prior training. OCC loss includes BCE only with root filtering.', fontsize=8)
        figure.tight_layout(rect=(0, 0.045, 1, 0.93))
        figure.savefig(output / filename, dpi=150)
        plt.close(figure)


def FlattenFields(record, prefix=''):
    fields = {}
    for key, value in record.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            fields.update(FlattenFields(value, name))
        else:
            fields[name] = json.dumps(value) if isinstance(value, (list, tuple)) else value
    return fields


def ExportCSVs(report, output):
    # CSV和JSON/PNG共用已读快照；比例保持0..1，地形误差单位为m。
    for filename, history_key in (('validation_metrics.csv', 'validation_history'),
                                  ('raw_loss_by_epoch.csv', 'training_epochs')):
        rows = []
        for stage_name, stage in report['stages'].items():
            coverage = {entry['epoch']: entry for entry in stage['validation_coverage']}
            for entry in stage[history_key]:
                b_full = (coverage[entry['epoch']]['b_full_count_observed'] if history_key == 'validation_history'
                          else entry['b_full_epoch_observed'])
                row = {'stage': stage_name, 'epoch': entry['epoch'],
                       'world_size': report['execution_settings'].get('world_size'),
                       'dataset_train_count': stage['data_counts']['train'],
                       'dataset_val_count': stage['data_counts']['val'],
                       'b_full_observed': b_full, 'b_partial': not b_full}
                if history_key == 'training_epochs':
                    row['raw_loss_scope'] = 'rank0_logged_updates_mean'
                row.update(FlattenFields(entry))
                rows.append(row)
        columns = list(rows[0]) if rows else ['stage', 'epoch', 'world_size', 'b_full_observed', 'b_partial']
        columns += sorted({key for row in rows for key in row} - set(columns))
        with (output / filename).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise NotADirectoryError(run_root)
    settings_path = run_root / 'execution_settings.json'
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    snapshots = {}
    report = {'run_root': str(run_root), 'created_utc': datetime.now(timezone.utc).isoformat(),
              'notes': NOTES, 'execution_settings': settings, 'source_snapshots': snapshots, 'stages': {}}
    for stage_name in settings.get('stages', [item[0] for item in STAGES]):
        stage = SummarizeStage(run_root / stage_name, settings.get('world_size'), snapshots)
        stage['b_scheduled'] = stage_name in settings.get('stages', [item[0] for item in STAGES])
        report['stages'][stage_name] = stage
    output = run_root / 'reports'
    output.mkdir(exist_ok=True)
    temporary = output / 'summary.json.tmp'
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(output / 'summary.json')
    PlotReports(report, output)
    ExportCSVs(report, output)
    print(output)


if __name__ == '__main__':
    Main()
