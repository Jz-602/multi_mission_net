"""执行动态半数目标八卡正式实验；旧固定抽样仅供显式历史复现。"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
PHASES = (5, 25, 48)
VALIDATION_EPOCHS = (*range(5, 40, 5), *range(40, 49))
STAGES = (
    ('single', False, None),
    ('temporal', True, 'single'),
    ('single_continuation', False, 'single'),
    ('single_no_root', False, None),
    ('temporal_no_root', True, 'single_no_root'),
)
XYZT_COMPLETED_STAGES = (*STAGES[:3], ('temporal_scratch', True, None))
FIGURES = (('train', 10), ('val', 0), ('val', 100), ('val', 300))

SCOPES = {
    'fixed86': {'stages': STAGES, 'train_count': 984, 'val_count': 520,
                'config_prefix': 'server_', 'output_name': 'server_runs_20260930'},
    'full50': {'stages': STAGES[:2], 'train_count': 11844, 'val_count': 5973,
               'config_prefix': 'server_full50_', 'output_name': 'server_runs_full50'},
    'dynamic50': {'stages': STAGES, 'train_count': 23688, 'epoch_train_count': 11844,
                  'val_count': 5973, 'config_prefix': 'server_dynamic50_',
                  'output_name': 'server_runs_dynamic50',
                  # 固定近距离骑行者样本，保留类别失败证据；原四个样本照常输出。
                  'figures': (*FIGURES, ('val', 2720))},
    'xyzt_completed': {'stages': XYZT_COMPLETED_STAGES, 'train_count': 23688,
                       'epoch_train_count': 11844, 'val_count': 5973,
                       'config_prefix': 'server_xyzt_completed_',
                       'output_name': 'server_runs_xyzt_completed_20261007',
                       'figures': (*FIGURES, ('val', 2720))},
}


def CheckFull50Labels(scope='full50'):
    """核对已验收actual记录、目标选择及当前文件大小；不重算标签内容SHA。"""
    # actual内容清单必须在主agent完成全部派生标签验收后生成，selection不能替代。
    from build_label_manifest import BuildTargetFiles, HashTextLines, LoadConfig, LoadSelectedInfos
    config_path = PROJECT_ROOT / 'configs' / f'{SCOPES[scope]["config_prefix"]}single.yaml'
    config = LoadConfig(config_path)
    if scope == 'xyzt_completed':
        sys.path.insert(0, str(PROJECT_ROOT))
        from run_options import CheckCompletedDetectionManifest
        CheckCompletedDetectionManifest(config, config_path)
        if config['model']['in_channels'] != 4 or not all(
                config['data'][f'{split}_dataset'].get('b_use_time_feature') is True
                for split in ('train', 'val')):
            raise RuntimeError('XYZT实验必须使用四通道和目标帧相对时间输入')
    if scope in ('dynamic50', 'xyzt_completed'):
        train_config = config['data']['train_dataset']
        val_config = config['data']['val_dataset']
        if (train_config.get('b_dynamic_epoch_sampling') is not True or
                train_config.get('sample_fraction') != 0.5 or train_config.get('sample_seed') != 0 or
                val_config.get('sample_fraction') != 1.0 or val_config.get('b_dynamic_epoch_sampling', False)):
            raise RuntimeError('dynamic50必须保留完整候选、seed0逐轮50%抽样及固定全部验证目标')
    manifest_path = (config_path.parent / config['label_manifest']).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('b_partial_manifest') is not False:
        raise RuntimeError('full50需完整schema1 actual label manifest；未知/partial清单禁止启动')
    data_root = Path(os.path.expandvars(os.environ.get('WAYMO_DATA_ROOT',
                     config['common_variables']['root_path']))).expanduser().resolve()
    if Path(manifest['data_root']).resolve() != data_root:
        raise RuntimeError('full50 actual manifest数据根目录不一致')
    for split in ('train', 'val'):
        dataset_config = config['data'][f'{split}_dataset']
        infos, _, _, _, _, _ = LoadSelectedInfos(data_root, dataset_config, config.get('run', {}), split)
        entry = manifest['splits'][split]
        targets = entry['targets']
        tokens = [str(info['token']) for info in infos]
        if (len(tokens) != SCOPES[scope][f'{split}_count'] or
                entry['target_count'] != len(tokens) or
                [target['token'] for target in targets] != tokens or
                entry['token_sha256'] != HashTextLines(tokens)):
            raise RuntimeError(f'full50 actual manifest未覆盖固定{split}目标')
        aggregate_hash = hashlib.sha256()
        total_bytes = 0
        for info, target in zip(infos, targets):
            sequence_name, timestamp_ms, files = BuildTargetFiles(data_root, dataset_config, info)
            if target['sequence_name'] != sequence_name or target['timestamp_ms'] != timestamp_ms:
                raise RuntimeError('full50 actual manifest目标序列/时间戳不一致')
            if set(target['files']) != {name for name, _ in files}:
                raise RuntimeError('full50 actual manifest缺少六项派生标签内容记录')
            for name, filename in files:
                record = target['files'][name]
                relative_path = filename.resolve().relative_to(data_root).as_posix()
                sha256 = record['sha256']
                if (record['path'] != relative_path or record['size_bytes'] <= 0 or
                        len(sha256) != 64 or any(value not in '0123456789abcdef' for value in sha256)):
                    raise RuntimeError('full50 actual manifest标签路径/大小/SHA记录无效')
                if not filename.is_file() or filename.stat().st_size != record['size_bytes']:
                    raise RuntimeError(f'full50 actual标签缺失或大小改变：{filename}')
                aggregate_hash.update(relative_path.encode('utf-8') + b'\0' + sha256.encode('ascii') + b'\n')
                total_bytes += record['size_bytes']
        if entry['label_content_sha256'] != aggregate_hash.hexdigest() or entry['total_label_bytes'] != total_bytes:
            raise RuntimeError('full50 actual manifest内容摘要不一致')


def Now():
    return datetime.now(timezone.utc).isoformat()


def EpochSteps(scope, world_size):
    count = SCOPES[scope].get('epoch_train_count', SCOPES[scope]['train_count'])
    return math.ceil(count / world_size)


def ReadRecords(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ValidationRecords(output, scope='fixed86'):
    records = ReadRecords(output / 'validation_metrics.jsonl')
    epochs = [record['epoch'] for record in records]
    if len(set(epochs)) != len(epochs) or set(epochs) - set(VALIDATION_EPOCHS):
        raise RuntimeError(f'验证epoch重复/超出固定计划，需主agent审计：{output}: {epochs}')
    if any(record['metrics'].get(task, {}).get('frame_count') != SCOPES[scope]['val_count']
           for record in records for task in ('terrain', 'occ')):
        raise RuntimeError(f'验证未覆盖完整{SCOPES[scope]["val_count"]}目标，需主agent审计：{output}')
    return records


def CheckpointMeta(path):
    import torch
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    meta = checkpoint['meta']
    epoch, iteration = meta['epoch'], meta['iter']
    if type(epoch) is not int or type(iteration) is not int or not 0 < epoch <= 48 or iteration <= 0:
        raise RuntimeError(f'需人工审计checkpoint meta: {path}: {meta}')
    return {'epoch': epoch, 'iter': iteration}


def TrainCommand(stage, b_temporal, output_root, world_size, extra, scope='fixed86'):
    command = [sys.executable]
    if world_size > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(world_size)]
    command += [str(PROJECT_ROOT / ('LYF_train_ODSS_multi.py' if b_temporal else 'LYF_train_ODSS.py')),
                '--config', str(PROJECT_ROOT / 'configs' / f'{SCOPES[scope]["config_prefix"]}{stage}.yaml'),
                '--output', str(output_root / stage)]
    return command + extra


def FigureCommand(stage, b_temporal, checkpoint, split, index, output, loss_log=None, scope='fixed86'):
    command = [sys.executable, str(PROJECT_ROOT / 'tools' / 'visualize_three_task.py'),
               '--config', str(PROJECT_ROOT / 'configs' / f'{SCOPES[scope]["config_prefix"]}{stage}.yaml'),
               '--checkpoint', str(checkpoint), '--split', split, '--index', str(index),
               '--output', str(output)]
    if b_temporal:
        command.append('--temporal')
    if loss_log is not None:
        command += ['--loss-log', str(loss_log)]
    return command


def AppendProgress(record):
    with (PROJECT_ROOT / 'SERVER_PROGRESS.md').open('a') as stream:
        stream.write('\n串行实验状态：' + json.dumps(record, ensure_ascii=False) + '\n')


def Run(command, stage, phase, action, output_root, gpu_ids, scope='fixed86'):
    # 复用生产JSONL写入；延迟导入让--help/--dry-run保持标准库入口。
    sys.path.insert(0, str(PROJECT_ROOT / 'trainer' / 'trainer'))
    from three_task_runtime import AppendJsonLine
    command_dir = output_root / 'commands' / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + f'_{stage}_{phase}_{action}')
    command_dir.mkdir(parents=True)
    output = output_root / stage
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpu_ids))
    record = dict(stage=stage, phase=phase, action=action, command=command,
                  gpu_ids=gpu_ids, started_at=Now(), log=str(command_dir / 'stdout.log'),
                  next_step='完成当前子命令后检查产物，继续当前阶段')
    if scope != 'fixed86':
        record['scope'] = scope
    AppendProgress(dict(record, status='running'))
    process = None
    exit_code = None
    try:
        with (command_dir / 'stdout.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            AppendJsonLine(str(output_root / 'commands.jsonl'), dict(record, event='start', pid=process.pid))
            exit_code = process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        if process is not None:
            exit_code = process.returncode
        # 每次入口原生保存run_history；这里只留本次manifest/完整config副本。
        for name in ('manifest.json', 'config.yaml'):
            if (output / name).exists():
                shutil.copyfile(output / name, command_dir / name)
        finished = dict(record, event='finish', pid=None if process is None else process.pid,
                        ended_at=Now(), exit_code=exit_code,
                        status='complete' if exit_code == 0 else 'failed')
        if exit_code != 0:
            finished['next_step'] = '主agent审计stdout和checkpoint/日志一致性后，重新调用同一脚本'
        AppendJsonLine(str(output_root / 'commands.jsonl'), finished)
        AppendProgress(finished)
    if exit_code != 0:
        raise RuntimeError(f'子命令失败，停止：{command_dir / "stdout.log"}')


def CheckManifest(output, output_root, scope='fixed86'):
    manifest = json.loads((output / 'manifest.json').read_text())
    if [manifest['splits'][split]['count'] for split in ('train', 'val')] != [SCOPES[scope]['train_count'], SCOPES[scope]['val_count']]:
        raise RuntimeError(f'需人工审计实际数据数量：{output}: {manifest["splits"]}')
    reference_path = output_root / 'single' / 'manifest.json'
    if reference_path.exists():
        reference = json.loads(reference_path.read_text())
        if reference['splits'] != manifest['splits']:
            raise RuntimeError(f'需人工审计实验token不一致：{output}')
        if scope == 'xyzt_completed':
            for key in ('infos', 'binding_sha256'):
                if (reference['label_provenance']['completed_detection'][key]
                        != manifest['label_provenance']['completed_detection'][key]):
                    raise RuntimeError('XYZT对照的补框标签内容不同')
    source_stage = next(item[2] for item in SCOPES[scope]['stages'] if item[0] == output.name)
    initial_runs = [run for run in ReadRecords(output / 'run_history.jsonl')
                    if run['arguments'].get('init_checkpoint')]
    if source_stage is None and initial_runs:
        raise RuntimeError(f'从头训练组含初始化权重记录，需主agent审计：{output}')
    if source_stage is not None:
        expected = output_root / source_stage / 'checkpoints' / 'epoch_48.pth'
        if not initial_runs or any(Path(run['arguments']['init_checkpoint']).resolve() != expected for run in initial_runs):
            raise RuntimeError(f'已有实验初始化路径不匹配，需主agent审计：{output}')
        if output.name == 'single_continuation':
            temporal_runs = [run for run in ReadRecords(output_root / 'temporal' / 'run_history.jsonl')
                             if run['arguments'].get('init_checkpoint')]
            if initial_runs[0]['checkpoint_provenance'] != temporal_runs[0]['checkpoint_provenance']:
                raise RuntimeError('时序和单帧继续训练初始化权重内容不同，需主agent审计')


def AuditStage(output, output_root, world_size, scope='fixed86'):
    latest = output / 'checkpoints' / 'latest.pth'
    train_records = ReadRecords(output / 'train_metrics.jsonl')
    validation_records = ValidationRecords(output, scope)
    if not latest.exists():
        if any((output / name).exists() for name in ('train_metrics.jsonl', 'validation_metrics.jsonl',
                                                    'validation_metrics.json')) or list((output / 'checkpoints').glob('*.pth')):
            raise RuntimeError(f'日志/权重存在但latest缺失，需主agent审计：{output}')
        return None
    meta = CheckpointMeta(latest)
    CheckManifest(output, output_root, scope)
    if meta['iter'] != meta['epoch'] * EpochSteps(scope, world_size):
        raise RuntimeError(f'checkpoint更新数与固定预算不一致，需主agent审计：{latest}: {meta}')
    iterations = [record['iter'] for record in train_records]
    if iterations != list(range(1, meta['iter'] + 1)):
        raise RuntimeError(f'训练日志与checkpoint不一致/重复，需主agent审计：{output}')
    if any(record['epoch'] > meta['epoch'] for record in validation_records):
        raise RuntimeError(f'验证历史超过checkpoint，需主agent审计：{output}')
    return meta


def CompletePhase(stage, b_temporal, phase, output_root, gpu_ids, scope='fixed86'):
    output = output_root / stage
    checkpoint = output / 'checkpoints' / f'epoch_{phase}.pth'
    meta = CheckpointMeta(checkpoint)
    if meta['epoch'] != phase or meta['iter'] != phase * EpochSteps(scope, len(gpu_ids)):
        raise RuntimeError(f'阶段checkpoint epoch错误：{checkpoint}: {meta}')
    completed_epochs = {record['epoch'] for record in ValidationRecords(output, scope)}
    for epoch in VALIDATION_EPOCHS:
        if epoch > phase or epoch in completed_epochs:
            continue
        evaluation_checkpoint = output / 'checkpoints' / f'epoch_{epoch}.pth'
        evaluation_meta = CheckpointMeta(evaluation_checkpoint)
        if evaluation_meta != {'epoch': epoch, 'iter': epoch * EpochSteps(scope, len(gpu_ids))}:
            raise RuntimeError(f'待补验证checkpoint错误：{evaluation_checkpoint}')
        command = TrainCommand(stage, b_temporal, output_root, len(gpu_ids),
                               ['--resume', str(evaluation_checkpoint), '--validate-only'], scope)
        Run(command, stage, epoch, 'validate', output_root, gpu_ids, scope)
        if not any(record['epoch'] == epoch for record in ValidationRecords(output, scope)):
            raise RuntimeError(f'验证成功退出但阶段记录缺失：{output}: {epoch}')
    if phase == 48 and sorted(record['epoch'] for record in ValidationRecords(output, scope)) != list(VALIDATION_EPOCHS):
        raise RuntimeError(f'最终验证历史不完整：{output}')
    image_dir = output / 'images' / f'epoch_{phase:02d}'
    image_dir.mkdir(parents=True, exist_ok=True)
    loss_log = image_dir / 'train_metrics.jsonl'
    if not loss_log.exists():
        with loss_log.open('x') as stream:
            for record in ReadRecords(output / 'train_metrics.jsonl'):
                if record['iter'] <= meta['iter']:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    for split, index in SCOPES[scope].get('figures', FIGURES):
        image = image_dir / f'{split}_{index:03d}.png'
        curves = image_dir / 'loss_curves.png'
        if split == 'train' and image.exists() and not curves.exists():
            image = image_dir / 'loss_train_010.png'
            if image.exists():
                raise RuntimeError(f'补曲线产物不完整，需主agent审计：{image_dir}')
        if image.exists():
            continue
        command = FigureCommand(stage, b_temporal, checkpoint, split, index, image,
                                loss_log if split == 'train' and not curves.exists() else None, scope)
        Run(command, stage, phase, f'image_{split}_{index}', output_root, gpu_ids[:1], scope)
        if not image.exists() or not curves.exists():
            raise RuntimeError(f'出图成功退出但PNG/曲线缺失：{image_dir}')
    record = dict(time=Now(), stage=stage, phase=phase, status='phase_complete',
                  checkpoint=str(checkpoint), meta=meta, validation_frames=SCOPES[scope]['val_count'],
                  next_step='继续下一phase/stage；主比较使用epoch_48')
    if scope != 'fixed86':
        record['scope'] = scope
    AppendProgress(record)


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=tuple(SCOPES), default='xyzt_completed')
    parser.add_argument('--stage', choices=tuple(dict.fromkeys(item[0] for item in (*STAGES, *XYZT_COMPLETED_STAGES))), default=None,
                        help='动态正式主线每次仅运行指定阶段，时序须先完成定向诊断')
    parser.add_argument('--allow-legacy', action='store_true',
                        help='显式允许历史固定抽样实验；过时自动任务不能默认重启')
    parser.add_argument('--output-root', type=Path, default=None)
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'),
                        help='同一scope全部实验统一使用的GPU列表，如0或0,1；出图只用首卡')
    parser.add_argument('--dry-run', action='store_true', help='仅展示从头运行的固定计划，不读取标签或checkpoint')
    args = parser.parse_args()
    if not args.allow_legacy and not args.dry_run:
        parser.error('旧48轮流程已退出当前主线；本轮使用run_detection_screen.py最多5轮逐轮验收。历史复现需显式--allow-legacy。')
    scope = args.scope
    configured_stages = SCOPES[scope]['stages']
    stages = tuple(item for item in configured_stages if args.stage is None or item[0] == args.stage)
    if not stages:
        parser.error('该scope没有指定阶段')
    gpu_ids = args.gpus.split(',')
    if any(not value.isdigit() for value in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error('--gpus须为互不重复的GPU整数编号')
    if scope in ('dynamic50', 'xyzt_completed') and (len(gpu_ids) != 8 or args.stage is None):
        parser.error('正式dynamic50必须指定八张GPU和当前--stage，不自动串接未经诊断的时序')
    if scope not in ('dynamic50', 'xyzt_completed') and not args.allow_legacy and not args.dry_run:
        parser.error('历史固定抽样已退出主线；仅显式--allow-legacy可复现')
    output_root = (args.output_root or PROJECT_ROOT / 'log' / SCOPES[scope]['output_name']).expanduser().resolve()
    if args.dry_run:
        for stage, b_temporal, source_stage in stages:
            for phase in PHASES:
                extra = ['--stop-after-epoch', str(phase)]
                if phase != PHASES[0]:
                    extra += ['--resume', str(output_root / stage / 'checkpoints' / 'latest.pth')]
                elif source_stage is not None:
                    extra += ['--init-checkpoint', str(output_root / source_stage / 'checkpoints' / 'epoch_48.pth')]
                print(json.dumps(TrainCommand(stage, b_temporal, output_root, len(gpu_ids), extra, scope)))
                for split, index in SCOPES[scope].get('figures', FIGURES):
                    image_dir = output_root / stage / 'images' / f'epoch_{phase:02d}'
                    print(json.dumps(FigureCommand(stage, b_temporal,
                        output_root / stage / 'checkpoints' / f'epoch_{phase}.pth', split, index,
                        image_dir / f'{split}_{index:03d}.png',
                        image_dir / 'train_metrics.jsonl' if split == 'train' else None, scope)))
        return
    if scope in ('full50', 'dynamic50', 'xyzt_completed'):
        CheckFull50Labels(scope)
    if scope in ('dynamic50', 'xyzt_completed') and args.stage.startswith('temporal'):
        decision_path = output_root / 'temporal_diagnostic_decision.json'
        decision = json.loads(decision_path.read_text())
        source_hashes = {name: hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest()
                         for name in ('multi_mission_model.py', 'multi_mission_model_memory.py')}
        if (decision.get('b_ready_for_long_training') is not True or
                decision.get('model_source_sha256') != source_hashes):
            raise RuntimeError('时序长训练前必须完成当前模型的定向退化诊断并记录结论')
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        settings_path = output_root / 'execution_settings.json'
        settings = {'world_size': len(gpu_ids), 'phases': list(PHASES), 'stages': [item[0] for item in configured_stages]}
        if scope != 'fixed86':
            settings['scope'] = scope
        if scope in ('dynamic50', 'xyzt_completed'):
            settings['sampling'] = {'candidate_count': 23688, 'fraction': 0.5,
                                    'epoch_target_count': 11844, 'seed': 0,
                                    'steps_per_epoch': 1481, 'padded_repeats_per_epoch': 4}
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != settings:
                raise RuntimeError('恢复时world_size/阶段预算改变，需主agent审计')
        else:
            with settings_path.open('x') as stream:
                json.dump(settings, stream, indent=2)
        for stage, b_temporal, source_stage in stages:
            output = output_root / stage
            for phase in PHASES:
                meta = AuditStage(output, output_root, len(gpu_ids), scope)
                if meta is None or meta['epoch'] < phase:
                    extra = ['--stop-after-epoch', str(phase)]
                    if meta is not None:
                        extra += ['--resume', str(output / 'checkpoints' / 'latest.pth')]
                    elif source_stage is not None:
                        source = output_root / source_stage / 'checkpoints' / 'epoch_48.pth'
                        if CheckpointMeta(source)['epoch'] != 48:
                            raise RuntimeError(f'初始化权重不是epoch48：{source}')
                        extra += ['--init-checkpoint', str(source)]
                    Run(TrainCommand(stage, b_temporal, output_root, len(gpu_ids), extra, scope),
                        stage, phase, 'train', output_root, gpu_ids, scope)
                    meta = AuditStage(output, output_root, len(gpu_ids), scope)
                    if meta is None or meta['epoch'] != phase:
                        raise RuntimeError(f'训练成功退出但阶段checkpoint不匹配：{output}: {meta}')
                CompletePhase(stage, b_temporal, phase, output_root, gpu_ids, scope)
        AppendProgress(dict(time=Now(), status='selected_stage_complete' if args.stage else ('five_stages_complete' if scope == 'fixed86' else 'full50_two_stages_complete'), stage=args.stage, output_root=str(output_root),
                            next_step='审查当前阶段完整指标/图片；按最新优先级在诊断通过后显式进入下一阶段'))


if __name__ == '__main__':
    try:
        Main()
    except (Exception, KeyboardInterrupt) as error:
        AppendProgress(dict(time=Now(), status='stopped', error=str(error),
                            next_step='主agent审计异常和最后命令；不自动修复日志或改预算'))
        raise
