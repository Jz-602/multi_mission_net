"""单候选0～5轮串行筛选；复用服务器监督，逐轮验收后才启动下一轮。"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from run_options import HashFile, WriteJsonAtomic
from detection_screen_gate import EvaluateGate
from detection_screen_runtime import CheckGateBinding, CheckRawUpdates, ConfigIdentity

SOURCE = PROJECT / 'log/server_runs_xyzt_completed_20261007/single/checkpoints/epoch_48.pth'
SOURCE_SHA = '57da573ac78160840cfad15cad3ad0646d77fbfa673a1f417361b97f5f8fc928'
OUTPUT_ROOT = PROJECT / 'log/server_runs_x4_detection_screen_20261008'
FIGURES = (('train', 10), ('val', 0), ('val', 100), ('val', 300), ('val', 2720))


class ManualPauseError(RuntimeError):
    pass


def Read(path):
    return json.loads(path.read_text())


def CheckActive():
    if Read(PROJECT / 'log/server_supervision/control.json').get('mode') != 'active':
        raise ManualPauseError('Manual pause/cancel; refusing next action')
    code = ('import json; from supervise_server_job import CodexClient; '
            'client=CodexClient(); print(json.dumps(client.Goal())); client.Close()')
    try:
        result = subprocess.run(['/home/gk/anaconda3/bin/python', '-c', code],
                                cwd=PROJECT / 'tools', capture_output=True, text=True,
                                timeout=45, check=True)
        goal = json.loads(result.stdout)
        if not goal or goal.get('status') != 'active':
            raise ValueError('Fixed-session goal is not active')
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        raise ManualPauseError(f'Goal status unavailable/inactive; refusing next action: {error}') from error


def CheckEpochEvidence(output, epoch):
    import torch
    checkpoint = output / f'checkpoints/epoch_{epoch}.pth'
    meta = torch.load(checkpoint, map_location='cpu', weights_only=False)['meta']
    if meta.get('epoch') != epoch or meta.get('iter') != epoch * 1481:
        raise RuntimeError('Screen checkpoint epoch/update budget mismatch')
    CheckRawUpdates(output, epoch)
    manifest = Read(output / 'manifest.json')
    baseline_manifest = Read(OUTPUT_ROOT / 'single/manifest.json')
    for split, count in (('train', 23688), ('val', 5973)):
        entry = manifest['splits'][split]
        if (entry['count'] != count or len(entry['sha256']) != 64
                or entry != baseline_manifest['splits'][split]):
            raise RuntimeError('Screen split signature differs from complete B0 targets')
    return manifest['splits']


def Execute(command, output, action):
    CheckActive()
    directory = output / 'commands' / action
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / (datetime.datetime.now().strftime('%Y%m%dT%H%M%S') + '.log')
    start = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=PROJECT, env=os.environ.copy(),
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        WriteJsonAtomic(output / 'process.json', dict(action=action, pid=process.pid,
                        command=command, log=str(log), started_at=datetime.datetime.now().astimezone().isoformat()))
        try:
            code = process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            raise
    record = dict(action=action, returncode=code, elapsed_seconds=time.monotonic()-start, log=str(log))
    with (output / 'timings.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    if code:
        raise RuntimeError(f'{action}: exit={code}, see {log}')


def Run(candidate):
    output = OUTPUT_ROOT / candidate
    output.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / 'failed_action.json').exists():
            raise RuntimeError('Previous failure requires explicit review; refusing blind retry')
        if (output / 'stage_result.json').exists():
            raise RuntimeError('Candidate already ended; no automatic restart')
        try:
            RunCandidate(candidate, output)
        except ManualPauseError:
            raise
        except BaseException as error:
            WriteJsonAtomic(output / 'failed_action.json', dict(
                error=str(error), error_type=type(error).__name__,
                time=datetime.datetime.now().astimezone().isoformat(),
                b_retry_requires_review=True))
            raise


def RunCandidate(candidate, output):
    if (output / 'stage_result.json').exists():
        raise RuntimeError('Candidate already ended; no automatic restart')
    if HashFile(SOURCE) != SOURCE_SHA:
        raise RuntimeError('Shared single checkpoint changed')
    config = PROJECT / f'configs/server_x4_screen_{candidate}.yaml'
    from tools.build_label_manifest import LoadConfig
    cfg = LoadConfig(config)
    screen = cfg['detection_screen']
    if (screen['max_new_epochs'] != 5 or screen['max_detection_drop_pp'] != 5.0
            or screen['expected_targets'] != 5973):
        raise RuntimeError('Screen cap/drop threshold cannot be relaxed')
    if cfg['trainer']['max_epochs'] != 48 or cfg['trainer']['eval_every_nepochs'] != 1:
        raise RuntimeError('Must preserve 48-epoch LR prefix and validate every new epoch')
    protocol = dict(config=str(config), config_identity=ConfigIdentity(config),
                    source=str(SOURCE), source_sha256=SOURCE_SHA, **screen)
    protocol_path = output / 'screen_protocol.json'
    if protocol_path.exists() and Read(protocol_path) != protocol:
        raise RuntimeError('Existing candidate config/label/source binding changed')
    WriteJsonAtomic(protocol_path, protocol)
    acceptance = output / 'acceptance'
    acceptance.mkdir(exist_ok=True)
    baseline_path = OUTPUT_ROOT / 'single/validation/epoch_0.json'
    if candidate != 'single' and not baseline_path.exists():
        raise RuntimeError('B0 full validation is required before temporal candidates')
    base_command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                    '--nproc_per_node', '8', str(PROJECT/'LYF_train_ODSS_multi.py'),
                    '--config', str(config), '--output', str(output)]
    (output / 'validation').mkdir(exist_ok=True)
    for epoch in range(6):
        CheckActive()
        gate_path = acceptance / f'epoch_{epoch}.json'
        if gate_path.exists():
            if Read(gate_path)['status'] != 'pass':
                raise RuntimeError('Previous rejected epoch cannot be automatically retried')
            CheckGateBinding(Read(gate_path), output, config)
            continue
        checkpoint = output / f'checkpoints/epoch_{epoch}.pth'
        validation = output / f'validation/epoch_{epoch}.json'
        if not validation.exists():
            if checkpoint.exists():
                # 训练已保存但验证中断：只补评价，禁止重复训练该轮。
                command = base_command + ['--resume', str(checkpoint), '--validate-only']
            elif epoch == 0:
                command = base_command + ['--init-checkpoint', str(SOURCE), '--validate-only']
            else:
                command = base_command + ['--resume', str(output/f'checkpoints/epoch_{epoch-1}.pth'),
                                           '--stop-after-epoch', str(epoch)]
            Execute(command, output, f'epoch_{epoch}')
            current = Read(output / 'validation_metrics.json')
            if current['epoch'] != epoch:
                raise RuntimeError('Missing current epoch validation')
            WriteJsonAtomic(validation, current)
        current = Read(validation)
        if current.get('epoch') != epoch:
            raise RuntimeError('Validation snapshot belongs to another epoch')
        split_signature = CheckEpochEvidence(output, epoch)
        baseline = current if candidate == 'single' and epoch == 0 else Read(baseline_path)
        gate = EvaluateGate(baseline, current, screen['max_detection_drop_pp'], screen['expected_targets'])
        gate.update(epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=HashFile(checkpoint),
                    validation=str(validation), validation_sha256=HashFile(validation),
                    baseline=str(baseline_path), baseline_sha256=HashFile(baseline_path),
                    protocol_sha256=HashFile(protocol_path), split_signature=split_signature)
        # 验证后先写拒绝结论；只有图片验收完成才开放下一轮。
        if gate['status'] != 'pass':
            WriteJsonAtomic(gate_path, gate)
        for split, index in FIGURES:
            image = output / f'images/epoch_{epoch:02d}/{split}_{index:03d}.png'
            curve = image.parent / 'loss_curves.png'
            if not image.exists() or (epoch >= 1 and split == 'train' and not curve.exists()):
                command = [sys.executable, str(PROJECT/'tools/visualize_three_task.py'), '--temporal',
                         '--config', str(config), '--checkpoint', str(checkpoint), '--split', split,
                         '--index', str(index), '--output', str(image)]
                if epoch >= 1 and split == 'train':
                    command += ['--loss-log', str(output / 'train_metrics.jsonl')]
                Execute(command, output, f'epoch_{epoch}_{split}_{index}')
            if image.stat().st_size < 1000:
                raise RuntimeError('Invalid figure output')
        if epoch >= 1 and not (output / f'images/epoch_{epoch:02d}/loss_curves.png').is_file():
            raise RuntimeError('Missing per-epoch raw loss curve')
        images = output / f'images/epoch_{epoch:02d}'
        Execute(['/home/gk/anaconda3/envs/vggt_py310/bin/python',
                 str(PROJECT / 'tools/summarize_detection_duplicates.py'),
                 '--directory', str(images)], output, f'epoch_{epoch}_duplicates')
        if len(Read(images / 'duplicate_diagnostics.json')['records']) != len(FIGURES):
            raise RuntimeError('Missing five-sample duplicate diagnostics')
        WriteJsonAtomic(gate_path, gate)
        timings = [json.loads(line) for line in (output/'timings.jsonl').read_text().splitlines()]
        elapsed = sum(row['elapsed_seconds'] for row in timings)
        epoch0_seconds = sum(row['elapsed_seconds'] for row in timings if row['action'].startswith('epoch_0'))
        state = dict(candidate=candidate, epoch=epoch, status=gate['status'], elapsed_seconds=elapsed,
                     checkpoint=str(checkpoint), gate=str(gate_path), max_new_epochs=5,
                     remaining_seconds_estimate=(5-epoch)*(elapsed-epoch0_seconds)/epoch
                     if epoch and gate['status'] == 'pass' else None)
        WriteJsonAtomic(output / 'state.json', state)
        with (PROJECT/'SERVER_PROGRESS.md').open('a') as stream:
            stream.write('\nX4逐轮验收：'+json.dumps(state, ensure_ascii=False)+'\n')
        if gate['status'] != 'pass':
            WriteJsonAtomic(output/'stage_result.json', dict(state, b_effectiveness_confirmed=False))
            return
    state = dict(candidate=candidate, epoch=5, status='five_epochs_finished_pending_effectiveness_review',
                 b_effectiveness_confirmed=False, max_new_epochs=5)
    WriteJsonAtomic(output/'stage_result.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', choices=('single', 'a', 'b', 'c', 'single_f', 'f', 'f_no_root'), required=True)
    Run(parser.parse_args().candidate)
