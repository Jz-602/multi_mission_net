"""论文统一协议：每轮训练、完整验证、图片、门禁，复用唯一服务器监督。"""
import argparse
import fcntl
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tools'))
from run_options import HashFile, WriteJsonAtomic
from detection_screen_runtime import ConfigIdentity, CheckRawUpdates
from thesis_v1_runtime import CheckAcceptance, CheckShortStageApproval
from thesis_v1_gate import EvaluateThesisGate, SelectThesisCheckpoint
from tools.run_detection_screen import Execute, CheckActive, ManualPauseError

OUTPUT = PROJECT / 'log/server_thesis_v1_20261011'
PREPARATION = PROJECT / 'log/server_preparation/thesis_v1_20261011'
CONFIGS = dict(s0='single', s='single_continuation', t='temporal',
               s_no_root='single_no_root', t_no_root='temporal_no_root')
STAGE_NAMES = dict(s0='S0', s='S', t='T', s_no_root='S-R', t_no_root='T-R')
FIGURES = (('train', 10), ('val', 0), ('val', 100), ('val', 300), ('val', 2720),
           ('val', 1166), ('val', 5047), ('val', 4346))


def Read(path):
    return json.loads(Path(path).read_text())


def Register(stage, output, config, limit):
    ready = Read(PREPARATION / 'formal_readiness.json')
    if ready.get('b_ready_for_s0') is not True:
        raise RuntimeError('Root decision and real eight-GPU preflight not accepted')
    if ready['config_identities'][stage] != ConfigIdentity(config):
        raise RuntimeError('Configuration changed after preflight/final protocol freeze')
    source, reference = None, None
    if stage != 's0':
        accepted = Read(OUTPUT / 's0/final_acceptance.json')
        if accepted.get('b_s0_accepted') is not True:
            raise RuntimeError('S0 metrics and fixed validation figures not yet accepted')
        source = Path(accepted['checkpoint'])
        if HashFile(source) != accepted['checkpoint_sha256']:
            raise RuntimeError('Accepted S0 checkpoint changed')
        reference = Read(accepted['validation'])
        if HashFile(accepted['validation']) != accepted['validation_sha256']:
            raise RuntimeError('Accepted S0 reference changed')
    protocol = dict(stage=stage, config_identity=ConfigIdentity(config),
                    source=str(source) if source else None,
                    source_sha256=HashFile(source) if source else None,
                    formal_readiness_sha256=HashFile(PREPARATION / 'formal_readiness.json'),
                    specification_sha256=HashFile(PROJECT / 'THESIS_V1_PROTOCOL.md'),
                    max_epochs=48, steps_per_epoch=1481, expected_validation_targets=5973)
    path = output / 'thesis_protocol.json'
    if path.exists() and Read(path) != protocol:
        raise RuntimeError('Existing experiment binding changed; do not silently mix protocols')
    WriteJsonAtomic(path, protocol)
    if stage != 's0' and limit > 5:
        CheckShortStageApproval(output, protocol)
    return protocol, reference


def CheckEpoch(output, epoch):
    import torch
    path = output / f'checkpoints/epoch_{epoch}.pth'
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['meta']['epoch'] != epoch or checkpoint['meta']['iter'] != epoch * 1481:
        raise RuntimeError('Checkpoint budget differs from completed epoch')
    scheduler = checkpoint['scheduler']
    if scheduler['total_steps'] != 71088 or scheduler['last_epoch'] != epoch * 1481:
        raise RuntimeError('48-epoch learning-rate schedule changed or reset')
    states = list(checkpoint['optimizer']['state'].values())
    if epoch and (not states or {int(state['step']) for state in states} != {epoch * 1481}):
        raise RuntimeError('Optimizer state steps mismatch')
    CheckRawUpdates(output, epoch)
    manifest = Read(output / 'manifest.json')
    for split, count in [('train', 23688), ('val', 5973)]:
        if manifest['splits'][split]['count'] != count:
            raise RuntimeError('Full candidate/validation split changed')
    return manifest['splits']


def Figures(stage, config, output, epoch, b_all=False):
    cases = FIGURES if b_all or epoch in (0, 1, 5, 24, 48) else (('train', 10), ('val', 1166))
    images = []
    for split, index in cases:
        path = output / f'images/epoch_{epoch:02d}/{split}_{index:04d}.png'
        if not path.exists():
            command = [sys.executable, str(PROJECT / 'tools/visualize_three_task.py'),
                       '--config', str(config), '--checkpoint', str(output / f'checkpoints/epoch_{epoch}.pth'),
                       '--split', split, '--index', str(index), '--output', str(path)]
            if stage != 's0':
                command.append('--temporal')
            if epoch and split == 'train':
                command += ['--loss-log', str(output / 'train_metrics.jsonl')]
            Execute(command, output, f'epoch_{epoch}_{split}_{index}')
        details = path.with_name(path.stem + '_terrain_details.png')
        if min(path.stat().st_size, details.stat().st_size, path.with_suffix('.pt').stat().st_size) < 1000:
            raise RuntimeError('Missing real three-task/terrain-validity figure')
        images.append(str(path))
    return images


def Run(stage, limit):
    if limit not in (5, 48) or stage == 's0' and limit != 48:
        raise ValueError('S0 trains48; continuations stop at5 for review or authorized48')
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / 'thesis_runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = OUTPUT / stage
        output.mkdir(exist_ok=True)
        if (output / 'failed_action.json').exists():
            raise RuntimeError('Previous failure requires explicit review')
        try:
            RunStage(stage, output, limit)
        except ManualPauseError:
            raise
        except BaseException as error:
            WriteJsonAtomic(output / 'failed_action.json', dict(error=str(error),
                            error_type=type(error).__name__, b_retry_requires_review=True))
            raise


def RunStage(stage, output, limit):
    CheckActive()
    config = PROJECT / f'configs/server_thesis_v1_{CONFIGS[stage]}.yaml'
    protocol, reference = Register(stage, output, config, limit)
    entry = 'LYF_train_ODSS.py' if stage == 's0' else 'LYF_train_ODSS_multi.py'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=8',
               str(PROJECT / entry), '--config', str(config), '--output', str(output)]
    (output / 'validation').mkdir(exist_ok=True)
    (output / 'acceptance').mkdir(exist_ok=True)
    previous = None
    records = []
    for epoch in range(limit + 1):
        CheckActive()
        checkpoint = output / f'checkpoints/epoch_{epoch}.pth'
        validation = output / f'validation/epoch_{epoch}.json'
        accepted = output / f'acceptance/epoch_{epoch}.json'
        if accepted.exists():
            gate = CheckAcceptance(output, epoch, config, b_allow_later_updates=True)
            current = Read(validation)
            previous = dict(record=current, gate=gate)
            records.append(current)
            continue
        if not validation.exists():
            if checkpoint.exists():
                action = command + ['--resume', str(checkpoint), '--validate-only']
            elif epoch == 0:
                action = command + ['--validate-only']
                if stage != 's0':
                    action += ['--init-checkpoint', protocol['source']]
            else:
                action = command + ['--resume', str(output / f'checkpoints/epoch_{epoch-1}.pth'),
                                    '--stop-after-epoch', str(epoch)]
            Execute(action, output, f'epoch_{epoch}')
            current = Read(output / 'validation_metrics.json')
            if current['epoch'] != epoch:
                raise RuntimeError('Validation belongs to another epoch')
            WriteJsonAtomic(validation, current)
        current = Read(validation)
        splits = CheckEpoch(output, epoch)
        gate = EvaluateThesisGate(current, reference, previous, STAGE_NAMES[stage])
        gate.update(checkpoint=str(checkpoint), checkpoint_sha256=HashFile(checkpoint),
                    validation=str(validation), validation_sha256=HashFile(validation), split_signature=splits,
                    protocol_sha256=HashFile(output / 'thesis_protocol.json'))
        if gate['status'] != 'pass':
            WriteJsonAtomic(accepted, gate)
        gate['images'] = Figures(stage, config, output, epoch)
        gate['image_artifacts'] = [dict(path=str(path), sha256=HashFile(path))
                                  for image in gate['images']
                                  for path in (Path(image), Path(image).with_suffix('.pt'),
                                               Path(image).with_name(Path(image).stem + '_terrain_details.png'))]
        WriteJsonAtomic(accepted, gate)
        timings = [json.loads(line) for line in (output / 'timings.jsonl').read_text().splitlines()]
        elapsed = sum(row['elapsed_seconds'] for row in timings)
        initial = sum(row['elapsed_seconds'] for row in timings if row['action'].startswith('epoch_0'))
        state = dict(stage=stage, epoch=epoch, status=gate['status'], b_eligible=gate['b_eligible'],
                     score=gate['score'], elapsed_seconds=elapsed, checkpoint=str(checkpoint),
                     remaining_seconds_estimate=(limit-epoch)*(elapsed-initial)/epoch if epoch else None,
                     current_execution_cap=limit, scheduler_epochs=48)
        WriteJsonAtomic(output / 'state.json', state)
        with (PROJECT / 'SERVER_PROGRESS.md').open('a') as stream:
            stream.write('\n论文v1逐轮验收：' + json.dumps(state, ensure_ascii=False) + '\n')
        records.append(current)
        previous = dict(record=current, gate=gate)
        if gate['status'] != 'pass':
            WriteJsonAtomic(output / 'stage_result.json', dict(state, reasons=gate['reasons'],
                            b_final_effectiveness_accepted=False))
            return
    selection = SelectThesisCheckpoint(records, reference, STAGE_NAMES[stage])
    WriteJsonAtomic(output / 'selection.json', selection)
    if selection.get('selected_epoch') is not None:
        Figures(stage, config, output, selection['selected_epoch'], b_all=True)
    WriteJsonAtomic(output / f'phase_{limit}_result.json', dict(stage=stage, epoch=limit,
                    status='complete_validation_pending_scientific_review', selection=selection,
                    b_final_effectiveness_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=CONFIGS, required=True)
    parser.add_argument('--limit-epoch', type=int, required=True)
    args = parser.parse_args()
    Run(args.stage, args.limit_epoch)
