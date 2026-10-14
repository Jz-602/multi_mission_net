"""恢复已验收F的5轮前缀至48轮；固定验证门禁，不启动其他候选。"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from run_options import HashFile, WriteJsonAtomic
from tools.build_label_manifest import LoadConfig
from detection_screen_runtime import CheckRawUpdates, ConfigIdentity
from detection_screen_gate import EvaluateGate
from run_detection_screen import CheckActive, Execute, FIGURES, ManualPauseError

SOURCE = PROJECT / 'log/server_runs_x4_detection_screen_20261008/f'
SOURCE_SHA = 'a32e4818afa0255dd8bc9c07f86a4ad3f88d66a31f2afe95c323ea396b16ad07'
BASELINE = SOURCE.parent / 'single/validation/epoch_0.json'
CONFIG = PROJECT / 'configs/server_x4_selected_f_long48.yaml'
OUTPUT = PROJECT / 'log/server_runs_x4_selected_long48_20261010/f'
PHASES = (6, 10, 15, 20, 25, 30, 35, *range(40, 49))


def Read(path):
    return json.loads(path.read_text())


def CheckConfig():
    old = LoadConfig(PROJECT / 'configs/server_x4_screen_f.yaml')
    new = LoadConfig(CONFIG)
    expected = json.loads(json.dumps(old))
    expected['detection_screen'] = None
    expected['collect_training_memory'] = True
    expected['trainer'].update(eval_every_nepochs=0, eval_epochs=list(PHASES))
    if new != expected or new['trainer']['max_epochs'] != 48:
        raise ValueError('Long training may only change execution cap/validation/resource logging')
    return ConfigIdentity(CONFIG)


def CheckCheckpoint(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    epoch = saved['meta']['epoch']
    step = epoch * 1481
    if not 5 <= epoch <= 48 or saved['meta']['iter'] != step:
        raise ValueError('Unexpected continuation checkpoint epoch/step')
    if (saved['scheduler']['total_steps'] != 71088 or saved['scheduler']['last_epoch'] != step
            or len(saved['optimizer']['state']) != 339):
        raise ValueError('Optimizer/scheduler continuation mismatch')
    for value in saved['optimizer']['state'].values():
        if int(value['step']) != step:
            raise ValueError('Optimizer step mismatch')
    for section in ('state_dict', 'optimizer', 'scheduler'):
        def CheckFinite(value):
            if isinstance(value, dict):
                for child in value.values():
                    CheckFinite(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    CheckFinite(child)
            elif torch.is_tensor(value) and not torch.isfinite(value).all():
                raise ValueError('Nonfinite checkpoint tensor')
        CheckFinite(saved[section])
    return epoch


def Prepare():
    identity = CheckConfig()
    checkpoint = SOURCE / 'checkpoints/epoch_5.pth'
    if HashFile(checkpoint) != SOURCE_SHA or CheckCheckpoint(checkpoint) != 5:
        raise ValueError('Accepted F5 changed')
    protocol = dict(source=str(SOURCE), source_checkpoint=str(checkpoint),
                    source_sha256=SOURCE_SHA, config_identity=identity,
                    start_epoch=5, final_epoch=48, validation_epochs=list(PHASES),
                    max_detection_drop_pp=5., baseline=str(BASELINE),
                    baseline_sha256=HashFile(BASELINE),
                    split_signature=Read(SOURCE/'manifest.json')['splits'],
                    b_resume_optimizer_and_scheduler=True,
                    comparison='S48 baseline vs S48 plus temporal48; not equal optimization budget')
    path = OUTPUT / 'long_protocol.json'
    if path.exists():
        if Read(path) != protocol:
            raise ValueError('Existing long training protocol changed')
        return
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise ValueError('Unregistered partial output requires review')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in ('checkpoints', 'acceptance', 'validation'):
        (OUTPUT / name).mkdir()
    # 日志复制后继续追加；权重和历史TB只读链接，不向旧筛选目录写入。
    for name in ('train_metrics.jsonl', 'validation_metrics.jsonl', 'validation_metrics.json',
                 'manifest.json', 'train_tokens.json', 'val_tokens.json'):
        shutil.copy2(SOURCE / name, OUTPUT / name)
    for name in ('epoch_sampling', 'training_resources', 'validation'):
        shutil.copytree(SOURCE / name, OUTPUT / name, dirs_exist_ok=True)
    (OUTPUT / 'checkpoints/epoch_5.pth').symlink_to(checkpoint)
    (OUTPUT / 'inherited_five_epoch_evidence').symlink_to(SOURCE, target_is_directory=True)
    CheckRawUpdates(OUTPUT, 5)
    WriteJsonAtomic(path, protocol)


def VerifyProtocol():
    protocol = Read(OUTPUT/'long_protocol.json')
    if (ConfigIdentity(CONFIG) != protocol['config_identity']
            or HashFile(BASELINE) != protocol['baseline_sha256']
            or Read(OUTPUT/'manifest.json')['splits'] != protocol['split_signature']):
        raise ValueError('Bound configuration/labels/baseline/targets changed')


def VerifyGate(epoch):
    VerifyProtocol()
    gate = Read(OUTPUT / f'acceptance/epoch_{epoch}.json')
    if (gate['status'] != 'pass' or gate['protocol_sha256'] != HashFile(OUTPUT/'long_protocol.json')
            or gate['checkpoint_sha256'] != HashFile(OUTPUT/f'checkpoints/epoch_{epoch}.pth')
            or gate['validation_sha256'] != HashFile(OUTPUT/f'validation/epoch_{epoch}.json')):
        raise ValueError('Recorded gate no longer matches checkpoint/protocol/validation')
    current = Read(OUTPUT/f'validation/epoch_{epoch}.json')
    if current['epoch'] != epoch or EvaluateGate(Read(BASELINE), current)['status'] != 'pass':
        raise ValueError('Recorded validation no longer passes')


def RunPhases():
    base = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '8',
            '--', str(PROJECT/'LYF_train_ODSS_multi.py'), '--config', str(CONFIG), '--output', str(OUTPUT)]
    previous = 5
    for epoch in PHASES:
        CheckActive()
        CheckConfig()
        VerifyProtocol()
        gate_path = OUTPUT / f'acceptance/epoch_{epoch}.json'
        if gate_path.exists():
            VerifyGate(epoch)
            previous = epoch
            continue
        checkpoint = OUTPUT / f'checkpoints/epoch_{epoch}.pth'
        validation = OUTPUT / f'validation/epoch_{epoch}.json'
        if not validation.exists():
            current = Read(OUTPUT/'validation_metrics.json')
            if checkpoint.exists() and current['epoch'] == epoch:
                WriteJsonAtomic(validation, current)
            else:
                resume = checkpoint if checkpoint.exists() else OUTPUT/'checkpoints/latest.pth'
                if not resume.exists():
                    resume = OUTPUT / f'checkpoints/epoch_{previous}.pth'
                restored_epoch = CheckCheckpoint(resume)
                if not previous <= restored_epoch <= epoch:
                    raise ValueError('Recovery checkpoint outside current validated phase')
                CheckRawUpdates(OUTPUT, restored_epoch)
                command = base + ['--resume', str(resume)]
                command += ['--validate-only'] if restored_epoch == epoch else ['--stop-after-epoch', str(epoch)]
                Execute(command, OUTPUT, f'epoch_{epoch}')
                current = Read(OUTPUT/'validation_metrics.json')
                if current['epoch'] != epoch:
                    raise ValueError('Missing final phase validation')
                WriteJsonAtomic(validation, current)
        if CheckCheckpoint(checkpoint) != epoch:
            raise ValueError('Validation checkpoint mismatch')
        CheckRawUpdates(OUTPUT, epoch)
        if Read(OUTPUT/'manifest.json')['splits'] != Read(SOURCE/'manifest.json')['splits']:
            raise ValueError('Candidate or validation targets changed')
        current = Read(validation)
        if current['epoch'] != epoch:
            raise ValueError('Validation snapshot belongs to another epoch')
        VerifyProtocol()
        gate = EvaluateGate(Read(BASELINE), current)
        gate.update(epoch=epoch, checkpoint_sha256=HashFile(checkpoint),
                    validation_sha256=HashFile(validation), protocol_sha256=HashFile(OUTPUT/'long_protocol.json'))
        # 拒绝状态先持久化，图片出错也不能绕过停止线。
        if gate['status'] != 'pass':
            WriteJsonAtomic(gate_path, gate)
        for split, index in FIGURES:
            image = OUTPUT/f'images/epoch_{epoch:02d}/{split}_{index:03d}.png'
            if not image.exists():
                command = [sys.executable, str(PROJECT/'tools/visualize_three_task.py'), '--temporal',
                           '--config', str(CONFIG), '--checkpoint', str(checkpoint),
                           '--split', split, '--index', str(index), '--output', str(image)]
                if split == 'train':
                    command += ['--loss-log', str(OUTPUT/'train_metrics.jsonl')]
                Execute(command, OUTPUT, f'epoch_{epoch}_{split}_{index}')
            if image.stat().st_size < 1000:
                raise ValueError('Missing or invalid real prediction image')
        if not (OUTPUT/f'images/epoch_{epoch:02d}/loss_curves.png').is_file():
            raise ValueError('Missing raw loss figure')
        WriteJsonAtomic(gate_path, gate)
        state = dict(epoch=epoch, status=gate['status'], checkpoint=str(checkpoint),
                     b_complete=False, next_epoch=next((e for e in PHASES if e>epoch), None))
        WriteJsonAtomic(OUTPUT/'state.json', state)
        if gate['status'] != 'pass':
            WriteJsonAtomic(OUTPUT/'stage_result.json', dict(state, b_review_required=True))
            return
        previous = epoch
    WriteJsonAtomic(OUTPUT/'stage_result.json', dict(epoch=48, status='training_complete_pending_acceptance',
                   b_complete=False, checkpoint=str(OUTPUT/'checkpoints/epoch_48.pth')))


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with (OUTPUT.parent/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT/'failed_action.json').exists() or (OUTPUT/'stage_result.json').exists():
            raise ValueError('Terminal/failed run requires explicit review; no automatic retry')
        CheckActive()
        Prepare()
        if args.prepare_only:
            return
        try:
            RunPhases()
        except ManualPauseError:
            raise
        except BaseException as error:
            WriteJsonAtomic(OUTPUT/'failed_action.json', dict(error=repr(error), b_retry_requires_review=True))
            raise


if __name__ == '__main__':
    Main()
