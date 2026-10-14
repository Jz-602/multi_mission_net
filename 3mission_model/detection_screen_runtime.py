"""筛选入口硬限制：每次最多一轮，必须有上一轮绑定权重的验收。"""
import json
import hashlib
from pathlib import Path

from run_options import HashFile
from tools.build_label_manifest import LoadConfig


CONTROL_PATH = Path(__file__).parent / 'log/server_supervision/control.json'


def CheckRawUpdates(output, epoch):
    raw = output / 'train_metrics.jsonl'
    records = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
    if [row['iter'] for row in records] != list(range(1, epoch * 1481 + 1)):
        raise RuntimeError('Raw updates missing/duplicated/ahead of saved epoch; review before recovery')


def ConfigIdentity(filename):
    """Bind expanded inherited config and detection-label manifest contents."""
    cfg = LoadConfig(filename)
    result = {'config_sha256': hashlib.sha256(
        json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()}
    from run_options import ResolveConfigArtifact
    for key in ('completed_detection_manifest', 'label_manifest'):
        if not cfg.get(key):
            raise ValueError(f'Screen requires {key}')
        result[key + '_sha256'] = HashFile(ResolveConfigArtifact(cfg[key], filename))
    return result


def CheckGateBinding(gate, output, config_filename):
    """A pass authorizes only its exact config, GT, validation and checkpoint."""
    protocol_path = output / 'screen_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if protocol.get('config_identity') != ConfigIdentity(config_filename):
        raise ValueError('Screen configuration or label manifest changed')
    if gate.get('protocol_sha256') != HashFile(protocol_path):
        raise ValueError('Gate belongs to a different screen protocol')
    epoch = gate.get('epoch')
    if type(epoch) is not int or not 0 <= epoch <= 5:
        raise ValueError('Invalid gate epoch')
    expected_checkpoint = output / f'checkpoints/epoch_{epoch}.pth'
    expected_validation = output / f'validation/epoch_{epoch}.json'
    for field, expected in (('checkpoint', expected_checkpoint), ('validation', expected_validation)):
        if (not gate.get(field) or Path(gate[field]).resolve() != expected.resolve()
                or gate.get(field + '_sha256') != HashFile(expected)):
            raise ValueError(f'Gate {field} changed')
    baseline = output.parent / 'single/validation/epoch_0.json'
    if (not gate.get('baseline') or Path(gate['baseline']).resolve() != baseline.resolve()
            or gate.get('baseline_sha256') != HashFile(baseline)):
        raise ValueError('Gate B0 baseline changed')
    current = json.loads(expected_validation.read_text())
    if current.get('epoch') != epoch:
        raise ValueError('Gate validation epoch mismatch')
    splits = json.loads((output / 'manifest.json').read_text())['splits']
    baseline_splits = json.loads((output.parent / 'single/manifest.json').read_text())['splits']
    if splits != gate.get('split_signature') or splits != baseline_splits:
        raise ValueError('Gate training/validation targets changed')
    if splits['train']['count'] != 23688 or splits['val']['count'] != 5973:
        raise ValueError('Gate does not cover the complete candidate and validation pools')
    from tools.detection_screen_gate import EvaluateGate
    fresh = EvaluateGate(json.loads(baseline.read_text()), current,
                         protocol['max_detection_drop_pp'], protocol['expected_targets'])
    if gate.get('status') != 'pass' or fresh['status'] != 'pass':
        raise ValueError('Prior gate is not a current valid pass')


def RunScreenStep(trainer, cfg, args):
    screen = cfg.detection_screen
    if screen.max_new_epochs != 5 or cfg.trainer.max_epochs != 48:
        raise ValueError('Screen requires a five-epoch cap and the original 48-epoch LR schedule')
    if cfg.trainer.eval_every_nepochs != 1:
        raise ValueError('Screen requires validation after every epoch')
    if type(trainer.epoch) is not int or not 0 <= trainer.epoch <= 5:
        raise ValueError('Checkpoint exceeds the new-epoch screen budget')
    if screen.max_detection_drop_pp != 5.0 or screen.expected_targets != 5973:
        raise ValueError('Screen drop threshold/complete validation cannot be relaxed')
    control = json.loads(CONTROL_PATH.read_text())
    if control.get('mode') != 'active':
        raise RuntimeError('Manual pause/cancel: no screen action')
    output = Path(trainer.checkpoint_dir).parent
    if (output / 'failed_action.json').exists():
        raise RuntimeError('Prior failed action needs explicit review before retry')
    protocol = json.loads((output / 'screen_protocol.json').read_text())
    if protocol.get('config_identity') != ConfigIdentity(args.config):
        raise ValueError('Screen configuration or labels differ from recorded protocol')
    if args.validate_only:
        if not (args.init_checkpoint or args.resume):
            raise ValueError('Screen validation needs an explicit checkpoint')
        if not args.resume and (trainer.epoch != 0 or
                HashFile(args.init_checkpoint) != protocol['source_sha256']):
            raise ValueError('Epoch0 must initialize from the bound single checkpoint')
        if args.resume and Path(args.resume).resolve() != (
                output / f'checkpoints/epoch_{trainer.epoch}.pth').resolve():
            raise ValueError('Validation recovery must use this screen epoch checkpoint')
        if trainer.rank == 0 and not args.resume:
            trainer.save_checkpoint('epoch_0.pth')
        trainer.val_epoch()
        return
    if not args.resume or args.stop_after_epoch != trainer.epoch + 1 or args.stop_after_epoch > 5:
        raise ValueError('Screen trains exactly one new epoch from an accepted checkpoint, capped at five')
    gate = json.loads((output / 'acceptance' / f'epoch_{trainer.epoch}.json').read_text())
    if (gate.get('status') != 'pass' or gate.get('epoch') != trainer.epoch
            or gate.get('checkpoint_sha256') != HashFile(args.resume)):
        raise ValueError('Previous full validation has not accepted this exact checkpoint')
    CheckGateBinding(gate, output, args.config)
    CheckRawUpdates(output, trainer.epoch)
    # 常规训练显存留痕；不额外前向、不引入覆盖统计，验证/出图不计入训练峰值。
    from training_memory import TrackTrainingMemory
    TrackTrainingMemory(trainer)
    trainer.max_epochs = args.stop_after_epoch
    trainer.fit()
