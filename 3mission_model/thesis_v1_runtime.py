"""新论文协议每次仅执行一个验收节点，保持48轮学习率预算。"""
import json
from pathlib import Path

from detection_screen_runtime import ConfigIdentity, CheckRawUpdates
from run_options import HashFile

CONTROL_PATH = Path(__file__).parent / 'log/server_supervision/control.json'


def Read(path):
    return json.loads(Path(path).read_text())


def CheckAcceptance(output, epoch, config, b_allow_later_updates=False):
    gate = Read(output / 'acceptance' / f'epoch_{epoch}.json')
    if gate['status'] != 'pass' or gate['epoch'] != epoch:
        raise ValueError('Previous complete validation is not accepted')
    protocol = output / 'thesis_protocol.json'
    if gate['protocol_sha256'] != HashFile(protocol):
        raise ValueError('Accepted protocol changed')
    if Read(protocol)['config_identity'] != ConfigIdentity(config):
        raise ValueError('Configuration or label binding changed')
    for key, path in [('checkpoint', output / f'checkpoints/epoch_{epoch}.pth'),
                      ('validation', output / f'validation/epoch_{epoch}.json')]:
        if Path(gate[key]).resolve() != path.resolve() or gate[key + '_sha256'] != HashFile(path):
            raise ValueError(f'Accepted {key} changed')
    if Read(output / 'manifest.json')['splits'] != gate.get('split_signature'):
        raise ValueError('Accepted target manifest changed')
    artifacts = gate.get('image_artifacts', [])
    expected_images = {str(Path(image).resolve()) for image in gate.get('images', [])}
    actual_images = {str(Path(item['path']).resolve()) for item in artifacts}
    required_images = set()
    for image in expected_images:
        path = Path(image)
        required_images.update((image, str(path.with_suffix('.pt')),
                               str(path.with_name(path.stem + '_terrain_details.png'))))
    if not expected_images or required_images != actual_images:
        raise ValueError('Accepted prediction figures/raw artifacts are incomplete')
    for item in artifacts:
        if HashFile(item['path']) != item['sha256']:
            raise ValueError('Accepted prediction artifact changed')
    if b_allow_later_updates:
        raw = output / 'train_metrics.jsonl'
        records = [json.loads(line) for line in raw.read_text().splitlines()] if raw.exists() else []
        updates = [row['iter'] for row in records]
        if len(updates) < epoch * 1481 or updates != list(range(1, len(updates) + 1)):
            raise ValueError('Historical accepted epoch lacks a complete unique update prefix')
    else:
        CheckRawUpdates(output, epoch)
    return gate


def CheckShortStageApproval(output, protocol):
    approval = Read(output / 'screen_acceptance.json')
    if (approval.get('b_continue_to48') is not True or approval.get('stage') != protocol['stage']
            or approval.get('source_sha256') != protocol['source_sha256']
            or approval.get('protocol_sha256') != HashFile(output / 'thesis_protocol.json')
            or approval.get('epoch5_checkpoint_sha256') != HashFile(output / 'checkpoints/epoch_5.pth')
            or approval.get('epoch5_validation_sha256') != HashFile(output / 'validation/epoch_5.json')):
        raise ValueError('Short-stage review does not authorize this exact continuation beyond epoch5')


def RunThesisEpoch(trainer, cfg, args):
    output = Path(trainer.checkpoint_dir).resolve().parent
    protocol = Read(output / 'thesis_protocol.json')
    if Read(CONTROL_PATH)['mode'] != 'active':
        raise RuntimeError('Manual pause/cancel prevents experiment advancement')
    if (output / 'failed_action.json').exists():
        raise RuntimeError('Previous failure needs explicit review')
    if protocol['config_identity'] != ConfigIdentity(args.config):
        raise ValueError('Config or labels differ from registered experiment')
    stage = cfg.thesis_v1.stage
    if stage != protocol['stage'] or stage not in ('s0', 's', 't', 's_no_root', 't_no_root'):
        raise ValueError('Unknown or inconsistent thesis stage')
    if cfg.trainer.max_epochs != 48 or cfg.trainer.eval_every_nepochs != 1:
        raise ValueError('Thesis training requires48 schedule and every-epoch validation')
    if trainer.lr_scheduler.total_steps != 71088 or len(trainer.train_dataloader) != 1481:
        raise ValueError('Full-pool eight-GPU optimizer budget changed')
    if not 0 <= trainer.epoch <= 48 or trainer.global_step != trainer.epoch * 1481:
        raise ValueError('Checkpoint is not a complete epoch boundary')
    model = getattr(trainer.model, 'module', trainer.model)
    model.config['validation_artifact_dir'] = str(output / 'validation_counts')
    if args.resume:
        expected = output / f'checkpoints/epoch_{trainer.epoch}.pth'
        if Path(args.resume).resolve() != expected.resolve():
            raise ValueError('Resume must use this experiment exact epoch checkpoint')
    elif stage == 's0':
        if args.init_checkpoint or trainer.epoch != 0:
            raise ValueError('S0 must start fresh; no old pretrained or temporal checkpoint')
    elif (not args.init_checkpoint or HashFile(args.init_checkpoint) != protocol['source_sha256']
          or Path(args.init_checkpoint).resolve() != Path(protocol['source']).resolve()):
        raise ValueError('All continuations must load the same accepted S0 checkpoint')
    if args.validate_only:
        if not args.resume and trainer.rank == 0:
            trainer.save_checkpoint('epoch_0.pth')
        trainer.val_epoch()
        return
    if not args.resume or args.stop_after_epoch != trainer.epoch + 1 or args.stop_after_epoch > 48:
        raise ValueError('Exactly one accepted new epoch may run per action')
    if stage != 's0' and args.stop_after_epoch > 5:
        CheckShortStageApproval(output, protocol)
    CheckAcceptance(output, trainer.epoch, args.config)
    trainer.train_epoch()
    if trainer.rank == 0:
        trainer.save_checkpoint()
        trainer.save_checkpoint('latest.pth')
    trainer.val_epoch()
