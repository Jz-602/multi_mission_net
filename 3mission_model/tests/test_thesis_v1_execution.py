"""CPU验证论文runner的推进、恢复和证据门禁；不运行训练子进程或GPU。"""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import thesis_v1_runtime as runtime
from tools import run_thesis_v1 as runner
from run_options import HashFile

IDENTITY = {'config_sha256': 'fixed'}
SPLITS = {'train': {'count': 23688}, 'val': {'count': 5973}}


def Write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def Artifacts(output, epoch):
    image = output / f'images/epoch_{epoch:02d}/val_0000.png'
    image.parent.mkdir(parents=True, exist_ok=True)
    for path in (image, image.with_suffix('.pt'), image.with_name(image.stem + '_terrain_details.png')):
        path.write_bytes(b'evidence' * 200)
    return [str(image)]


def Evidence(output, epoch, b_accept=True):
    checkpoint = output / f'checkpoints/epoch_{epoch}.pth'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f'checkpoint{epoch}'.encode())
    validation = output / f'validation/epoch_{epoch}.json'
    Write(validation, {'epoch': epoch, 'metrics': {}})
    Write(output / 'manifest.json', {'splits': SPLITS})
    images = Artifacts(output, epoch)
    gate = dict(status='pass', epoch=epoch, checkpoint=str(checkpoint), checkpoint_sha256=HashFile(checkpoint),
                validation=str(validation), validation_sha256=HashFile(validation), split_signature=SPLITS,
                protocol_sha256=HashFile(output / 'thesis_protocol.json'), images=images)
    gate['image_artifacts'] = [dict(path=str(path), sha256=HashFile(path)) for image in images
                              for path in (Path(image), Path(image).with_suffix('.pt'),
                                           Path(image).with_name(Path(image).stem + '_terrain_details.png'))]
    if b_accept:
        Write(output / f'acceptance/epoch_{epoch}.json', gate)
    (output / 'train_metrics.jsonl').write_text(''.join(json.dumps({'iter': i}) + '\n' for i in range(1, epoch * 1481 + 1)))
    return gate


class Trainer:
    def __init__(self, output, epoch=0):
        self.checkpoint_dir = str(output / 'checkpoints')
        self.epoch, self.global_step, self.rank = epoch, epoch * 1481, 0
        self.lr_scheduler = SimpleNamespace(total_steps=71088)
        self.train_dataloader = range(1481)
        self.model = SimpleNamespace(config={})
        self.calls = []

    def save_checkpoint(self, name=None):
        self.calls.append(('save', name))

    def val_epoch(self):
        self.calls.append(('val', self.epoch))

    def train_epoch(self):
        self.calls.append(('train', self.epoch))
        self.epoch += 1
        self.global_step += 1481


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 's0'
        self.output.mkdir()
        self.control = self.root / 'control.json'
        Write(self.control, {'mode': 'active'})
        self.protocol = dict(stage='s0', config_identity=IDENTITY, source=None, source_sha256=None)
        Write(self.output / 'thesis_protocol.json', self.protocol)
        self.config = self.root / 'config.yaml'
        self.config.write_text('config')
        self.patches = [patch.object(runtime, 'CONTROL_PATH', self.control),
                        patch.object(runtime, 'ConfigIdentity', return_value=IDENTITY),
                        patch.object(runner, 'ConfigIdentity', return_value=IDENTITY)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def Args(self, epoch=0, b_resume=False, b_validate=False):
        return SimpleNamespace(config=str(self.config), resume=str(self.output / f'checkpoints/epoch_{epoch}.pth') if b_resume else None,
                               init_checkpoint=None, validate_only=b_validate, stop_after_epoch=epoch + 1)

    def Cfg(self, stage='s0'):
        return SimpleNamespace(thesis_v1=SimpleNamespace(stage=stage), trainer=SimpleNamespace(max_epochs=48, eval_every_nepochs=1))

    def TestRandomS0AndOneEpoch(self):
        trainer = Trainer(self.output)
        runtime.RunThesisEpoch(trainer, self.Cfg(), self.Args(b_validate=True))
        self.assertEqual(trainer.calls, [('save', 'epoch_0.pth'), ('val', 0)])
        Evidence(self.output, 0)
        runtime.RunThesisEpoch(trainer, self.Cfg(), self.Args(b_resume=True))
        self.assertEqual(trainer.calls[-4:], [('train', 0), ('save', None), ('save', 'latest.pth'), ('val', 1)])
        self.assertEqual(trainer.lr_scheduler.total_steps, 71088)
        for bad_stop in (1, 3, 49):
            args = self.Args(1, b_resume=True)
            args.stop_after_epoch = bad_stop
            with self.assertRaises(ValueError):
                runtime.RunThesisEpoch(trainer, self.Cfg(), args)
        trainer.lr_scheduler.total_steps = 5 * 1481
        with self.assertRaises(ValueError):
            runtime.RunThesisEpoch(trainer, self.Cfg(), self.Args(1, b_resume=True))

    def TestMissingValidationCheckpointAndImagesBlock(self):
        trainer = Trainer(self.output)
        with self.assertRaises(FileNotFoundError):
            runtime.RunThesisEpoch(trainer, self.Cfg(), self.Args(b_resume=True))
        self.assertEqual(trainer.calls, [])
        gate = Evidence(self.output, 0)
        runtime.CheckAcceptance(self.output, 0, self.config)
        image = Path(gate['images'][0])
        image.unlink()
        with self.assertRaises(FileNotFoundError):
            runtime.CheckAcceptance(self.output, 0, self.config)
        Artifacts(self.output, 0)
        (self.output / 'checkpoints/epoch_0.pth').write_bytes(b'altered')
        with self.assertRaises(ValueError):
            runtime.CheckAcceptance(self.output, 0, self.config)
        self.assertEqual(trainer.calls, [])

    def TestHistoricalAcceptanceAllowsOnlyCompleteUniquePrefix(self):
        Evidence(self.output, 0)
        Evidence(self.output, 1)
        runtime.CheckAcceptance(self.output, 0, self.config, b_allow_later_updates=True)
        with self.assertRaises(RuntimeError):
            runtime.CheckAcceptance(self.output, 0, self.config)
        with (self.output / 'train_metrics.jsonl').open('a') as stream:
            stream.write('{"iter":1481}\n')
        with self.assertRaises(ValueError):
            runtime.CheckAcceptance(self.output, 0, self.config, b_allow_later_updates=True)

    def TestContinuationsShareSourceAndFiveEpochGate(self):
        source = self.root / 'accepted_s0.pth'
        source.write_bytes(b'one common S0')
        for stage in ('s', 't', 's_no_root', 't_no_root'):
            protocol = dict(self.protocol, stage=stage, source=str(source), source_sha256=HashFile(source))
            Write(self.output / 'thesis_protocol.json', protocol)
            args = self.Args(b_validate=True)
            args.init_checkpoint = str(source)
            trainer = Trainer(self.output)
            runtime.RunThesisEpoch(trainer, self.Cfg(stage), args)
            self.assertEqual(trainer.calls, [('save', 'epoch_0.pth'), ('val', 0)])
            wrong = self.root / 'wrong.pth'
            wrong.write_bytes(b'wrong')
            args.init_checkpoint = str(wrong)
            with self.assertRaises(ValueError):
                runtime.RunThesisEpoch(Trainer(self.output), self.Cfg(stage), args)
        Evidence(self.output, 5)
        trainer = Trainer(self.output, 5)
        with self.assertRaises(FileNotFoundError):
            runtime.RunThesisEpoch(trainer, self.Cfg('t_no_root'), self.Args(5, b_resume=True))
        approval = dict(b_continue_to48=True, stage='t_no_root', source_sha256=HashFile(source),
                        protocol_sha256=HashFile(self.output / 'thesis_protocol.json'),
                        epoch5_checkpoint_sha256=HashFile(self.output / 'checkpoints/epoch_5.pth'),
                        epoch5_validation_sha256=HashFile(self.output / 'validation/epoch_5.json'))
        Write(self.output / 'screen_acceptance.json', dict(approval, b_continue_to48='true'))
        with self.assertRaises(ValueError):
            runtime.RunThesisEpoch(trainer, self.Cfg('t_no_root'), self.Args(5, b_resume=True))
        Write(self.output / 'screen_acceptance.json', approval)
        runtime.RunThesisEpoch(trainer, self.Cfg('t_no_root'), self.Args(5, b_resume=True))
        self.assertEqual(trainer.calls, [('train', 5), ('save', None), ('save', 'latest.pth'), ('val', 6)])

    def TestPauseFailureAndValidationOnlyRecovery(self):
        trainer = Trainer(self.output, 2)
        args = self.Args(2, b_resume=True, b_validate=True)
        Write(self.control, {'mode': 'paused'})
        with self.assertRaises(RuntimeError):
            runtime.RunThesisEpoch(trainer, self.Cfg(), args)
        Write(self.control, {'mode': 'active'})
        Write(self.output / 'failed_action.json', {'reason': 'previous error'})
        with self.assertRaises(RuntimeError):
            runtime.RunThesisEpoch(trainer, self.Cfg(), args)
        (self.output / 'failed_action.json').unlink()
        runtime.RunThesisEpoch(trainer, self.Cfg(), args)
        self.assertEqual(trainer.calls, [('val', 2)])

    def TestCheckEpochOptimizerAndSchedule(self):
        Evidence(self.output, 1)
        path = self.output / 'checkpoints/epoch_1.pth'
        checkpoint = dict(meta=dict(epoch=1, iter=1481), scheduler=dict(total_steps=71088, last_epoch=1481),
                          optimizer=dict(state={0: dict(step=torch.tensor(1481))}))
        torch.save(checkpoint, path)
        self.assertEqual(runner.CheckEpoch(self.output, 1), SPLITS)
        checkpoint['optimizer']['state'] = {}
        torch.save(checkpoint, path)
        with self.assertRaises(RuntimeError):
            runner.CheckEpoch(self.output, 1)
        checkpoint['scheduler']['total_steps'] = 7405
        torch.save(checkpoint, path)
        with self.assertRaises(RuntimeError):
            runner.CheckEpoch(self.output, 1)

    def TestRunnerFailureStopsAutomaticRetry(self):
        with patch.object(runner, 'OUTPUT', self.root), patch.object(runner, 'RunStage', side_effect=RuntimeError('mock failed')) as action:
            with self.assertRaises(RuntimeError):
                runner.Run('s0', 48)
            self.assertTrue((self.output / 'failed_action.json').exists())
            with self.assertRaises(RuntimeError):
                runner.Run('s0', 48)
            self.assertEqual(action.call_count, 1)

    def TestRegisterReadinessAndCommonS0(self):
        prep = self.root / 'prep'
        Write(prep / 'formal_readiness.json', dict(b_ready_for_s0=False, config_identities={name: IDENTITY for name in runner.CONFIGS}))
        (self.root / 'THESIS_V1_PROTOCOL.md').write_text('frozen specification')
        with patch.object(runner, 'PREPARATION', prep), patch.object(runner, 'PROJECT', self.root), patch.object(runner, 'OUTPUT', self.root):
            with self.assertRaises(RuntimeError):
                runner.Register('s0', self.output, self.config, 48)
            Write(prep / 'formal_readiness.json', dict(b_ready_for_s0=True, config_identities={name: IDENTITY for name in runner.CONFIGS}))
            gate = Evidence(self.output, 0)
            Write(self.output / 'final_acceptance.json', dict(b_s0_accepted=True, checkpoint=gate['checkpoint'], checkpoint_sha256=gate['checkpoint_sha256'],
                                                           validation=gate['validation'], validation_sha256=gate['validation_sha256']))
            for stage in ('s', 't', 's_no_root', 't_no_root'):
                output = self.root / stage
                output.mkdir()
                protocol, reference = runner.Register(stage, output, self.config, 5)
                self.assertEqual(protocol['source_sha256'], gate['checkpoint_sha256'])
                self.assertEqual(reference['epoch'], 0)
                with self.assertRaises(FileNotFoundError):
                    runner.Register(stage, output, self.config, 48)

    def TestFiguresMissingDetailsCannotPass(self):
        image = self.output / 'images/epoch_02/train_0010.png'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'x' * 2000)
        with patch.object(runner, 'Execute') as action:
            with self.assertRaises(FileNotFoundError):
                runner.Figures('s0', self.config, self.output, 2)
            action.assert_not_called()

    def TestRunnerResumeMissingValidationThenSkipsAcceptedHistory(self):
        protocol = dict(self.protocol, stage='s', source=str(self.root / 's0_source.pth'))
        Write(self.output / 'thesis_protocol.json', protocol)
        Evidence(self.output, 0, b_accept=False)
        (self.output / 'validation/epoch_0.json').unlink()
        actions = []

        def Execute(command, output, action):
            actions.append(command)
            epoch = int(action.split('_')[1])
            Evidence(output, epoch, b_accept=False)
            # 真实子进程写validation_metrics；runner负责创建validation/epoch_N绑定副本。
            (output / f'validation/epoch_{epoch}.json').unlink()
            Write(output / 'validation_metrics.json', {'epoch': epoch, 'metrics': {}})
            with (output / 'timings.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(action=action, elapsed_seconds=1)) + '\n')

        def Gate(record, reference, previous, stage):
            self.assertEqual(stage, 'S')
            self.assertEqual(previous['record']['epoch'] if previous else -1, record['epoch'] - 1)
            return dict(status='pass', epoch=record['epoch'], b_eligible=True, score=1.)

        with patch.object(runner, 'PROJECT', self.root), patch.object(runner, 'CheckActive'), \
                patch.object(runner, 'Register', return_value=(protocol, None)), \
                patch.object(runner, 'Execute', side_effect=Execute), \
                patch.object(runner, 'CheckEpoch', return_value=SPLITS), \
                patch.object(runner, 'EvaluateThesisGate', side_effect=Gate), \
                patch.object(runner, 'SelectThesisCheckpoint', return_value={'selected_epoch': 5}), \
                patch.object(runner, 'Figures', side_effect=lambda stage, config, output, epoch, **kw: Artifacts(output, epoch)):
            runner.RunStage('s', self.output, 5)
            self.assertEqual(len(actions), 6)
            self.assertIn('--resume', actions[0])
            self.assertIn('--validate-only', actions[0])
            self.assertNotIn('--stop-after-epoch', actions[0])
            self.assertEqual([int(command[-1]) for command in actions[1:]], [1, 2, 3, 4, 5])
            runner.RunStage('s', self.output, 5)
            self.assertEqual(len(actions), 6, 'Accepted completed history must not be trained or validated again')


if __name__ == '__main__':
    torch.set_num_threads(2)
    suite = unittest.TestLoader()
    suite.testMethodPrefix = 'Test'
    result = unittest.TextTestRunner(verbosity=2).run(suite.loadTestsFromTestCase(ExecutionTests))
    raise SystemExit(not result.wasSuccessful())
