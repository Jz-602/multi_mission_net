"""CPU mocks exercise orchestration; they do not claim CUDA/DDP/model validation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as Namespace
import unittest
from unittest.mock import patch

import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tools'))
from test_detection_screen_gate import Record
import run_detection_screen as runner
import detection_screen_runtime as runtime
from run_options import HashFile, WriteJsonAtomic


class ScreenRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.root = self.project / 'runs'
        self.source = self.project / 'source.pth'
        self.source.write_bytes(b'fixed initialization')
        self.control = self.project / 'log/server_supervision/control.json'
        self.control.parent.mkdir(parents=True)
        WriteJsonAtomic(self.control, {'mode': 'active'})
        (self.project / 'configs').mkdir()
        (self.project / 'tools').mkdir()
        self.calls = []
        self.aps = (.6, .6, .6)
        self.b_missing_metric = False
        self.b_failure = False
        self.splits = {'train': {'count': 23688, 'sha256': 'a' * 64},
                       'val': {'count': 5973, 'sha256': 'b' * 64}}
        self.cfg = {'detection_screen': {'max_new_epochs': 5, 'max_detection_drop_pp': 5.,
                                         'expected_targets': 5973},
                    'trainer': {'max_epochs': 48, 'eval_every_nepochs': 1}}
        for key in ('completed_detection_manifest', 'label_manifest'):
            filename = self.project / (key + '.json')
            filename.write_text('{}')
            self.cfg[key] = str(filename)
        for candidate in ('single', 'a', 'b'):
            (self.project / f'configs/server_x4_screen_{candidate}.yaml').write_text(yaml.safe_dump(self.cfg))
        for module, name, value in ((runner, 'PROJECT', self.project), (runner, 'OUTPUT_ROOT', self.root),
                                    (runner, 'SOURCE', self.source), (runner, 'SOURCE_SHA', HashFile(self.source)),
                                    (runtime, 'CONTROL_PATH', self.control)):
            item = patch.object(module, name, value); item.start(); self.addCleanup(item.stop)
        goal = patch.object(runner.subprocess, 'run', return_value=Namespace(stdout='{"status":"active"}'))
        goal.start(); self.addCleanup(goal.stop)
        execute = patch.object(runner, 'Execute', side_effect=self.FakeExecute)
        self.real_execute = runner.Execute
        self.execute = execute.start(); self.addCleanup(execute.stop)

    def SaveEpoch(self, output, epoch):
        (output / 'checkpoints').mkdir(parents=True, exist_ok=True)
        torch.save({'meta': {'epoch': epoch, 'iter': epoch * 1481}}, output / f'checkpoints/epoch_{epoch}.pth')
        (output / 'train_metrics.jsonl').write_text(''.join(json.dumps({'iter': i}) + '\n'
                                                          for i in range(1, epoch * 1481 + 1)))
        WriteJsonAtomic(output / 'manifest.json', {'splits': self.splits})

    def FakeExecute(self, command, output, action):
        runner.CheckActive()
        self.calls.append((action, command))
        if self.b_failure:
            raise RuntimeError('mock child failed')
        if action.endswith('_duplicates'):
            directory = Path(command[command.index('--directory') + 1])
            WriteJsonAtomic(directory / 'duplicate_diagnostics.json', {'records': [{}] * 5})
        elif '--split' in command:
            image = Path(command[command.index('--output') + 1])
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b'mock PNG, not real inference' * 100)
            if '--loss-log' in command:
                (image.parent / 'loss_curves.png').write_bytes(b'mock loss curve' * 100)
        else:
            epoch = int(action.split('_')[1])
            if '--stop-after-epoch' in command:
                self.assertEqual(int(command[command.index('--stop-after-epoch') + 1]), epoch)
                previous = runner.Read(output / f'acceptance/epoch_{epoch-1}.json')
                self.assertEqual(previous['status'], 'pass')
                runtime.CheckGateBinding(previous, output, self.project / f'configs/server_x4_screen_{output.name}.yaml')
            self.SaveEpoch(output, epoch)
            record = Record(self.aps, epoch)
            if self.b_missing_metric:
                del record['metrics']['det']['cyclist_ap_l2']
            WriteJsonAtomic(output / 'validation_metrics.json', record)
        with (output / 'timings.jsonl').open('a') as stream:
            stream.write(json.dumps({'action': action, 'elapsed_seconds': .1}) + '\n')

    def Baseline(self):
        output = self.root / 'single'
        (output / 'validation').mkdir(parents=True, exist_ok=True)
        WriteJsonAtomic(output / 'validation/epoch_0.json', Record())
        WriteJsonAtomic(output / 'manifest.json', {'splits': self.splits})

    def TestNormalZeroThroughFiveAndLossDiagnostics(self):
        runner.Run('single')
        actions = [(action, cmd) for action, cmd in self.calls if action in [f'epoch_{i}' for i in range(6)]]
        self.assertEqual([a for a, _ in actions], [f'epoch_{i}' for i in range(6)])
        self.assertIn('--validate-only', actions[0][1])
        self.assertEqual(sum('--stop-after-epoch' in cmd for _, cmd in actions), 5)
        output = self.root / 'single'
        self.assertFalse(runner.Read(output / 'stage_result.json')['b_effectiveness_confirmed'])
        for epoch in range(6):
            self.assertEqual(runner.Read(output / f'acceptance/epoch_{epoch}.json')['status'], 'pass')
            self.assertTrue((output / f'images/epoch_{epoch:02d}/duplicate_diagnostics.json').exists())
            if epoch:
                self.assertTrue((output / f'images/epoch_{epoch:02d}/loss_curves.png').exists())
        count = len(self.calls)
        with self.assertRaises(RuntimeError):
            runner.Run('single')
        self.assertEqual(count, len(self.calls))
        self.assertFalse((output / 'failed_action.json').exists())

    def TestEpochZeroEarlyStopAndMissingMetricsDoNotTrain(self):
        self.Baseline()
        self.aps = (.55,) * 3
        runner.Run('a')
        self.assertEqual(runner.Read(self.root / 'a/stage_result.json')['status'], 'early_stop')
        self.assertFalse(any('--stop-after-epoch' in cmd for _, cmd in self.calls))
        self.b_missing_metric = True
        runner.Run('b')
        self.assertEqual(runner.Read(self.root / 'b/stage_result.json')['status'], 'review_required')
        self.assertFalse(any('--stop-after-epoch' in cmd for _, cmd in self.calls))

    def TestFailurePersistsAndNoBlindRetry(self):
        self.b_failure = True
        with self.assertRaises(RuntimeError):
            runner.Run('single')
        self.assertTrue((self.root / 'single/failed_action.json').exists())
        count = len(self.calls)
        self.b_failure = False
        with self.assertRaises(RuntimeError):
            runner.Run('single')
        self.assertEqual(len(self.calls), count)

    def TestManualPauseAndNativeGoalFailClosed(self):
        for goal in ({'status': 'paused'}, None):
            with patch.object(runner.subprocess, 'run', return_value=Namespace(stdout=json.dumps(goal))):
                with self.assertRaises(runner.ManualPauseError):
                    runner.Run('single')
        with patch.object(runner.subprocess, 'run', side_effect=subprocess.TimeoutExpired('goal', 45)):
            with self.assertRaises(runner.ManualPauseError):
                runner.Run('single')
        WriteJsonAtomic(self.control, {'mode': 'paused'})
        with self.assertRaises(runner.ManualPauseError):
            runner.Run('single')
        self.assertEqual(self.calls, [])
        self.assertFalse((self.root / 'single/failed_action.json').exists())

    def TestSavedCheckpointRecoversValidationWithoutRepeatingTraining(self):
        output = self.root / 'single'
        self.SaveEpoch(output, 0)
        runner.Run('single')
        first = self.calls[0][1]
        self.assertIn('--resume', first)
        self.assertIn('--validate-only', first)
        self.assertNotIn('--init-checkpoint', first)
        # Actual CPU metadata/raw checks reject an ahead-of-checkpoint interrupted log.
        with (output / 'train_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'iter': 7406}) + '\n')
        with self.assertRaisesRegex(RuntimeError, 'ahead'):
            runner.CheckEpochEvidence(output, 5)
        torch.save({'meta': {'epoch': 5, 'iter': 7404}}, output / 'checkpoints/epoch_5.pth')
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            runner.CheckEpochEvidence(output, 5)

    def TestGateBindingRejectsChangedConfigValidationCheckpointAndTargets(self):
        runner.Run('single')
        output = self.root / 'single'
        gate = runner.Read(output / 'acceptance/epoch_5.json')
        config = self.project / 'configs/server_x4_screen_single.yaml'
        for filename in (config, output / 'validation/epoch_5.json', output / 'checkpoints/epoch_5.pth',
                         self.project / 'completed_detection_manifest.json'):
            original = filename.read_bytes()
            filename.write_bytes(original + (b'\nchanged_model_flag: true\n' if filename == config else b'\n#changed'))
            with self.assertRaises((ValueError, RuntimeError)):
                runtime.CheckGateBinding(gate, output, config)
            filename.write_bytes(original)
        manifest = runner.Read(output / 'manifest.json')
        manifest['splits']['val']['sha256'] = 'c' * 64
        WriteJsonAtomic(output / 'manifest.json', manifest)
        with self.assertRaisesRegex(ValueError, 'targets changed'):
            runtime.CheckGateBinding(gate, output, config)

    def TestRuntimeHardCapAndPauseBeforeValidation(self):
        runner.Run('single')
        output = self.root / 'single'
        events = []
        trainer = Namespace(epoch=5, rank=0, checkpoint_dir=str(output / 'checkpoints'),
                            fit=lambda: events.append('fit'), val_epoch=lambda: events.append('val'),
                            save_checkpoint=lambda name: events.append(name))
        cfg = Namespace(detection_screen=Namespace(**self.cfg['detection_screen']),
                        trainer=Namespace(**self.cfg['trainer']))
        args = Namespace(config=str(self.project / 'configs/server_x4_screen_single.yaml'),
                         init_checkpoint=None, resume=str(output / 'checkpoints/epoch_5.pth'),
                         validate_only=False, stop_after_epoch=6)
        with self.assertRaisesRegex(ValueError, 'capped at five'):
            runtime.RunScreenStep(trainer, cfg, args)
        trainer.epoch = 6
        with self.assertRaisesRegex(ValueError, 'budget'):
            runtime.RunScreenStep(trainer, cfg, args)
        trainer.epoch = 5
        args.validate_only = True
        WriteJsonAtomic(self.control, {'mode': 'paused'})
        with self.assertRaisesRegex(RuntimeError, 'pause'):
            runtime.RunScreenStep(trainer, cfg, args)
        self.assertEqual(events, [])

    def TestChildExitAndInterruptPreserveFailureEvidence(self):
        output = self.root / 'single'
        output.mkdir(parents=True)
        process = Namespace(pid=123456, wait=lambda: 7)
        with patch.object(runner.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'exit=7'):
                self.real_execute(['mock-trainer'], output, 'epoch_1')
        self.assertEqual(json.loads((output / 'timings.jsonl').read_text())['returncode'], 7)
        self.assertEqual(runner.Read(output / 'process.json')['pid'], 123456)
        from unittest.mock import Mock
        process.wait = Mock(side_effect=[KeyboardInterrupt(), 0])
        with patch.object(runner.subprocess, 'Popen', return_value=process), patch.object(runner.os, 'killpg') as kill:
            with self.assertRaises(KeyboardInterrupt):
                self.real_execute(['mock-trainer'], output, 'epoch_1_recovery')
            kill.assert_called_once_with(123456, runner.signal.SIGTERM)

    def TestRuntimeAcceptedOneEpochAndRejectSkippedEpoch(self):
        runner.Run('single')
        output = self.root / 'single'
        events = []
        trainer = Namespace(epoch=4, rank=0, checkpoint_dir=str(output / 'checkpoints'),
                            train_epoch=lambda: None, fit=lambda: events.append('fit'))
        cfg = Namespace(detection_screen=Namespace(**self.cfg['detection_screen']),
                        trainer=Namespace(**self.cfg['trainer']))
        args = Namespace(config=str(self.project / 'configs/server_x4_screen_single.yaml'),
                         init_checkpoint=None, resume=str(output / 'checkpoints/epoch_4.pth'),
                         validate_only=False, stop_after_epoch=5)
        with self.assertRaisesRegex(RuntimeError, 'ahead'):
            runtime.RunScreenStep(trainer, cfg, args)
        raw = output / 'train_metrics.jsonl'
        raw.write_text('\n'.join(raw.read_text().splitlines()[:4 * 1481]) + '\n')
        runtime.RunScreenStep(trainer, cfg, args)
        self.assertEqual(events, ['fit'])
        self.assertEqual(trainer.max_epochs, 5)
        trainer.epoch = 0
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            runtime.RunScreenStep(trainer, cfg, args)


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    result = unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromTestCase(ScreenRunnerTests))
    sys.exit(0 if result.wasSuccessful() else 1)
