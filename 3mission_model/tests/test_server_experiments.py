"""CPU checks for cohort isolation, recovery and actual-manifest launch guards."""
import contextlib
import copy
import io
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_server_experiments as runner
from build_label_manifest import BuildSplitManifest, BuildTargetFiles, LoadConfig


class ServerExperimentsTest(unittest.TestCase):
    def TestDynamicContinuationMatchesTemporalOptimization(self):
        configs = {stage: LoadConfig(runner.PROJECT_ROOT/'configs'/f'server_dynamic50_{stage}.yaml')
                   for stage in ('single', 'temporal', 'single_continuation')}
        for key in ('optimizer', 'scheduler'):
            self.assertEqual(configs['single_continuation'][key], configs['temporal'][key])
        for stage, config in configs.items():
            self.assertEqual(config['trainer']['max_epochs'], 48)
            self.assertTrue(config['data']['train_dataset']['b_dynamic_epoch_sampling'])
            self.assertEqual(config['data']['train_dataset']['sample_fraction'], 0.5)
            self.assertEqual(config['data']['train_dataset']['sample_seed'], 0)
            self.assertEqual(config['data']['val_dataset']['info_path'], configs['single']['data']['val_dataset']['info_path'])

    def TestDynamicPlanRequiresEightGpusAndOneExplicitStage(self):
        stream = io.StringIO()
        with patch.object(sys, 'argv', ['runner', '--scope', 'dynamic50', '--stage', 'single',
                                      '--gpus', '0,1,2,3,4,5,6,7', '--dry-run']), contextlib.redirect_stdout(stream):
            runner.Main()
        commands = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(commands), 18)
        trains = [command for command in commands if '--stop-after-epoch' in command]
        self.assertEqual([command[command.index('--nproc_per_node')+1] for command in trains], ['8']*3)
        self.assertTrue(all('server_dynamic50_single.yaml' in ' '.join(command) for command in commands))
        self.assertEqual(sum('--index' in command and command[command.index('--index')+1] == '2720'
                             for command in commands), 3)
        for arguments in (['--scope','dynamic50','--stage','single','--gpus','0'],
                          ['--scope','dynamic50','--gpus','0,1,2,3,4,5,6,7'],
                          ['--scope','fixed86','--gpus','0']):
            with patch.object(sys, 'argv', ['runner']+arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.Main()

    def TestDynamicEpochStepsUseSelectedCountInsteadOfCandidatePool(self):
        self.assertEqual(runner.SCOPES['dynamic50']['train_count'], 23688)
        self.assertEqual(runner.EpochSteps('dynamic50', 8), 1481)
        self.assertEqual(runner.EpochSteps('dynamic50', 8)*48, 71088)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root/'single'
            (output/'checkpoints').mkdir(parents=True)
            (output/'checkpoints/latest.pth').touch()
            (output/'train_metrics.jsonl').write_text(''.join(json.dumps({'iter':index})+'\n' for index in range(1,1482)))
            with patch.object(runner,'CheckpointMeta',return_value={'epoch':1,'iter':1481}), patch.object(runner,'CheckManifest'):
                self.assertEqual(runner.AuditStage(output,root,8,'dynamic50'),{'epoch':1,'iter':1481})
            with patch.object(runner,'CheckpointMeta',return_value={'epoch':1,'iter':2961}), patch.object(runner,'CheckManifest'):
                with self.assertRaisesRegex(RuntimeError,'固定预算'):
                    runner.AuditStage(output,root,8,'dynamic50')

    def TestSummarySeparatesDynamicCandidatesAndEpochUpdates(self):
        from summarize_server_experiments import SummarizeStage
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = {'splits': {'train': {'count': 23688}, 'val': {'count': 5973}},
                        'epoch_sampling': {'epoch_sample_fraction': 0.5}}
            (output/'manifest.json').write_text(json.dumps(manifest))
            tasks = [{'task': [task], 'loss': 1.0} for task in ('det', 'terrain', 'occ')]
            (output/'train_metrics.jsonl').write_text(''.join(
                json.dumps({'epoch': 0, 'iter': index, 'tasks': tasks})+'\n' for index in range(1, 1482)))
            summary = SummarizeStage(output, 8, {})
            self.assertEqual(summary['data_counts']['train'], 23688)
            self.assertEqual(summary['epoch_train_count'], 11844)
            self.assertEqual(summary['expected_updates_per_epoch'], 1481)
            self.assertTrue(summary['training_epochs'][0]['b_full_epoch_observed'])

    def TestFull50PlanAndInitialization(self):
        stream = io.StringIO()
        with patch.object(sys, 'argv', ['runner', '--scope', 'full50', '--gpus', '0,1,2,3', '--dry-run']), contextlib.redirect_stdout(stream):
            runner.Main()
        commands = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(len(commands), 30)
        trains = [command for command in commands if '--stop-after-epoch' in command]
        self.assertEqual([command[command.index('--stop-after-epoch') + 1] for command in trains], ['5', '25', '48'] * 2)
        for command in trains:
            self.assertIn('4', command)
            self.assertIn('server_full50_', command[command.index('--config') + 1])
        initial = trains[3]
        self.assertTrue(initial[initial.index('--init-checkpoint') + 1].endswith('server_runs_full50/single/checkpoints/epoch_48.pth'))
        self.assertNotIn('--init-checkpoint', trains[0])

    def TestLegacyExecutionSettingsResume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'execution_settings.json'
            settings = {'world_size': 1, 'phases': [5, 25, 48], 'stages': [item[0] for item in runner.STAGES]}
            path.write_text(json.dumps(settings))
            with patch.object(sys, 'argv', ['runner', '--scope', 'fixed86', '--allow-legacy', '--output-root', directory, '--gpus', '0']), patch.object(runner, 'AuditStage', return_value={'epoch': 48}), patch.object(runner, 'CompletePhase') as complete, patch.object(runner, 'Run') as run, patch.object(runner, 'AppendProgress'), patch.object(runner, 'CheckFull50Labels') as labels:
                runner.Main()
            self.assertEqual(complete.call_count, 15)
            run.assert_not_called()
            labels.assert_not_called()
            self.assertEqual(json.loads(path.read_text()), settings)

    def TestValidationCountIsScopeSpecific(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = {'epoch': 5, 'metrics': {'occ': {'frame_count': 5973}, 'terrain': {'frame_count': 5973}}}
            (output / 'validation_metrics.jsonl').write_text(json.dumps(record) + '\n')
            self.assertEqual(runner.ValidationRecords(output, 'full50'), [record])
            with self.assertRaisesRegex(RuntimeError, '520'):
                runner.ValidationRecords(output)
            record['metrics']['terrain']['frame_count'] = 5972
            (output / 'validation_metrics.jsonl').write_text(json.dumps(record) + '\n')
            with self.assertRaisesRegex(RuntimeError, '5973'):
                runner.ValidationRecords(output, 'full50')
            del record['metrics']['terrain']
            (output / 'validation_metrics.jsonl').write_text(json.dumps(record) + '\n')
            with self.assertRaisesRegex(RuntimeError, '5973'):
                runner.ValidationRecords(output, 'full50')

    def TestFull50RecoveryUses11844Targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'single'
            (output / 'checkpoints').mkdir(parents=True)
            (output / 'checkpoints' / 'latest.pth').touch()
            iterations = 11844 // 4
            (output / 'train_metrics.jsonl').write_text(''.join(json.dumps({'iter': index}) + '\n' for index in range(1, iterations + 1)))
            meta = {'epoch': 1, 'iter': iterations}
            with patch.object(runner, 'CheckpointMeta', return_value=meta), patch.object(runner, 'CheckManifest'):
                self.assertEqual(runner.AuditStage(output, root, 4, 'full50'), meta)
                with self.assertRaisesRegex(RuntimeError, '固定预算'):
                    runner.AuditStage(output, root, 4)

    def TestFull50ConfigsPreserveTrainingContract(self):
        for stage in ('single', 'temporal'):
            base = LoadConfig(runner.PROJECT_ROOT / 'configs' / f'server_{stage}.yaml')
            config = LoadConfig(runner.PROJECT_ROOT / 'configs' / f'server_full50_{stage}.yaml')
            for key in set(base) - {'data', 'label_manifest'}:
                self.assertEqual(config[key], base[key])
            self.assertEqual(config['data']['train_dataset']['sample_fraction'], 0.5)
            self.assertEqual(config['data']['train_dataset']['sample_seed'], 0)
            for split in ('train', 'val'):
                prior = copy.deepcopy(base['data'][f'{split}_dataset'])
                changed = copy.deepcopy(config['data'][f'{split}_dataset'])
                for key in ('info_path', 'full_info_path', 'sample_fraction', 'sample_seed'):
                    prior.pop(key, None)
                    changed.pop(key, None)
                self.assertEqual(changed, prior)
            if stage == 'temporal':
                self.assertEqual(config['data']['train_dataset']['full_info_path'], 'waymo_infos_training_point_only.pkl')
                self.assertEqual(config['data']['val_dataset']['full_info_path'], 'waymo_infos_validation_point_only.pkl')

    def TestMissingActualManifestBlocksBeforeOutputOrGpu(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'unused'
            with patch.object(sys, 'argv', ['runner', '--scope', 'full50', '--allow-legacy', '--output-root', str(output), '--gpus', '0']), patch.object(runner, 'CheckFull50Labels', side_effect=FileNotFoundError('actual missing')), patch.object(runner, 'Run') as run:
                with self.assertRaises(FileNotFoundError):
                    runner.Main()
            run.assert_not_called()
            self.assertFalse(output.exists())

    def TestPartialOrUnknownActualManifestRejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            manifest_path = root / 'actual.json'
            config = {'label_manifest': '../actual.json'}
            for value in ({'schema_version': 1, 'b_partial_manifest': True}, {'schema_version': 1}, {'schema_version': 2, 'b_partial_manifest': False}):
                manifest_path.write_text(json.dumps(value))
                with patch.object(runner, 'PROJECT_ROOT', root), patch('build_label_manifest.LoadConfig', return_value=config):
                    with self.assertRaisesRegex(RuntimeError, '未知/partial'):
                        runner.CheckFull50Labels()

    def TestActualContentRecordsAndMissingLabels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            config = {'label_manifest': '../actual.json', 'common_variables': {'root_path': str(root)}, 'data': {}}
            manifest = {'schema_version': 1, 'b_partial_manifest': False, 'data_root': str(root), 'splits': {}}
            for split in ('train', 'val'):
                info = {'token': split, 'sequence_name': split, 'frame_timestamp_micros': 1000}
                dataset = {'info_path': f'{split}.pkl'}
                config['data'][f'{split}_dataset'] = dataset
                (root / dataset['info_path']).write_bytes(pickle.dumps([info]))
                for _, filename in BuildTargetFiles(root, dataset, info)[2]:
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename.write_bytes(b'label')
                manifest['splits'][split] = BuildSplitManifest(root, dataset, {}, split, None)
            manifest_path = root / 'actual.json'
            manifest_path.write_text(json.dumps(manifest))
            scope = dict(runner.SCOPES['full50'], train_count=1, val_count=1)
            with patch.object(runner, 'PROJECT_ROOT', root), patch.dict(os.environ, {'WAYMO_DATA_ROOT': str(root)}), patch.dict(runner.SCOPES, {'full50': scope}), patch('build_label_manifest.LoadConfig', return_value=config):
                runner.CheckFull50Labels()
                manifest['splits']['train']['target_count'] = 2
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, '未覆盖固定train'):
                    runner.CheckFull50Labels()
                manifest['splits']['train']['target_count'] = 1
                manifest['splits']['train']['label_content_sha256'] = '0' * 64
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, '内容摘要不一致'):
                    runner.CheckFull50Labels()
                manifest['splits']['train'] = BuildSplitManifest(root, config['data']['train_dataset'], {}, 'train', None)
                manifest_path.write_text(json.dumps(manifest))
                filename.unlink()
                with self.assertRaisesRegex(RuntimeError, '缺失或大小改变'):
                    runner.CheckFull50Labels()


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    unittest.main(testLoader=loader)
