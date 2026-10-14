"""补框infos内容绑定及四组运行计划回归，不读取真实大数据。"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tools'))
from run_options import CheckCompletedDetectionManifest
from run_server_experiments import SCOPES, TrainCommand, EpochSteps


class CompletedDetectionBindingTest(unittest.TestCase):
    def test_content_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            info = root / 'new.pkl'
            info.write_bytes(b'original-boxes-and-supplements')
            entry = {'path': info.name, 'sha256': hashlib.sha256(info.read_bytes()).hexdigest()}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'b_complete': True, 'binding_sha256': 'test',
                                           'info_files': {'training': entry, 'validation': entry}}))
            dataset = {'b_use_completed_detection_labels': True, 'info_path': info.name}
            cfg = {'data': {'train_dataset': dataset, 'val_dataset': dict(dataset)},
                   'completed_detection_manifest': '$WAYMO_DATA_ROOT/manifest.json'}
            with patch.dict(os.environ, WAYMO_DATA_ROOT=directory):
                result = CheckCompletedDetectionManifest(cfg, root / 'config.yaml')
                self.assertEqual(result['infos']['train']['sha256'], entry['sha256'])
                info.write_bytes(b'changed-labels')
                with self.assertRaisesRegex(ValueError, 'content changed'):
                    CheckCompletedDetectionManifest(cfg, root / 'config.yaml')

    def test_legacy_config_has_no_new_requirement(self):
        self.assertIsNone(CheckCompletedDetectionManifest({'data': {'train_dataset': {}}}, 'unused.yaml'))
        with self.assertRaisesRegex(ValueError, 'explicit content manifest'):
            CheckCompletedDetectionManifest({'data': {'train_dataset': {'b_use_completed_detection_labels': True}}}, 'unused.yaml')

    def test_four_stage_budget_and_scratch_entry(self):
        stages = SCOPES['xyzt_completed']['stages']
        self.assertEqual(stages, (('single', False, None), ('temporal', True, 'single'),
                                  ('single_continuation', False, 'single'), ('temporal_scratch', True, None)))
        self.assertEqual(EpochSteps('xyzt_completed', 8), 1481)
        command = TrainCommand('temporal_scratch', True, Path('/tmp/test-run'), 8,
                               ['--stop-after-epoch', '5'], 'xyzt_completed')
        self.assertIn(str(PROJECT / 'LYF_train_ODSS_multi.py'), command)
        self.assertNotIn('--init-checkpoint', command)
        self.assertIn(str(PROJECT / 'configs/server_xyzt_completed_temporal_scratch.yaml'), command)


if __name__ == '__main__':
    unittest.main()
