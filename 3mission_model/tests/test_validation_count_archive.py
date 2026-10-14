"""完整验证的顺序绑定、去重计数和可重算归档，不运行GPU。"""
import gzip
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.trainer import multitask_validator as validator
from occ_metrics import FrameMetrics


class ValidationArchiveTest(unittest.TestCase):
    def test_ordered_binding_and_recomputable_counts(self):
        batch = dict(terrain_height=torch.zeros(1, 1, 2, 2),
                     terrain_valid_mask=torch.ones(1, 1, 2, 2, dtype=torch.bool),
                     terrain_train_weight=torch.ones(1, 1, 2, 2),
                     terrain_inpaint_distance=torch.zeros(1, 1, 2, 2))
        terrain = validator.BuildTerrainFrameStatistics(
            dict(height=torch.zeros(1, 2, 2), validity_logits=torch.ones(1, 2, 2)), batch, 0)
        xyz, labels = torch.tensor([[0, 0, 0]]), torch.tensor([1])
        occ = FrameMetrics(xyz, labels, xyz, labels, (4, 4, 4), xyz, xyz)
        records = {token: dict(terrain=terrain, occ=occ) for token in ('a', 'b')}
        dataset = SimpleNamespace(infos=[{'token': 'b'}, {'token': 'a'}],
                                  b_use_completed_detection_labels=False)
        det = {f'{category}_{metric}_{level}': .5 for category in ('vehicle', 'pedestrian', 'cyclist')
               for metric in ('ap', 'aph') for level in ('l1', 'l2')}
        det.update({f'{metric}_{level}': .5 for metric in ('map', 'maph') for level in ('l1', 'l2')})
        infos = ([{'score': np.array([.5])}] * 2, [{'gt_objects': []}] * 2)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(validator, '_BuildDetectionInfos', return_value=infos), \
                patch.object(validator.WaymoDetectionMetricsEstimator, 'Evaluate', return_value=det):
            config = EasyDict(point_cloud_range=[0, 0, 0, 1, 1, 1], validation_artifact_dir=directory)
            result = validator.FinalizeThreeTaskMetrics(records, dataset, config, 1, logging.getLogger())
            audit = result['validation_audit']
            self.assertTrue(audit['b_complete'])
            self.assertEqual(audit['unique_target_count'], 2)
            self.assertEqual(audit['token_sha256'], hashlib.sha256(b'b\na').hexdigest())
            with gzip.open(audit['per_frame_counts'], 'rt') as stream:
                saved = [json.loads(line) for line in stream]
            self.assertEqual([row['token'] for row in saved], ['b', 'a'])
            self.assertEqual(validator.SummarizeTerrainMetrics(row['terrain'] for row in saved), result['terrain'])
            self.assertEqual(validator.SummarizeOCCMetrics(row['occ'] for row in saved), result['occ'])
            del records['a']
            result = validator.FinalizeThreeTaskMetrics(records, dataset, config, 2, logging.getLogger())
            self.assertFalse(result['validation_audit']['b_complete'])
            self.assertEqual(result['validation_audit']['missing_count'], 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
