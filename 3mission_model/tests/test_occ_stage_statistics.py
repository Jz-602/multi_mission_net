"""CPU set accounting, boundaries, ignores, root supervision and token dedup."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occ_stage_statistics import FrameStatistics, StatisticsCollector, MergeStatistics
from occ_metrics import FrameMetrics


def Leaves(roots, shape=(6, 6, 6), b_expand=True):
    expanded = {tuple(np.array(root)+offset) for root in roots
                for offset in itertools.product((-1, 0, 1) if b_expand else (0,), repeat=3)
                if all(0 <= value < limit for value, limit in zip(np.array(root)+offset, shape))}
    xyz = np.array([np.array(root)*4+offset for root in sorted(expanded)
                    for offset in itertools.product(range(4), repeat=3)], dtype=np.int64).reshape(-1, 3)
    return np.array(sorted(expanded), dtype=np.int64).reshape(-1, 3), xyz


def Fixture():
    roots = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0], [3, 3, 3]])
    kept = roots[:1]
    expanded, leaves = Leaves(kept)
    gt = np.array([[1, 1, 1], [4, 0, 0], [5, 1, 1], [12, 0, 0], [20, 20, 20], [0, 4, 0], [0, 0, 0]])
    labels = np.array([1, 3, 3, 2, 1, 2, 255])
    prediction = np.zeros(len(leaves), dtype=np.int64)
    for xyz, label in [([1, 1, 1], 1), ([4, 0, 0], 2), ([0, 4, 0], 1), ([1, 0, 0], 3), ([0, 0, 0], 3)]:
        prediction[(leaves == xyz).all(axis=1)] = label
    return dict(target_id='fixed', pred_xyz=leaves, pred_labels=prediction, gt_xyz=gt, gt_labels=labels,
                grid_shape_xyz=[24]*3, root_xyz=roots, kept_root_xyz=kept, expanded_root_xyz=expanded,
                current_points_xyz=np.array([[.3, .3, .3], [.3, .3, .3], [.01, .01, .01], [4.8, 0, 0], [-.1, 0, 0]]),
                history_points_xyz=np.array([[.9, .1, .1], [2.5, .1, .1]]),
                pc_range=[0, 0, 0, 4.8, 4.8, 4.8], b_valid_roi_confirmed=True)


class OCCStatisticsTests(unittest.TestCase):
    def TestFourFatesAndExistingMetricAgreement(self):
        data = Fixture(); result = FrameStatistics(**data); c = result['counts']
        self.assertEqual(c['gt_fates'], [1, 1, 1, 3])
        self.assertEqual(sum(c['gt_fates']), c['gt'])
        self.assertEqual((c['occupancy_tp'], c['occupancy_fp'], c['occupancy_fn']), (3, 1, 3))
        self.assertEqual((c['candidate_gt_nonempty_correct'], c['candidate_gt_nonempty_wrong_nonempty']), (1, 2))
        # A deleted positive root's GT remains reachable from a retained neighbor:
        # two deleted positive roots are NOT two GT leaves lost to deletion.
        self.assertEqual(c['occupied_root_deleted'], 2)
        self.assertEqual(c['root_deletion_lost_gt'], 1)
        self.assertEqual(c['empty_root_deleted'], 1)
        self.assertEqual(c['expanded_root'], 8)  # Boundary clips 27 neighbors to eight.
        self.assertEqual(c['leaf_raw'], 512)
        self.assertEqual(c['leaf'], 511)  # Explicit ignore never becomes empty/FP.
        self.assertEqual(c['expansion_added_gt'], 1)
        self.assertEqual(c['input_leaf'], 1)
        self.assertEqual(c['input_gt'], 1)
        self.assertEqual(c['history_gt'], 2)
        self.assertEqual(c['history_current_gt'], 3)
        self.assertEqual(c['rejected_current_points'], 2)
        old = FrameMetrics(*[torch.tensor(data[k]) for k in ('pred_xyz','pred_labels','gt_xyz','gt_labels')],
                           data['grid_shape_xyz'], torch.tensor(data['root_xyz']), torch.tensor(data['kept_root_xyz']))
        self.assertEqual(c['full_confusion'], old['confusion'])
        for key in ('occupancy_tp','occupancy_fp','occupancy_fn','tp','fp','fn'):
            self.assertEqual(c[key], old[key])
        self.assertEqual(c['lplus_gt'], old['reachable_gt'])
        self.assertEqual(c['leaf_gt'], old['covered_gt'])
        self.assertEqual(c['root_deletion_lost_gt'], old['removed_gt'])

    def TestDuplicatesIgnoresInvalidDomainAndIncompleteGeometry(self):
        data = Fixture(); reference = FrameStatistics(**data)
        for coordinates, labels in [('pred_xyz','pred_labels'), ('gt_xyz','gt_labels')]:
            data[coordinates] = np.concatenate((data[coordinates], data[coordinates][:1]))
            data[labels] = np.concatenate((data[labels], data[labels][:1]))
        for key in ('root_xyz','kept_root_xyz','expanded_root_xyz'):
            data[key] = np.concatenate((data[key], data[key][:1]))
        self.assertEqual(FrameStatistics(**data), reference)
        data['gt_labels'][-1] = 2
        with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'): FrameStatistics(**data)
        data = Fixture(); data['b_valid_roi_confirmed'] = False
        with self.assertRaisesRegex(ValueError, 'confirm'): FrameStatistics(**data)
        data = Fixture(); data['pred_xyz'] = data['pred_xyz'][:-1]; data['pred_labels'] = data['pred_labels'][:-1]
        with self.assertRaisesRegex(ValueError, 'truncation'): FrameStatistics(**data)
        data = Fixture(); data['gt_xyz'][0] = [24, 0, 0]
        with self.assertRaisesRegex(ValueError, 'outside'): FrameStatistics(**data)

    def TestAllKeptNoExpandAndEmptyDenominators(self):
        data = Fixture(); data['kept_root_xyz'] = data['root_xyz']; data['b_expand_roots'] = False
        data['expanded_root_xyz'], data['pred_xyz'] = Leaves(data['root_xyz'], b_expand=False)
        data['pred_labels'] = np.zeros(len(data['pred_xyz']), dtype=np.int64)
        result = FrameStatistics(**data)
        self.assertEqual(result['counts']['deleted_root'], 0)
        self.assertEqual(result['counts']['root_deletion_lost_gt'], 0)
        self.assertEqual(result['counts']['expansion_added_leaf'], 0)
        self.assertIsNone(result['ratios']['deletion_precision']['value'])
        self.assertIsNone(result['ratios']['occupancy_precision']['value'])
        for key in ('gt_xyz','root_xyz','kept_root_xyz','expanded_root_xyz','pred_xyz','current_points_xyz','history_points_xyz'):
            data[key] = np.empty((0, 3), dtype=np.int64)
        data['gt_labels'] = data['pred_labels'] = np.empty(0, dtype=np.int64)
        result = FrameStatistics(**data)
        self.assertIsNone(result['ratios']['occupancy_iou']['value'])
        self.assertIsNone(result['ratios']['semantic_miou'])
        self.assertIsNone(result['ratios']['input_gt_coverage']['value'])
        self.assertEqual(result['counts']['full_confusion'][0][0], 24**3)

    def TestDisabledAndDeduplicatedAggregation(self):
        self.assertIsNone(StatisticsCollector().ComputeAndAdd(invalid=True))
        data = Fixture(); frame = FrameStatistics(**data)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'frames.jsonl'; collector = StatisticsCollector(True, path)
            self.assertTrue(collector.Add(frame)); self.assertFalse(collector.Add(frame))
            other = dict(frame, target_id='second'); collector.Add(other)
            summary = collector.Summary()
            self.assertEqual(summary['frame_count'], 2)
            self.assertEqual(summary['counts']['gt'], 12)
            self.assertEqual(summary['ratios']['occupancy_iou']['value'], 3/7)
            self.assertEqual(len(path.read_text().splitlines()), 2)
            merged = MergeStatistics([path, path], expected_target_ids=['fixed', 'second'])
            self.assertEqual(merged['frame_count'], 2)
            self.assertEqual(merged['duplicate_records_removed'], 2)
            self.assertEqual(merged['counts'], summary['counts'])
            with self.assertRaisesRegex(ValueError, 'fixed validation'):
                MergeStatistics([path], expected_target_ids=['fixed'])
            altered = json.loads(json.dumps(frame)); altered['counts']['gt'] = 99
            with self.assertRaisesRegex(ValueError, 'Conflicting duplicate'): collector.Add(altered)


if __name__ == '__main__':
    torch.set_num_threads(2)
    loader = unittest.TestLoader(); loader.testMethodPrefix = 'Test'
    unittest.main(testLoader=loader)
