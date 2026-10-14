"""Every-validation OCC stages share the audited offline counter and token identity."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occ_metrics import FrameMetrics, SummarizeMetrics
from test_occ_stage_statistics import Fixture, Leaves


def Evaluate(data, b_enabled=True):
    return FrameMetrics(*[torch.tensor(data[key]) for key in ('pred_xyz','pred_labels','gt_xyz','gt_labels')],
        data['grid_shape_xyz'], torch.tensor(data['root_xyz']), torch.tensor(data['kept_root_xyz']),
        data.get('b_expand_roots', True), b_collect_occ_stage_statistics=b_enabled,
        current_points_xyz=torch.tensor(data['current_points_xyz']), pc_range=data['pc_range'],
        expanded_root_xyz=torch.tensor(data['expanded_root_xyz']), target_id=data['target_id'],
        b_valid_roi_confirmed=data['b_valid_roi_confirmed'])


class OCCValidationStageTests(unittest.TestCase):
    def test_four_fates_observed_groups_and_wrong_semantics(self):
        result = Evaluate(Fixture())
        stage = result['stages']; counts = stage['counts']
        self.assertEqual(counts['gt_fates'], [1,1,1,3])
        self.assertEqual(counts['observed_gt'], 1)
        self.assertEqual(counts['observed_gt_tp'], 1)
        self.assertEqual(counts['unobserved_gt'], 5)
        self.assertEqual(counts['unobserved_gt_tp'], 2)
        self.assertEqual(stage['ratios']['observed_gt_recall']['value'], 1.)
        self.assertEqual(stage['ratios']['unobserved_gt_recall']['value'], .4)
        self.assertFalse(stage['b_history_observation_collected'])
        self.assertNotIn('history_leaf', counts)
        self.assertNotIn('history_current_gt_coverage', stage['ratios'])
        self.assertEqual(counts['occupied_root_deleted'], 2)
        self.assertEqual(counts['root_deletion_lost_gt'], 1)
        self.assertEqual(counts['candidate_gt_nonempty_correct'], 1)
        self.assertEqual(counts['candidate_gt_nonempty_wrong_nonempty'], 2)
        self.assertEqual(counts['leaf'], 511)
        self.assertEqual(counts['full_confusion'], result['confusion'])

    def test_root_off_and_boundary_counts(self):
        data = Fixture(); data['kept_root_xyz'] = data['root_xyz']
        data['expanded_root_xyz'], data['pred_xyz'] = Leaves(data['root_xyz'])
        data['pred_labels'] = np.zeros(len(data['pred_xyz']), dtype=np.int64)
        result = Evaluate(data); counts = result['stages']['counts']
        self.assertEqual(counts['gt_fates'], [1,0,5,0])
        self.assertEqual(counts['root_deletion_lost_gt'], 0)
        self.assertEqual(counts['lplus'], counts['leaf'])
        self.assertEqual(counts['leaf_raw'], 64*counts['expanded_root'])
        self.assertEqual(counts['gt'], sum(counts['gt_fates']))

    def test_global_sum_after_token_dedup_and_disabled_path(self):
        first = Evaluate(Fixture()); data = Fixture(); data['target_id'] = 'second'
        second = Evaluate(data)
        records = {'fixed':first, 'second':second}; records['second'] = second
        summary = SummarizeMetrics(records.values())
        self.assertEqual(summary['frame_count'], 2)
        self.assertEqual(summary['stages']['frame_count'], 2)
        self.assertEqual(summary['stages']['counts']['gt'], 12)
        self.assertEqual(summary['stages']['ratios']['unobserved_gt_recall']['value'], .4)
        self.assertEqual(summary['occupancy_precision'], .75)
        self.assertEqual(summary['occupancy_recall'], .5)
        self.assertEqual(summary['stages']['counts']['full_confusion'], summary['confusion'])
        self.assertGreater(summary['stage_statistics_overhead']['mean_per_target_seconds'], 0.)
        with self.assertRaisesRegex(ValueError, 'Deduplicate'):
            SummarizeMetrics([first, first])
        disabled = Evaluate(data, False)
        self.assertNotIn('stages', disabled)
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            SummarizeMetrics([first, disabled])

    def test_root_diagnostics_weighted_means_not_mean_of_frames(self):
        from occ_head import GeneratedOCCHead
        first = Evaluate(Fixture(), False)
        second = Evaluate(Fixture(), False)
        first['root_diagnostics'] = GeneratedOCCHead.RootGroupStatistics(torch.tensor([0., 1.]), torch.tensor([0., 1.]))
        second['root_diagnostics'] = GeneratedOCCHead.RootGroupStatistics(torch.tensor([2., 2., 2.]), torch.tensor([0., 0., 0.]))
        summary = SummarizeMetrics([first, second])['root_diagnostics']
        self.assertEqual(summary['groups']['positive']['count'], 4)
        self.assertEqual(summary['groups']['positive']['logit_mean'], 1.5)
        self.assertEqual(summary['groups']['empty']['count'], 1)
        self.assertEqual(summary['groups']['empty']['logit_mean'], 1.)
        self.assertNotIn('logit_p50', summary['groups']['positive'])
        self.assertIn('root_positive_logit_p50', first['root_diagnostics'])

    def test_actual_export_wiring_and_training_stage_off(self):
        from test_occ_head import MakeHead, Batch
        head = MakeHead().eval()
        head.b_collect_occ_stage_statistics = True
        head.b_collect_root_diagnostics = True
        roots = np.array([[0,0,0], [1,0,0]])
        expanded, leaves = Leaves(roots[:1], shape=(4,4,4))
        from occ_metrics import BZYX
        prediction = {
            'indices': BZYX(torch.tensor(leaves)), 'logits': torch.zeros(len(leaves),4),
            'root_indices': BZYX(torch.tensor(roots)), 'root_logits': torch.tensor([-2.,2.]),
            'root_keep_mask': torch.tensor([True,False]), 'root_spatial_shape': (4,4,4),
            'expanded_root_indices': BZYX(torch.tensor(expanded)), 'batch_size': 1,
        }
        batch = Batch([[1,1,1],[4,0,0]], [1,3])
        batch.update(token=['real_export_toy'], points=torch.tensor([[0.,.3,.3,.3,0.]]))
        result = head.ExportPredictions(prediction,batch)[0]['metrics']
        self.assertIn('stages', result)
        self.assertEqual(result['root_diagnostics']['root_positive_count'], 2)
        self.assertEqual(result['root_diagnostics']['root_empty_count'], 0)
        self.assertEqual(result['root_diagnostics']['root_positive_logit_sum'], 0.)
        self.assertEqual(result['stages']['counts']['occupied_root_deleted'], 1)
        self.assertEqual(result['stages']['counts']['root_deletion_lost_gt'], 0)
        head.train()
        self.assertNotIn('stages', head.ExportPredictions(prediction,batch)[0]['metrics'])
        head.eval(); head.b_collect_occ_stage_statistics = False
        self.assertNotIn('stages', head.ExportPredictions(prediction,batch)[0]['metrics'])

    def test_coordinate_duplicates_ignored_and_invalid_roi_rejected(self):
        data = Fixture(); reference = Evaluate(data)
        for key in ('root_xyz','kept_root_xyz','expanded_root_xyz'):
            data[key] = np.concatenate((data[key], data[key][:1]))
        duplicate = Evaluate(data)
        self.assertEqual(duplicate['stages'], reference['stages'])
        self.assertEqual(duplicate['root_count'], reference['root_count'])
        data['b_valid_roi_confirmed'] = False
        with self.assertRaisesRegex(ValueError, 'confirm'):
            Evaluate(data)
        data = Fixture(); data['pred_xyz'] = data['pred_xyz'][:-1]; data['pred_labels'] = data['pred_labels'][:-1]
        with self.assertRaisesRegex(ValueError, 'truncation'):
            Evaluate(data)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
