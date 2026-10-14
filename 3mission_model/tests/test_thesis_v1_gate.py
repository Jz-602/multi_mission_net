"""Pure CPU frozen-protocol health/selection tests using production metric layout."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
from test_detection_screen_gate import Record as DetectionRecord
from terrain_metrics import SummarizeTerrainValidity
from thesis_v1_gate import (EvaluateThesisGate, SelectThesisCheckpoint, EXPECTED_TOKEN_SHA256,
                            GT_VALID_CELLS, GT_HOLE_CELLS, OLD_MAP, OLD_OCC_IOU)


def Record(epoch=0, aps=(.6, .6, .6), root_recall=.9):
    record = DetectionRecord(aps, epoch)
    record['metrics']['validation_audit'] = dict(unique_target_count=5973, expected_target_count=5973,
        token_sha256=EXPECTED_TOKEN_SHA256, expected_token_sha256=EXPECTED_TOKEN_SHA256, b_complete=True)
    frame = dict(terrain_gt_valid_count=GT_VALID_CELLS,
                 terrain_unweighted_abs_error_sum=GT_VALID_CELLS * .08,
                 terrain_unweighted_squared_error_sum=GT_VALID_CELLS * .23 ** 2,
                 terrain_ignore_count=0, terrain_classification_count=GT_VALID_CELLS + GT_HOLE_CELLS,
                 terrain_validity_loss_sum=1000., terrain_validity_tp=int(GT_VALID_CELLS * .98),
                 terrain_validity_fn=GT_VALID_CELLS - int(GT_VALID_CELLS * .98),
                 terrain_validity_tn=int(GT_HOLE_CELLS * .8),
                 terrain_validity_fp=GT_HOLE_CELLS - int(GT_HOLE_CELLS * .8),
                 terrain_joint_10cm_count=int(GT_VALID_CELLS * .7),
                 terrain_joint_20cm_count=int(GT_VALID_CELLS * .8))
    record['metrics']['terrain'].update(SummarizeTerrainValidity([frame]))
    record['metrics']['occ'].update(root_count=2000, kept_root_count=1500,
                                   occupied_root_count=1000, occupied_root_kept=round(root_recall * 1000),
                                   occupied_root_recall=root_recall)
    return record


def Previous(record, reference=None, stage='S0', previous=None):
    return {'record': record, 'gate': EvaluateThesisGate(record, reference, previous, stage)}


class ThesisV1GateTests(unittest.TestCase):
    def TestS0EligibleAndFrozenScore(self):
        record = Record()
        result = EvaluateThesisGate(record, None, None, 'S0')
        self.assertTrue(result['b_healthy'], result['reasons'])
        self.assertTrue(result['b_eligible'], result['eligibility_reasons'])
        self.assertTrue(result['b_root_healthy_seen'])
        self.assertAlmostEqual(result['score'], (.6 / OLD_MAP + .4 / OLD_OCC_IOU + record['metrics']['terrain']['validity']['joint_20cm_recall']) / 3)

    def TestRandomS0NotPausedByPretrainedAPOrRoot(self):
        first = Record(aps=(0., 0., 0.), root_recall=1.)
        prior = Previous(first)
        second = Record(1, aps=(.01, 0., 0.), root_recall=.01)
        result = EvaluateThesisGate(second, None, prior, 'S0')
        self.assertTrue(result['b_healthy'], result['reasons'])
        self.assertFalse(result['b_root_healthy_seen'])
        self.assertFalse(result['b_eligible'])

    def TestS0ArmedRootGateRemainsArmedBelowEligibility(self):
        prior = Previous(Record(root_recall=.99))
        result = EvaluateThesisGate(Record(1, aps=(.01,) * 3, root_recall=.59), None, prior, 'S0')
        self.assertFalse(result['b_can_continue'])
        self.assertTrue(result['b_root_healthy_seen'])
        self.assertIn('occupied-root recall below', ' '.join(result['reasons']))

    def TestFirstEligibleS0StillNeedsHealthyRoot(self):
        result = EvaluateThesisGate(Record(root_recall=.59), None, None, 'S0')
        self.assertFalse(result['b_healthy'])
        self.assertFalse(result['b_eligible'])
        self.assertIn('first numerically eligible', ' '.join(result['reasons']))

    def TestCommonContinuationRuleAndNoShortBudgetGainRequirement(self):
        reference = Record()
        for stage in ('S', 'T', 'S-R', 'T-R'):
            current = Record()
            if stage.endswith('-R'):
                current['metrics']['occ'].update(kept_root_count=2000, occupied_root_kept=1000, occupied_root_recall=1.)
            result = EvaluateThesisGate(current, reference, None, stage)
            self.assertTrue(result['b_healthy'], result['reasons'])
            self.assertTrue(result['b_eligible'])
            self.assertAlmostEqual(result['score'], 1.)
            self.assertFalse(result['b_final_experiment_complete'])

    def TestFivePPBoundaryAndStricterEligibility(self):
        reference = Record()
        for ap, b_healthy, b_eligible in ((.55, False, False), (.55000001, True, False), (.58, True, True)):
            result = EvaluateThesisGate(Record(aps=(ap,) * 3), reference, None, 'T')
            self.assertEqual(result['b_healthy'], b_healthy, result)
            self.assertEqual(result['b_eligible'], b_eligible, result)

    def TestClassAndAPHCollapseCannotHideInMean(self):
        reference = Record()
        for aps, b_healthy in (((.95, .85, .119), False), ((.95, .85, .12), True)):
            result = EvaluateThesisGate(Record(aps=aps), reference, None, 'T')
            self.assertEqual(result['b_healthy'], b_healthy, result['reasons'])
        current = Record()
        det = current['metrics']['det']
        det['cyclist_aph_l2'] = .01
        det['maph_l2'] = sum(det[f'{name}_aph_l2'] for name in ('vehicle', 'pedestrian', 'cyclist')) / 3
        result = EvaluateThesisGate(current, reference, None, 'T')
        self.assertFalse(result['b_healthy'])
        self.assertIn('cyclist_aph_l2', ' '.join(result['reasons']))

    def TestRootRecallAndDropUnits(self):
        reference = Record(root_recall=.99)
        for recall, b_healthy in ((.74, True), (.739, False), (.59, False), (.60, False)):
            result = EvaluateThesisGate(Record(root_recall=recall), reference, None, 'T')
            self.assertEqual(result['b_healthy'], b_healthy, result['reasons'])
        result = EvaluateThesisGate(Record(root_recall=.60), Record(root_recall=.80), None, 'T')
        self.assertTrue(result['b_healthy'])

    def TestRootOffMustReallyKeepRoots(self):
        result = EvaluateThesisGate(Record(), Record(), None, 'T-R')
        self.assertFalse(result['b_healthy'])
        self.assertIn('did not retain all roots', ' '.join(result['reasons']))

    def TestProductionOptionalOCCStatsAllowUndefinedRatios(self):
        import torch
        from occ_head import GeneratedOCCHead
        from occ_metrics import SummarizeMetrics
        from test_occ_validation_stages import Evaluate, Fixture, Leaves
        torch.set_num_threads(2)
        data = Fixture()
        data['kept_root_xyz'] = data['root_xyz']
        data['expanded_root_xyz'], data['pred_xyz'] = Leaves(data['root_xyz'])
        import numpy as np
        data['pred_labels'] = np.zeros(len(data['pred_xyz']), dtype=np.int64)
        frame = Evaluate(data)
        frame['root_diagnostics'] = GeneratedOCCHead.RootGroupStatistics(torch.tensor([-2., -2.]), torch.tensor([0., 0.]))
        summary = SummarizeMetrics([frame])
        self.assertIsNone(summary['deletion_precision'])
        self.assertIsNone(summary['occupancy_precision'])
        self.assertEqual(summary['root_diagnostics']['groups']['empty']['count'], 0)
        current = Record(aps=(0.,) * 3, root_recall=1.)
        # Attach actual optional schemas to the complete synthetic full-val fixture;
        # toy counts themselves are not represented as 5973-frame evidence.
        for key in ('stages', 'root_diagnostics', 'deletion_precision', 'occupancy_precision'):
            current['metrics']['occ'][key] = summary[key]
        current['metrics']['occ']['kept_root_count'] = 2000
        current['metrics']['terrain']['distance_bins'] = {'empty_bin': {'mae_m': None, 'rmse_m': None}}
        result = EvaluateThesisGate(current, None, None, 'S0')
        self.assertTrue(result['b_healthy'], result['reasons'])

    def TestMalformedMissingAndNonfiniteNeverPass(self):
        cases = []
        current = Record(); current['metrics']['validation_audit']['unique_target_count'] = 5972; cases.append(current)
        current = Record(); current['metrics']['validation_audit']['token_sha256'] = 'wrong'; cases.append(current)
        current = Record(); current['metrics']['validation_audit']['b_complete'] = False; cases.append(current)
        current = Record(); current['metrics']['terrain'].pop('validity'); cases.append(current)
        current = Record(); current['metrics']['occ']['occupied_root_recall'] = float('nan'); cases.append(current)
        current = Record(); current['unrelated_diagnostic'] = float('inf'); cases.append(current)
        current = Record(); current['metrics']['terrain']['validity']['joint_20cm_recall'] /= 2; cases.append(current)
        current = Record(); current['metrics']['terrain']['validity']['validity_tp'] = 0; cases.append(current)
        current = Record(); current['metrics']['det']['ground_truth_audit']['frame_count'] = 5976; cases.append(current)
        for current in cases + [None, {}, {'epoch': 0, 'metrics': {}}]:
            with self.subTest(current=current):
                result = EvaluateThesisGate(current, Record(), None, 'T')
                self.assertFalse(result['b_healthy'])
                self.assertIsNone(result['score'])

    def TestGroundTruthCountsAndJointDenominatorFixed(self):
        current = Record()
        validity = current['metrics']['terrain']['validity']
        validity['joint_20cm_count'] = validity['validity_tp'] + 1
        self.assertFalse(EvaluateThesisGate(current, None, None, 'S0')['b_healthy'])
        current = Record(); current['metrics']['det']['ground_truth_audit']['supplemental_eligible_count'] += 1
        self.assertIn('GT mismatch', ' '.join(EvaluateThesisGate(current, Record(), None, 'T')['reasons']))

    def TestAllValidAndHeightLimitsCannotWinSelection(self):
        current = Record()
        v = current['metrics']['terrain']['validity']
        v.update(validity_tp=GT_VALID_CELLS, validity_fn=0, validity_fp=GT_HOLE_CELLS, validity_tn=0,
                 valid_recall=1., hole_iou=0., miou=GT_VALID_CELLS / (GT_VALID_CELLS + GT_HOLE_CELLS) / 2)
        result = EvaluateThesisGate(current, None, None, 'S0')
        self.assertTrue(result['b_healthy'], result['reasons'])
        self.assertFalse(result['b_eligible'])
        self.assertIn('hole IoU', ' '.join(result['eligibility_reasons']))
        for key, value in (('mae_m', .121), ('rmse_m', .321)):
            current = Record(); current['metrics']['terrain'][key] = value
            self.assertFalse(EvaluateThesisGate(current, None, None, 'S0')['b_eligible'])
        current = Record(); current['metrics']['terrain']['mae_m'] = .08801
        self.assertFalse(EvaluateThesisGate(current, Record(), None, 'T')['b_eligible'])

    def TestPreviousMustBeBoundAndDiagnosticsBlockProgress(self):
        self.assertFalse(EvaluateThesisGate(Record(1), Record(), None, 'T')['b_can_continue'])
        prior = Previous(Record(), Record(), 'T')
        wrong = copy.deepcopy(prior); wrong['gate']['stage'] = 'S'
        self.assertFalse(EvaluateThesisGate(Record(1), Record(), wrong, 'T')['b_healthy'])
        self.assertFalse(EvaluateThesisGate(Record(2), Record(), prior, 'T')['b_healthy'])
        wrong = copy.deepcopy(prior); wrong['gate']['status'] = 'diagnostic_required'; wrong['gate']['b_healthy'] = False
        self.assertFalse(EvaluateThesisGate(Record(1), Record(), wrong, 'T')['b_healthy'])
        broken = Record(1); broken['metrics']['terrain']['mae_m'] = float('nan')
        self.assertTrue(EvaluateThesisGate(broken, Record(), prior, 'T')['b_root_healthy_seen'])

    def TestSelectionSameCheckpointTieEarlyNoAutomaticVictory(self):
        records = [Record(index) for index in range(49)]
        original = copy.deepcopy(records)
        result = SelectThesisCheckpoint(records, None, 'S0')
        self.assertEqual(result['selected_epoch'], 0)
        self.assertTrue(result['b_complete_48_epochs'])
        self.assertFalse(result['evaluations'][-1]['b_can_continue'])
        self.assertFalse(result['b_final_experiment_complete'])
        self.assertEqual(records, original)
        self.assertEqual(SelectThesisCheckpoint([Record(0), Record(2)], None, 'S0')['status'], 'diagnostic_required')
        self.assertEqual(SelectThesisCheckpoint([Record(0), Record(0)], None, 'S0')['status'], 'diagnostic_required')
        self.assertEqual(SelectThesisCheckpoint([Record(aps=(0.,) * 3)], None, 'S0')['status'], 'no_eligible_checkpoint')
        self.assertFalse(EvaluateThesisGate(Record(49), Record(), None, 'T')['b_can_continue'])


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    result = unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromTestCase(ThesisV1GateTests))
    sys.exit(0 if result.wasSuccessful() else 1)
