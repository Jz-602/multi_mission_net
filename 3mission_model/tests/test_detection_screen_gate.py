"""Pure standard-library checks for fail-closed detection screening decisions."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from detection_screen_gate import EvaluateGate


def Record(aps=(0.6, 0.6, 0.6), epoch=0):
    det = {}
    for level in ('l1', 'l2'):
        for name, ap in zip(('vehicle', 'pedestrian', 'cyclist'), aps):
            det[f'{name}_ap_{level}'] = ap
            det[f'{name}_aph_{level}'] = ap * 0.95
        det[f'map_{level}'] = sum(aps) / 3
        det[f'maph_{level}'] = sum(ap * 0.95 for ap in aps) / 3
    det['ground_truth_audit'] = {
        'protocol': 'completed_detection_l2_v1', 'frame_count': 5973,
        'original_eligible_count': 71577, 'supplemental_eligible_count': 1783,
        'supplemental_zero_point_count': 1310,
        'eligible_by_class': {'vehicle': 44909, 'pedestrian': 27713, 'cyclist': 738},
    }
    return {'epoch': epoch, 'metrics': {
        'det': det, 'terrain': {'frame_count': 5973, 'mae_m': .08, 'rmse_m': .23},
        'occ': {'frame_count': 5973, 'occupancy_iou': .4, 'semantic_miou': .36},
    }}


class DetectionScreenGateTests(unittest.TestCase):
    def TestNormalAndNoFinalSuccess(self):
        result = EvaluateGate(Record(), Record((0.61, 0.59, 0.63), epoch=5))
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['action'], 'pass_to_next_epoch')
        self.assertTrue(result['b_can_continue'])
        self.assertFalse(result['b_final_experiment_complete'])

    def TestExactlyFivePercentagePointsAndJustBelow(self):
        result = EvaluateGate(Record(), Record((0.55,) * 3))
        self.assertEqual(result['status'], 'early_stop')
        self.assertFalse(result['b_can_continue'])
        self.assertAlmostEqual(result['map_drop_pp'], 5.)
        self.assertEqual(EvaluateGate(Record(), Record((0.55000001,) * 3))['status'], 'pass')
        # 5% relative of baseline .6 is .03, which is only 3 percentage points.
        self.assertEqual(EvaluateGate(Record(), Record((0.57,) * 3))['status'], 'pass')

    def TestNonfiniteAndMissingMetrics(self):
        for key in ('map_l2', 'vehicle_ap_l1', 'cyclist_aph_l2'):
            for bad in (float('nan'), float('inf'), None, '0.6'):
                current = Record(); current['metrics']['det'][key] = bad
                self.assertEqual(EvaluateGate(Record(), current)['status'], 'review_required')
        current = Record(); del current['metrics']['det']['cyclist_ap_l2']
        self.assertEqual(EvaluateGate(Record(), current)['status'], 'review_required')
        current = Record(); current['metrics']['occ']['occupancy_iou'] = float('nan')
        self.assertEqual(EvaluateGate(Record(), current)['status'], 'review_required')
        current = Record(); del current['metrics']['terrain']['mae_m']
        self.assertEqual(EvaluateGate(Record(), current)['status'], 'review_required')
        current = Record(); current['metrics']['det']['map_l2'] = .1
        self.assertEqual(EvaluateGate(Record(), current)['status'], 'review_required')

    def TestMissingFullTargetsAndProtocol(self):
        for task in ('det', 'terrain', 'occ'):
            for count in (None, 5972, 5973.0):
                current = Record()
                entry = (current['metrics']['det']['ground_truth_audit'] if task == 'det'
                         else current['metrics'][task])
                if count is None:
                    del entry['frame_count']
                else:
                    entry['frame_count'] = count
                self.assertEqual(EvaluateGate(Record(), current)['status'], 'review_required')
        current = Record(); del current['metrics']['det']['ground_truth_audit']
        self.assertEqual(EvaluateGate(Record(), current)['status'], 'review_required')
        current = Record(); current['metrics']['det']['ground_truth_audit']['supplemental_zero_point_count'] -= 1
        self.assertEqual(EvaluateGate(Record(), current)['status'], 'review_required')

    def TestClassCollapseEvenWithSmallMeanDrop(self):
        baseline = Record((.8, .6, .4))
        current = Record((.9, .72, .08), epoch=1)
        result = EvaluateGate(baseline, current)
        self.assertLess(result['map_drop_pp'], 5.)
        self.assertEqual(result['status'], 'early_stop')
        self.assertTrue(result['class_comparison']['cyclist']['b_collapse'])
        self.assertTrue(any('cyclist AP_L2 collapse' in reason for reason in result['reasons']))
        # Strictly below 25%, not <=25%; other classes offset aggregate drop.
        self.assertEqual(EvaluateGate(baseline, Record((.95, .75, .1)))['status'], 'pass')

    def TestMissingConfigurationAndInputImmutable(self):
        baseline, current = Record(), Record(epoch=1)
        saved = copy.deepcopy((baseline, current))
        for threshold, targets in ((None, 5973), (float('nan'), 5973), (0, 5973),
                                   (5, None), (5, 0), (True, 5973)):
            self.assertEqual(EvaluateGate(baseline, current, threshold, targets)['status'],
                             'review_required')
        self.assertEqual((baseline, current), saved)
        self.assertEqual(EvaluateGate({}, current)['status'], 'review_required')

    def TestCLIExitCodesAndActualEvidence(self):
        script = Path(__file__).resolve().parents[1] / 'tools/detection_screen_gate.py'
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / 'baseline.json'
            current = Path(directory) / 'current.json'
            output = Path(directory) / 'acceptance.json'
            base.write_text(json.dumps(Record()))
            for value, expected in ((Record(), 0), (Record((.55,) * 3), 3), ({}, 4)):
                current.write_text(json.dumps(value))
                result = subprocess.run([sys.executable, str(script), '--baseline', str(base),
                                         '--current', str(current), '--output', str(output)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)
                evidence = json.loads(output.read_text())
                self.assertEqual(evidence['b_can_continue'], expected == 0)
                self.assertFalse(evidence['b_final_experiment_complete'])
            current.unlink()
            result = subprocess.run([sys.executable, str(script), '--baseline', str(base),
                                     '--current', str(current), '--output', str(output)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 4)
            self.assertEqual(json.loads(output.read_text())['status'], 'review_required')


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    result = unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromTestCase(DetectionScreenGateTests))
    sys.exit(0 if result.wasSuccessful() else 1)
