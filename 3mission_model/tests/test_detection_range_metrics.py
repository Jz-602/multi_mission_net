"""Installed official Waymo CPU operator RANGE tests; no GPU or manual range cuts."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.trainer.multitask_validator import WaymoDetectionMetricsEstimator


def Frame(center, name='vehicle', b_prediction=True, b_supplemental=False):
    # [N,7]: target LiDAR XYZ center, L/W/H meters, yaw radians.
    box = np.array([*center, 4., 2., 2., 0.], dtype=np.float32)
    truth = {'gt_objects': [{'box': box, 'type_name': name,
                            'num_points_in_top_lidar': 0 if b_supplemental else 10,
                            'difficulty': 2 if b_supplemental else 1,
                            'b_is_supplemental': b_supplemental}]}
    prediction = {'name': [name] if b_prediction else [],
                  'boxes_lidar': box[None].copy() if b_prediction else np.empty((0, 7), np.float32),
                  'score': np.ones(1, np.float32) if b_prediction else np.empty(0, np.float32)}
    return prediction, truth


def Evaluate(predictions, truths, b_ranges):
    return WaymoDetectionMetricsEstimator().Evaluate(predictions, truths,
        b_use_completed_detection_labels=True, b_collect_detection_range_metrics=b_ranges)


class DetectionRangeTests(unittest.TestCase):
    def TestOfficialXYZRangeBoundariesAndMainUnchanged(self):
        predictions, truths = [], []
        # Isolated frames avoid matching ambiguity; z-only boundaries distinguish
        # true 3D norm from a BEV-only range implementation.
        centers = [(29.99, 0., 0.), (30., 0., 0.), (49.99, 0., 0.), (50., 0., 0.),
                   (0., 0., 30.), (18., 24., 0.), (0., 0., 50.)]
        for name in ('vehicle', 'pedestrian', 'cyclist'):
            for center in centers:
                prediction, truth = Frame(center, name)
                predictions.append(prediction); truths.append(truth)
        plain = Evaluate(predictions, truths, False)
        ranged = Evaluate(predictions, truths, True)
        self.assertEqual(plain, {key: ranged[key] for key in plain})
        for entry, support in zip(ranged['distance_metrics']['ranges'], (1, 4, 2)):
            self.assertEqual(len(entry['supported_classes']), 3)
            for values in entry['classes'].values():
                self.assertEqual(values['gt_count'], support)
                for level in (1, 2):
                    self.assertAlmostEqual(values[f'ap_l{level}'], 1., places=5)
                    self.assertAlmostEqual(values[f'aph_l{level}'], 1., places=5)
        # Test z alone with no other vehicle GT, checking RAW official range outputs.
        prediction, truth = Frame((0., 0., 30.))
        z_result = Evaluate([prediction], [truth], True)['distance_metrics']['ranges']
        self.assertEqual(z_result[0]['classes']['vehicle']['raw_operator_ap_l2'], 0.)
        self.assertAlmostEqual(z_result[1]['classes']['vehicle']['raw_operator_ap_l2'], 1., places=5)
        self.assertEqual(z_result[2]['classes']['vehicle']['raw_operator_ap_l2'], 0.)

    def TestEmptyRangeAndEmptyPredictionsHaveDifferentMeaning(self):
        prediction, truth = Frame((10., 0., 0.), b_prediction=False)
        plain = Evaluate([prediction], [truth], False)
        ranged = Evaluate([prediction], [truth], True)
        self.assertEqual(plain, {key: ranged[key] for key in plain})
        entries = ranged['distance_metrics']['ranges']
        self.assertEqual(entries[0]['classes']['vehicle']['ap_l2'], 0.)
        self.assertEqual(entries[0]['supported_classes'], ['vehicle'])
        self.assertIsNone(entries[0]['classes']['cyclist']['ap_l2'])
        self.assertIsNone(entries[2]['map_l2'])
        self.assertEqual(entries[2]['classes']['vehicle']['raw_operator_ap_l2'], 0.)
        all_empty = Evaluate([prediction], [{'gt_objects': []}], True)
        self.assertTrue(all(entry['map_l2'] is None for entry in all_empty['distance_metrics']['ranges']))
        self.assertEqual(all_empty['map_l2'], 0.)

    def TestBoundaryMatchAndSupplementalDifficultyPreserved(self):
        prediction, truth = Frame((29.95, 0., 0.), b_supplemental=True)
        prediction['boxes_lidar'][0, 0] = 30.05
        plain = Evaluate([prediction], [truth], False)
        ranged = Evaluate([prediction], [truth], True)
        self.assertEqual(plain, {key: ranged[key] for key in plain})
        self.assertAlmostEqual(ranged['vehicle_ap_l2'], 1., places=5)
        entries = ranged['distance_metrics']['ranges']
        self.assertEqual(entries[0]['classes']['vehicle']['gt_count'], 1)
        self.assertEqual(entries[0]['classes']['vehicle']['prediction_count'], 0)
        self.assertEqual(entries[1]['classes']['vehicle']['prediction_count'], 1)
        self.assertEqual(entries[0]['classes']['vehicle']['gt_difficulty_le_l1_count'], 0)
        self.assertEqual(entries[0]['classes']['vehicle']['gt_difficulty_le_l2_count'], 1)
        # No GT in midrange stays N/A even when predictions exist there.
        self.assertIsNone(entries[1]['classes']['vehicle']['ap_l2'])


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    result = unittest.TextTestRunner(verbosity=2).run(loader.loadTestsFromTestCase(DetectionRangeTests))
    sys.exit(0 if result.wasSuccessful() else 1)
