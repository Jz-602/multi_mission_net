"""CPU Waymo metric regressions for empty GT/predictions and frame isolation."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.trainer.multitask_validator import (
    SummarizeCompletedDetectionGroundTruth,
    WaymoDetectionMetricsEstimator,
)


def TestEmptyFrames():
    # [N,7] boxes: vehicle-frame XYZ center, length/width/height, heading radians.
    box = np.array([0., 0., 0., 4., 2., 2., 0.], dtype=np.float32)
    empty_gt = {'gt_objects': []}
    object_gt = {'gt_objects': [{'box': box, 'type_name': 'vehicle',
                               'num_points_in_top_lidar': 10, 'difficulty': 0}]}
    empty_prediction = {'name': [], 'boxes_lidar': np.empty((0, 7), dtype=np.float32),
                        'score': np.empty(0, dtype=np.float32)}
    object_prediction = {'name': ['vehicle'], 'boxes_lidar': box[None],
                         'score': np.ones(1, dtype=np.float32)}
    gt_infos = [empty_gt, object_gt, empty_gt]
    estimator = WaymoDetectionMetricsEstimator()
    gt_arrays = estimator.GenerateWaymoTypeResults(gt_infos, ['vehicle'], b_is_gt=True)
    assert gt_arrays[0].tolist() == [1], 'Empty GT must preserve subsequent frame IDs'
    assert gt_arrays[1].shape == (1, 7)
    assert gt_arrays[-1].tolist() == [1], 'Point-count difficulty inference changed'
    for infos, b_is_gt in (([empty_gt], True), ([empty_prediction], False)):
        arrays = estimator.GenerateWaymoTypeResults(infos, ['vehicle'], b_is_gt=b_is_gt)
        assert arrays[1].shape == (0, 7)
        assert all(array.shape == (0,) for index, array in enumerate(arrays) if index != 1)

    metrics = WaymoDetectionMetricsEstimator().Evaluate(
        [empty_prediction, object_prediction, empty_prediction], gt_infos)
    for level in ('l1', 'l2'):
        for metric in ('ap', 'aph'):
            np.testing.assert_allclose(metrics[f'vehicle_{metric}_{level}'], 1., atol=2e-6)
    cases = (
        # Same box in a frame with empty GT must not match GT in frame 1.
        ([object_prediction, empty_prediction, empty_prediction], gt_infos),
        ([empty_prediction] * 3, gt_infos),
        ([object_prediction], [empty_gt]),
        ([empty_prediction], [empty_gt]),
    )
    for predictions, truth in cases:
        metrics = WaymoDetectionMetricsEstimator().Evaluate(predictions, truth)
        assert all(value == 0. for value in metrics.values()), metrics
    print('PASS CPU empty GT/prediction shapes, preserved frame IDs, self-match, '
          'empty-frame false positives cannot match another frame, all-empty inputs')


def TestCompletedGroundTruth():
    # [N,7] boxes: target-frame LiDAR XYZ, length/width/height, heading radians.
    boxes = np.array([[float(index * 10), 0., 0., 4., 2., 2., 0.]
                      for index in range(5)], dtype=np.float32)
    objects = [
        {'box': box, 'type_name': 'vehicle', 'num_points_in_top_lidar': count,
         'difficulty': 2 if b_supplemental else 0, 'b_is_supplemental': b_supplemental}
        for box, count, b_supplemental in zip(
            boxes, [0, 10, 0, 3, 8], [False, False, True, True, True])
    ]
    truth = [{'gt_objects': objects}]
    estimator = WaymoDetectionMetricsEstimator()
    old_arrays = estimator.GenerateWaymoTypeResults(truth, ['vehicle'], b_is_gt=True)
    new_arrays = estimator.GenerateWaymoTypeResults(
        truth, ['vehicle'], b_is_gt=True, b_use_completed_detection_labels=True)
    np.testing.assert_array_equal(old_arrays[1], boxes[[1, 3, 4]])
    np.testing.assert_array_equal(new_arrays[1], boxes[[1, 2, 3, 4]])
    np.testing.assert_array_equal(new_arrays[-1], [1, 2, 2, 2])
    assert [obj['num_points_in_top_lidar'] for obj in objects] == [0, 10, 0, 3, 8]
    assert [obj['difficulty'] for obj in objects] == [0, 0, 2, 2, 2]
    audit = SummarizeCompletedDetectionGroundTruth(truth)
    assert audit['original_count'] == 2 and audit['original_eligible_count'] == 1
    assert audit['supplemental_count'] == audit['supplemental_eligible_count'] == 3
    assert audit['supplemental_zero_point_count'] == 1
    assert audit['real_point_count_histogram']['supplemental'] == {'0': 1, '3': 1, '8': 1}

    # Run the installed TF/C++ operator: zero-point supplemental GT must become
    # a real L2 target; with the legacy gate it is absent and the prediction is FP.
    supplemental_truth = [{'gt_objects': [objects[2]]}]
    prediction = [{'name': ['vehicle'], 'boxes_lidar': boxes[2:3],
                   'score': np.ones(1, dtype=np.float32)}]
    original_metrics = WaymoDetectionMetricsEstimator().Evaluate(prediction, supplemental_truth)
    completed_metrics = WaymoDetectionMetricsEstimator().Evaluate(
        prediction, supplemental_truth, b_use_completed_detection_labels=True)
    assert original_metrics['vehicle_ap_l2'] == 0.
    np.testing.assert_allclose(completed_metrics['vehicle_ap_l2'], 1., atol=2e-6)
    np.testing.assert_allclose(completed_metrics['vehicle_aph_l2'], 1., atol=2e-6)
    # Installed Waymo matching can count a matched difficulty-2 GT as an L1 TP;
    # difficulty controls unmatched FN eligibility, not prefiltering matched GT.
    np.testing.assert_allclose(completed_metrics['vehicle_ap_l1'], 1., atol=2e-6)

    # A missed supplemental target must reduce L2 recall/AP, not just survive
    # serialization. Two original/supplemental targets in separate frames.
    two_truth = [{'gt_objects': [objects[1]]}, supplemental_truth[0]]
    two_prediction = [
        {'name': ['vehicle'], 'boxes_lidar': boxes[1:2], 'score': np.ones(1, dtype=np.float32)},
        {'name': [], 'boxes_lidar': np.empty((0, 7), dtype=np.float32),
         'score': np.empty(0, dtype=np.float32)},
    ]
    old_metrics = WaymoDetectionMetricsEstimator().Evaluate(two_prediction, two_truth)
    new_metrics = WaymoDetectionMetricsEstimator().Evaluate(
        two_prediction, two_truth, b_use_completed_detection_labels=True)
    np.testing.assert_allclose(old_metrics['vehicle_ap_l2'], 1., atol=2e-6)
    assert 0. < new_metrics['vehicle_ap_l2'] < old_metrics['vehicle_ap_l2']
    np.testing.assert_allclose(new_metrics['vehicle_ap_l1'], 1., atol=2e-6)
    print('PASS installed CPU Waymo op: supplemental zero real points is L2 GT, '
          'miss reduces AP, source/count audit preserved; disabled flag unchanged; '
          f'missed supplemental L2 AP={new_metrics["vehicle_ap_l2"]:.6f}')


if __name__ == '__main__':
    TestEmptyFrames()
    TestCompletedGroundTruth()
