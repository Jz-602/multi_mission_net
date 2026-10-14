"""CPU checks for target-relative XYZT and supplemental-box supervision."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))
from WaymoDatasetV2 import BaseDataset, WaymoDataset
from WaymoDataset_multi import ClipAugmentationParams, WaymoDatasetMulti
from multi_mission_model import BuildBatchVoxels


class XYZTInputTests(unittest.TestCase):
    def MakeDataset(self, b_temporal=True, b_time=True):
        dataset = (WaymoDatasetMulti if b_temporal else WaymoDataset).__new__(
            WaymoDatasetMulti if b_temporal else WaymoDataset
        )
        dataset.b_use_time_feature = b_time
        dataset.b_use_completed_detection_labels = False
        dataset.loading_pipelines = ['LoadTestPoints']
        dataset.b_load_three_task_labels = False
        dataset.sampler = None
        dataset.augmentations = {}
        dataset.prepare_label = {}
        self.shared_points = np.array([[.11, .22, .33], [.44, .55, .66]], dtype=np.float32)

        def LoadTestPoints(res, info):
            res['points'] = self.shared_points
            return res

        dataset.LoadTestPoints = LoadTestPoints
        base_micros = 1_550_000_000_000_001
        dataset.infos = [dict(token=f'seq-{i}', sample_idx=i,
                              frame_timestamp_micros=base_micros + offset,
                              pose=np.eye(4, dtype=np.float32))
                         for i, offset in enumerate([0, 100_013, 310_019])]
        if b_temporal:
            dataset.warmup_len = 3
            dataset.train_len = 1
            dataset.seq_len = 4
            dataset.b_random_history_interval = False
            dataset.eval_history_interval = 3
            dataset.b_shared_clip_augmentation = True
            dataset._cfg_augmentation_items = []
            dataset.full_sequence_info_dict = {'seq': dataset.infos}
            dataset.anchor_list = [dict(sequence_name='seq', target_full_idx=i,
                                        anchor_info=dataset.infos[i]) for i in [1, 2]]
        return dataset

    def TestIrregularMicrosecondTimingPaddingAndClipIsolation(self):
        dataset = self.MakeDataset()
        original = self.shared_points.copy()
        first = dataset[0]
        second = dataset[1]
        np.testing.assert_array_equal(first['full_frame_indices'], [0, 0, 0, 1])
        np.testing.assert_array_equal(second['full_frame_indices'], [0, 0, 1, 2])
        for sample, expected in [(first, [-.100013, -.100013, -.100013, 0.]),
                                 (second, [-.310019, -.310019, -.210006, 0.])]:
            for frame, delta in zip(sample['frames'], expected):
                self.assertEqual(frame['points'].dtype, np.float32)
                np.testing.assert_array_equal(frame['points'][:, :3], original)
                np.testing.assert_allclose(frame['points'][:, 3], delta, rtol=0, atol=2e-8)
        np.testing.assert_array_equal(self.shared_points, original)
        np.testing.assert_array_equal(dataset[0]['frames'][0]['points'], first['frames'][0]['points'])

    def TestSpatialAugmentationDoesNotTransformTime(self):
        dataset = self.MakeDataset()
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] *= 2.
        matrix[:3, 3] = [.5, 1., -1.]
        dataset.BuildClipAugmentationList = lambda: [ClipAugmentationParams(aug_matrix=matrix)] * 4
        sample = dataset[1]
        for frame, delta in zip(sample['frames'], [-.310019, -.310019, -.210006, 0.]):
            np.testing.assert_allclose(frame['points'][:, :3], self.shared_points * 2 + matrix[:3, 3])
            np.testing.assert_allclose(frame['points'][:, 3], delta, rtol=0, atol=2e-8)

    def TestSingleZeroAndDisabledLegacyEquivalence(self):
        single = self.MakeDataset(b_temporal=False)
        np.testing.assert_array_equal(single[1]['points'][:, 3], [0., 0.])
        single.b_use_time_feature = False
        np.testing.assert_array_equal(single[1]['points'], self.shared_points)
        temporal = self.MakeDataset(b_time=False)
        for frame in temporal[0]['frames']:
            np.testing.assert_array_equal(frame['points'], self.shared_points)

    def TestCollateAndVoxelizationKeepTimeWithoutFourthSpatialAxis(self):
        dataset = self.MakeDataset()
        batch = dataset.CollateBatch([dataset[0], dataset[1]])
        points = batch['frames'][0]['points']
        self.assertEqual(tuple(points.shape), (4, 5))  # batch,x,y,z,t
        features, coords, _, mask, shape = BuildBatchVoxels(
            points, [.1, .1, .1], [-1., -1., -1., 1., 1., 1.], 4
        )
        self.assertEqual(tuple(features.shape), (4, 4))
        self.assertEqual(tuple(coords.shape), (4, 4))  # batch,z,y,x
        self.assertTrue(bool(mask.all()))
        self.assertEqual(shape.tolist(), [20, 20, 20])
        for rank, expected in [(0, -.100013), (1, -.310019)]:
            torch.testing.assert_close(features[coords[:, 0] == rank, 3], torch.full((2,), expected))
        xyz_points = points[:, :4]
        xyz_features, xyz_coords, _, _, xyz_shape = BuildBatchVoxels(
            xyz_points, [.1, .1, .1], [-1., -1., -1., 1., 1., 1.], 3
        )
        torch.testing.assert_close(features[:, :3], xyz_features)
        torch.testing.assert_close(coords, xyz_coords)
        torch.testing.assert_close(shape, xyz_shape)

    def TestOnlyExplicitSupplementalZeroPointBoxesRetained(self):
        dataset = self.MakeDataset(b_temporal=False)
        info = {'objects': [
            dict(box=[1., 0., 0., 4., 2., 1.5, 0.], type=1, num_points_in_top_lidar=3),
            dict(box=[2., 0., 0., 4., 2., 1.5, 0.], type=1, num_points_in_top_lidar=0),
            dict(box=[3., 0., 0., 4., 2., 1.5, 0.], type=1, num_points_in_top_lidar=0,
                 b_is_supplemental=True),
        ]}
        legacy = dataset.load_box3d({}, info)['annotations']['gt_boxes']
        np.testing.assert_array_equal(legacy[:, 0], [1.])
        dataset.b_use_completed_detection_labels = True
        completed = dataset.load_box3d({}, info)['annotations']['gt_boxes']
        np.testing.assert_array_equal(completed[:, 0], [1., 3.])
        empty = dataset.load_box3d({}, {'objects': []})['annotations']
        self.assertEqual(empty['gt_boxes'].shape, (0, 7))


if __name__ == '__main__':
    loader = unittest.TestLoader()
    loader.testMethodPrefix = 'Test'
    unittest.main(testLoader=loader)
