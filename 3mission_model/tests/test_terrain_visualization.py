import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terrain_visualization import SaveTerrainDetails


class TerrainVisualizationTest(unittest.TestCase):
    def test_predicted_mask_never_uses_gt(self):
        target = torch.tensor([[[[1., -123456., 3.], [4., 5., -123456.]]]])
        b_valid = target != -123456.
        probability = 1. - b_valid.float()
        prediction = torch.full_like(target, 2.)
        batch = {'terrain_height': target, 'terrain_valid_mask': b_valid,
                 'terrain_ignore_mask': torch.zeros_like(b_valid)}
        terrain = {'height': prediction, 'validity_probability': probability}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'plot.png'
            SaveTerrainDetails(terrain, batch, [-24.,-24.,-7.,24.,24.,5.8], destination,
                               {'split': 'synthetic_test', 'token': 'axis_check'})
            with np.load(destination.with_suffix('.npz')) as data:
                np.testing.assert_array_equal(data['pred_valid_mask'], ~b_valid.numpy().squeeze())
                np.testing.assert_array_equal(np.isnan(data['pred_height_masked']), b_valid.numpy().squeeze())
                np.testing.assert_array_equal(np.isnan(data['abs_error_gt_valid_auxiliary']), ~b_valid.numpy().squeeze())
                self.assertEqual(data['gt_height'].shape, (2,3))
            metadata = json.loads(destination.with_suffix('.json').read_text())
            self.assertFalse(metadata['b_prediction_uses_gt_mask'])
            self.assertEqual(metadata['height_color_range_m'], [-7.,5.8])
            self.assertGreater(destination.stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
