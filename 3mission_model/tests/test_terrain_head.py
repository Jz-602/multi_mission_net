import math
import os
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gt_generate.common.constants import INVALID_HEIGHT
from terrain_head import TerrainHead, TerrainLoss


class TerrainHeadTest(unittest.TestCase):
    def test_forward_shape_and_orientation_for_rectangular_roi(self):
        torch.manual_seed(0)
        model = TerrainHead(in_channels=8, hidden_channels=4)
        captured = {}

        def SaveBEVOrderFeatures(_module, _inputs, output):
            captured["bev_order_features"] = output.detach()

        hook = model.output_refine.register_forward_hook(SaveBEVOrderFeatures)
        outputs = model(torch.randn(2, 8, 8, 10))
        hook.remove()

        self.assertEqual(tuple(outputs["terrain_features"].shape), (2, 4, 20, 16))
        self.assertEqual(tuple(outputs["terrain_height_pred"].shape), (2, 1, 20, 16))
        expected = captured["bev_order_features"].transpose(-2, -1).flip((-2, -1))
        torch.testing.assert_close(outputs["terrain_features"], expected)
        self.assertTrue(bool(torch.isfinite(outputs["terrain_height_pred"]).all()))


class TerrainLossTest(unittest.TestCase):
    def setUp(self):
        self.loss_module = TerrainLoss()

    @staticmethod
    def BuildBatch(target, valid_mask=None, exported_weight=None, distance=None):
        if valid_mask is None:
            valid_mask = torch.ones_like(target, dtype=torch.bool)
        if exported_weight is None:
            exported_weight = torch.ones_like(target)
        if distance is None:
            distance = torch.zeros_like(target)
        return {
            "terrain_height": target,
            "terrain_valid_mask": valid_mask,
            "terrain_train_weight": exported_weight,
            "terrain_inpaint_distance": distance,
        }

    def test_equal_prediction_has_zero_loss_with_invalid_values_sanitized(self):
        target = torch.tensor(
            [[0.0, 1.0, INVALID_HEIGHT], [2.0, float("nan"), float("inf")]]
        )
        pred = torch.tensor(
            [[[[0.0, 1.0, 9.0], [2.0, 8.0, 7.0]]]],
            requires_grad=True,
        )
        batch = self.BuildBatch(
            target,
            exported_weight=torch.tensor(
                [[1.0, float("nan"), 1.0], [float("inf"), 1.0, 1.0]]
            ),
            distance=torch.tensor(
                [[0.0, 0.0, INVALID_HEIGHT], [0.0, 0.0, 0.0]]
            ),
        )

        loss, logs, b_has_supervision = self.loss_module.ComputeLoss(
            {"terrain_height_pred": pred}, batch
        )
        self.assertTrue(b_has_supervision)
        self.assertEqual(int(logs["terrain_valid_count"]), 3)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))

    def test_empty_supervision_returns_connected_zero(self):
        pred = torch.randn(1, 1, 2, 3, requires_grad=True)
        batch = self.BuildBatch(
            torch.full((1, 2, 3), INVALID_HEIGHT),
            valid_mask=torch.zeros(1, 2, 3, dtype=torch.bool),
            exported_weight=torch.full((1, 2, 3), float("nan")),
            distance=torch.full((1, 2, 3), float("nan")),
        )

        loss, logs, b_has_supervision = self.loss_module.ComputeLoss(
            {"terrain_height_pred": pred}, batch
        )
        self.assertFalse(b_has_supervision)
        self.assertFalse(logs["terrain_has_supervision"])
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_weighted_metrics_and_distance_bins_use_exact_valid_subset(self):
        target = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        pred = torch.zeros(1, 1, 2, 2, requires_grad=True)
        distance = torch.tensor([[[0.5, 1.0], [3.0, 6.0]]])
        batch = self.BuildBatch(
            target,
            exported_weight=torch.zeros_like(target),
            distance=distance,
        )

        loss, logs, b_has_supervision = self.loss_module.ComputeLoss(
            {"terrain_height_pred": pred}, batch
        )
        self.assertTrue(b_has_supervision)
        weights = torch.exp(-distance / 3.0)
        expected_mae = (target * weights).sum() / weights.sum()
        expected_rmse = torch.sqrt((target.square() * weights).sum() / weights.sum())
        torch.testing.assert_close(logs["terrain_mae_m"], expected_mae)
        torch.testing.assert_close(logs["terrain_rmse_m"], expected_rmse)
        for bin_name in ("0_1m", "1_3m", "3_6m", "6m_plus"):
            self.assertEqual(
                int(logs[f"terrain_distance_{bin_name}_valid_count"]), 1
            )
        self.assertTrue(math.isfinite(float(loss.detach())))

    def test_gradient_loss_requires_two_supervised_endpoints(self):
        target = torch.tensor([[1.0, 5.0]])
        pred = torch.tensor([[[[2.0, -10.0]]]], requires_grad=True)
        batch = self.BuildBatch(
            target,
            valid_mask=torch.tensor([[True, True]]),
            exported_weight=torch.tensor([[1.0, 0.0]]),
            distance=torch.full((1, 2), INVALID_HEIGHT),
        )

        _, logs, _ = self.loss_module.ComputeLoss(
            {"terrain_height_pred": pred}, batch
        )
        self.assertEqual(float(logs["terrain_gradient_loss"]), 0.0)
        self.assertEqual(int(logs["terrain_valid_count"]), 1)

    def test_rank_three_batch_is_normalized_and_nonfinite_prediction_rejected(self):
        pred = torch.zeros(2, 1, 2, 3)
        target = torch.zeros(2, 2, 3)
        _, _, b_has_supervision = self.loss_module.ComputeLoss(
            {"terrain_height_pred": pred}, self.BuildBatch(target)
        )
        self.assertTrue(b_has_supervision)

        pred[0, 0, 0, 0] = float("nan")
        with self.assertRaises(FloatingPointError):
            self.loss_module.ComputeLoss(
                {"terrain_height_pred": pred}, self.BuildBatch(target)
            )

    @unittest.skipUnless(
        os.environ.get("RUN_TERRAIN_CUDA_TEST") == "1" and torch.cuda.is_available(),
        "set RUN_TERRAIN_CUDA_TEST=1 when a coordinated CUDA slot is available",
    )
    def test_cpu_prediction_accepts_cuda_validation_labels(self):
        pred = torch.zeros(1, 1, 2, 2, device="cpu", requires_grad=True)
        target = torch.ones(1, 1, 2, 2, device="cuda")
        loss, logs, b_has_supervision = self.loss_module.ComputeLoss(
            {"terrain_height_pred": pred},
            self.BuildBatch(target),
        )
        self.assertTrue(b_has_supervision)
        self.assertEqual(loss.device.type, "cpu")
        self.assertEqual(logs["terrain_mae_m"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
