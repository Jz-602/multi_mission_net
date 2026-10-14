import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terrain_head import TerrainHead, TerrainLoss, INVALID_HEIGHT
from terrain_metrics import SummarizeTerrainValidity, TERRAIN_UNWEIGHTED_STATISTICS, TERRAIN_VALIDITY_STATISTICS


class TerrainValidityTest(unittest.TestCase):
    @staticmethod
    def Batch(target, mask=None, ignore=None):
        batch = {
            'terrain_height': target,
            'terrain_valid_mask': (target != INVALID_HEIGHT) if mask is None else mask,
            'terrain_train_weight': torch.ones_like(target),
            'terrain_inpaint_distance': torch.zeros_like(target),
        }
        if ignore is not None:
            batch['terrain_ignore_mask'] = ignore
        return batch

    @staticmethod
    def Frame(logs):
        return {key: logs[key].item() for key in TERRAIN_UNWEIGHTED_STATISTICS + TERRAIN_VALIDITY_STATISTICS}

    def test_holes_supervised_and_height_not_hidden(self):
        target = torch.tensor([[1., INVALID_HEIGHT], [2., INVALID_HEIGHT]])
        prediction = torch.zeros(1, 1, 2, 2, requires_grad=True)
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        loss, logs, supervised = TerrainLoss().ComputeLoss(
            {'terrain_height_pred': prediction, 'terrain_validity_logits': logits}, self.Batch(target))
        self.assertTrue(supervised)
        self.assertAlmostEqual(logs['terrain_validity_loss'].item(), 0.69314718, places=6)
        loss.backward()
        self.assertTrue((logits.grad[0, 0, :, 0] < 0).all())
        self.assertTrue((logits.grad[0, 0, :, 1] > 0).all())
        self.assertTrue((prediction.grad[0, 0, :, 0] < 0).all())
        self.assertTrue((prediction.grad[0, 0, :, 1] == 0).all())
        for value in (-20., 20.):
            _, current_logs, _ = TerrainLoss().ComputeLoss(
                {'terrain_height_pred': prediction, 'terrain_validity_logits': torch.full_like(logits, value)}, self.Batch(target))
            torch.testing.assert_close(current_logs['terrain_main_loss'], logs['terrain_main_loss'])
            torch.testing.assert_close(current_logs['terrain_gradient_loss'], logs['terrain_gradient_loss'])
            self.assertEqual(current_logs['terrain_gt_valid_count'].item(), 2)

    def test_corrupt_unknown_ignored_but_sentinel_hole_not_ignored(self):
        target = torch.tensor([[INVALID_HEIGHT, float('nan'), 2., 3.]])
        mask = torch.tensor([[False, False, False, True]])
        logits = torch.zeros(1, 1, 1, 4, requires_grad=True)
        batch = self.Batch(target, mask, torch.tensor([[False, False, False, True]]))
        loss, logs, supervised = TerrainLoss().ComputeLoss(
            {'terrain_height_pred': torch.zeros_like(logits), 'terrain_validity_logits': logits}, batch)
        loss.backward()
        self.assertTrue(supervised)
        self.assertEqual(logs['terrain_ignore_count'].item(), 3)
        self.assertEqual(logs['terrain_classification_count'].item(), 1)
        self.assertGreater(logits.grad[0, 0, 0, 0].item(), 0.)
        self.assertEqual(logits.grad[0, 0, 0, 1:].abs().sum().item(), 0.)

    def test_all_holes_all_valid_and_all_ignored(self):
        for target_value in (INVALID_HEIGHT, 1.):
            target = torch.full((2, 3), target_value)
            logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
            loss, logs, supervised = TerrainLoss().ComputeLoss(
                {'terrain_height_pred': torch.zeros_like(logits), 'terrain_validity_logits': logits}, self.Batch(target))
            self.assertTrue(supervised)
            self.assertTrue(torch.isfinite(loss))
            summary = SummarizeTerrainValidity([self.Frame(logs)])['validity']
            self.assertEqual(summary['classification_count'], 6)
            if target_value == INVALID_HEIGHT:
                self.assertIsNone(summary['valid_recall'])
                self.assertIsNone(summary['joint_10cm_recall'])
                self.assertEqual(summary['hole_recall'], 0.)
            else:
                self.assertEqual(summary['valid_recall'], 1.)
                self.assertIsNone(summary['hole_recall'])
        target = torch.zeros(2, 3)
        logits = torch.zeros(1, 1, 2, 3, requires_grad=True)
        loss, logs, supervised = TerrainLoss().ComputeLoss(
            {'terrain_height_pred': torch.zeros_like(logits), 'terrain_validity_logits': logits},
            self.Batch(target, ignore=torch.ones_like(target, dtype=torch.bool)))
        loss.backward()
        self.assertFalse(supervised)
        self.assertEqual(logits.grad.abs().sum().item(), 0.)
        self.assertIsNone(SummarizeTerrainValidity([self.Frame(logs)])['validity']['miou'])

    def test_metrics_use_all_gt_valid_even_when_height_weight_zero(self):
        target = torch.tensor([[0., 1., INVALID_HEIGHT]])
        batch = self.Batch(target)
        batch['terrain_train_weight'].zero_()
        batch['terrain_inpaint_distance'].fill_(INVALID_HEIGHT)
        _, logs, _ = TerrainLoss().ComputeLoss(
            {'terrain_height_pred': torch.zeros(1, 1, 1, 3), 'terrain_validity_logits': torch.tensor([[[[1., -1., -1.]]]])}, batch)
        summary = SummarizeTerrainValidity([self.Frame(logs)])
        self.assertEqual(logs['terrain_valid_count'], 0)
        self.assertEqual(summary['unweighted_mae_m'], .5)
        self.assertAlmostEqual(summary['unweighted_rmse_m'], 2 ** -.5)
        self.assertEqual(summary['validity']['joint_10cm_recall'], .5)
        self.assertEqual(summary['validity']['valid_recall'], .5)
        self.assertEqual(summary['validity']['hole_recall'], 1.)

    def test_global_counts_not_mean_of_frame_ratios(self):
        frames = []
        for target, logits in (
            (torch.tensor([[0.]]), torch.tensor([[[[1.]]]])),
            (torch.tensor([[0., 0., 0., INVALID_HEIGHT]]), torch.tensor([[[[-1., -1., -1., 1.]]]])),
        ):
            _, logs, _ = TerrainLoss().ComputeLoss({'terrain_height_pred': torch.zeros_like(logits), 'terrain_validity_logits': logits}, self.Batch(target))
            frames.append(self.Frame(logs))
        summary = SummarizeTerrainValidity(frames)['validity']
        self.assertEqual(summary['valid_recall'], .25)
        self.assertEqual(summary['valid_precision'], .5)
        self.assertEqual(summary['valid_iou'], .2)
        self.assertEqual(summary['miou'], .1)
        self.assertEqual(summary['joint_20cm_recall'], .25)
        self.assertEqual(summary['all_valid_baseline']['miou'], .4)
        self.assertEqual(summary['confusion_gt_rows_pred_columns'], [[0, 1], [3, 1]])

    def test_head_shared_orientation_and_inference_has_no_gt(self):
        torch.manual_seed(4)
        model = TerrainHead(in_channels=8, hidden_channels=4, b_predict_validity=True).eval()
        captured = {}
        def Capture(_module, _inputs, output):
            captured['bev'] = output
        handle = model.output_refine.register_forward_hook(Capture)
        output = model(torch.randn(2, 8, 8, 10))
        handle.remove()
        expected = captured['bev'].transpose(-2, -1).flip((-2, -1))
        torch.testing.assert_close(output['terrain_validity_logits'], model.validity_head(expected))
        torch.testing.assert_close(output['terrain_height_pred'], model.height_head(expected))
        self.assertEqual(output['terrain_validity_logits'].shape, (2, 1, 20, 16))
        torch.testing.assert_close(torch.isnan(output['terrain_height_masked']), ~output['terrain_validity_mask'])
        (output['terrain_height_pred'].mean() + output['terrain_validity_logits'].mean()).backward()
        for parameter in (model.height_head.weight, model.validity_head.weight, model.output_upsample.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_validator_frame_adapter_and_global_aggregation(self):
        from trainer.trainer.multitask_validator import BuildTerrainFrameStatistics, SummarizeTerrainMetrics
        batch = self.Batch(torch.tensor([[[[0., INVALID_HEIGHT]]], [[[1., INVALID_HEIGHT]]]]))
        results = [
            {'height': torch.zeros(1, 1, 2), 'validity_logits': torch.tensor([[[1., -1.]]])},
            {'height': torch.zeros(1, 1, 2), 'validity_logits': torch.tensor([[[-1., -1.]]])},
        ]
        # Records remain keyed by target ID, exactly as distributed validation deduplicates.
        records = {}
        for index in (0, 1, 1):
            records[str(index)] = BuildTerrainFrameStatistics(results[index], batch, index)
        metrics = SummarizeTerrainMetrics(records.values())
        self.assertEqual(metrics['frame_count'], 2)
        self.assertEqual(metrics['validity']['classification_count'], 4)
        self.assertEqual(metrics['validity']['gt_valid_count'], 2)
        self.assertEqual(metrics['unweighted_mae_m'], .5)
        self.assertEqual(metrics['validity']['valid_recall'], .5)
        self.assertEqual(metrics['validity']['hole_recall'], 1.)
        self.assertEqual(metrics['validity']['joint_20cm_recall'], .5)

    def test_backward_compatibility_and_nonfinite_logits(self):
        self.assertNotIn('validity_head.weight', TerrainHead(in_channels=8, hidden_channels=4).state_dict())
        with self.assertRaises(FloatingPointError):
            TerrainLoss().ComputeLoss({'terrain_height_pred': torch.zeros(1, 1, 1, 1), 'terrain_validity_logits': torch.full((1, 1, 1, 1), float('nan'))}, self.Batch(torch.zeros(1, 1)))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
