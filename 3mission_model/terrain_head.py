import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as functional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.network_prediction.training_model import DownBlock, SmallConvBlock, UpBlock


TERRAIN_DISTANCE_BINS_M = (
    ("0_1m", 0.0, 1.0),
    ("1_3m", 1.0, 3.0),
    ("3_6m", 3.0, 6.0),
    ("6m_plus", 6.0, math.inf),
)


class TerrainHead(nn.Module):
    """Predict a local-frame terrain height grid from BEV features.

    Input uses BEV order [B, C, Y, X]. Both returned tensors use the terrain
    label order [B, C, X_decreasing, Y_decreasing].
    """

    def __init__(self, in_channels=256, hidden_channels=32, b_predict_validity=False):
        super().__init__()
        mid_channels = int(hidden_channels)
        deep_channels = mid_channels * 2
        bottleneck_channels = mid_channels * 4

        self.enc1 = DownBlock(int(in_channels), mid_channels)
        self.enc2 = DownBlock(mid_channels, deep_channels)
        self.bottleneck = SmallConvBlock(deep_channels, bottleneck_channels)
        self.dec2 = UpBlock(bottleneck_channels, deep_channels, deep_channels)
        self.dec1 = UpBlock(deep_channels, mid_channels, mid_channels)
        self.output_upsample = nn.ConvTranspose2d(
            mid_channels,
            mid_channels,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.output_refine = SmallConvBlock(mid_channels, mid_channels)
        self.height_head = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.validity_head = nn.Conv2d(mid_channels, 1, kernel_size=1) if b_predict_validity else None

    def forward(self, bev_feature_map):
        if bev_feature_map.ndim != 4:
            raise ValueError(
                "bev_feature_map must have shape [B,C,Y,X], "
                f"got {tuple(bev_feature_map.shape)}"
            )

        skip1, down1 = self.enc1(bev_feature_map)
        skip2, down2 = self.enc2(down1)
        bottleneck = self.bottleneck(down2)
        up2 = self.dec2(bottleneck, skip2)
        bev_order_features = self.dec1(up2, skip1)
        bev_order_features = self.output_upsample(bev_order_features)
        bev_order_features = self.output_refine(bev_order_features)

        terrain_features = bev_order_features.transpose(-2, -1).flip((-2, -1))
        terrain_height_pred = self.height_head(terrain_features)
        outputs = {
            "terrain_height_pred": terrain_height_pred,
            "terrain_features": terrain_features,
        }
        if self.validity_head is not None:
            logits = self.validity_head(terrain_features)
            probability = torch.sigmoid(logits)
            b_valid = probability >= 0.5
            outputs.update({
                "terrain_validity_logits": logits,
                "terrain_validity_probability": probability,
                "terrain_validity_mask": b_valid,
                "terrain_height_masked": terrain_height_pred.masked_fill(~b_valid, float("nan")),
            })
        return outputs


class TerrainLoss(nn.Module):
    def __init__(
        self,
        beta=0.15,
        gradient_weight=0.2,
        distance_sigma_m=3.0,
        eps=1e-6,
        validity_weight=1.0,
    ):
        super().__init__()
        self.beta = max(float(beta), 1e-6)
        self.gradient_weight = float(gradient_weight)
        self.distance_sigma_m = max(float(distance_sigma_m), 1e-6)
        self.eps = float(eps)
        self.validity_weight = float(validity_weight)

    @staticmethod
    def _NormalizeGridRank(grid_tensor, field_name):
        if not torch.is_tensor(grid_tensor):
            grid_tensor = torch.as_tensor(grid_tensor)
        if grid_tensor.ndim == 2:
            return grid_tensor.unsqueeze(0).unsqueeze(0)
        if grid_tensor.ndim == 3:
            return grid_tensor.unsqueeze(1)
        if grid_tensor.ndim == 4 and grid_tensor.shape[1] == 1:
            return grid_tensor
        raise ValueError(
            f"{field_name} must have shape [H,W], [B,H,W], or [B,1,H,W], "
            f"got {tuple(grid_tensor.shape)}"
        )

    @staticmethod
    def _WeightedMean(value_map, weight_map, eps):
        return torch.sum(value_map * weight_map) / torch.clamp(
            torch.sum(weight_map),
            min=float(eps),
        )

    def _SmoothL1LossMap(self, pred_map, target_map):
        diff = torch.abs(pred_map - target_map)
        return torch.where(
            diff < self.beta,
            0.5 * diff * diff / self.beta,
            diff - 0.5 * self.beta,
        )

    def _GradientLoss(self, pred_map, target_map, supervised_mask, weight_map):
        direction_losses = []
        for spatial_dim in (-1, -2):
            if pred_map.shape[spatial_dim] < 2:
                direction_losses.append(pred_map.sum() * 0.0)
                continue

            first_slice = [slice(None)] * pred_map.ndim
            second_slice = [slice(None)] * pred_map.ndim
            first_slice[spatial_dim] = slice(None, -1)
            second_slice[spatial_dim] = slice(1, None)
            first_slice = tuple(first_slice)
            second_slice = tuple(second_slice)

            pair_mask = supervised_mask[first_slice] & supervised_mask[second_slice]
            pair_weight = 0.5 * (
                weight_map[first_slice] + weight_map[second_slice]
            )
            pair_weight = torch.where(pair_mask, pair_weight, torch.zeros_like(pair_weight))
            pred_delta = pred_map[second_slice] - pred_map[first_slice]
            target_delta = target_map[second_slice] - target_map[first_slice]
            direction_losses.append(
                self._WeightedMean(
                    torch.abs(pred_delta - target_delta),
                    pair_weight,
                    self.eps,
                )
            )
        return 0.5 * (direction_losses[0] + direction_losses[1])

    def _BuildDistanceBinLogs(
        self,
        abs_error,
        squared_error,
        weight_map,
        supervised_mask,
        distance_map,
        distance_valid_mask,
    ):
        logs = {}
        for bin_name, lower_m, upper_m in TERRAIN_DISTANCE_BINS_M:
            bin_mask = supervised_mask & distance_valid_mask & (distance_map >= lower_m)
            if math.isfinite(upper_m):
                bin_mask &= distance_map < upper_m
            bin_weight = torch.where(bin_mask, weight_map, torch.zeros_like(weight_map))
            weight_sum = bin_weight.sum()
            abs_error_sum = (abs_error * bin_weight).sum()
            squared_error_sum = (squared_error * bin_weight).sum()
            metric_denominator = torch.clamp(weight_sum, min=self.eps)
            prefix = f"terrain_distance_{bin_name}"
            logs[f"{prefix}_valid_count"] = bin_mask.sum().detach()
            logs[f"{prefix}_weight_sum"] = weight_sum.detach()
            logs[f"{prefix}_abs_error_sum"] = abs_error_sum.detach()
            logs[f"{prefix}_squared_error_sum"] = squared_error_sum.detach()
            logs[f"{prefix}_mae_m"] = (abs_error_sum / metric_denominator).detach()
            logs[f"{prefix}_rmse_m"] = torch.sqrt(
                squared_error_sum / metric_denominator
            ).detach()
        return logs

    def ComputeLoss(self, outputs, batch):
        pred_map = self._NormalizeGridRank(
            outputs["terrain_height_pred"],
            "terrain_height_pred",
        )
        if not torch.is_floating_point(pred_map):
            raise TypeError("terrain_height_pred must be a floating-point tensor")
        if not bool(torch.isfinite(pred_map).all().item()):
            raise FloatingPointError("terrain_height_pred contains NaN or Inf")

        field_names = (
            "terrain_height",
            "terrain_valid_mask",
            "terrain_train_weight",
            "terrain_inpaint_distance",
        )
        normalized_batch = {
            field_name: self._NormalizeGridRank(batch[field_name], field_name).to(
                device=pred_map.device
            )
            for field_name in field_names
        }
        for field_name, grid_tensor in normalized_batch.items():
            if grid_tensor.shape != pred_map.shape:
                raise ValueError(
                    f"{field_name} shape {tuple(grid_tensor.shape)} does not match "
                    f"terrain_height_pred shape {tuple(pred_map.shape)}"
                )

        target_raw = normalized_batch["terrain_height"].to(dtype=pred_map.dtype)
        valid_mask_raw = normalized_batch["terrain_valid_mask"]
        exported_weight_raw = normalized_batch["terrain_train_weight"].to(
            dtype=pred_map.dtype
        )
        distance_raw = normalized_batch["terrain_inpaint_distance"].to(
            dtype=pred_map.dtype
        )

        height_valid_mask = torch.isfinite(target_raw) & (target_raw != INVALID_HEIGHT)
        # [B,1,X_decreasing,Y_decreasing], local vehicle coordinates. Sentinel holes
        # are negative classification targets; corrupt or contradictory labels are ignored.
        b_ignore = ~torch.isfinite(target_raw) | (valid_mask_raw.bool() != height_valid_mask)
        if "terrain_ignore_mask" in batch:
            explicit_ignore = self._NormalizeGridRank(batch["terrain_ignore_mask"], "terrain_ignore_mask")
            if explicit_ignore.shape != pred_map.shape:
                raise ValueError("terrain_ignore_mask shape does not match terrain_height_pred")
            b_ignore = b_ignore | explicit_ignore.to(device=pred_map.device, dtype=torch.bool)
        valid_mask = valid_mask_raw.to(dtype=torch.bool) & height_valid_mask & ~b_ignore
        exported_weight = torch.where(
            torch.isfinite(exported_weight_raw),
            torch.clamp(exported_weight_raw, min=0.0),
            torch.zeros_like(exported_weight_raw),
        )
        distance_valid_mask = torch.isfinite(distance_raw) & (distance_raw != INVALID_HEIGHT)
        safe_distance = torch.where(
            distance_valid_mask,
            torch.clamp(distance_raw, min=0.0),
            torch.zeros_like(distance_raw),
        )
        distance_weight = torch.where(
            distance_valid_mask,
            torch.exp(-safe_distance / self.distance_sigma_m),
            torch.zeros_like(safe_distance),
        )
        weight_map = torch.maximum(exported_weight, distance_weight)
        weight_map = torch.where(valid_mask, weight_map, torch.zeros_like(weight_map))
        supervised_mask = valid_mask & torch.isfinite(weight_map) & (weight_map > 0.0)
        weight_map = torch.where(supervised_mask, weight_map, torch.zeros_like(weight_map))
        target_map = torch.where(supervised_mask, target_raw, torch.zeros_like(target_raw))

        b_has_supervision = bool(torch.any(supervised_mask).item())
        if b_has_supervision:
            main_loss = self._WeightedMean(
                self._SmoothL1LossMap(pred_map, target_map),
                weight_map,
                self.eps,
            )
            gradient_loss = self._GradientLoss(
                pred_map,
                target_map,
                supervised_mask,
                weight_map,
            )
            total_loss = main_loss + self.gradient_weight * gradient_loss
        else:
            total_loss = pred_map.sum() * 0.0
            main_loss = total_loss
            gradient_loss = total_loss

        abs_error = torch.abs(pred_map - target_map)
        squared_error = (pred_map - target_map).square()
        weight_sum = weight_map.sum()
        abs_error_sum = (abs_error * weight_map).sum()
        squared_error_sum = (squared_error * weight_map).sum()
        metric_denominator = torch.clamp(weight_sum, min=self.eps)
        logs = {
            "terrain_loss": total_loss.detach(),
            "terrain_main_loss": main_loss.detach(),
            "terrain_gradient_loss": gradient_loss.detach(),
            "terrain_valid_count": supervised_mask.sum().detach(),
            "terrain_weight_sum": weight_sum.detach(),
            "terrain_abs_error_sum": abs_error_sum.detach(),
            "terrain_squared_error_sum": squared_error_sum.detach(),
            "terrain_mae_m": (abs_error_sum / metric_denominator).detach(),
            "terrain_rmse_m": torch.sqrt(
                squared_error_sum / metric_denominator
            ).detach(),
            "terrain_has_supervision": b_has_supervision,
        }
        safe_target = torch.where(valid_mask, target_raw, torch.zeros_like(target_raw))
        unweighted_error = torch.where(valid_mask, pred_map - safe_target, torch.zeros_like(pred_map))
        logs.update({
            "terrain_gt_valid_count": valid_mask.sum().detach(),
            "terrain_unweighted_abs_error_sum": unweighted_error.abs().double().sum().detach(),
            "terrain_unweighted_squared_error_sum": unweighted_error.double().square().sum().detach(),
            "terrain_ignore_count": b_ignore.sum().detach(),
        })
        if "terrain_validity_logits" in outputs:
            logits = self._NormalizeGridRank(outputs["terrain_validity_logits"], "terrain_validity_logits")
            if logits.shape != pred_map.shape:
                raise ValueError("terrain_validity_logits shape does not match terrain_height_pred")
            if not bool(torch.isfinite(logits).all().item()):
                raise FloatingPointError("terrain_validity_logits contains NaN or Inf")
            b_classify = ~b_ignore
            classification_loss_map = functional.binary_cross_entropy_with_logits(
                logits, valid_mask.to(dtype=logits.dtype), reduction="none"
            )
            validity_loss = classification_loss_map[b_classify].mean() if bool(b_classify.any()) else logits.sum() * 0.0
            total_loss = total_loss + self.validity_weight * validity_loss
            b_has_supervision = b_has_supervision or bool(b_classify.any())
            b_prediction = logits >= 0.0
            b_hole = ~valid_mask & b_classify
            logs.update({
                "terrain_loss": total_loss.detach(),
                "terrain_validity_loss": validity_loss.detach(),
                "terrain_validity_loss_sum": classification_loss_map[b_classify].double().sum().detach(),
                "terrain_classification_count": b_classify.sum().detach(),
                "terrain_validity_tp": (b_prediction & valid_mask).sum().detach(),
                "terrain_validity_fn": (~b_prediction & valid_mask).sum().detach(),
                "terrain_validity_fp": (b_prediction & b_hole).sum().detach(),
                "terrain_validity_tn": (~b_prediction & b_hole).sum().detach(),
                "terrain_joint_10cm_count": (b_prediction & valid_mask & (unweighted_error.abs() <= 0.1)).sum().detach(),
                "terrain_joint_20cm_count": (b_prediction & valid_mask & (unweighted_error.abs() <= 0.2)).sum().detach(),
                "terrain_has_supervision": b_has_supervision,
            })
        logs.update(
            self._BuildDistanceBinLogs(
                abs_error,
                squared_error,
                weight_map,
                supervised_mask,
                safe_distance,
                distance_valid_mask,
            )
        )
        return total_loss, logs, b_has_supervision


def ComputeLoss(outputs, batch, loss_module=None):
    if loss_module is None:
        loss_module = TerrainLoss()
    return loss_module.ComputeLoss(outputs, batch)
