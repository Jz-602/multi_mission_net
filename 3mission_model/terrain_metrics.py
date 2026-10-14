"""Global terrain metrics from deduplicated target-frame sufficient statistics."""
import math


TERRAIN_UNWEIGHTED_STATISTICS = (
    "terrain_gt_valid_count", "terrain_unweighted_abs_error_sum",
    "terrain_unweighted_squared_error_sum", "terrain_ignore_count",
)
TERRAIN_VALIDITY_STATISTICS = (
    "terrain_classification_count", "terrain_validity_loss_sum",
    "terrain_validity_tp", "terrain_validity_fn", "terrain_validity_fp", "terrain_validity_tn",
    "terrain_joint_10cm_count", "terrain_joint_20cm_count",
)


def SummarizeTerrainValidity(frames):
    """Ratios use summed cell counts, never a mean of frame ratios; undefined = None."""
    def Ratio(numerator, denominator):
        return numerator / denominator if denominator else None

    output = {}
    if not frames or "terrain_gt_valid_count" not in frames[0]:
        return output
    if any("terrain_gt_valid_count" not in frame for frame in frames):
        raise ValueError("Mixed terrain metric protocols")
    counts = {key.removeprefix("terrain_"): sum(frame[key] for frame in frames)
              for key in TERRAIN_UNWEIGHTED_STATISTICS}
    output.update(counts)
    output["unweighted_mae_m"] = Ratio(counts["unweighted_abs_error_sum"], counts["gt_valid_count"])
    squared_mean = Ratio(counts["unweighted_squared_error_sum"], counts["gt_valid_count"])
    output["unweighted_rmse_m"] = math.sqrt(squared_mean) if squared_mean is not None else None
    if "terrain_classification_count" not in frames[0]:
        if any("terrain_classification_count" in frame for frame in frames):
            raise ValueError("Mixed terrain validity-head protocols")
        return output
    if any("terrain_classification_count" not in frame for frame in frames):
        raise ValueError("Mixed terrain validity-head protocols")
    counts.update({key.removeprefix("terrain_"): sum(frame[key] for frame in frames)
                   for key in TERRAIN_VALIDITY_STATISTICS})
    tp, fn, fp, tn = [int(counts[f"validity_{name}"]) for name in ("tp", "fn", "fp", "tn")]
    count = tp + fn + fp + tn
    if count != counts["classification_count"] or tp + fn != counts["gt_valid_count"]:
        raise ValueError("Terrain confusion counts do not conserve classification/GT cells")
    valid_iou, hole_iou = Ratio(tp, tp + fp + fn), Ratio(tn, tn + fn + fp)
    defined_ious = [value for value in (valid_iou, hole_iou) if value is not None]
    all_valid_iou = Ratio(tp + fn, count)
    output["validity"] = {
        **counts, "threshold": 0.5, "confusion_gt_rows_pred_columns": [[tn, fp], [fn, tp]],
        "class_order": ["hole", "valid"], "valid_iou": valid_iou, "hole_iou": hole_iou,
        "miou": sum(defined_ious) / len(defined_ious) if defined_ious else None,
        "valid_precision": Ratio(tp, tp + fp), "valid_recall": Ratio(tp, tp + fn),
        "hole_precision": Ratio(tn, tn + fn), "hole_recall": Ratio(tn, tn + fp),
        "gt_valid_fraction": Ratio(tp + fn, count), "gt_hole_fraction": Ratio(tn + fp, count),
        "pred_valid_fraction": Ratio(tp + fp, count), "pred_hole_fraction": Ratio(tn + fn, count),
        "joint_10cm_recall": Ratio(counts["joint_10cm_count"], tp + fn),
        "joint_20cm_recall": Ratio(counts["joint_20cm_count"], tp + fn),
        "bce_mean": Ratio(counts["validity_loss_sum"], count),
        "zero_denominator_rule": "null; mIoU averages classes with nonzero IoU denominator",
        "all_valid_baseline": {
            "valid_iou": all_valid_iou, "hole_iou": 0.0 if tn + fp else None,
            "miou": (all_valid_iou / 2 if tn + fp else all_valid_iou),
            "valid_precision": all_valid_iou, "valid_recall": 1.0 if tp + fn else None,
            "hole_precision": None, "hole_recall": 0.0 if tn + fp else None,
            "pred_valid_fraction": 1.0 if count else None,
        },
    }
    return output
