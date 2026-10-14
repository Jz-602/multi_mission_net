"""Terrain grids use local LiDAR [X decreasing, Y decreasing] order."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def SaveTerrainDetails(terrain, batch, pc_range, destination, metadata):
    def Array(value):
        return value.detach().cpu().numpy().squeeze() if hasattr(value, 'detach') else np.asarray(value).squeeze()

    destination = Path(destination)
    gt_height = Array(batch['terrain_height'])
    b_gt_valid = Array(batch['terrain_valid_mask']).astype(bool)
    b_ignore = Array(batch['terrain_ignore_mask']).astype(bool) if 'terrain_ignore_mask' in batch else np.zeros_like(b_gt_valid)
    b_gt_valid &= ~b_ignore
    height = Array(terrain['height'])
    probability = Array(terrain['validity_probability'])
    b_prediction = probability >= 0.5
    masked_height = np.where(b_prediction, height, np.nan)
    abs_error_gt_valid = np.where(b_gt_valid, np.abs(height - gt_height), np.nan)
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    arrays = {
        'gt_height': gt_height, 'pred_height_raw': height, 'gt_valid_mask': b_gt_valid,
        'ignore_mask': b_ignore, 'pred_valid_probability': probability,
        'pred_valid_mask': b_prediction, 'pred_height_masked': masked_height,
        'abs_error_gt_valid_auxiliary': abs_error_gt_valid,
    }
    if 'validity_logits' in terrain:
        arrays['pred_valid_logits'] = Array(terrain['validity_logits'])
    panels = [
        (np.where(b_gt_valid, gt_height, np.nan), 'GT height (GT valid)', 'terrain', zmin, zmax),
        (height, 'Prediction: raw height (no mask)', 'terrain', zmin, zmax),
        (np.where(b_ignore, np.nan, b_gt_valid.astype(float)), 'GT valid mask (ignore omitted)', 'viridis', 0, 1),
        (probability, 'Predicted valid probability', 'viridis', 0, 1),
        (b_prediction, 'Predicted valid mask (threshold 0.5)', 'viridis', 0, 1),
        (masked_height, 'Prediction: height masked by prediction only', 'terrain', zmin, zmax),
        (abs_error_gt_valid, 'Auxiliary: absolute height error on GT valid cells', 'magma', 0, .5),
        (b_ignore, 'Ignored corrupt/explicit-ignore cells', 'gray', 0, 1),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(19, 9), layout='constrained')
    for ax, (grid, title, cmap, lo, hi) in zip(axes.flat, panels):
        image = ax.imshow(grid.T[:, ::-1], extent=[xmin, xmax, ymin, ymax], origin='upper', cmap=cmap, vmin=lo, vmax=hi)
        ax.set(title=title, xlabel='x (m)', ylabel='y (m)')
        fig.colorbar(image, ax=ax, shrink=.75)
    fig.suptitle(f"{metadata.get('split', 'diagnostic')} | {metadata.get('token', '')}\nTerrain validity is label availability, not traversability")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    np.savez_compressed(destination.with_suffix('.npz'), **arrays)
    details = dict(metadata, threshold=.5, pc_range=list(pc_range), grid_order='X decreasing, Y decreasing',
                   coordinate_frame='target-frame LiDAR', height_color_range_m=[zmin,zmax], error_color_range_m=[0,.5],
                   b_prediction_uses_gt_mask=False, b_error_panel_uses_gt_mask=True,
                   masked_empty_representation='NaN', task='terrain label validity, not traversability')
    destination.with_suffix('.json').write_text(json.dumps(details, ensure_ascii=False, indent=2)+'\n')
    return details
