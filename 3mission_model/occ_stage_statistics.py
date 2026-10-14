"""Opt-in CPU OCC stage accounting; never called by training or model forward.

All grid inputs are [N,3] XYZ in one target vehicle-frame ROI. Point inputs are
real [N,3] XYZ metres in that same target frame, after any historical pose warp.
Sparse missing GT means empty ONLY after the caller confirms full ROI validity.
"""
import itertools
import hashlib
import json
from pathlib import Path

import numpy as np


def Array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def Keys(xyz, shape):
    xyz = Array(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or not np.equal(xyz, np.floor(xyz)).all():
        raise ValueError('Expected finite integer XYZ coordinates [N,3]')
    xyz = xyz.astype(np.int64)
    if np.any(xyz < 0) or np.any(xyz >= shape):
        raise ValueError('Grid coordinates outside confirmed ROI')
    return (xyz[:, 0] * shape[1] + xyz[:, 1]) * shape[2] + xyz[:, 2]


def XYZ(keys, shape):
    return np.column_stack((keys // (shape[1]*shape[2]), keys // shape[2] % shape[1], keys % shape[2]))


def UniqueLabeled(xyz, labels, shape, b_gt=False):
    keys = Keys(xyz, shape)
    labels = Array(labels)
    allowed = [0, 1, 2, 3, 255] if b_gt else [0, 1, 2, 3]
    if labels.shape != keys.shape or not np.isin(labels, allowed).all():
        raise ValueError('Invalid coordinate/label shape or class')
    unique, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    if b_gt and not np.equal(labels, labels[first][inverse]).all():
        raise ValueError('Conflicting duplicate GT classes')
    # Match existing FrameMetrics: duplicate prediction coordinates retain first.
    return unique, labels[first].astype(np.int64)


def Lookup(keys, gt_keys, gt_labels):
    target = np.zeros(len(keys), dtype=np.int64)
    if len(gt_keys):
        rows = np.searchsorted(gt_keys, keys)
        b_hit = rows < len(gt_keys)
        b_hit[b_hit] &= gt_keys[rows[b_hit]] == keys[b_hit]
        target[b_hit] = gt_labels[rows[b_hit]]
    return target


def ExpandRoots(root_keys, root_shape, b_expand_roots):
    if not b_expand_roots or not len(root_keys):
        return root_keys
    xyz = XYZ(root_keys, root_shape)
    expanded = xyz[:, None, :] + np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    expanded = expanded.reshape(-1, 3)
    expanded = expanded[((expanded >= 0) & (expanded < root_shape)).all(axis=1)]
    return np.unique(Keys(expanded, root_shape))


def PointKeys(points_xyz, pc_range, shape):
    points = Array(points_xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Real points must be target-frame metres [N,3]')
    b_inside = np.isfinite(points).all(axis=1) & (points >= pc_range[:3]).all(axis=1) & (points < pc_range[3:]).all(axis=1)
    voxels = np.floor((points[b_inside].astype(np.float64)-pc_range[:3])/.2).astype(np.int64)
    return np.unique(Keys(voxels, shape)), int((~b_inside).sum())


def Ratios(counts):
    def Ratio(numerator, denominator):
        return dict(numerator=int(numerator), denominator=int(denominator),
                    value=float(numerator/denominator) if denominator else None)
    c = counts
    result = {name: Ratio(c[num], c[den]) for name, num, den in (
        ('input_gt_coverage', 'input_gt', 'gt'), ('history_gt_coverage', 'history_gt', 'gt'),
        ('history_current_gt_coverage', 'history_current_gt', 'gt'),
        ('l0_gt_coverage', 'l0_gt', 'gt'), ('lplus_gt_coverage', 'lplus_gt', 'gt'), ('actual_gt_coverage', 'leaf_gt', 'gt'),
        ('expansion_added_nonempty_fraction', 'expansion_added_gt', 'expansion_added_leaf'),
        ('kept_expansion_added_nonempty_fraction', 'kept_expansion_added_gt', 'kept_expansion_added_leaf'),
        ('root_deletion_fraction', 'deleted_root', 'root'), ('occupied_root_recall', 'occupied_root_kept', 'occupied_root'),
        ('deletion_precision', 'empty_root_deleted', 'deleted_root'), ('empty_root_deletion_recall', 'empty_root_deleted', 'empty_root'),
        ('root_deletion_gt_coverage_loss', 'root_deletion_lost_gt', 'gt'),
        ('candidate_nonempty_fraction', 'leaf_gt', 'leaf'), ('candidate_occupancy_recall', 'occupancy_tp', 'leaf_gt'),
        ('occupancy_precision', 'occupancy_tp', 'pred_positive'), ('occupancy_recall', 'occupancy_tp', 'gt'),
        ('observed_gt_recall', 'observed_gt_tp', 'observed_gt'),
        ('unobserved_gt_recall', 'unobserved_gt_tp', 'unobserved_gt')) if num in c and den in c}
    result['occupancy_iou'] = Ratio(c['occupancy_tp'], c['occupancy_tp']+c['occupancy_fp']+c['occupancy_fn'])
    confusion = np.array(c['full_confusion'], dtype=np.int64)
    tp = confusion.diagonal(); union = confusion.sum(0)+confusion.sum(1)-tp
    result['class_iou'] = [Ratio(t, u) for t, u in zip(tp, union)]
    valid = [r['value'] for r in result['class_iou'][1:] if r['value'] is not None]
    result['semantic_miou'] = sum(valid)/len(valid) if valid else None
    result['semantic_miou_valid_classes'] = [i for i in (1, 2, 3) if union[i]]
    result['candidate_gt_class_fraction'] = [Ratio(v, c['leaf']) for v in c['candidate_gt_classes']]
    result['candidate_nonempty_class_fraction'] = [Ratio(v, c['leaf_gt']) for v in c['candidate_gt_classes'][1:]]
    result['gt_fates_fraction'] = [Ratio(v, c['gt']) for v in c['gt_fates']]
    return result


def FrameStatistics(target_id, pred_xyz, pred_labels, gt_xyz, gt_labels, grid_shape_xyz,
                    root_xyz, kept_root_xyz, expanded_root_xyz, current_points_xyz,
                    history_points_xyz, pc_range, b_expand_roots=True,
                    b_valid_roi_confirmed=False, actual_counts=None):
    """Compute additive counts from one target's existing OCC forward output.

    L0=64 children per R0; L+=64 children per clipped expanded R0; L is actual
    generated support. Explicit ignore cells are excluded from all leaf sets.
    history_points_xyz=[] means no history; None explicitly means NOT collected.
    Network feature support is never an observation substitute. Timing disables this call.
    """
    if not b_valid_roi_confirmed:
        raise ValueError('Must confirm full labeled valid ROI before treating sparse absence as empty')
    shape = np.array(grid_shape_xyz, dtype=np.int64)
    pc_range = np.array(pc_range, dtype=np.float64)
    if shape.shape != (3,) or np.any(shape <= 0) or np.any(shape % 4) or pc_range.shape != (6,) or not np.allclose((pc_range[3:]-pc_range[:3])/.2, shape, atol=1e-4, rtol=0):
        raise ValueError('Expected aligned 0.2m ROI with four leaves per root axis')
    root_shape = shape//4
    gt_keys, labels = UniqueLabeled(gt_xyz, gt_labels, shape, b_gt=True)
    positive = gt_keys[(labels > 0) & (labels < 4)]
    ignore = gt_keys[labels == 255]
    positive_parents = Keys(XYZ(positive, shape)//4, root_shape)
    ignore_parents = Keys(XYZ(ignore, shape)//4, root_shape)
    roots = np.unique(Keys(root_xyz, root_shape)); kept = np.unique(Keys(kept_root_xyz, root_shape))
    expanded = np.unique(Keys(expanded_root_xyz, root_shape))
    if not np.isin(kept, roots).all():
        raise ValueError('Rk must be subset of R0')
    all_expanded = ExpandRoots(roots, root_shape, b_expand_roots)
    expected_expanded = ExpandRoots(kept, root_shape, b_expand_roots)
    if not np.array_equal(expanded, expected_expanded):
        raise ValueError('Observed expanded roots differ from production 3x3x3 geometry; investigate truncation separately')
    leaf_raw, predictions_raw = UniqueLabeled(pred_xyz, pred_labels, shape)
    target_raw = Lookup(leaf_raw, gt_keys, labels)
    b_valid = target_raw != 255
    leaves, predictions, target = leaf_raw[b_valid], predictions_raw[b_valid], target_raw[b_valid]
    leaf_parents = Keys(XYZ(leaf_raw, shape)//4, root_shape)
    # Actual model guarantees all 64 leaves. Fail rather than attribute a decoder
    # truncation/top-k gap to root deletion.
    if len(leaf_raw) != 64*len(expanded) or not np.isin(leaf_parents, expanded).all():
        raise ValueError('Actual leaf support not complete 64-fold expanded roots; separate decoder truncation required')
    if not np.isin(leaf_parents, all_expanded).all():
        raise ValueError('L must be subset of L+')
    def LeafCount(root_keys):
        return 64*len(root_keys)-int(np.isin(ignore_parents, root_keys).sum())
    def GTCount(root_keys):
        return int(np.isin(positive_parents, root_keys).sum())
    b_root_positive = np.isin(roots, positive_parents); b_kept = np.isin(roots, kept)
    pred_positive = leaves[predictions > 0]
    confusion = np.bincount(4*target+predictions, minlength=16).reshape(4, 4)
    b_gt_lplus = np.isin(positive_parents, all_expanded)
    b_gt_leaf = np.isin(positive, leaves); b_gt_pred = np.isin(positive, pred_positive)
    fates = [int((~b_gt_lplus).sum()), int((b_gt_lplus & ~b_gt_leaf).sum()),
             int((b_gt_leaf & ~b_gt_pred).sum()), int(b_gt_pred.sum())]
    full = confusion.copy()
    missing_positive_labels = labels[(labels > 0) & (labels < 4)][~b_gt_leaf]
    full[:, 0] += np.bincount(missing_positive_labels, minlength=4)
    full[0, 0] += int(np.prod(shape))-len(ignore)-len(leaves)-len(missing_positive_labels)
    tp = full.diagonal(); fp = full.sum(0)-tp; fn = full.sum(1)-tp
    input_keys, rejected_points = PointKeys(current_points_xyz, pc_range, shape)
    b_history_observation_collected = history_points_xyz is not None
    history_keys, rejected_history = PointKeys(history_points_xyz, pc_range, shape) if b_history_observation_collected else (np.empty(0, dtype=np.int64), 0)
    input_keys = input_keys[~np.isin(input_keys, ignore)]
    history_keys = history_keys[~np.isin(history_keys, ignore)]
    union_input = np.union1d(input_keys, history_keys)
    added_roots = np.setdiff1d(all_expanded, roots)
    kept_added_roots = np.setdiff1d(expanded, kept)
    b_gt_observed = np.isin(positive, input_keys)
    counts = dict(observed_gt=int(b_gt_observed.sum()), unobserved_gt=int((~b_gt_observed).sum()),
        observed_gt_tp=int((b_gt_observed & b_gt_pred).sum()),
        unobserved_gt_tp=int((~b_gt_observed & b_gt_pred).sum()), gt=len(positive), ignored_leaf=len(ignore), valid_roi_leaf=int(np.prod(shape))-len(ignore),
        input_leaf=len(input_keys), input_gt=int(np.isin(input_keys, positive).sum()),
        history_leaf=len(history_keys), history_gt=int(np.isin(history_keys, positive).sum()),
        history_current_leaf=len(union_input), history_current_gt=int(np.isin(union_input, positive).sum()),
        rejected_current_points=rejected_points, rejected_history_points=rejected_history,
        root=len(roots), kept_root=len(kept), deleted_root=len(roots)-len(kept),
        occupied_root=int(b_root_positive.sum()), occupied_root_kept=int((b_root_positive & b_kept).sum()),
        occupied_root_deleted=int((b_root_positive & ~b_kept).sum()), empty_root=int((~b_root_positive).sum()),
        empty_root_deleted=int((~b_root_positive & ~b_kept).sum()), empty_root_kept=int((~b_root_positive & b_kept).sum()),
        l0_raw=64*len(roots), l0=LeafCount(roots), l0_gt=GTCount(roots),
        lplus_raw=64*len(all_expanded), lplus=LeafCount(all_expanded), lplus_gt=GTCount(all_expanded),
        all_kept_counterfactual_expanded_root=len(all_expanded),
        kept_direct_leaf=LeafCount(kept), kept_direct_gt=GTCount(kept),
        expansion_added_leaf=LeafCount(added_roots), expansion_added_gt=GTCount(added_roots),
        kept_expansion_added_leaf=LeafCount(kept_added_roots), kept_expansion_added_gt=GTCount(kept_added_roots),
        expanded_root=len(expanded), y3_raw=8*len(expanded), leaf_raw=len(leaf_raw), leaf=len(leaves),
        leaf_ignored=int((~b_valid).sum()), leaf_gt=int(b_gt_leaf.sum()), empty_candidate=int((target == 0).sum()),
        candidate_gt_classes=np.bincount(target, minlength=4).tolist(), candidate_confusion=confusion.tolist(),
        candidate_gt_nonempty_pred_empty=fates[2], candidate_gt_nonempty_wrong_nonempty=int(confusion[1:, 1:].sum()-np.trace(confusion[1:, 1:])),
        candidate_gt_nonempty_correct=int(np.trace(confusion[1:, 1:])),
        root_deletion_lost_gt=fates[1], gt_fates=fates, pred_positive=len(pred_positive),
        occupancy_tp=fates[3], occupancy_fp=len(pred_positive)-fates[3], occupancy_fn=len(positive)-fates[3],
        full_confusion=full.tolist(), tp=tp.tolist(), fp=fp.tolist(), fn=fn.tolist())
    if sum(fates) != len(positive) or counts['lplus']-counts['l0'] != counts['expansion_added_leaf']:
        raise AssertionError('OCC set accounting conservation failure')
    if actual_counts is not None:
        mapping = {'root_count':'root', 'kept_root_count':'kept_root', 'deleted_root_count':'deleted_root',
                   'expanded_root_count':'expanded_root', 'y3_count':'y3_raw', 'leaf_count':'leaf_raw'}
        if any(int(actual_counts[k]) != counts[v] for k, v in mapping.items()):
            raise ValueError('Existing forward counts disagree with independent support accounting')
    if not b_history_observation_collected:
        for key in ('history_leaf', 'history_gt', 'history_current_leaf', 'history_current_gt', 'rejected_history_points'):
            counts.pop(key)
    return dict(schema_version=1, target_id=str(target_id), counts=counts, ratios=Ratios(counts),
                b_history_observation_collected=b_history_observation_collected,
                grid_shape_xyz=shape.tolist(), pc_range=pc_range.tolist(), voxel_size_xyz=[.2]*3,
                b_expand_roots=bool(b_expand_roots),
                b_complete_leaf_geometry=True, b_valid_roi_confirmed=True)


class StatisticsCollector:
    """Explicit opt-in; disabled calls do no array work or output writes."""
    def __init__(self, b_enabled=False, output_jsonl=None):
        self.b_enabled = bool(b_enabled)
        self.output_jsonl = Path(output_jsonl) if output_jsonl else None
        self.frames = {}

    def Add(self, frame):
        if not self.b_enabled:
            return False
        token = frame['target_id']
        if self.frames:
            reference = next(iter(self.frames.values()))
            if frame.get('b_history_observation_collected', True) != reference.get('b_history_observation_collected', True):
                raise ValueError('Cannot mix collected and uncollected history observations')
            if any(frame[key] != reference[key] for key in ('schema_version', 'grid_shape_xyz', 'pc_range', 'voxel_size_xyz', 'b_expand_roots')):
                raise ValueError('Cannot merge statistics with different ROI or expansion protocols')
        if token in self.frames:
            if self.frames[token] != frame:
                raise ValueError('Conflicting duplicate target statistics')
            return False
        self.frames[token] = frame
        if self.output_jsonl:
            self.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.output_jsonl.open('a') as stream:
                stream.write(json.dumps(frame, allow_nan=False)+'\n')
        return True

    def ComputeAndAdd(self, **kwargs):
        if not self.b_enabled:
            return None
        frame = FrameStatistics(**kwargs)
        self.Add(frame)
        return frame

    def Summary(self):
        if not self.frames:
            return dict(frame_count=0, counts={}, ratios={})
        counts = {key: np.sum([frame['counts'][key] for frame in self.frames.values()], axis=0).tolist()
                  for key in next(iter(self.frames.values()))['counts']}
        distributions = {}
        for key in ('input_gt_coverage', 'history_current_gt_coverage', 'actual_gt_coverage', 'root_deletion_gt_coverage_loss', 'occupancy_iou'):
            if key not in next(iter(self.frames.values()))['ratios']:
                continue
            values = [f['ratios'][key]['value'] for f in self.frames.values() if f['ratios'][key]['value'] is not None]
            distributions[key] = dict(defined_frames=len(values), quantiles=np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else None)
        return dict(frame_count=len(self.frames), counts=counts, ratios=Ratios(counts), frame_distributions=distributions)


def MergeStatistics(paths, expected_target_ids=None):
    """Merge rank JSONL by exact target ID; conflicting repeats fail loudly."""
    collector = StatisticsCollector(b_enabled=True)
    duplicate_count = 0
    for path in paths:
        with Path(path).open() as stream:
            for line in stream:
                if line.strip():
                    duplicate_count += not collector.Add(json.loads(line))
    if expected_target_ids is not None and set(collector.frames) != set(map(str, expected_target_ids)):
        raise ValueError('Statistics target IDs do not exactly match fixed validation split')
    summary = collector.Summary()
    summary['duplicate_records_removed'] = duplicate_count
    summary['target_ids_sha256'] = hashlib.sha256('\n'.join(sorted(collector.frames)).encode()).hexdigest()
    return summary
