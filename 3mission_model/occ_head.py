"""Generated static OCC in vehicle coordinates; sparse indices always [batch,z,y,x]."""
from __future__ import annotations

import itertools
import math
from contextlib import contextmanager, nullcontext

import spconv.pytorch as spconv
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from loss_segmentation.dice_loss_v1 import DiceLossV1
from loss_segmentation.lovasz_losses import lovasz_softmax_flat



@contextmanager
def PreserveBatchNormRunningStats(module):
    """Undo only recomputation's BN buffer updates; retain ordinary training BN.

    Snapshot at recomputation entry, after the real forward has already updated
    these buffers. Keeping track_running_stats enabled makes the recomputation
    save the same tensors as the original non-reentrant checkpoint forward.
    """
    snapshots = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm) and child.training and child.track_running_stats:
            for name in ('running_mean', 'running_var', 'num_batches_tracked'):
                buffer = getattr(child, name)
                if buffer is not None:
                    snapshots.append((buffer, buffer.detach().clone()))
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, saved_value in snapshots:
                buffer.copy_(saved_value)


def CheckpointSparseRefinement(refinement, features, indices, spatial_shape, batch_size):
    """Checkpoint features on one fixed support, rebuilding indice caches each run."""
    def Refine(input_features):
        sparse = spconv.SparseConvTensor(input_features, indices.int(), spatial_shape, batch_size)
        return refinement(sparse).features

    refined_features = checkpoint(
        Refine, features, use_reentrant=False,
        context_fn=lambda: (nullcontext(), PreserveBatchNormRunningStats(refinement)),
    )
    return spconv.SparseConvTensor(refined_features, indices.int(), spatial_shape, batch_size)



def CheckpointGeneratedStage(generator, refinement, parent_features, child_indices,
                             skip_features, skip_indices, spatial_shape, batch_size, out_channels):
    """Keep compact parent/encoder inputs instead of generated/concatenated leaves.

    Checkpoint each residual block separately: recomputing both blocks together
    would recreate all of their saved activations at the same backward peak.
    The first block's checkpoint also recomputes generation and the skip join.
    """
    blocks = list(refinement.children())
    first_block = blocks[0]

    def GenerateAndRefine(input_parent_features, input_skip_features):
        generated = F.relu(generator(input_parent_features).reshape(-1, out_channels))
        matched_skip = GatherSkip(child_indices, skip_indices, input_skip_features, spatial_shape)
        merged = torch.cat((generated, matched_skip), dim=1)
        # They are not needed as Python locals during sparse residual processing.
        del generated, matched_skip
        sparse = spconv.SparseConvTensor(merged, child_indices.int(), spatial_shape, batch_size)
        return first_block(sparse).features

    features = checkpoint(
        GenerateAndRefine, parent_features, skip_features, use_reentrant=False,
        context_fn=lambda: (nullcontext(), PreserveBatchNormRunningStats(first_block)),
    )
    current = spconv.SparseConvTensor(features, child_indices.int(), spatial_shape, batch_size)
    for block in blocks[1:]:
        current = CheckpointSparseRefinement(block, current.features, child_indices, spatial_shape, batch_size)
    return current


def CoordinateKeys(indices, spatial_shape):
    """Int64 keys for [N,4] BZYX indices in the supplied ZYX grid."""
    z_size, y_size, x_size = map(int, spatial_shape)
    coords = indices.long()
    return ((coords[:, 0] * z_size + coords[:, 1]) * y_size + coords[:, 2]) * x_size + coords[:, 3]


def UniqueRows(indices, spatial_shape):
    """Return sorted unique coordinate rows; duplicate ties retain their first row."""
    keys = CoordinateKeys(indices, spatial_shape)
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    b_first = torch.ones_like(sorted_keys, dtype=torch.bool)
    b_first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return order[b_first]


def MatchRows(query, source, spatial_shape):
    """Return source row per query or -1, preserving query order and batch isolation."""
    if source.shape[0] == 0:
        return torch.full((query.shape[0],), -1, dtype=torch.long, device=query.device)
    unique_rows = UniqueRows(source, spatial_shape)
    source_keys = CoordinateKeys(source[unique_rows], spatial_shape)
    query_keys = CoordinateKeys(query, spatial_shape)
    locations = torch.searchsorted(source_keys, query_keys)
    b_hit = locations < source_keys.numel()
    safe_locations = locations.clamp(max=source_keys.numel() - 1)
    b_hit &= source_keys[safe_locations] == query_keys
    return torch.where(b_hit, unique_rows[safe_locations], -1)


def GatherSkip(query, source_indices, source_features, spatial_shape):
    rows = MatchRows(query, source_indices, spatial_shape)
    result = source_features.new_zeros((len(query), source_features.shape[1]))
    b_hit = rows >= 0
    result[b_hit] = source_features[rows[b_hit]]
    # Preserve the shared skip path even when no generated position hits it.
    return result + source_features.sum() * 0.0


def GenerateChildren(indices):
    offsets = torch.tensor(list(itertools.product((0, 1), repeat=3)), device=indices.device, dtype=indices.dtype)
    children = indices[:, None, :].expand(-1, 8, -1).clone()
    children[:, :, 1:] = 2 * children[:, :, 1:] + offsets[None]
    return children.reshape(-1, 4)


def ClassificationLoss(logits, target, loss_types=None, loss_weights=None, ignore_index=255):
    """Batch leaf means; empty is a supervised class, 255 alone is ignored."""
    loss_types = loss_types or ('CELoss', 'DiceLossV1', 'LovLoss')
    loss_weights = loss_weights or (1.0, 1.0, 1.0)
    if len(loss_types) != len(loss_weights) or set(loss_types) != {'CELoss', 'DiceLossV1', 'LovLoss'}:
        raise ValueError('OCC needs exactly CELoss, DiceLossV1 and LovLoss with matching weights')
    if not torch.isfinite(logits).all():
        raise ValueError('Non-finite OCC logits')
    b_valid = target != ignore_index
    valid_count = int(b_valid.sum())
    zero = logits.sum() * 0.0
    if valid_count == 0:
        return zero, {name: zero for name in loss_types}, 0
    valid_logits, valid_target = logits[b_valid], target[b_valid].long()
    losses = {
        'CELoss': F.cross_entropy(valid_logits, valid_target),
        'DiceLossV1': DiceLossV1(ignore_index)(valid_logits, valid_target),
        # Explicit present-class list avoids the legacy helper's string identity comparison.
        'LovLoss': lovasz_softmax_flat(valid_logits.softmax(dim=1), valid_target,
                                      classes=torch.unique(valid_target).tolist()),
    }
    return sum(losses[name] * weight for name, weight in zip(loss_types, loss_weights)), losses, valid_count


class GeneratedOCCHead(nn.Module):
    """x4/x3/x2: 256/128/64 channels at 0.8/0.4/0.2 m, same ROI origin.

    config is the independent OCC mapping. pc_range describes the model ROI;
    voxel_size_xyz is the leaf resolution (0.2 m), not the input resolution.
    Sparse block factories reuse the main encoder's implementation without imports
    back into the main model. Changed coordinate sets always get fresh tensors.
    """
    def __init__(self, config, sparse_basic_block_cls, sparse_conv_bn_relu_cls):
        super().__init__()
        config = dict(config or {})
        self.pc_range = tuple(config.get('pc_range', [-80, -80, -7, 80, 80, 5.8]))
        self.voxel_size_xyz = tuple(config.get('voxel_size_xyz', [.2, .2, .2]))
        if len(self.pc_range) != 6 or len(self.voxel_size_xyz) != 3 or not all(math.isfinite(value) for value in (*self.pc_range, *self.voxel_size_xyz)):
            raise ValueError('OCC pc_range and voxel_size_xyz must be finite XYZ metadata')
        if any(value <= 0 for value in self.voxel_size_xyz):
            raise ValueError('OCC voxel_size_xyz must be positive')
        grid_extent = tuple((self.pc_range[i + 3] - self.pc_range[i]) / self.voxel_size_xyz[i] for i in range(3))
        if any(not math.isclose(value, round(value), rel_tol=0., abs_tol=1e-4) for value in grid_extent):
            raise ValueError('OCC ROI extent must be an integer number of output voxels')
        self.grid_shape_xyz = tuple(round(value) for value in grid_extent)
        if any(abs(v - .2) > 1e-6 for v in self.voxel_size_xyz) or any(n <= 0 or n % 4 for n in self.grid_shape_xyz):
            raise ValueError('OCC needs 0.2m leaves and an ROI with each axis divisible by 0.8m')
        if config.get('num_classes', 4) != 4 or config.get('ignore_index', 255) != 255:
            raise ValueError('OCC classes are empty/pole/general/vegetation and ignore_index=255')
        self.class_names = config.get('class_names', ['EMPTY', 'POLE', 'GENERAL_OBSTACLE', 'VEGETATION'])
        self.loss_types = config.get('loss_types', ['CELoss', 'DiceLossV1', 'LovLoss'])
        self.loss_weights = config.get('loss_weights', [1., 1., 1.])
        self.root_loss_weight = float(config.get('root_loss_weight', 1.))
        self.delete_threshold = float(config.get('delete_threshold', .5))
        self.b_enable_root_filter = bool(config.get('b_enable_root_filter', True))
        self.root_input_normalization = str(config.get('root_input_normalization', 'none'))
        if self.root_input_normalization not in ('none', 'layer_norm'):
            raise ValueError('root_input_normalization must be none or layer_norm')
        self.b_collect_root_diagnostics = bool(config.get('b_collect_root_diagnostics', False))
        self.b_collect_occ_stage_statistics = bool(config.get('b_collect_occ_stage_statistics', False))
        self.b_expand_roots = bool(config.get('b_expand_roots', True))
        # Also recompute generation/skip; each residual block has its own checkpoint.
        self.b_checkpoint_refinement = bool(config.get('b_checkpoint_refinement', False))
        if self.b_enable_root_filter:
            self.root_score = nn.Linear(256, 1)
            initial_prob = float(config.get('init_delete_prob', .01))
            if not 0 < initial_prob <= self.delete_threshold < 1:
                raise ValueError('Root initialization must keep all roots at the deletion threshold')
            nn.init.zeros_(self.root_score.weight)
            nn.init.constant_(self.root_score.bias, math.log(initial_prob / (1 - initial_prob)))
        else:
            self.root_score = None
        self.root_expand = sparse_conv_bn_relu_cls(256, 256, kernel_size=3, stride=1, padding=1,
            conv_type='spconv', indice_key='occ_root_expand') if self.b_expand_roots else None
        self.generate_y3 = nn.Linear(256, 8 * 256, bias=False)
        self.generate_y2 = nn.Linear(256, 8 * 128, bias=False)
        self.refine_y3 = self.BuildRefinement(384, 256, 'occ_y3', sparse_basic_block_cls, sparse_conv_bn_relu_cls)
        self.refine_y2 = self.BuildRefinement(192, 128, 'occ_y2', sparse_basic_block_cls, sparse_conv_bn_relu_cls)
        self.classifier = nn.Linear(128, 4)

    @staticmethod
    def BuildRefinement(in_channels, out_channels, key, block_cls, conv_cls):
        projection = conv_cls(in_channels, out_channels, kernel_size=1, conv_type='subm',
                              indice_key=key + '_projection', b_activate=False)
        return nn.Sequential(block_cls(in_channels, out_channels, indice_key=key, downsample=projection),
                             block_cls(out_channels, out_channels, indice_key=key))

    def ZeroConnection(self, *features):
        zero = self.classifier.weight.reshape(-1)[0] * 0.0
        for parameter in self.parameters():
            if parameter.requires_grad and parameter.numel():
                zero = zero + parameter.reshape(-1)[0] * 0.0
        for feature in features:
            zero = zero + feature.sum() * 0.0
        return zero

    def RootInputFeatures(self, root_features):
        """目标车体系0.8m根[N,256]；只归一化删根头输入，不改共享F4或解码输入。"""
        if self.root_input_normalization == 'layer_norm':
            return F.layer_norm(root_features, (256,), weight=None, bias=None, eps=1e-5)
        return root_features

    @staticmethod
    @torch.no_grad()
    def RootGroupStatistics(root_logits, target_delete, feature_statistics=None):
        """所有原始根的分组统计；target=0保留非空根，target=1删除自身为空的根。"""
        logits = root_logits.detach().float()
        targets = target_delete.detach().float()
        if logits.shape != targets.shape:
            raise ValueError('Root diagnostic logits/targets must have matching shapes')
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probabilities = logits.sigmoid()
        result = {}
        for group, mask in (('positive', targets == 0), ('empty', targets == 1)):
            prefix = f'root_{group}_'
            result[prefix + 'count'] = int(mask.sum())
            if not bool(mask.any()):
                continue  # 分母为0时不伪造均值；count=0明确表示该组不适用。
            result[prefix + 'bce_sum'] = float(bce[mask].double().sum())
            result[prefix + 'bce_mean'] = float(bce[mask].mean())
            values = {'logit': logits, 'probability': probabilities, **(feature_statistics or {})}
            for name, value in values.items():
                selected = value.detach().float()[mask]
                result[prefix + name + '_sum'] = float(selected.double().sum())
                result[prefix + name + '_mean'] = float(selected.mean())
                quantiles = torch.quantile(selected, selected.new_tensor([.05, .5, .95]))
                for suffix, quantile in zip(('p05', 'p50', 'p95'), quantiles):
                    result[prefix + name + '_' + suffix] = float(quantile)
                result[prefix + name + '_absmax'] = float(selected.abs().max())
        return result

    def forward(self, x4, x3, x2, b_apply_root_filter=True):
        shape = tuple(map(int, x4.spatial_shape))
        if tuple(v * 4 for v in shape[::-1]) != self.grid_shape_xyz:
            raise ValueError(f'x4 shape {shape} does not match OCC ROI {self.grid_shape_xyz}')
        roots = UniqueRows(x4.indices, shape)
        root_indices, root_features = x4.indices[roots], x4.features[roots]
        if not torch.isfinite(root_features).all():
            raise ValueError('Non-finite shared x4 features')
        root_inputs = self.RootInputFeatures(root_features)
        root_logits = self.root_score(root_inputs).flatten() if self.root_score is not None else root_features.new_empty(0)
        if not torch.isfinite(root_logits).all():
            raise ValueError('Non-finite OCC root deletion logits')
        b_keep = ~(root_logits.sigmoid() > self.delete_threshold) if self.root_score is not None and b_apply_root_filter else torch.ones(len(roots), device=roots.device, dtype=torch.bool)
        zero = self.ZeroConnection(x4.features, x3.features, x2.features)
        result = dict(root_indices=root_indices, root_logits=root_logits, root_keep_mask=b_keep,
                      root_spatial_shape=shape, spatial_shape=tuple(v * 4 for v in shape),
                      batch_size=x4.batch_size, zero_connection=zero, b_expand_roots=self.b_expand_roots)
        if self.b_collect_root_diagnostics and self.root_score is not None:
            result['root_feature_statistics'] = {
                'feature_l2': root_features.detach().norm(dim=1),
                'input_l2': root_inputs.detach().norm(dim=1),
                'feature_absmax': root_features.detach().abs().amax(dim=1),
                'input_absmax': root_inputs.detach().abs().amax(dim=1),
            }
        if not bool(b_keep.any()):
            result.update(indices=root_indices.new_empty((0, 4)), logits=root_features.new_empty((0, 4)) + zero,
                          expanded_root_indices=root_indices.new_empty((0, 4)),
                          counts={'root_count': len(roots), 'kept_root_count': 0, 'deleted_root_count': len(roots), 'expanded_root_count': 0, 'y3_count': 0, 'leaf_count': 0})
            return result
        current = spconv.SparseConvTensor(root_features[b_keep], root_indices[b_keep].int(), shape, x4.batch_size)
        if self.root_expand is not None:
            current = self.root_expand(current)
        expanded_indices = current.indices
        expanded_count = len(expanded_indices)
        for generator, skip, refinement, out_channels in (
            (self.generate_y3, x3, self.refine_y3, 256), (self.generate_y2, x2, self.refine_y2, 128)):
            indices = GenerateChildren(current.indices)
            next_shape = tuple(int(v) * 2 for v in current.spatial_shape)
            if tuple(map(int, skip.spatial_shape)) != next_shape:
                raise ValueError('OCC skip and generated lattice shapes differ')
            if self.b_checkpoint_refinement and self.training and torch.is_grad_enabled():
                current = CheckpointGeneratedStage(
                    generator, refinement, current.features, indices, skip.features, skip.indices,
                    next_shape, x4.batch_size, out_channels,
                )
            else:
                features = F.relu(generator(current.features).reshape(-1, out_channels))
                skip_features = GatherSkip(indices, skip.indices, skip.features, next_shape)
                merged_features = torch.cat((features, skip_features), dim=1)
                current = spconv.SparseConvTensor(merged_features, indices.int(), next_shape, x4.batch_size)
                current = refinement(current)
        result.update(indices=current.indices, logits=self.classifier(current.features) + zero,
                      expanded_root_indices=expanded_indices,
                      counts={'root_count': len(roots), 'kept_root_count': int(b_keep.sum()),
                              'deleted_root_count': int((~b_keep).sum()), 'expanded_root_count': expanded_count, 'y3_count': expanded_count * 8,
                              'leaf_count': len(current.indices)})
        return result

    def PrepareGT(self, batch, device, batch_size=None):
        """Crop validated file-grid XYZ GT into model ROI, return local BZYX + class."""
        coordinate_list, label_list = batch['occ_voxel_indices'], batch['occ_labels']
        expected_count = len(coordinate_list) if batch_size is None else int(batch_size)
        if len(coordinate_list) != expected_count or len(label_list) != expected_count:
            raise ValueError('OCC GT coordinate/label lists must match prediction batch_size')
        for field, width in (('occ_pc_range', 6), ('occ_voxel_size_xyz', 3), ('occ_grid_shape', 3)):
            if tuple(batch[field].shape) != (expected_count, width):
                raise ValueError(f'{field} must have shape [{expected_count},{width}] matching OCC batch_size')
        coordinates, labels = [], []
        origin = torch.tensor(self.pc_range[:3], device=device, dtype=torch.float64)
        resolution = torch.tensor(self.voxel_size_xyz, device=device, dtype=torch.float64)
        shape_xyz = torch.tensor(self.grid_shape_xyz, device=device)
        for batch_index, (xyz, classes) in enumerate(zip(coordinate_list, label_list)):
            xyz, classes = xyz.to(device=device, dtype=torch.long), classes.to(device=device, dtype=torch.long)
            gt_origin = batch['occ_pc_range'][batch_index, :3].to(device=device, dtype=torch.float64)
            gt_resolution = batch['occ_voxel_size_xyz'][batch_index].to(device=device, dtype=torch.float64)
            gt_shape = batch['occ_grid_shape'][batch_index].to(device=device, dtype=torch.long)
            if xyz.ndim != 2 or xyz.shape[1] != 3 or classes.ndim != 1 or len(xyz) != len(classes):
                raise ValueError('Invalid sparse OCC GT coordinate/label shape')
            if not torch.allclose(gt_resolution, resolution, atol=1e-6, rtol=0):
                raise ValueError('OCC GT and leaf resolutions differ')
            if bool(((xyz < 0) | (xyz >= gt_shape)).any()):
                raise ValueError('OCC GT outside its source grid')
            if bool(((classes < 0) | ((classes > 3) & (classes != 255))).any()):
                raise ValueError('Unknown OCC class')
            offset_float = (origin - gt_origin) / resolution
            if not torch.allclose(offset_float, offset_float.round(), atol=1e-4, rtol=0):
                raise ValueError('Model OCC ROI origin is not aligned to GT lattice')
            offset = offset_float.round().long()
            if bool(((offset < 0) | (offset + shape_xyz > gt_shape)).any()):
                raise ValueError('Model OCC ROI must be contained in labeled source grid')
            local_xyz = xyz - offset
            b_inside = ((local_xyz >= 0) & (local_xyz < shape_xyz)).all(dim=1)
            local_xyz, classes = local_xyz[b_inside], classes[b_inside]
            bzyx = torch.cat((torch.full((len(local_xyz), 1), batch_index, device=device, dtype=torch.long), local_xyz[:, [2, 1, 0]]), dim=1)
            coordinates.append(bzyx)
            labels.append(classes)
        if not coordinates:
            return torch.empty((0, 4), dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
        coordinates, labels = torch.cat(coordinates), torch.cat(labels)
        rows = UniqueRows(coordinates, self.grid_shape_xyz[::-1])
        # Duplicate coordinates are legal only if their class agrees.
        matched = MatchRows(coordinates, coordinates[rows], self.grid_shape_xyz[::-1])
        if bool((labels != labels[rows][matched]).any()):
            raise ValueError('Conflicting classes at duplicate OCC GT coordinate')
        return coordinates[rows], labels[rows]

    def Targets(self, pred, batch):
        gt_indices, gt_labels = self.PrepareGT(batch, pred['logits'].device, pred.get('batch_size'))
        rows = MatchRows(pred['indices'], gt_indices, pred['spatial_shape'])
        labels = torch.zeros(len(rows), dtype=torch.long, device=rows.device)
        b_hit = rows >= 0
        labels[b_hit] = gt_labels[rows[b_hit]]
        b_positive = (gt_labels > 0) & (gt_labels < 4)
        parents = gt_indices[b_positive].clone()
        parents[:, 1:] //= 4
        parent_rows = MatchRows(pred['root_indices'], parents, pred['root_spatial_shape'])
        target_delete = (parent_rows < 0).to(dtype=pred['logits'].dtype)
        return labels, target_delete

    def ComputeLoss(self, pred, batch):
        target, target_delete = self.Targets(pred, batch)
        class_loss, parts, valid_count = ClassificationLoss(pred['logits'], target, self.loss_types, self.loss_weights)
        root_count = len(pred['root_logits'])
        root_loss = F.binary_cross_entropy_with_logits(pred['root_logits'], target_delete) if root_count else pred['zero_connection']
        loss = class_loss + self.root_loss_weight * root_loss + pred['zero_connection']
        b_positive = target_delete == 0
        b_keep = pred['root_keep_mask']
        logs = {name: value.detach() for name, value in parts.items()}
        logs.update(pred['counts'])
        logs.update(loss=loss.detach(), class_loss=class_loss.detach(), root_loss=root_loss.detach(),
                    valid_leaf_count=valid_count, root_supervision_count=root_count,
                    occupied_root_count=int(b_positive.sum()), occupied_root_kept=int((b_positive & b_keep).sum()),
                    occupied_root_deleted=int((b_positive & ~b_keep).sum()),
                    empty_root_deleted=int((~b_positive & ~b_keep).sum()), empty_root_kept=int((~b_positive & b_keep).sum()))
        if self.b_collect_root_diagnostics and root_count:
            logs.update(self.RootGroupStatistics(pred['root_logits'], target_delete,
                                                 pred.get('root_feature_statistics')))
        predicted_labels = pred['logits'].detach().argmax(dim=1)
        for category in range(4):
            logs[f'leaf_gt_class_{category}'] = int((target == category).sum())
            logs[f'leaf_pred_class_{category}'] = int((predicted_labels == category).sum())
        return loss, logs, valid_count > 0 or root_count > 0

    @torch.no_grad()
    def ExportPredictions(self, pred, batch):
        from occ_metrics import FrameMetrics
        b_has_gt = 'occ_voxel_indices' in batch and 'occ_labels' in batch
        if b_has_gt:
            gt_indices, gt_labels = self.PrepareGT(batch, pred['logits'].device, pred.get('batch_size'))
        classes = pred['logits'].argmax(dim=1)
        root_targets = None
        if b_has_gt and self.b_collect_root_diagnostics and self.root_score is not None:
            # Own-root labels include ALL original roots, regardless of keep/delete.
            positive_parents = gt_indices[(gt_labels > 0) & (gt_labels < 4)].clone()
            positive_parents[:, 1:] //= 4
            root_targets = (MatchRows(pred['root_indices'], positive_parents,
                                     pred['root_spatial_shape']) < 0).to(pred['root_logits'].dtype)
        outputs = []
        for batch_index in range(pred['batch_size']):
            b_leaf = pred['indices'][:, 0] == batch_index
            b_gt = gt_indices[:, 0] == batch_index if b_has_gt else None
            b_root = pred['root_indices'][:, 0] == batch_index
            indices = pred['indices'][b_leaf, 1:][:, [2, 1, 0]]
            labels = classes[b_leaf]
            roots = pred['root_indices'][b_root, 1:][:, [2, 1, 0]]
            kept_roots = roots[pred['root_keep_mask'][b_root]]
            stage_arguments = {}
            if b_has_gt and self.b_collect_occ_stage_statistics and not self.training:
                b_current_points = batch['points'][:, 0] == batch_index
                b_expanded = pred['expanded_root_indices'][:, 0] == batch_index
                stage_arguments = dict(
                    b_collect_occ_stage_statistics=True,
                    current_points_xyz=batch['points'][b_current_points, 1:4],
                    expanded_root_xyz=pred['expanded_root_indices'][b_expanded, 1:][:, [2, 1, 0]],
                    pc_range=self.pc_range, target_id=batch['token'][batch_index],
                    # PrepareGT checked the complete model ROI lies inside the labeled grid.
                    b_valid_roi_confirmed=True,
                )
            statistics = FrameMetrics(indices, labels, gt_indices[b_gt, 1:][:, [2, 1, 0]], gt_labels[b_gt],
                                      self.grid_shape_xyz, roots, kept_roots, self.b_expand_roots,
                                      **stage_arguments) if b_has_gt else {}
            if root_targets is not None:
                feature_statistics = pred.get('root_feature_statistics')
                frame_features = ({key: value[b_root] for key, value in feature_statistics.items()}
                                  if feature_statistics is not None else None)
                statistics['root_diagnostics'] = self.RootGroupStatistics(
                    pred['root_logits'][b_root], root_targets[b_root], frame_features)
            outputs.append(dict(voxel_indices=indices.cpu(), labels=labels.cpu(),
                                voxel_size_xyz=torch.tensor(self.voxel_size_xyz), pc_range=torch.tensor(self.pc_range),
                                grid_shape=torch.tensor(self.grid_shape_xyz), metrics=statistics))
        return outputs
