"""CPU geometric/loss/metric regressions; --gpu checks actual sparse decoding."""
import argparse
import ast
from typing import Optional
import itertools
import sys
from pathlib import Path

import torch
import spconv.pytorch as spconv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occ_head import ClassificationLoss, GenerateChildren, GatherSkip, GeneratedOCCHead, MatchRows
from occ_metrics import FrameMetrics, SummarizeMetrics


def MakeHead(b_expand_roots=True, b_enable_root_filter=True):
    # Load the existing block definitions without importing detection CUDA extensions.
    source_path = Path(__file__).resolve().parents[1] / 'multi_mission_model.py'
    syntax = ast.parse(source_path.read_text())
    definitions = [node for node in syntax.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                   and node.name in {'ReplaceFeature', 'SparseBasicBlock', 'SparseConvBNReLU'}]
    namespace = dict(torch=torch, nn=torch.nn, spconv=spconv, Optional=Optional,
                     SparseConvTensor=spconv.SparseConvTensor, SparseModule=spconv.SparseModule, Tensor=torch.Tensor)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source_path), 'exec'), namespace)
    SparseBasicBlock, SparseConvBNReLU = namespace['SparseBasicBlock'], namespace['SparseConvBNReLU']
    return GeneratedOCCHead(dict(pc_range=[0, 0, 0, 3.2, 3.2, 3.2],
        b_expand_roots=b_expand_roots, b_enable_root_filter=b_enable_root_filter),
        SparseBasicBlock, SparseConvBNReLU)


def Batch(gt_xyz, labels, device='cpu'):
    return dict(occ_voxel_indices=[torch.tensor(gt_xyz, device=device, dtype=torch.long).reshape(-1, 3)],
        occ_labels=[torch.tensor(labels, device=device, dtype=torch.long)],
        occ_pc_range=torch.tensor([[0., 0., 0., 3.2, 3.2, 3.2]], device=device),
        occ_voxel_size_xyz=torch.tensor([[.2, .2, .2]], device=device),
        occ_grid_shape=torch.tensor([[16, 16, 16]], device=device))


def SparseInput(device='cpu'):
    root_indices = torch.tensor([[0, 1, 1, 1], [0, 2, 2, 2]], dtype=torch.int32, device=device)
    values = []
    for scale, channels in ((1, 256), (2, 128), (4, 64)):
        coords = root_indices.clone()
        coords[:, 1:] *= scale
        features = torch.randn(2, channels, device=device, requires_grad=True)
        values.append(spconv.SparseConvTensor(features, coords, [4 * scale] * 3, 1))
    return values


def TestGeometry():
    parents = torch.tensor([[0, 1, 2, 3], [1, 1, 2, 3]], dtype=torch.int32)
    children = GenerateChildren(parents)
    grandchildren = GenerateChildren(children)
    assert len(children) == 16 and len(grandchildren) == 128
    assert len(set(map(tuple, grandchildren.tolist()))) == 128
    expected = {(0, z, y, x) for z in range(4, 8) for y in range(8, 12) for x in range(12, 16)}
    assert set(map(tuple, grandchildren[:64].tolist())) == expected
    assert torch.equal(grandchildren[:, 1:] // 4, parents[:, None, 1:].expand(-1, 64, -1).reshape(-1, 3))
    source = children[[10, 1, 8]]
    features = torch.tensor([[10., 11.], [1., 2.], [8., 9.]], requires_grad=True)
    gathered = GatherSkip(children, source, features, [8, 8, 8])
    assert torch.equal(gathered[[10, 1, 8]], features)
    assert gathered[0].eq(0).all()
    gathered.sum().backward()
    assert features.grad.eq(1).all()
    assert MatchRows(children, source.new_empty((0, 4)), [8] * 3).eq(-1).all()
    print('PASS child coordinates, complete empty-space coverage, shuffled skip lookup, batch isolation and gradients')


def TestLosses():
    targets = torch.tensor([0, 1, 2, 3])
    bad = torch.zeros((4, 4), requires_grad=True)
    good = torch.eye(4) * 10
    bad_loss, bad_parts, count = ClassificationLoss(bad, targets)
    good_loss, good_parts, _ = ClassificationLoss(good, targets)
    assert count == 4 and good_loss < bad_loss
    assert all(good_parts[key] < bad_parts[key] for key in bad_parts)
    expected_ce = torch.tensor(4.).log()
    assert torch.allclose(bad_parts['CELoss'], expected_ce)
    # Dice at p=1/4: (2*.25+1)/(4*.25**2+1+1)=2/3, loss=1/3.
    assert torch.allclose(bad_parts['DiceLossV1'], torch.tensor(1 / 3))
    empty_gt_loss, _, count = ClassificationLoss(torch.zeros((3, 4)), torch.zeros(3, dtype=torch.long))
    assert count == 3 and empty_gt_loss > 0
    for logits, target in ((torch.empty((0, 4), requires_grad=True), torch.empty(0, dtype=torch.long)),
                           (torch.randn(3, 4, requires_grad=True), torch.full((3,), 255))):
        loss, _, count = ClassificationLoss(logits, target)
        assert count == 0 and loss.item() == 0 and loss.requires_grad
        loss.backward()
        assert logits.grad is not None and logits.grad.eq(0).all()
    bad_loss.backward()
    assert bad.grad[0, 0] < 0
    print('PASS CE hand value, Dice hand value, all three class losses improve, empty class trained, empty/all-ignore connected zero')


def TestMetrics():
    pred_xyz = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 2], [3, 3, 3]])
    pred_labels = torch.tensor([1, 2, 3, 3, 3])
    gt_xyz = torch.tensor([[0, 0, 0], [1, 1, 1], [4, 4, 4], [3, 3, 3]])
    gt_labels = torch.tensor([1, 3, 2, 255])
    stats = FrameMetrics(pred_xyz, pred_labels, gt_xyz, gt_labels, (8, 8, 8))
    assert stats['occupancy_tp'] == 2 and stats['occupancy_fp'] == 1 and stats['occupancy_fn'] == 1
    assert stats['tp'][1:] == [1, 0, 0] and stats['fp'][1:] == [0, 1, 1] and stats['fn'][1:] == [0, 1, 1]
    assert sum(map(sum, stats['confusion'])) == 8**3 - 1
    # Independent dense-grid count serves as a reference only on this tiny ROI.
    pred_dense, gt_dense = torch.zeros(8, 8, 8, dtype=torch.long), torch.zeros(8, 8, 8, dtype=torch.long)
    for xyz, label in zip(pred_xyz, pred_labels): pred_dense[tuple(xyz)] = label
    for xyz, label in zip(gt_xyz, gt_labels): gt_dense[tuple(xyz)] = label
    b_valid = gt_dense != 255
    dense_confusion = torch.bincount(gt_dense[b_valid] * 4 + pred_dense[b_valid], minlength=16).reshape(4, 4)
    assert dense_confusion.tolist() == stats['confusion']
    summary = SummarizeMetrics([stats])
    assert summary['occupancy_iou'] == .5
    empty = torch.empty((0, 3), dtype=torch.long)
    roots = torch.tensor([[0, 0, 0], [1, 0, 0]])
    # Deleted positive root remains covered by surviving empty neighbor; unreachable GT remains FN.
    stats = FrameMetrics(empty, torch.empty(0, dtype=torch.long), torch.tensor([[0, 0, 0], [15, 15, 15]]),
                         torch.tensor([1, 2]), (16, 16, 16), roots, roots[1:], True)
    assert stats['reachable_gt'] == 1 and stats['removed_gt'] == 0 and stats['unreachable_gt'] == 1
    assert stats['occupied_root_deleted'] == 1 and stats['covered_gt'] == 1
    stats = FrameMetrics(empty, torch.empty(0, dtype=torch.long), torch.tensor([[0, 0, 0]]),
                         torch.tensor([1]), (16, 16, 16), roots, empty, True)
    assert stats['removed_gt'] == 1
    print('PASS sparse/dense full-grid confusion, missing-candidate FN, ignored voxels, deterministic dedup, root/coverage decomposition')


def TestRootAndEmpty():
    head = MakeHead()
    inputs = SparseInput()
    roots = inputs[0].indices
    pred = dict(indices=GenerateChildren(GenerateChildren(roots)), root_indices=roots,
                logits=torch.zeros((128, 4)), spatial_shape=(16, 16, 16), root_spatial_shape=(4, 4, 4))
    labels, root_targets = head.Targets(pred, Batch([[7, 7, 7], [0, 0, 0]], [3, 2]))
    assert root_targets.tolist() == [0., 1.]
    assert (labels == 3).sum() == 1  # GT absent from encoder skip still receives supervision.
    assert torch.allclose(head.root_score(inputs[0].features).sigmoid(), torch.full((2, 1), .01))
    with torch.no_grad(): head.root_score.bias.fill_(10.)
    output = head(*inputs)
    assert output['logits'].shape == (0, 4) and output['counts']['kept_root_count'] == 0
    assert head.ExportPredictions(output, {})[0]['metrics'] == {}
    loss, logs, b_valid = head.ComputeLoss(output, Batch([[7, 7, 7]], [3]))
    assert b_valid and logs['root_supervision_count'] == 2 and logs['valid_leaf_count'] == 0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.root_score.bias.grad.abs().sum() > 0
    for name, parameter in head.named_parameters():
        if not name.startswith('root_score'):
            assert parameter.grad.eq(0).all(), name
    assert all(item.features.grad is not None for item in inputs)
    head.zero_grad(set_to_none=True)
    empty_inputs = [spconv.SparseConvTensor(torch.empty((0, channels), requires_grad=True),
                    torch.empty((0, 4), dtype=torch.int32), [size] * 3, 1)
                    for channels, size in ((256, 4), (128, 8), (64, 16))]
    empty_prediction = head(*empty_inputs)
    empty_loss, _, b_empty_valid = head.ComputeLoss(empty_prediction, Batch([], []))
    assert not b_empty_valid and empty_loss.item() == 0
    empty_loss.backward()
    assert all(parameter.grad is not None and parameter.grad.eq(0).all() for parameter in head.parameters())
    disabled = MakeHead(b_enable_root_filter=False)
    assert not any('root_score' in name for name, _ in disabled.named_parameters())
    # Nonzero learned score weights must pass root BCE gradients into x4.
    head.zero_grad(set_to_none=True)
    inputs = SparseInput()
    with torch.no_grad(): head.root_score.weight.fill_(.01)
    output = head(*inputs)
    loss, _, _ = head.ComputeLoss(output, Batch([[7, 7, 7]], [3]))
    loss.backward()
    assert inputs[0].features.grad.abs().sum() > 0
    print('PASS own-root GT target, new-location GT, all-keep init, all-deleted all-parameter gradients, learned score x4 gradient, root ablation removes parameters')


def TestGPU():
    device = 'cuda'
    head = MakeHead().to(device)
    inputs = SparseInput(device)
    prediction = head(*inputs)
    expected_roots = set()
    for row in inputs[0].indices.tolist():
        for offset in itertools.product((-1, 0, 1), repeat=3):
            coord = tuple(row[axis + 1] + offset[axis] for axis in range(3))
            if all(0 <= value < 4 for value in coord): expected_roots.add((0,) + coord)
    assert set(map(tuple, prediction['expanded_root_indices'].tolist())) == expected_roots
    assert len(prediction['indices']) == len(expected_roots) * 64
    assert not inputs[0].indice_dict
    loss, logs, b_valid = head.ComputeLoss(prediction, Batch([[7, 7, 7]], [3], device))
    loss.backward()
    assert b_valid and torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    for name in ('generate_y3', 'generate_y2', 'classifier'):
        assert getattr(head, name).weight.grad.abs().sum() > 0
    assert all(item.features.grad.abs().sum() > 0 for item in inputs)
    assert len(head.ExportPredictions(prediction, Batch([[7, 7, 7]], [3], device))) == 1
    print('PASS GPU ring coordinates, 8/64-fold decode, original cache preserved, all path gradients and frame metric export', prediction['counts'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    arguments = parser.parse_args()
    TestGeometry()
    TestLosses()
    TestMetrics()
    TestRootAndEmpty()
    if arguments.gpu: TestGPU()
