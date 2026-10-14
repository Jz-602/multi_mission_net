"""Activation-checkpoint equivalence including exact BN buffer preservation.

CPU tests the fixed-support checkpoint wrapper and all-deleted safety.
--gpu tests the actual complete OCC head; --ddp tests asymmetric Gloo ranks.
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
from pathlib import Path
import sys
import tempfile

import torch
from torch import nn
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
import spconv.pytorch as spconv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occ_head import (CheckpointSparseRefinement, CheckpointGeneratedStage, GenerateChildren, GatherSkip,
                      MatchRows, UniqueRows, GeneratedOCCHead)
from test_occ_head import MakeHead, SparseInput, Batch


def BatchNormState(module):
    return {name: value.detach().clone() for name, value in module.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def AssertBuffersEqual(expected, module):
    actual = BatchNormState(module)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name


class CPUSparseRefinement(nn.Module):
    """Exercise real BN/checkpoint/cache handling without requiring CPU sparse GEMM."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 12), nn.BatchNorm1d(12, momentum=None), nn.ReLU(),
                                    nn.Linear(12, 8), nn.BatchNorm1d(8))
        self.calls = 0

    def forward(self, sparse):
        assert not sparse.indice_dict, 'Checkpoint reused stale sparse indice cache'
        sparse.indice_dict['test-support-cache'] = True
        self.calls += 1
        return sparse.replace_feature(self.layers(sparse.features))


def TestCPU():
    torch.manual_seed(53)
    baseline = CPUSparseRefinement().train()
    checked = copy.deepcopy(baseline)
    values = torch.randn((12, 8))
    source = values.clone().requires_grad_()
    source_checked = values.clone().requires_grad_()
    coordinates = torch.tensor([[0, 0, index // 4, index % 4] for index in range(12)], dtype=torch.int32)
    expected = baseline(spconv.SparseConvTensor(source, coordinates, [4] * 3, 1)).features
    actual = CheckpointSparseRefinement(checked, source_checked, coordinates, [4] * 3, 1)
    assert not actual.indice_dict
    torch.testing.assert_close(expected, actual.features, rtol=0, atol=0)
    checked_forward_buffers = BatchNormState(checked)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual.features * probe).sum().backward()
    assert checked.calls == 2 and baseline.calls == 1
    AssertBuffersEqual(checked_forward_buffers, checked)
    AssertBuffersEqual(BatchNormState(baseline), checked)
    torch.testing.assert_close(source.grad, source_checked.grad, rtol=0, atol=0)
    for (name, parameter), (checked_name, checked_parameter) in zip(baseline.named_parameters(), checked.named_parameters()):
        assert name == checked_name
        torch.testing.assert_close(parameter.grad, checked_parameter.grad, rtol=0, atol=0)
    # Empty candidate branch remains safe when the feature checkpoint option is enabled.
    head = MakeHead()
    head.b_checkpoint_refinement = True
    with torch.no_grad(): head.root_score.bias.fill_(10.)
    inputs = SparseInput()
    pred = head(*inputs)
    loss, _, b_valid = head.ComputeLoss(pred, Batch([[7, 7, 7]], [3]))
    loss.backward()
    assert b_valid and pred['counts']['leaf_count'] == 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
    print('PASS CPU checkpoint outputs/input+parameter gradients bitwise equal, BN EMA/cumulative stats restored exactly, fresh caches and all-deleted connections', flush=True)




class CPUDecoderBlock(CPUSparseRefinement):
    def forward(self, sparse):
        self.calls += 1
        return sparse.replace_feature(self.layers(sparse.features))


def TestDecoderBoundaryCPU():
    torch.manual_seed(61)
    baseline = nn.ModuleDict(dict(generator=nn.Linear(8, 32, bias=False),
                                  refinement=nn.Sequential(CPUDecoderBlock(), CPUDecoderBlock())))
    checked = copy.deepcopy(baseline)
    parent_indices = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.int32)
    child_indices = GenerateChildren(parent_indices)
    skip_indices = child_indices[[0, 8, 15]]
    parent = torch.randn(2, 8, requires_grad=True)
    skip = torch.randn(3, 4, requires_grad=True)
    checked_parent = parent.detach().clone().requires_grad_()
    checked_skip = skip.detach().clone().requires_grad_()
    generated = torch.relu(baseline['generator'](parent).reshape(-1, 4))
    merged = torch.cat((generated, GatherSkip(child_indices, skip_indices, skip, [4] * 3)), dim=1)
    expected = baseline['refinement'](spconv.SparseConvTensor(merged, child_indices, [4] * 3, 1)).features
    actual = CheckpointGeneratedStage(checked['generator'], checked['refinement'], checked_parent,
        child_indices, checked_skip, skip_indices, [4] * 3, 1, 4).features
    buffers_after_forward = BatchNormState(checked)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    AssertBuffersEqual(buffers_after_forward, checked)
    AssertBuffersEqual(BatchNormState(baseline), checked)
    torch.testing.assert_close(parent.grad, checked_parent.grad, rtol=0, atol=0)
    torch.testing.assert_close(skip.grad, checked_skip.grad, rtol=0, atol=0)
    for (name, parameter), (checked_name, checked_parameter) in zip(baseline.named_parameters(), checked.named_parameters()):
        assert name == checked_name
        torch.testing.assert_close(parameter.grad, checked_parameter.grad, rtol=0, atol=0, msg=name)
    assert all(block.calls == 2 for block in checked['refinement'])
    print('PASS CPU generation+skip+split-residual checkpoint boundary: exact output/all gradients/BN, both blocks recomputed', flush=True)


def TestInputGuards():
    def ExpectValueError(callback, expected_message):
        try:
            callback()
        except ValueError as error:
            assert expected_message in str(error), str(error)
        else:
            raise AssertionError('Invalid OCC geometry/batch was silently accepted')

    # Geometry rejection happens before factories are used.
    ExpectValueError(lambda: GeneratedOCCHead({'pc_range': [0, 0, 0, 3.25, 3.2, 3.2]}, None, None), 'integer number')
    head = MakeHead()
    valid = Batch([], [])
    ExpectValueError(lambda: head.PrepareGT(valid, 'cpu', batch_size=2), 'batch_size')
    mismatched = dict(valid, occ_labels=[])
    ExpectValueError(lambda: head.PrepareGT(mismatched, 'cpu'), 'batch_size')
    for field in ('occ_pc_range', 'occ_voxel_size_xyz', 'occ_grid_shape'):
        mismatched = dict(valid)
        mismatched[field] = valid[field].repeat(2, 1)
        ExpectValueError(lambda: head.PrepareGT(mismatched, 'cpu'), field)
    coordinates, labels = head.PrepareGT(valid, 'cpu', batch_size=1)
    assert coordinates.shape == (0, 4) and labels.shape == (0,)
    print('PASS non-grid-aligned ROI and unequal GT-list/metadata/prediction batch sizes rejected', flush=True)


def CloneSparse(sparse):
    return spconv.SparseConvTensor(sparse.features.detach().clone().requires_grad_(), sparse.indices.clone(),
                                   sparse.spatial_shape, sparse.batch_size)


def TestGPU():
    torch.manual_seed(59)
    baseline = MakeHead().cuda().train()
    checked = copy.deepcopy(baseline)
    checked.b_checkpoint_refinement = True
    def StableRootOrder(module, inputs, sparse):
        # Test-only: spconv expansion emits [N,4] vehicle-grid BZYX rows in an
        # atomic order. Stabilize before BN: row-dependent rounding otherwise
        # perturbs nonsmooth gradients even between two ordinary forwards.
        rows = UniqueRows(sparse.indices, sparse.spatial_shape)
        assert len(rows) == len(sparse.indices)
        return spconv.SparseConvTensor(sparse.features[rows], sparse.indices[rows],
                                       sparse.spatial_shape, sparse.batch_size)
    baseline.root_expand.conv.register_forward_hook(StableRootOrder)
    checked.root_expand.conv.register_forward_hook(StableRootOrder)
    inputs = SparseInput('cuda')
    checked_inputs = [CloneSparse(sparse) for sparse in inputs]
    pred = baseline(*inputs)
    checked_pred = checked(*checked_inputs)
    assert pred['counts'] == checked_pred['counts']
    rows = MatchRows(pred['indices'], checked_pred['indices'], pred['spatial_shape'])
    assert bool((rows >= 0).all())
    torch.testing.assert_close(pred['logits'], checked_pred['logits'][rows], rtol=2e-4, atol=2e-5)
    checked_forward_buffers = BatchNormState(checked)
    batch = Batch([[7, 7, 7]], [3], 'cuda')
    loss, _, _ = baseline.ComputeLoss(pred, batch)
    checked_loss, _, _ = checked.ComputeLoss(checked_pred, batch)
    torch.testing.assert_close(loss, checked_loss, rtol=1e-5, atol=1e-6)
    loss.backward()
    checked_loss.backward()
    AssertBuffersEqual(checked_forward_buffers, checked)
    max_gradient_difference = 0.
    for (name, parameter), (checked_name, checked_parameter) in zip(baseline.named_parameters(), checked.named_parameters()):
        assert name == checked_name and parameter.grad is not None and checked_parameter.grad is not None
        torch.testing.assert_close(parameter.grad, checked_parameter.grad, rtol=1e-3, atol=3e-5, msg=name)
        max_gradient_difference = max(max_gradient_difference, float((parameter.grad - checked_parameter.grad).abs().max()))
    for original, check in zip(inputs, checked_inputs):
        torch.testing.assert_close(original.features.grad, check.features.grad, rtol=1e-3, atol=3e-5)
    for name, expected in BatchNormState(baseline).items():
        actual = BatchNormState(checked)[name]
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6, msg=name)
    print('PASS GPU complete OCC checkpoint on/off support, logits, loss, every parameter/input gradient and exact post-forward BN buffer preservation',
          dict(counts=pred['counts'], max_parameter_gradient_difference=max_gradient_difference), flush=True)


def DDPWorker(rank, rendezvous):
    from test_three_task_temporal import ConditionalOCC
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    distributed.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        local = ConditionalOCC().cuda()
        local.head.b_checkpoint_refinement = True
        model = nn.parallel.DistributedDataParallel(local, device_ids=[0], find_unused_parameters=False, broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, leaves = model(1. if rank == 0 else -1.)
            assert (leaves == 0) == (rank == 0)
            buffers = BatchNormState(local)
            loss.backward()
            AssertBuffersEqual(buffers, local)
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
            norm = local.head.generate_y3.weight.grad.norm()
            assert norm > 0
            shared_norms = [torch.empty_like(norm) for _ in range(2)]
            distributed.all_gather(shared_norms, norm)
            torch.testing.assert_close(shared_norms[0], shared_norms[1])
            optimizer.step()
            print(f'PASS checkpoint same-GPU Gloo DDP rank={rank} step={step} leaves={leaves} BN buffers unchanged by backward', flush=True)
    finally:
        distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--ddp', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    TestCPU()
    TestDecoderBoundaryCPU()
    TestInputGuards()
    if args.gpu: TestGPU()
    if args.ddp:
        with tempfile.TemporaryDirectory() as directory:
            multiprocessing.spawn(DDPWorker, args=('file://' + str(Path(directory) / 'rendezvous'),), nprocs=2, join=True)
