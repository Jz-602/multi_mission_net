"""Bounded temporal and conditional-DDP regressions; CPU default, GPU explicit.

--gpu checks actual shared encoder/three-task heads on a tiny 3.2m ROI.
--ddp checks two Gloo ranks on one GPU; it is not a multi-GPU NCCL test.
"""
from __future__ import annotations

import argparse
import ast
from datetime import timedelta
from pathlib import Path
import sys
import tempfile
import types

import torch
from torch import nn
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
import spconv.pytorch as spconv
from easydict import EasyDict
import yaml

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))
from multi_mission_model import MultiMissionModel as SingleModel, SparseBasicBlock, SparseConvBNReLU
from multi_mission_model_memory import MultiMissionModel as TemporalModel
from occ_head import GeneratedOCCHead, MatchRows


def LoadDetachMemory():
    """Exercise the real trainer helper without importing unrelated trainer services."""
    source_path = MODEL_DIR / 'trainer/trainer/trainer_multi.py'
    syntax = ast.parse(source_path.read_text())
    definition = next(node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name == 'DetachMemory')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace['DetachMemory']


def Config():
    config = EasyDict(yaml.safe_load((MODEL_DIR / 'LYF_ODSS_config_multi.yaml').read_text()))
    model = config.model
    model.point_cloud_range = [-1.6, -1.6, -1.6, 1.6, 1.6, 1.6]
    for name in ('tasks', 'voxel_size', 'out_size_factor'):
        model[name] = config.common_variables[name]
    model.post_processing.post_center_limit_range = model.point_cloud_range
    return model


def PoseAndFeatures(device='cpu', batch_size=2):
    current_indices = torch.tensor([[0, 3, 3, 1], [1, 3, 3, 1]][:batch_size], device=device, dtype=torch.int32)
    current_features = torch.randn(batch_size, 128, device=device, requires_grad=True)
    current = spconv.SparseConvTensor(current_features, current_indices, [8, 8, 8], batch_size)
    history_indices = [[0, 3, 3, 3], [0, 3, 0, 3]]
    if batch_size == 2:
        history_indices.append([1, 3, 3, 3])
    history_raw_features = torch.randn(len(history_indices), 128, device=device, requires_grad=True)
    history = dict(coords=torch.tensor(history_indices, device=device, dtype=torch.int32),
                   feats=history_raw_features, spatial_shape=[8, 8, 8], batch_size=batch_size, b_empty=False)
    history = LoadDetachMemory()(history)
    pose = torch.eye(4, device=device).repeat(batch_size, 1, 1)
    # Batch 0: +90deg about z followed by +0.4m x; one history point leaves ROI.
    pose[0, 0, 0], pose[0, 0, 1], pose[0, 1, 0], pose[0, 1, 1] = 0., -1., 1., 0.
    pose[0, 0, 3] = .4
    if batch_size == 2:
        pose[1, 0, 3] = -.4
    return current, history, history_raw_features, dict(pose_to_prev=pose, token=[f'synthetic-{i}' for i in range(batch_size)])


def TestPoseUnionCPU():
    model = TemporalModel.__new__(TemporalModel)
    nn.Module.__init__(model)
    model.config = Config()
    current, history, history_raw_features, batch = PoseAndFeatures()
    aligned = model.AlignMemoryToCurrentX3(batch, history, current)
    expected = {(0, 3, 3, 5), (1, 3, 3, 2)}
    assert set(map(tuple, aligned['coords'].tolist())) == expected
    fused = model.MemoryUpdate(current, aligned, 2)
    assert set(map(tuple, fused['coords'].tolist())) == expected | {(0, 3, 3, 1), (1, 3, 3, 1)}
    assert len(fused['coords']) == 4
    fused['feats'].square().sum().backward()
    assert current.features.grad is not None and current.features.grad.abs().sum() > 0
    assert history_raw_features.grad is None and not history['feats'].requires_grad
    # Two geometrically overlapping rows coalesce; current gradient remains nonzero.
    overlap = dict(coords=current.indices.clone(), feats=torch.ones_like(current.features), distances=torch.full((2,), 100.))
    current.features.grad = None
    fused = model.MemoryUpdate(current, overlap, 2)
    assert len(fused['coords']) == 2
    fused['feats'].sum().backward()
    torch.testing.assert_close(current.features.grad, torch.full_like(current.features, .7))
    assert model.AlignMemoryToCurrentX3(batch, None, current) is None
    print('PASS CPU rigid pose, ROI clipping, batch isolation, historical/current union, overlap weighting and trainer detach boundary', flush=True)


def AssertNestedClose(left, right, path='outputs'):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6, msg=path)
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left: AssertNestedClose(left[key], right[key], path + '.' + str(key))
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)): AssertNestedClose(a, b, path + f'[{index}]')
    else:
        assert left == right, path


def TestNoHistoryGPU():
    torch.manual_seed(19)
    single = SingleModel(Config()).cuda().eval()
    temporal = TemporalModel(Config()).cuda().eval()
    state = temporal.load_state_dict(single.state_dict(), strict=False)
    assert not state.unexpected_keys
    assert state.missing_keys and all(name.startswith('x3_memory_transformer.') for name in state.missing_keys)
    generator = torch.Generator().manual_seed(8)
    xyz = torch.rand((96, 3), generator=generator) * 3.1 - 1.55
    batch = dict(points=torch.cat((torch.zeros((len(xyz), 1)), xyz), dim=1).cuda(), token=['same-state'])
    with torch.no_grad():
        before = single.ForwardNetwork(batch)
        after = temporal.ForwardNetwork(batch, memory=None)
    AssertNestedClose(before['det'], after['det'], 'det')
    AssertNestedClose(before['terrain'], after['terrain'], 'terrain')
    # GPU sparse indice emission order is not a semantic ordering guarantee.
    rows = MatchRows(before['occ']['indices'], after['occ']['indices'], before['occ']['spatial_shape'])
    assert len(before['occ']['indices']) == len(after['occ']['indices']) and bool((rows >= 0).all())
    AssertNestedClose(before['occ']['logits'], after['occ']['logits'][rows], 'occ.logits_by_coordinate')
    for name in ('root_indices', 'root_logits', 'root_keep_mask'):
        AssertNestedClose(before['occ'][name], after['occ'][name], 'occ.' + name)
    assert before['occ']['counts'] == after['occ']['counts']
    print('PASS GPU no-history temporal == single-frame shared-state raw detections, terrain and OCC', before['occ']['counts'], flush=True)


def TestHistoryPathGPU():
    torch.manual_seed(23)
    model = TemporalModel(Config()).cuda().eval()
    current, history, raw_history, batch = PoseAndFeatures('cuda', batch_size=1)
    x2_indices = current.indices.clone()
    x2_indices[:, 1:] *= 2
    x2 = spconv.SparseConvTensor(torch.randn(1, 64, device='cuda', requires_grad=True), x2_indices, [16] * 3, 1)
    model.EncodeCurrent = types.MethodType(lambda self, unused_batch: (x2, current), model)
    captured = {}
    def CaptureFused(_module, args):
        captured['fused_coords'] = args[0].indices.detach().clone()
        captured['fresh_cache'] = not bool(args[0].indice_dict)
    def CaptureX4(_module, _args, output):
        captured['x4_coords'] = output.indices.detach().clone()
    def CaptureBEV(_module, _args, output):
        captured['bev'] = output.detach().clone()
    handles = [model.encoder_stage4.register_forward_pre_hook(CaptureFused),
               model.encoder_stage4.register_forward_hook(CaptureX4),
               model.detection_bev_projector.register_forward_hook(CaptureBEV)]
    with torch.no_grad():
        baseline = model.ForwardNetwork(batch, memory=None)
    baseline_bev = captured['bev'].clone()
    output = model.ForwardNetwork(batch, history)
    assert set(map(tuple, captured['fused_coords'].tolist())) == {(0, 3, 3, 1), (0, 3, 3, 5)}
    assert set(map(tuple, captured['x4_coords'].tolist())) == {(0, 1, 1, 0), (0, 1, 1, 2)}
    assert captured['fresh_cache']
    assert output['occ']['counts']['leaf_count'] > baseline['occ']['counts']['leaf_count']
    assert not torch.allclose(captured['bev'], baseline_bev, atol=1e-8, rtol=1e-5)
    loss = output['occ']['logits'].square().mean() + output['terrain']['terrain_height_pred'].square().mean()
    loss.backward()
    assert current.features.grad is not None and torch.isfinite(current.features.grad).all() and current.features.grad.abs().sum() > 0
    assert x2.features.grad is not None and x2.features.grad.abs().sum() > 0
    assert raw_history.grad is None
    attention_grads = [parameter.grad for parameter in model.x3_memory_transformer.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in attention_grads)
    for handle in handles: handle.remove()
    print('PASS GPU posed history-only x3 -> fresh union -> x4 -> BEV/OCC, current/skip gradient and detached history', output['occ']['counts'], flush=True)


class ConditionalOCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = GeneratedOCCHead(dict(pc_range=[0., 0., 0., 3.2, 3.2, 3.2]), SparseBasicBlock, SparseConvBNReLU)
        with torch.no_grad():
            self.head.root_score.weight.fill_(1 / 256)
            self.head.root_score.bias.zero_()

    def forward(self, sign):
        root_indices = torch.tensor([[0, 1, 1, 1]], device='cuda', dtype=torch.int32)
        sparse = []
        for scale, channels in ((1, 256), (2, 128), (4, 64)):
            indices = root_indices.clone()
            indices[:, 1:] *= scale
            features = torch.full((1, channels), float(sign), device='cuda')
            sparse.append(spconv.SparseConvTensor(features, indices, [4 * scale] * 3, 1))
        pred = self.head(*sparse)
        batch = dict(occ_voxel_indices=[torch.empty((0, 3), dtype=torch.long, device='cuda')],
                     occ_labels=[torch.empty(0, dtype=torch.long, device='cuda')],
                     occ_pc_range=torch.tensor([[0., 0., 0., 3.2, 3.2, 3.2]], device='cuda'),
                     occ_voxel_size_xyz=torch.tensor([[.2] * 3], device='cuda'),
                     occ_grid_shape=torch.tensor([[16] * 3], device='cuda'))
        loss, _, b_valid = self.head.ComputeLoss(pred, batch)
        assert b_valid
        return loss, pred['counts']['leaf_count']


def DDPWorker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    distributed.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        model = torch.nn.parallel.DistributedDataParallel(ConditionalOCC().cuda(), device_ids=[0],
                                                        find_unused_parameters=False, broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, leaf_count = model(1. if rank == 0 else -1.)
            assert (leaf_count == 0) == (rank == 0), (rank, leaf_count)
            loss.backward()
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
            # The all-deleted rank receives the surviving rank's decoder gradient.
            norm = model.module.head.generate_y3.weight.grad.norm()
            assert norm > 0
            norms = [torch.empty_like(norm) for _ in range(2)]
            distributed.all_gather(norms, norm)
            torch.testing.assert_close(norms[0], norms[1])
            optimizer.step()
            print(f'PASS same-GPU Gloo DDP rank={rank} step={step} leaves={leaf_count} loss={float(loss):.6f} shared_decoder_grad={float(norm):.6f}', flush=True)
    finally:
        distributed.destroy_process_group()


def TestConditionalDDP():
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = 'file://' + str(Path(directory) / 'gloo-rendezvous')
        multiprocessing.spawn(DDPWorker, args=(rendezvous,), nprocs=2, join=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--ddp', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    TestPoseUnionCPU()
    if args.gpu:
        TestNoHistoryGPU()
        TestHistoryPathGPU()
    if args.ddp:
        TestConditionalDDP()
