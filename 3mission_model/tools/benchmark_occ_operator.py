"""Linear eight-child generation versus true generative transpose convolution.

CPU verifies against dense ConvTranspose3d; --gpu checks installed spconv and
records synchronized operator-only forward/backward/first-AdamW-step resources.
These local checks do not measure the complete OCC decoder or model.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
import spconv.pytorch as spconv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from occ_head import GenerateChildren, MatchRows


def Inputs(device, root_count):
    if root_count < 1 or root_count > 2 * 16**3:
        raise ValueError('root_count must be within [1,8192]')
    generator = torch.Generator().manual_seed(37)
    flat = torch.randperm(2 * 16**3, generator=generator)[:root_count]
    indices = torch.stack((flat // (16**3), (flat // 256) % 16, (flat // 16) % 16, flat % 16), dim=1)
    return indices.to(device=device, dtype=torch.int32), torch.randn((root_count, 256), generator=generator, device='cpu').to(device)


def SparseWeightFromLinear(weight, out_channels):
    # Local spconv 2.3.6 default implicit-GEMM transpose uses KRSC:
    # [C_out,kz,ky,kx,C_in], whereas Linear groups offsets before C_out.
    return weight.reshape(2, 2, 2, out_channels, 256).permute(3, 0, 1, 2, 4).contiguous()


def CheckEquivalence(device='cpu'):
    summaries = []
    for out_channels in (256, 128):
        torch.manual_seed(41 + out_channels)
        indices, input_values = Inputs(device, 4)
        # CPU dense reference remains tiny (4^3 input) but has two batch members.
        indices = torch.tensor([[0, 0, 0, 0], [0, 2, 1, 3], [1, 1, 1, 1], [1, 3, 3, 3]], device=device, dtype=torch.int32)
        linear = nn.Linear(256, 8 * out_channels, bias=False).to(device)
        explicit_input = input_values.detach().clone().requires_grad_()
        reference_input = input_values.detach().clone().requires_grad_()
        child_indices = GenerateChildren(indices)
        explicit_output = linear(explicit_input).reshape(-1, out_channels)
        if device == 'cuda':
            reference = spconv.SparseConvTranspose3d(256, out_channels, kernel_size=2, stride=2,
                padding=0, bias=False, indice_key='operator_check').to(device)
            assert tuple(reference.weight.shape) == (out_channels, 2, 2, 2, 256), reference.weight.shape
            with torch.no_grad(): reference.weight.copy_(SparseWeightFromLinear(linear.weight, out_channels))
            tensor = spconv.SparseConvTensor(reference_input, indices, [4] * 3, 2)
            output = reference(tensor)
            rows = MatchRows(child_indices, output.indices, [8] * 3)
            assert len(output.indices) == len(child_indices) and bool((rows >= 0).all())
            reference_output = output.features[rows]
        else:
            reference = nn.ConvTranspose3d(256, out_channels, kernel_size=2, stride=2, padding=0, bias=False)
            with torch.no_grad():
                reference.weight.copy_(linear.weight.reshape(2, 2, 2, out_channels, 256).permute(4, 3, 0, 1, 2))
            dense_input = reference_input.new_zeros((2, 256, 4, 4, 4))
            b, z, y, x = indices.long().unbind(1)
            dense_input[b, :, z, y, x] = reference_input
            dense_output = reference(dense_input)
            b, z, y, x = child_indices.long().unbind(1)
            reference_output = dense_output[b, :, z, y, x]
        probe = torch.randn_like(explicit_output)
        (explicit_output * probe).sum().backward()
        (reference_output * probe).sum().backward()
        if device == 'cuda':
            reference_weight_gradient = reference.weight.grad.permute(1, 2, 3, 0, 4).reshape_as(linear.weight)
        else:
            reference_weight_gradient = reference.weight.grad.permute(2, 3, 4, 1, 0).reshape_as(linear.weight)
        errors = dict(forward=float((explicit_output - reference_output).abs().max()),
                      input_gradient=float((explicit_input.grad - reference_input.grad).abs().max()),
                      weight_gradient=float((linear.weight.grad - reference_weight_gradient).abs().max()))
        torch.testing.assert_close(explicit_output, reference_output, rtol=2e-4, atol=2e-5)
        torch.testing.assert_close(explicit_input.grad, reference_input.grad, rtol=3e-4, atol=1e-4)
        torch.testing.assert_close(linear.weight.grad, reference_weight_gradient, rtol=2e-4, atol=2e-5)
        summaries.append(dict(device=device, in_channels=256, out_channels=out_channels,
                              roots=len(indices), leaves=len(child_indices), max_abs_errors=errors))
    return summaries


def MeasureGPU(method, out_channels, root_count, iterations):
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(43)
    indices, features = Inputs('cuda', root_count)
    if method == 'linear':
        operator = nn.Linear(256, 8 * out_channels, bias=False).cuda()
    else:
        operator = spconv.SparseConvTranspose3d(256, out_channels, kernel_size=2, stride=2,
                                               padding=0, bias=False, indice_key='benchmark_transpose').cuda()
    def Forward(values):
        if method == 'linear':
            return operator(values).reshape(-1, out_channels), GenerateChildren(indices)
        sparse = operator(spconv.SparseConvTensor(values, indices, [16] * 3, 2))
        return sparse.features, sparse.indices
    # Warm up kernel selection and allocation, but not optimizer state allocation.
    values = features.detach().requires_grad_()
    prediction, coordinates = Forward(values)
    prediction.square().mean().backward()
    operator.zero_grad(set_to_none=True)
    del values, prediction, coordinates
    torch.cuda.synchronize()
    optimizer = torch.optim.AdamW(operator.parameters(), lr=.001)
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        values = features.detach().requires_grad_()
        prediction, coordinates = Forward(values)
        prediction.square().mean().backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return dict(method=method, roots=root_count, leaves=len(coordinates), in_channels=256,
                out_channels=out_channels, dtype=str(features.dtype), iterations=iterations,
                mean_forward_backward_step_ms=1000 * elapsed / iterations,
                baseline_allocated_mib=baseline_bytes / 2**20,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                peak_extra_allocated_mib=(torch.cuda.max_memory_allocated() - baseline_bytes) / 2**20,
                scope='operator, coordinates, square loss, backward, first AdamW state allocation; no decoder refinement/skips')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--roots', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.iterations < 1: raise ValueError('iterations must be positive')
    torch.set_num_threads(2)
    results = {'equivalence': CheckEquivalence('cpu')}
    if args.gpu:
        results['device'] = torch.cuda.get_device_name(0)
        results['equivalence'].extend(CheckEquivalence('cuda'))
        results['resource_smoke'] = [MeasureGPU(method, channels, args.roots, args.iterations)
                                     for channels in (256, 128) for method in ('linear', 'spconv_transpose')]
    encoded = json.dumps(results, indent=2)
    print(encoded, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n')
