"""CPU Gloo regression for actual temporal validator's first-history BN buffers.

Compile the production validator/helpers directly from their AST to avoid importing
unrelated TensorFlow/Waymo scoring services. Only device/record/scoring services
are replaced; the model.eval, buffer broadcast, history and target paths are real.
Toy inputs are [2,1] BN features, with 15 history frames and one target per clip.
"""
import ast
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

MODEL_DIR = Path(__file__).resolve().parents[1]
SOURCE_PATH = MODEL_DIR / 'trainer/trainer/trainer_multi.py'


class ToyMemory(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(1, momentum=1.0)
        self.config = {}
        self.history_outputs = []
        self.target_outputs = []

    def forward(self, batch, memory=None, b_memory_only=False):
        features = self.bn(batch['features'])
        if b_memory_only:
            memory = features if memory is None else features + memory
            self.history_outputs.append(memory.detach().clone())
            return memory
        output = features if memory is None else features + memory
        self.target_outputs.append(output.detach().clone())
        return output


class ToyLoader(list):
    dataset = None


def LoadValidator(b_legacy=False):
    syntax = ast.parse(SOURCE_PATH.read_text())
    names = {'GetModelModule', 'DetachMemory', 'ValidateMultiTaskMemory'}
    nodes = [node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == 3
    if b_legacy:
        validator = next(node for node in nodes if node.name == 'ValidateMultiTaskMemory')
        sync_nodes = [node for node in validator.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == 'world_size > 1']
        assert len(sync_nodes) == 1
        validator.body.remove(sync_nodes[0])
    namespace = {'torch': torch, 'dist': dist,
                 'example_to_device': lambda batch, *args, **kwargs: batch,
                 'AccumulateBatchRecords': lambda *args: None,
                 'GatherTokenRecords': lambda records, *args: records,
                 'FinalizeThreeTaskMetrics': lambda *args: {}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE_PATH), 'exec'), namespace)
    return namespace['ValidateMultiTaskMemory']


def Buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def RunValidation(rank, world_size, b_legacy):
    module = ToyMemory()
    model = DistributedDataParallel(module) if world_size > 1 else module
    model.train()
    model({'features': torch.tensor([[0.0], [2.0]]) + rank * 10}).square().sum().backward()
    module.target_outputs.clear()
    parameters_before = [value.detach().clone() for value in module.parameters()]
    train_buffers = Buffers(module)
    features = torch.tensor([[3.0], [5.0]])
    loader = ToyLoader([{'frames': [{'features': features} for _ in range(16)],
                        'warmup_len': 15, 'train_len': 1} for _ in range(2)])
    LoadValidator(b_legacy)(model, loader, torch.device('cpu'), rank, world_size, 1, None)
    parameters_after = list(module.parameters())
    assert all(torch.equal(a, b) for a, b in zip(parameters_before, parameters_after))
    rank0_buffers = {name: value.clone() for name, value in train_buffers.items()}
    if world_size > 1:
        for buffer in rank0_buffers.values():
            dist.broadcast(buffer, src=0)
    checkpoint_bn = torch.nn.BatchNorm1d(1)
    checkpoint_bn.load_state_dict({**module.bn.state_dict(),
                                  'running_mean': rank0_buffers['bn.running_mean'],
                                  'running_var': rank0_buffers['bn.running_var'],
                                  'num_batches_tracked': rank0_buffers['bn.num_batches_tracked']})
    checkpoint_bn.eval()
    with torch.no_grad():
        expected = checkpoint_bn(features) * 16
    outputs = module.target_outputs
    assert len(outputs) == 2 and len(module.history_outputs) == 30
    errors = [float((output - expected).abs().max()) for output in outputs]
    if b_legacy and rank != 0:
        assert errors[0] > 0
    else:
        assert errors[0] < 1e-5
    assert errors[1] < 1e-5
    assert all(torch.equal(value, rank0_buffers[name]) for name, value in Buffers(module).items())
    if rank == 0:
        assert all(torch.equal(value, train_buffers[name]) for name, value in Buffers(module).items())
    return {'rank': rank, 'world_size': world_size, 'b_legacy': b_legacy,
            'train_running_mean': train_buffers['bn.running_mean'].tolist(),
            'target_checkpoint_max_errors': errors,
            'targets': [output.tolist() for output in outputs],
            'history_final': [module.history_outputs[index].tolist() for index in (14, 29)],
            'b_parameters_unchanged': True, 'b_final_buffers_equal_rank0_checkpoint': True}


def Worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        results = [RunValidation(rank, 2, b_legacy) for b_legacy in (True, False)]
        Path(output_dir, f'rank{rank}.json').write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


class TemporalValidationBuffersTest(unittest.TestCase):
    def TestTwoRankActualValidator(self):
        with tempfile.TemporaryDirectory(prefix='validation_buffers_cpu_') as work_dir:
            mp.spawn(Worker, args=(str(Path(work_dir, 'rendezvous')), work_dir), nprocs=2, join=True)
            ranks = [json.loads(Path(work_dir, f'rank{rank}.json').read_text()) for rank in range(2)]
        self.assertNotEqual(ranks[0][0]['train_running_mean'], ranks[1][0]['train_running_mean'])
        self.assertNotEqual(ranks[0][0]['history_final'][0], ranks[1][0]['history_final'][0])
        self.assertEqual(ranks[0][1]['history_final'], ranks[1][1]['history_final'])
        self.assertEqual(ranks[0][1]['targets'], ranks[1][1]['targets'])
        self.assertEqual(ranks[0][0]['targets'], ranks[0][1]['targets'])
        print(json.dumps({'two_rank_results': ranks}))

    def TestWorldOneNoCollective(self):
        broadcast = dist.broadcast
        def RejectBroadcast(*args, **kwargs):
            raise AssertionError('world1 validator must not broadcast')
        dist.broadcast = RejectBroadcast
        try:
            result = RunValidation(0, 1, False)
        finally:
            dist.broadcast = broadcast
        self.assertTrue(result['b_parameters_unchanged'])
        print(json.dumps({'world1_result': result}))


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_loader = unittest.TestLoader()
    test_loader.testMethodPrefix = 'Test'
    unittest.main(testLoader=test_loader)
