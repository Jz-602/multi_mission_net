"""CPU toy calls and AST checks; no CUDA initialization or model imports."""
import ast
from pathlib import Path
import sys
import time
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_timing import ModuleTiming


def BuildBatchVoxels(value):
    return value + 1


def rotate_nms_pcdet(value):
    return value[:1]


class Event:
    def __init__(self, **kwargs):
        self.timestamp = None

    def record(self):
        self.timestamp = time.perf_counter()

    def elapsed_time(self, other):
        return 1000 * (other.timestamp - self.timestamp)


class CUDA:
    Event = Event
    sync_count = 0

    @classmethod
    def synchronize(cls):
        cls.sync_count += 1


class Block:
    def __call__(self, value):
        return self.forward(value)

    def forward(self, value):
        return value + 1

    def predict(self, value):
        return rotate_nms_pcdet(value)

    def ExportPredictions(self, value):
        return value


class Toy:
    def __init__(self):
        for name in ('conv_input', 'encoder_stage1', 'encoder_stage2', 'encoder_stage3',
                     'encoder_stage4', 'detection_bev_projector', 'detection_neck',
                     'detection_head', 'terrain_head', 'occ_head'):
            setattr(self, name, Block())

    def EncodeCurrent(self, value):
        value = BuildBatchVoxels(value)
        for block in (self.conv_input, self.encoder_stage1, self.encoder_stage2, self.encoder_stage3):
            value = block(value)
        return value

    def ForwardNetwork(self, value):
        value = self.encoder_stage4(self.EncodeCurrent(value))
        shared = self.detection_bev_projector(value)
        return [self.occ_head(value), self.terrain_head(shared), self.detection_head(self.detection_neck(shared))]

    def ExportPredictions(self, outputs, value=None):
        return self.detection_head.predict(outputs), self.occ_head.ExportPredictions(outputs)


class TemporalToy(Toy):
    def __init__(self):
        super().__init__()
        self.x4_memory_transformer = Block()
        self.terrain_bev_projector = Block()

    def AlignMemoryToCurrentX4(self, value):
        return value

    def MemoryUpdate(self, value):
        return value

    def SelectDetectionFeatures(self, value):
        return value

    def ForwardNetwork(self, value):
        value = self.encoder_stage4(self.EncodeCurrent(value))
        value = self.MemoryUpdate(self.x4_memory_transformer(self.AlignMemoryToCurrentX4(value)))
        return [self.occ_head(value), self.terrain_head(self.terrain_bev_projector(value)),
                self.detection_head(self.detection_neck(self.detection_bev_projector(self.SelectDetectionFeatures(value))))]


class TestInferenceTiming(unittest.TestCase):
    def test_output_nesting_shared_projection_and_restore(self):
        model = Toy()
        expected = model.ExportPredictions(model.ForwardNetwork(1))
        original = BuildBatchVoxels
        CUDA.sync_count = 0
        with ModuleTiming(model, CUDA) as timer:
            timer.BeginIteration()
            timer.context = {'phase': 'target'}
            self.assertEqual(model.ExportPredictions(model.ForwardNetwork(1)), expected)
            rows = timer.FinishIteration()
            names = [row['name'] for row in rows]
            self.assertEqual(names.count('bev.shared_projection'), 1)
            self.assertNotIn('bev.detection_projection', names)
            self.assertEqual(CUDA.sync_count, 0)
            nms = next(row for row in rows if row['name'] == 'export.rotated_nms_inclusive')
            self.assertEqual(rows[nms['parent_call_id']]['name'], 'export.CenterHead.predict_inclusive')
            self.assertTrue(all(row['b_completed'] and row['host_inclusive_ms'] >= 0 for row in rows))
            self.assertTrue(all(row['context']['phase'] == 'target' for row in rows))
        self.assertIs(BuildBatchVoxels, original)
        self.assertNotIn('ForwardNetwork', vars(model))
        self.assertNotIn('forward', vars(model.occ_head))
        self.assertEqual(model.ExportPredictions(model.ForwardNetwork(1)), expected)

    def test_real_torch_module_hooks_are_preserved(self):
        import torch
        class RealBlock(torch.nn.Module):
            def forward(self, value):
                return value + 1
            def ExportPredictions(self, value):
                return value
        model = Toy()
        model.occ_head = RealBlock()
        observed = []
        handle = model.occ_head.register_forward_hook(lambda module, inputs, output: observed.append(output))
        try:
            expected = model.ForwardNetwork(1)
            with ModuleTiming(model, CUDA) as timer:
                self.assertEqual(model.ForwardNetwork(1), expected)
                self.assertTrue(timer.FinishIteration())
            self.assertEqual(len(observed), 2)
            self.assertNotIn('forward', vars(model.occ_head))
        finally:
            handle.remove()

    def test_temporal_scopes(self):
        model = TemporalToy()
        expected = model.ForwardNetwork(1)
        with ModuleTiming(model, CUDA) as timer:
            timer.BeginIteration()
            self.assertEqual(model.ForwardNetwork(1), expected)
            names = [row['name'] for row in timer.FinishIteration()]
            for name in ('fusion.AlignMemoryToCurrentX4', 'fusion.MemoryUpdate',
                         'fusion.SelectDetectionFeatures', 'fusion.x4_attention',
                         'bev.terrain_projection', 'bev.detection_projection'):
                self.assertEqual(names.count(name), 1)
            self.assertNotIn('bev.shared_projection', names)

    def test_exception_and_partial_install_restored(self):
        model = Toy()
        original = BuildBatchVoxels
        def Fail(value):
            raise RuntimeError('toy error')
        model.occ_head.forward = Fail
        with self.assertRaisesRegex(RuntimeError, 'toy error'):
            with ModuleTiming(model, CUDA):
                model.ForwardNetwork(1)
        self.assertIs(model.occ_head.forward, Fail)
        self.assertIs(BuildBatchVoxels, original)
        self.assertNotIn('ForwardNetwork', vars(model))
        del model.encoder_stage3
        with self.assertRaises(AttributeError):
            with ModuleTiming(model, CUDA):
                pass
        self.assertIs(BuildBatchVoxels, original)
        self.assertNotIn('forward', vars(model.conv_input))

    def test_real_benchmark_target_has_no_pipeline_sync(self):
        source = Path(__file__).resolve().parents[1] / 'tools/benchmark_three_task_inference.py'
        tree = ast.parse(source.read_text())
        target = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'Target')
        class Model:
            def ForwardNetwork(self, frame):
                return {'counts': {}, 'occ': {'counts': {}}}
            def ExportPredictions(self, outputs, frame):
                return {'unchanged': True}
        for mode, expected_syncs in [('pipeline', 0), ('modules', 0), ('legacy', 3)]:
            CUDA.sync_count = 0
            first, last = Event(), Event()
            first.record(); last.record()
            namespace = dict(args=types.SimpleNamespace(timing_mode=mode), torch=types.SimpleNamespace(cuda=CUDA),
                             model=Model(), X4Model=Toy, time=time, occ_events=[(first, last)])
            exec(compile(ast.Module(body=[target], type_ignores=[]), str(source), 'exec'), namespace)
            memory, row = namespace['Target']({'token': ['test']}, None)
            self.assertEqual(CUDA.sync_count, expected_syncs)
            self.assertEqual(row['token'], 'test')
            self.assertEqual('occ_forward_gpu_ms' in row, mode == 'legacy')
        # New modes don't install legacy OCC hooks or a background RSS sampler.
        text = source.read_text()
        self.assertIn("if args.timing_mode == 'legacy' else []", text)
        self.assertIn("if args.timing_mode == 'legacy' else nullcontext(None)", text)


if __name__ == '__main__':
    unittest.main()
