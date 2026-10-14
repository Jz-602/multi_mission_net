"""CPU X4 geometry, exact S initialization and task-support isolation checks."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from spconv.pytorch import SparseConvTensor
from easydict import EasyDict
import yaml

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))
from multi_mission_model import MultiMissionModel as SingleModel
from multi_mission_model_memory import X3MemoryCrossAttentionTransformer
from multi_mission_model_memory_x4 import MultiMissionModel
from occ_head import MatchRows


def Config():
    cfg = EasyDict(yaml.safe_load((MODEL_DIR/'LYF_ODSS_config_multi.yaml').read_text()))
    for key in ('tasks', 'voxel_size', 'out_size_factor'): cfg.model[key] = cfg.common_variables[key]
    cfg.model.point_cloud_range = [0., 0., 0., 3.2, 3.2, 3.2]
    cfg.model.post_processing.post_center_limit_range = cfg.model.point_cloud_range
    cfg.model.in_channels = 4
    return cfg.model


def Sparse(coords, value=1., channels=256, shape=(4, 4, 4), batch_size=1):
    indices = torch.tensor(coords, dtype=torch.int32)
    features = torch.full((len(coords), channels), value, requires_grad=True)
    return SparseConvTensor(features, indices, shape, batch_size)


class X4MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = Config()
        cls.model = MultiMissionModel(cls.config)

    def TestSingleLoadAndProjectorIsolation(self):
        single = SingleModel(self.config)
        with torch.no_grad():
            single.detection_bev_projector.height_compress[1].running_mean.fill_(.75)
        report = self.model.LoadSingleState(single.state_dict())
        actual = self.model.state_dict()
        self.assertEqual(report['matched_shared_count'], len(single.state_dict()))
        for key, expected in single.state_dict().items(): torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
        det = self.model.detection_bev_projector.state_dict()
        terrain = self.model.terrain_bev_projector.state_dict()
        for key in det:
            torch.testing.assert_close(det[key], terrain[key], rtol=0, atol=0)
            self.assertNotEqual(det[key].data_ptr(), terrain[key].data_ptr())
        invalid = dict(single.state_dict()); invalid.pop('conv_input.0.conv.weight')
        with self.assertRaises(RuntimeError): self.model.LoadSingleState(invalid)
        self.assertEqual(self.model.x4_memory_transformer.norm_query_list[0].normalized_shape, (256,))
        torch.testing.assert_close(self.model.x4_memory_transformer.x3_voxel_size_tensor, torch.full((3,), .8))

    def TestX4PoseMetricPositionAndLegacyDefault(self):
        current = Sparse([[0, 1, 1, 1], [1, 1, 1, 1]], batch_size=2)
        history = dict(coords=current.indices.clone(), feats=torch.ones(2, 256))
        pose = torch.eye(4).repeat(2, 1, 1); pose[0, 0, 3] = .8; pose[1, 0, 3] = -.8
        aligned = self.model.AlignMemoryToCurrentX4({'pose_to_prev': pose}, history, current)
        self.assertEqual(set(map(tuple, aligned['coords'].tolist())), {(0, 1, 1, 2), (1, 1, 1, 0)})
        xyz = self.model.x4_memory_transformer.CoordsToXYZ(current.indices, torch.float32)
        torch.testing.assert_close(xyz, torch.full((2, 3), 1.2))
        default = self.model.AlignMemoryToCurrentX3({'pose_to_prev': pose}, history, current)
        explicit = self.model.AlignMemoryToCurrentX3({'pose_to_prev': pose}, history, current, feature_stride=8)
        for key in default: torch.testing.assert_close(default[key], explicit[key], rtol=0, atol=0)
        options = dict(channels=8, point_cloud_range=self.config.point_cloud_range, voxel_size=[.05]*3, depth=1)
        torch.manual_seed(17); legacy = X3MemoryCrossAttentionTransformer(**options)
        torch.manual_seed(17); explicit_transformer = X3MemoryCrossAttentionTransformer(**options, feature_stride=8)
        query = torch.randn(2, 8); memory = dict(coords=current.indices, feats=torch.randn(2, 8))
        torch.testing.assert_close(legacy(query, current.indices, memory, [4]*3),
                                  explicit_transformer(query, current.indices, memory, [4]*3), rtol=0, atol=0)

    def TestCurrentF4UsesUnionMixturesAndFull3DKeys(self):
        current = Sparse([[0, 1, 1, 1], [0, 2, 1, 1]], 1.)
        history = dict(coords=torch.tensor([[0, 1, 1, 1], [0, 3, 1, 1]], dtype=torch.int32), feats=torch.full((2, 256), 3.))
        memory = self.model.MemoryUpdate(current, history, 1, feature_stride=16)
        fused = SparseConvTensor(memory['feats'], memory['coords'], [4]*3, 1)
        self.model.x4_detection_source = 'current_f4'
        detected = self.model.SelectDetectionFeatures(current, current, fused)
        torch.testing.assert_close(detected.indices, current.indices)
        torch.testing.assert_close(detected.features[0], torch.full((256,), 2.))
        torch.testing.assert_close(detected.features[1], torch.ones(256))
        self.assertEqual(len(fused.indices), 3)
        detected.features.sum().backward()
        torch.testing.assert_close(current.features.grad[:, 0], torch.tensor([.5, 1.]))
        missing = Sparse([[0, 0, 0, 0]])
        with self.assertRaises(RuntimeError): self.model.SelectDetectionFeatures(missing, missing, fused)
        self.model.x4_detection_source = 'full_f4'
        self.assertIs(self.model.SelectDetectionFeatures(current, current, fused), fused)

    def TestEnhancedE4RouteAndActualAttentionGradient(self):
        config = EasyDict(dict(self.config, x4_detection_source='enhanced_e4'))
        model = MultiMissionModel(config)
        # C adds no state: strict A/B load remains exact, including both projectors and BN.
        model.load_state_dict(self.model.state_dict(), strict=True)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, self.model.state_dict()[key], rtol=0, atol=0)
        current = Sparse([[0, 1, 1, 1], [0, 2, 1, 1]])
        aligned = dict(coords=torch.tensor([[0, 1, 1, 1], [0, 3, 1, 1]], dtype=torch.int32),
                       feats=torch.randn(2, 256))
        enhanced_features = model.x4_memory_transformer(
            current.features, current.indices, aligned, current.spatial_shape)
        enhanced = SparseConvTensor(enhanced_features, current.indices, current.spatial_shape, 1)
        memory = model.MemoryUpdate(enhanced, aligned, 1, feature_stride=16)
        fused = SparseConvTensor(memory['feats'], memory['coords'], current.spatial_shape, 1)
        detected = model.SelectDetectionFeatures(current, enhanced, fused)
        self.assertIs(detected, enhanced)
        self.assertEqual(len(fused.indices), 3)
        self.assertEqual(len(detected.indices), 2)
        detected.features.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(current.features.grad).all()))
        self.assertGreater(float(current.features.grad.abs().sum()), 0.)
        gradients = [parameter.grad for parameter in model.x4_memory_transformer.parameters()]
        self.assertTrue(all(value is not None and bool(torch.isfinite(value).all()) for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0.)
        self.assertIsNone(aligned['feats'].grad)

    def TestRealOCCAllDeleteDoesNotMutateF4(self):
        x4 = Sparse([[0, 1, 1, 1], [0, 2, 1, 1]])
        x3 = Sparse([[0, 2, 2, 2]], channels=128, shape=(8, 8, 8))
        x2 = Sparse([[0, 4, 4, 4]], channels=64, shape=(16, 16, 16))
        before = (x4.indices.clone(), x4.features.detach().clone())
        head = self.model.occ_head
        saved = head.root_score.bias.detach().clone()
        with torch.no_grad(): head.root_score.bias.fill_(100.)
        result = head(x4, x3, x2)
        with torch.no_grad(): head.root_score.bias.copy_(saved)
        self.assertEqual(int(result['root_keep_mask'].sum()), 0)
        torch.testing.assert_close(x4.indices, before[0]); torch.testing.assert_close(x4.features, before[1])

    def TestFrozenPretrainedBNScopeBuffersAndAffineGradients(self):
        config = EasyDict(dict(self.config, x4_detection_source='enhanced_e4',
                               b_freeze_pretrained_bn_stats=True))
        model = MultiMissionModel(config)
        model.load_state_dict(self.model.state_dict(), strict=True)
        frozen_names = {'conv_input', 'encoder_stage1', 'encoder_stage2', 'encoder_stage3',
                        'encoder_stage4', 'detection_bev_projector', 'terrain_bev_projector'}
        self.assertIs(model.train(), model)
        checked_names = set()
        for name, module in model.named_modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            b_frozen = name.split('.')[0] in frozen_names
            self.assertEqual(module.training, not b_frozen, name)
            checked_names.add(name.split('.')[0])
            before = {key: value.clone() for key, value in module.named_buffers()}
            # Direct real BN calls avoid CPU-unsupported sparse convolutions. The first
            # 15 calls reproduce no-grad history; the final call has affine gradients.
            shape = (4, module.num_features, 3, 3) if isinstance(module, nn.BatchNorm2d) else (16, module.num_features)
            features = torch.randn(shape) + 2
            with torch.no_grad():
                for _ in range(15): module(features)
            output = module(features)
            (output.square().mean() + output.mean()).backward()
            for parameter in (module.weight, module.bias):
                self.assertTrue(parameter.requires_grad, name)
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0., name)
            if b_frozen:
                for key, value in module.named_buffers():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0, msg=name+'.'+key)
            else:
                self.assertEqual(int(module.num_batches_tracked-before['num_batches_tracked']), 16, name)
        self.assertTrue(frozen_names.issubset(checked_names))
        self.assertTrue({'detection_neck', 'detection_head', 'occ_head'}.issubset(checked_names))
        self.assertTrue(all(module.training for module in model.x4_memory_transformer.modules()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.x4_memory_transformer.parameters()))
        # Validation must be all-eval; the next training epoch must reapply just this scope.
        self.assertIs(model.eval(), model)
        self.assertTrue(all(not module.training for module in model.modules()))
        model.train()
        for name, module in model.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                self.assertEqual(module.training, name.split('.')[0] not in frozen_names, name)
        self.assertTrue(model.training)
        self.assertTrue(model.x4_memory_transformer.norm_query_list[0].training)

    def TestDefaultBNPolicyRetainsTrainState(self):
        # The absent flag preserves historical A/B/C behavior; no state keys are added.
        model = MultiMissionModel(self.config)
        model.eval().train()
        self.assertTrue(all(module.training for module in model.modules()))
        bn = model.encoder_stage4[0].norm
        before = bn.num_batches_tracked.clone()
        with torch.no_grad(): bn(torch.randn(16, bn.num_features))
        self.assertEqual(int(bn.num_batches_tracked-before), 1)

    def TestForwardRoutingWithSparseKernelStubs(self):
        # Execute real new ForwardNetwork; only CPU-unsupported sparse kernels/heads are replaced.
        model = MultiMissionModel.__new__(MultiMissionModel); nn.Module.__init__(model)
        model.config = self.config
        x2 = Sparse([[0, 4, 4, 4]], channels=64, shape=(16, 16, 16))
        x3 = Sparse([[0, 2, 2, 2]], channels=128, shape=(8, 8, 8))
        c4 = Sparse([[0, 1, 1, 1]])
        model.EncodeCurrent = lambda batch: (x2, x3)
        observations = {}
        class Stage(nn.Module):
            def forward(self, value):
                assert value is x3; observations['stage4_current_only'] = True; return c4
        class Attention(nn.Module):
            def forward(self, x3_features, **kwargs): return x3_features + 2
        class Projector(nn.Module):
            def __init__(self, name): super().__init__(); self.name = name
            def forward(self, value): observations[self.name] = value; return value.features
        class OCC(nn.Module):
            def forward(self, full, skip3, skip2, b_apply_root_filter=True):
                observations['b_apply_root_filter'] = b_apply_root_filter
                assert skip3 is x3 and skip2 is x2; observations['occ'] = full; return {}
        model.encoder_stage4 = Stage(); model.x4_memory_transformer = Attention()
        model.occ_head = OCC(); model.terrain_bev_projector = Projector('terrain')
        model.detection_bev_projector = Projector('det')
        model.terrain_head = nn.Identity(); model.detection_neck = nn.Identity(); model.detection_head = nn.Identity()
        history = dict(coords=torch.tensor([[0, 1, 1, 1], [0, 3, 1, 1]], dtype=torch.int32), feats=torch.full((2, 256), 5.))
        batch = dict(pose_to_prev=torch.eye(4).unsqueeze(0))
        for mode, count in [('full_f4', 2), ('current_f4', 1), ('enhanced_e4', 1)]:
            model.x4_detection_source = mode
            result = model.ForwardNetwork(batch, history)
            self.assertTrue(observations['b_apply_root_filter'])
            self.assertEqual(len(observations['det'].indices), count)
            self.assertEqual(len(observations['terrain'].indices), 2)
            self.assertIs(observations['terrain'], observations['occ'])
            self.assertEqual(len(result['memory']['coords']), 2)
            rows = MatchRows(c4.indices, result['memory']['coords'], [4]*3)
            torch.testing.assert_close(result['memory']['feats'][rows], torch.full((1, 256), 4.))
            if mode == 'enhanced_e4':
                # E4=1+2=3; detector must not accidentally take F4[S4]=(3+5)/2=4.
                torch.testing.assert_close(observations['det'].features, torch.full((1, 256), 3.))
        model.config.occ.b_apply_root_filter = False
        model.ForwardNetwork(batch, history)
        self.assertFalse(observations['b_apply_root_filter'])
        del model.config.occ.b_apply_root_filter
        output = model.ForwardNetwork(batch, history, b_memory_only=True)
        self.assertEqual(set(output), {'memory'})
        for mode in ('full_f4', 'current_f4', 'enhanced_e4'):
            model.x4_detection_source = mode
            output = model.ForwardNetwork(batch, memory=None)
            torch.testing.assert_close(output['memory']['coords'], c4.indices)
            torch.testing.assert_close(output['memory']['feats'], c4.features)
            torch.testing.assert_close(output['det'], c4.features)
            torch.testing.assert_close(output['terrain'], c4.features)


if __name__ == '__main__':
    loader = unittest.TestLoader(); loader.testMethodPrefix = 'Test'
    unittest.main(testLoader=loader)
