"""X4 rolling memory; detection support is either full F4 or F4 sampled at current S4."""
from copy import deepcopy
from contextlib import nullcontext

from spconv.pytorch import SparseConvTensor
import torch
from torch import nn

from multi_mission_model import MultiMissionModel as SingleFrameModel, ReplaceFeature
from multi_mission_model_memory import MultiMissionModel as X3MemoryModel, X3MemoryCrossAttentionTransformer
from occ_head import MatchRows


X4_STRIDE = 16


class MultiMissionModel(X3MemoryModel):
    """独立编码当前 C4，再融合历史；地形/OCC 始终读取完整 F4。

    C4/E4/F4 均为当前帧 LiDAR 坐标系、[N,256] 特征及 [N,4] BZYX 索引。
    S4 是融合前 C4 的完整三维支持；M4 是上一帧 LiDAR 坐标系的 rolling memory。
    输入 .05m 时 X4 栅格为 .8m，OCC 跳连仍是当前帧 .4m x3/.2m x2。
    历史 no_grad/train-BN 策略沿用 trainer；本模型不隐式切换 BN 状态。
    """

    def __init__(self, config):
        SingleFrameModel.__init__(self, config)
        self.x4_detection_source = str(self.config.get('x4_detection_source', 'full_f4'))
        if self.x4_detection_source not in ('full_f4', 'current_f4', 'enhanced_e4'):
            raise ValueError(f'unsupported x4_detection_source={self.x4_detection_source}')
        fusion_seed = self.config.get('x4_fusion_init_seed')
        # 本轮-R少构造一个Linear，不应改变新增融合权重；默认旧实验保持原RNG路径。
        # 网络此时在CPU构造，只隔离CPU RNG，不初始化或改动CUDA随机状态。
        with torch.random.fork_rng(devices=[]) if fusion_seed is not None else nullcontext():
            if fusion_seed is not None:
                torch.random.default_generator.manual_seed(int(fusion_seed))
            self.x4_memory_transformer = X3MemoryCrossAttentionTransformer(
                channels=self.config.encoder_channels[3],
                point_cloud_range=self.config.point_cloud_range,
                voxel_size=self.config.voxel_size,
                depth=int(self.config.memory_transformer_depth),
                num_heads=int(self.config.memory_num_heads),
                topk=int(self.config.memory_topk),
                coord_radius=self.config.memory_coord_radius,
                feature_stride=X4_STRIDE,
            )
        # 两组都独立训练这两个 projector，防止 full/current 支持混用同一 BN 统计。
        self.terrain_bev_projector = deepcopy(self.detection_bev_projector)

    def train(self, mode=True):
        super().train(mode)
        if mode and bool(self.config.get('b_freeze_pretrained_bn_stats', False)):
            # F只固定预训练主干及两条BEV投影的统计；affine、任务头及新attention仍可训练。
            for name in ('conv_input', 'encoder_stage1', 'encoder_stage2', 'encoder_stage3',
                         'encoder_stage4', 'detection_bev_projector', 'terrain_bev_projector'):
                for module in getattr(self, name).modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
        return self

    def LoadSingleState(self, source_state):
        """严格加载单帧 state_dict，再复制已加载的投影参数/BN；恢复 X4 用普通 strict load。"""
        target = self.state_dict()
        new_prefixes = ('x4_memory_transformer.', 'terrain_bev_projector.')
        expected = {key for key in target if not key.startswith(new_prefixes)}
        # -R确实不构造根分类器；仅允许忽略单帧来源中的这两个根参数。
        b_root_disabled = (not bool(self.config.occ.get('b_enable_root_filter', True))
                           and self.occ_head.root_score is None)
        ignored = ({'occ_head.root_score.weight', 'occ_head.root_score.bias'} & set(source_state)
                   if b_root_disabled else set())
        shared_state = {key: value for key, value in source_state.items() if key not in ignored}
        missing = sorted(expected - set(shared_state))
        unexpected = sorted(set(shared_state) - expected)
        mismatched = [key for key in expected & set(shared_state) if target[key].shape != shared_state[key].shape]
        if missing or unexpected or mismatched:
            raise RuntimeError(f'incompatible single state: missing={missing[:10]}, unexpected={unexpected[:10]}, shape={mismatched[:10]}')
        target.update(shared_state)
        self.load_state_dict(target, strict=True)
        # 构造时复制随机权重不等于复制单帧预训练权重，必须在上面的加载之后再次复制。
        self.terrain_bev_projector.load_state_dict(self.detection_bev_projector.state_dict(), strict=True)
        return {'matched_shared_count': len(shared_state),
                'x4_fusion_init_seed': self.config.get('x4_fusion_init_seed'),
                'loaded_keys': sorted(shared_state), 'ignored_source_keys': sorted(ignored),
                'random_initialized_keys': sorted(key for key in target if key.startswith('x4_memory_transformer.')),
                'copied_projector_keys': sorted(key for key in target if key.startswith('terrain_bev_projector.')),
                'new_attention_count': sum(key.startswith('x4_memory_transformer.') for key in target),
                'copied_projector_count': len(self.detection_bev_projector.state_dict())}

    def AlignMemoryToCurrentX4(self, batch_dict, memory, current_x4):
        # 复用相同位姿/碰撞合并规则，但所有中心、量化、距离均按 stride16 的 X4 网格。
        return self.AlignMemoryToCurrentX3(batch_dict, memory, current_x4, feature_stride=X4_STRIDE)

    def SelectDetectionFeatures(self, current_x4, enhanced_x4, fused_x4):
        """检测路由唯一接口；current_f4 取 F4 在 S4 的值，绝非 E4/C4 或二维 BEV 掩码。"""
        if self.x4_detection_source == 'full_f4':
            return fused_x4
        if self.x4_detection_source == 'enhanced_e4':
            # C仅改变检测读取：保留历史注意力，但不再读取memory合并后的值。
            return enhanced_x4
        rows = MatchRows(current_x4.indices, fused_x4.indices, fused_x4.spatial_shape)
        if not bool((rows >= 0).all()):
            raise RuntimeError('F4 union is missing current S4 coordinates')
        # 新建 sparse tensor，不能沿用完整并集的 indice 缓存。
        return SparseConvTensor(fused_x4.features[rows], current_x4.indices,
                                fused_x4.spatial_shape, fused_x4.batch_size)

    def ForwardNetwork(self, batch_dict, memory=None, b_memory_only=False):
        x2, x3 = self.EncodeCurrent(batch_dict)
        current_x4 = self.encoder_stage4(x3)
        aligned_memory = self.AlignMemoryToCurrentX4(batch_dict, memory, current_x4)
        if aligned_memory is not None and aligned_memory['coords'].shape[0] > 0:
            enhanced_features = self.x4_memory_transformer(
                x3_features=current_x4.features, x3_coords=current_x4.indices,
                aligned_memory=aligned_memory, spatial_shape=current_x4.spatial_shape,
            )
        else:
            zero = sum(parameter.reshape(-1)[0]*0 for parameter in self.x4_memory_transformer.parameters())
            enhanced_features = current_x4.features + zero
        enhanced_x4 = ReplaceFeature(current_x4, enhanced_features)
        memory_next = self.MemoryUpdate(enhanced_x4, aligned_memory, current_x4.batch_size, feature_stride=X4_STRIDE)
        if b_memory_only:
            return {'memory': memory_next}
        fused_x4 = SparseConvTensor(memory_next['feats'], memory_next['coords'].int(),
                                    memory_next['spatial_shape'], memory_next['batch_size'])
        detection_x4 = self.SelectDetectionFeatures(current_x4, enhanced_x4, fused_x4)
        # OCC 筛根只在自身内部执行，不把裁剪结果写回 F4 或 rolling memory。
        occ = self.occ_head(fused_x4, x3, x2,
                            b_apply_root_filter=bool(self.config.occ.get('b_apply_root_filter', True)))
        terrain = self.terrain_head(self.terrain_bev_projector(fused_x4))
        det = self.detection_head(self.detection_neck(self.detection_bev_projector(detection_x4)))
        return {'det': det, 'terrain': terrain, 'occ': occ, 'memory': memory_next,
                'counts': {'x2': len(x2.indices), 'x3': len(x3.indices), 'x4': len(fused_x4.indices),
                           'x4_current': len(current_x4.indices), 'x4_detection': len(detection_x4.indices)}}
