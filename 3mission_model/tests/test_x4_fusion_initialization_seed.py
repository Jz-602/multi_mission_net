"""CPU真实模块构造：-R和有根X4同S0、新融合同种子；默认旧初始化逐值不变。"""
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys

from easydict import EasyDict
import torch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from test_x4_memory import Config
from multi_mission_model import MultiMissionModel as SingleModel
from multi_mission_model_memory import X3MemoryCrossAttentionTransformer
from multi_mission_model_memory_x4 import MultiMissionModel, X4_STRIDE


def StateHashes(module):
    return {key: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()
            for key, value in module.state_dict().items()}


def Main():
    torch.set_num_threads(2)
    config = Config()
    config.terrain = EasyDict(b_predict_validity=True)
    config.occ = EasyDict(b_enable_root_filter=True)
    source = SingleModel(deepcopy(config)).state_dict()
    hashes = []
    for b_root_enabled, outer_seed in ((True, 0), (False, 0), (False, 37)):
        cfg = deepcopy(config)
        cfg.occ.b_enable_root_filter = b_root_enabled
        cfg.x4_fusion_init_seed = 0
        torch.manual_seed(outer_seed)
        model = MultiMissionModel(cfg)
        report = model.LoadSingleState(source)
        assert report['x4_fusion_init_seed'] == 0
        for key in report['loaded_keys']:
            assert torch.equal(model.state_dict()[key], source[key]), key
        for key, value in model.detection_bev_projector.state_dict().items():
            assert torch.equal(value, model.terrain_bev_projector.state_dict()[key]), key
        hashes.append(StateHashes(model))
        del model
        gc.collect()
    common = sorted(set(hashes[0]) & set(hashes[1]))
    mismatched = [key for key in common if any(item[key] != hashes[0][key] for item in hashes[1:])]
    assert not mismatched, mismatched
    # 隔离上下文恢复到父单帧构造后的RNG；两个投影的deepcopy不额外消耗随机数。
    torch.manual_seed(19)
    single = SingleModel(deepcopy(config))
    after_single = torch.get_rng_state().clone()
    legacy_attention = X3MemoryCrossAttentionTransformer(
        channels=config.encoder_channels[3], point_cloud_range=config.point_cloud_range,
        voxel_size=config.voxel_size, depth=int(config.memory_transformer_depth),
        num_heads=int(config.memory_num_heads), topk=int(config.memory_topk),
        coord_radius=config.memory_coord_radius, feature_stride=X4_STRIDE)
    legacy_hashes = StateHashes(legacy_attention)
    del single, legacy_attention
    gc.collect()
    torch.manual_seed(19)
    seeded_cfg = deepcopy(config)
    seeded_cfg.x4_fusion_init_seed = 0
    seeded = MultiMissionModel(seeded_cfg)
    assert torch.equal(after_single, torch.get_rng_state())
    del seeded
    gc.collect()
    torch.manual_seed(19)
    default = MultiMissionModel(deepcopy(config))
    assert StateHashes(default.x4_memory_transformer) == legacy_hashes
    result = dict(status='pass', shared_state_count=len(common), shared_state_mismatch_count=len(mismatched),
                  fusion_state_count=sum(key.startswith('x4_memory_transformer.') for key in common),
                  b_default_legacy_exact=True, b_cpu_rng_restored=True,
                  root_on_outer_seed=0, root_off_outer_seeds=[0, 37], fixed_fusion_seed=0,
                  b_source_loaded_exact=True, b_projector_copied_exact=True,
                  scope='Actual CPU model constructors/state load; no sparse forward or GPU')
    print(json.dumps(result))
    return result


if __name__ == '__main__':
    Main()
