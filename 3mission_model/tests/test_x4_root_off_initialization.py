"""CPU小模型核对S0到-R的精确根参数白名单；不构造稀疏网络或使用GPU。"""
from copy import deepcopy
from pathlib import Path
import sys

from easydict import EasyDict
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multi_mission_model_memory_x4 import MultiMissionModel


class MockX4(nn.Module):
    LoadSingleState = MultiMissionModel.LoadSingleState

    def __init__(self, b_root_enabled):
        super().__init__()
        self.config = EasyDict(occ={'b_enable_root_filter': b_root_enabled})
        self.backbone = nn.Linear(3, 4)
        self.occ_head = nn.Module()
        self.occ_head.root_score = nn.Linear(4, 1) if b_root_enabled else None
        self.detection_bev_projector = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        self.terrain_bev_projector = deepcopy(self.detection_bev_projector)
        self.x4_memory_transformer = nn.Linear(4, 4)


def Main():
    torch.set_num_threads(2)
    source_model = MockX4(True)
    source = {key: value.clone() for key, value in source_model.state_dict().items()
              if not key.startswith(('x4_memory_transformer.', 'terrain_bev_projector.'))}
    root_keys = {'occ_head.root_score.weight', 'occ_head.root_score.bias'}
    for b_root_enabled in (True, False):
        model = MockX4(b_root_enabled)
        attention_before = deepcopy(model.x4_memory_transformer.state_dict())
        report = model.LoadSingleState(source)
        ignored = set() if b_root_enabled else root_keys
        assert set(report['ignored_source_keys']) == ignored
        assert set(report['loaded_keys']) == set(source) - ignored
        assert report['matched_shared_count'] == len(source) - len(ignored)
        assert set(report['random_initialized_keys']) == {'x4_memory_transformer.weight', 'x4_memory_transformer.bias'}
        for key in report['loaded_keys']:
            assert torch.equal(model.state_dict()[key], source[key])
        for key, value in model.x4_memory_transformer.state_dict().items():
            assert torch.equal(value, attention_before[key])
        for key, value in model.detection_bev_projector.state_dict().items():
            assert torch.equal(value, model.terrain_bev_projector.state_dict()[key])
        assert (model.occ_head.root_score is None) == (not b_root_enabled)
        invalid_states = []
        missing = dict(source)
        missing.pop('backbone.weight')
        invalid_states.append(missing)
        invalid_states.append(dict(source, **{'unrelated.weight': torch.ones(1)}))
        invalid_states.append(dict(source, **{'backbone.weight': torch.ones(2)}))
        invalid_states.append(dict(source, **{'occ_head.root_score.extra': torch.ones(1)}))
        if b_root_enabled:
            invalid_states.append({key: value for key, value in source.items() if key not in root_keys})
        for invalid in invalid_states:
            try:
                model.LoadSingleState(invalid)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Unexpected source mismatch was silently accepted')
    # 配置没有明确关闭时，不能仅凭模块不存在就吞掉root来源键。
    model = MockX4(False)
    model.config.occ.b_enable_root_filter = True
    try:
        model.LoadSingleState(source)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Absent root head without explicit config disable was accepted')
    print('PASS CPU root-on strict load; root-off exactly two allowlisted keys; actual loaded/random/copied reports; unrelated missing/unexpected/shape mismatches rejected')


if __name__ == '__main__':
    Main()
