"""CPU：删根输入归一化兼容性、分组监督与恢复；不执行稀疏卷积GPU。"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_occ_head import MakeHead, SparseInput, Batch


def Main():
    torch.set_num_threads(2)
    raw = MakeHead()
    normalized = MakeHead()
    normalized.root_input_normalization = 'layer_norm'
    normalized.load_state_dict(raw.state_dict(), strict=True)
    assert raw.state_dict().keys() == normalized.state_dict().keys()
    features = torch.randn(6, 256, requires_grad=True) * 10000
    assert raw.RootInputFeatures(features) is features
    copied = features.detach().clone()
    inputs = normalized.RootInputFeatures(features)
    assert torch.equal(features.detach(), copied)
    torch.testing.assert_close(inputs.mean(1), torch.zeros(6), atol=1e-6, rtol=0)
    torch.testing.assert_close(inputs.var(1, unbiased=False), torch.ones(6), atol=1e-6, rtol=0)
    # 原Linear零权重初始化保证任意特征尺度下概率0.01，全根保留。
    for head in (raw, normalized):
        probability = head.root_score(head.RootInputFeatures(features)).sigmoid()
        torch.testing.assert_close(probability, torch.full_like(probability, .01))
        assert bool((probability <= head.delete_threshold).all())
    with torch.no_grad():
        normalized.root_score.weight.fill_(.01)
        normalized.root_score.weight[0, 0] = .1
        normalized.root_score.bias.zero_()
    leaves = torch.randn(6, 256, requires_grad=True)
    logits = normalized.root_score(normalized.RootInputFeatures(leaves)).flatten()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.] * 3)).backward()
    assert torch.isfinite(leaves.grad).all() and leaves.grad.abs().sum() > 0
    assert normalized.root_score.weight.grad.abs().sum() > 0
    # 恢复模型只读参数/缓冲；归一化为配置，不新增任何state_dict键。
    restored = MakeHead()
    restored.root_input_normalization = 'layer_norm'
    restored.load_state_dict(normalized.state_dict(), strict=True)
    assert torch.equal(restored.root_score(restored.RootInputFeatures(leaves)),
                       normalized.root_score(normalized.RootInputFeatures(leaves)))
    normalized.b_collect_root_diagnostics = True
    with torch.no_grad():
        normalized.root_score.weight.zero_()
        normalized.root_score.bias.fill_(10.)
    output = normalized(*SparseInput())
    loss, logs, b_valid = normalized.ComputeLoss(output, Batch([[7, 7, 7]], [3]))
    assert b_valid and logs['root_supervision_count'] == 2 and logs['valid_leaf_count'] == 0
    assert logs['root_positive_count'] == logs['root_empty_count'] == 1
    assert logs['root_positive_bce_mean'] > 9.9 and logs['root_empty_bce_mean'] < .001
    assert abs(float(logs['root_loss']) - (logs['root_positive_bce_mean'] + logs['root_empty_bce_mean']) / 2) < 1e-6
    assert 'root_positive_feature_l2_p95' in logs and 'root_empty_probability_p50' in logs
    loss.backward()
    assert normalized.root_score.bias.grad.abs().sum() > 0
    empty = normalized.RootGroupStatistics(torch.empty(0), torch.empty(0))
    assert empty == {'root_positive_count': 0, 'root_empty_count': 0}
    print('PASS default bitwise path, affine-free norm, all-keep init, strict load/restore, gradient to features/head, all-root BCE after total deletion, group metrics and absent groups')


if __name__ == '__main__':
    Main()
