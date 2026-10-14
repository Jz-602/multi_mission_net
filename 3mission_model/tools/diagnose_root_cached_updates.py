"""CPU缓存机制探针：固定epoch20 F4，仅更新root Linear；不是整网续训或精度实验。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def Main():
    import torch
    from occ_head import GeneratedOCCHead
    from run_options import HashFile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-directory', default=str(ROOT / 'log/server_preparation/x4_selected_long48_20261010/root_operating_point_gpu'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'log/server_runs_x4_selected_long48_20261010/f/checkpoints/epoch_20.pth'))
    parser.add_argument('--initialization', choices=['checkpoint', 'all_keep'], default='checkpoint')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.steps <= 200:
        parser.error('This finite CPU probe permits 1..200 steps')
    torch.set_num_threads(2)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert checkpoint['meta']['epoch'] == 20
    checkpoint_hash = HashFile(args.checkpoint)
    settings = checkpoint['optimizer']['param_groups'][0]
    caches, sources = {}, {}
    for name in ['train_010', 'val_100', 'val_2720']:
        path = Path(args.cache_directory) / f'{name}_epoch_20.pt'
        cache = torch.load(path, map_location='cpu', weights_only=False)
        assert cache['checkpoint_sha256'] == checkpoint_hash
        caches[name] = cache
        sources[name] = {'path': str(path), 'sha256': HashFile(path)}
    report = {'b_complete': False, 'b_gpu_executed': False, 'source_checkpoint_sha256': checkpoint_hash,
              'source_cache': sources, 'initialization': args.initialization, 'steps': args.steps,
              'optimizer': {'kind': 'fresh AdamW, no inherited moment estimates', 'lr': settings['lr'],
                            'betas': settings['betas'], 'weight_decay': settings['weight_decay'],
                            'b_scheduler_used': False, 'b_grad_clipping_used': False},
              'scope': 'frozen epoch20 train10 F4 root-BCE updates only; val100/2720 never used for updates',
              'limits': ['This does not restore full-network optimizer or train other parameters.',
                         'Raw root-only gradients have no shared-task weighting or full-model clipping.',
                         'Repeated one-clip root fitting is a finite mechanism probe, not a candidate quality decision.',
                         'Checkpoint head plus normalization changes initial calibration; all_keep resets BOTH arms if selected.'],
              'candidates': {}}
    for mode in ['none', 'layer_norm']:
        head = GeneratedOCCHead.__new__(GeneratedOCCHead)
        torch.nn.Module.__init__(head)
        head.root_input_normalization = mode
        head.root_score = torch.nn.Linear(256, 1)
        head.root_score.load_state_dict({'weight': checkpoint['state_dict']['occ_head.root_score.weight'],
                                        'bias': checkpoint['state_dict']['occ_head.root_score.bias']})
        if args.initialization == 'all_keep':
            with torch.no_grad():
                head.root_score.weight.zero_()
                head.root_score.bias.fill_(torch.log(torch.tensor(.01 / .99)).item())
        optimizer = torch.optim.AdamW(head.root_score.parameters(), lr=settings['lr'], betas=settings['betas'],
                                      weight_decay=settings['weight_decay'], eps=settings['eps'])
        records, snapshots = [], {}
        for step in range(args.steps + 1):
            if step in {0, 1, 5, 20, args.steps}:
                snapshots[str(step)] = {}
                with torch.no_grad():
                    for name, cache in caches.items():
                        x, y = cache['features_f4'], cache['target_delete']
                        transformed = head.RootInputFeatures(x)
                        logits = head.root_score(transformed).flatten()
                        b_keep, b_positive = logits.sigmoid() <= cache['delete_threshold'], y == 0
                        snapshots[str(step)][name] = {
                            'bce': float(torch.nn.functional.binary_cross_entropy_with_logits(logits, y)),
                            'positive_recall': float((b_keep & b_positive).sum()) / int(b_positive.sum()),
                            'kept_fraction': float(b_keep.float().mean()),
                            'groups': head.RootGroupStatistics(logits, y, {'feature_l2': x.norm(dim=1), 'input_l2': transformed.norm(dim=1)})}
            if step == args.steps:
                break
            cache = caches['train_010']
            x = cache['features_f4'].detach().requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            logits = head.root_score(head.RootInputFeatures(x)).flatten()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, cache['target_delete'])
            loss.backward()
            weight = head.root_score.weight.detach().clone()
            bias = head.root_score.bias.detach().clone()
            row = {'step': step + 1, 'root_bce': float(loss), 'raw_weight_gradient_l2': float(head.root_score.weight.grad.norm()),
                   'raw_bias_gradient': float(head.root_score.bias.grad), 'feature_gradient_l2': float(x.grad.norm()),
                   'weight_l2_before': float(weight.norm()), 'bias_before': float(bias)}
            optimizer.step()
            delta = float((head.root_score.weight.detach() - weight).norm())
            row.update(weight_update_l2=delta, relative_weight_update=delta / float(weight.norm()) if bool(weight.norm() > 0) else None,
                       bias_update=float(head.root_score.bias.detach() - bias), weight_l2_after=float(head.root_score.weight.detach().norm()))
            records.append(row)
        report['candidates'][mode] = {'snapshots': snapshots, 'updates': records}
        (output / f'{mode}_updates.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in records))
    report['b_complete'] = True
    (output / 'cached_root_probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(output / 'cached_root_probe.json')


if __name__ == '__main__':
    Main()
