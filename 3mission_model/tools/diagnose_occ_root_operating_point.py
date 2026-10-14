"""只读固定片段：比较 F4 特征与线性删根头；不解码、不训练、不改阈值。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = (('train', 10), ('val', 100), ('val', 2720))
EPOCHS = (20, 25, 48)


def Distribution(values):
    import torch
    values = values.detach().cpu().float().flatten()
    if not len(values):
        return {'count': 0}
    if not bool(torch.isfinite(values).all()):
        raise ValueError('Non-finite diagnostic values')
    quantiles = torch.quantile(values, torch.tensor([0., .01, .05, .5, .95, .99, 1.]))
    return dict(count=len(values), mean=float(values.mean()),
                **dict(zip(('min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max'), quantiles.tolist())))


def Summarize(features, target_delete, weight, bias, threshold, actual_logits=None):
    """当前目标车体系 F4 [N,256]；target_delete [N]：自身根含非空GT为0，否则1。"""
    import torch
    dot = torch.nn.functional.linear(features, weight).flatten()
    logits = dot + bias.flatten()[0]
    b_keep = ~(logits.sigmoid() > threshold)
    b_positive = target_delete == 0
    result = {'root_count': len(features), 'positive_count': int(b_positive.sum()),
              'kept_count': int(b_keep.sum()), 'positive_kept': int((b_keep & b_positive).sum()),
              'empty_kept': int((b_keep & ~b_positive).sum()), 'threshold': threshold,
              'head_weight_l2': float(weight.norm()), 'head_bias': float(bias.item()), 'groups': {}}
    result['positive_recall'] = result['positive_kept'] / result['positive_count'] if result['positive_count'] else None
    for name, mask in [('all', torch.ones(len(features), dtype=torch.bool)),
                       ('positive_gt', b_positive), ('empty_gt', ~b_positive)]:
        result['groups'][name] = {'feature_l2': Distribution(features[mask].norm(dim=1)),
                                 'dot_without_bias': Distribution(dot[mask]),
                                 'delete_logits': Distribution(logits[mask]),
                                 'delete_probability': Distribution(logits[mask].sigmoid())}
    if actual_logits is not None:
        b_actual_keep = ~(actual_logits.sigmoid() > threshold)
        result['native_gpu_linear'] = {
            'cpu_gpu_max_logit_difference': float((actual_logits - logits).abs().max()),
            'cpu_gpu_mask_disagreements': int((b_keep != b_actual_keep).sum()),
            'kept_count': int(b_actual_keep.sum()),
            'positive_kept': int((b_actual_keep & b_positive).sum()),
            'empty_kept': int((b_actual_keep & ~b_positive).sum()),
            'positive_delete_logits': Distribution(actual_logits[b_positive]),
            'empty_delete_logits': Distribution(actual_logits[~b_positive])}
    return result


def RootTargets(head, coords, features, shape, batch):
    """复用生产Targets：根为目标车体系0.8m BZYX，GT为同ROI的0.2m格子。"""
    pred = dict(root_indices=coords, root_spatial_shape=shape, batch_size=1,
                indices=coords.new_empty((0, 4)), logits=features.new_empty((0, 4)),
                spatial_shape=tuple(int(value) * 4 for value in shape))
    return head.Targets(pred, batch)[1]


def CPUCheck():
    import torch
    from occ_head import GeneratedOCCHead
    features = torch.tensor([[-2., 0.], [0., 0.], [2., 0.], [4., 0.]])
    target = torch.tensor([0., 0., 1., 1.])
    weight, bias = torch.tensor([[1., 0.]]), torch.tensor([0.])
    logits = torch.nn.functional.linear(features, weight, bias).flatten()
    result = Summarize(features, target, weight, bias, .5, logits)
    assert result['kept_count'] == 2 and result['positive_kept'] == 2
    assert result['native_gpu_linear']['cpu_gpu_mask_disagreements'] == 0
    assert Summarize(features, target, -weight, bias, .5)['positive_kept'] == 1
    head = GeneratedOCCHead.__new__(GeneratedOCCHead)
    torch.nn.Module.__init__(head)
    head.pc_range = (0., 0., 0., 1.6, .8, .8)
    head.voxel_size_xyz = (.2, .2, .2)
    head.grid_shape_xyz = (8, 4, 4)
    coords = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32)
    batch = dict(occ_voxel_indices=[torch.tensor([[0, 0, 0], [4, 0, 0]])],
                 occ_labels=[torch.tensor([1, 255])],
                 occ_pc_range=torch.tensor([head.pc_range]),
                 occ_voxel_size_xyz=torch.tensor([head.voxel_size_xyz]),
                 occ_grid_shape=torch.tensor([head.grid_shape_xyz]))
    targets = RootTargets(head, coords, torch.zeros(2, 256), (1, 1, 2), batch)
    assert targets.tolist() == [0., 1.]
    return {'b_pass': True, 'b_gpu_executed': False,
            'checks': ['zero-logit threshold retains root', 'positive/empty count and opposite head',
                       'native/CPU logit parity fixture', 'production Targets own-root GT and ignore255'],
            'fixture': result}


def Run(args):
    import torch
    from easydict import EasyDict
    from LYF_train_ODSS import CfgFromYamlFile, SetRandomSeed
    from WaymoDataset_multi import WaymoDatasetMulti
    from multi_mission_model_memory_x4 import MultiMissionModel
    from trainer.trainer.three_task_runtime import MoveToDevice
    from occ_head import UniqueRows
    from run_options import CheckRunConfig, HashFile, WriteJsonAtomic

    run = Path(args.run_directory).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = CfgFromYamlFile(str(run / 'config.yaml'), EasyDict())
    CheckRunConfig(cfg)
    for key in ('tasks', 'out_size_factor', 'voxel_size'):
        cfg.model[key] = cfg.common_variables[key]
    cfg.model.point_cloud_range = cfg.common_variables.pc_range
    report = {'b_complete': False, 'b_read_only': True, 'config_sha256': HashFile(run / 'config.yaml'),
              'script_sha256': HashFile(__file__), 'checkpoints': {}, 'cases': [],
              'scope': '9 fixed 15+1 clips; actual memory-only F4 path, native root linear; no task decoding',
              'limits': ['Three fixed diagnostic examples, not full-validation results.',
                         'Mixed feature/head pairs are attribution probes, not trained model metrics.',
                         'No prediction quality or same-weight root on/off decoding here; run that separately.',
                         'CPU cross-head arithmetic may differ slightly from native GPU linear; native pair records both.']}
    checkpoints, heads = {}, {}
    for epoch in EPOCHS:
        path = run / f'checkpoints/epoch_{epoch}.pth'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['meta']['epoch'] == epoch
        assert checkpoint['meta']['model_config'] == cfg.model, 'Checkpoint and run model config differ'
        state = checkpoint['state_dict']
        checkpoints[epoch] = state
        heads[epoch] = (state['occ_head.root_score.weight'], state['occ_head.root_score.bias'])
        report['checkpoints'][str(epoch)] = {'path': str(path), 'sha256': HashFile(path)}
        del checkpoint
    model = MultiMissionModel(cfg.model).cuda().eval()
    started = time.monotonic()
    try:
        for split, index in CASES:
            # 与出图工具相同：每例重设种子，训练图历史间隔也固定为同一次抽样。
            SetRandomSeed(0, 0)
            dataset = WaymoDatasetMulti(cfg.data[f'{split}_dataset'], cfg.common_variables)
            sample = dataset[index]
            batch = dataset.CollateBatch([sample])
            del dataset
            case = {'split': split, 'index': index, 'token': sample['token_seq'][-1],
                    'temporal_clip': {'token_seq': sample['token_seq'],
                                      'full_frame_indices': sample['full_frame_indices'].tolist(),
                                      'history_interval': int(sample['history_interval'])}, 'epochs': {}}
            reference_coords, reference_targets = None, None
            for epoch in EPOCHS:
                reference = json.loads((run / f'images/epoch_{epoch:02d}/{split}_{index:03d}.json').read_text())
                assert reference['token'] == case['token']
                assert reference['temporal_clip']['token_seq'] == sample['token_seq'], 'Historical figure clip differs'
                model.load_state_dict(checkpoints[epoch], strict=True)
                memory = None
                with torch.no_grad():
                    for frame in batch['frames']:
                        target = MoveToDevice(frame, 'cuda')
                        memory = model(target, memory, b_memory_only=True)
                    rows = UniqueRows(memory['coords'], memory['spatial_shape'])
                    coords = memory['coords'][rows]
                    features = memory['feats'][rows]
                    targets = RootTargets(model.occ_head, coords, features, memory['spatial_shape'], target)
                    actual_logits = model.occ_head.root_score(features).flatten().cpu()
                coords, features, targets = coords.cpu(), features.cpu(), targets.cpu()
                if reference_coords is not None:
                    assert torch.equal(coords, reference_coords) and torch.equal(targets, reference_targets)
                reference_coords, reference_targets = coords, targets
                threshold = model.occ_head.delete_threshold
                pairs = {}
                for head_epoch in ((20, 25) if epoch in (20, 25) else (48,)):
                    weight, bias = heads[head_epoch]
                    pairs[str(head_epoch)] = Summarize(features, targets, weight, bias, threshold,
                                                       actual_logits if head_epoch == epoch else None)
                native = pairs[str(epoch)]['native_gpu_linear']
                expected = reference['occ']
                assert len(coords) == expected['root_count']
                assert int((targets == 0).sum()) == expected['occupied_root_count']
                # 生产图使用GPU筛选；近阈值数值差异须留痕，不以CPU交叉计算冒充原始mask。
                case['epochs'][str(epoch)] = {'head_pairings': pairs,
                    'saved_figure_counts': {key: expected[key] for key in ('root_count', 'kept_root_count', 'occupied_root_count', 'occupied_root_kept')},
                    'native_vs_saved_figure_kept_difference': native['kept_count'] - expected['kept_root_count'],
                    'native_vs_saved_figure_positive_kept_difference': native['positive_kept'] - expected['occupied_root_kept']}
                cache = output / f'{split}_{index:03d}_epoch_{epoch:02d}.pt'
                torch.save({'coords_bzyx': coords, 'features_f4': features, 'target_delete': targets,
                            'native_root_logits': actual_logits, 'root_shape_zyx': memory['spatial_shape'],
                            'head_weight': heads[epoch][0], 'head_bias': heads[epoch][1],
                            'delete_threshold': threshold, 'token': case['token'],
                            'checkpoint_sha256': report['checkpoints'][str(epoch)]['sha256']}, cache)
                case['epochs'][str(epoch)]['cache'] = {'path': str(cache), 'sha256': HashFile(cache)}
                del memory, target, features, actual_logits
                torch.cuda.empty_cache()
            report['cases'].append(case)
            WriteJsonAtomic(output / f'{split}_{index:03d}.json', case)
            WriteJsonAtomic(output / 'root_operating_point.json', report)
            del batch
        report['b_saved_prediction_root_counts_match'] = all(
            item['native_vs_saved_figure_kept_difference'] == 0
            and item['native_vs_saved_figure_positive_kept_difference'] == 0
            for case in report['cases'] for item in case['epochs'].values())
        report['b_complete'] = True
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        WriteJsonAtomic(output / 'root_operating_point.json', report)


def Main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--run-directory', default=str(ROOT / 'log/server_runs_x4_selected_long48_20261010/f'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.run:
        Run(args)
    else:
        Path(args.output).write_text(json.dumps(CPUCheck(), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    Main()
