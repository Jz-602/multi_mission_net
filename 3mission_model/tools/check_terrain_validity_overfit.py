"""Small real-frame terrain diagnostic, not formal training or a generalization claim.

Cache fresh single-model BEV features with training-mode backbone BN, then fit only
its complete terrain decoder and both output heads. Train/validation frames remain
separate. Full three-task backward and DDP checks are handled by formal preflight.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from easydict import EasyDict
from LYF_train_ODSS import CfgFromYamlFile, SetRandomSeed
from WaymoDatasetV2 import WaymoDataset
from multi_mission_model import MultiMissionModel
from terrain_head import TerrainLoss
from terrain_metrics import SummarizeTerrainValidity, TERRAIN_UNWEIGHTED_STATISTICS, TERRAIN_VALIDITY_STATISTICS
from terrain_visualization import SaveTerrainDetails
from tools.visualize_three_task import ToDevice
from run_options import CheckRunConfig


def SelectSamples(dataset, indices):
    candidates = []
    for index in indices:
        sample = dataset[index]
        batch = dataset.CollateBatch([sample])
        b_valid = batch['terrain_valid_mask']
        b_ignore = batch.get('terrain_ignore_mask', torch.zeros_like(b_valid))
        count = int((~b_ignore).sum())
        fraction = float((b_valid & ~b_ignore).sum()) / count if count else 0.
        candidates.append((fraction, int(index), batch))
    candidates.sort(key=lambda item: item[0])
    if len(candidates) < 2 or candidates[0][1] == candidates[-1][1]:
        raise ValueError('Provide at least two distinct target indices per split')
    return [candidates[0], candidates[-1]]


def Evaluate(head, criterion, cache, destination=None, pc_range=None):
    head.eval()
    records = []
    per_frame = []
    with torch.no_grad():
        for entry in cache:
            output = head(entry['bev'])
            loss, logs, _ = criterion.ComputeLoss(output, entry['batch'])
            frame = {key: logs[key].item() for key in TERRAIN_UNWEIGHTED_STATISTICS + TERRAIN_VALIDITY_STATISTICS}
            records.append(frame)
            metrics = SummarizeTerrainValidity([frame])
            per_frame.append({'token': entry['token'], 'index': entry['index'], 'split': entry['split'],
                              'loss': float(loss), 'metrics': metrics})
            if destination is not None:
                terrain = {'height': output['terrain_height_pred'],
                           'validity_logits': output['terrain_validity_logits'],
                           'validity_probability': output['terrain_validity_probability']}
                SaveTerrainDetails(terrain, entry['batch'], pc_range,
                                   destination / f"{entry['split']}_{entry['index']}_terrain.png",
                                   {'token': entry['token'], 'split': entry['split'],
                                    'b_diagnostic_only': True})
    return {'aggregate': SummarizeTerrainValidity(records), 'frames': per_frame}


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-indices', default='0,10,25,50,100,200')
    parser.add_argument('--val-indices', default='0,10,25,50,100,200')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError('steps and eval-every must be positive')
    start = time.monotonic()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    SetRandomSeed(args.seed, 0)
    cfg = CfgFromYamlFile(args.config, EasyDict())
    CheckRunConfig(cfg)
    cfg.model.terrain = EasyDict(b_predict_validity=True, validity_weight=1.)
    for key in ('tasks', 'out_size_factor', 'voxel_size'):
        cfg.model[key] = cfg.common_variables[key]
    cfg.model.point_cloud_range = cfg.common_variables.pc_range
    model = MultiMissionModel(cfg.model).to(args.device).train()
    cache = []
    selections = {}
    for split in ('train', 'val'):
        dataset = WaymoDataset(cfg.data[f'{split}_dataset'], cfg.common_variables)
        indices = [int(item) for item in getattr(args, f'{split}_indices').split(',')]
        selected = SelectSamples(dataset, indices)
        selections[split] = [{'index': index, 'gt_valid_fraction': fraction, 'token': batch['token'][0]}
                             for fraction, index, batch in selected]
        for fraction, index, batch in selected:
            device_batch = ToDevice(batch, args.device)
            with torch.no_grad():
                x2, x3 = model.EncodeCurrent(device_batch)
                x4 = model.encoder_stage4(x3)
                bev = model.detection_bev_projector(x4).detach()
                del x2, x3, x4
            cache.append({'bev': bev, 'batch': device_batch, 'token': batch['token'][0],
                          'split': split, 'index': index})
    if set(row['token'] for row in cache if row['split'] == 'train') & set(row['token'] for row in cache if row['split'] == 'val'):
        raise ValueError('Train and validation tokens overlap')
    head = model.terrain_head
    criterion = TerrainLoss(validity_weight=1.)
    del model
    train_cache = [entry for entry in cache if entry['split'] == 'train']
    val_cache = [entry for entry in cache if entry['split'] == 'val']
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    baseline = {'train': Evaluate(head, criterion, train_cache), 'val': Evaluate(head, criterion, val_cache)}
    history = [{'step': 0, **baseline}]
    first_gradients = {}
    for step in range(1, args.steps + 1):
        head.train()
        entry = train_cache[(step - 1) % len(train_cache)]
        optimizer.zero_grad(set_to_none=True)
        output = head(entry['bev'])
        loss, logs, _ = criterion.ComputeLoss(output, entry['batch'])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'nonfinite terrain loss at {step}')
        loss.backward()
        if step == 1:
            first_gradients = {name: float(parameter.grad.norm()) for name, parameter in head.named_parameters()
                               if name in ('height_head.weight', 'validity_head.weight', 'output_refine.block.0.weight')}
            if any(not torch.isfinite(parameter.grad).all() for parameter in head.parameters() if parameter.grad is not None):
                raise FloatingPointError('nonfinite terrain gradient')
            for name in ('height_head.weight', 'validity_head.weight'):
                if first_gradients.get(name, 0.) <= 0.:
                    raise RuntimeError(f'missing gradient: {name}')
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            row = {'step': step, 'loss': float(loss),
                   'train': Evaluate(head, criterion, train_cache), 'val': Evaluate(head, criterion, val_cache)}
            history.append(row)
            print(json.dumps({'step': step, 'loss': float(loss), 'train_miou': row['train']['aggregate']['validity']['miou'],
                              'val_miou': row['val']['aggregate']['validity']['miou']}), flush=True)
            (destination / 'history.json').write_text(json.dumps(history, indent=2)+'\n')
    final = {split: Evaluate(head, criterion, [item for item in cache if item['split'] == split], destination, cfg.common_variables.pc_range)
             for split in ('train', 'val')}
    validity = final['train']['aggregate']['validity']
    b_train_fit = (validity['hole_recall'] is not None and validity['hole_recall'] >= .5 and
                   validity['valid_recall'] is not None and validity['valid_recall'] >= .8 and
                   validity['miou'] > validity['all_valid_baseline']['miou'] + .05)
    report = {'status': 'train_fit_passed_validation_requires_review' if b_train_fit else 'train_fit_failed',
              'b_formal_result': False, 'b_train_fit_passed': b_train_fit,
              'b_distinct_hole_rates': all(abs(rows[0]['gt_valid_fraction']-rows[1]['gt_valid_fraction']) >= .02 for rows in selections.values()),
              'config': args.config, 'config_sha256': hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
              'seed': args.seed, 'steps': args.steps, 'selections': selections, 'first_gradients': first_gradients,
              'initial': baseline, 'final': final, 'elapsed_seconds': time.monotonic()-start,
              'initialization': 'fresh random single backbone and terrain head; no old checkpoint',
              'cache_protocol': 'training-mode backbone BN; real-frame BEV cached once, no further backbone update',
              'acceptance': 'train holeRecall>=.5, validRecall>=.8, binarymIoU>all-valid+.05; validation recorded independently, not proof of generalization'}
    torch.save({'state_dict': head.state_dict(), 'optimizer': optimizer.state_dict(), 'steps': args.steps}, destination/'terrain_head_diagnostic.pth')
    (destination/'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')
    if not b_train_fit or not report['b_distinct_hole_rates']:
        raise RuntimeError('Diagnostic acceptance not met; inspect saved evidence before retrying')


if __name__ == '__main__':
    Main()
