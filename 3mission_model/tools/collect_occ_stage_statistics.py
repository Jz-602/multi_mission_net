"""Independent eight-GPU full-validation OCC accounting; not a speed benchmark.

Uses the production dataset, checkpoint and raw ForwardNetwork. No training,
label generation, NMS or production validation path is changed.
"""
import argparse
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from occ_stage_statistics import Array, MergeStatistics, StatisticsCollector


def FirstSample(samples):
    return samples[0]


def AlignedRealPoints(sample):
    """Warp real history XYZ to final target using dataset prev->current poses.

    Frames contain [N,3/4] local LiDAR input; only XYZ transforms, never dt.
    A chain T_target<-i uses the exact pose_to_prev matrices used by memory,
    composed in float64. Repeated padded observations remain real observations;
    downstream voxel sets deduplicate them and clip only in the target ROI.
    """
    frames = sample['frames']
    current = Array(frames[-1]['points'])[:, :3]
    transform = np.eye(4, dtype=np.float64)
    histories = []
    for index in range(len(frames)-2, -1, -1):
        transform = transform @ Array(frames[index+1]['pose_to_prev']).astype(np.float64)
        points = Array(frames[index]['points'])[:, :3].astype(np.float64)
        histories.append(points @ transform[:3, :3].T + transform[:3, 3])
    history = np.concatenate(histories) if histories else np.empty((0, 3), dtype=np.float64)
    return current, history


def WriteTables(output, summary):
    """Global additive counts/ratios and mutually exclusive per-100-GT table."""
    with (output/'global_counts.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['count', 'value'])
        for name, value in summary['counts'].items(): writer.writerow([name, json.dumps(value)])
    with (output/'global_ratios.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['metric', 'numerator', 'denominator', 'value'])
        for name, value in summary['ratios'].items():
            if isinstance(value, dict): writer.writerow([name, value['numerator'], value['denominator'], value['value']])
    counts = summary['counts']; denominator = counts['gt']
    with (output/'gt_fates_per_100.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['exclusive_fate', 'count', 'gt_denominator', 'per_100_gt'])
        for name, value in zip(('outside_Lplus', 'root_deletion_lost', 'generated_pred_empty', 'pred_nonempty'), counts['gt_fates']):
            writer.writerow([name, value, denominator, 100*value/denominator if denominator else None])
    summary['gt_fates_per_100'] = [100*v/denominator if denominator else None for v in counts['gt_fates']]
    summary['fate4_semantic_split_per_100'] = {key:100*counts[key]/denominator if denominator else None
        for key in ('candidate_gt_nonempty_correct', 'candidate_gt_nonempty_wrong_nonempty')}


def Run(args):
    import torch
    import torch.distributed as dist
    from easydict import EasyDict
    from torch.utils.data import DataLoader
    from build_label_manifest import LoadConfig
    from LYF_train_ODSS_multi import SetRandomSeed
    from WaymoDataset_multi import WaymoDatasetMulti
    from multi_mission_model_memory_x4 import MultiMissionModel
    from trainer.trainer.three_task_runtime import MoveToDevice
    from run_options import CheckRunConfig, HashFile, WriteJsonAtomic

    rank, world_size, local_rank = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world_size != 8:
        raise ValueError('Full statistics collection requires the authorized eight GPUs')
    torch.set_num_threads(args.cpu_threads); torch.cuda.set_device(local_rank)
    dist.init_process_group('gloo', timeout=timedelta(hours=3))
    output = Path(args.output).resolve()
    if rank == 0: output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    cfg = EasyDict(LoadConfig(args.config)); CheckRunConfig(cfg)
    SetRandomSeed(int(cfg.get('random_seed', 0)), rank)
    for key in ('tasks', 'out_size_factor', 'voxel_size'): cfg.model[key] = cfg.common_variables[key]
    cfg.model.point_cloud_range = cfg.common_variables.pc_range
    if args.root_filter != 'configured':
        cfg.model.occ.b_apply_root_filter = args.root_filter == 'on'
    if not cfg.model.occ.get('b_enable_root_filter', True):
        raise ValueError('Selected on/off protocol must retain root head and BCE parameters')
    if cfg.data.val_dataset.get('b_random_history_interval', True):
        raise ValueError('Fixed validation requires deterministic history interval')
    dataset = WaymoDatasetMulti(cfg.data.val_dataset, cfg.common_variables)
    expected = json.loads(Path(args.expected_tokens).read_text())
    tokens = [item['anchor_info']['token'] for item in dataset.anchor_list]
    if len(tokens) != 5973 or len(set(tokens)) != 5973 or tokens != expected:
        raise ValueError('Dataset must match all5973 fixed validation IDs in exact saved order')
    model = MultiMissionModel(cfg.model).to(local_rank).eval()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    if int(checkpoint['meta']['epoch']) != args.expected_epoch:
        raise ValueError('Checkpoint epoch differs from the registered comparison')
    if rank == 0:
        WriteJsonAtomic(output/'manifest.json', dict(config=str(Path(args.config).resolve()), config_sha256=HashFile(args.config),
            resolved_config=cfg, checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=HashFile(args.checkpoint),
            checkpoint_meta=checkpoint['meta'], expected_tokens_sha256=HashFile(args.expected_tokens),
            source_sha256={str(path.relative_to(PROJECT)):HashFile(path) for path in
                (Path(__file__), PROJECT/'occ_stage_statistics.py', PROJECT/'occ_head.py', PROJECT/'multi_mission_model_memory_x4.py', PROJECT/'WaymoDataset_multi.py')},
            b_apply_root_filter=bool(cfg.model.occ.get('b_apply_root_filter', True)),
            b_statistics_enabled=True, b_timing_is_inference_benchmark=False,
            shard_rule='indices rank, rank+8,...; no padding; token merge still deduplicates',
            history_alignment='target<-frame composition of dataset pose_to_prev, real inputXYZ only',
            validity='LoadOCCLabelData validates contained source-grid ROI; PrepareGT validates local lattice/classes; sparse absence is empty only within confirmed ROI'))
        WriteJsonAtomic(output/'val_tokens.json', expected)
    del checkpoint
    indices = list(range(rank, len(dataset), world_size))
    loader = DataLoader(dataset, batch_size=1, sampler=indices, num_workers=args.workers,
                        collate_fn=FirstSample, pin_memory=False)
    collector = StatisticsCollector(b_enabled=True, output_jsonl=output/f'frames_rank{rank}.jsonl')
    started = time.monotonic()
    with torch.no_grad():
        for index, sample in zip(indices, loader):
            token = sample['token_seq'][-1]
            if token != tokens[index]: raise ValueError('Target changed during loading')
            current_points, history_points = AlignedRealPoints(sample)
            batch = dataset.CollateBatch([sample]); memory = None
            for frame in batch['frames'][:-1]:
                memory = model(MoveToDevice(frame, local_rank), memory=memory, b_memory_only=True)
            target = MoveToDevice(batch['frames'][-1], local_rank)
            raw = model.ForwardNetwork(target, memory=memory)
            pred = raw['occ']
            gt_indices, gt_labels = model.occ_head.PrepareGT(target, pred['logits'].device, 1)
            def GridXYZ(indices):
                if len(indices) and not bool((indices[:, 0] == 0).all()):
                    raise ValueError('Per-target statistics unexpectedly contain another batch')
                return indices[:, [3, 2, 1]].detach().cpu()
            frame_record = collector.ComputeAndAdd(target_id=token,
                pred_xyz=GridXYZ(pred['indices']), pred_labels=pred['logits'].argmax(1).cpu(),
                gt_xyz=GridXYZ(gt_indices), gt_labels=gt_labels.cpu(), grid_shape_xyz=model.occ_head.grid_shape_xyz,
                root_xyz=GridXYZ(pred['root_indices']), kept_root_xyz=GridXYZ(pred['root_indices'][pred['root_keep_mask']]),
                expanded_root_xyz=GridXYZ(pred['expanded_root_indices']), current_points_xyz=current_points,
                history_points_xyz=history_points, pc_range=model.occ_head.pc_range,
                b_expand_roots=model.occ_head.b_expand_roots, b_valid_roi_confirmed=True, actual_counts=pred['counts'])
            del frame_record, raw, pred, memory, target, batch, sample, current_points, history_points
    WriteJsonAtomic(output/f'rank{rank}_complete.json', dict(rank=rank, expected_count=len(indices),
        frame_count=len(collector.frames), elapsed_seconds_including_cpu_statistics=time.monotonic()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    dist.barrier()
    if rank == 0:
        summary = MergeStatistics([output/f'frames_rank{i}.jsonl' for i in range(world_size)], expected_target_ids=expected)
        WriteTables(output, summary)
        summary.update(b_complete=True, b_inference_benchmark=False, b_effectiveness_conclusion=False)
        WriteJsonAtomic(output/'summary.json', summary)
    dist.barrier(); dist.destroy_process_group()


def CPUCheck():
    # Noncommuting rotation/translation catches reverse transform composition.
    first = np.eye(4); first[:3, 3] = [1, 2, 0]
    second = np.eye(4); second[:2, :2] = [[0, -1], [1, 0]]; second[:3, 3] = [3, 0, 0]
    points = np.array([[2., 0., 0., -1.]])
    sample = {'frames':[dict(points=points, pose_to_prev=np.eye(4)), dict(points=points, pose_to_prev=first),
                        dict(points=points, pose_to_prev=second)]}
    current, history = AlignedRealPoints(sample)
    np.testing.assert_array_equal(current, points[:, :3])
    np.testing.assert_allclose(history, [[3, 2, 0], [1, 3, 0]])
    _, empty = AlignedRealPoints({'frames':sample['frames'][-1:]})
    assert empty.shape == (0, 3)
    shards = [list(range(rank, 5973, 8)) for rank in range(8)]
    assert sorted(sum(shards, [])) == list(range(5973))
    return dict(b_cpu_pass=True, b_gpu_executed=False, checks=['exact prev-to-current transform composition',
        'timefeature untouched/excluded fromXYZ', 'singleframe emptyhistory', '8rank5973 no-padding partition'])


def Main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true'); mode.add_argument('--cpu-check', action='store_true')
    for name in ('config', 'checkpoint', 'expected-tokens', 'output'): parser.add_argument('--'+name)
    parser.add_argument('--root-filter', choices=('configured', 'on', 'off'), default='configured')
    parser.add_argument('--expected-epoch', type=int, choices=range(1, 49), default=5)
    parser.add_argument('--workers', type=int, default=2); parser.add_argument('--cpu-threads', type=int, default=2)
    args = parser.parse_args()
    if args.cpu_check:
        result = CPUCheck()
        if args.output: Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result)); return
    if not all((args.config, args.checkpoint, args.expected_tokens, args.output)):
        parser.error('--run requires config,checkpoint,expected-tokens,output')
    Run(args)


if __name__ == '__main__': Main()
