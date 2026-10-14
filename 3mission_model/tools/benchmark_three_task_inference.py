"""Batch1 FP32 three-task inference; cold clips and genuine rolling online streams.

Online points [N,4] are current-vehicle xyz+time(0s); poses [4,4] map
vehicle->world. Memory keeps production features/coordinates in the prior frame.
Cold clips retain production target-relative history time features. No GT metrics.
"""
import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def Statistics(values):
    import numpy as np
    return {'count': len(values), 'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'minimum': min(values), 'maximum': max(values)}


class RSSSampler:
    """Sum RSS of main and recursive children; shared pages are counted repeatedly."""
    def __init__(self, interval):
        import psutil
        self.process = psutil.Process()
        self.interval = interval
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.Run, daemon=True)

    def Run(self):
        import psutil
        while not self.stop.is_set():
            rows = []
            for process in [self.process] + self.process.children(recursive=True):
                try:
                    rows.append((process.pid, process.memory_info().rss))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.samples.append({'wall_time': time.time(), 'processes': rows,
                                 'sum_rss_bytes': sum(row[1] for row in rows)})
            self.stop.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=['single', 'cold', 'online'], default='cold')
    parser.add_argument('--indices', type=int, nargs='+', default=[100, 2720])
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--input-metadata', type=Path,
                        help='Verified input-only manifest; fixed val targets, no GT-bearing infos opened')
    parser.add_argument('--root', choices=['config', 'on', 'off'], default='config')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--timing-mode', choices=['legacy', 'pipeline', 'modules'], default='legacy',
                        help='pipeline: outer-sync only, no diagnostic hooks; modules: separate inclusive call diagnostics')
    parser.add_argument('--rss-interval', type=float, default=.02)
    parser.add_argument('--collect-pss', dest='b_collect_pss', action='store_true',
                        help='Independent 1-second PSS/RSS resource pass; timings are not formal latency')
    parser.add_argument('--cpu-check', dest='b_cpu_check', action='store_true')
    args = parser.parse_args()
    if args.b_cpu_check:
        import numpy as np
        assert Statistics([1., 2., 3.])['median'] == 2.
        assert Statistics([1., 2., 9.])['mean'] == 4.
        previous_pose, current_pose = np.eye(4), np.eye(4)
        previous_pose[0, 3], current_pose[0, 3] = 2., 5.
        assert (np.linalg.inv(current_pose) @ previous_pose @ np.array([1., 0., 0., 1.]))[0] == -2.
        if args.b_collect_pss:
            import tempfile
            sys.path.insert(0, str(ROOT))
            from inference_memory import InferenceMemorySampler
            with tempfile.TemporaryDirectory() as directory:
                with InferenceMemorySampler(Path(directory) / 'cpu_memory.jsonl') as sampler:
                    time.sleep(1.1)
                assert sampler.samples and sampler.sampling_error is None
                assert sampler.Report()['windows']['whole_resource_pass']['complete_pss_samples'] > 0
        else:
            with RSSSampler(args.rss_interval) as sampler:
                sampler.stop.wait(args.rss_interval * 3)
            assert sampler.samples and all(row['sum_rss_bytes'] > 0 for row in sampler.samples)
        print(json.dumps({'b_cpu_check_pass': True, 'b_gpu_executed': False,
                          'statistics': Statistics([1., 2., 3.]), 'rss_samples': len(sampler.samples)}))
        return
    if not (args.config and args.checkpoint and args.output):
        parser.error('--config, --checkpoint and --output are required for GPU measurement')
    if args.repeats < 5 or args.warmup < 1 or args.rss_interval <= 0:
        parser.error('require repeats>=5, warmup>=1, positive RSS interval')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from easydict import EasyDict
    from torch.utils.data import DataLoader, Subset
    from LYF_train_ODSS import CfgFromYamlFile, SetRandomSeed
    from WaymoDataset_multi import WaymoDatasetMulti, TARGET_LABEL_PIPELINES
    from multi_mission_model import MultiMissionModel as SingleModel
    from multi_mission_model_memory_x4 import MultiMissionModel as X4Model
    from run_options import CheckRunConfig
    from inference_metadata import ConfigureInferenceMetadata
    from inference_memory import InferenceMemorySampler
    from inference_timing import ModuleTiming
    from trainer.trainer.three_task_runtime import MoveToDevice
    torch.set_num_threads(4)
    SetRandomSeed(0, 0)
    report = {'arguments': vars(args) | {'output': str(args.output)}, 'cases': [],
              'scope': 'batch1 eval/no_grad FP32; target ForwardNetwork + production ExportPredictions (NMS, OCC argmax, CPU transfers); no GT metrics, losses or plots',
              'rss_scope': 'main+recursive loader children RSS sum, includes CPU dataset/model/loading; shared pages repeated; sampling may miss short peaks',
              'online_scope': '15 actual consecutive history frames build memory, then real consecutive new frames update it; warmup frames discarded; no replay/frozen-memory target timings',
              'online_time_feature': 'each newly encoded frame time=0s; prior memory features are not re-encoded as their relative age changes; cold uses production target-relative history times'}
    report['measurement_role'] = ('resource_only' if args.b_collect_pss else
                                  'latency' if args.timing_mode == 'pipeline' else 'diagnostic')
    report['b_latency_eligible'] = not args.b_collect_pss and args.timing_mode == 'pipeline'
    report['timing_mode'] = args.timing_mode
    report['pipeline_timing_scope'] = 'CPU-preloaded input to H2D + production voxelization + all cold history/target forwards + ExportPredictions + final CUDA synchronization; loader CPU preparation/I/O separately measured. Online additionally includes pose preparation; online and cold input protocols remain distinct.'
    report['timing_instrumentation'] = ('No OCC hooks, module wrappers, intermediate benchmark synchronization or background memory sampler unless --collect-pss. Production internal synchronizations/transfers unchanged.' if args.timing_mode == 'pipeline' else 'Diagnostic timings include instrumentation; not formal pipeline latency.')
    if args.b_collect_pss:
        report['timing_policy'] = 'PSS sampling adds host work: all timings are diagnostic; repeat without --collect-pss for formal latency.'
    resource_sampler = (InferenceMemorySampler(args.output / 'cpu_memory.jsonl')
                        if args.b_collect_pss else RSSSampler(args.rss_interval)
                        if args.timing_mode == 'legacy' else nullcontext(None))
    with resource_sampler as sampler:
        if args.b_collect_pss:
            sampler.Begin('initialization')
        cfg = CfgFromYamlFile(args.config, EasyDict())
        CheckRunConfig(cfg)
        report.update(config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                      checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                      source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in (Path(__file__), ROOT/'occ_head.py', ROOT/'terrain_head.py',
                                                  ROOT/'multi_mission_model.py', ROOT/'multi_mission_model_memory_x4.py',
                                                  ROOT/'WaymoDataset_multi.py', ROOT/'inference_metadata.py',
                                                  ROOT/'inference_memory.py', ROOT/'training_memory.py', ROOT/'inference_timing.py')},
                      device=torch.cuda.get_device_name(), precision='float32',
                      b_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      b_tf32_cudnn=torch.backends.cudnn.allow_tf32,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda)
        data_cfg = copy.deepcopy(cfg.data[f'{args.split}_dataset'])
        data_cfg.loading_pipelines = [name for name in data_cfg.loading_pipelines if name not in TARGET_LABEL_PIPELINES]
        data_cfg.prepare_label = None
        data_cfg.augmentation = None
        data_cfg.sampler = None
        data_cfg.b_use_gt_sampling = False
        data_cfg.b_random_history_interval = False
        if args.mode == 'single':
            data_cfg.warmup_len = 0
        report['input_metadata'] = (
            ConfigureInferenceMetadata(data_cfg, args.input_metadata, args.split)
            if args.input_metadata else
            {'b_strict_input_only_metadata': False,
             'scope': 'Per-frame GT loaders disabled; legacy infos may still contain embedded annotations.'})
        dataset = WaymoDatasetMulti(data_cfg, cfg.common_variables)
        if args.mode != 'single' and dataset.warmup_len != 15:
            raise ValueError('temporal benchmark requires exactly 15 history frames')
        for key in ('tasks', 'out_size_factor', 'voxel_size'):
            cfg.model[key] = cfg.common_variables[key]
        cfg.model.point_cloud_range = cfg.common_variables.pc_range
        # 正式配置开启的根特征诊断只服务训练/评价，不计入模型推理开销。
        report['disabled_measurement_statistics'] = {
            key: bool(cfg.model.occ.get(key, False))
            for key in ('b_collect_root_diagnostics', 'b_collect_occ_stage_statistics')}
        for key in report['disabled_measurement_statistics']:
            cfg.model.occ[key] = False
        model = (X4Model if 'x4_detection_source' in cfg.model else SingleModel)(cfg.model).cuda().eval()
        if args.mode != 'single' and not isinstance(model, X4Model):
            raise ValueError('temporal benchmark expects screened X4 model')
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        # 同权重取消筛选，保留 root_score/BCE 定义；入口将此配置传给 OCC forward。
        if args.root != 'config':
            if not isinstance(model, X4Model):
                raise ValueError('selection-only root override requires the screened X4 model')
            if model.occ_head.root_score is None:
                raise ValueError('controlled root comparison requires the original root classifier')
            model.config.occ.b_apply_root_filter = args.root == 'on'
        report['effective_model_config'] = dict(cfg.model)
        report['effective_data_config'] = dict(data_cfg)
        report['checkpoint_meta'] = checkpoint.get('meta', {})
        report['b_root_classifier_present'] = model.occ_head.root_score is not None
        report['b_apply_root_filter'] = bool(model.config.occ.get('b_apply_root_filter', True))
        report['root_comparison'] = (
            'same-weight selection override; root classifier retained, no losses computed'
            if args.root != 'config' else
            'checkpoint architecture and configured root selection; no losses computed')
        del checkpoint
        occ_events = []
        def OCCStart(module, inputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            occ_events.append([event])
        def OCCEnd(module, inputs, outputs):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            occ_events[-1].append(event)
        handles = ([model.occ_head.register_forward_pre_hook(OCCStart), model.occ_head.register_forward_hook(OCCEnd)]
                   if args.timing_mode == 'legacy' else [])
        module_timer = ModuleTiming(model, torch.cuda) if args.timing_mode == 'modules' else None
        if args.b_collect_pss:
            sampler.End('initialization')
        with torch.no_grad(), (module_timer if module_timer else nullcontext()):
            if module_timer:
                report['module_timing_definition'] = module_timer.Definition()
            for case_number, index in enumerate(args.indices):
                window_prefix = f'case_{case_number}_index_{index}'
                if args.b_collect_pss:
                    sampler.Begin(window_prefix)
                    sampler.Begin(window_prefix + '/loader')
                memory = None
                rss_start = len(sampler.samples) if sampler else 0
                anchor = dataset.anchor_list[index]
                if args.mode == 'online':
                    stream = copy.copy(dataset)
                    stream.warmup_len, stream.seq_len = 0, 1
                    sequence = dataset.full_sequence_info_dict[anchor['sequence_name']]
                    stop = anchor['target_full_idx'] + 1
                    start = stop - (15 + args.warmup + args.repeats)
                    if start < 0:
                        raise ValueError(f'index {index}: insufficient true preceding frames for online benchmark')
                    stream.anchor_list = [{'sequence_name': anchor['sequence_name'], 'target_full_idx': i,
                                           'anchor_info': sequence[i]} for i in range(start, stop)]
                    selected = stream
                else:
                    selected = Subset(dataset, [index])
                loader = DataLoader(selected, batch_size=1, num_workers=args.workers,
                                    collate_fn=dataset.CollateBatch, shuffle=False)
                # CPU loading is measured separately; GPU scopes exclude loader I/O.
                load_start = time.perf_counter()
                batches = list(loader)
                load_seconds = time.perf_counter() - load_start
                del loader
                if args.b_collect_pss:
                    sampler.End(window_prefix + '/loader')
                torch.cuda.empty_cache()
                baseline_allocated = torch.cuda.memory_allocated()
                baseline_reserved = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                memory, rows, tokens = None, [], []
                prefill_module_calls = []
                def Target(frame, memory):
                    if args.timing_mode == 'legacy':
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                    outputs = model.ForwardNetwork(frame, memory) if isinstance(model, X4Model) else model.ForwardNetwork(frame)
                    if args.timing_mode == 'legacy':
                        torch.cuda.synchronize()
                        forward_ms = 1000 * (time.perf_counter() - tick)
                    prediction = model.ExportPredictions(outputs, frame)
                    record = {'counts': outputs['counts'], 'occ_counts': outputs['occ']['counts'],
                              'token': frame['token'][0]}
                    if args.timing_mode == 'legacy':
                        torch.cuda.synchronize()
                        record.update(target_forward_ms=forward_ms,
                                      target_with_export_ms=1000 * (time.perf_counter() - tick),
                                      occ_forward_gpu_ms=occ_events[-1][0].elapsed_time(occ_events[-1][1]))
                    del prediction
                    return outputs.get('memory'), record
                if args.mode == 'online':
                    if args.b_collect_pss:
                        sampler.Begin(window_prefix + '/history_prefill')
                    previous_pose = None
                    for position, batch in enumerate(batches):
                        if args.b_collect_pss and position in (15, 15 + args.warmup):
                            torch.cuda.synchronize()
                            sampler.End(window_prefix + ('/history_prefill' if position == 15 else '/warmup'))
                            sampler.Begin(window_prefix + ('/warmup' if position == 15 else '/measurement'))
                        if module_timer:
                            module_timer.BeginIteration()
                            module_timer.context = {'position': position, 'phase': 'history' if position < 15 else 'target'}
                        if args.timing_mode != 'legacy':
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                        frame = batch['frames'][0]
                        pose = frame['pose'][0].numpy()
                        # Previous vehicle -> world -> current vehicle; augmentation disabled.
                        if previous_pose is not None:
                            frame['pose_to_prev'] = torch.from_numpy((np.linalg.inv(pose) @ previous_pose).astype(np.float32))[None]
                        previous_pose = pose.copy()
                        tokens.append(frame['token'][0])
                        if args.timing_mode == 'legacy':
                            torch.cuda.synchronize()
                            tick = time.perf_counter()
                        frame = MoveToDevice(frame, 'cuda')
                        if position < 15:
                            memory = model.ForwardNetwork(frame, memory, b_memory_only=True)['memory']
                            if module_timer:
                                torch.cuda.synchronize()
                                prefill_module_calls.append(module_timer.FinishIteration())
                        else:
                            memory, row = Target(frame, memory)
                            if args.timing_mode != 'legacy':
                                torch.cuda.synchronize()
                            row['pipeline_ms'] = 1000 * (time.perf_counter() - tick)
                            if module_timer:
                                row['module_calls'] = module_timer.FinishIteration()
                            if position >= 15 + args.warmup:
                                rows.append(row)
                        del frame
                else:
                    if args.b_collect_pss:
                        sampler.Begin(window_prefix + '/warmup')
                    for repeat in range(args.warmup + args.repeats):
                        if args.b_collect_pss and repeat == args.warmup:
                            torch.cuda.synchronize()
                            sampler.End(window_prefix + '/warmup')
                            sampler.Begin(window_prefix + '/measurement')
                        if module_timer:
                            module_timer.BeginIteration()
                        memory = None
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        for history_index, cpu_frame in enumerate(batches[0]['frames'][:-1]):
                            if module_timer:
                                module_timer.context = {'position': history_index, 'phase': 'history'}
                            frame = MoveToDevice(cpu_frame, 'cuda')
                            memory = model.ForwardNetwork(frame, memory, b_memory_only=True)['memory']
                            del frame
                        if module_timer:
                            module_timer.context = {'position': len(batches[0]['frames']) - 1, 'phase': 'target'}
                        frame = MoveToDevice(batches[0]['frames'][-1], 'cuda')
                        target_memory, row = Target(frame, memory)
                        if args.timing_mode != 'legacy':
                            torch.cuda.synchronize()
                        row['pipeline_ms'] = 1000 * (time.perf_counter() - tick)
                        if module_timer:
                            row['module_calls'] = module_timer.FinishIteration()
                        if repeat >= args.warmup:
                            rows.append(row)
                        del frame, memory, target_memory
                    memory = None
                    tokens = [frame['token'][0] for frame in batches[0]['frames']]
                torch.cuda.synchronize()
                if args.b_collect_pss:
                    sampler.End(window_prefix + '/measurement')
                    sampler.End(window_prefix)
                del memory
                case = {'index': index, 'anchor_token': anchor['anchor_info']['token'], 'sequence': anchor['sequence_name'],
                        'tokens': tokens, 'loader_seconds': load_seconds, 'samples': rows,
                        'input_scope': 'CPU-preloaded frame batches; loader time separate, pipeline includes H2D and production voxelization (including its CPU work), excludes CPU dataset I/O/preparation',
                        'statistics_ms': {key: Statistics([row[key] for row in rows]) for key in
                                          (('pipeline_ms', 'target_forward_ms', 'target_with_export_ms', 'occ_forward_gpu_ms')
                                           if args.timing_mode == 'legacy' else ('pipeline_ms',))},
                        'cuda_scope': 'reset before initial warmup; includes resident model, H2D frames, memory, raw outputs, export temporaries; caching allocator reserved peak',
                        'baseline_allocated_bytes': baseline_allocated,
                        'baseline_reserved_bytes': baseline_reserved,
                        'peak_extra_allocated_bytes': torch.cuda.max_memory_allocated() - baseline_allocated,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'rss_sum_peak_bytes': (None if args.b_collect_pss or sampler is None else
                                               max((row['sum_rss_bytes'] for row in sampler.samples[rss_start:]), default=None))}
                if args.b_collect_pss:
                    case['cpu_memory_window'] = window_prefix
                    case['b_latency_eligible'] = False
                if module_timer:
                    case['online_prefill_module_calls'] = prefill_module_calls
                    groups = {}
                    for row in rows:
                        for call in row['module_calls']:
                            key = call['context']['phase'] + '/' + call['name']
                            groups.setdefault(key, []).append(call)
                    case['module_per_call_statistics_ms'] = {
                        key: {metric: Statistics([call[metric] for call in calls])
                              for metric in ('host_inclusive_ms', 'cuda_stream_span_inclusive_ms')}
                        for key, calls in groups.items()}
                    case['module_statistics_scope'] = 'Per invocation in measured iterations only, grouped by history/target and call name; warmup/prefill excluded. Inclusive nested spans are not additive. Raw online prefill calls retained separately.'
                case['b_latency_eligible'] = report['b_latency_eligible']
                report['cases'].append(case)
                (args.output / 'partial.json').write_text(json.dumps(report, default=str, indent=2))
                del batches
        for handle in handles:
            handle.remove()
    if args.b_collect_pss:
        report['cpu_memory'] = sampler.Report()
        report['rss_scope'] = report['cpu_memory']['scope'] + ' ' + report['cpu_memory']['aggregation']
        report['rss_sum_peak_bytes'] = report['cpu_memory']['windows']['whole_resource_pass']['sampled_peak_rss_sum_bytes']
        for case in report['cases']:
            window = report['cpu_memory']['windows'][case['cpu_memory_window']]
            case['rss_sum_peak_bytes'] = window['sampled_peak_rss_sum_bytes']
            case['pss_sum_peak_bytes'] = window['sampled_peak_pss_sum_bytes']
    elif sampler is not None:
        report['rss_sum_peak_bytes'] = max(row['sum_rss_bytes'] for row in sampler.samples)
        report['rss_samples'] = sampler.samples
    else:
        report['rss_sum_peak_bytes'] = None
        report['rss_samples'] = []
        report['rss_scope'] = 'Not sampled in pipeline/modules pass without --collect-pss; use an independent resource pass.'
    report['b_complete'] = True
    (args.output / 'report.json').write_text(json.dumps(report, default=str, indent=2) + '\n')
    print(str(args.output / 'report.json'))


if __name__ == '__main__':
    Main()
