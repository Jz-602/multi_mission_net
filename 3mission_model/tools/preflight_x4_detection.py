"""Bounded real-data X4 interface checks; never starts a formal training stage."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools')]


def Write(path, record):
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def Setup(args):
    import torch
    from easydict import EasyDict
    from build_label_manifest import LoadConfig
    from run_options import CheckRunConfig
    from WaymoDataset_multi import WaymoDatasetMulti
    cfg = EasyDict(LoadConfig(args.config))
    cfg.model.x4_detection_source = args.route
    for key in ('tasks', 'voxel_size', 'out_size_factor'): cfg.model[key] = cfg.common_variables[key]
    cfg.model.point_cloud_range = cfg.common_variables.pc_range
    CheckRunConfig(cfg)
    torch.set_num_threads(4)
    dataset = WaymoDatasetMulti(cfg.data.train_dataset, cfg.common_variables)
    lookup = {item['anchor_info']['token']: index for index, item in enumerate(dataset.anchor_list)}
    plan = json.loads(Path(args.sample_plan).read_text())
    indices = [lookup[token] for token in plan['train_tokens']]
    return cfg, dataset, indices, plan


def Batch(dataset, index, device):
    from trainer.trainer.three_task_runtime import MoveToDevice
    return MoveToDevice(dataset.CollateBatch([dataset[index]]), device)


def Digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def Snapshot(output):
    import numpy as np
    import torch
    result = {}
    def Collect(value, prefix):
        if torch.is_tensor(value): result[prefix] = value.detach().cpu()
        elif isinstance(value, dict):
            for key, child in value.items(): Collect(child, prefix+'/'+str(key))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value): Collect(child, prefix+'/'+str(index))
    Collect(output['det'], 'det'); Collect(output['terrain'], 'terrain')
    occ = output['occ']
    for coordinates, linked in [('indices', ['logits']), ('root_indices', ['root_logits', 'root_keep_mask']), ('expanded_root_indices', [])]:
        array = occ[coordinates].detach().cpu()
        order = torch.from_numpy(np.lexsort(array.numpy().T[::-1]).copy())
        result['occ/'+coordinates] = array[order]
        for key in linked: result['occ/'+key] = occ[key].detach().cpu()[order]
    return result


def SingleGPU(args):
    import torch
    from multi_mission_model import MultiMissionModel as Single
    from multi_mission_model_memory_x4 import MultiMissionModel as X4
    from occ_head import MatchRows
    from trainer.trainer.trainer_multi import DetachMemory
    from trainer.trainer.three_task_runtime import CheckFiniteGradients, ScalarizeLogs
    cfg, dataset, indices, plan = Setup(args)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    hard_token = plan['required_high_tokens'][0]
    hard_index = next(index for index in indices if dataset.anchor_list[index]['anchor_info']['token'] == hard_token)
    batch = Batch(dataset, hard_index, torch.device('cuda:0'))
    target = batch['frames'][-1]
    single = Single(cfg.model).cuda().eval(); single.load_state_dict(source, strict=True)
    with torch.no_grad():
        before = Snapshot(single.ForwardNetwork(target))
        repeat = Snapshot(single.ForwardNetwork(target))
    del single; torch.cuda.empty_cache()
    report = dict(b_pass=False, token=hard_token, nohistory={}, history={}, b_real_full_roi=True)
    for route in (('enhanced_e4',) if args.route == 'enhanced_e4' else ('full_f4', 'current_f4')):
        cfg.model.x4_detection_source = route
        torch.manual_seed(0); model = X4(cfg.model).cuda().eval()
        report.setdefault('initialization', {})[route] = model.LoadSingleState(source)
        with torch.no_grad(): actual = Snapshot(model.ForwardNetwork(target, memory=None))
        assert before.keys() == actual.keys()
        differences = {}
        for key in before:
            assert before[key].shape == actual[key].shape, key
            if before[key].dtype in (torch.bool, torch.int32, torch.int64):
                assert torch.equal(before[key], actual[key]), key
            elif before[key].numel():
                finite = torch.isfinite(before[key])
                if not bool(finite.all()):
                    # 预测洞本来就以NaN保存；必须先证明洞位置一致，不能忽略任意非有限值。
                    assert key == 'terrain/terrain_height_masked', key
                    for value in (before[key], repeat[key], actual[key]):
                        assert not bool(torch.isinf(value).any()), key
                        assert torch.equal(torch.isnan(value), ~finite), key
                assert torch.equal(torch.isfinite(repeat[key]), finite), key
                assert torch.equal(torch.isfinite(actual[key]), finite), key
                if not bool(finite.any()):
                    differences[key] = dict(finite_count=0, b_nan_mask_equal=True)
                    continue
                noise = float((before[key][finite]-repeat[key][finite]).abs().max())
                error = float((before[key][finite]-actual[key][finite]).abs().max())
                limit = max(1e-6, 10*noise, 1e-5*float(before[key][finite].abs().max()))
                assert error <= limit, (key, error, noise, limit)
                differences[key] = dict(max_abs=error, repeated_s_noise=noise, tolerance=limit,
                                        finite_count=int(finite.sum()), b_nan_mask_equal=True)
        report['nohistory'][route] = differences
        model.train(); torch.cuda.reset_peak_memory_stats()
        memory = None
        with torch.no_grad():
            for frame in batch['frames'][:-1]: memory = DetachMemory(model(frame, memory=memory, b_memory_only=True))
        observed = {}
        def Current(_module, _args, value): observed['current'] = value
        def Det(_module, arguments): observed['det'] = arguments[0]
        def Enhanced(_module, _args, value): observed['enhanced'] = value
        def Terrain(_module, arguments): observed['terrain'] = arguments[0]
        def OCC(_module, arguments):
            observed['occ'] = arguments[0]
            observed['occ_before'] = (arguments[0].indices.clone(), arguments[0].features.detach().clone())
            observed['skip'] = [len(value.indices) for value in arguments[1:]]
        handles = [model.encoder_stage4.register_forward_hook(Current), model.x4_memory_transformer.register_forward_hook(Enhanced), model.detection_bev_projector.register_forward_pre_hook(Det),
                   model.terrain_bev_projector.register_forward_pre_hook(Terrain), model.occ_head.register_forward_pre_hook(OCC)]
        output = model.ForwardNetwork(target, memory=memory)
        loss, logs = model.ComputeLoss(output, target)
        loss.backward(); CheckFiniteGradients(model)
        for name in ('encoder_stage4', 'x4_memory_transformer', 'detection_bev_projector', 'terrain_bev_projector', 'occ_head'):
            gradients = [parameter.grad for parameter in getattr(model, name).parameters() if parameter.requires_grad]
            assert all(value is not None for value in gradients), name
            assert sum(float(value.abs().sum()) for value in gradients) > 0, name
        if cfg.model.get('terrain', {}).get('b_predict_validity', False):
            report['terrain_head_gradients'] = {}
            for name in ('height_head', 'validity_head'):
                gradient = getattr(model.terrain_head, name).weight.grad
                assert gradient is not None and float(gradient.norm()) > 0, name
                report['terrain_head_gradients'][name] = float(gradient.norm())
        assert observed['occ'] is observed['terrain']
        torch.testing.assert_close(observed['occ'].indices, observed['occ_before'][0], rtol=0, atol=0)
        torch.testing.assert_close(observed['occ'].features, observed['occ_before'][1], rtol=0, atol=0)
        expected = observed['occ'].indices if route == 'full_f4' else observed['current'].indices
        torch.testing.assert_close(observed['det'].indices, expected, rtol=0, atol=0)
        rows = MatchRows(observed['det'].indices, observed['occ'].indices, observed['occ'].spatial_shape)
        assert bool((rows >= 0).all())
        expected_features = observed['enhanced'] if route == 'enhanced_e4' else observed['occ'].features[rows]
        torch.testing.assert_close(observed['det'].features, expected_features, rtol=0, atol=0)
        report['history'][route] = dict(loss=float(loss), tasks=ScalarizeLogs(logs), counts=output['counts'],
            skip_counts=observed['skip'], peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, b_detection_route_and_root_isolation=True)
        for handle in handles: handle.remove()
        del output, loss, logs, memory, observed, model; torch.cuda.empty_cache()
    report['b_pass'] = True; Write(Path(args.output)/'single_gpu.json', report)


def DDPWorker(args):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from multi_mission_model_memory_x4 import MultiMissionModel
    from trainer.trainer.trainer_multi import DetachMemory
    from trainer.trainer.three_task_runtime import CheckFiniteGradients, CheckFiniteLoss, ScalarizeLogs
    rank = int(os.environ['RANK']); torch.cuda.set_device(rank)
    dist.init_process_group('nccl'); device = torch.device('cuda', rank)
    cfg, dataset, indices, plan = Setup(args)
    import numpy as np
    torch.manual_seed(0)
    model = MultiMissionModel(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **{key: value for key, value in cfg.optimizer.items() if key != '_target_'})
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.steps_per_epoch*int(cfg.trainer.max_epochs),
        **{key: value for key, value in cfg.scheduler.items() if key not in ('_target_', 'epochs', 'steps_per_epoch', 'total_steps')})
    output = Path(args.output)
    if args.start_step:
        saved = torch.load(output/'ddp_step4.pth', map_location=device, weights_only=False)
        assert saved['step'] == args.start_step
        model.load_state_dict(saved['state_dict'], strict=True); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
    else:
        source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict']
        model.LoadSingleState(source); del source
    wrapped = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    wrapped.train(); torch.cuda.reset_peak_memory_stats()
    records = []
    for step in range(args.start_step+1, args.start_step+5):
        torch.manual_seed(step*100+rank); np.random.seed(step*100+rank)
        batch = Batch(dataset, indices[(step-1)*8+rank], device)
        memory = None
        with torch.no_grad():
            for frame in batch['frames'][:-1]: memory = DetachMemory(model(frame, memory=memory, b_memory_only=True))
        loss, logs, _ = wrapped(batch['frames'][-1], memory=memory)
        CheckFiniteLoss(loss); optimizer.zero_grad(); loss.backward(); CheckFiniteGradients(wrapped)
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 10.)
        optimizer.step(); scheduler.step()
        record = dict(step=step, loss=float(loss), tasks=ScalarizeLogs(logs), token=batch['frames'][-1]['token'])
        if step in (1, 4, 5, 8):
            signature = dict(parameters=Digest(model.named_parameters()), gradients=Digest((name, value.grad) for name, value in model.named_parameters() if value.grad is not None))
            signatures = [None]*8; dist.all_gather_object(signatures, signature)
            assert all(value == signature for value in signatures)
            record.update(b_all_ranks_equal=True, signature=signature)
        records.append(record)
        assert {int(state['step']) for state in optimizer.state.values()} == {step}
        assert scheduler.last_epoch == step
    record = dict(rank=rank, b_pass=True, records=records, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, scheduler_total_steps=args.steps_per_epoch*int(cfg.trainer.max_epochs),
                  sampler_boundary='64 fixed interface-test clips, no claim of formal epoch sampling or validation')
    Write(output/f'ddp_rank{rank}_through{step}.json', record)
    if rank == 0: torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), step=step), output/f'ddp_step{step}.pth')
    dist.barrier(); dist.destroy_process_group()


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(PROJECT/'configs/server_x4_screen_b.yaml'))
    parser.add_argument('--checkpoint', default=str(PROJECT/'log/server_runs_xyzt_completed_20261007/single/checkpoints/epoch_48.pth'))
    parser.add_argument('--sample-plan', default=str(PROJECT/'log/server_preparation/dynamic50_transition/temporal_preflight/cpu_preparation.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--route', choices=('full_f4', 'current_f4', 'enhanced_e4'), default='current_f4')
    parser.add_argument('--phase', choices=('run', 'single', 'ddp'), default='run')
    parser.add_argument('--start-step', type=int, default=0)
    parser.add_argument('--steps-per-epoch', type=int, default=4,
                        help='LR schedule denominator only; diagnostic still executes four plus four updates')
    args = parser.parse_args()
    if args.phase == 'single': SingleGPU(args); return
    if args.phase == 'ddp': DDPWorker(args); return
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip(), 'GPU busy'
    started = time.monotonic(); deadline = started+1200
    report = dict(b_pass=False, b_formal_training=False, commands=[])
    common = ['--config',args.config,'--checkpoint',args.checkpoint,'--sample-plan',args.sample_plan,'--output',str(output),'--route',args.route,
              '--steps-per-epoch',str(args.steps_per_epoch)]
    commands = [[sys.executable,'-B',__file__,*common,'--phase','single']]
    commands += [[sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',__file__,*common,'--phase','ddp','--start-step',str(start)] for start in (0, 4)]
    try:
        for index, command in enumerate(commands):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0' if index == 0 else '0,1,2,3,4,5,6,7', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', TF_CPP_MIN_LOG_LEVEL='3')
            with (output/f'phase{index}.log').open('x') as stream:
                process = subprocess.Popen(command, env=env, cwd=PROJECT.parent, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                item = dict(command=command, pid=process.pid); report['commands'].append(item); Write(output/'summary.json', report)
                try: item['returncode'] = process.wait(timeout=max(1,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                    raise RuntimeError('X4 preflight 20-minute budget exhausted')
                assert item['returncode'] == 0, item
        assert json.loads((output/'single_gpu.json').read_text())['b_pass']
        for rank in range(8):
            for step in (4,8): assert json.loads((output/f'ddp_rank{rank}_through{step}.json').read_text())['b_pass']
        report['b_pass'] = True
    except BaseException as error: report['error'] = repr(error); raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started; Write(output/'summary.json', report)
    Write(output/'success.json', report)


if __name__ == '__main__': Main()
