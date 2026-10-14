"""Fresh single-frame thesis model: 8-GPU four updates, restart, four updates, eight-target validation.

No pretrained/model-initialization checkpoint argument is accepted. The only resume
is this tool's own step-4 checkpoint. Uses the exact formal 1481*48 LR prefix; eight
actual micro-updates are interface diagnostics and cannot estimate a formal epoch.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / 'tools')]
from preflight_x4_detection import Batch, Digest, Write


def Worker(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from easydict import EasyDict
    from build_label_manifest import LoadConfig
    from run_options import CheckRunConfig, HashFile
    from WaymoDatasetV2 import WaymoDataset
    from multi_mission_model import MultiMissionModel
    from trainer.trainer.three_task_runtime import CheckFiniteLoss, CheckFiniteGradients, ScalarizeLogs
    from trainer.trainer.multitask_validator import AccumulateBatchRecords, GatherTokenRecords, FinalizeThreeTaskMetrics
    from training_memory import TrainingCPUMemorySampler

    rank, local_rank, world_size = (int(os.environ[key]) for key in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    if world_size != 8:
        raise ValueError('This preflight requires eight real DDP ranks')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    torch.set_num_threads(4)
    torch.manual_seed(0); np.random.seed(0)
    cfg = EasyDict(LoadConfig(args.config)); CheckRunConfig(cfg)
    if int(cfg.trainer.max_epochs) != 48 or not cfg.model.get('terrain', {}).get('b_predict_validity', False):
        raise ValueError('Expected the new 48-epoch single config with terrain-validity head')
    if 'x4_detection_source' in cfg.model:
        raise ValueError('This preflight is single-frame only; no temporal initialization')
    for key in ('tasks', 'voxel_size', 'out_size_factor'):
        cfg.model[key] = cfg.common_variables[key]
    cfg.model.point_cloud_range = cfg.common_variables.pc_range
    cfg.model.occ.b_collect_occ_stage_statistics = True
    cfg.model.occ.b_collect_root_diagnostics = True
    dataset = WaymoDataset(cfg.data.train_dataset, cfg.common_variables)
    val_dataset = WaymoDataset(cfg.data.val_dataset, cfg.common_variables)
    lookup = {info['token']: index for index, info in enumerate(dataset.infos)}
    val_lookup = {info['token']: index for index, info in enumerate(val_dataset.infos)}
    plan = json.loads(Path(args.sample_plan).read_text())
    if len(plan['train_tokens']) < 64 or not set(plan['required_high_tokens']) <= set(plan['train_tokens'][:64]):
        raise ValueError('Micro-plan must include 64 real targets and the established high-occupancy anchors')
    if len(plan['val_tokens']) < 8:
        raise ValueError('Need eight independent validation targets')
    indices = [lookup[token] for token in plan['train_tokens'][:64]]
    val_indices = [val_lookup[token] for token in plan['val_tokens'][:8]]
    model = MultiMissionModel(cfg.model).to(device)
    if model.terrain_head.validity_head is None:
        raise AssertionError('New validity head missing')
    optimizer = torch.optim.AdamW(model.parameters(), **{key:value for key,value in cfg.optimizer.items() if key != '_target_'})
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=1481*48,
        **{key:value for key,value in cfg.scheduler.items() if key not in ('_target_','epochs','steps_per_epoch','total_steps')})
    output = Path(args.output)
    config_signature = json.dumps(cfg, sort_keys=True)
    restored = None
    if args.start_step:
        saved = torch.load(output/'ddp_step4.pth', map_location=device, weights_only=False)
        if saved['step'] != 4 or args.start_step != 4 or saved['config_signature'] != config_signature:
            raise ValueError('Only the same-config step-4 diagnostic may resume')
        model.load_state_dict(saved['state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        if scheduler.last_epoch != 4 or scheduler.total_steps != 71088:
            raise AssertionError('Formal LR prefix was not restored')
        if {int(state['step']) for state in optimizer.state.values()} != {4}:
            raise AssertionError('Optimizer step mismatch on restore')
        restored = {'checkpoint_sha256': HashFile(output/'ddp_step4.pth'),
                    'parameters': Digest(model.named_parameters()), 'scheduler_last_epoch': scheduler.last_epoch}
        if restored['parameters'] != saved['parameter_digest']:
            raise AssertionError('Restored parameters differ from saved checkpoint')
        del saved
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    wrapped.train(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    sampler = TrainingCPUMemorySampler(output/f'cpu_rank{rank}_from{args.start_step}.jsonl')
    sampler.Start(); started = time.monotonic(); records = []
    try:
        for step in range(args.start_step+1, args.start_step+5):
            torch.manual_seed(step*100+rank); np.random.seed(step*100+rank)
            batch = Batch(dataset, indices[(step-1)*8+rank], device)
            optimizer.zero_grad(set_to_none=True)
            lr = [group['lr'] for group in optimizer.param_groups]
            loss, logs = wrapped(batch)
            CheckFiniteLoss(loss); loss.backward(); CheckFiniteGradients(wrapped)
            gradients = {}
            for name in ('terrain_head.height_head.weight','terrain_head.validity_head.weight','occ_head.classifier.weight','detection_head.tasks.0.hm.0.weight'):
                parameter = dict(model.named_parameters()).get(name)
                if parameter is not None:
                    gradients[name] = None if parameter.grad is None else float(parameter.grad.norm())
            gradients['detection_head_all'] = sum(float(parameter.grad.norm()) for parameter in model.detection_head.parameters() if parameter.grad is not None)
            for name in ('terrain_head.height_head.weight','terrain_head.validity_head.weight','occ_head.classifier.weight','detection_head_all'):
                if gradients.get(name, 0.) is None or gradients.get(name, 0.) <= 0.:
                    raise AssertionError(f'No nonzero task-head gradient: {name}')
            tasks = ScalarizeLogs(logs)
            terrain = next(row for row in tasks if row['task'] == ['terrain'])
            for key in ('terrain_main_loss','terrain_gradient_loss','terrain_validity_loss'):
                if not np.isfinite(terrain[key]):
                    raise AssertionError(f'Nonfinite terrain subloss {key}')
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(wrapped.parameters(), float(cfg.trainer.clip_grad_val)))
            optimizer.step(); scheduler.step()
            record = dict(step=step, loss=float(loss), tasks=tasks, token=batch['token'],
                          parameter_group_lr=lr, task_head_gradient_norms=gradients,
                          gradient_norm_before_clip=gradient_norm)
            if step in (1,4,5,8):
                signature = dict(parameters=Digest(model.named_parameters()),
                                 gradients=Digest((name,p.grad) for name,p in model.named_parameters() if p.grad is not None))
                signatures = [None]*world_size; dist.all_gather_object(signatures, signature)
                if any(item != signature for item in signatures):
                    raise AssertionError('DDP gradients or parameters differ across ranks')
                record.update(b_all_ranks_equal=True, signature=signature)
            records.append(record)
            if {int(state['step']) for state in optimizer.state.values()} != {step} or scheduler.last_epoch != step:
                raise AssertionError('Optimizer/scheduler did not advance exactly once per update')
            del loss, logs, batch
        torch.cuda.synchronize()
    finally:
        cpu_memory = sampler.Stop()
    cpu_memory['micro_scope'] = 'Four diagnostic updates in this rank; direct synchronous dataset fetch, no data-loader subprocesses. Not a formal epoch peak.'
    training = dict(elapsed_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3, cpu_memory=cpu_memory)
    if cpu_memory['sampling_error'] or cpu_memory['complete_pss_samples'] == 0:
        raise RuntimeError('CPU PSS sampling failed; inspect raw samples')
    if rank == 0:
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        step=step, config_signature=config_signature, parameter_digest=Digest(model.named_parameters())), output/f'ddp_step{step}.pth')
    dist.barrier()
    if step == 8:
        # Match formal DDP eval: broadcast rank0 buffers before all ranks evaluate.
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)
        wrapped.eval(); torch.cuda.reset_peak_memory_stats()
        val_started = time.monotonic(); batch = Batch(val_dataset, val_indices[rank], device)
        with torch.no_grad():
            prediction = model(batch)
        records_by_token = {}; AccumulateBatchRecords(records_by_token, prediction, batch, model.terrain_criterion)
        frame = records_by_token[batch['token'][0]]
        if model.occ_head.root_score is not None and 'root_diagnostics' not in frame['occ']:
            raise AssertionError('Prefilter root group diagnostics missing from validation')
        if 'terrain_classification_count' not in frame['terrain'] or 'stages' not in frame['occ']:
            raise AssertionError('New validity/OCC-stage metrics missing from actual evaluation')
        all_records = GatherTokenRecords(records_by_token, rank, world_size)
        Write(output/f'validation_rank{rank}.json', dict(token=batch['token'][0],
            elapsed_seconds_including_statistics=time.monotonic()-val_started,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
            stage_statistics_wall_seconds=frame['occ']['stage_statistics_wall_seconds']))
        if rank == 0:
            metrics = FinalizeThreeTaskMetrics(all_records, val_dataset, cfg.model, 0, logging.getLogger('preflight'))
            if len(all_records) != 8 or metrics['occ']['stages']['frame_count'] != 8:
                raise AssertionError('Micro-validation target count incorrect')
            if 'validity' not in metrics['terrain']:
                raise AssertionError('Global terrain validity metrics missing')
            Write(output/'validation_metrics.json', dict(b_formal_validation=False, target_count=8,
                metrics=metrics, raw_token_records={token:{'terrain':record['terrain'],'occ':record['occ']} for token,record in all_records.items()}))
    report = dict(rank=rank, b_pass=True, records=records, resources=training, restored=restored,
                  initialization='new random single model; only own step4 resumes', scheduler_total_steps=71088,
                  formal_updates_per_epoch=1481, micro_updates_this_process=4, b_formal_training=False,
                  sample_plan_sha256=HashFile(args.sample_plan), config_sha256=HashFile(args.config),
                  limitation='Eight updates and eight validation targets only; not formal throughput or accuracy evidence')
    Write(output/f'ddp_rank{rank}_through{step}.json', report)
    dist.barrier(); dist.destroy_process_group()


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--sample-plan', default=str(PROJECT/'log/server_preparation/dynamic50_transition/temporal_preflight/cpu_preparation.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase', choices=('run','ddp'), default='run')
    parser.add_argument('--start-step', type=int, choices=(0,4), default=0)
    parser.add_argument('--timeout-seconds', type=int, default=1200)
    args = parser.parse_args()
    if args.phase == 'ddp':
        Worker(args); return
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU busy: schedule the single preflight after the active job')
    started = time.monotonic(); report = dict(b_pass=False, b_formal_training=False, commands=[])
    try:
        for start in (0,4):
            command = [sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=8',__file__,
                       '--config',args.config,'--sample-plan',args.sample_plan,'--output',str(output),'--phase','ddp','--start-step',str(start)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TF_CPP_MIN_LOG_LEVEL='3')
            with (output/f'phase_from{start}.log').open('x') as stream:
                process = subprocess.Popen(command, env=env, cwd=PROJECT.parent, stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                item = dict(command=command,pid=process.pid); report['commands'].append(item); Write(output/'summary.json', report)
                try:
                    item['returncode'] = process.wait(timeout=max(1,args.timeout_seconds-(time.monotonic()-started)))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
                    raise RuntimeError('Thesis single preflight time budget exceeded')
                if item['returncode']:
                    raise RuntimeError(f'DDP preflight phase failed: {start}, exit={item["returncode"]}')
        for rank in range(8):
            for step in (4,8):
                if not json.loads((output/f'ddp_rank{rank}_through{step}.json').read_text())['b_pass']:
                    raise AssertionError('Missing passing rank record')
        if json.loads((output/'validation_metrics.json').read_text())['target_count'] != 8:
            raise AssertionError('Missing eight-target validation')
        report['b_pass'] = True
    except BaseException as error:
        report['error'] = repr(error); raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started; Write(output/'summary.json', report)
    Write(output/'success.json', report)


if __name__ == '__main__':
    Main()
