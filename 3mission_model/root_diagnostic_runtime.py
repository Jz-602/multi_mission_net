"""有限步真实恢复诊断；复用训练步，不将半轮检查点伪装成完整epoch。"""
import hashlib
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist

from dynamic_sampler import PrepareEpochSampling, SamplingDefinition
from run_options import HashFile
from trainer.trainer.three_task_runtime import AppendJsonLine, BuildCheckpointMeta, SaveJson, TaskLogMap
from trainer.utils.checkpoint import save_checkpoint


ARMS = {'original_resume': ('none', False), 'reset_none': ('none', True),
        'reset_layer_norm': ('layer_norm', True)}


def ResetRootHead(head, optimizer):
    """只重置root Linear及其Adam状态；共享主干、其他任务与优化器状态保留。"""
    parameters = list(head.root_score.parameters())
    before_count = len(optimizer.state)
    removed = sum(parameter in optimizer.state for parameter in parameters)
    for parameter in parameters:
        optimizer.state.pop(parameter, None)
    with torch.no_grad():
        head.root_score.weight.zero_()
        head.root_score.bias.fill_(math.log(.01 / .99))
    return {'optimizer_states_before': before_count, 'optimizer_states_removed': removed,
            'optimizer_states_after': len(optimizer.state), 'delete_probability': .01}


def RootStateSHA(head):
    digest = hashlib.sha256()
    for parameter in head.root_score.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def CheckRootAgreement(head):
    hashes = [RootStateSHA(head)]
    if dist.is_initialized():
        hashes = [None] * dist.get_world_size()
        dist.all_gather_object(hashes, RootStateSHA(head))
    if len(set(hashes)) != 1:
        raise RuntimeError('Root classifier differs between DDP ranks')
    return hashes


def InstrumentOptimizeStep(trainer, head, record):
    """仍执行原optimize_step；在其真实clip调用前后读取已DDP归约的梯度。"""
    original_step = trainer.optimize_step
    parameters = dict(head.root_score.named_parameters())

    def WrappedStep(loss):
        before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        record.clear()
        record['learning_rates_before'] = list(trainer.current_lr)
        original_clip = torch.nn.utils.clip_grad_norm_

        def ObserveClip(all_parameters, max_norm, *args, **kwargs):
            record['root_gradient_pre_clip'] = {
                name: float(parameter.grad.detach().norm()) if parameter.grad is not None else None
                for name, parameter in parameters.items()}
            total_norm = original_clip(all_parameters, max_norm, *args, **kwargs)
            record['global_gradient_norm_pre_clip'] = float(total_norm)
            record['clip_max_norm'] = float(max_norm)
            record['global_clip_factor'] = min(1., float(max_norm) / (float(total_norm) + 1e-6))
            record['root_gradient_post_clip'] = {
                name: float(parameter.grad.detach().norm()) if parameter.grad is not None else None
                for name, parameter in parameters.items()}
            return total_norm

        if trainer.clip_grad_val <= 0:
            raise ValueError('Root diagnostic expects original positive global clipping threshold')
        # 当前rank只在同步训练线程使用此上下文；finally保证异常不遗留替换。
        torch.nn.utils.clip_grad_norm_ = ObserveClip
        try:
            original_step(loss)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        if 'global_clip_factor' not in record:
            raise RuntimeError('Original optimizer did not invoke its configured gradient clipping')
        record['learning_rates_after'] = list(trainer.current_lr)
        record['root_parameter_updates'] = {}
        for name, parameter in parameters.items():
            delta = parameter.detach() - before[name]
            norm = float(before[name].norm())
            record['root_parameter_updates'][name] = {
                'parameter_l2_before': norm, 'parameter_l2_after': float(parameter.detach().norm()),
                'delta_l2': float(delta.norm()), 'delta_absmax': float(delta.abs().max()),
                'relative_delta_l2': float(delta.norm()) / norm if norm else None}
            if name == 'bias':
                record['root_parameter_updates'][name].update(
                    value_before=float(before[name].item()), value_after=float(parameter.detach().item()))
    return WrappedStep


def TrackRootOptimization(trainer):
    """正式训练每10次更新读根梯度/参数，不改变原优化步骤；关闭时无日志或包装。"""
    model = getattr(trainer.model, 'module', trainer.model)
    head = getattr(model, 'occ_head', None)
    if head is None or not getattr(head, 'b_collect_root_diagnostics', False):
        return
    if getattr(trainer, 'b_root_optimization_tracking', False):
        return
    trainer.b_root_optimization_tracking = True
    path = Path(trainer.checkpoint_dir).parent / f'root_optimization_rank{trainer.rank}.jsonl'
    if head.root_score is None:
        AppendJsonLine(str(path), {'status': 'disabled', 'reason': 'root classifier absent',
                                  'rank': trainer.rank, 'global_step': trainer.global_step,
                                  'b_root_loss_active': False, 'b_root_filter_active': False})
        return
    original_train_iter = trainer.train_iter
    original_optimize = trainer.optimize_step
    gradient_record = {}
    observed_optimize = InstrumentOptimizeStep(trainer, head, gradient_record)
    AppendJsonLine(str(path), {'status': 'enabled', 'rank': trainer.rank,
                              'sample_every_updates': 10, 'global_step': trainer.global_step,
                              'scope': 'DDP-reduced gradients before/after original global clip; root257 parameters only; post-step task logs',
                              'root_input_normalization': head.root_input_normalization})

    def TrainIterWithRootStatistics(batch):
        b_sample = (trainer.global_step + 1) % 10 == 0
        if not b_sample:
            return original_train_iter(batch)
        trainer.optimize_step = observed_optimize
        try:
            result = original_train_iter(batch)
        finally:
            trainer.optimize_step = original_optimize
        _, logs = result
        occ_logs = TaskLogMap(logs)['occ']
        task_weight = float(occ_logs['loss_weight'])
        AppendJsonLine(str(path), dict(
            gradient_record, status='sample', rank=trainer.rank,
            epoch=int(trainer.epoch), global_step=int(trainer.global_step),
            root_loss_weight=float(head.root_loss_weight), occ_task_weight=task_weight,
            effective_root_bce_weight=task_weight * float(head.root_loss_weight),
            root_loss=float(occ_logs['root_loss']),
            root_count=int(occ_logs['root_count']),
            occupied_root_count=int(occ_logs['occupied_root_count']),
            occupied_root_kept=int(occ_logs['occupied_root_kept'])))
        return result

    trainer.train_iter = TrainIterWithRootStatistics


def RunRootDiagnostic(trainer, cfg, args):
    options = cfg.root_diagnostic
    arm = str(options['arm'])
    steps = int(options.get('max_steps', 100))
    if arm not in ARMS or not 1 <= steps <= 400:
        raise ValueError('Root diagnostic requires a registered arm and 1..400 steps')
    if not args.resume or args.validate_only or args.init_checkpoint or args.stop_after_epoch is not None:
        raise ValueError('Root diagnostic requires --resume only; no validate-only/init/epoch-stop options')
    if cfg.get('detection_screen'):
        raise ValueError('Root diagnostic and screening runner are mutually exclusive')
    if trainer.epoch != 20 or trainer.global_step != 20 * len(trainer.train_dataloader):
        raise ValueError('Root diagnostic must begin at the complete epoch20 boundary')
    if steps >= len(trainer.train_dataloader):
        raise ValueError('Diagnostic must stop before completing the next epoch')
    model = getattr(trainer.model, 'module', trainer.model)
    head = model.occ_head
    normalization, b_reset = ARMS[arm]
    if head.root_score is None or head.root_input_normalization != normalization:
        raise ValueError('Diagnostic arm and actual root classifier/input normalization differ')
    if not head.b_collect_root_diagnostics or head.delete_threshold != .5:
        raise ValueError('Enable root group diagnostics and preserve deletion threshold .5')
    if trainer.b_freeze_x3_encoder:
        raise ValueError('This diagnostic expects the original F model training strategy')
    source_epoch, source_iter = trainer.epoch, trainer.global_step
    source_scheduler_step = trainer.lr_scheduler.last_epoch
    if source_scheduler_step != source_iter or trainer.lr_scheduler.total_steps != 48 * len(trainer.train_dataloader):
        raise ValueError('Diagnostic scheduler must retain original48 budget and epoch20 progress')
    output = Path(trainer.checkpoint_dir).parent
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f'root_diagnostic_rank{trainer.rank}.jsonl'
    if log_path.exists() or (output / 'root_diagnostic_result.json').exists():
        raise FileExistsError('Do not overwrite or silently restart an existing diagnostic')
    protocol = {'arm': arm, 'max_steps': steps, 'source_checkpoint': str(Path(args.resume).resolve()),
                'source_sha256': HashFile(args.resume), 'source_epoch': source_epoch, 'source_iter': source_iter,
                'world_size': trainer.world_size, 'normalization': normalization, 'b_reset_root_head': b_reset,
                'scheduler_total_steps': trainer.lr_scheduler.total_steps, 'scheduler_start_step': source_scheduler_step,
                'root_sha_before_reset': CheckRootAgreement(head),
                'b_partial_epoch': True, 'b_formal_training_epoch_completed': False,
                'scope': 'Real DDP trainer.train_iter updates; full pool epoch21 selection, finite prefix only; no validation claim'}
    if b_reset:
        protocol['root_reset'] = ResetRootHead(head, trainer.optimizer)
    protocol['root_sha_initial'] = CheckRootAgreement(head)
    if trainer.rank == 0:
        SaveJson(str(output / 'root_diagnostic_protocol.json'), protocol)
    trainer.model.train()
    PrepareEpochSampling(trainer.train_dataloader, source_epoch, trainer.rank, trainer.checkpoint_dir)
    original_optimize = trainer.optimize_step
    gradient_record = {}
    trainer.optimize_step = InstrumentOptimizeStep(trainer, head, gradient_record)
    started = time.monotonic()
    completed = 0
    b_success = False
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for index, batch in zip(range(steps), trainer.train_dataloader):
            trainer.inner_iter = index
            loss, logs = trainer.train_iter(batch)
            completed += 1
            tasks = TaskLogMap(logs)
            occ = tasks['occ']
            row = dict(gradient_record, diagnostic_step=completed, global_step=trainer.global_step,
                       source_epoch=source_epoch, total_loss=loss, elapsed_seconds=time.monotonic() - started,
                       tokens=batch['frames'][-1]['token'],
                       root_logs={key: float(value.detach()) if torch.is_tensor(value) else value
                                  for key, value in occ.items() if key.startswith('root_') or key in
                                  ('occupied_root_count', 'occupied_root_kept', 'kept_root_count', 'deleted_root_count', 'class_loss')},
                       task_weights={task: float(item['loss_weight']) for task, item in tasks.items()},
                       task_raw_losses={task: float(item['loss']) for task, item in tasks.items()})
            AppendJsonLine(str(log_path), row)
        if completed != steps or trainer.epoch != source_epoch or trainer.global_step != source_iter + steps:
            raise RuntimeError('Diagnostic update budget or unchanged epoch invariant failed')
        final_hashes = CheckRootAgreement(head)
        b_success = True
    finally:
        trainer.optimize_step = original_optimize
        result = {'b_complete': b_success, 'arm': arm, 'completed_steps': completed,
                  'source_epoch': source_epoch, 'epoch': trainer.epoch, 'iter': trainer.global_step,
                  'scheduler_step': trainer.lr_scheduler.last_epoch,
                  'elapsed_seconds': time.monotonic() - started, 'b_partial_epoch': True,
                  'b_full_validation_performed': False}
        if device.type == 'cuda':
            result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        SaveJson(str(output / f'root_diagnostic_status_rank{trainer.rank}.json'), result)
        if trainer.rank == 0:
            checkpoint = Path(trainer.checkpoint_dir) / f'diagnostic_after_{completed:03d}_steps.pth'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            meta = BuildCheckpointMeta(trainer.model, source_epoch, trainer.global_step)
            meta.update(diagnostic_steps=completed, diagnostic_source_epoch=source_epoch,
                        diagnostic_source_iter=source_iter, diagnostic_arm=arm, b_partial_epoch=True,
                        epoch_sampling=SamplingDefinition(trainer.train_dataloader))
            save_checkpoint(trainer.model, str(checkpoint), trainer.optimizer, trainer.lr_scheduler, meta)
            result.update(checkpoint=str(checkpoint), checkpoint_sha256=HashFile(checkpoint), protocol=protocol)
            if b_success:
                result['root_sha_final_all_ranks'] = final_hashes
            SaveJson(str(output / 'root_diagnostic_result.json'), result)
