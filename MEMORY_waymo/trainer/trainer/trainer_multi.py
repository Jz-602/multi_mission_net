import os
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.progressbar import ProgressBar

def example_to_device(example, device, non_blocking=False):
    ## 多帧 batch 是 dict/list/dict 的嵌套结构，这里递归搬运 tensor，字符串和普通数值保持原样。
    if isinstance(example, torch.Tensor):
        return example.to(device, non_blocking=non_blocking)
    if isinstance(example, dict):
        return {
            key: example_to_device(value, device, non_blocking=non_blocking)
            for key, value in example.items()
        }
    if isinstance(example, list):
        return [
            example_to_device(value, device, non_blocking=non_blocking)
            for value in example
        ]
    if isinstance(example, tuple):
        return tuple(
            example_to_device(value, device, non_blocking=non_blocking)
            for value in example
        )
    return example


def DetachMemory(memory):
    ## 每个时间步只把 memory state 作为下一帧输入，不让梯度无限穿过整段序列。
    if memory is None:
        return None
    if isinstance(memory, torch.Tensor):
        return memory.detach()
    if isinstance(memory, dict):
        return {key: DetachMemory(value) for key, value in memory.items()}
    if isinstance(memory, list):
        return [DetachMemory(value) for value in memory]
    if isinstance(memory, tuple):
        return tuple(DetachMemory(value) for value in memory)
    return memory


def GetModelModule(model):
    if hasattr(model, 'module'):
        return model.module
    return model


X3_ENCODER_MODULE_NAMES = (
    'conv_input',
    'encoder_stage1',
    'encoder_stage2',
    'encoder_stage3',
)


def FreezeX3Encoder(model, logger=None):
    model_module = GetModelModule(model)
    frozen_param_count = 0
    newly_frozen_param_count = 0
    frozen_module_names = []
    missing_module_names = []

    for module_name in X3_ENCODER_MODULE_NAMES:
        if not hasattr(model_module, module_name):
            missing_module_names.append(module_name)
            continue

        frozen_module_names.append(module_name)
        module = getattr(model_module, module_name)
        for param in module.parameters():
            param_count = param.numel()
            frozen_param_count += param_count
            if param.requires_grad:
                newly_frozen_param_count += param_count
            param.requires_grad = False

    if logger is not None:
        logger.info(
            'freeze x3 encoder modules: %s | frozen params: %d | newly frozen params: %d',
            ', '.join(frozen_module_names) if len(frozen_module_names) > 0 else 'none',
            frozen_param_count,
            newly_frozen_param_count,
        )
        if len(missing_module_names) > 0:
            logger.info('missing x3 encoder modules when freezing: %s', ', '.join(missing_module_names))

    return frozen_module_names, frozen_param_count


def SetFrozenX3EncoderEval(model):
    model_module = GetModelModule(model)
    for module_name in X3_ENCODER_MODULE_NAMES:
        if hasattr(model_module, module_name):
            getattr(model_module, module_name).eval()


def ExtractCheckpointStateDict(checkpoint, filename):
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise RuntimeError("No state_dict found in checkpoint file {}".format(filename))

    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    return state_dict


def BuildSegmentationConfusionMatrix(point_pred, point_label, num_classes, ignore_index):
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.float64,
        device=point_pred.device,
    )

    b_valid = (point_label >= 0) & (point_label < num_classes)
    if ignore_index is not None:
        b_valid &= (point_label != ignore_index)

    if not torch.any(b_valid):
        return confusion_matrix

    point_pred = point_pred[b_valid].to(torch.long)
    point_label = point_label[b_valid].to(torch.long)
    linear_indices = point_label * num_classes + point_pred
    confusion_matrix += torch.bincount(
        linear_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return confusion_matrix


def ComputeSegmentationMetrics(confusion_matrix, class_names, ignore_index):
    num_classes = confusion_matrix.shape[0]
    if class_names is None:
        class_names = []
    class_names = list(class_names)
    if len(class_names) < num_classes:
        class_names.extend(
            [f'class_{class_idx:02d}' for class_idx in range(len(class_names), num_classes)]
        )

    true_positive = torch.diag(confusion_matrix)
    gt_count = confusion_matrix.sum(dim=1)
    pred_count = confusion_matrix.sum(dim=0)
    union = gt_count + pred_count - true_positive

    class_iou = torch.zeros(num_classes, dtype=torch.float64, device=confusion_matrix.device)
    class_acc = torch.zeros(num_classes, dtype=torch.float64, device=confusion_matrix.device)

    b_valid_iou = union > 0
    b_valid_acc = gt_count > 0
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        b_valid_iou[ignore_index] = False
        b_valid_acc[ignore_index] = False

    class_iou[b_valid_iou] = true_positive[b_valid_iou] / union[b_valid_iou]
    class_acc[b_valid_acc] = true_positive[b_valid_acc] / gt_count[b_valid_acc]

    miou = class_iou[b_valid_iou].mean().item() if torch.any(b_valid_iou) else 0.0
    macc = class_acc[b_valid_acc].mean().item() if torch.any(b_valid_acc) else 0.0
    all_acc_denominator = gt_count.sum().item()
    all_acc = true_positive.sum().item() / all_acc_denominator if all_acc_denominator > 0 else 0.0

    class_result_list = []
    for class_idx, class_name in enumerate(class_names[:num_classes]):
        class_result_list.append(
            {
                'class_idx': class_idx,
                'class_name': class_name,
                'iou': class_iou[class_idx].item(),
                'acc': class_acc[class_idx].item(),
                'support': int(gt_count[class_idx].item()),
                'b_valid_iou': bool(b_valid_iou[class_idx].item()),
                'b_valid_acc': bool(b_valid_acc[class_idx].item()),
            }
        )

    return {
        'miou': miou,
        'macc': macc,
        'all_acc': all_acc,
        'class_result_list': class_result_list,
    }


def FormatSegmentationMetrics(epoch, metric_dict):
    message_lines = [
        f"Validation Epoch {epoch} Segmentation",
        (
            f"mIoU: {metric_dict['miou'] * 100.0:.2f} | "
            f"mAcc: {metric_dict['macc'] * 100.0:.2f} | "
            f"allAcc: {metric_dict['all_acc'] * 100.0:.2f}"
        ),
        "| Class                | IoU (%) |",
        "|----------------------|---------|",
    ]
    for class_result in metric_dict['class_result_list']:
        if class_result['b_valid_iou']:
            message_lines.append(
                f"| {class_result['class_name']:<20} | {class_result['iou'] * 100.0:7.2f} |"
            )
    return '\n'.join(message_lines)


class Trainer(object):
    def __init__(self, model, train_dataloader=None,  val_dataloader=None,
                 optimizer=None, lr_scheduler=None, clip_grad_val=0.0, max_epochs=0,
                 eval_every_nepochs=1, eval_epochs=None, logger=None, log_every_niters=50, tb_log = None,
                 freeze_x3_encoder=False, checkpoint_dir="./checkpoints_multi"):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_every_niters = log_every_niters

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.epoch = 0
        self.global_step = 0
        self.inner_iter = 0
        self.max_epochs = max_epochs

        self.clip_grad_val = clip_grad_val
        self.eval_every_nepochs = eval_every_nepochs
        self.eval_epochs = eval_epochs

        self.logger = logger
        self.tb_log = tb_log
        self.b_resume_from_single_model_loaded = False
        self.checkpoint_dir = checkpoint_dir
        self.b_freeze_x3_encoder = bool(freeze_x3_encoder)
        if self.b_freeze_x3_encoder:
            FreezeX3Encoder(self.model, logger=self.logger)

    @property
    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError(
                "lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    @property
    def device(self):
        """
        :return: an instance of `torch.device`
        """
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        else:
            return torch.device('cpu')

    def load_checkpoint(self, filename, map_location="cpu", strict=False):
        self.logger.info("load checkpoint from %s", filename)
        return load_checkpoint(self.model, filename, map_location, strict)

    def resume_from_single_model(self, filename, map_location=torch.device("cpu")):
        if not os.path.isfile(filename):
            raise IOError("{} is not a checkpoint file".format(filename))

        self.logger.info("load single-frame model weights from %s", filename)
        try:
            checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filename, map_location=map_location)

        source_state_dict = ExtractCheckpointStateDict(checkpoint, filename)
        model_module = GetModelModule(self.model)
        target_state_dict = model_module.state_dict()

        matched_state_dict = {}
        skipped_unexpected_keys = []
        skipped_shape_keys = []
        for key, value in source_state_dict.items():
            if key not in target_state_dict:
                skipped_unexpected_keys.append(key)
                continue
            if target_state_dict[key].shape != value.shape:
                skipped_shape_keys.append((key, tuple(value.shape), tuple(target_state_dict[key].shape)))
                continue
            matched_state_dict[key] = value

        if len(matched_state_dict) == 0:
            raise RuntimeError(
                "No matching parameters were loaded from single-frame checkpoint {}".format(filename)
            )

        target_state_dict.update(matched_state_dict)
        model_module.load_state_dict(target_state_dict, strict=True)
        self.epoch = 0
        self.global_step = 0
        self.b_resume_from_single_model_loaded = True

        missing_target_keys = [
            key for key in target_state_dict.keys()
            if key not in matched_state_dict
        ]
        self.logger.info(
            'loaded single-frame weights: matched=%d | target_uninitialized=%d | unexpected=%d | shape_mismatch=%d',
            len(matched_state_dict),
            len(missing_target_keys),
            len(skipped_unexpected_keys),
            len(skipped_shape_keys),
        )
        if len(skipped_shape_keys) > 0:
            preview_items = [
                f'{key}: source{source_shape}->target{target_shape}'
                for key, source_shape, target_shape in skipped_shape_keys[:10]
            ]
            self.logger.info('single-frame shape-mismatch keys preview: %s', '; '.join(preview_items))
        return checkpoint

    def save_checkpoint(self, filename_tmpl="epoch_{}.pth", save_optimizer=True):
        meta = dict(epoch=self.epoch, iter=self.global_step)
        filepath = filename_tmpl.format(self.epoch)
        optimizer = self.optimizer if save_optimizer else None
        scheduler = self.lr_scheduler if save_optimizer else None

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = os.path.join(self.checkpoint_dir, filepath)
        save_checkpoint(self.model, filename, optimizer=optimizer,
                        scheduler=scheduler, meta=meta)

    def resume(self, checkpoint, resume_optimizer=True, map_location=torch.device("cpu")):
        checkpoint = self.load_checkpoint(
            checkpoint, map_location=map_location, strict=True)

        self.epoch = checkpoint['meta']['epoch']
        self.global_step = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and resume_optimizer:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])

        self.logger.info("resumed epoch %d, iter %d",
                         self.epoch, self.global_step)

    def optimize_step(self, loss):
        """
        Performs a single optimization step and updates the parameters which have been passed to ``self.optimizer``.
        """
        # optimization
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_val)

        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def train_iter(self, data_batch):
        data_batch = example_to_device(
            data_batch, self.device, non_blocking=False)
        frames = data_batch["frames"]
        seq_len = len(frames)

        ## train_start / train_end 决定哪些时间步只 warmup memory，哪些时间步参与 loss。
        warmup_len = int(data_batch.get("warmup_len", 0))
        train_len = int(data_batch.get("train_len", max(1, seq_len - warmup_len)))
        train_start = min(warmup_len, seq_len)
        train_end = min(seq_len, train_start + train_len)

        memory = None
        loss = None
        logs = {}
        train_step_count = 0

        ## warmup 阶段只递推 memory，不反传；此阶段用于让当前 clip 的历史帧进入 memory 状态。
        for frame_idx in range(train_start):
            with torch.no_grad():
                memory_next = self.model(
                    frames[frame_idx],
                    memory=memory,
                    b_memory_only=True,
                )
                memory = DetachMemory(memory_next)

        ## train 阶段继续沿用 warmup 得到的 memory，并对后续帧累计 segmentation loss。
        for frame_idx in range(train_start, train_end):
            loss_t, logs_t, memory_next = self.model(frames[frame_idx], memory=memory)
            loss = loss_t if loss is None else loss + loss_t
            logs = logs_t
            memory = DetachMemory(memory_next)
            train_step_count += 1

        ## 多个训练时间步时取平均，保持 loss 尺度不随 train_len 改变。
        loss = loss / max(1, train_step_count)
        self.optimize_step(loss)
        loss_value = float(loss.detach().cpu())

        # update global step
        self.global_step += 1

        if self.tb_log is not None:
            self.tb_log.add_scalar('train/loss', loss, self.global_step)
            current_lr_list = self.current_lr
            self.tb_log.add_scalar('train/lr', current_lr_list[0], self.global_step)
            if len(current_lr_list) > 1:
                for group_idx, lr in enumerate(current_lr_list):
                    self.tb_log.add_scalar(f'train/lr_group_{group_idx}', lr, self.global_step)
            log_item_list = logs if isinstance(logs, list) else [logs]
            for log_item in log_item_list:
                task_name = log_item.get('task', ['train'])[0]
                for key, val in log_item.items():
                    if key.endswith('loss') or key in ['DiceLossV1', 'CELoss', 'LovLoss']:
                        self.tb_log.add_scalar('train/' + task_name + key, val, self.global_step)

        return loss_value, logs

    def train_epoch(self):
        self.model.train()
        if self.b_freeze_x3_encoder:
            SetFrozenX3EncoderEval(self.model)
        train_dataset = self.train_dataloader.dataset
        sample_count = max(1, int(np.ceil(len(train_dataset.sample_source_infos) * train_dataset.sample_fraction)))
        rng = np.random.default_rng(train_dataset.sample_seed + self.epoch)
        sampled_indices = np.sort(
            rng.choice(len(train_dataset.sample_source_infos), size=sample_count, replace=False)
        )
        train_dataset.infos = [
            train_dataset.sample_source_infos[int(sample_idx)]
            for sample_idx in sampled_indices
        ]
        if hasattr(train_dataset, "SetCurrentEpoch"):
            train_dataset.SetCurrentEpoch(self.epoch, self.max_epochs)
        train_dataset.anchor_list = train_dataset.BuildAnchorList()
        if self.world_size > 1:
            self.train_dataloader.sampler.set_epoch(self.epoch)

        if self.rank == 0:
            progress_bar = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f'Train Epoch {self.epoch + 1}/{self.max_epochs}',
                dynamic_ncols=True,
            )
        else:
            progress_bar = enumerate(self.train_dataloader)

        for i, data_batch in progress_bar:
            self.inner_iter = i  # iteration in current epoch
            loss_value, logs = self.train_iter(data_batch)
            if self.rank == 0:
                postfix_dict = {
                    'loss': f'{loss_value:.4f}',
                    'lr': f'{self.current_lr[0]:.5f}',
                }
                log_item_list = logs if isinstance(logs, list) else [logs]
                seg_log = None
                for log_item in log_item_list:
                    if log_item.get('task', [''])[0] == 'seg':
                        seg_log = log_item
                        break

                if seg_log is not None:
                    model_module = GetModelModule(self.model)
                    loss_type_list = list(getattr(model_module, 'loss_types', []))
                    for loss_name in loss_type_list:
                        if loss_name in seg_log:
                            loss_value_item = seg_log[loss_name]
                            if isinstance(loss_value_item, torch.Tensor):
                                loss_value_item = float(loss_value_item.detach().cpu())
                            postfix_dict[loss_name] = f'{float(loss_value_item):.4f}'

                progress_bar.set_postfix(postfix_dict)

        self.epoch += 1

    @torch.no_grad()
    def val_epoch(self):
        if self.val_dataloader is None:
            return

        self.model.eval()
        model_module = GetModelModule(self.model)
        model_config = getattr(model_module, 'config', None)
        if model_config is None:
            raise AttributeError('Segmentation validation requires model.config.')

        num_classes = int(model_config.num_classes)
        ignore_index = getattr(model_config, 'ignore_index', None)
        class_names = getattr(model_config, 'class_names', None)
        device = self.device

        confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.float64,
            device=device,
        )
        val_loss_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
        val_batch_count = torch.tensor(0, dtype=torch.long, device=device)

        if self.rank == 0:
            prog_bar = ProgressBar(len(self.val_dataloader))

        for i, data_batch in enumerate(self.val_dataloader):
            self.inner_iter = i
            data_batch = example_to_device(data_batch, device, non_blocking=False)

            frames = data_batch["frames"]
            seq_len = len(frames)

            ## 验证阶段同样先 warmup memory，只统计 warmup 之后 train_len 个时间步的分割指标。
            warmup_len = int(data_batch.get("warmup_len", 0))
            train_len = int(data_batch.get("train_len", max(1, seq_len - warmup_len)))
            eval_start = min(warmup_len, seq_len)
            eval_end = min(seq_len, eval_start + train_len)
            if eval_end <= eval_start:
                eval_start = max(0, seq_len - 1)
                eval_end = seq_len

            memory = None
            for frame_idx, frame_batch in enumerate(frames):
                if frame_idx >= eval_end:
                    break
                if frame_idx < eval_start:
                    memory_next = model_module(
                        frame_batch,
                        memory=memory,
                        b_memory_only=True,
                    )
                    memory = DetachMemory(memory_next)
                    continue

                ## b_return_details=True 时 model 返回 logits、label 计算所需中间结果和 memory_next。
                forward_dict = model_module(
                    frame_batch,
                    memory=memory,
                    b_return_details=True,
                )
                memory = DetachMemory(forward_dict.get("memory", None))

                ## 只对选定验证时间步计算 loss / confusion matrix，避免 warmup 帧重复进入验证指标。
                val_loss, _ = model_module.ComputeLoss(frame_batch, forward_dict)
                point_logits = torch.cat(forward_dict['point_logits_list'], dim=0)  # [sum_N, num_classes]
                b_valid_mask = forward_dict['b_valid_mask']
                point_label = frame_batch['semantic_label'][b_valid_mask].to(point_logits.device).long()  # [sum_N]
                point_pred = point_logits.argmax(dim=1)  # [sum_N]

                confusion_matrix += BuildSegmentationConfusionMatrix(
                    point_pred=point_pred,
                    point_label=point_label,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                )
                val_loss_sum += val_loss.detach().to(torch.float64)
                val_batch_count += 1
            if self.rank == 0:
                prog_bar.update()

        if self.world_size > 1:
            dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_batch_count, op=dist.ReduceOp.SUM)

        if self.rank != 0:
            return

        if val_batch_count.item() == 0:
            self.logger.info('Validation dataloader is empty, skip validation metrics.')
            return

        mean_val_loss = (val_loss_sum / val_batch_count).item()
        metric_dict = ComputeSegmentationMetrics(
            confusion_matrix=confusion_matrix,
            class_names=class_names,
            ignore_index=ignore_index,
        )

        self.logger.info(FormatSegmentationMetrics(self.epoch, metric_dict))

        if self.tb_log is not None:
            self.tb_log.add_scalar('val/loss', mean_val_loss, self.epoch)
            self.tb_log.add_scalar('val/mIoU', metric_dict['miou'], self.epoch)
            self.tb_log.add_scalar('val/mAcc', metric_dict['macc'], self.epoch)
            self.tb_log.add_scalar('val/allAcc', metric_dict['all_acc'], self.epoch)
            for class_result in metric_dict['class_result_list']:
                if class_result['b_valid_iou']:
                    self.tb_log.add_scalar(
                        'val_iou/' + class_result['class_name'],
                        class_result['iou'],
                        self.epoch,
                    )

    def fit(self):
        self.logger.info("max: %d epochs", self.max_epochs)
        while self.epoch < self.max_epochs:
            self.train_epoch()
            b_validate = (
                (self.eval_every_nepochs > 0 and self.epoch % self.eval_every_nepochs == 0)
                or (self.eval_epochs is not None and self.epoch in self.eval_epochs)
            )
            if b_validate:
                if self.rank == 0:
                    self.save_checkpoint()
                self.val_epoch()

    def _convert_to_str(self, log_dict):
        def _convert_to_precision4(val):
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().tolist()
            if isinstance(val, float):
                val = "{:.4f}".format(val)
            elif isinstance(val, list):
                val = [_convert_to_precision4(v) for v in val]

            return val

        def _convert_dict_to_str(log_vars):
            log_items = []
            for name, val in log_vars.items():
                log_items.append("{}: {}".format(
                    name, _convert_to_precision4(val)))

            log_str = ", ".join(log_items)

            return log_str

        if isinstance(log_dict, list):
            logs = [_convert_dict_to_str(log_vars) for log_vars in log_dict]
            log_str = '\n'.join(logs)
        else:
            log_str = _convert_dict_to_str(log_dict)

        log_str += "\n"

        return log_str
