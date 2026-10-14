import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from ..utils.checkpoint import LoadCompatibleModelWeights, load_checkpoint, save_checkpoint
from ..utils.progressbar import ProgressBar

def example_to_device(example, device, non_blocking=False):
    example_torch = {}
    for k, v in example.items():
        if k in ['token']:
            example_torch[k] = v
        elif isinstance(v, list):
            example_torch[k] = [
                res.to(device, non_blocking=non_blocking) for res in v]
        else:
            example_torch[k] = v.to(device, non_blocking=non_blocking)

    return example_torch


def GetModelModule(model):
    if hasattr(model, 'module'):
        return model.module
    return model


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


class Trainer(object):
    def __init__(self, model, train_dataloader=None,  val_dataloader=None,
                 optimizer=None, lr_scheduler=None, clip_grad_val=0.0, max_epochs=0,
                 eval_every_nepochs=1, eval_epochs=None, logger=None, log_every_niters=50, tb_log = None,
                 checkpoint_dir='./checkpoints', init_from_checkpoint=None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.log_every_niters = log_every_niters
        self.checkpoint_dir = checkpoint_dir
        self.init_from_checkpoint_path = init_from_checkpoint

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

    def LoadCompatibleCheckpoint(self, filename, map_location=torch.device("cpu")):
        self.logger.info("initialize model weights from %s", filename)
        checkpoint, load_info = LoadCompatibleModelWeights(
            self.model,
            filename,
            map_location=map_location,
        )
        self.epoch = 0
        self.global_step = 0

        self.logger.info(
            'initialized model weights: matched=%d | target_uninitialized=%d | unexpected=%d | '
            'shape_mismatch=%d | expanded_input=%d',
            load_info["matched"],
            load_info["target_uninitialized"],
            load_info["unexpected"],
            load_info["shape_mismatch"],
            len(load_info["expanded_input_keys"]),
        )
        if len(load_info["expanded_input_keys"]) > 0:
            self.logger.info('expanded input-channel keys: %s', '; '.join(load_info["expanded_input_keys"]))
        if len(load_info["skipped_shape_keys"]) > 0:
            preview_items = [
                f'{key}: source{source_shape}->target{target_shape}'
                for key, source_shape, target_shape in load_info["skipped_shape_keys"][:10]
            ]
            self.logger.info('shape-mismatch keys preview: %s', '; '.join(preview_items))
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
        loss, logs = self.model(data_batch)
        self.optimize_step(loss)
        loss_value = float(loss.detach().cpu())

        # if (self.inner_iter + 1) % self.log_every_niters == 0:
        #     log_str = "Epoch [{}/{}][{}/{}]\tlr: {:.5f}, ".format(
        #         self.epoch + 1,
        #         self.max_epochs,
        #         self.inner_iter + 1,
        #         len(self.train_dataloader),
        #         self.current_lr[0])
        #     self.logger.info(log_str)
        #     self.logger.info(self._convert_to_str(logs))

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
        if self.rank == 0:
            self.save_checkpoint()

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
        class_ce_loss_sum = torch.zeros(
            (num_classes,),
            dtype=torch.float64,
            device=device,
        )
        class_ce_count = torch.zeros(
            (num_classes,),
            dtype=torch.float64,
            device=device,
        )

        if self.rank == 0:
            prog_bar = ProgressBar(len(self.val_dataloader))

        for i, data_batch in enumerate(self.val_dataloader):
            self.inner_iter = i
            data_batch = example_to_device(data_batch, device, non_blocking=False)

            forward_dict = model_module.ForwardNetwork(data_batch)
            val_loss, _ = model_module.ComputeLoss(data_batch, forward_dict)
            point_logits = torch.cat(forward_dict['point_logits_list'], dim=0)  # [sum_N, num_classes]
            b_valid_mask = forward_dict['b_valid_mask']
            point_label = data_batch['semantic_label'][b_valid_mask].to(point_logits.device).long()  # [sum_N]
            point_pred = point_logits.argmax(dim=1)  # [sum_N]
            point_ce_loss = F.cross_entropy(
                point_logits,
                point_label,
                ignore_index=ignore_index,
                reduction='none',
            )
            b_ce_valid = (point_label >= 0) & (point_label < num_classes)
            if ignore_index is not None:
                b_ce_valid &= point_label != ignore_index
            if torch.any(b_ce_valid):
                valid_ce_label = point_label[b_ce_valid]
                valid_ce_loss = point_ce_loss[b_ce_valid].to(torch.float64)
                class_ce_loss_sum += torch.bincount(
                    valid_ce_label,
                    weights=valid_ce_loss,
                    minlength=num_classes,
                ).to(torch.float64)
                class_ce_count += torch.bincount(
                    valid_ce_label,
                    minlength=num_classes,
                ).to(torch.float64)

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
            dist.all_reduce(class_ce_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_ce_count, op=dist.ReduceOp.SUM)

        if self.rank != 0:
            return

        if val_batch_count.item() == 0:
            self.logger.info('Validation dataloader is empty, skip validation metrics.')
            return

        mean_val_loss = (val_loss_sum / val_batch_count).item()
        class_ce_mean = class_ce_loss_sum / torch.clamp(class_ce_count, min=1.0)
        total_ce_loss_sum = class_ce_loss_sum.sum()
        metric_dict = ComputeSegmentationMetrics(
            confusion_matrix=confusion_matrix,
            class_names=class_names,
            ignore_index=ignore_index,
        )

        seg_message_lines = [
            f'Validation Epoch {self.epoch} Segmentation',
            f'loss    : {mean_val_loss:.6f}',
            f"mIoU    : {metric_dict['miou'] * 100.0:.2f}",
            f"mAcc    : {metric_dict['macc'] * 100.0:.2f}",
            f"allAcc  : {metric_dict['all_acc'] * 100.0:.2f}",
            '| Class                | IoU (%) |',
            '|----------------------|---------|',
        ]
        for class_result in metric_dict['class_result_list']:
            if class_result['b_valid_iou']:
                seg_message_lines.append(
                    f"| {class_result['class_name']:<20} | {class_result['iou'] * 100.0:7.2f} |"
                )
        seg_message = '\n'.join(seg_message_lines)
        self.logger.info(seg_message)

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
            for class_result in metric_dict['class_result_list']:
                class_idx = int(class_result['class_idx'])
                if class_ce_count[class_idx].item() <= 0:
                    continue
                self.tb_log.add_scalar(
                    'val_ce_loss_mean/' + class_result['class_name'],
                    class_ce_mean[class_idx].item(),
                    self.epoch,
                )
                self.tb_log.add_scalar(
                    'val_ce_loss_sum/' + class_result['class_name'],
                    class_ce_loss_sum[class_idx].item(),
                    self.epoch,
                )
                if total_ce_loss_sum.item() > 0:
                    self.tb_log.add_scalar(
                        'val_ce_loss_ratio/' + class_result['class_name'],
                        (class_ce_loss_sum[class_idx] / total_ce_loss_sum).item(),
                        self.epoch,
                    )
                self.tb_log.add_scalar(
                    'val_ce_count/' + class_result['class_name'],
                    class_ce_count[class_idx].item(),
                    self.epoch,
                )

    def fit(self):
        self.logger.info("max: %d epochs", self.max_epochs)
        while self.epoch < self.max_epochs:
            self.train_epoch()
            if (self.eval_every_nepochs > 0 and self.epoch % self.eval_every_nepochs == 0) or \
               (self.eval_epochs is not None and self.epoch in self.eval_epochs):
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
