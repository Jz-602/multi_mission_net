import torch
from ..utils.checkpoint import load_checkpoint, save_checkpoint
import torch.distributed as dist
from .multitask_validator import ValidateMultiTask

from tqdm import tqdm


import os

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


class Trainer(object):
    def __init__(self, model, train_dataloader=None,  val_dataloader=None,
                 optimizer=None, lr_scheduler=None, clip_grad_val=0.0, max_epochs=0,
                 eval_every_nepochs=1, eval_epochs=None, logger=None, log_every_niters=50, tb_log = None):
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

    def save_checkpoint(self, filename_tmpl="epoch_{}.pth", save_optimizer=True):
        meta = dict(epoch=self.epoch, iter=self.global_step)
        filepath = filename_tmpl.format(self.epoch)
        optimizer = self.optimizer if save_optimizer else None
        scheduler = self.lr_scheduler if save_optimizer else None

        os.makedirs('./checkpoints/', exist_ok=True)
        filename = './checkpoints/' + filepath
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

        # update global step
        self.global_step += 1

        if self.tb_log is not None:
            ## 训练阶段只保留核心主指标、任务权重和分割子 loss，避免 TensorBoard 里重复项过多。
            self.tb_log.add_scalar('train/loss', loss, self.global_step)
            current_lr_list = self.current_lr
            self.tb_log.add_scalar('train/lr', current_lr_list[0], self.global_step)

            seg_log = [log_item for log_item in logs if log_item['task'][0] == 'seg'][0]
            det_logs = [log_item for log_item in logs if log_item['task'][0] != 'seg']

            seg_total_loss = seg_log['loss']
            det_total_loss = sum([det_log['loss'] for det_log in det_logs])
            det_hm_loss = sum([det_log['hm_loss'] for det_log in det_logs])
            det_loc_loss = sum([det_log['loc_loss'] for det_log in det_logs])

            self.tb_log.add_scalar('train/seg_loss', seg_total_loss, self.global_step)
            self.tb_log.add_scalar('train/det_loss', det_total_loss, self.global_step)
            self.tb_log.add_scalar('train/det_hm_loss', det_hm_loss, self.global_step)
            self.tb_log.add_scalar('train/det_loc_loss', det_loc_loss, self.global_step)

            ## 保留多任务权重，便于观察训练过程中分割与检测任务的相对占比变化。
            if 'log_var' in seg_log:
                self.tb_log.add_scalar('train/seg_loss_weight', seg_log['loss_weight'], self.global_step)
                self.tb_log.add_scalar('train/seg_weighted_loss', seg_log['weighted_loss'], self.global_step)

            det_weight_logs = [det_log for det_log in det_logs if 'log_var' in det_log]
            if len(det_weight_logs) > 0:
                det_weight_log = det_weight_logs[0]
                self.tb_log.add_scalar('train/det_loss_weight', det_weight_log['loss_weight'], self.global_step)
                self.tb_log.add_scalar('train/det_weighted_loss', det_weight_log['weighted_loss'], self.global_step)

            ## 保留分割子 loss，方便单独观察 Dice / CE / Lovasz 的相对变化。
            for key in ['DiceLossV1', 'CELoss', 'LovLoss']:
                if key in seg_log:
                    self.tb_log.add_scalar(f'train/seg{key}', seg_log[key], self.global_step)

        return loss_value, logs

    def train_epoch(self):
        self.model.train()
        train_dataset = self.train_dataloader.dataset
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
            # if i >= 2:
            #     break
            self.inner_iter = i  # iteration in current epoch
            loss_value, logs = self.train_iter(data_batch)
            if self.rank == 0:
                progress_bar.set_postfix_str(self._build_train_tqdm_str(loss_value, logs))

        self.epoch += 1
        if self.rank == 0:
            self.save_checkpoint()

    @torch.no_grad()
    def val_epoch(self):
        metric_dict = ValidateMultiTask(
            model=self.model,
            val_dataloader=self.val_dataloader,
            device=self.device,
            rank=self.rank,
            world_size=self.world_size,
            epoch=self.epoch,
            logger=self.logger,
        )

        ## 记录验证指标到 TensorBoard，便于直接观察分割 mIoU 与检测 mAP 曲线。
        if self.rank == 0 and self.tb_log is not None and metric_dict is not None:
            seg_metric_dict = metric_dict['seg']
            det_metric_dict = metric_dict['det']

            ## 分割整体指标
            self.tb_log.add_scalar('val/seg_miou', seg_metric_dict['miou'], self.epoch)
            self.tb_log.add_scalar('val/seg_macc', seg_metric_dict['macc'], self.epoch)
            self.tb_log.add_scalar('val/seg_all_acc', seg_metric_dict['all_acc'], self.epoch)

            ## 分割分类别指标
            for class_result in seg_metric_dict['class_result_list']:
                class_name = class_result['class_name']
                if class_result['b_valid_iou']:
                    self.tb_log.add_scalar(
                        f'val_seg_class_iou/{class_name}',
                        class_result['iou'],
                        self.epoch,
                    )
                if class_result['b_valid_acc']:
                    self.tb_log.add_scalar(
                        f'val_seg_class_acc/{class_name}',
                        class_result['acc'],
                        self.epoch,
                    )

            ## 检测整体与分类别指标
            for key, val in det_metric_dict.items():
                self.tb_log.add_scalar(f'val_det/{key}', val, self.epoch)

        return metric_dict

    def fit(self):
        # self.val_epoch()
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

    def _build_train_tqdm_str(self, total_loss, logs):
        seg_log = [log_item for log_item in logs if log_item['task'][0] == 'seg'][0]
        det_logs = [log_item for log_item in logs if log_item['task'][0] != 'seg']

        def _ToFloat(val):
            if isinstance(val, torch.Tensor):
                return float(val.detach().cpu())
            return float(val)

        raw_seg_loss = _ToFloat(seg_log['loss'])
        seg_display_loss = _ToFloat(seg_log.get('weighted_loss', seg_log['loss']))
        seg_loss_items = [f'raw_seg_loss={raw_seg_loss:.3f}']
        if 'loss_weight' in seg_log:
            seg_loss_items.append(f'seg_loss_weight={_ToFloat(seg_log["loss_weight"]):.3f}')
        for key, val in seg_log.items():
            if key in ['task', 'loss', 'weighted_loss', 'log_var', 'loss_weight']:
                continue
            seg_loss_items.append(f'{key}={_ToFloat(val):.3f}')

        raw_det_loss = 0.0
        hm_loss = 0.0
        loc_loss = 0.0
        iou_loss = 0.0
        iou_reg_loss = 0.0
        det_weight_log = None
        for det_log in det_logs:
            det_task_loss = _ToFloat(det_log['loss'])
            raw_det_loss += det_task_loss
            hm_loss += _ToFloat(det_log['hm_loss'])
            loc_loss += _ToFloat(det_log['loc_loss'])
            if 'iou_loss' in det_log:
                iou_loss += _ToFloat(det_log['iou_loss'])
            if 'iou_reg_loss' in det_log:
                iou_reg_loss += _ToFloat(det_log['iou_reg_loss'])
            if det_weight_log is None and 'weighted_loss' in det_log:
                det_weight_log = det_log

        det_display_loss = raw_det_loss
        det_loss_items = [f'raw_det_loss={raw_det_loss:.3f}']
        if det_weight_log is not None:
            det_display_loss = _ToFloat(det_weight_log['weighted_loss'])
            det_loss_items.append(f'det_loss_weight={_ToFloat(det_weight_log["loss_weight"]):.3f}')
        det_loss_items.extend([f'hm_loss={hm_loss:.3f}', f'loc_loss={loc_loss:.3f}'])
        if iou_loss > 0:
            det_loss_items.append(f'iou_loss={iou_loss:.3f}')
        if iou_reg_loss > 0:
            det_loss_items.append(f'iou_reg_loss={iou_reg_loss:.3f}')
        return (
            f"lr={self.current_lr[0]:.5f} | total_loss={total_loss:.4f} | "
            f"seg_loss={seg_display_loss:.4f}:[{','.join(seg_loss_items)}] | "
            f"det_loss={det_display_loss:.4f}:[{','.join(det_loss_items)}]"
        )
