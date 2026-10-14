from dynamic_sampler import PrepareEpochSampling, SamplingDefinition, ValidateSamplingResume
import os
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .multitask_validator import ValidateMultiTask
from .three_task_runtime import (
    AppendJsonLine,
    BuildCheckpointMeta,
    BuildProgressText,
    CheckFiniteGradients,
    CheckFiniteLoss,
    MoveToDevice,
    SaveJson,
    ScalarizeLogs,
    ValidateResumeCheckpoint,
    WriteTrainTensorboard,
    WriteValidationTensorboard,
)

def example_to_device(example, device, non_blocking=False):
    return MoveToDevice(example, device, non_blocking=non_blocking)


class Trainer(object):
    def __init__(self, model, train_dataloader=None,  val_dataloader=None,
                 optimizer=None, lr_scheduler=None, clip_grad_val=0.0, max_epochs=0,
                 eval_every_nepochs=1, eval_epochs=None, logger=None, log_every_niters=50, tb_log=None,
                 checkpoint_dir="./checkpoints"):
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
        self.checkpoint_dir = checkpoint_dir
        output_dir = Path(checkpoint_dir).resolve().parent
        self.train_metrics_filename = str(output_dir / "train_metrics.jsonl")
        self.validation_metrics_filename = str(output_dir / "validation_metrics.json")
        self.validation_metrics_history_filename = str(output_dir / "validation_metrics.jsonl")

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
        meta = BuildCheckpointMeta(self.model, self.epoch, self.global_step)
        sampling_definition = SamplingDefinition(self.train_dataloader)
        if sampling_definition is not None:
            meta["epoch_sampling"] = sampling_definition
        filepath = filename_tmpl.format(self.epoch)
        optimizer = self.optimizer if save_optimizer else None
        scheduler = self.lr_scheduler if save_optimizer else None

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = os.path.join(self.checkpoint_dir, filepath)
        save_checkpoint(self.model, filename, optimizer=optimizer,
                        scheduler=scheduler, meta=meta)

    def resume(self, checkpoint, resume_optimizer=True, map_location=torch.device("cpu")):
        validated_checkpoint = ValidateResumeCheckpoint(
            checkpoint,
            self.model,
            map_location=map_location,
            b_require_optimizer=bool(resume_optimizer and self.optimizer is not None),
            b_require_scheduler=bool(resume_optimizer and self.lr_scheduler is not None),
        )
        ValidateSamplingResume(self.train_dataloader, validated_checkpoint, self.lr_scheduler)
        checkpoint = self.load_checkpoint(
            checkpoint, map_location=map_location, strict=True)

        self.epoch = validated_checkpoint['meta']['epoch']
        self.global_step = validated_checkpoint['meta']['iter']
        if resume_optimizer and self.optimizer is not None:
            self.optimizer.load_state_dict(validated_checkpoint['optimizer'])
        if resume_optimizer and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(validated_checkpoint['scheduler'])

        self.logger.info("resumed epoch %d, iter %d",
                         self.epoch, self.global_step)

    def optimize_step(self, loss):
        """
        Performs a single optimization step and updates the parameters which have been passed to ``self.optimizer``.
        """
        CheckFiniteLoss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        CheckFiniteGradients(self.model)
        if self.clip_grad_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad_val)
            CheckFiniteGradients(self.model)

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

        WriteTrainTensorboard(
            self.tb_log,
            loss,
            logs,
            self.global_step,
            self.current_lr[0],
        )
        if self.rank == 0:
            AppendJsonLine(self.train_metrics_filename, {
                'epoch': int(self.epoch),
                'iter': int(self.global_step),
                'total_loss': loss_value,
                'learning_rate': float(self.current_lr[0]),
                'tasks': ScalarizeLogs(logs),
            })

        return loss_value, logs

    def train_epoch(self):
        self.model.train()
        train_dataset = self.train_dataloader.dataset
        if hasattr(train_dataset, "SetCurrentEpoch"):
            train_dataset.SetCurrentEpoch(self.epoch, self.max_epochs)
        PrepareEpochSampling(self.train_dataloader, self.epoch, self.rank, self.checkpoint_dir)

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

        if self.rank == 0 and metric_dict is not None:
            WriteValidationTensorboard(self.tb_log, metric_dict, self.epoch)
            validation_record = {'epoch': int(self.epoch), 'metrics': metric_dict}
            SaveJson(self.validation_metrics_filename, validation_record)
            AppendJsonLine(self.validation_metrics_history_filename, validation_record)

        return metric_dict

    def fit(self):
        # self.val_epoch()
        self.logger.info("max: %d epochs", self.max_epochs)
        while self.epoch < self.max_epochs:
            self.train_epoch()
            b_validate = (
                (self.eval_every_nepochs > 0 and self.epoch % self.eval_every_nepochs == 0)
                or (self.eval_epochs is not None and self.epoch in self.eval_epochs)
            )
            if self.rank == 0:
                self.save_checkpoint('latest.pth')
                if b_validate or self.epoch == self.max_epochs:
                    self.save_checkpoint()
            if b_validate:
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
        return BuildProgressText(total_loss, logs, self.current_lr[0])
