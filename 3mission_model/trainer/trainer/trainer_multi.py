from dynamic_sampler import PrepareEpochSampling, SamplingDefinition, ValidateSamplingResume
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from ..utils.checkpoint import load_checkpoint, save_checkpoint
from .multitask_validator import (
    AccumulateBatchRecords,
    FinalizeThreeTaskMetrics,
    GatherTokenRecords,
)
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


def DetachMemory(memory):
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
    if hasattr(model, "module"):
        return model.module
    return model


X3_ENCODER_MODULE_NAMES = (
    "conv_input",
    "encoder_stage1",
    "encoder_stage2",
    "encoder_stage3",
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
            "freeze x3 encoder modules: %s | frozen params: %d | newly frozen params: %d",
            ", ".join(frozen_module_names) if len(frozen_module_names) > 0 else "none",
            frozen_param_count,
            newly_frozen_param_count,
        )
        if len(missing_module_names) > 0:
            logger.info("missing x3 encoder modules when freezing: %s", ", ".join(missing_module_names))
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


@torch.no_grad()
def ValidateMultiTaskMemory(model, val_dataloader, device, rank, world_size, epoch, logger):
    if val_dataloader is None:
        return None
    model.eval()
    model_module = GetModelModule(model)
    # History bypasses DDP; align eval buffers before it builds the first memory.
    if world_size > 1:
        for buffer in model_module.buffers():
            dist.broadcast(buffer, src=0)
    terrain_criterion = getattr(model_module, "terrain_criterion", None)
    token_records = {}

    for data_batch in val_dataloader:
        data_batch = example_to_device(data_batch, device, non_blocking=False)
        frames = data_batch["frames"]
        warmup_len = min(int(data_batch.get("warmup_len", 0)), len(frames))
        train_len = int(data_batch.get("train_len", max(1, len(frames) - warmup_len)))
        train_end = min(len(frames), warmup_len + train_len)
        memory = None

        # Warmup bypasses DDP because it has no loss/backward counterpart.
        for frame_idx in range(warmup_len):
            memory = model_module(
                frames[frame_idx],
                memory=memory,
                b_memory_only=True,
            )
            memory = DetachMemory(memory)

        for frame_idx in range(warmup_len, train_end):
            batch_dict = frames[frame_idx]
            batch_result = model(batch_dict, memory=memory)
            AccumulateBatchRecords(
                token_records,
                batch_result,
                batch_dict,
                terrain_criterion,
            )

    token_records = GatherTokenRecords(token_records, rank, world_size)
    if rank != 0:
        return None
    return FinalizeThreeTaskMetrics(
        token_records,
        val_dataloader.dataset,
        model_module.config,
        epoch,
        logger,
    )


class Trainer(object):
    def __init__(self, model, train_dataloader=None,  val_dataloader=None,
                 optimizer=None, lr_scheduler=None, clip_grad_val=0.0, max_epochs=0,
                 eval_every_nepochs=1, eval_epochs=None, logger=None, log_every_niters=50, tb_log=None,
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
        self.b_freeze_x3_encoder = bool(freeze_x3_encoder)
        self.checkpoint_dir = checkpoint_dir
        output_dir = Path(checkpoint_dir).resolve().parent
        self.train_metrics_filename = str(output_dir / "train_metrics.jsonl")
        self.validation_metrics_filename = str(output_dir / "validation_metrics.json")
        self.validation_metrics_history_filename = str(output_dir / "validation_metrics.jsonl")
        if self.b_freeze_x3_encoder:
            FreezeX3Encoder(self.model, logger=self.logger)

    @property
    def current_lr(self):
        if self.optimizer is None:
            raise RuntimeError("lr is not applicable because optimizer does not exist.")
        return [group["lr"] for group in self.optimizer.param_groups]

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.cuda.current_device()
        return torch.device("cpu")

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
        if hasattr(model_module, 'LoadSingleState'):
            report = model_module.LoadSingleState(source_state_dict)
            target_keys = set(model_module.state_dict())
            report.setdefault('loaded_keys', sorted(source_state_dict))
            report.update(source_checkpoint=str(Path(filename).resolve()),
                          new_keys=sorted(target_keys - set(source_state_dict)))
            if self.rank == 0:
                SaveJson(str(Path(self.checkpoint_dir).parent / 'initialization.json'), report)
            self.epoch = 0
            self.global_step = 0
            self.b_resume_from_single_model_loaded = True
            self.logger.info('X4 single initialization: %s', report)
            return checkpoint
        target_state_dict = model_module.state_dict()

        matched_state_dict = {}
        unexpected_keys = []
        shape_mismatch_keys = []
        for key, value in source_state_dict.items():
            if key not in target_state_dict:
                unexpected_keys.append(key)
                continue
            if target_state_dict[key].shape != value.shape:
                shape_mismatch_keys.append((key, tuple(value.shape), tuple(target_state_dict[key].shape)))
                continue
            matched_state_dict[key] = value

        missing_shared_keys = [
            key for key in target_state_dict
            if key not in matched_state_dict and not key.startswith("x3_memory_transformer.")
        ]
        if unexpected_keys or shape_mismatch_keys or missing_shared_keys:
            raise RuntimeError(
                "single-frame checkpoint is incompatible with temporal shared model: "
                f"unexpected={unexpected_keys[:10]}, "
                f"shape_mismatch={shape_mismatch_keys[:10]}, "
                f"missing_shared={missing_shared_keys[:10]}"
            )

        target_state_dict.update(matched_state_dict)
        model_module.load_state_dict(target_state_dict, strict=True)
        self.epoch = 0
        self.global_step = 0
        self.b_resume_from_single_model_loaded = True

        missing_target_keys = [key for key in target_state_dict if key not in matched_state_dict]
        self.logger.info(
            "loaded single-frame weights: matched=%d | target_uninitialized=%d | unexpected=%d | shape_mismatch=%d",
            len(matched_state_dict),
            len(missing_target_keys),
            len(unexpected_keys),
            len(shape_mismatch_keys),
        )
        return checkpoint

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
        save_checkpoint(self.model, filename, optimizer=optimizer, scheduler=scheduler, meta=meta)

    def resume(self, checkpoint, resume_optimizer=True, map_location=torch.device("cpu")):
        validated_checkpoint = ValidateResumeCheckpoint(
            checkpoint,
            self.model,
            map_location=map_location,
            b_require_optimizer=bool(resume_optimizer and self.optimizer is not None),
            b_require_scheduler=bool(resume_optimizer and self.lr_scheduler is not None),
        )
        ValidateSamplingResume(self.train_dataloader, validated_checkpoint, self.lr_scheduler)
        checkpoint = self.load_checkpoint(checkpoint, map_location=map_location, strict=True)
        self.epoch = validated_checkpoint["meta"]["epoch"]
        self.global_step = validated_checkpoint["meta"]["iter"]
        if resume_optimizer and self.optimizer is not None:
            self.optimizer.load_state_dict(validated_checkpoint["optimizer"])
        if resume_optimizer and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(validated_checkpoint["scheduler"])
        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.global_step)

    def optimize_step(self, loss):
        CheckFiniteLoss(loss)
        self.optimizer.zero_grad()
        loss.backward()
        CheckFiniteGradients(self.model)
        if self.clip_grad_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_val)
            CheckFiniteGradients(self.model)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def train_iter(self, data_batch):
        data_batch = example_to_device(data_batch, self.device, non_blocking=False)
        frames = data_batch["frames"]
        seq_len = len(frames)
        warmup_len = int(data_batch.get("warmup_len", 0))
        train_len = int(data_batch.get("train_len", max(1, seq_len - warmup_len)))
        train_start = min(warmup_len, seq_len)
        train_end = min(seq_len, train_start + train_len)

        memory = None
        loss = None
        logs = {}
        train_step_count = 0

        for frame_idx in range(train_start):
            with torch.no_grad():
                memory_next = GetModelModule(self.model)(
                    frames[frame_idx],
                    memory=memory,
                    b_memory_only=True,
                )
                memory = DetachMemory(memory_next)

        for frame_idx in range(train_start, train_end):
            loss_t, logs_t, memory_next = self.model(frames[frame_idx], memory=memory)
            loss = loss_t if loss is None else loss + loss_t
            logs = logs_t
            memory = DetachMemory(memory_next)
            train_step_count += 1

        loss = loss / max(1, train_step_count)
        self.optimize_step(loss)
        loss_value = float(loss.detach().cpu())
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
                "epoch": int(self.epoch),
                "iter": int(self.global_step),
                "total_loss": loss_value,
                "learning_rate": float(self.current_lr[0]),
                "tasks": ScalarizeLogs(logs),
            })

        return loss_value, logs

    def train_epoch(self):
        self.model.train()
        if self.b_freeze_x3_encoder:
            SetFrozenX3EncoderEval(self.model)
        PrepareEpochSampling(self.train_dataloader, self.epoch, self.rank, self.checkpoint_dir)

        if self.rank == 0:
            progress_bar = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f"Train Epoch {self.epoch + 1}/{self.max_epochs}",
                dynamic_ncols=True,
            )
        else:
            progress_bar = enumerate(self.train_dataloader)

        for i, data_batch in progress_bar:
            self.inner_iter = i
            loss_value, logs = self.train_iter(data_batch)
            if self.rank == 0:
                progress_bar.set_postfix_str(self._build_train_tqdm_str(loss_value, logs))

        self.epoch += 1

    @torch.no_grad()
    def val_epoch(self):
        metric_dict = ValidateMultiTaskMemory(
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
            validation_record = {"epoch": int(self.epoch), "metrics": metric_dict}
            SaveJson(self.validation_metrics_filename, validation_record)
            AppendJsonLine(self.validation_metrics_history_filename, validation_record)

        return metric_dict

    def fit(self):
        self.logger.info("max: %d epochs", self.max_epochs)
        while self.epoch < self.max_epochs:
            self.train_epoch()
            b_validate = (
                (self.eval_every_nepochs > 0 and self.epoch % self.eval_every_nepochs == 0)
                or (self.eval_epochs is not None and self.epoch in self.eval_epochs)
            )
            if self.rank == 0:
                self.save_checkpoint("latest.pth")
                if b_validate or self.epoch == self.max_epochs:
                    self.save_checkpoint()
            if b_validate:
                self.val_epoch()

    def _build_train_tqdm_str(self, total_loss, logs):
        return BuildProgressText(total_loss, logs, self.current_lr[0])
