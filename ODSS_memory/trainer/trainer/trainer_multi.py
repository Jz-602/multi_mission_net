import os
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.progressbar import ProgressBar
from .multitask_validator import (
    BuildSegmentationConfusionMatrix,
    ComputeSegmentationMetrics,
    FormatDetectionMetrics,
    FormatSegmentationMetrics,
    WAYMO_INT_TO_TYPE_DICT,
    WaymoDetectionMetricsEstimator,
    label_to_type,
)


def example_to_device(example, device, non_blocking=False):
    ## 多帧 batch 是 dict/list/dict 的嵌套结构，这里递归搬运 tensor。
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


def ValidateMultiTaskMemory(model, val_dataloader, device, rank, world_size, epoch, logger):
    if val_dataloader is None:
        return None

    model.eval()
    model_module = GetModelModule(model)
    model_config = getattr(model_module, "config", None)
    num_classes = int(model_config.num_classes)
    ignore_index = getattr(model_config, "ignore_index", None)
    class_names = getattr(model_config, "class_names", None)

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.float64,
        device=device,
    )
    det_prediction_dict = {}

    if rank == 0:
        progress_bar = ProgressBar(len(val_dataloader))

    for data_batch in val_dataloader:
        data_batch = example_to_device(data_batch, device, non_blocking=False)
        frames = data_batch["frames"]
        seq_len = len(frames)
        warmup_len = int(data_batch.get("warmup_len", 0))
        train_len = int(data_batch.get("train_len", max(1, seq_len - warmup_len)))
        train_start = min(warmup_len, seq_len)
        train_end = min(seq_len, train_start + train_len)

        memory = None
        for frame_idx in range(train_start):
            memory_next = model(
                frames[frame_idx],
                memory=memory,
                b_memory_only=True,
            )
            memory = DetachMemory(memory_next)

        for frame_idx in range(train_start, train_end):
            batch_dict = frames[frame_idx]
            batch_result = model(batch_dict, memory=memory)
            point_pred = torch.cat(
                [batch_result[token]["seg"]["point_pred"].to(device) for token in batch_dict["token"]],
                dim=0,
            )
            point_coords_xyz_all = batch_dict["points"][:, 1:4]
            point_cloud_range = point_coords_xyz_all.new_tensor(model_config.point_cloud_range)
            point_cloud_min_xyz = point_cloud_range[:3]
            point_cloud_max_xyz = point_cloud_range[3:]
            b_valid_mask = (
                (point_coords_xyz_all >= point_cloud_min_xyz)
                & (point_coords_xyz_all < point_cloud_max_xyz)
            ).all(dim=1)
            point_label = batch_dict["semantic_label"][b_valid_mask].to(device).long()
            confusion_matrix += BuildSegmentationConfusionMatrix(
                point_pred=point_pred,
                point_label=point_label,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )

            for token in batch_dict["token"]:
                det_result = batch_result[token].get("det", {})
                det_prediction_dict[token] = {
                    "box3d_lidar": det_result.get("box3d_lidar", torch.zeros((0, 7), dtype=torch.float32)),
                    "scores": det_result.get("scores", torch.zeros((0,), dtype=torch.float32)),
                    "label_preds": det_result.get("label_preds", torch.zeros((0,), dtype=torch.long)),
                }

        if rank == 0:
            progress_bar.update()

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        all_det_predictions = [None for _ in range(world_size)]
        dist.all_gather_object(all_det_predictions, det_prediction_dict)

    if rank != 0:
        return None

    if world_size > 1:
        merged_det_prediction_dict = {}
        for det_prediction_part in all_det_predictions:
            merged_det_prediction_dict.update(det_prediction_part)
    else:
        merged_det_prediction_dict = det_prediction_dict

    seg_metric_dict = ComputeSegmentationMetrics(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        ignore_index=ignore_index,
    )

    dataset = val_dataloader.dataset
    gt_infos = []
    pred_infos = []
    for info in dataset.infos:
        token = info["token"]
        if token not in merged_det_prediction_dict:
            continue

        det_result = merged_det_prediction_dict[token]
        box_3d_lidar = det_result["box3d_lidar"].detach().cpu().numpy()
        scores = det_result["scores"].detach().cpu().numpy()
        label_preds = det_result["label_preds"].detach().cpu().numpy()

        gt_object_list = []
        for gt_object in info["objects"]:
            difficulty = 0
            if "difficulty_level" in gt_object and len(gt_object["difficulty_level"]) > 0:
                difficulty_val = gt_object["difficulty_level"][0]
                if not np.isnan(difficulty_val):
                    difficulty = int(difficulty_val)

            gt_object_list.append({
                "box": np.asarray(gt_object["box"], dtype=np.float32),
                "type_name": WAYMO_INT_TO_TYPE_DICT[int(gt_object["type"])],
                "num_points_in_top_lidar": int(gt_object["num_points_in_top_lidar"]),
                "difficulty": difficulty,
            })

        gt_infos.append({
            "token": token,
            "gt_objects": gt_object_list,
        })
        pred_infos.append({
            "name": [WAYMO_INT_TO_TYPE_DICT[label_to_type(int(label))] for label in label_preds],
            "score": scores,
            "boxes_lidar": box_3d_lidar,
            "sequence_name": token.split("-")[0],
            "frame_timestamp": int(token.split("-")[1]),
        })

    logger.info(
        "Validation Epoch %d: start Waymo detection metric, frames=%d, pred_boxes=%d, gt_boxes=%d",
        epoch,
        len(pred_infos),
        int(sum(len(info["score"]) for info in pred_infos)),
        int(sum(len(info["gt_objects"]) for info in gt_infos)),
    )
    det_metric_dict = WaymoDetectionMetricsEstimator().Evaluate(pred_infos, gt_infos)

    logger.info(FormatSegmentationMetrics(epoch, seg_metric_dict))
    logger.info(FormatDetectionMetrics(epoch, det_metric_dict))
    return {"seg": seg_metric_dict, "det": det_metric_dict}


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
            "loaded single-frame weights: matched=%d | target_uninitialized=%d | unexpected=%d | shape_mismatch=%d",
            len(matched_state_dict),
            len(missing_target_keys),
            len(skipped_unexpected_keys),
            len(skipped_shape_keys),
        )
        if len(skipped_shape_keys) > 0:
            preview_items = [
                f"{key}: source{source_shape}->target{target_shape}"
                for key, source_shape, target_shape in skipped_shape_keys[:10]
            ]
            self.logger.info("single-frame shape-mismatch keys preview: %s", "; ".join(preview_items))
        return checkpoint

    def save_checkpoint(self, filename_tmpl="epoch_{}.pth", save_optimizer=True):
        meta = dict(epoch=self.epoch, iter=self.global_step)
        filepath = filename_tmpl.format(self.epoch)
        optimizer = self.optimizer if save_optimizer else None
        scheduler = self.lr_scheduler if save_optimizer else None
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = os.path.join(self.checkpoint_dir, filepath)
        save_checkpoint(self.model, filename, optimizer=optimizer, scheduler=scheduler, meta=meta)

    def resume(self, checkpoint, resume_optimizer=True, map_location=torch.device("cpu")):
        checkpoint = self.load_checkpoint(checkpoint, map_location=map_location, strict=True)
        self.epoch = checkpoint["meta"]["epoch"]
        self.global_step = checkpoint["meta"]["iter"]
        if "optimizer" in checkpoint and resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint and resume_optimizer:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler"])
        self.logger.info("resumed epoch %d, iter %d", self.epoch, self.global_step)

    def optimize_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_val)
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
                memory_next = self.model(
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

        if self.tb_log is not None:
            self.WriteTrainTensorboard(loss, logs)

        return loss_value, logs

    def WriteTrainTensorboard(self, loss, logs):
        self.tb_log.add_scalar("train/loss", loss, self.global_step)
        current_lr_list = self.current_lr
        self.tb_log.add_scalar("train/lr", current_lr_list[0], self.global_step)

        seg_log = [log_item for log_item in logs if log_item["task"][0] == "seg"][0]
        det_logs = [log_item for log_item in logs if log_item["task"][0] != "seg"]
        self.tb_log.add_scalar("train/seg_loss", seg_log["loss"], self.global_step)
        self.tb_log.add_scalar("train/det_loss", sum([det_log["loss"] for det_log in det_logs]), self.global_step)
        self.tb_log.add_scalar("train/det_hm_loss", sum([det_log["hm_loss"] for det_log in det_logs]), self.global_step)
        self.tb_log.add_scalar("train/det_loc_loss", sum([det_log["loc_loss"] for det_log in det_logs]), self.global_step)

        if "log_var" in seg_log:
            self.tb_log.add_scalar("train/seg_loss_weight", seg_log["loss_weight"], self.global_step)
            self.tb_log.add_scalar("train/seg_weighted_loss", seg_log["weighted_loss"], self.global_step)
        det_weight_logs = [det_log for det_log in det_logs if "log_var" in det_log]
        if len(det_weight_logs) > 0:
            self.tb_log.add_scalar("train/det_loss_weight", det_weight_logs[0]["loss_weight"], self.global_step)
            self.tb_log.add_scalar("train/det_weighted_loss", det_weight_logs[0]["weighted_loss"], self.global_step)

        for key in ["DiceLossV1", "CELoss", "LovLoss"]:
            if key in seg_log:
                self.tb_log.add_scalar(f"train/seg{key}", seg_log[key], self.global_step)

    def train_epoch(self):
        self.model.train()
        if self.b_freeze_x3_encoder:
            SetFrozenX3EncoderEval(self.model)
        if self.world_size > 1:
            self.train_dataloader.sampler.set_epoch(self.epoch)

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

        if self.rank == 0 and self.tb_log is not None and metric_dict is not None:
            seg_metric_dict = metric_dict["seg"]
            det_metric_dict = metric_dict["det"]
            self.tb_log.add_scalar("val/seg_miou", seg_metric_dict["miou"], self.epoch)
            self.tb_log.add_scalar("val/seg_macc", seg_metric_dict["macc"], self.epoch)
            self.tb_log.add_scalar("val/seg_all_acc", seg_metric_dict["all_acc"], self.epoch)
            for class_result in seg_metric_dict["class_result_list"]:
                class_name = class_result["class_name"]
                if class_result["b_valid_iou"]:
                    self.tb_log.add_scalar(f"val_seg_class_iou/{class_name}", class_result["iou"], self.epoch)
                if class_result["b_valid_acc"]:
                    self.tb_log.add_scalar(f"val_seg_class_acc/{class_name}", class_result["acc"], self.epoch)
            for key, val in det_metric_dict.items():
                self.tb_log.add_scalar(f"val_det/{key}", val, self.epoch)

        return metric_dict

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

    def _build_train_tqdm_str(self, total_loss, logs):
        seg_log = [log_item for log_item in logs if log_item["task"][0] == "seg"][0]
        det_logs = [log_item for log_item in logs if log_item["task"][0] != "seg"]

        def ToFloat(val):
            if isinstance(val, torch.Tensor):
                return float(val.detach().cpu())
            return float(val)

        raw_seg_loss = ToFloat(seg_log["loss"])
        seg_display_loss = ToFloat(seg_log.get("weighted_loss", seg_log["loss"]))
        seg_loss_items = [f"raw_seg_loss={raw_seg_loss:.3f}"]
        if "loss_weight" in seg_log:
            seg_loss_items.append(f"seg_loss_weight={ToFloat(seg_log['loss_weight']):.3f}")
        for key, val in seg_log.items():
            if key in ["task", "loss", "weighted_loss", "log_var", "loss_weight"]:
                continue
            seg_loss_items.append(f"{key}={ToFloat(val):.3f}")

        raw_det_loss = 0.0
        hm_loss = 0.0
        loc_loss = 0.0
        det_weight_log = None
        for det_log in det_logs:
            raw_det_loss += ToFloat(det_log["loss"])
            hm_loss += ToFloat(det_log["hm_loss"])
            loc_loss += ToFloat(det_log["loc_loss"])
            if det_weight_log is None and "weighted_loss" in det_log:
                det_weight_log = det_log

        det_display_loss = raw_det_loss
        det_loss_items = [f"raw_det_loss={raw_det_loss:.3f}"]
        if det_weight_log is not None:
            det_display_loss = ToFloat(det_weight_log["weighted_loss"])
            det_loss_items.append(f"det_loss_weight={ToFloat(det_weight_log['loss_weight']):.3f}")
        det_loss_items.extend([f"hm_loss={hm_loss:.3f}", f"loc_loss={loc_loss:.3f}"])

        return (
            f"lr={self.current_lr[0]:.5f} | total_loss={total_loss:.4f} | "
            f"seg_loss={seg_display_loss:.4f}:[{','.join(seg_loss_items)}] | "
            f"det_loss={det_display_loss:.4f}:[{','.join(det_loss_items)}]"
        )
