import json
import numbers
import os

import torch


TASK_NAMES = ("det", "terrain", "occ")


def MoveToDevice(value, device, non_blocking=False):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: MoveToDevice(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [MoveToDevice(item, device, non_blocking=non_blocking) for item in value]
    if isinstance(value, tuple):
        return tuple(MoveToDevice(item, device, non_blocking=non_blocking) for item in value)
    return value


def GetModelModule(model):
    return model.module if hasattr(model, "module") else model


def ToScalar(value):
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"training log tensor must be scalar, got shape {tuple(value.shape)}")
        return value.detach().cpu().item()
    if isinstance(value, numbers.Number):
        return value.item() if hasattr(value, "item") else value
    if isinstance(value, str) or value is None:
        return value
    raise TypeError(f"training log value must be scalar, got {type(value).__name__}")


def ScalarizeLogs(logs):
    output = []
    for log_item in logs:
        scalar_item = {}
        for key, value in log_item.items():
            if key == "task":
                scalar_item[key] = list(value)
            else:
                scalar_item[key] = ToScalar(value)
        output.append(scalar_item)
    return output


def TaskLogMap(logs):
    task_logs = {}
    for log_item in logs:
        task_names = log_item.get("task", [])
        if len(task_names) != 1 or task_names[0] not in TASK_NAMES:
            raise ValueError(f"expected one task in {TASK_NAMES}, got {task_names}")
        task_name = task_names[0]
        if task_name in task_logs:
            raise ValueError(f"duplicate task log: {task_name}")
        task_logs[task_name] = log_item
    missing_tasks = [task_name for task_name in TASK_NAMES if task_name not in task_logs]
    if missing_tasks:
        raise ValueError(f"missing task logs: {missing_tasks}")
    return task_logs


def AppendJsonLine(filename, record):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def SaveJson(filename, record):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    temporary_filename = filename + ".tmp"
    with open(temporary_filename, "w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, allow_nan=False, indent=2)
    os.replace(temporary_filename, filename)


def CheckFiniteLoss(loss):
    if loss.numel() != 1 or not bool(torch.isfinite(loss.detach()).item()):
        raise FloatingPointError(f"training loss is not finite: {loss.detach().cpu()}")


def CheckFiniteGradients(model):
    # 正常路径把各参数的 GPU bool 标量合并，只同步一次；异常时保留原定位逻辑。
    b_finite_checks = [
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not b_finite_checks or bool(torch.stack(b_finite_checks).all().item()):
        return
    nonfinite_names = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
            nonfinite_names.append(name)
            if len(nonfinite_names) >= 10:
                break
    if nonfinite_names:
        raise FloatingPointError(
            "non-finite gradients detected before optimizer.step: "
            + ", ".join(nonfinite_names)
        )


def ConfigForCheckpoint(value):
    if isinstance(value, dict):
        return {str(key): ConfigForCheckpoint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ConfigForCheckpoint(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def BuildCheckpointMeta(model, epoch, global_step):
    model_module = GetModelModule(model)
    task_names = list(getattr(model_module.task_uncertainty_loss, "task_names", TASK_NAMES))
    config = ConfigForCheckpoint(getattr(model_module, "config", {}))
    return {
        "epoch": int(epoch),
        "iter": int(global_step),
        "task_names": task_names,
        "model_config": config,
    }


def ValidateResumeCheckpoint(
    filename,
    model,
    map_location="cpu",
    b_require_optimizer=True,
    b_require_scheduler=True,
):
    if not os.path.isfile(filename):
        raise IOError(f"{filename} is not a checkpoint file")
    try:
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("resume checkpoint must contain full training state")
    meta = checkpoint.get("meta")
    if not isinstance(meta, dict):
        raise RuntimeError("resume checkpoint is missing meta")
    missing_meta = [key for key in ("epoch", "iter", "task_names") if key not in meta]
    if missing_meta:
        raise RuntimeError(f"resume checkpoint meta is missing: {missing_meta}")
    expected_tasks = list(
        getattr(GetModelModule(model).task_uncertainty_loss, "task_names", TASK_NAMES)
    )
    if list(meta["task_names"]) != expected_tasks:
        raise RuntimeError(
            f"resume task_names mismatch: checkpoint={meta['task_names']}, model={expected_tasks}"
        )
    if b_require_optimizer and "optimizer" not in checkpoint:
        raise RuntimeError("resume checkpoint is missing optimizer state")
    if b_require_scheduler and "scheduler" not in checkpoint:
        raise RuntimeError("resume checkpoint is missing scheduler state")
    return checkpoint


def WriteTrainTensorboard(tb_log, total_loss, logs, global_step, learning_rate):
    if tb_log is None:
        return
    tb_log.add_scalar("train/loss", total_loss, global_step)
    tb_log.add_scalar("train/lr", learning_rate, global_step)
    for task_name, log_item in TaskLogMap(logs).items():
        for key, value in log_item.items():
            if key == "task" or not isinstance(value, (bool, int, float, torch.Tensor)):
                continue
            if isinstance(value, torch.Tensor) and value.numel() != 1:
                continue
            tb_log.add_scalar(f"train_{task_name}/{key}", value, global_step)


def WriteValidationTensorboard(tb_log, metrics, epoch):
    if tb_log is None or metrics is None:
        return
    for task_name, task_metrics in metrics.items():
        for key, value in task_metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                tb_log.add_scalar(f"val_{task_name}/{key}", value, epoch)
            elif isinstance(value, list):
                for item_idx, item in enumerate(value):
                    if isinstance(item, (int, float)) and item is not None:
                        tb_log.add_scalar(f"val_{task_name}/{key}_{item_idx}", item, epoch)


def BuildProgressText(total_loss, logs, learning_rate):
    task_logs = TaskLogMap(logs)
    parts = [f"lr={learning_rate:.5f}", f"total={float(total_loss):.4f}"]
    for task_name in TASK_NAMES:
        log_item = task_logs[task_name]
        raw_loss = float(ToScalar(log_item["loss"]))
        display_loss = float(ToScalar(log_item.get("weighted_loss", log_item["loss"])))
        task_text = f"{task_name}={display_loss:.4f}(raw={raw_loss:.4f}"
        if "loss_weight" in log_item:
            task_text += f",w={float(ToScalar(log_item['loss_weight'])):.3f}"
        task_text += ")"
        parts.append(task_text)
    return " | ".join(parts)
