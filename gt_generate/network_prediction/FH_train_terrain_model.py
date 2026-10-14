import json
import os
import sys
from argparse import Namespace

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gt_generate.common.constants import INVALID_HEIGHT
from gt_generate.common.io_utils import EnsureDir, SaveJson, UpdateNamespaceFromCli
from gt_generate.ground_truth.data_paths import DEFAULT_FH_WAYMO_ROOT, DEFAULT_FH_WAYMO_SEQUENCE_NAME, ApplyResolvedDataPaths, BuildDefaultTerrainDir, ResolveTerrainOnlyDataPaths
from gt_generate.network_prediction.training_data import TerrainOnlyFrameDataset
from gt_generate.network_prediction.training_model import TerrainOnlyNet
from gt_generate.common.training_torch import NumpyToTorch, StackBatchItems


def BuildProgressBar(iterable, total=None, desc=""):
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def BuildArgs():
    return Namespace(
        data_path=DEFAULT_FH_WAYMO_ROOT,
        sequence_name=DEFAULT_FH_WAYMO_SEQUENCE_NAME,
        pcd_dir="",
        semantic_label_dir="",
        pose_filename="",
        terrain_dir="",
        manifest_filename="",
        run_dir="",
        resume_checkpoint="",
        epoch_num=20,
        batch_size=2,
        learning_rate=1e-3,
        hidden_channels=32,
        val_ratio=0.15,
        val_interval=2,
        max_samples=0,
        input_point_stride=1,
        b_cache_observation_inputs=True,
        terrain_main_beta=0.15,
        terrain_grad_weight=0.2,
        terrain_distance_weight_sigma_m=3.0,
        loss_eps=1e-6,
        device="cuda",
        seed=0,
    )


def CollateTerrainOnlyBatch(sample_list):
    torch_samples = [NumpyToTorch(sample) for sample in sample_list]
    return {
        "timestamp": [sample["timestamp"] for sample in torch_samples],
        "input": StackBatchItems([sample["input"] for sample in torch_samples]),
        "target": StackBatchItems([sample["target"] for sample in torch_samples]),
        "mask": StackBatchItems([sample["mask"] for sample in torch_samples]),
        "weight": StackBatchItems([sample["weight"] for sample in torch_samples]),
        "meta": [sample["meta"] for sample in torch_samples],
    }


def BuildInputTensor(batch_input, input_field_names):
    return torch.cat([batch_input[field_name].float() for field_name in input_field_names], dim=1)


def MoveBatchToDevice(batch_dict, device):
    output_dict = {"timestamp": batch_dict["timestamp"], "input": {}, "target": {}, "mask": {}, "weight": {}, "meta": batch_dict["meta"]}
    for group_name in ["input", "target", "mask", "weight"]:
        for key, value in batch_dict[group_name].items():
            output_dict[group_name][key] = value.to(device=device, dtype=torch.float32)
    return output_dict


def WeightedMean(value_map, weight_map, eps):
    return torch.sum(value_map * weight_map) / torch.clamp(torch.sum(weight_map), min=float(eps))


def SmoothL1LossMap(pred_map, target_map, beta):
    diff = torch.abs(pred_map - target_map)
    beta = max(float(beta), 1e-6)
    return torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)


def TerrainGradientLoss(pred_map, target_map, valid_mask, weight_map, eps):
    pred_dx = pred_map[:, :, :, 1:] - pred_map[:, :, :, :-1]
    pred_dy = pred_map[:, :, 1:, :] - pred_map[:, :, :-1, :]
    target_dx = target_map[:, :, :, 1:] - target_map[:, :, :, :-1]
    target_dy = target_map[:, :, 1:, :] - target_map[:, :, :-1, :]
    valid_dx = valid_mask[:, :, :, 1:] * valid_mask[:, :, :, :-1]
    valid_dy = valid_mask[:, :, 1:, :] * valid_mask[:, :, :-1, :]
    weight_dx = 0.5 * (weight_map[:, :, :, 1:] + weight_map[:, :, :, :-1])
    weight_dy = 0.5 * (weight_map[:, :, 1:, :] + weight_map[:, :, :-1, :])
    loss_dx = WeightedMean(torch.abs(pred_dx - target_dx), valid_dx * weight_dx, eps)
    loss_dy = WeightedMean(torch.abs(pred_dy - target_dy), valid_dy * weight_dy, eps)
    return 0.5 * (loss_dx + loss_dy)


def ComputeLoss(pred_dict, batch_dict, args):
    pred_map = pred_dict["terrain_z_inpainted"]
    target_map = batch_dict["target"]["terrain_z_inpainted"].float()
    valid_mask = batch_dict["mask"]["terrain_inpainted_mask"].float()
    distance_map = batch_dict["weight"]["terrain_inpaint_distance"].float()
    distance_valid_mask = (distance_map != INVALID_HEIGHT).float()
    distance_weight = torch.exp(-torch.clamp(distance_map, min=0.0) / max(float(args.terrain_distance_weight_sigma_m), 1e-6))
    exported_weight = batch_dict["weight"]["terrain_train_weight"].float()
    weight_map = torch.maximum(exported_weight, distance_weight * distance_valid_mask) * valid_mask
    main_loss = WeightedMean(SmoothL1LossMap(pred_map, target_map, args.terrain_main_beta), weight_map, args.loss_eps)
    grad_loss = TerrainGradientLoss(pred_map, target_map, valid_mask, weight_map, args.loss_eps)
    total_loss = main_loss + float(args.terrain_grad_weight) * grad_loss
    abs_err = torch.abs(pred_map - target_map)
    return total_loss, {
        "total": float(total_loss.detach().cpu().item()),
        "terrain_main": float(main_loss.detach().cpu().item()),
        "terrain_grad": float(grad_loss.detach().cpu().item()),
        "terrain_mae_m": float(WeightedMean(abs_err, weight_map, args.loss_eps).detach().cpu().item()),
    }


def AverageMetrics(metric_items):
    output = {}
    for key in metric_items[0].keys():
        output[key] = float(sum(item[key] for item in metric_items) / max(len(metric_items), 1))
    return output


def BuildTrainValDatasets(dataset, args):
    sample_num = len(dataset)
    if sample_num <= 1 or float(args.val_ratio) <= 0.0:
        return dataset, None
    val_num = max(1, int(round(sample_num * float(args.val_ratio))))
    train_num = max(1, sample_num - val_num)
    index_list = list(range(sample_num))
    random_state = np.random.RandomState(int(args.seed))
    random_state.shuffle(index_list)
    return Subset(dataset, index_list[:train_num]), Subset(dataset, index_list[train_num:train_num + val_num])


def RunEpoch(model, data_loader, optimizer, args, device, b_train):
    model.train(mode=bool(b_train))
    metric_items = []
    stage_name = "train" if bool(b_train) else "val"
    progress_iter = BuildProgressBar(data_loader, total=len(data_loader), desc=f"TerrainOnly-{stage_name}")
    for batch_dict in progress_iter:
        moved_batch = MoveBatchToDevice(batch_dict, device)
        input_tensor = BuildInputTensor(moved_batch["input"], data_loader.dataset.dataset.input_field_names if isinstance(data_loader.dataset, Subset) else data_loader.dataset.input_field_names)
        pred_dict = model(input_tensor)
        loss, metrics = ComputeLoss(pred_dict, moved_batch, args)
        if bool(b_train):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metric_items.append(metrics)
        if hasattr(progress_iter, "set_postfix"):
            progress_iter.set_postfix(
                total=f"{metrics['total']:.4f}",
                main=f"{metrics['terrain_main']:.4f}",
                mae=f"{metrics['terrain_mae_m']:.3f}",
            )
    return AverageMetrics(metric_items)


def SaveCheckpoint(model, optimizer, epoch_idx, run_dir, metrics, filename):
    EnsureDir(os.path.join(run_dir, "checkpoints"))
    torch.save(
        {
            "epoch": int(epoch_idx),
            "input_channel_num": int(model.enc1.conv.block[0].weight.shape[1]),
            "hidden_channels": int(model.enc1.conv.block[0].weight.shape[0]),
            "input_field_names": list(getattr(model, "input_field_names", [])),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        os.path.join(run_dir, "checkpoints", filename),
    )


def LoadResumeCheckpoint(model, optimizer, checkpoint_filename, device):
    checkpoint = torch.load(checkpoint_filename, map_location=device, weights_only=False)
    current_channel_num = int(model.enc1.conv.block[0].weight.shape[1])
    checkpoint_channel_num = int(checkpoint.get("input_channel_num", current_channel_num))
    if checkpoint_channel_num != current_channel_num:
        raise RuntimeError(f"Resume checkpoint input_channel_num mismatch: checkpoint={checkpoint_channel_num}, current={current_channel_num}")
    checkpoint_input_field_names = list(checkpoint.get("input_field_names", []))
    if checkpoint_input_field_names and checkpoint_input_field_names != list(getattr(model, "input_field_names", [])):
        raise RuntimeError("Resume checkpoint input_field_names mismatch. Please resume from a checkpoint trained with the same input fields.")
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def DrawMetricCurve(canvas, x_values, y_values, color, x_min, x_max, y_min, y_max, margin):
    import cv2

    points = []
    plot_width = canvas.shape[1] - 2 * margin
    plot_height = canvas.shape[0] - 2 * margin
    for x_value, y_value in zip(x_values, y_values):
        if y_value is None:
            continue
        x_norm = 0.0 if x_max <= x_min else (float(x_value) - float(x_min)) / max(float(x_max - x_min), 1e-6)
        y_norm = 0.0 if y_max <= y_min else (float(y_value) - float(y_min)) / max(float(y_max - y_min), 1e-6)
        px = int(round(margin + x_norm * plot_width))
        py = int(round(canvas.shape[0] - margin - y_norm * plot_height))
        points.append((px, py))
    for point_idx in range(len(points) - 1):
        cv2.line(canvas, points[point_idx], points[point_idx + 1], color, 2, cv2.LINE_AA)


def SaveLossCurveImages(train_log, run_dir):
    try:
        import cv2
    except ImportError:
        return
    metric_names = ["total", "terrain_main", "terrain_grad", "terrain_mae_m"]
    curve_dir = os.path.join(run_dir, "curves")
    EnsureDir(curve_dir)
    epoch_logs = list(train_log.get("epoch_logs", []))
    if not epoch_logs:
        return
    x_values = [int(item["epoch"]) for item in epoch_logs]
    for metric_name in metric_names:
        train_values = [float(item["train"][metric_name]) for item in epoch_logs]
        val_values = [
            None if item.get("val") is None else float(item["val"][metric_name])
            for item in epoch_logs
        ]
        all_values = [value for value in train_values + val_values if value is not None]
        if not all_values:
            continue
        y_min = min(all_values)
        y_max = max(all_values)
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0
        canvas = np.full((480, 720, 3), 245, dtype=np.uint8)
        margin = 60
        cv2.rectangle(canvas, (margin, margin), (canvas.shape[1] - margin, canvas.shape[0] - margin), (70, 70, 70), 1)
        DrawMetricCurve(canvas, x_values, train_values, (255, 120, 0), min(x_values), max(x_values), y_min, y_max, margin)
        DrawMetricCurve(canvas, x_values, val_values, (0, 120, 255), min(x_values), max(x_values), y_min, y_max, margin)
        cv2.putText(canvas, f"{metric_name}", (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.putText(canvas, "train", (canvas.shape[1] - 190, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "val", (canvas.shape[1] - 95, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"min={y_min:.4f}", (margin, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"max={y_max:.4f}", (canvas.shape[1] - 180, canvas.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(curve_dir, f"{metric_name}.png"), canvas)


def Main():
    args = BuildArgs()
    UpdateNamespaceFromCli(args, sys.argv)
    data_paths = ResolveTerrainOnlyDataPaths(args, b_require_semantic=False, b_require_pose=False)
    ApplyResolvedDataPaths(args, data_paths)
    if not str(args.terrain_dir).strip():
        args.terrain_dir = BuildDefaultTerrainDir(args)
    if not str(args.manifest_filename).strip():
        args.manifest_filename = os.path.join(args.terrain_dir, "terrain_only_frame_gt_manifest.json")
    if not str(args.run_dir).strip():
        args.run_dir = os.path.join(args.terrain_dir, "training_runs", "terrain_only_v1")
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(int(args.seed))
    EnsureDir(args.run_dir)
    SaveJson(os.path.join(args.run_dir, "args.json"), vars(args))

    dataset = TerrainOnlyFrameDataset(args)
    train_dataset, val_dataset = BuildTrainValDatasets(dataset, args)
    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, collate_fn=CollateTerrainOnlyBatch)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=CollateTerrainOnlyBatch)

    first_batch = next(iter(train_loader))
    input_channel_num = int(BuildInputTensor(first_batch["input"], dataset.input_field_names).shape[1])
    model = TerrainOnlyNet(input_channel_num=input_channel_num, hidden_channels=int(args.hidden_channels)).to(device=args.device)
    model.input_field_names = list(dataset.input_field_names)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
    resume_epoch = 0
    if str(args.resume_checkpoint).strip():
        resume_epoch = LoadResumeCheckpoint(model, optimizer, str(args.resume_checkpoint), args.device)
        print(f"[TerrainOnlyTrain] resumed checkpoint={args.resume_checkpoint} resume_epoch={resume_epoch}")
    print(
        "[TerrainOnlyTrain] "
        f"device={args.device} "
        f"cuda_available={torch.cuda.is_available()} "
        f"train_samples={len(train_dataset)} "
        f"val_samples={0 if val_dataset is None else len(val_dataset)} "
        f"input_channels={input_channel_num}"
    )

    train_log = {"epoch_logs": []}
    best_val = None
    start_epoch = int(resume_epoch) + 1
    end_epoch = int(resume_epoch) + int(args.epoch_num)
    epoch_iter = BuildProgressBar(range(start_epoch, end_epoch + 1), total=int(args.epoch_num), desc="TerrainOnly-Epoch")
    for epoch_idx in epoch_iter:
        train_metrics = RunEpoch(model, train_loader, optimizer, args, args.device, b_train=True)
        b_run_val = val_loader is not None and (epoch_idx % int(args.val_interval) == 0 or epoch_idx == end_epoch)
        val_metrics = None
        if b_run_val:
            with torch.no_grad():
                val_metrics = RunEpoch(model, val_loader, optimizer, args, args.device, b_train=False)
        epoch_log = {"epoch": int(epoch_idx), "train": train_metrics, "val": val_metrics}
        train_log["epoch_logs"].append(epoch_log)
        SaveJson(os.path.join(args.run_dir, "train_log.json"), train_log)
        SaveLossCurveImages(train_log, args.run_dir)
        SaveCheckpoint(model, optimizer, epoch_idx, args.run_dir, epoch_log, f"epoch_{epoch_idx:03d}.pt")
        monitor_value = train_metrics["total"] if val_metrics is None else val_metrics["total"]
        if best_val is None or monitor_value < best_val:
            best_val = float(monitor_value)
            SaveCheckpoint(model, optimizer, epoch_idx, args.run_dir, epoch_log, "best.pt")
        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                train=f"{train_metrics['total']:.4f}",
                val="none" if val_metrics is None else f"{val_metrics['total']:.4f}",
                best=f"{best_val:.4f}",
            )
        print(f"[TerrainOnlyTrain] epoch={epoch_idx} train_total={train_metrics['total']:.4f} val_total={None if val_metrics is None else round(val_metrics['total'], 4)}")


if __name__ == "__main__":
    Main()
