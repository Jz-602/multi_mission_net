"""真实数据整网前反向与显存检查，沿用正式模型、Dataset 和损失。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from easydict import EasyDict
from LYF_train_ODSS import CfgFromYamlFile, Instantiate, SetRandomSeed
from WaymoDatasetV2 import WaymoDataset
from WaymoDataset_multi import WaymoDatasetMulti
from multi_mission_model import MultiMissionModel as SingleModel
from multi_mission_model_memory import MultiMissionModel as TemporalModel
from run_options import CheckRunConfig
from trainer.trainer.three_task_runtime import MoveToDevice, ScalarizeLogs
from trainer.trainer.trainer_multi import DetachMemory


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--index', type=int, default=10)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--encoder-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    SetRandomSeed(0, 0)
    cfg = CfgFromYamlFile(args.config, EasyDict())
    CheckRunConfig(cfg)
    dataset = (WaymoDatasetMulti if args.temporal else WaymoDataset)(cfg.data.train_dataset, cfg.common_variables)
    batch = MoveToDevice(dataset.CollateBatch([dataset[args.index]]), 'cuda')
    for key in ('tasks', 'out_size_factor', 'voxel_size'):
        cfg.model[key] = cfg.common_variables[key]
    cfg.model.point_cloud_range = cfg.common_variables.pc_range
    model = (TemporalModel if args.temporal else SingleModel)(cfg.model).cuda().train()
    optimizer = Instantiate(cfg.optimizer, params=model.parameters())
    records = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        memory = None
        if args.temporal:
            for frame in batch['frames'][:-1]:
                with torch.no_grad():
                    memory = DetachMemory(model(frame, memory, b_memory_only=True))
            target = batch['frames'][-1]
        else:
            target = batch
        torch.cuda.synchronize()
        target_start = time.perf_counter()
        if args.encoder_only:
            with torch.no_grad():
                x2, x3 = model.EncodeCurrent(target)
                current_count = len(x3.indices)
                if args.temporal:
                    aligned = model.AlignMemoryToCurrentX3(target, memory, x3)
                    updated = model.MemoryUpdate(x3, aligned, x3.batch_size)
                    from spconv.pytorch import SparseConvTensor
                    x3 = SparseConvTensor(updated['feats'], updated['coords'].int(), updated['spatial_shape'], updated['batch_size'])
                x4 = model.encoder_stage4(x3)
                from spconv.pytorch import SparseConvTensor
                root = SparseConvTensor(x4.features,x4.indices,x4.spatial_shape,x4.batch_size)
                root = model.occ_head.root_expand(root) if model.occ_head.root_expand is not None else root
                record = {'x3_current': current_count, 'x3_fused': len(x3.indices), 'x4': len(x4.indices),
                          'expanded_roots': len(root.indices), 'leaves_before_filter': len(root.indices)*64}
        else:
            outputs = model.ForwardNetwork(target, memory) if args.temporal else model.ForwardNetwork(target)
            loss, logs = model.ComputeLoss(outputs, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite total loss')
            loss.backward()
            missing = [name for name,param in model.named_parameters() if param.requires_grad and param.grad is None]
            nonfinite = [name for name,param in model.named_parameters() if param.grad is not None and not bool(torch.isfinite(param.grad).all())]
            if missing or nonfinite:
                raise AssertionError({'missing_gradient':missing,'nonfinite_gradient':nonfinite})
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.trainer.clip_grad_val), error_if_nonfinite=True)
            optimizer.step()
            record = {'total_loss':float(loss.detach()), 'tasks':ScalarizeLogs(logs),
                      'counts':outputs['counts'], 'occ_counts':outputs['occ']['counts'],
                      'all_parameter_gradients_finite':True}
            del outputs, loss, logs
        torch.cuda.synchronize()
        record.update(step=step, token=target['token'][0], total_seconds=time.perf_counter()-start,
                      target_seconds=time.perf_counter()-target_start,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        records.append(record)
        print(json.dumps(record), flush=True)
    report = {'device':torch.cuda.get_device_name(), 'pc_range':cfg.common_variables.pc_range,
              'voxel_size':cfg.common_variables.voxel_size,'temporal':args.temporal,
              'warmup_len':getattr(dataset,'warmup_len',0),'allow_tf32':cfg.get('allow_tf32',False),
              'precision':'float32','first_step_includes_kernel_initialization':True,
              'config':str(Path(args.config).resolve()),'records':records}
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))

if __name__ == '__main__':
    Main()
