"""Read-only, identical-real-clip comparison of a checkpoint and bounded root-diagnostic arms.

The input JSON is {"arms":[{"name":"old20","config":"...","checkpoint":"..."},...]};
optional indices default to six fixed validation targets. Checkpoints load strictly
as model state ONLY: optimizer/scheduler/diagnostic-run controllers are not restored.
No mini-set detection mAP is computed or claimed. This is not full validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools')]


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    import torch
    from easydict import EasyDict
    from build_label_manifest import LoadConfig
    from run_options import CheckRunConfig, HashFile
    from LYF_train_ODSS import SetRandomSeed
    from WaymoDataset_multi import WaymoDatasetMulti
    from multi_mission_model_memory_x4 import MultiMissionModel
    from trainer.trainer.multitask_validator import BuildTerrainFrameStatistics, SummarizeTerrainMetrics
    from occ_metrics import SummarizeMetrics
    from visualize_three_task import ToDevice, PlotComparison
    from preflight_x4_detection import Write

    plan = json.loads(Path(args.plan).read_text())
    indices = plan.get('indices', [0,100,2720,1166,5047,4346])
    if len(set(indices)) != len(indices) or not plan.get('arms'):
        raise ValueError('Need distinct validation indices and explicit arms')
    names = [arm['name'] for arm in plan['arms']]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError('Arm names must be unique plain path components')
    destination = Path(args.output); destination.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); SetRandomSeed(0, 0)
    reference_clips = {}; reference_protocol = None
    report = dict(b_complete=False, b_full_validation=False, b_detection_map_computed=False,
                  checkpoint_handling='strict model-only state load; no optimizer resume and no training controller',
                  plan_sha256=HashFile(args.plan), indices=indices, arms=[])
    start = time.monotonic()
    try:
        for arm in plan['arms']:
            arm_started = time.monotonic(); output = destination/arm['name']; output.mkdir()
            cfg = EasyDict(LoadConfig(arm['config'])); CheckRunConfig(cfg)
            # Remove the bounded-training controller configuration; this entry never invokes it.
            cfg.pop('root_diagnostic', None)
            for key in ('tasks','voxel_size','out_size_factor'):
                cfg.model[key] = cfg.common_variables[key]
            cfg.model.point_cloud_range = cfg.common_variables.pc_range
            if cfg.model.get('x4_detection_source') != 'enhanced_e4':
                raise ValueError('Expected fixed F / enhanced_e4 architecture for all arms')
            cfg.model.occ.b_collect_occ_stage_statistics = False
            cfg.model.occ.b_collect_root_diagnostics = True
            protocol = dict(pc_range=list(cfg.common_variables.pc_range), voxel_size=list(cfg.common_variables.voxel_size),
                            val_dataset=dict(cfg.data.val_dataset), post_processing=dict(cfg.model.post_processing))
            if reference_protocol is None:
                reference_protocol = protocol
            elif json.dumps(reference_protocol, sort_keys=True) != json.dumps(protocol, sort_keys=True):
                raise ValueError('Arms must share validation data/history/ROI/postprocessing')
            dataset = WaymoDatasetMulti(cfg.data.val_dataset, cfg.common_variables)
            model = MultiMissionModel(cfg.model).to(args.device).eval()
            saved = torch.load(arm['checkpoint'], map_location='cpu', weights_only=False)
            if 'state_dict' not in saved:
                raise ValueError('Expected checkpoint state_dict; no implicit format conversion')
            loaded = model.load_state_dict(saved['state_dict'], strict=True)
            arm_report = dict(name=arm['name'], config=arm['config'], config_sha256=HashFile(arm['config']),
                              checkpoint=arm['checkpoint'], checkpoint_sha256=HashFile(arm['checkpoint']),
                              checkpoint_epoch=saved.get('meta', {}).get('epoch'), checkpoint_step=saved.get('meta', {}).get('iter'),
                              strict_missing_keys=list(loaded.missing_keys), strict_unexpected_keys=list(loaded.unexpected_keys),
                              root_input_normalization=model.occ_head.root_input_normalization,
                              delete_threshold=model.occ_head.delete_threshold, b_heavy_stage_statistics=False,
                              effective_model_config_sha256=hashlib.sha256(json.dumps(cfg.model, sort_keys=True).encode()).hexdigest(),
                              frames=[])
            del saved
            terrain_frames=[]; occ_frames=[]
            for index in indices:
                frame_started=time.monotonic(); sample=dataset[index]
                batch=dataset.CollateBatch([sample]); target=batch['frames'][-1]
                token=target['token'][0]
                clip=dict(token=token, token_seq=sample['token_seq'],
                          full_frame_indices=sample['full_frame_indices'].tolist(),
                          history_interval=int(sample['history_interval']), warmup_len=int(sample['warmup_len']),
                          train_len=int(sample['train_len']), b_history_padded=bool(sample['b_history_padded']))
                if index in reference_clips and clip != reference_clips[index]:
                    raise ValueError(f'Validation clip changed between arms at index {index}')
                reference_clips[index]=clip
                with torch.no_grad():
                    memory=None
                    for frame in batch['frames'][:-1]:
                        memory=model(ToDevice(frame,args.device),memory,b_memory_only=True)
                    prediction=model(ToDevice(target,args.device),memory)[token]
                terrain=BuildTerrainFrameStatistics(prediction['terrain'],target,0,model.terrain_criterion)
                occ=prediction['occ']['metrics']; terrain_frames.append(terrain); occ_frames.append(occ)
                det=prediction['det']; scores=det['scores']; labels=det['label_preds']
                detection=dict(post_nms_count=len(scores), post_nms_class_counts=[int((labels==c).sum()) for c in range(3)],
                               score_ge_015_class_counts=[int(((labels==c)&(scores>=.15)).sum()) for c in range(3)],
                               score_mean=float(scores.mean()) if len(scores) else None,
                               score_max=float(scores.max()) if len(scores) else None,
                               scope='post-NMS output counts/scores only; not detection AP')
                frame_report=dict(index=index,clip=clip,terrain=SummarizeTerrainMetrics([terrain]),
                                  occ=SummarizeMetrics([occ]),detection=detection,
                                  terrain_raw_counts=terrain,occ_raw_counts=occ,
                                  elapsed_seconds_including_diagnostics=time.monotonic()-frame_started)
                arm_report['frames'].append(frame_report)
                Write(output/f'val_{index}.json',frame_report)
                if index == 100:
                    info=next(item for item in dataset.infos if item['token']==token)
                    PlotComparison(prediction,target,info,cfg,output/'val_100_three_task.png','val',True)
                    torch.save(prediction,output/'val_100_prediction.pt')
                del prediction,memory,batch,target
            arm_report.update(terrain=SummarizeTerrainMetrics(terrain_frames),occ=SummarizeMetrics(occ_frames),
                              elapsed_seconds=time.monotonic()-arm_started)
            Write(output/'summary.json',arm_report); report['arms'].append(arm_report)
            Write(destination/'summary.json',report)
            del model,dataset; torch.cuda.empty_cache()
        report['b_complete']=True
    except BaseException as error:
        report['error']=repr(error); raise
    finally:
        report['elapsed_seconds']=time.monotonic()-start
        Write(destination/'summary.json',report)
    Write(destination/'success.json',dict(b_complete=True,b_full_validation=False,
          b_detection_map_computed=False,models=len(report['arms']),targets_per_model=len(indices),
          elapsed_seconds=report['elapsed_seconds']))


if __name__=='__main__':
    Main()
