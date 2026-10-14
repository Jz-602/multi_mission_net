"""Read-only fixed-clip C4/E4/F4 statistics; no heads, optimizer or training."""
import argparse
import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def Distribution(values):
    import torch
    if not values.numel(): return {'count': 0}
    values = values.float()
    return dict(count=values.numel(), mean=float(values.mean()),
                p50=float(torch.quantile(values, .5)), p95=float(torch.quantile(values, .95)), maximum=float(values.max()))


def Compare(left, right):
    """Row-wise right-minus-left; relative norms use nonzero left rows only."""
    import torch
    a = torch.linalg.vector_norm(left, dim=1)
    b = torch.linalg.vector_norm(right, dim=1)
    delta = torch.linalg.vector_norm(right-left, dim=1)
    b_nonzero = a > 1e-12
    b_cosine = b_nonzero & (b > 1e-12)
    cosine = (left[b_cosine]*right[b_cosine]).sum(1)/(a[b_cosine]*b[b_cosine])
    return dict(left_l2=Distribution(a), right_l2=Distribution(b), delta_l2=Distribution(delta),
                relative_delta_l2=Distribution(delta[b_nonzero]/a[b_nonzero]), cosine=Distribution(cosine),
                left_zero_norm_rows=int((~b_nonzero).sum()), cosine_valid_rows=int(b_cosine.sum()))


def Summarize(current, enhanced, fused, aligned, pc_range, voxel_size):
    import torch
    from occ_head import MatchRows
    from multi_mission_model_memory import MEMORY_ALPHA_EQUAL, MEMORY_ALPHA_FAR, MEMORY_ALPHA_NEAR, MEMORY_DISTANCE_MARGIN
    # All captured sparse coordinates are current-target LiDAR BZYX; features are [N,C4].
    coords, c4 = current['coords'], current['feats']
    assert torch.equal(coords, enhanced['coords'])
    rows = MatchRows(coords, fused['coords'], fused['spatial_shape']); assert bool((rows >= 0).all())
    e4, f4 = enhanced['feats'], fused['feats'][rows]
    if aligned is None:
        memory_rows = torch.full((len(coords),), -1, dtype=torch.long)
    else:
        memory_rows = MatchRows(coords, aligned['coords'], fused['spatial_shape'])
    b_overlap = memory_rows >= 0
    alpha = torch.ones(len(coords), dtype=c4.dtype)
    matched_memory = torch.zeros_like(c4)
    if bool(b_overlap.any()):
        xyz = torch.tensor(pc_range[:3])+(coords[:, [3, 2, 1]].float()+.5)*torch.tensor(voxel_size)*16
        current_distances = current.get('distances', torch.linalg.vector_norm(xyz, dim=1))
        memory_distances = aligned['distances'][memory_rows[b_overlap]]
        overlap_distances = current_distances[b_overlap]
        alpha[b_overlap] = torch.where(overlap_distances+MEMORY_DISTANCE_MARGIN < memory_distances, MEMORY_ALPHA_NEAR,
                            torch.where(overlap_distances > memory_distances+MEMORY_DISTANCE_MARGIN, MEMORY_ALPHA_FAR, MEMORY_ALPHA_EQUAL))
        matched_memory[b_overlap] = aligned['feats'][memory_rows[b_overlap]]
    expected = alpha[:, None]*e4+(1-alpha[:, None])*matched_memory
    torch.testing.assert_close(f4, expected, rtol=1e-5, atol=1e-6)
    full_to_current = MatchRows(fused['coords'], coords, fused['spatial_shape'])
    b_history_only = full_to_current < 0
    assert len(fused['coords']) == len(coords)+int(b_history_only.sum())
    groups = {'all_current': torch.ones(len(coords), dtype=torch.bool), 'current_only': ~b_overlap, 'overlap': b_overlap}
    groups.update({f'alpha_{value:.1f}': b_overlap & (alpha == value) for value in (MEMORY_ALPHA_FAR, MEMORY_ALPHA_EQUAL, MEMORY_ALPHA_NEAR)})
    result = dict(current_count=len(coords), aligned_memory_count=0 if aligned is None else len(aligned['coords']),
                  full_f4_count=len(fused['coords']), overlap_count=int(b_overlap.sum()),
                  history_only_count=int(b_history_only.sum()), current_only_count=int((~b_overlap).sum()),
                  full_f4_l2=Distribution(torch.linalg.vector_norm(fused['feats'], dim=1)),
                  history_only_f4_l2=Distribution(torch.linalg.vector_norm(fused['feats'][b_history_only], dim=1)),
                  mixture_reconstruction_max_abs=float((f4-expected).abs().max()), groups={})
    for name, mask in groups.items():
        result['groups'][name] = dict(rows=int(mask.sum()), c4_to_e4=Compare(c4[mask], e4[mask]),
                                     e4_to_current_f4=Compare(e4[mask], f4[mask]), c4_to_current_f4=Compare(c4[mask], f4[mask]))
    return result


def CPUCheck():
    import torch
    from easydict import EasyDict
    from spconv.pytorch import SparseConvTensor
    from multi_mission_model_memory_x4 import MultiMissionModel
    model = MultiMissionModel.__new__(MultiMissionModel); torch.nn.Module.__init__(model)
    model.config = EasyDict(point_cloud_range=[0,0,0,6.4,6.4,6.4], voxel_size=[.05]*3)
    coords = torch.tensor([[0,1,1,1],[0,2,2,2],[0,3,3,3],[0,4,4,4]], dtype=torch.int32)
    current = dict(coords=coords, feats=torch.arange(8).reshape(4,2).float()+1)
    enhanced = dict(coords=coords, feats=current['feats']+.1)
    distances = torch.linalg.vector_norm((coords[:3, [3,2,1]].float()+.5)*.8,dim=1)
    aligned = dict(coords=torch.cat((coords[:3],torch.tensor([[0,5,5,5]], dtype=torch.int32))),
                   feats=torch.full((4,2),10.), distances=torch.cat((distances+torch.tensor([-2.,0.,2.]),torch.tensor([1.]))))
    sparse = SparseConvTensor(enhanced['feats'],coords,[8]*3,1)
    fused = model.MemoryUpdate(sparse,aligned,1,feature_stride=16)
    result = Summarize(current,enhanced,fused,aligned,model.config.point_cloud_range,model.config.voxel_size)
    assert [result['groups'][f'alpha_{alpha:.1f}']['rows'] for alpha in (.2,.5,.7)] == [1,1,1]
    assert result['current_only_count']==1 and result['history_only_count']==1
    assert result['mixture_reconstruction_max_abs'] < 1e-6
    # Explicit missing current-row fixture must be rejected, not silently zero-filled.
    bad = dict(fused,coords=fused['coords'][1:],feats=fused['feats'][1:])
    try: Summarize(current,enhanced,bad,aligned,model.config.point_cloud_range,model.config.voxel_size)
    except AssertionError: pass
    else: raise AssertionError('Missing S4 accepted')
    return dict(b_cpu_pass=True,b_gpu_executed=False,checks=['actual MemoryUpdate alpha .2/.5/.7/current-only reconstruction','history-only support count','missing current support rejected','row norm/delta/cosine statistics'],synthetic_result=result)


def Run(args):
    import torch
    from easydict import EasyDict
    from LYF_train_ODSS import CfgFromYamlFile, SetRandomSeed
    from WaymoDataset_multi import WaymoDatasetMulti
    from multi_mission_model_memory_x4 import MultiMissionModel
    from trainer.trainer.three_task_runtime import MoveToDevice
    from run_options import CheckRunConfig, HashFile
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cfg=CfgFromYamlFile(args.config,EasyDict());CheckRunConfig(cfg)
    for key in ('tasks','out_size_factor','voxel_size'):cfg.model[key]=cfg.common_variables[key]
    cfg.model.point_cloud_range=cfg.common_variables.pc_range
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['meta']['epoch']==0 and checkpoint['meta']['iter']==0
    comparison=torch.load(args.compare_checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['state_dict'].keys()==comparison['state_dict'].keys()
    assert all(torch.equal(value,comparison['state_dict'][key]) for key,value in checkpoint['state_dict'].items()), 'A/B epoch0 weights differ; one traversal would not represent both'
    del comparison
    model=MultiMissionModel(cfg.model).cuda().eval();model.load_state_dict(checkpoint['state_dict'],strict=True)
    del checkpoint
    torch.set_num_threads(4)
    captured={};b_target=False
    def Current(_module,_inputs,result):
        if b_target:captured['current']=result
    original_update=model.MemoryUpdate
    def Update(self,enhanced,aligned,batch_size,feature_stride=8):
        fused=original_update(enhanced,aligned,batch_size,feature_stride=feature_stride)
        if b_target:
            # Same device/dtype as production; avoid CPU rounding moving a distance across the 1m alpha threshold.
            xyz=enhanced.features.new_tensor(cfg.model.point_cloud_range[:3])+(enhanced.indices[:,[3,2,1]].to(enhanced.features.dtype)+.5)*(enhanced.features.new_tensor(cfg.model.voxel_size)*feature_stride)
            captured.update(enhanced=enhanced,aligned=aligned,fused=fused,current_distances=torch.linalg.norm(xyz,dim=1).cpu())
        return fused
    handle=model.encoder_stage4.register_forward_hook(Current);model.MemoryUpdate=types.MethodType(Update,model)
    report=dict(b_complete=False,b_read_only=True,checkpoint=args.checkpoint,compare_checkpoint=args.compare_checkpoint,
                checkpoint_sha256=HashFile(args.checkpoint),compare_checkpoint_sha256=HashFile(args.compare_checkpoint),
                b_epoch0_states_equal=True,config=args.config,clips=[],
                limits=['Only train10 and val100; no population claim or AP conclusion.','No detection/terrain/OCC head execution; isolates actual memory feature computation, not decoded detection causality.','History eval and no_grad; exact production forward path invoked with memory_only.','Norm/cosine differences alone do not select a repair; compare attention E4-C4 versus mixture F4[S4]-E4.'])
    started=time.monotonic()
    try:
        for split,index in [('train',10),('val',100)]:
            SetRandomSeed(0,0)
            dataset=WaymoDatasetMulti(cfg.data[f'{split}_dataset'],cfg.common_variables)
            sample=dataset[index];batch=dataset.CollateBatch([sample]);del dataset
            metadata=json.loads((Path(args.reference_images)/f'{split}_{index:03d}.json').read_text())
            assert metadata['token']==sample['token_seq'][-1]
            assert metadata['temporal_clip']['token_seq']==sample['token_seq'], 'Clip differs from saved A0 visualization'
            memory=None;b_target=False
            with torch.no_grad():
                for frame in batch['frames'][:-1]:memory=model(MoveToDevice(frame,'cuda'),memory,b_memory_only=True)
                b_target=True;model(MoveToDevice(batch['frames'][-1],'cuda'),memory,b_memory_only=True)
            def SparseCPU(value):return dict(coords=value.indices.cpu(),feats=value.features.cpu())
            current=SparseCPU(captured['current']);current['distances']=captured['current_distances'];enhanced=SparseCPU(captured['enhanced'])
            fused={key:value.cpu() if torch.is_tensor(value) else value for key,value in captured['fused'].items()}
            aligned=None if captured['aligned'] is None else {key:value.cpu() for key,value in captured['aligned'].items()}
            stats=Summarize(current,enhanced,fused,aligned,cfg.model.point_cloud_range,cfg.model.voxel_size)
            report['clips'].append(dict(split=split,index=index,token=metadata['token'],temporal_clip=metadata['temporal_clip'],statistics=stats))
            (output/f'{split}_{index:03d}.json').write_text(json.dumps(report['clips'][-1],indent=2,allow_nan=False)+'\n')
            captured.clear();del memory,batch,current,enhanced,fused,aligned;torch.cuda.empty_cache()
        report['b_complete']=True
    finally:
        handle.remove();report['elapsed_seconds']=time.monotonic()-started
        (output/'feature_mixing.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


def Main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--cpu-check',action='store_true')
    parser.add_argument('--config',default=str(ROOT/'configs/server_x4_screen_a.yaml'))
    base=ROOT/'log/server_runs_x4_detection_screen_20261008'
    parser.add_argument('--checkpoint',default=str(base/'a/checkpoints/epoch_0.pth'))
    parser.add_argument('--compare-checkpoint',default=str(base/'b/checkpoints/epoch_0.pth'))
    parser.add_argument('--reference-images',default=str(base/'a/images/epoch_00'))
    parser.add_argument('--output',required=True);args=parser.parse_args()
    if args.run:
        assert not args.cpu_check;Run(args)
    else:
        Path(args.output).write_text(json.dumps(CPUCheck(),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':Main()
