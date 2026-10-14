"""导出同一车体系的三任务预测/真值对比；可用于训练样本诊断和验证样本。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch
from easydict import EasyDict

from LYF_train_ODSS import CfgFromYamlFile, SetRandomSeed
from WaymoDatasetV2 import WaymoDataset
from WaymoDataset_multi import WaymoDatasetMulti
from multi_mission_model import MultiMissionModel as SingleModel
from multi_mission_model_memory import MultiMissionModel as TemporalModel
from run_options import CheckRunConfig
from terrain_visualization import SaveTerrainDetails

COLORS = np.array([[.16, .65, .88], [.94, .51, .19], [.26, .70, .39]])


def ToDevice(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: ToDevice(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [ToDevice(v, device) for v in value]
    return value


def DrawBoxes(ax, boxes, labels, scores=None):
    for index, (box, category) in enumerate(zip(boxes, labels)):
        x, y, z, length, width, height, yaw = box
        corners = np.array([[-1,-1],[1,-1],[1,1],[-1,1],[-1,-1]]) * [length/2,width/2]
        rot = np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        corners = corners @ rot.T + [x,y]
        ax.plot(corners[:,0],corners[:,1],color=COLORS[int(category)%3],linewidth=1.2)
        if scores is not None:
            ax.text(x,y,f'{scores[index]:.2f}',fontsize=5,color='black')


def PlotComparison(output, batch, info, cfg, destination, split, b_temporal):
    pc_range = cfg.common_variables.pc_range
    xmin,ymin,zmin,xmax,ymax,zmax = pc_range
    points = batch['points'].cpu().numpy()[:,1:4]
    points = points[((points >= np.array(pc_range[:3])) & (points < np.array(pc_range[3:]))).all(1)]
    if len(points)>30000:
        points = points[np.linspace(0,len(points)-1,30000).astype(int)]
    fig, axes = plt.subplots(3,2,figsize=(13,16),layout='constrained')
    for ax in axes[0]:
        ax.scatter(points[:,0],points[:,1],s=.12,c='#aaaaaa',alpha=.4,rasterized=True)
        ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),xlabel='x (m)',ylabel='y (m)',aspect='equal')
    type_to_label={1:0,2:1,4:2}
    b_completed = bool(cfg.data[f'{split}_dataset'].get('b_use_completed_detection_labels', False))
    gt=[v for v in info['objects'] if int(v['type']) in type_to_label
        and (int(v.get('num_points_in_top_lidar',1))>0 or (b_completed and v.get('b_is_supplemental', False)))
        and len(v['box'])>=7]
    # 当前车体系框[N,7]；与验证一致，按三维中心ROI筛选，保留跨边界的有效框。
    gt_boxes=np.asarray([v['box'][:7] for v in gt],dtype=np.float32).reshape(-1,7)
    b_gt_in_roi=np.isfinite(gt_boxes).all(1)&((gt_boxes[:,:3]>=np.asarray(pc_range[:3]))&(gt_boxes[:,:3]<np.asarray(pc_range[3:]))).all(1)
    gt_labels=np.asarray([type_to_label[int(v['type'])] for v in gt],dtype=np.int64)
    DrawBoxes(axes[0,0],gt_boxes[b_gt_in_roi],gt_labels[b_gt_in_roi])
    det=output['det'];keep=det['scores'].numpy()>=.15
    DrawBoxes(axes[0,1],det['box3d_lidar'].numpy()[keep],det['label_preds'].numpy()[keep],det['scores'].numpy()[keep])
    axes[0,0].set_title('Detection ground truth (original + completed)\n+ current LiDAR'
                        if b_completed else 'Detection ground truth + current LiDAR')
    axes[0,1].set_title('Detection prediction (score >= 0.15)')
    gt_height=batch['terrain_height'][0,0].cpu().numpy()
    valid=batch['terrain_valid_mask'][0,0].cpu().numpy()
    pred_height=output['terrain']['height'].squeeze().numpy()
    b_has_validity = 'validity_probability' in output['terrain']
    if 'terrain_ignore_mask' in batch:
        valid = valid & ~batch['terrain_ignore_mask'][0,0].cpu().numpy()
    if b_has_validity:
        # Same fixed ROI z limits across samples/checkpoints; prediction never uses GT mask.
        lo, hi = zmin, zmax
        pred_mask = output['terrain']['validity_probability'].squeeze().numpy() >= .5
        displayed_prediction = np.where(pred_mask, pred_height, np.nan)
    else:
        lo,hi=np.percentile(gt_height[valid],[2,98]) if valid.any() else (-3,1)
        displayed_prediction = pred_height
    pred_title = 'Terrain prediction (predicted valid mask)' if b_has_validity else 'Terrain prediction'
    for ax,grid,title in zip(axes[1],[np.where(valid,gt_height,np.nan),displayed_prediction],['Terrain ground truth (valid cells)',pred_title]):
        # Terrain row=x descending/col=y descending -> imshow row=y descending/col=x increasing.
        display=grid.T[:,::-1]
        im=ax.imshow(display,extent=[xmin,xmax,ymin,ymax],origin='upper',vmin=lo,vmax=hi,cmap='terrain')
        ax.set(title=title,xlabel='x (m)',ylabel='y (m)')
        fig.colorbar(im,ax=ax,shrink=.8,label='height (m)')
    occ=output['occ'];pred_xyz=occ['voxel_indices'].numpy();pred_label=occ['labels'].numpy()
    gt_xyz=batch['occ_voxel_indices'][0].cpu().numpy();gt_label=batch['occ_labels'][0].cpu().numpy()
    for index,(xyz,label,title) in enumerate([(gt_xyz,gt_label,'Static OCC ground truth'),(pred_xyz,pred_label,'Static OCC prediction')]):
        axes[2,index].remove();ax=fig.add_subplot(3,2,5+index,projection='3d')
        keep=(label>0)&(label<4);xyz=xyz[keep];label=label[keep]
        if len(xyz)>60000:
            rows=np.linspace(0,len(xyz)-1,60000).astype(int);xyz=xyz[rows];label=label[rows]
        physical=np.asarray(pc_range[:3])+(xyz+.5)*.2
        ax.scatter(physical[:,0],physical[:,1],physical[:,2],s=.6,c=COLORS[label-1],alpha=.85,rasterized=True)
        ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),zlim=(zmin,zmax),title=title,xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
        ax.set_box_aspect((xmax-xmin,ymax-ymin,20));ax.view_init(elev=32,azim=-60)
    b_has_history = b_temporal and int(cfg.data[f'{split}_dataset'].get('warmup_len', 0)) > 0
    fig.suptitle(f'{"Temporal" if b_has_history else "Single frame"} three-task model | {split} sample\n{info["token"]}\nOCC: blue=pole, orange=general obstacle, green=vegetation; empty omitted',fontsize=12)
    destination.parent.mkdir(parents=True,exist_ok=True);fig.savefig(destination,dpi=160);plt.close(fig)
    if b_has_validity:
        SaveTerrainDetails(output['terrain'], batch, pc_range,
                           destination.with_name(destination.stem + '_terrain_details.png'),
                           {'token': info['token'], 'split': split, 'b_temporal': b_temporal})


def PlotLosses(logfile, destination):
    rows=[json.loads(line) for line in Path(logfile).read_text().splitlines() if line.strip()]
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for ax,task in zip(axes,['det','terrain','occ']):
        values=[];steps=[]
        for row in rows:
            value=row.get(f'{task}/loss',row.get(f'{task}_loss'))
            if value is None and isinstance(row.get('tasks'),dict):value=row['tasks'].get(task,{}).get('loss')
            if value is None and isinstance(row.get('tasks'),list):
                value=next((entry.get('loss') for entry in row['tasks'] if entry.get('task')==[task]),None)
            if value is not None:values.append(value);steps.append(row.get('step',row.get('global_step',row.get('iter',len(steps)+1))))
        ax.plot(steps,values);ax.set(title=f'{task}: raw task loss',xlabel='optimizer updates',ylabel='loss');ax.grid(alpha=.2)
    fig.savefig(destination,dpi=160);plt.close(fig)


def Main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--split',choices=['train','val'],default='train');parser.add_argument('--index',type=int,default=10)
    parser.add_argument('--temporal',action='store_true');parser.add_argument('--loss-log');args=parser.parse_args()
    torch.set_num_threads(4);SetRandomSeed(0,0)
    cfg=CfgFromYamlFile(args.config,EasyDict());CheckRunConfig(cfg)
    data_cfg=cfg.data[f'{args.split}_dataset'];cls=WaymoDatasetMulti if args.temporal else WaymoDataset
    dataset=cls(data_cfg,cfg.common_variables);sample=dataset[args.index];batch=dataset.CollateBatch([sample])
    for key in ['tasks','out_size_factor','voxel_size']:cfg.model[key]=cfg.common_variables[key]
    cfg.model.point_cloud_range=cfg.common_variables.pc_range
    model_class = TemporalModel if args.temporal else SingleModel
    if args.temporal and 'x4_detection_source' in cfg.model:
        from multi_mission_model_memory_x4 import MultiMissionModel as X4Model
        model_class = X4Model
    model = model_class(cfg.model).cuda()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    with torch.no_grad():
        if args.temporal:
            memory=None
            for frame in batch['frames'][:-1]:memory=model(ToDevice(frame,'cuda'),memory,b_memory_only=True)
            target=batch['frames'][-1];result=model(ToDevice(target,'cuda'),memory)
        else:
            target=batch;result=model(ToDevice(target,'cuda'))
    token=target['token'][0];info=next(v for v in dataset.infos if v['token']==token)
    output=Path(args.output);PlotComparison(result[token],target,info,cfg,output,args.split,args.temporal)
    torch.save(result,output.with_suffix('.pt'))
    metadata={'token':token,'split':args.split,'checkpoint':args.checkpoint,'occ':result[token]['occ'].get('metrics',{})}
    metadata['b_use_completed_detection_labels'] = bool(data_cfg.get('b_use_completed_detection_labels', False))
    metadata['b_use_time_feature'] = bool(data_cfg.get('b_use_time_feature', False))
    if args.temporal:
        # 时间顺序为历史帧到目标帧；记录实际推理片段，便于跨检查点核对固定样本。
        metadata['temporal_clip']={
            'token_seq':sample['token_seq'],
            'full_frame_indices':sample['full_frame_indices'].tolist(),
            'history_interval':int(sample['history_interval']),
            'b_history_padded':bool(sample['b_history_padded']),
            'warmup_len':int(sample['warmup_len']),
            'train_len':int(sample['train_len']),
        }
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    if args.loss_log:PlotLosses(args.loss_log,output.with_name('loss_curves.png'))
    print(output)

if __name__=='__main__':Main()
