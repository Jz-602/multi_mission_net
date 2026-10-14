"""由真实F48产物生成唯一主报告；CPU离线统计，不启动或修改实验。"""
import argparse
import csv
import datetime
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import statistics
from urllib.parse import unquote, urlsplit

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT/'log/server_runs_x4_selected_long48_20261010/f'
PREFIX = PROJECT/'log/server_runs_x4_detection_screen_20261008/f'
PREP = PROJECT/'log/server_preparation/x4_selected_long48_20261010'
OUTPUT = PROJECT.parent/'实验报告/2026-10-10起_选定X4时序48轮与根筛选消融'
EPOCHS = [*range(6), 6, 10, 15, 20, 25, 30, 35, *range(40, 49)]
LOSS_FIELDS = {'det': ('loss', 'hm_loss', 'loc_loss', 'iou_loss', 'iou_reg_loss'),
               'terrain': ('loss', 'terrain_main_loss', 'terrain_gradient_loss'),
               'occ': ('loss', 'class_loss', 'root_loss', 'CELoss', 'DiceLossV1', 'LovLoss')}


def Read(path):
    return json.loads(path.read_text())


def SHA(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def Write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def CSV(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def Table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def Link(path, label, output):
    return '<a href="'+html.escape(os.path.relpath(path,output),quote=True)+'">'+html.escape(label)+'</a>'


def MetricsRow(name, record):
    m=record['metrics'];d,t,o=m['det'],m['terrain'],m['occ']
    return [name, f"{d['map_l2']*100:.4f}",f"{d['maph_l2']*100:.4f}",f"{t['mae_m']:.6f}",f"{t['rmse_m']:.6f}",f"{o['occupancy_iou']*100:.4f}",f"{o['semantic_miou']*100:.4f}"]


def FlattenMetrics(record):
    row={'epoch':record['epoch']}
    for task,values in record['metrics'].items():
        for key,value in values.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool):row[task+'_'+key]=value
            elif isinstance(value,list) and all(isinstance(v,(int,float)) for v in value):
                row.update({task+'_'+key+'_'+str(i):v for i,v in enumerate(value)})
    return row


def LossRows():
    buckets={epoch:{} for epoch in range(1,49)};counts={epoch:0 for epoch in buckets};step=0
    with (RUN/'train_metrics.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);step+=1
            if row['iter']!=step or row['epoch']!=(step-1)//1481:raise ValueError('Non-contiguous actual training log')
            epoch=row['epoch']+1;counts[epoch]+=1
            values={'total_loss':row['total_loss'],'learning_rate':row['learning_rate']}
            for name,task in zip(('det','terrain','occ'),row['tasks']):
                if task['task']!=[name]:raise ValueError('Unexpected task mapping')
                values.update({name+'_'+key:task[key] for key in LOSS_FIELDS[name]})
            for key,value in values.items():
                if not math.isfinite(value):raise ValueError('Nonfinite raw loss')
                buckets[epoch].setdefault(key,[]).append(value)
    if step!=71088 or set(counts.values())!={1481}:raise ValueError('Not a complete 48-epoch run')
    rows=[]
    for epoch,metrics in buckets.items():
        row={'epoch':epoch,'updates':counts[epoch],'first_iter':(epoch-1)*1481+1,'last_iter':epoch*1481}
        for key,values in metrics.items():
            row[key+'_mean']=statistics.mean(values);row[key+'_median']=statistics.median(values)
            if key=='learning_rate':row['lr_first']=values[0];row['lr_last']=values[-1]
        rows.append(row)
    return rows


def Plot(output, records, losses, baseline):
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font_path)
    font_name=font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font_name],'axes.unicode_minus':False})
    fig,axes=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
    fields=[('det','map_l2','检测 mAP L2 (%)',100),('det','maph_l2','检测 mAPH L2 (%)',100),
            ('terrain','mae_m','地形加权 MAE (m)',1),('terrain','rmse_m','地形加权 RMSE (m)',1),
            ('occ','occupancy_iou','OCC 占据 IoU (%)',100),('occ','semantic_miou','OCC 语义 mIoU (%)',100)]
    for ax,(task,key,title,scale) in zip(axes.flat,fields):
        ax.plot([r['epoch'] for r in records],[r['metrics'][task][key]*scale for r in records],'.-',label='F：完整验证')
        ax.axhline(baseline['metrics'][task][key]*scale,color='#777',ls='--',label='S48 同口径重评 B0')
        ax.axvline(25,color='#b82222',alpha=.45);ax.set(title=title,xlabel='新增时序训练轮次');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('同一检查点三任务指标；未拼接各任务最佳值');fig.savefig(output/'validation_curves.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
    for ax,key,title in zip(axes.flat,('det_loss','terrain_loss','occ_loss','det_hm_loss','det_loc_loss','occ_root_loss'),
                            ('检测 raw loss','地形 raw loss','OCC raw loss','检测分类 hm_loss','检测回归 loc_loss（另存 IoU loss）','根删除二分类 root_loss')):
        ax.plot([r['epoch'] for r in losses],[r[key+'_mean'] for r in losses],label='逐步算术均值')
        ax.plot([r['epoch'] for r in losses],[r[key+'_median'] for r in losses],ls='--',label='逐步中位数')
        ax.set(title=title,xlabel='新增时序训练轮次');ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('真实每轮1481步：三任务原始损失、分类/回归与根监督');fig.savefig(output/'raw_loss_curves.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for key,label in [('occupied_root_recall','正根召回'),('candidate_gt_coverage','非空GT候选覆盖'),('occupancy_iou','最终占据IoU')]:
        axes[0].plot([r['epoch'] for r in records],[r['metrics']['occ'][key]*100 for r in records],'.-',label=label)
    axes[0].axvline(25,color='#b82222',ls='--');axes[0].set(title='根筛选失稳与覆盖损失',xlabel='时序轮次',ylabel='%');axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot([r['epoch'] for r in losses],[r['learning_rate_mean'] for r in losses]);axes[1].set(title='原48轮 OneCycle 学习率轨迹',xlabel='时序轮次',ylabel='每轮 mean LR');axes[1].grid(alpha=.2)
    fig.savefig(output/'root_instability_and_lr.png',dpi=140);plt.close(fig)


def RootAblation(output):
    stats_root=PROJECT/'log/server_occ_stage_statistics_20261011'
    perf_root=PROJECT/'log/server_inference_performance_20261011'
    evidence={'b_occ_pair_accepted':False,'b_performance_accepted':False,'b_independent_root_off48_accepted':False}
    body='<h2>4. 时序OCC根删除开／关小型消融</h2>'
    accepted=[PREP/'f48_root_on_acceptance.json',PREP/'f48_same_weight_root_off_acceptance.json']
    if not all(p.exists() and Read(p).get('b_pass') is True for p in accepted):
        return body+'<p class="note">同权重根开关完整验证仍待两组验收；不填入未经验收的数值。独立无根48轮尚待用户决定。</p>',evidence
    groups=['f48_root_on','f48_same_weight_root_off']
    summaries=[Read(stats_root/g/'summary.json') for g in groups]
    source_sha=Read(PREP/'final_cpu_review.json')['checkpoint']['sha256']
    for group,summary,acceptance in zip(groups,summaries,accepted):
        if (summary['frame_count']!=5973 or not summary['b_complete'] or Read(acceptance)['checkpoint_sha256']!=source_sha
                or not Read(acceptance).get('b_per_frame_and_global_counts_verified')):raise ValueError('Unaccepted full root-switch summary')
    on,off=summaries
    if not Read(accepted[1]).get('same_weight_pre_filter_geometry_and_real_input_counts_equal'):raise ValueError('Same-root counterfactual not verified')
    for key in ['gt','input_leaf','input_gt','history_leaf','history_gt','history_current_leaf','history_current_gt','root','l0','l0_gt','lplus','lplus_gt']:
        if on['counts'][key]!=off['counts'][key]:raise ValueError('Same-weight pre-filter geometry changed: '+key)
    counts=[];ratios=[]
    for group,summary in zip(groups,summaries):
        for key,value in summary['counts'].items():counts.append({'group':group,'metric':key,'value':json.dumps(value) if isinstance(value,(dict,list)) else value})
        for key,value in summary['ratios'].items():
            if isinstance(value,dict):ratios.append({'group':group,'metric':key,**value})
            elif isinstance(value,(int,float)):ratios.append({'group':group,'metric':key,'value':value})
            else:ratios.append({'group':group,'metric':key,'value':json.dumps(value)})
    CSV(output/'root_switch_global_counts.csv',counts);CSV(output/'root_switch_ratios.csv',ratios)
    fate_rows=[{'group':g,**{f'fate_{i+1}_per100':v for i,v in enumerate(s['gt_fates_per_100'])},**s['fate4_semantic_split_per_100']} for g,s in zip(groups,summaries)]
    CSV(output/'root_switch_gt_fates_per100.csv',fate_rows)
    labels=['不删根也不可达','因删根失去覆盖','候选已覆盖却判为空','非空位置与语义正确','非空位置正确、语义错误']
    colors=['#91a5b6','#d66b55','#edbc61','#4c9a85','#8a76b3']
    fig,ax=plt.subplots(figsize=(14,4),constrained_layout=True)
    left=[0.,0.]
    components=[[*s['gt_fates_per_100'][:3],s['fate4_semantic_split_per_100']['candidate_gt_nonempty_correct'],s['fate4_semantic_split_per_100']['candidate_gt_nonempty_wrong_nonempty']] for s in summaries]
    for i,(label,color) in enumerate(zip(labels,colors)):
        widths=[r[i] for r in components];ax.barh([0,1],widths,left=left,label=label,color=color)
        for row,width in enumerate(widths):
            if width>4:ax.text(left[row]+width/2,row,f'{width:.2f}',ha='center',va='center',fontsize=10)
            left[row]+=width
    ax.set(yticks=[0,1],yticklabels=['同F48：开启根删除','同F48：关闭根删除'],xlim=(0,100),xlabel='每100个有效非空GT的位置去向')
    ax.invert_yaxis();ax.legend(loc='lower left',bbox_to_anchor=(0,1.04),ncol=3,fontsize=9);fig.savefig(output/'root_switch_gt_fates.png',dpi=150);plt.close(fig)
    reduction=1-on['counts']['leaf']/off['counts']['leaf']
    body+='<p><b>同一F48权重、完整固定5973帧，两个条件均已验收。</b> 根删除仅作用于OCC，开关保留同一根分类器；不重新训练、不调整阈值。下列数字由全验证逐帧计数累加得到，非逐帧比例平均。稀疏标签缺失按已验收有效ROI的空处理，忽略/范围外位置不临时变成空。</p>'
    body+=Table(['F48同权重条件','占据IoU %','语义mIoU %','Precision %','Recall %','正根召回 %','GT候选覆盖 %','实际叶格数'],[[name,f"{s['ratios']['occupancy_iou']['value']*100:.6f}",f"{s['ratios']['semantic_miou']*100:.6f}"]+[f"{s['ratios'][k]['value']*100:.4f}" for k in ['occupancy_precision','occupancy_recall','occupied_root_recall','actual_gt_coverage']]+[f"{s['counts']['leaf']:,}"] for name,s in zip(['开启删除','关闭删除'],summaries)])
    body+=f'<p>开启删除使实际叶候选减少 <b>{reduction*100:.2f}%</b>；占据IoU相对关闭删除为 {(on["ratios"]["occupancy_iou"]["value"]-off["ratios"]["occupancy_iou"]["value"])*100:+.6f} 个百分点。这个极小差异不是显著提升结论。候选数量下降不能直接当作整网耗时或显存下降，资源收益见下方独立实测。</p>'
    stage_fields=[('gt','非空GT总数 G'),('input_leaf','当前点云0.2米格子 I'),('input_gt','当前点命中GT'),('history_current_gt','历史与当前观测命中GT'),('root','筛选前0.8米粗根 R0'),('kept_root','保留根 Rk'),('occupied_root_deleted','误删含GT根数'),('l0','原始根直接展开 L0'),('l0_gt','L0命中GT'),('lplus','不删根外扩潜在叶格 L+'),('lplus_gt','L+命中GT'),('expansion_added_leaf','不删根外扩增加候选'),('expansion_added_gt','不删根外扩新增覆盖GT'),('expanded_root','实际外扩去重根数'),('y3_raw','实际0.4米子格数'),('leaf','实际0.2米叶格 L'),('leaf_gt','候选覆盖非空GT'),('empty_candidate','候选内有效空格'),('candidate_gt_nonempty_pred_empty','候选内GT非空却判为空'),('candidate_gt_nonempty_wrong_nonempty','非空语义类判错'),('occupancy_tp','全范围占据TP'),('occupancy_fp','全范围占据FP'),('occupancy_fn','全范围占据FN')]
    body+=Table(['阶段原始计数','开启删除','关闭删除'],[[label,*[f"{s['counts'][key]:,}" for s in summaries]] for key,label in stage_fields])
    body+='<p>OCC流程：原始0.8米粗根→二分类根删除→保留根外扩一圈→0.4米八子格→0.2米八叶格→空/杆状/一般障碍物/植被分类。L0、L+是同一R0的几何反事实；L是真实生成候选。粗根有覆盖不等于最终分类为非空。根自身为空与外扩能帮助邻居GT是不同判断。</p>'
    body+='<img src="root_switch_gt_fates.png" alt="同权重根开关每100个非空GT去向">'
    body+=Table(['条件',*labels],[[name,*[f'{v:.4f}' for v in values]] for name,values in zip(['开启删除','关闭删除'],components)])
    body+='<p>五段和为100：最后两段合计为占据TP，语义错误仍计占据找回。FN含未生成候选的GT。OCC语义mIoU仅均值类1/2/3，空类单列；两组逐帧G、R0、L0、L+及真实输入计数完全相同。开启删除统计与原epoch48验证有约1e−8级浮点重算差异，不声称逐位相同。全部统计耗时不计入模型推理时间；外扩边界尝试次数未单独记录，不能由去重前后之差直接归为越界。</p>'
    body+=Table(['非空类别IoU % / 空类单列','开启删除','关闭删除'],[[label,*[f"{s['ratios']['class_iou'][i]['value']*100:.6f}" for s in summaries]] for i,label in enumerate(['空（不进语义mIoU）','杆状','一般障碍物','植被'])])
    body+='<p>'+Link(output/'root_switch_global_counts.csv','全部计数、4类混淆矩阵 CSV',output)+' · '+Link(output/'root_switch_ratios.csv','分子/分母与比率 CSV',output)+' · '+Link(output/'root_switch_gt_fates_per100.csv','每100GT原始数值',output)+'</p><ul>'+''.join('<li>'+Link(stats_root/g,'逐帧8rank JSONL / 汇总：'+g,output)+' · '+Link(a,'验收',output)+'</li>' for g,a in zip(groups,accepted))+'</ul>'
    evidence.update(b_occ_pair_accepted=True,occ_groups={g:s for g,s in zip(groups,summaries)},occ_acceptance={str(p):{'sha256':SHA(p),'value':Read(p)} for p in accepted},leaf_candidate_reduction=reduction)
    head_path=PREP/'root_operating_point_acceptance.json'
    if head_path.exists() and Read(head_path).get('b_pass'):
        head=Read(head_path);head_rows=[]
        for case in head['case_audits']:
            for pair in case['pairs']:head_rows.append({'case':case['case'],**pair})
        CSV(output/'root_head_crossed_diagnostic.csv',head_rows)
        body+='<h3>第25轮根失稳：已验收的定向归因</h3><p>9个真实clip缓存、15个特征/根头组合、3个固定案例：在相同缓存特征上换成第25轮根头，正根召回随之下降；第25轮特征配第20轮根头明显恢复。当前证据主要指向根头工作状态，而不是汇总计数错误；根头为何在优化中失稳尚未查明。交叉head只是诊断，没有执行叶格解码，不能据此宣称修复了OCC、完整验证或长期训练稳定性。</p>'
        body+=Table(['案例','特征轮次','根头轮次','正根召回 %','正根保留/总数'],[[r['case'],r['feature_epoch'],r['head_epoch'],f"{r['positive_recall']*100:.4f}",f"{r['positive_kept']}/{r['positive_count']}"] for r in head_rows])
        body+='<p>'+Link(head_path,'根头诊断独立验收与解释边界',output)+' · '+Link(output/'root_head_crossed_diagnostic.csv','诊断计数 CSV',output)+'</p>'
        evidence['root_head_diagnostic_acceptance']={'path':str(head_path),'sha256':SHA(head_path),'cache_count':head['audited_cache_count'],'pair_count':head['audited_pair_count']}
    perf_acceptance=PREP/'f48_inference_acceptance.json'
    body+='<h3>实测推理：只比较时序有根／无根</h3>'
    if not perf_acceptance.exists() or not (Read(perf_acceptance).get('b_pass') is True and Read(perf_acceptance).get('b_figures_verified') is True):
        body+='<p class="note">四种条件的推理与关闭删除五张图尚待正式验收。即使后台report.json已出现，也不提前填入性能结论。</p>'
    else:
        selection=Read(perf_root/'sample_selection.json');samples={s['index']:s for s in selection['samples']}
        if selection['fixed_validation_count']!=5973 or len(samples)!=2:raise ValueError('Benchmark selection not full5973 representative pair')
        rows=[];raw=[];reports={}
        for mode in ['cold','online']:
            for setting in ['on','off']:
                group=f'f48_{mode}_{setting}';path=perf_root/group/'report.json';report=Read(path)
                if (not report.get('b_complete') or report['checkpoint_sha256']!=source_sha or report['checkpoint_meta']['epoch']!=48
                        or report['arguments']['mode']!=mode or report['b_apply_root_filter']!=(setting=='on')
                        or report['arguments']['warmup']!=3 or report['arguments']['repeats']!=10 or report['precision']!='float32'):
                    raise ValueError('Benchmark final48/mode/root/protocol mismatch')
                if {case['index'] for case in report['cases']}!=set(samples):raise ValueError('Representative frames changed')
                for case in report['cases']:
                    if case['anchor_token']!=samples[case['index']]['token'] or len(case['samples'])!=10:raise ValueError('Frame identity mismatch')
                    row={'group':group,'mode':mode,'root_deletion':setting,'index':case['index'],'occupancy_quantile':samples[case['index']]['quantile'],'token':case['anchor_token'],'checkpoint_sha256':source_sha}
                    for field in ['pipeline_ms','target_forward_ms','target_with_export_ms','occ_forward_gpu_ms']:
                        values=[s[field] for s in case['samples']]
                        if any(v<=0 or not math.isfinite(v) for v in values):raise ValueError('Invalid timing')
                        values_sorted=sorted(values);position=(len(values_sorted)-1)*.95;left=int(position)
                        p95=values_sorted[left]+(values_sorted[min(left+1,len(values_sorted)-1)]-values_sorted[left])*(position-left)
                        median=statistics.median(values)
                        if not math.isclose(median,case['statistics_ms'][field]['median'],abs_tol=1e-7) or not math.isclose(p95,case['statistics_ms'][field]['p95'],abs_tol=1e-7):raise ValueError('Timing recomputation mismatch')
                        row[field+'_median']=median;row[field+'_p95']=p95
                    for field in ['baseline_allocated_bytes','baseline_reserved_bytes','peak_extra_allocated_bytes','peak_allocated_bytes','peak_reserved_bytes','rss_sum_peak_bytes']:row[field]=case[field]
                    row['run_rss_sum_peak_bytes']=report['rss_sum_peak_bytes'];rows.append(row)
                    raw.extend({'group':group,'index':case['index'],'measurement':i,**s} for i,s in enumerate(case['samples']))
                reports[group]={'path':str(path),'sha256':SHA(path),'scope':report['scope'],'rss_scope':report['rss_scope'],'online_scope':report['online_scope'],'online_time_feature':report['online_time_feature'],'source_sha256':report['source_sha256']}
        if len(rows)!=8 or len(raw)!=80:raise ValueError('Incomplete 4 conditions x 2 samples x 10 repeats')
        CSV(output/'root_switch_inference_summary.csv',rows);CSV(output/'root_switch_inference_raw80.csv',raw)
        body+='<p>GPU0 RTX4090，batch1，FP32/原TF32设置；每条件3次预热＋10次同步测量。代表样本来自F48全部5973验证目标的生成叶格数量分布：中位数附近与p95附近，各一个。<b>这是两个代表样本，不是全验证耗时分布</b>；下表的计时p95是在各样本10次测量内计算，不能与候选规模p95混淆。</p>'
        body+=Table(['模式','删除','目标ID / 候选规模分位','整网pipeline median/p95 ms','目标forward median/p95 ms','OCC median/p95 ms'],[[r['mode'],r['root_deletion'],str(r['index'])+' / '+str(r['occupancy_quantile']),*[f"{r[k+'_median']:.3f} / {r[k+'_p95']:.3f}" for k in ['pipeline_ms','target_forward_ms','occ_forward_gpu_ms']]] for r in rows])
        body+=Table(['模式','删除','目标ID','峰allocated GiB','峰reserved GiB','额外allocated GiB','该样本RSS合计峰 GiB','整次运行RSS峰 GiB'],[[r['mode'],r['root_deletion'],r['index'],*[f"{r[k]/2**30:.4f}" for k in ['peak_allocated_bytes','peak_reserved_bytes','peak_extra_allocated_bytes','rss_sum_peak_bytes','run_rss_sum_peak_bytes']]] for r in rows])
        comparisons=[]
        for mode in ['cold','online']:
            for index in samples:
                a=next(r for r in rows if r['mode']==mode and r['index']==index and r['root_deletion']=='on');b=next(r for r in rows if r['mode']==mode and r['index']==index and r['root_deletion']=='off')
                comparisons.append({'mode':mode,'index':index,**{key+'_reduction':1-a[key]/b[key] for key in ['pipeline_ms_median','occ_forward_gpu_ms_median','peak_allocated_bytes','peak_reserved_bytes']}})
        CSV(output/'root_switch_measured_reductions.csv',comparisons)
        body+=Table(['模式','目标ID','整网时间下降 %','OCC时间下降 %','峰allocated下降 %','峰reserved下降 %'],[[r['mode'],r['index'],*[f"{r[key+'_reduction']*100:.2f}" for key in ['pipeline_ms_median','occ_forward_gpu_ms_median','peak_allocated_bytes','peak_reserved_bytes']]] for r in comparisons])
        body+='<p class="note">cold包含从零处理15个历史帧＋目标帧；online先建立memory，再处理真实连续新帧。online新帧t=0，缓存历史特征不按年龄重编码，与cold按目标时间差重建历史的输入语义不同，不声称二者准确率等价。只在各自模式内比较开／关删除。pipeline含模型路径及导出，CPU数据读取另计；重型覆盖统计/出图不在测量内。OCC耗时为目标帧OCC阶段CUDA事件；训练峰值与推理峰值分表。torch allocator峰值不包括全部外部CUDA内存；RSS为主进程及递归子进程合计、名义20ms采样，共享页可能重复、短峰可能漏采；当前case采样窗口未保存边界索引，逐样本RSS峰只能引用已验收测量记录，不能由整次采样流独立重算。</p>'
        body+='<p>'+Link(output/'root_switch_inference_summary.csv','8行性能/内存明细',output)+' · '+Link(output/'root_switch_inference_raw80.csv','80次原始计时',output)+' · '+Link(perf_acceptance,'推理与真实图像验收',output)+'</p>'
        off_images=[]
        for split,index in [('train',10),('val',0),('val',100),('val',300),('val',2720)]:
            image=perf_root/f'root_off_epoch48/{split}_{index:03d}.png'
            if not image.exists():raise ValueError('Accepted root-off fixed image missing')
            off_images.append({'split':split,'index':index,'path':str(image),'sha256':SHA(image)})
        body+=Table(['固定样本','F48有根图','同F48关闭删除图'],[[('训练' if r['split']=='train' else '验证')+str(r['index']),Link(RUN/f"images/epoch_48/{r['split']}_{r['index']:03d}.png",'开启删除',output),Link(Path(r['path']),'关闭删除',output)] for r in off_images])
        evidence.update(b_performance_accepted=True,performance_acceptance={'path':str(perf_acceptance),'sha256':SHA(perf_acceptance)},performance_selection=selection,performance_reports=reports,performance_rows=rows,performance_reductions=comparisons,root_off_images=off_images)
    historical=PROJECT/'log/server_runs_x4_detection_screen_20261008'
    f5=Read(historical/'f/validation/epoch_5.json');n5=Read(historical/'f_no_root/validation/epoch_5.json')
    body+='<h3>独立训练精度证据：已有5轮，48轮尚待用户决定</h3><p>下表是分别训练的F有根/无根5轮，同S初始化、目标清单和LR前缀，不能与上方同权重F48开关混成一组。独立无根48轮未完成，本报告不将其列为已验收，也不自动扩展训练。</p>'
    body+=Table(['独立训练模型','mAP %','mAPH %','地形MAE m','RMSE m','OCC IoU %','语义mIoU %'],[MetricsRow('历史F有根5轮',f5),MetricsRow('历史F无根5轮',n5)])
    evidence['independent_training_status']='Historical matched5 completed; independent root-off48 awaits user decision; same-weight F48 switch does not substitute for separately trained48.'
    return body,evidence


class LocalLinks(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):self.paths.extend(v for k,v in attrs if k in ('href','src'))


def Build(output):
    output.mkdir(parents=True,exist_ok=True)
    audit=Read(PREP/'final_cpu_review.json')
    if not audit['b_execution_complete'] or audit['checkpoint']['epoch']!=48:raise ValueError('Final CPU acceptance required')
    protocol=Read(RUN/'long_protocol.json');baseline=Read(Path(protocol['baseline']));records=[Read(RUN/f'validation/epoch_{e}.json') for e in EPOCHS]
    for epoch,record in zip(EPOCHS,records):
        if record['epoch']!=epoch:raise ValueError('Validation epoch mismatch')
        m=record['metrics']
        if any(n!=5973 for n in [m['det']['ground_truth_audit']['frame_count'],m['terrain']['frame_count'],m['occ']['frame_count']]):raise ValueError('Incomplete fixed validation')
    loss_rows=LossRows();metric_rows=[FlattenMetrics(r) for r in records]
    CSV(output/'validation_metrics.csv',metric_rows);CSV(output/'raw_loss_epoch_mean_median.csv',loss_rows)
    CSV(output/'training_resources_384_rank_epoch.csv',audit['resources']['epoch_records'])
    CSV(output/'training_resources_per_rank.csv',audit['resources']['per_rank'])
    Plot(output,records,loss_rows,baseline)
    root_body,root_evidence=RootAblation(output)
    final=records[-1];f5=records[5];b0=baseline['metrics'];f48=final['metrics'];delta={task:{key:f48[task][key]-b0[task][key] for key in keys} for task,keys in [('det',['map_l2','maph_l2']),('terrain',['mae_m','rmse_m']),('occ',['occupancy_iou','semantic_miou'])]}
    heading='选定 X4 时序48轮与根筛选消融'
    body='<p class="muted">更新：'+datetime.datetime.now().astimezone().isoformat()+' · 单一主报告，后续根开关验收在此补齐。</p>'
    status_text='F48正式训练和同权重根开关均已验收；独立无根48轮尚待用户决定。' if root_evidence['b_performance_accepted'] else 'F48正式训练已完成；同权重根开关仍待性能验收，独立无根48轮尚待用户决定。'
    body+='<div class="note"><b>'+status_text+'</b> F采用检测读取E4、固定预训练主干/BEV的BN统计、地形与OCC使用完整F4。时序从S48初始化，复用已验收前5轮，再恢复43轮；没有从零训练。</div>'
    body+='<h2>1. 同一最终检查点与单帧基线</h2>'+Table(['模型 / 预算','mAP L2 %','mAPH L2 %','地形 MAE m','地形 RMSE m','OCC IoU %','语义 mIoU %'],[MetricsRow('单帧 S48：训练48轮，B0重评',baseline),MetricsRow('F5：S48 + 时序5轮（历史筛选）',f5),MetricsRow('F48：S48 + 时序48轮（最终）',final)])
    body+=f'<p>F48相对B0：检测mAP <b>{delta["det"]["map_l2"]*100:+.4f}个百分点</b>，mAPH {delta["det"]["maph_l2"]*100:+.4f}个百分点；地形MAE {delta["terrain"]["mae_m"]:+.6f}米，RMSE {delta["terrain"]["rmse_m"]:+.6f}米；OCC IoU {delta["occ"]["occupancy_iou"]*100:+.4f}个百分点、语义mIoU {delta["occ"]["semantic_miou"]*100:+.4f}个百分点。</p>'
    body+='<p class="note">此主表预算不相等：S48为48轮，F48在S48之上新增48轮。不能把全部差值归因于时序信息本身。固定BN的single_f5只与F5构成等预算短训对照，不能代替F48等预算单帧。所有指标来自同一epoch48；不拼接检测、地形、OCC各自最佳轮。</p>'
    body+=Table(['模型','车辆AP / APH %','行人AP / APH %','骑行者AP / APH %'],[[name]+[f"{r['metrics']['det'][c+'_ap_l2']*100:.4f} / {r['metrics']['det'][c+'_aph_l2']*100:.4f}" for c in ['vehicle','pedestrian','cyclist']] for name,r in [('S48 B0',baseline),('F48',final)]])
    body+='<h2>2. 22次固定完整验证与48轮真实损失</h2><p>798个训练序列、202个验证序列；训练候选23688，每轮seed＋epoch随机抽11844，DDP补4，每轮1481次更新；固定5973个验证目标。验证各rank按目标去重。输入0.05米、输出0.2米、48×48米、无数据增强；15历史＋1目标。检测使用补框GT口径 completed_detection_l2_v1，不称为官方原始Waymo标签结果。</p>'
    body+='<img src="validation_curves.png" alt="三任务完整验证曲线"><img src="raw_loss_curves.png" alt="逐轮原始损失均值与中位数">'
    body+='<p>'+Link(output/'validation_metrics.csv','完整验证数值 CSV',output)+' · '+Link(output/'raw_loss_epoch_mean_median.csv','48轮 raw loss均值/中位数、分类/回归/根loss及LR CSV',output)+' · '+Link(RUN/'train_metrics.jsonl','71088步原始日志',output)+' · '+Link(RUN/'tensorboard_minkunet_seg_multi','续训TensorBoard',output)+' · '+Link(PREFIX/'tensorboard_minkunet_seg_multi','前5轮TensorBoard',output)+'</p>'
    body+=Table(['时序轮次','mAP L2 %','mAPH %','MAE m','RMSE m','OCC IoU %','语义mIoU %'],[MetricsRow(str(r['epoch']),r) for r in records])
    body+='<h2>3. 保留根筛选失稳证据</h2><p class="note">第25轮检测仍通过门禁，但OCC IoU降至3.2167%，正根召回1.1583%，候选覆盖7.2069%；训练状态、数据、LR、123个冻结BN缓冲验收正常。后续第30轮恢复不能抹去这次崩溃。第40～48轮OCC仍波动，因此不把“训练完成”写成三任务稳定提升。根打分异常的原因需结合定向诊断，不能仅凭总loss下降判断正常。</p><img src="root_instability_and_lr.png" alt="根召回、覆盖、IoU和学习率">'
    selected=[r for r in records if r['epoch'] in [10,20,25,30,48]]
    body+=Table(['轮次','占据IoU %','正根召回 %','GT候选覆盖 %','保留根数','保留正根数'],[[r['epoch']]+[f"{r['metrics']['occ'][k]*100:.4f}" for k in ['occupancy_iou','occupied_root_recall','candidate_gt_coverage']]+[r['metrics']['occ'][k] for k in ['kept_root_count','occupied_root_kept']] for r in selected])
    for e in [20,25,30,48]:
        image=RUN/f'images/epoch_{e:02d}/val_100.png';body+='<details'+(' open' if e==25 else '')+'><summary>固定验证样本100：第'+str(e)+'轮 GT /预测（保留失败案例）</summary>'+Link(image,'打开原始PNG',output)+'<img loading="lazy" src="'+html.escape(os.path.relpath(image,output),quote=True)+'" alt="验证样本100第'+str(e)+'轮"></details>'
    body+='<p>'+Link(PREP/'epoch25_cpu_review.json','第25轮独立CPU异常验收',output)+' · '+Link(PREP/'epoch25_occ_root_diagnostic.json','根异常定向诊断原始记录',output)+' · '+Link(PREP/'epoch30_occ_followup.json','第30轮恢复记录',output)+'</p>'
    body+=root_body
    old=PROJECT.parent/'实验报告/2026-10-09_推理性能与短预算最终总览/index.html'
    body+='<p>'+Link(old,'历史F5根删除短训与推理报告（5轮口径）',output)+'</p>'
    times=audit['timings'];body+='<h2>5. 实际训练耗时与八卡显存</h2>'+Table(['范围','实测'],[['复用前5轮训练/验证/保存/出图动作',f"{times['inherited_five_epoch_actions_seconds']/3600:.3f}小时"],['新增43轮训练/验证/保存/出图动作',f"{times['new_43_epoch_actions_seconds']/3600:.3f}小时"],['完整48轮累计动作（不含两阶段间等待）',f"{times['combined_actions_seconds']/3600:.3f}小时"],['纯训练：每轮8rank最大耗时之和',f"{times['sum_each_epoch_max_rank_train_seconds']/3600:.3f}小时"]])
    body+=Table(['GPU rank','训练峰值 allocated GiB','训练峰值 reserved GiB'],[[r['rank'],f"{r['peak_allocated_bytes']/2**30:.4f}",f"{r['peak_reserved_bytes']/2**30:.4f}"] for r in audit['resources']['per_rank']])
    body+='<p>RTX4090×8，batch每卡1，FP32/TF32配置按原方案。torch.cuda分配器峰值按每轮训练采集，排除验证/图片；allocated/reserved不等于nvidia-smi全进程显存。CPU训练峰值未采集，不追溯虚构。'+Link(output/'training_resources_384_rank_epoch.csv','384条rank×epoch资源明细',output)+'。</p>'
    body+='<h2>6. 全部固定图片与复现入口</h2><p>每轮固定图：训练样本10；验证样本0、100、300、2720。下面明确区分训练与验证，链接110张实际图片。</p>'
    rows=[]
    for e in EPOCHS:
        image_run=PREFIX if e<=5 else RUN
        rows.append([e]+[Link(image_run/f'images/epoch_{e:02d}/{s}_{i:03d}.png',('训练' if s=='train' else '验证')+str(i),output) for s,i in [('train',10),('val',0),('val',100),('val',300),('val',2720)]])
    body+=Table(['轮次','训练10','验证0','验证100','验证300','验证2720'],rows)
    body+='<p>F48权重 SHA256：<code>'+audit['checkpoint']['sha256']+'</code></p>'
    single_protocol=Read(PREFIX/'screen_protocol.json')
    body+='<p>单帧S48权重 SHA256：<code>'+single_protocol['source_sha256']+'</code></p>'
    paths=[(Path(single_protocol['source']),'单帧S48起点权重'),(PREFIX/'screen_protocol.json','F前5轮共享S初始化协议'),(Path(audit['checkpoint']['path']),'F48最终权重'),(RUN/'config.yaml','实际展开配置'),(RUN/'long_protocol.json','长训协议/单帧起点与F5来源'),(RUN/'run_history.jsonl','16阶段启动与源码SHA'),(RUN/'train_tokens.json','训练完整候选清单'),(RUN/'val_tokens.json','固定验证目标清单'),(RUN/'label_manifest.json','地形/OCC标签版本清单'),(RUN/'completed_detection_manifest.json','补框标签清单'),(RUN/'epoch_sampling','逐轮50%抽样清单'),(PREP/'final_cpu_review.json','48轮独立CPU验收'),(PREP/'final_figures_review.json','最终图像内容审查'),(PROJECT/'configs/server_x4_selected_f_long48.yaml','实际配置入口')]
    body+='<ul>'+''.join('<li>'+Link(p,label,output)+'</li>' for p,label in paths)+'</ul><p>原48轮失败时序、旧5轮筛选、原Git分界线均保留；本报告不改写历史。</p>'
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+heading+'</title><style>body{max-width:1450px;margin:30px auto;padding:0 24px;font:16px/1.7 system-ui;color:#20334a}h1,h2{line-height:1.3}h2{margin-top:40px}a{color:#08699a}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th,td{padding:9px;border-bottom:1px solid #d7dee6;text-align:left}th{background:#edf3f7}.note{background:#fff1d8;border-left:4px solid #b88018;padding:16px}.muted{color:#647486}img{max-width:100%;height:auto}code{overflow-wrap:anywhere}details{margin:16px 0;padding:12px;background:#f5f7fa}</style></head><body><h1>'+heading+'</h1>'+body+'</body></html>'
    (output/'index.html').write_text(document)
    parser=LocalLinks();parser.feed(document);broken=[]
    for url in parser.paths:
        parts=urlsplit(url)
        if parts.scheme or not parts.path:continue
        if not (output/unquote(parts.path)).resolve().exists():broken.append(url)
    if broken:raise ValueError('Broken report links: '+str(broken))
    evidence={'generated_at':datetime.datetime.now().astimezone().isoformat(),'status':'F48_and_same_weight_root_switch_accepted' if root_evidence['b_performance_accepted'] else 'F48_complete_root_switch_performance_pending_acceptance','baseline':baseline,'F48':final,'delta_F48_minus_B0':delta,'epoch_count':48,'optimizer_updates':71088,'validation_epochs':EPOCHS,'validation_targets':5973,'budget_warning':'S48 vs S48+temporal48, not equal optimization budget','root25_instability_preserved':True,'source_sha256':{str(p):SHA(p) for p in [Path(__file__),PREFIX/'screen_protocol.json',PREP/'final_cpu_review.json',RUN/'train_metrics.jsonl',RUN/'config.yaml',RUN/'long_protocol.json',Path(protocol['baseline']),*[RUN/f'validation/epoch_{e}.json' for e in EPOCHS]]},'root_ablation':root_evidence}
    Write(output/'evidence.json',evidence);Write(output/'link_check.json',{'b_pass':True,'checked_links':len(parser.paths),'broken':[]})
    Write(output/'SHA256.json',{p.name:SHA(p) for p in sorted(output.iterdir()) if p.is_file() and p.name!='SHA256.json'})
    print(json.dumps({'output':str(output/'index.html'),'validation_records':22,'raw_loss_epochs':48,'prediction_links':110,'links_checked':len(parser.paths),'status':evidence['status']},ensure_ascii=False))


def Main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    Build(parser.parse_args().output.resolve())


if __name__=='__main__':Main()
