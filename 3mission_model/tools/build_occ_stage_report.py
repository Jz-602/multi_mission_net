"""Build an HTML OCC report only after all four complete fixed-set collections."""
import argparse
import base64
from datetime import datetime
import html
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from occ_stage_statistics import Ratios

GROUPS = ('single_f5', 'f5_root_on', 'f5_same_weight_root_off', 'f_no_root5')
LABELS = ('单帧固定BN 5轮', 'F5 开根筛选', '同F5权重关闭删除', 'F无删除独立训练5轮')
PLOT_LABELS = ('Single 5', 'F5 root on', 'F5 same weights off', 'F off trained 5')
FATES = ('潜在候选仍不可达', '根删除造成覆盖损失', '已生成但预测为空', '预测为非空')


def LoadComplete(directory):
    summary = json.loads((directory/'summary.json').read_text())
    manifest = json.loads((directory/'manifest.json').read_text())
    if not summary.get('b_complete') or summary['frame_count'] != 5973:
        raise ValueError(f'{directory}: complete5973 target evidence required')
    roi = manifest['resolved_config']['model']['point_cloud_range']
    if not np.allclose((np.array(roi[3:])-roi[:3])/.2, [240,240,64]):
        raise ValueError('This report must be bound to the agreed48m ROI and64-height leaf grid')
    counts = summary['counts']
    if sum(counts['gt_fates']) != counts['gt'] or counts['leaf_raw'] != 64*counts['expanded_root'] or counts['y3_raw'] != 8*counts['expanded_root'] or counts['leaf_raw']-counts['leaf'] != counts['leaf_ignored']:
        raise ValueError(f'{directory}: stage conservation failure')
    if summary['ratios'] != Ratios(counts):
        raise ValueError(f'{directory}: ratios must be recomputed from additive counts')
    return dict(directory=directory, summary=summary, manifest=manifest)


def Format(value, b_percent=False):
    if value is None: return '未定义（分母为0）'
    if b_percent: return f'{100*value:.4f}%'
    if isinstance(value, int): return f'{value:,}'
    return f'{value:.6g}'


def Table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def Figure(groups, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = ['#8294ae', '#cf624b', '#d5ad44', '#3d917d']
    positions = np.arange(4); left = np.zeros(4)
    for index, name in enumerate(('Unreachable', 'Root-deletion loss', 'Generated empty', 'Predicted occupied')):
        values = [100*g['summary']['ratios']['gt_fates_fraction'][index]['value']
                  if g['summary']['ratios']['gt_fates_fraction'][index]['value'] is not None else np.nan for g in groups]
        axes[0, 0].barh(positions, values, left=left, label=name, color=colors[index]); left += values
    axes[0, 0].set(yticks=positions, yticklabels=PLOT_LABELS, xlabel='Per 100 occupied GT voxels', xlim=(0, 100))
    axes[0, 0].legend(fontsize=8, loc='lower left', bbox_to_anchor=(0, 1.01), ncol=2, borderaxespad=0)
    axes[0, 0].set_title('Four mutually exclusive GT outcomes', pad=46)
    ratio_names = ('occupancy_precision', 'occupancy_recall', 'occupancy_iou', 'semantic_miou')
    for group_index, group in enumerate(groups):
        ratios = group['summary']['ratios']
        values = [ratios[name] if name == 'semantic_miou' else ratios[name]['value'] for name in ratio_names]
        axes[0, 1].bar(np.arange(4)+(group_index-1.5)*.2, [100*v if v is not None else np.nan for v in values], width=.2, label=PLOT_LABELS[group_index])
        stages = ('input_gt_coverage', 'history_current_gt_coverage', 'l0_gt_coverage', 'lplus_gt_coverage', 'actual_gt_coverage', 'occupancy_recall')
        axes[1, 0].plot(range(6), [100*ratios[k]['value'] if ratios[k]['value'] is not None else np.nan for k in stages], marker='o', label=PLOT_LABELS[group_index])
    axes[0, 1].set(xticks=range(4), xticklabels=('Precision', 'Recall', 'Occ IoU', 'Nonempty mIoU'), ylabel='%')
    axes[0, 1].legend(fontsize=8); axes[0, 1].set_title('Full valid ROI; ungenerated GT contributes FN')
    axes[1, 0].set(xticks=range(6), xticklabels=('Current I', 'I + history', 'L0', 'L+', 'Actual L', 'Occupied P'), ylabel='% of all occupied GT')
    axes[1, 0].legend(fontsize=8); axes[1, 0].set_title('Observed-point coverage and network coverage are distinct')
    boxes=[]; labels=[]
    for group, label in zip(groups, PLOT_LABELS):
        distribution = group['summary']['frame_distributions']['occupancy_iou']
        q = distribution['quantiles']
        if q is not None:
            boxes.append(dict(whislo=100*q[0], q1=100*q[1], med=100*q[2], q3=100*q[3], whishi=100*q[4], fliers=[])); labels.append(label)
    axes[1, 1].bxp(boxes, showfliers=False)
    axes[1, 1].set(xticklabels=labels, ylabel='Frame occupancy IoU (%)')
    axes[1, 1].tick_params(axis='x', labelrotation=15)
    axes[1, 1].set_title('Per-frame min / quartiles / max; not confidence intervals')
    figure.savefig(filename, dpi=160); plt.close(figure)


def Build(args):
    base = Path(args.input_root).resolve()
    groups = [LoadComplete(base/name) for name in GROUPS]
    if len({g['summary']['target_ids_sha256'] for g in groups}) != 1 or len({g['summary']['counts']['gt'] for g in groups}) != 1:
        raise ValueError('Four groups must share exact targetIDs and occupied GT count')
    if groups[1]['manifest']['checkpoint_sha256'] != groups[2]['manifest']['checkpoint_sha256']:
        raise ValueError('Immediate root on/off comparison must use exactly the same F5 weights')
    if not groups[1]['manifest']['b_apply_root_filter'] or groups[2]['manifest']['b_apply_root_filter'] or groups[3]['manifest']['b_apply_root_filter']:
        raise ValueError('Root on/off intervention labels disagree with recorded execution')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    figure_path = output/'occ_stages.png'; Figure(groups, figure_path)
    document=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>OCC完整固定验证阶段统计</title>',
        '<style>body{max-width:1500px;margin:32px auto;padding:0 22px;font:16px/1.6 sans-serif;color:#172c39;background:#f6f8fa}table{border-collapse:collapse;width:100%;margin:18px 0;background:white}th,td{border:1px solid #cbd5df;padding:7px;text-align:right}th:first-child,td:first-child{text-align:left}h1,h2{color:#244e61}img{width:100%}.note{background:#fff4d9;padding:15px}code{overflow-wrap:anywhere}</style>',
        '<h1>OCC完整固定验证：覆盖、删根与分类</h1>', f'<p>生成时间：{html.escape(datetime.now().isoformat())}。四组各5973个固定目标；所有总体比率先累加分子分母再计算。</p>',
        '<p class="note">本报告是已完成统计的描述性比较，不自动认定根筛选或时序提高效果。检测短预算通过不等于OCC改善。统计采集包含数据读取和重型CPU集合运算，不能作为推理速度基准。</p>',
        '<h2>三类证据不能互相替代</h2><ol><li>同一R0的几何反事实：L0直接展开，L+将所有R0保留并外扩；不重新预测。</li><li>同一F5权重的实际删除开／关：比较真实预测。性能必须使用统计关闭的独立计时。</li><li>F与F无删除分别训练：两者均从同源S开始、固定BN、最多5新轮；关闭删除仍保留根评分器和BCE。此结果包含训练轨迹差异。</li></ol>',
        '<h2>集合与监督口径</h2><p>所有叶集位于同一有效ROI、按0.2米去重。G为全部非空GT；I为当前真实点云格。历史真实点经位姿对齐后另列并集。R0为原根，Rk为实际保留根，L0与L+为潜在候选，L为实际生成候选，P为L中预测非空格。候选非空比例不是预测precision。显式255不参与监督或统计；源标签有效域已包含ROI，域内稀疏缺失沿既有标签约定为空。</p><p>每个原根仅按自身0.8米立方体是否含非空GT监督，不以邻根扩圈覆盖代替自身监督。叶监督只作用于实际生成候选，empty是显式训练类别；未生成GT仍计入全ROI FN。</p>',
        '<img alt="OCC阶段对照、每100GT去向与逐帧分布" src="data:image/png;base64,'+base64.b64encode(figure_path.read_bytes()).decode()+'">']
    def CountTable(title, fields):
        document.append('<h2>'+title+'</h2>')
        document.append(Table(['计数', *LABELS], [[label, *[Format(g['summary']['counts'][key]) for g in groups]] for label, key in fields]))
    def RatioTable(title, fields):
        document.append('<h2>'+title+'</h2>')
        document.append(Table(['指标', *LABELS], [[label, *[Format(g['summary']['ratios'][key]['value'], True) for g in groups]] for label, key in fields]))
    CountTable('输入真实覆盖与各级候选', [('非空GT |G|','gt'),('当前输入格 |I|','input_leaf'),('当前命中GT','input_gt'),('历史输入格','history_leaf'),('历史命中GT','history_gt'),('当前＋历史并集格','history_current_leaf'),('当前＋历史命中GT','history_current_gt'),('原始根 |R0|','root'),('保留根 |Rk|','kept_root'),('直接展开有效叶 |L0|','l0'),('L0命中GT','l0_gt'),('全保留外扩有效叶 |L+|','lplus'),('L+命中GT','lplus_gt'),('外扩新增有效候选','expansion_added_leaf'),('外扩新增GT','expansion_added_gt'),('实际保留根外扩新增有效叶','kept_expansion_added_leaf'),('实际保留根外扩新增GT','kept_expansion_added_gt'),('实际外扩后根','expanded_root'),('0.4米子级（未剔ignore）','y3_raw'),('0.2米叶级（未剔ignore）','leaf_raw'),('实际有效候选 |L|','leaf'),('候选非空GT','leaf_gt'),('候选空GT','empty_candidate')])
    RatioTable('覆盖与根自身监督', [('当前真实点覆盖','input_gt_coverage'),('当前＋历史真实点覆盖','history_current_gt_coverage'),('L0覆盖','l0_gt_coverage'),('L+覆盖','lplus_gt_coverage'),('实际L覆盖','actual_gt_coverage'),('外扩新增候选的非空比例','expansion_added_nonempty_fraction'),('实际保留根外扩增量非空比例','kept_expansion_added_nonempty_fraction'),('根删除比例','root_deletion_fraction'),('自身含GT根保留召回','occupied_root_recall'),('删除precision（删中自身空）','deletion_precision'),('自身空根删除召回','empty_root_deletion_recall'),('删根造成GT覆盖损失 / G','root_deletion_gt_coverage_loss')])
    CountTable('根删留与分类错误', [('自身含GT根','occupied_root'),('误删自身含GT根','occupied_root_deleted'),('保留自身含GT根','occupied_root_kept'),('删除自身空根','empty_root_deleted'),('保留自身空根','empty_root_kept'),('候选GT非空却预测空','candidate_gt_nonempty_pred_empty'),('GT非空预测错误非空类','candidate_gt_nonempty_wrong_nonempty'),('GT非空语义正确','candidate_gt_nonempty_correct'),('全场TP（含错非空类）','occupancy_tp'),('全场FP','occupancy_fp'),('全场FN（含未生成）','occupancy_fn')])
    RatioTable('候选组成与完整有效范围预测', [('候选非空GT占比（不是precision）','candidate_nonempty_fraction'),('候选内占据recall','candidate_occupancy_recall'),('全ROI occupancy precision','occupancy_precision'),('全ROI occupancy recall','occupancy_recall'),('全ROI occupancy IoU','occupancy_iou')])
    rows=[]
    for index, name in enumerate(('empty IoU（单列）','pole IoU','general obstacle IoU','vegetation IoU')):
        rows.append([name, *[Format(g['summary']['ratios']['class_iou'][index]['value'], True) for g in groups]])
    rows.append(['非空类mIoU', *[Format(g['summary']['ratios']['semantic_miou'], True) for g in groups]])
    document.append(Table(['语义指标', *LABELS], rows))
    document.append('<h2>四个互斥GT去向：每100个GT</h2>')
    document.append(Table(['去向', *LABELS], [[label,*[Format(g['summary']['ratios']['gt_fates_fraction'][i]['value'], True) for g in groups]] for i,label in enumerate(FATES)]))
    document.append(Table(['第四项内部拆分（每100个非空GT）', *LABELS], [
        [label, *[Format(g['summary']['counts'][key]/g['summary']['counts']['gt']*100)
                   if g['summary']['counts']['gt'] else Format(None) for g in groups]]
        for label, key in [('非空且语义正确', 'candidate_gt_nonempty_correct'),
                           ('非空但语义错误', 'candidate_gt_nonempty_wrong_nonempty')]]))
    for group, label in zip(groups, LABELS):
        document.append('<h2>'+label+'：候选内4×4混淆</h2><p>行=GT，列=预测。全部候选经过有效域和ignore过滤；未生成GT另计完整范围FN。</p>')
        document.append(Table(['GT／预测','empty','pole','general','vegetation'], [[name,*row] for name,row in zip(('empty','pole','general','vegetation'),group['summary']['counts']['candidate_confusion'])]))
        composition = group['summary']['counts']['candidate_gt_classes']
        document.append(Table(['候选GT类','格数','占全部有效候选'], [[name,Format(v),Format(v/group['summary']['counts']['leaf'],True) if group['summary']['counts']['leaf'] else Format(None)] for name,v in zip(('empty','pole','general','vegetation'),composition)]))
    document.append('<h2>边界与8倍生成关系</h2><p>当前ROI叶网格240×240×64，各轴整除4。每帧统计已验证完整父子支持：0.4米子级=外扩后根数×8，0.2米原始叶=子级×8=根数×64。因此本生成阶段没有子叶边界裁剪或中间截断，计数为0；有效叶减量由显式ignore解释。扩圈本身采用ROI内3×3×3邻域，边缘越界邻居已裁剪。</p><p class="note">扩圈“越界尝试数”未单独保存，不能从27×根数−去重外扩根数推导，因为该差还含根间重叠。此数记未记录，不记0；若将来需要，须额外记录根坐标或边界计数。没有为此重复整组GPU统计。</p>')
    CountTable('可由现有原始计数精确核实的ignore例外', [('完整有效ROI显式ignore数','ignored_leaf'),('实际生成叶中ignore数','leaf_ignored')])
    document.append('<h2>数据与复核入口</h2><p>summary保存总体原始counts及全部比率分子／分母；frames_rank*.jsonl保存逐目标记录。逐帧分布图为最小值／四分位数／最大值，不是置信区间。非空mIoU只平均并集非零的类，缺失类不擅自置0。</p>')
    for group,label in zip(groups,LABELS):
        manifest=group['manifest']; directory=group['directory']
        document.append('<p>'+label+'：<a href="'+html.escape(str(directory/'summary.json'))+'">summary</a> / <a href="'+html.escape(str(directory/'manifest.json'))+'">manifest</a><br>权重：<code>'+html.escape(manifest['checkpoint'])+'</code><br>SHA256：<code>'+manifest['checkpoint_sha256']+'</code></p>')
    document.append('</html>'); (output/'报告.html').write_text('\n'.join(document),encoding='utf-8')
    provenance=dict(b_complete=True,b_all_four_statistics_complete=True,b_inference_benchmark=False,
        inputs={name:str(g['directory']) for name,g in zip(GROUPS,groups)},frame_count_each=5973,
        target_ids_sha256=groups[0]['summary']['target_ids_sha256'],expansion_boundary_attempt_count=None,
        expansion_boundary_note='not recoverable separately from overlap using saved counts',
        b_method_effectiveness_automatically_concluded=False)
    (output/'report_provenance.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')


def Main():
    parser=argparse.ArgumentParser();parser.add_argument('--input-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();Build(args)


if __name__=='__main__':Main()
