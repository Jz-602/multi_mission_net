"""CPU-only final inference/experiment report; requires seven real complete benchmarks."""
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
import shutil
import statistics
import sys
from urllib.parse import unquote, urlsplit

PROJECT = Path(__file__).resolve().parents[1]
if PROJECT.name != '3mission_model':
    PROJECT = PROJECT.parent / '3mission_model'
GROUPS = {
    'single_f5_single': ('single_f', 'single', True),
    'f5_cold_on': ('f', 'cold', True),
    'f5_online_on': ('f', 'online', True),
    'f5_cold_off': ('f', 'cold', False),
    'f5_online_off': ('f', 'online', False),
    'f_no_root5_cold_off': ('f_no_root', 'cold', False),
    'f_no_root5_online_off': ('f_no_root', 'online', False),
}
TIMES = ('target_forward_ms', 'target_with_export_ms', 'pipeline_ms', 'occ_forward_gpu_ms')


def Read(path):
    return json.loads(path.read_text())


def SHA(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def Write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def Quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left = int(position)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (position - left)


def CheckNumbers(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite benchmark evidence')
    if isinstance(value, dict):
        for item in value.values(): CheckNumbers(item)
    elif isinstance(value, list):
        for item in value: CheckNumbers(item)


def CSV(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def Table(headers, rows):
    return '<div class="scroll"><table><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</table></div>'


def Link(path, label, output):
    target = os.path.relpath(path, output)
    return '<a href="' + html.escape(target, quote=True) + '">' + html.escape(label) + '</a>'


def Document(title, body):
    return '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>body{max-width:1500px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#20334a}table{border-collapse:collapse;white-space:nowrap;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eef3f7}.scroll{overflow:auto;margin:18px 0}a{color:#06679b}.note{background:#fff1d8;padding:15px}code{overflow-wrap:anywhere}</style><h1>' + title + '</h1>' + body + '</html>'


def LoadEvidence(args):
    root, runs, prep = args.input_root.resolve(), args.runs.resolve(), args.prep.resolve()
    missing = [str(root / name / 'report.json') for name in GROUPS if not (root / name / 'report.json').is_file()]
    if missing:
        raise ValueError('Seven real benchmark groups are not complete; no report generated. Missing: ' + ', '.join(missing))
    if not args.occ_report.is_file() or not args.screen_report.is_file():
        raise ValueError('Existing OCC and screening HTML reports are required for the final overview')
    selection_path = prep / 'inference_performance_sample_selection.json'
    selection = Read(selection_path)
    if selection['distribution']['frame_count'] != 5973:
        raise ValueError('Sample selection must derive from the full fixed validation set')
    selected = {x['index']: x for x in selection['selected']}
    if set(selected) != {3626, 167}:
        raise ValueError('Unexpected representative benchmark indices')
    reports, rows, identities, raw_timings = {}, [], [], []
    source_hashes = {str(selection_path): SHA(selection_path)}
    for name, (candidate, mode, b_root) in GROUPS.items():
        path = root / name / 'report.json'; report = Read(path); CheckNumbers(report)
        run_args = report['arguments']; gate = Read(runs / candidate / 'acceptance/epoch_5.json')
        if (report.get('b_complete') is not True or run_args['mode'] != mode
                or run_args['repeats'] != 10 or run_args['warmup'] != 3 or run_args['split'] != 'val'
                or set(run_args['indices']) != set(selected) or report['precision'] != 'float32'
                or report['b_apply_root_filter'] != b_root or not report['b_root_classifier_present']
                or report['checkpoint_meta']['epoch'] != 5
                or report['checkpoint_sha256'] != gate['checkpoint_sha256']):
            raise ValueError(f'{name}: protocol/checkpoint/root/mode binding mismatch')
        model_cfg = report['effective_model_config']
        if (model_cfg['point_cloud_range'] != [-24, -24, -7, 24, 24, 5.8]
                or model_cfg['voxel_size'] != [.05, .05, .05]
                or model_cfg['x4_detection_source'] != 'enhanced_e4'
                or model_cfg.get('b_freeze_pretrained_bn_stats') is not True
                or model_cfg['occ'].get('b_enable_root_filter') is not True):
            raise ValueError(f'{name}: benchmark model range/input/route/BN/root classifier differs')
        if len(report['cases']) != 2 or {x['index'] for x in report['cases']} != set(selected):
            raise ValueError(f'{name}: expected both representative cases exactly once')
        for case in report['cases']:
            sample = selected[case['index']]
            if case['anchor_token'] != sample['token'] or case['sequence'] != sample['sequence'] or len(case['samples']) != 10:
                raise ValueError(f'{name}: target identity or measured-frame count mismatch')
            tokens = [x['token'] for x in case['samples']]
            if mode == 'online':
                if case['tokens'] != sample['online_token_seq'] or tokens != case['tokens'][-10:] or len(set(tokens)) != 10:
                    raise ValueError(f'{name}: measured frames are not the registered real online stream')
            elif tokens != [sample['token']] * 10 or case['tokens'][-1] != sample['token'] or len(case['tokens']) != (1 if mode == 'single' else 16):
                raise ValueError(f'{name}: cold/single repeat frame identity mismatch')
            row = {'group': name, 'candidate': candidate, 'mode': mode, 'root_deletion': b_root,
                   'sample': sample['label'], 'index': case['index'], 'anchor_token': case['anchor_token'],
                   'repeats': 10, 'warmup': 3, 'loader_seconds_separate': case['loader_seconds'],
                   'checkpoint_sha256': report['checkpoint_sha256']}
            for key in TIMES:
                values = [x[key] for x in case['samples']]
                if any(x <= 0 for x in values): raise ValueError('Nonpositive measured duration')
                expected = {'count': 10, 'median': statistics.median(values), 'p95': Quantile(values, .95), 'minimum': min(values), 'maximum': max(values)}
                if any(not math.isclose(case['statistics_ms'][key][k], v, rel_tol=1e-9, abs_tol=1e-9) for k, v in expected.items()):
                    raise ValueError(f'{name}: stored timing statistics disagree with samples')
                row[key + '_median'] = expected['median']; row[key + '_p95'] = expected['p95']
            for key in ('baseline_allocated_bytes','baseline_reserved_bytes','peak_extra_allocated_bytes','peak_allocated_bytes','peak_reserved_bytes','rss_sum_peak_bytes'):
                if case[key] < 0: raise ValueError('Negative measured memory')
                row[key] = case[key]; row[key.replace('_bytes', '_GiB')] = case[key] / 2**30
            row['run_rss_peak_GiB'] = report['rss_sum_peak_bytes'] / 2**30
            row['input_scope'] = case['input_scope']; row['cuda_scope'] = case['cuda_scope']; row['rss_scope'] = report['rss_scope']
            rows.append(row)
            identities.append({'group':name,'index':case['index'],'anchor_token':case['anchor_token'],'context_tokens':case['tokens'],'measured_tokens':tokens,'online_time_feature':report['online_time_feature']})
            raw_timings.extend(dict(group=name,index=case['index'],measurement=i,**{k:x[k] for k in (*TIMES,'token')}) for i,x in enumerate(case['samples']))
        reports[name] = report; source_hashes[str(path)] = SHA(path)
    if len({(x['device'],x['precision'],x['b_tf32_matmul'],x['b_tf32_cudnn'],x['torch_version'],x['cuda_version']) for x in reports.values()}) != 1:
        raise ValueError('Seven benchmarks must share device/precision/backend settings')
    training, per_rank = [], []
    for candidate in ('single_f','f','f_no_root'):
        records = []
        for epoch in range(1,6):
            for rank in range(8):
                path = runs / candidate / f'training_resources/epoch_{epoch}_rank{rank}.json'
                record = Read(path); CheckNumbers(record)
                if record['epoch'] != epoch or record['rank'] != rank or any(record[k] <= 0 for k in ('elapsed_seconds','peak_allocated_bytes','peak_reserved_bytes')):
                    raise ValueError(f'{candidate}: training resource record mismatch')
                source_hashes[str(path)] = SHA(path)
                row = dict(candidate=candidate,**record,peak_allocated_GiB=record['peak_allocated_bytes']/2**30,peak_reserved_GiB=record['peak_reserved_bytes']/2**30)
                training.append(row); records.append(row)
        for rank in range(8):
            part = [x for x in records if x['rank']==rank]
            per_rank.append({'candidate':candidate,'rank':rank,'epochs':5,'max_allocated_GiB':max(x['peak_allocated_GiB'] for x in part),'max_reserved_GiB':max(x['peak_reserved_GiB'] for x in part),'epoch_seconds_median':statistics.median(x['elapsed_seconds'] for x in part),'epoch_seconds_p95':Quantile([x['elapsed_seconds'] for x in part],.95),'epoch_seconds_sum':sum(x['elapsed_seconds'] for x in part),'cpu_peak_rss':'not_recorded'})
    fixed_metrics = []
    for candidate in ('single_f','f','f_no_root'):
        path = runs/candidate/'validation/epoch_5.json'; value=Read(path)
        if value['epoch'] != 5: raise ValueError('Overview requires fixed epoch5')
        for task in ('terrain','occ'):
            if value['metrics'][task]['frame_count'] != 5973: raise ValueError('Incomplete fixed validation')
        if value['metrics']['det']['ground_truth_audit']['frame_count'] != 5973: raise ValueError('Incomplete fixed detection')
        fixed_metrics.append({'candidate':candidate,'record':value,'checkpoint':Read(runs/candidate/'acceptance/epoch_5.json')['checkpoint'],'checkpoint_sha256':Read(runs/candidate/'acceptance/epoch_5.json')['checkpoint_sha256']})
        source_hashes[str(path)] = SHA(path)
    return dict(generated_at=datetime.datetime.now().astimezone().isoformat(),reports=reports,performance_rows=rows,raw_timings=raw_timings,frame_identities=identities,selection=selection,training_records=training,training_per_rank=per_rank,fixed_metrics=fixed_metrics,source_sha256=source_hashes)


class LocalLinks(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):self.links.extend(v for k,v in attrs if k in ('href','src'))


def Build(args):
    evidence = LoadEvidence(args)  # Fail closed before creating any output directory/artifact.
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    Write(output/'evidence.json', evidence)
    CSV(output/'inference_summary.csv',evidence['performance_rows'])
    CSV(output/'inference_raw_timings.csv',evidence['raw_timings'])
    CSV(output/'training_40rank_records_per_model.csv',evidence['training_records'])
    CSV(output/'training_per_rank.csv',evidence['training_per_rank'])
    Write(output/'frame_ids_and_scope.json',evidence['frame_identities'])
    first = next(iter(evidence['reports'].values()))
    body = '<p>七组真实基准全部完成；每组2个代表样本、3次暖up、10次测量。样本依据完整5,973验证目标的F5原始叶格数量，确定性选择中位附近index3626和p95附近index167；这是样本复杂度分位，不是把两样本的速度称为全验证总体p95。</p>'
    body += '<p>环境：'+html.escape(first['device'])+'，FP32，batch1，eval/no_grad；TF32 matmul='+str(first['b_tf32_matmul'])+'、cuDNN='+str(first['b_tf32_cudnn'])+'。下表 median/p95 来自每个样本自己的10条实际测量。</p>'
    body += '<p class="note">冷clip每次重新构建15帧历史（目标相对时间差），然后预测目标；online先以15个真实连续历史帧建memory，再丢弃3个新帧暖up、测10个不同的连续新帧。online每个新编码帧time=0，已存memory不随年龄重新编码；cold还沿生产历史采样间隔取帧。因此两者不是相同时间特征/相同历史采样的准确率一致验证，不能将online速度直接替代正式cold验证结果。</p>'
    body += '<p>model forward 是同步墙钟；with export 累计包含模型前向、生产NMS/OCC argmax/CPU传输；pipeline包含H2D、生产体素化及其CPU工作，cold还含建历史memory，online仅当前新帧更新。CPU数据读取/预处理已预加载，单独列loader耗时，不含在pipeline中。OCC为CUDA事件跨度，只测目标OCC调用；不同计时口径不作简单相减。</p>'
    for title,modes in [('单帧及冷clip：重复同一输入',('single','cold')),('在线连续新帧：真实滚动memory',('online',))]:
        rows=[r for r in evidence['performance_rows'] if r['mode'] in modes]
        body += '<h2>'+title+'</h2>'+Table(['组','样本(index)','forward ms 中位/p95','含导出 ms 中位/p95','pipeline ms 中位/p95','OCC GPU ms 中位/p95','loader秒（单列）'],[[r['group'],f"{r['sample']} ({r['index']})"]+[f"{r[k+'_median']:.3f} / {r[k+'_p95']:.3f}" for k in TIMES]+[f"{r['loader_seconds_separate']:.3f}"] for r in rows])
    body += '<h2>推理内存实测与范围</h2><p>GiB=2³⁰字节。CUDA峰在初次暖up前reset，因此包含常驻模型、H2D、memory、原始输出、导出临时量及暖up，不只是10次稳态计算；reserved是缓存分配器保留量，不能当作张量实际占用。RSS按主进程加递归子进程求和、共享页可能重复，采样可能漏短峰；case峰与含模型/数据加载的整次run峰分别列出。</p>'
    body += Table(['组','样本','allocated峰GiB','reserved峰GiB','相对基线额外allocated GiB','case CPU RSS峰GiB','run CPU RSS峰GiB'],[[r['group'],r['sample']]+[f"{r[k]:.4f}" for k in ('peak_allocated_GiB','peak_reserved_GiB','peak_extra_allocated_GiB','rss_sum_peak_GiB','run_rss_peak_GiB')] for r in evidence['performance_rows']])
    body += '<h2>三组训练：每组5轮×8rank的真实记录</h2><p>每卡峰值为五轮实测峰的最大值；时间中位/p95按该卡5轮记录计算，八卡并行时间不得跨rank相加为工期。范围仅train_epoch，排除验证、保存、图片；完整120条记录见CSV。训练CPU峰值当时未采集，不以推理RSS代替。</p>'
    body += Table(['训练组','rank','allocated最大GiB','reserved最大GiB','单轮秒 中位/p95','五轮秒合计（该rank）','训练CPU峰'],[[r['candidate'],r['rank'],f"{r['max_allocated_GiB']:.4f}",f"{r['max_reserved_GiB']:.4f}",f"{r['epoch_seconds_median']:.2f}/{r['epoch_seconds_p95']:.2f}",f"{r['epoch_seconds_sum']:.2f}",'未采集'] for r in evidence['training_per_rank']])
    body += '<h2>原始证据</h2><p>'+ ' · '.join(Link(output/f,label,output) for f,label in [('evidence.json','全部汇总证据'),('inference_summary.csv','性能CSV'),('inference_raw_timings.csv','140条真实时序测量'),('frame_ids_and_scope.json','完整帧IDs与口径'),('training_40rank_records_per_model.csv','120条训练rank记录'),('training_per_rank.csv','24行每卡训练汇总')])+'</p>'
    for name in GROUPS: body += '<p>'+Link(args.input_root/name/'report.json',name+' 原始report.json',output)+'</p>'
    (output/'performance.html').write_text(Document('真实推理性能：冷clip与在线新帧',body))
    overview='<p class="note">这是既定短预算实验的真实产物总览，固定比较single_f5、F5与F_no_root5，不拼接不同最好轮次。F仅通过短预算检测筛选：相对匹配single_f5有小幅收益，但逐轮有波动、单seed且轮次相关，未证明稳定或统计显著优势，也不代表三任务全面改善。</p>'
    overview += '<p>'+Link(args.screen_report,'完整检测筛选、根消融、loss及真实效果图',output)+' · '+Link(args.occ_report,'完整固定验证OCC阶段统计',output)+' · '+Link(output/'performance.html','实际推理与训练资源报告',output)+'</p>'
    rows=[]
    for row in evidence['fixed_metrics']:
        m=row['record']['metrics'];rows.append([row['candidate'],'5',f"{m['det']['map_l2']*100:.4f}",f"{m['det']['maph_l2']*100:.4f}",f"{m['terrain']['mae_m']:.6f}",f"{m['terrain']['rmse_m']:.6f}",f"{m['occ']['occupancy_iou']*100:.4f}",f"{m['occ']['semantic_miou']*100:.4f}"])
        overview+='<p>'+html.escape(row['candidate'])+'权重：'+Link(Path(row['checkpoint']),'固定epoch5.pth',output)+'<br>SHA256：<code>'+row['checkpoint_sha256']+'</code></p>'
    overview += Table(['模型','固定轮','mAP L2 %','mAPH L2 %','地形MAE m','RMSE m','OCC IoU %','语义mIoU %'],rows)
    overview += '<p>同F5权重关闭历史是输入分布干预，不等于独立single_f；同F5即时关闭根删除与独立训练F_no_root也属于不同问题。F_no_root保留原根BCE，OCC较好不能概括成检测/三任务全面改善。两代表样本的性能不能冒充全验证集推理分布；更不能据online测试宣称准确率与cold相同。权重、指标和样本范围均以原始证据绑定为准。</p>'
    (output/'index.html').write_text(Document('短预算三任务实验：固定终点与实际产物总览',overview))
    shutil.copyfile(__file__,output/'生成脚本.py')
    links=[]
    for f in ('index.html','performance.html'):
        parser=LocalLinks();parser.feed((output/f).read_text());links.extend(parser.links)
    missing=[x for x in set(links) if not urlsplit(x).scheme and not (output/unquote(urlsplit(x).path)).exists()]
    Write(output/'link_check.json',{'b_pass':not missing,'unique_links':len(set(links)),'missing':missing})
    if missing: raise ValueError('Missing final report links: '+repr(missing))
    Write(output/'SHA256.json',{f.name:SHA(f) for f in output.iterdir() if f.is_file() and f.name!='SHA256.json'})
    print(json.dumps({'b_complete':True,'output':str(output),'benchmarks':7,'performance_rows':14,'raw_timings':140,'training_epoch_rank_records':120,'links':len(set(links))},ensure_ascii=False))


def Main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,default=PROJECT/'log/server_inference_performance_20261009')
    parser.add_argument('--runs',type=Path,default=PROJECT/'log/server_runs_x4_detection_screen_20261008')
    parser.add_argument('--prep',type=Path,default=PROJECT/'log/server_preparation/x4_detection_screen_20261008')
    parser.add_argument('--screen-report',type=Path,default=PROJECT.parent/'实验报告/2026-10-08至09_X4检测筛选_最多五轮/index.html')
    parser.add_argument('--occ-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try: Build(args)
    except (OSError,ValueError,KeyError,TypeError) as error:
        print(str(error),file=sys.stderr);sys.exit(4)


if __name__=='__main__':Main()
