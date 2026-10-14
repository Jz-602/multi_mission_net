"""CPU-only snapshot of existing X4 screen evidence; never starts or evaluates a model."""
import argparse
import csv
import datetime
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit

PROJECT = Path(__file__).resolve().parents[1]
if PROJECT.name != '3mission_model':  # 报告目录内保留的脚本副本也可原位重跑。
    PROJECT = PROJECT.parent / '3mission_model'
REPORT_NAME = '2026-10-08至09_X4检测筛选_最多五轮'
NAMES = {'single': '共享单帧（同结构，无历史）', 'a': 'A：检测读完整 F4',
         'b': 'B：检测按当前 S4 读取 F4', 'c': 'C：检测读取 E4',
         'single_f': 'single_f：固定预训练 BN 统计的公平单帧对照',
         'f': 'F：E4 路径＋固定预训练 BN 统计',
         'f_no_root': 'F_no_root：不删根，保留相同根辅助监督'}
MATCHED_CONTROLS = {'single': 'single', 'a': 'single', 'b': 'single',
                    'c': 'single', 'single_f': 'single_f', 'f': 'single_f', 'f_no_root': 'f'}


def Read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def Write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def SHA(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ReadLines(path):
    if not path.exists():
        return [], False
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    b_incomplete = bool(lines and not lines[-1].endswith('\n'))
    if b_incomplete:
        lines.pop()  # 活动训练可能正在写最后一行；只读取已完成记录。
    return [json.loads(line) for line in lines if line.strip()], b_incomplete


def CSV(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def Link(path, label):
    return f'<a href="{html.escape(str(path), quote=True)}">{html.escape(str(label))}</a>'


def Table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{html.escape(v)}</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def Number(value, percent=False):
    return '—' if value is None else f'{value * (100 if percent else 1):.4f}'


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for key, value in attrs if key in ('href', 'src'))


def HistoryDiagnostic(output, runs):
    off_path = runs / 'f_nohistory_eval/validation_metrics.json'
    acceptance_path = PROJECT / 'log/server_supervision/20261009T125529_x4_f5_nohistory_full_validation/final_acceptance.json'
    if not off_path.exists() or not acceptance_path.exists():
        return '', None
    on = Read(runs / 'f/validation/epoch_5.json')
    off = Read(off_path)
    acceptance = Read(acceptance_path)
    gate = Read(runs / 'f/acceptance/epoch_5.json')
    manifest = Read(runs / 'f_nohistory_eval/manifest.json')
    if (acceptance['status'] != 'accepted_full_validation_diagnostic'
            or acceptance['checkpoint_sha256'] != gate['checkpoint_sha256']
            or manifest['checkpoint_provenance']['resume']['sha256'] != gate['checkpoint_sha256']
            or not manifest['arguments']['validate_only']):
        raise ValueError('F5 history diagnostic has no accepted identical-checkpoint binding')
    rows, metric_rows, class_rows = [], [], []
    for name, record in [('开启历史：15＋1', on), ('关闭历史：仅目标帧', off)]:
        det, terrain, occ = [record['metrics'][task] for task in ('det', 'terrain', 'occ')]
        if record['epoch'] != 5 or any(count != 5973 for count in (
                det['ground_truth_audit']['frame_count'], terrain['frame_count'], occ['frame_count'])):
            raise ValueError('Incomplete F5 history diagnostic')
        metric_rows.append([name, '5', '5973', Number(det['map_l2'], True), Number(det['maph_l2'], True),
                            Number(terrain['mae_m']), Number(terrain['rmse_m']),
                            Number(occ['occupancy_iou'], True), Number(occ['semantic_miou'], True)])
        class_rows.append([name] + [Number(det[f'{cls}_{metric}_l2'], True)
                           for cls in ('vehicle', 'pedestrian', 'cyclist') for metric in ('ap', 'aph')])
        row = {'condition': name, 'epoch': 5, 'checkpoint_sha256': gate['checkpoint_sha256']}
        for task, values in record['metrics'].items():
            row.update({task + '_' + key: value for key, value in values.items()
                        if isinstance(value, (int, float)) and not isinstance(value, bool)})
        rows.append(row)
    evidence = {'checkpoint': gate['checkpoint'], 'checkpoint_sha256': gate['checkpoint_sha256'],
                'history_on': on, 'history_off': off, 'acceptance': acceptance,
                'b_same_checkpoint': True, 'b_training_performed': False,
                'limitation': 'Closing history changes the input distribution of a temporally trained checkpoint; this inference intervention does not replace independently trained single_f or establish the causal effect of temporal training.'}
    Write(output / 'f5_history_diagnostic.json', evidence)
    CSV(output / 'f5_history_diagnostic.csv', rows)
    section = '<h2>同一 F5 权重：开／关历史的完整验证诊断</h2><p>固定同一第5轮权重、补框新真值和5,973个验证目标，仅改变推理历史输入，未进行训练。关闭历史改变了时序模型的输入分布；这个结果反映该权重对历史输入的依赖，<strong>不等于独立训练的 single_f，不替代公平训练对照，也不能单独证明训练收益的因果来源。</strong></p>'
    section += '<p>同一权重 SHA256：<code>' + gate['checkpoint_sha256'] + '</code></p>'
    section += Table(['推理条件','检查点轮次','目标数','mAP L2 %','mAPH L2 %','地形 MAE m','RMSE m','OCC IoU %','语义 mIoU %'], metric_rows)
    section += Table(['推理条件','车辆 AP %','车辆 APH %','行人 AP %','行人 APH %','骑行者 AP %','骑行者 APH %'], class_rows)
    section += '<p>' + ' · '.join(Link(path, label) for path, label in [
        ('f5_history_diagnostic.json','完整诊断原始JSON与验收记录'),
        ('f5_history_diagnostic.csv','独立诊断CSV（不混入训练轮次表）'),
        ('完整运行记录/f/validation/epoch_5.json','开启历史原始验证'),
        ('完整运行记录/f_nohistory_eval/validation_metrics.json','关闭历史原始验证'),
        ('完整运行记录/f_nohistory_eval/manifest.json','关闭历史运行与权重绑定'),
        ('../../3mission_model/log/server_supervision/20261009T125529_x4_f5_nohistory_full_validation/final_acceptance.json','后台最终验收')]) + '</p>'
    return section, evidence


def RootAblation(output, prep):
    path = prep / 'f_no_root_effectiveness_review.json'
    if not path.exists():
        return '', None
    review = Read(path)
    pairs = review['paired_metrics_by_epoch']
    final = pairs[-1]
    Write(output / 'f_root_ablation_review.json', review)
    CSV(output / 'f_root_ablation_deltas.csv', [dict(epoch=row['epoch'], **row['delta_off_minus_on']) for row in pairs])
    CSV(output / 'f_root_ablation_raw_loss.csv', review['raw_loss_epoch_means'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for axis, (task, key, title, scale) in zip(axes.flat, [
            ('det', 'map_l2', 'Detection mAP L2 (%)', 100),
            ('det', 'maph_l2', 'Detection mAPH L2 (%)', 100),
            ('terrain', 'mae_m', 'Terrain weighted MAE (m)', 1),
            ('terrain', 'rmse_m', 'Terrain weighted RMSE (m)', 1),
            ('occ', 'occupancy_iou', 'OCC occupancy IoU (%)', 100),
            ('occ', 'semantic_miou', 'OCC semantic mIoU (%)', 100)]):
        for candidate, label, color in [('f', 'F: root deletion ON', '#17629c'),
                                       ('f_no_root', 'F_no_root: deletion OFF, BCE kept', '#d5651d')]:
            axis.plot([row['epoch'] for row in pairs], [row[candidate][task][key] * scale for row in pairs],
                      marker='o', label=label, color=color)
        axis.set_title(title)
        axis.set_xlabel('New training epoch (0 = initialization)')
        axis.set_xticks(range(6))
        axis.grid(alpha=.25)
    axes.flat[0].legend(fontsize=8)
    figure.savefig(output / 'f_root_ablation_metrics.png', dpi=160)
    plt.close(figure)
    section = '<h2>受控根删除消融：F 与 F_no_root</h2><p><strong>' + html.escape(review['conclusion_zh']) + '</strong></p>'
    section += '<p>两组均从同一 S 开始，最多5轮、每轮完整验证；仅改变根删除选择，保留原根 BCE 及其权重。检测与静态分支共享训练过程，因此最终差异是整条受控训练路径的差异，不能解释成单次推理开关的效果。第0轮供初始化对照；正式终点固定第5轮，不挑不同最佳轮拼表。</p>'
    section += '<img src="f_root_ablation_metrics.png" alt="F和F_no_root六次完整验证三任务曲线">'
    metric_rows, class_rows = [], []
    for candidate, label in [('f', 'F5：有根删除'), ('f_no_root', 'F_no_root5：不删根，保留BCE')]:
        m = final[candidate]
        metric_rows.append([label, Number(m['det']['map_l2'], True), Number(m['det']['maph_l2'], True),
                            Number(m['terrain']['mae_m']), Number(m['terrain']['rmse_m']),
                            Number(m['occ']['occupancy_iou'], True), Number(m['occ']['semantic_miou'], True)])
        class_rows.append([label] + [Number(m['det'][f'{cls}_{metric}_l2'], True)
                           for cls in ('vehicle', 'pedestrian', 'cyclist') for metric in ('ap', 'aph')])
    section += Table(['固定终点','mAP L2 %','mAPH L2 %','地形 MAE m','RMSE m','OCC IoU %','语义 mIoU %'], metric_rows)
    section += Table(['固定终点','车辆 AP %','车辆 APH %','行人 AP %','行人 APH %','骑行者 AP %','骑行者 APH %'], class_rows)
    section += '<p>下表统一为“不删根−有根删除”；AP/IoU 为百分点，地形为米。负的地形差表示误差下降；负的 AP/IoU 差表示精度下降。</p>'
    section += Table(['轮次','ΔmAP pp','ΔmAPH pp','ΔMAE m','ΔRMSE m','ΔOCC IoU pp','Δ语义 mIoU pp'], [[str(row['epoch'])] + [Number(row['delta_off_minus_on'][key]) for key in ('map_l2','maph_l2','terrain_mae_m','terrain_rmse_m','occ_occupancy_iou_pp','occ_semantic_miou_pp')] for row in pairs])
    section += '<p>' + '</p><p>'.join(html.escape(value) for value in review['interpretation']) + '</p>'
    section += '<p>原始 rank0 记录确认全部7,405步根监督数仍等于输入根数，根 BCE 有非零值；F_no_root 每步 deleted_root_count=0。逐轮根 BCE 均值见下表；仅日志均值，不是全 rank 统计。</p>'
    section += Table(['实验','轮次','检测 raw','地形 raw','OCC raw','根 BCE 均值','根监督数合计 rank0'], [[row['candidate'],str(row['epoch'])] + [Number(row[key]) for key in ('det_raw_loss_mean','terrain_raw_loss_mean','occ_raw_loss_mean','root_bce_mean')] + [str(row['root_supervision_count_sum_rank0'])] for row in review['raw_loss_epoch_means']])
    performance_dir = output.parent / '2026-10-09_推理性能与短预算最终总览'
    performance_evidence = Read(performance_dir / 'evidence.json', {})
    reports = performance_evidence.get('reports', {})
    b_performance_complete = ((performance_dir / 'performance.html').exists()
                              and len(reports) == 7 and all(row.get('b_complete') for row in reports.values()))
    if b_performance_complete:
        section += '<p class="warning">七组真实推理资源专测现已完成，见 ' + Link('../2026-10-09_推理性能与短预算最终总览/performance.html','冷clip与在线新帧性能报告') + '。历史根消融审查JSON中的“未测”状态仅对应其生成时刻；性能结论以该独立报告的实测范围为准，不能外推为全验证集速度分布。训练CPU峰未采集。保留已选F，不因单项OCC较高后验改选模型。</p>'
    else:
        section += '<p class="warning">推理实际资源专测尚未采集。已有训练显存记录或预检不能替代同条件推理延迟/显存测试；本表不宣称删根已经带来实测加速或内存收益。保留已选 F，不因单项 OCC 较高后验改选模型。</p>'
    section += '<p>' + ' · '.join(Link(file, label) for file, label in [
        ('f_root_ablation_review.json','完整受控效果审查JSON'),
        ('f_root_ablation_deltas.csv','逐轮同检查点差值CSV'),
        ('f_root_ablation_raw_loss.csv','原始任务损失与根BCE CSV'),
        ('准备与诊断/fairness_review_no_root.json','开关、前向与BCE保留公平性证据')]) + '</p>'
    return section, review


def Build(output):
    output.mkdir(parents=True, exist_ok=True)
    runs = PROJECT / 'log/server_runs_x4_detection_screen_20261008'
    prep = PROJECT / 'log/server_preparation/x4_detection_screen_20261008'
    effectiveness_decision = Read(prep / 'f_effectiveness_decision.json')
    for name, target in [('完整运行记录', runs), ('准备与诊断', prep)]:
        link = output / name
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        if link.resolve() != target.resolve():
            raise ValueError(f'Existing evidence link points elsewhere: {link}')
    snapshot = {'snapshot_at': datetime.datetime.now().astimezone().isoformat(),
                'b_final_paper_effectiveness_claim': False, 'candidates': {},
                'effectiveness_decision': effectiveness_decision,
                'protocol': 'completed_detection_l2_v1; original plus completed GT; XYZT; 5973 fixed validation targets',
                'metric_units': 'JSON/CSV AP/APH/IoU fractions [0,1], terrain metres; HTML AP/IoU in percent',
                'limitations': ['Snapshot of existing files; no fresh inference or GPU execution.',
                               'Active candidate has only completed validation records; raw loss can include a partial epoch.',
                               'Checkpoint SHA comes from saved acceptance binding; this generator does not reload/copy weights.']}
    validations, losses, timings, candidate_rows = [], [], [], []
    sections = []
    for candidate, name in NAMES.items():
        run = runs / candidate
        registration = Read(prep / f'candidate_{candidate}_registration.json', {})
        if not run.exists():
            pending_status = '已注册，尚无正式结果' if registration else '计划内，尚无正式结果'
            snapshot['candidates'][candidate] = {
                'name': name, 'status': pending_status,
                'matched_control': MATCHED_CONTROLS[candidate],
                'registration': registration, 'validation': [], 'raw_epoch_means': []}
            candidate_rows.append([html.escape(name), pending_status, '—', '—', '—', '待执行，不据此判断效果'])
            part = '<h2>' + html.escape(name) + '</h2><p>尚无正式训练/完整验证结果。公平对照：<strong>' + MATCHED_CONTROLS[candidate] + '</strong>。</p>'
            if registration:
                part += '<p>' + html.escape(registration.get('comparison', '')) + '</p>'
                part += Link(f'准备与诊断/candidate_{candidate}_registration.json', '实验注册、初始化与预算')
            sections.append(part)
            continue
        protocol = Read(run / 'screen_protocol.json', {})
        stage = Read(run / 'stage_result.json')
        failed = Read(run / 'failed_action.json')
        state = Read(run / 'state.json', {})
        status = ('失败待核查' if failed else '退化早停' if stage and stage.get('status') == 'early_stop'
                  else '五轮已结束；待综合效果判断' if stage else '进行中；非最终结果')
        if (candidate == 'f' and effectiveness_decision
                and effectiveness_decision.get('b_detection_screen_passed') and stage and not failed):
            status = '通过短预算检测筛选；有限收益，非稳定/统计显著结论'
        final_review = Read(run / 'final_review.json')
        if final_review:
            if (final_review['candidate'] != candidate or
                    SHA(Path(final_review['external_final_acceptance'])) != final_review['external_final_acceptance_sha256']
                    or SHA(run / 'stage_result.json') != final_review['original_stage_result_sha256']):
                raise ValueError('Final reviewed status is not bound to external acceptance/original termination evidence')
            status = {'completed_control': '单帧对照已完成并验收',
                      'early_stop': '真实完整验证触发早停，已验收',
                      'passed_short_budget_detection_screen': '通过短预算检测筛选；有限收益，非稳定/统计显著结论',
                      'completed_controlled_ablation': '受控根消融已完成；检测低/OCC高，不是新通过候选'}[final_review['status']]
        raw, b_incomplete = ReadLines(run / 'train_metrics.jsonl')
        time_records, _ = ReadLines(run / 'timings.jsonl')
        for row in time_records:
            timings.append(dict(candidate=candidate, **row))
        entry = {'name': name, 'status': status, 'matched_control': MATCHED_CONTROLS[candidate],
                 'final_review': final_review,
                 'stage_result': stage, 'state': state,
                 'failed_action': failed, 'registration': registration, 'protocol': protocol,
                 'completed_actions_seconds': sum(x['elapsed_seconds'] for x in time_records),
                 'raw_updates_observed': len(raw), 'b_incomplete_trailing_raw_line_omitted': b_incomplete,
                 'validation': [], 'raw_epoch_means': []}
        for epoch in sorted(set(row['epoch'] for row in raw)):
            rows = [row for row in raw if row['epoch'] == epoch]
            values = {'candidate': candidate, 'matched_control': MATCHED_CONTROLS[candidate],
                      'epoch': epoch + 1, 'updates_observed': len(rows),
                      'b_complete_1481_updates': len(rows) == 1481,
                      'first_iter': rows[0]['iter'], 'last_iter': rows[-1]['iter'],
                      'lr_first': rows[0]['learning_rate'], 'lr_last': rows[-1]['learning_rate'],
                      'total_loss_mean': sum(row['total_loss'] for row in rows) / len(rows)}
            for index, task in enumerate(('det', 'terrain', 'occ')):
                values[task + '_raw_loss_mean'] = sum(row['tasks'][index]['loss'] for row in rows) / len(rows)
            losses.append(values)
            entry['raw_epoch_means'].append(values)
        for file in sorted((run / 'validation').glob('epoch_*.json'), key=lambda f: int(f.stem.split('_')[-1])):
            record = Read(file)
            epoch = record['epoch']
            metrics = record['metrics']
            gate = Read(run / f'acceptance/epoch_{epoch}.json', {})
            # 只有本检查点对应的三个任务进入同一行，不挑不同最佳轮拼表。
            flat = {'candidate': candidate, 'matched_control': MATCHED_CONTROLS[candidate],
                    'epoch': epoch, 'gate_status': gate.get('status', 'pending'),
                    'gate_reasons': '; '.join(gate.get('reasons', [])),
                    'checkpoint': gate.get('checkpoint', str(run / f'checkpoints/epoch_{epoch}.pth')),
                    'checkpoint_sha256': gate.get('checkpoint_sha256'),
                    'validation_sha256': SHA(file)}
            for task, values in metrics.items():
                flat.update({task + '_' + key: value for key, value in values.items()
                             if isinstance(value, (int, float)) and not isinstance(value, bool)})
            flat['det_frame_count'] = metrics['det']['ground_truth_audit']['frame_count']
            validations.append(flat)
            entry['validation'].append({'record': record, 'gate': gate})
        snapshot['candidates'][candidate] = entry
        last = entry['validation'][-1] if entry['validation'] else None
        reasons = '; '.join(last['gate'].get('reasons', [])) if last else ''
        candidate_rows.append([html.escape(name), html.escape(status), str(last['record']['epoch']) if last else '—',
                               Number(last['record']['metrics']['det']['map_l2'], True) if last else '—',
                               f'{entry["completed_actions_seconds"] / 60:.2f}', html.escape(reasons or '未触发停止条件')])
        part = f'<h2>{html.escape(name)}</h2><p><strong>{status}</strong>；公平对照：{MATCHED_CONTROLS[candidate]}。已落盘动作耗时 {entry["completed_actions_seconds"] / 60:.2f} 分钟（进行中的动作尚未计入），观察到 raw update {len(raw)} 条。</p>'
        if final_review:
            part += '<p>' + Link(f'完整运行记录/{candidate}/final_review.json','当前最终验收状态及其绑定') + ' · ' + Link(final_review['external_final_acceptance'],'外部最终验收记录') + ' · ' + Link(f'完整运行记录/{candidate}/stage_result.json','原始程序退出状态（保留不改，不代表当前待办）') + '</p>'
        if registration:
            part += '<p>' + html.escape(registration.get('comparison', '')) + '</p>'
            part += '<p>事前假设：' + html.escape(registration.get('hypothesis', '')) + '</p>'
            part += Link(f'准备与诊断/candidate_{candidate}_registration.json', '实验注册与预算')
        part += '<p>初始化 S：<code>' + html.escape(protocol.get('source', '未知')) + '</code><br>SHA256：<code>' + html.escape(protocol.get('source_sha256', '未知')) + '</code></p>'
        part += '<p>' + Link(f'完整运行记录/{candidate}/screen_protocol.json', '协议/配置及标签绑定') + ' · ' + Link(f'完整运行记录/{candidate}/manifest.json', '运行清单') + '</p>'
        metric_rows, class_rows, root_rows = [], [], []
        for result in entry['validation']:
            record, gate = result['record'], result['gate']
            epoch, m = record['epoch'], record['metrics']
            det, terrain, occ = m['det'], m['terrain'], m['occ']
            metric_rows.append([str(epoch), str(det['ground_truth_audit']['frame_count']), Number(det['map_l2'], True), Number(det['maph_l2'], True), Number(terrain['mae_m']), Number(terrain['rmse_m']), Number(occ['occupancy_iou'], True), Number(occ['semantic_miou'], True), html.escape(gate.get('status', 'pending')), Link(f'完整运行记录/{candidate}/validation/epoch_{epoch}.json', '完整数值'), Link(f'完整运行记录/{candidate}/checkpoints/epoch_{epoch}.pth', '权重')])
            class_rows.append([str(epoch)] + [Number(det[f'{cls}_{metric}_l2'], True) for cls in ('vehicle', 'pedestrian', 'cyclist') for metric in ('ap', 'aph')])
            root_rows.append([str(epoch)] + [Number(occ.get(key), True) for key in ('deletion_fraction','occupied_root_recall','deletion_precision','candidate_gt_coverage')])
        part += Table(['轮次','目标数','mAP L2 %','mAPH L2 %','地形 MAE m','RMSE m','OCC IoU %','语义 mIoU %','门禁','指标','检查点'], metric_rows)
        part += Table(['轮次','车辆 AP %','车辆 APH %','行人 AP %','行人 APH %','骑行者 AP %','骑行者 APH %'], class_rows)
        part += Table(['轮次','根删除率 %','占据根保留召回 %','删除精确率 %','候选覆盖 %'], root_rows)
        part += '<details><summary>原始 loss（rank0；按轮均值，活动轮可能未完成）</summary>' + Table(['轮次','已记录步数','完整轮','检测 raw','地形 raw','OCC raw','首 LR','末 LR'], [[str(x['epoch']),str(x['updates_observed']),str(x['b_complete_1481_updates']),Number(x['det_raw_loss_mean']),Number(x['terrain_raw_loss_mean']),Number(x['occ_raw_loss_mean']),f"{x['lr_first']:.9g}",f"{x['lr_last']:.9g}"] for x in entry['raw_epoch_means']]) + '</details>'
        for result in entry['validation']:
            epoch = result['record']['epoch']
            directory = run / f'images/epoch_{epoch:02d}'
            part += f'<details><summary>第 {epoch} 轮固定训练/验证样本与损失图</summary><div class="gallery">'
            for image in sorted(directory.glob('*.png')):
                relative = f'完整运行记录/{candidate}/images/epoch_{epoch:02d}/{image.name}'
                title = ('训练样本' if image.name.startswith('train') else '验证样本' if image.name.startswith('val') else '原始损失曲线') + ' · ' + image.name
                part += f'<figure>{Link(relative,title)}<img loading="lazy" src="{relative}" alt="{html.escape(title)}"></figure>'
            part += '</div>'
            if (directory / 'duplicate_diagnostics.json').exists():
                part += Link(f'完整运行记录/{candidate}/images/epoch_{epoch:02d}/duplicate_diagnostics.json','后 NMS 重叠框诊断（不等同于已证明误检）')
            part += '</details>'
        sections.append(part)
    history_section, history_evidence = HistoryDiagnostic(output, runs)
    if history_evidence:
        snapshot['same_f5_history_diagnostic'] = history_evidence
    root_section, root_evidence = RootAblation(output, prep)
    if root_evidence:
        snapshot['controlled_root_ablation'] = root_evidence
    Write(output / 'results_snapshot.json', snapshot)
    CSV(output / 'validation_metrics.csv', validations)
    CSV(output / 'raw_loss_by_epoch.csv', losses)
    CSV(output / 'action_timings.csv', timings)
    intro = '<h1>X4 检测筛选：最多五轮</h1><p>快照时间：' + snapshot['snapshot_at'] + '</p><p class="warning">本轮仍在验证方案。A/B/C 的真实停止状态与 F 新对照进度见下表。第 0 轮恢复基线只说明初始预测接近，不能说明训练后已解决检测问题，更不能作为最终论文效果。</p>'
    intro += '<p>本表统一使用补框新真值、XYZT、完整训练候选池 23,688（每轮 seed+epoch 动态抽 11,844，八卡补齐4）、固定验证 5,973、48m 范围、0.05m 输入/0.2m 输出、无增强。共享单帧和候选均从同一 S 初始化；新的优化器，沿用48轮 OneCycle 前缀，但本轮最多5轮。检测表列 L2；完整 L1/L2 和各任务原始统计保存在 CSV/JSON。AP/IoU 在 HTML 为百分数，CSV/JSON 为0～1比例；地形以米计。</p>'
    intro += '<p>标签协议为 completed_detection_l2_v1：原框加补框，新增框 difficulty=2，保留真实零点数；验证有效补框1,783，其中零 TOP 点1,310。旧标签48轮结果没有混入本表。每一行三任务指标来自同一次检查点验证。</p>'
    intro += '<p>门禁：相对共享第0轮 mAP L2 下降≥5个百分点，或单类 AP 低于基线25%且绝对下降≥10个百分点，则停止；缺指标/非有限/目标数或绑定不符则待核查。门禁通过仅允许下一轮，不宣布有效。</p>'
    intro += '<p><strong>F 的公平对照是新增 single_f，不是旧 single 第5轮。</strong>single_f 与 F 均从同一预训练 S 重新初始化并重置优化器，各自最多5轮；不继承旧 single5 或 C1。F 相对 C 仅固定预训练主干和两个 BEV 投影器的 BN 运行统计，BN affine 仍可训练，其他新分支仍可训练；single_f 使用完全相同的 BN 策略但无历史。先完成 single_f，再运行 F。旧 single0～5 仅与 A/B/C 匹配。</p>'
    if effectiveness_decision:
        intro += '<p class="warning"><strong>F 已通过本轮短预算检测筛选，结构与预定第5轮检查点固定。</strong>相对 single_f5：mAP ' + f"{effectiveness_decision['map_delta_pp']:+.4f}" + ' 个百分点、mAPH ' + f"{effectiveness_decision['maph_delta_pp']:+.4f}" + ' 个百分点。五轮收益有波动、单种子，尚未确认稳定或统计显著提升；地形也未全面改善。此结论不授权长训练，不再增加检测候选。</p><p>' + Link('准备与诊断/f_effectiveness_decision.json','正式筛选决策与固定检查点SHA') + ' · ' + Link('准备与诊断/f_effectiveness_review.json','独立效果审查及限制') + '</p>'
    intro += '<p><strong>F_no_root 仅与 F 比较：不删根，保留相同根辅助监督。</strong>它从同一 S 重新初始化，保留相同种子、BN策略、LR前缀、动态采样及每轮完整验证，最多5轮并继续应用5个百分点停止线；不从F5继续训练。唯一方法配置变化为 <code>model.occ.b_apply_root_filter=false</code>，复用已有前向选择开关；<code>b_enable_root_filter</code>仍为true，根打分器、根BCE及权重、全部S参数、扩根和OCC分类损失定义均保留。根辅助损失数值会随训练而变化，但定义不变。未采用旧架构开关 <code>b_enable_root_filter=false</code>，因为它会同时移除根打分器和根监督；本轮结果不得与该旧实验混作同一消融。尚无正式结果时不声明完成。</p>'
    if (prep / 'fairness_review_no_root.json').exists():
        intro += '<p>' + Link('准备与诊断/fairness_review_no_root.json','根关闭消融公平性与损失语义审查') + '</p>'
    intro += Table(['实验','状态','最新完整验证轮','mAP L2 %','已完成动作分钟','停止原因'],candidate_rows)
    intro += '<p>' + ' · '.join(Link(f,label) for f,label in [('results_snapshot.json','完整快照JSON'),('validation_metrics.csv','同检查点全部验证CSV'),('raw_loss_by_epoch.csv','逐轮原始loss CSV'),('action_timings.csv','实测动作耗时CSV'),('生成脚本.py','可重复生成脚本'),('链接检查.json','本页链接检查'),('SHA256.json','小文件SHA清单')]) + '</p>'
    intro += '<h2>已有诊断与解释边界</h2><p>NMS 已进行真实算子回放及独立 Shapely 核对，10例、30次类别调用的保留ID/顺序一致；车辆/行人/骑行者阈值为0.7/0.2/0.25。该核验使用历史模型缓存，仅作为后处理机制诊断，不是本轮新标签对照成绩。无证据表明可把所有邻近框当作误检，也不能用 NMS 核验替代检测效果。</p>'
    intro += '<p>' + Link('../2026-10-08_X4筛选前_重复框与NMS核验/index.html','NMS真实核验报告') + ' · ' + Link('准备与诊断/shared_single5_final_cpu_review.json','共享单帧五轮独立验收') + ' · ' + Link('准备与诊断/a_epoch0_degradation_review.json','A退化与初始化核查') + ' · ' + Link('准备与诊断/c_route_cpu_review.json','C读取路径与两例特征混合诊断') + '</p>'
    intro += '<p>A/B分别检验完整F4和当前支持点处F4；C保留历史注意力后的E4、绕过检测读取处的额外历史值合并，静态分支仍读取F4。两例中 E4→当前F4 改变量大于注意力残差只支持这个定向假设，不能作为全验证退化的因果定论。共享单帧第5轮也存在OCC和地形回落；以下保留所有轮次及失败图，不拼接不同最佳轮。</p>'
    intro += '<p>C 首轮真实完整验证已早停。后续两个固定样本的反事实诊断显示：仅替换已训练 attention 参数也能在保留旧 BN 时引发特征放大；同时恢复 attention 和预训练 BN 可改善局部框数，但不等于完整 mAP 或三任务恢复。因此不能仅归因 BN，也不预设 F 有效。</p><p>' + ' · '.join(Link('准备与诊断/' + filename, label) for filename, label in [('c_epoch1_collapse_review.json','C首轮崩溃CPU验收'),('c_bn_counterfactual/summary.json','BN反事实诊断'),('c_attention_counterfactual/interpretation.json','attention反事实诊断及限制')]) + '</p>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>X4检测筛选实验记录</title><style>body{max-width:1400px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#20334a}a{color:#06679b}h2{margin-top:42px}table{border-collapse:collapse;white-space:nowrap;font-size:14px;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf3f6}.scroll{overflow:auto;margin:18px 0}.warning{background:#fff0d6;padding:18px;border-left:5px solid #e6a329}.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:15px}figure{margin:0}img{width:100%;height:auto}code{overflow-wrap:anywhere}details{margin:16px 0;border:1px solid #ddd;padding:12px}summary{cursor:pointer;font-weight:600}</style>' + intro + history_section + root_section + ''.join(sections) + '</html>'
    (output / 'index.html').write_text(document)
    shutil.copyfile(__file__, output / '生成脚本.py')
    # 先创建本页所链接的验收清单，再验证所有本地href/src均实际存在。
    Write(output / '链接检查.json', {})
    Write(output / 'SHA256.json', {})
    parser = LocalLinks(); parser.feed(document)
    links = sorted(set(parser.links))
    missing = [link for link in links if not urlsplit(link).scheme and not (output / unquote(urlsplit(link).path)).exists()]
    Write(output / '链接检查.json', {'checked_at':snapshot['snapshot_at'],'unique_local_links':len(links),'missing':missing,'b_pass':not missing,'scope':'Existence only; no weight copy or GPU inference'})
    Write(output / 'SHA256.json', {f.name:SHA(f) for f in output.iterdir() if f.is_file() and not f.is_symlink() and f.name!='SHA256.json'})
    if missing:
        raise RuntimeError(f'Missing report links: {missing}')
    index = output.parent / '索引.html'
    target = REPORT_NAME + '/index.html'
    if index.exists() and target not in index.read_text():
        content = index.read_text()
        item = '<h2>2026-10-08至09 X4 检测筛选</h2><p>' + Link(target,'最多五轮：A/B/C筛选结果与F公平对照（按带时间快照更新）') + '</p>'
        index.write_text(content.replace('</html>', item + '</html>'))
    elif index.exists():
        index.write_text(index.read_text().replace('最多五轮：共享单帧、A/B早停与C进行中（按带时间快照更新）',
                                                  '最多五轮：A/B/C筛选结果与F公平对照（按带时间快照更新）'))
    print(json.dumps({'report':str(output/'index.html'),'validation_rows':len(validations),'raw_loss_rows':len(losses),'local_links':len(links),'missing_links':missing,'snapshot_at':snapshot['snapshot_at']},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PROJECT.parent / '实验报告' / REPORT_NAME)
    Build(parser.parse_args().output.resolve())
