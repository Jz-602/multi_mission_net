"""缓存原检测NMS输入及保留索引；GPU推理与独立Shapely复查分开执行。"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ('vehicle', 'pedestrian', 'cyclist')


def SaveJSON(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def BuildPolygons(boxes):
    import numpy as np
    import shapely

    # [N,7]目标车体系中心/XYZ尺寸（米）/yaw（弧度逆时针），只计算旋转BEV。
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    corners = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
    local = corners[None] * boxes[:, None, 3:5] / 2
    angle = boxes[:, 6, None]
    xy = np.stack((local[:,:,0]*np.cos(angle)-local[:,:,1]*np.sin(angle),
                   local[:,:,0]*np.sin(angle)+local[:,:,1]*np.cos(angle)), -1)
    return shapely.polygons(xy + boxes[:, None, :2])


def Crosscheck(output):
    import numpy as np
    import shapely
    from shapely import STRtree

    reviews = []
    for path in sorted(output.glob('*/nms_inputs.json')):
        case = json.loads(path.read_text())
        classes = []
        for record in case['classes']:
            # [N,7]为目标车体系cx,cy,cz,dx,dy,dz,yaw；长度米、yaw弧度逆时针。
            boxes = np.asarray(record['boxes'], dtype=np.float64).reshape(-1, 7)
            order = np.asarray(record['sort_order_pre'], dtype=np.int64)
            polygons = BuildPolygons(boxes)
            tree = STRtree(polygons)
            areas = shapely.area(polygons)
            b_removed = np.ones(len(boxes), dtype=bool)
            b_removed[order] = False
            kept = []
            for index in order:
                if b_removed[index]:
                    continue
                kept.append(int(index))
                neighbors = tree.query(polygons[index])
                neighbors = neighbors[~b_removed[neighbors]]
                overlap = shapely.area(shapely.intersection(polygons[index], polygons[neighbors]))
                iou = overlap / np.maximum(areas[index]+areas[neighbors]-overlap, 1e-8)
                b_removed[neighbors[iou > record['threshold']]] = True
                b_removed[index] = True
            kept = kept[:record['post_max_size']]
            actual = record['selected_class_indices']
            disagreements = sorted(set(kept).symmetric_difference(actual))
            boundary_pairs = []
            for index in disagreements:
                neighbors = tree.query(polygons[index])
                neighbors = neighbors[neighbors != index]
                overlap = shapely.area(shapely.intersection(polygons[index], polygons[neighbors]))
                iou = overlap / np.maximum(areas[index]+areas[neighbors]-overlap, 1e-8)
                for neighbor, value in zip(neighbors, iou):
                    if abs(value-record['threshold']) < .01:
                        boundary_pairs.append([index, int(neighbor), float(value)])
            classes.append({'class': record['class'], 'input_count': len(boxes),
                            'gpu_keep_count': len(actual), 'shapely_keep_count': len(kept),
                            'b_indices_and_order_match': kept == actual,
                            'different_indices': disagreements, 'near_threshold_pairs': boundary_pairs})
        review = {'case': path.parent.name, 'token': case['token'], 'classes': classes,
                  'shapely_version': shapely.__version__, 'python': sys.executable,
                  'note': 'Independent polygon greedy NMS uses exported torch order. Float32 kernel inclusion margin may differ near thresholds.'}
        SaveJSON(path.parent/'shapely_review.json', review)
        reviews.append(review)
    SaveJSON(output/'shapely_summary.json', reviews)
    print(json.dumps({'cases': len(reviews), 'mismatches': [r['case'] for r in reviews
          if not all(c['b_indices_and_order_match'] for c in r['classes'])]}))


def MatchCases(output, archive):
    import html
    import pickle
    import shutil
    import numpy as np
    import shapely
    import torch
    import yaml
    from scipy.optimize import linear_sum_assignment

    metadata = {path.parent.name: json.loads(path.read_text()) for path in output.glob('*/summary.json')}
    gt_by_token = {}
    for split in ('train', 'val'):
        cases = [value for value in metadata.values() if value['sample'].split(':')[0] == split]
        config = yaml.safe_load(Path(cases[0]['config']).read_text())
        infos_path = Path(config['common_variables']['root_path'])/config['data'][f'{split}_dataset']['info_path']
        requested = {value['token'] for value in cases}
        roi = np.asarray(config['common_variables']['pc_range'])
        with infos_path.open('rb') as stream:
            infos = pickle.load(stream)
        for info in infos:
            if info['token'] not in requested:
                continue
            objects = []
            for index, obj in enumerate(info['objects']):
                box = np.asarray(obj['box'], dtype=np.float64)[:7]
                if int(obj['type']) not in (1, 2, 4) or int(obj.get('num_points_in_top_lidar', 1)) <= 0:
                    continue
                if len(box) != 7 or not np.isfinite(box).all() or not ((box[:3]>=roi[:3]) & (box[:3]<roi[3:])).all():
                    continue
                objects.append({'object_index': index, 'label': {1:0, 2:1, 4:2}[int(obj['type'])], 'box': box.tolist()})
            gt_by_token[info['token']] = {'source_infos': str(infos_path), 'objects': objects}
        del infos
    assert set(gt_by_token) == {value['token'] for value in metadata.values()}
    cases = [];csv_rows = []
    for case_name, meta in sorted(metadata.items()):
        gt = gt_by_token[meta['token']]['objects'];variants = []
        for threshold in (.7, .6, .5):
            prediction = torch.load(output/case_name/f'veh_{threshold:.1f}.pt', map_location='cpu', weights_only=False)
            boxes = prediction['box3d_lidar'].numpy();labels = prediction['label_preds'].numpy()
            scores = prediction['scores'].numpy();b_visible = scores >= .15
            classes = []
            for label, class_name in enumerate(CLASS_NAMES):
                matching_threshold = .7 if label == 0 else .5
                gt_objects = [obj for obj in gt if obj['label'] == label]
                gt_polygons = BuildPolygons([obj['box'] for obj in gt_objects])
                pred_ids = np.where(b_visible & (labels==label))[0]
                pred_polygons = BuildPolygons(boxes[pred_ids])
                overlap = shapely.area(shapely.intersection(gt_polygons[:,None], pred_polygons[None,:]))
                union = shapely.area(gt_polygons)[:,None]+shapely.area(pred_polygons)[None,:]-overlap
                iou = overlap/np.maximum(union, 1e-8)
                b_valid = iou >= matching_threshold
                # 最大匹配数优先，再最大总IoU；独立局部BEV诊断，非官方AP或按分数扫阈值。
                gt_ids, pred_local_ids = linear_sum_assignment(b_valid*(max(iou.shape, default=0)+1+iou), maximize=True)
                matched = [{'gt_object_index': gt_objects[g]['object_index'], 'pred_index': int(pred_ids[p]),
                            'iou': float(iou[g,p]), 'score': float(scores[pred_ids[p]])}
                           for g,p in zip(gt_ids, pred_local_ids) if b_valid[g,p]]
                matched_gt = {value['gt_object_index'] for value in matched}
                matched_pred = {value['pred_index'] for value in matched}
                class_result = {'class': class_name, 'BEV_matching_threshold': matching_threshold,
                    'gt_count': len(gt_objects), 'visible_pred_count': len(pred_ids), 'matched_gt_count': len(matched),
                    'unmatched_gt_indices': [obj['object_index'] for obj in gt_objects if obj['object_index'] not in matched_gt],
                    'unmatched_prediction_indices': [int(i) for i in pred_ids if i not in matched_pred], 'matches': matched,
                    'candidate_exists_by_gt': {str(obj['object_index']): bool(b_valid[g].any()) for g,obj in enumerate(gt_objects)}}
                classes.append(class_result)
                csv_rows.append({'case': case_name, 'sample': meta['sample'], 'vehicle_NMS_threshold': threshold,
                    'class': class_name, 'BEV_matching_threshold': matching_threshold, 'GT': len(gt_objects),
                    'visible_predictions': len(pred_ids), 'matched_GT': len(matched),
                    'unmatched_predictions': len(pred_ids)-len(matched), 'unmatched_GT': len(gt_objects)-len(matched)})
            variants.append({'vehicle_NMS_threshold': threshold, 'classes': classes})
        baseline = variants[0]['classes'][0]
        baseline_gt = {value['gt_object_index'] for value in baseline['matches']}
        losses = []
        for variant in variants[1:]:
            strict = variant['classes'][0]
            lost = sorted(baseline_gt-{value['gt_object_index'] for value in strict['matches']})
            lost_details = []
            for index in lost:
                obj = next(value for value in gt if value['object_index']==index)
                neighbors = [value for value in gt if value['label']==0 and value['object_index']!=index]
                neighbor = min(neighbors, key=lambda value: np.linalg.norm(np.asarray(value['box'][:2])-obj['box'][:2]), default=None)
                lost_details.append({'gt_object_index': index, 'box': obj['box'],
                    'b_any_retained_prediction_still_reaches_matching_threshold': strict['candidate_exists_by_gt'][str(index)],
                    'nearest_other_vehicle_GT': neighbor,
                    'nearest_GT_center_distance_m': float(np.linalg.norm(np.asarray(neighbor['box'][:2])-obj['box'][:2])) if neighbor else None})
            losses.append({'vehicle_NMS_threshold': variant['vehicle_NMS_threshold'], 'lost_baseline_matched_GT': lost_details})
        cases.append({'case': case_name, 'sample': meta['sample'], 'token': meta['token'],
                      'original_GT': gt_by_token[meta['token']], 'variants': variants, 'strict_vehicle_match_losses': losses})
    review = {'protocol': 'Fixed5-target independent rotated BEV one-to-one diagnostic; NOT official Waymo AP.',
        'score_cutoff': .15, 'matching_thresholds': {'vehicle':.7, 'pedestrian':.5, 'cyclist':.5},
        'GT_filter': 'Original infos types1/2/4; positive LiDAR point count; finite [N,7] target-car boxes; half-open center ROI48m.',
        'assignment': 'Maximum match cardinality first, total IoU second, Hungarian. Compare retained candidate existence when identities switch.',
        'b_CPU_only': True, 'shapely_version': shapely.__version__, 'cases': cases,
        'limitations': 'Five fixed targets/model at score.15; BEV rather than official3D IoU, no difficulty/NLZ/AP-score sweep. Unmatched prediction is a local pairing result, not an official FP. No formal threshold change or population-level quality claim.'}
    SaveJSON(output/'case_matching.json', review)
    with (output/'case_matching.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    if archive is not None:
        archive.mkdir(parents=True, exist_ok=True)
        (archive/'raw_cache').symlink_to(output.resolve(), target_is_directory=True)
        for name in ('case_matching.json','case_matching.csv','counts.json','counts.csv','nms_gpu_review.json','shapely_summary.json'):
            shutil.copy2(output/name, archive/name)
        shutil.copy2(output.parent/'nms_static_review.json', archive/'nms_static_review.json')
        image_rows = []
        for case in cases:
            folder = archive/'images'/case['case'];folder.mkdir(parents=True)
            for threshold in (.7,.6,.5):
                shutil.copy2(output/case['case']/f'veh_{threshold:.1f}.png', folder/f'veh_{threshold:.1f}.png')
            vehicle_rows = [value for value in csv_rows if value['case']==case['case'] and value['class']=='vehicle']
            cells = ''.join(f'<td><p>NMS {row["vehicle_NMS_threshold"]}：GT {row["GT"]} / 匹配 {row["matched_GT"]} / 未匹配预测 {row["unmatched_predictions"]}</p><img src="images/{case["case"]}/veh_{row["vehicle_NMS_threshold"]:.1f}.png"></td>' for row in vehicle_rows)
            image_rows.append(f'<h2>{html.escape(case["case"])}</h2><p>{html.escape(case["token"])}</p><table><tr>{cells}</tr></table>')
        losses_count = sum(len(value['lost_baseline_matched_GT']) for case in cases for value in case['strict_vehicle_match_losses'])
        page = '<!doctype html><meta charset="utf-8"><title>重复框与NMS核验</title><style>body{font:16px sans-serif;margin:24px}table{width:100%;table-layout:fixed}td{vertical-align:top}img{width:100%}p{overflow-wrap:anywhere}</style><h1>2026-10-08 X4筛选前：重复框与NMS核验</h1><p>旧dynamic50 single48 / temporal48，固定train10与val0/100/300/2720。真实GPU NMS的30次类别保留索引/顺序与独立Shapely完全一致，未发现这批样本的实现错误；原阈值缓存重放与模型导出逐tensor一致。</p><p>车辆配对门槛BEV IoU≥0.7；仅显示score≥0.15。表中采用最大匹配数优先的一对一配对，不是官方Waymo AP、3D匹配或全5973评价。严格车辆NMS .6/.5仅为缓存诊断，正式阈值未改。</p>'+f'<p>相对.7出现{losses_count}条“基线已匹配GT在严格阈值下未匹配”的记录（含两个严格档分别计数）；具体邻近GT、仍有无合格候选见<a href="case_matching.json">配对证据JSON</a>。不能把减少重框直接解释为AP提升。</p><p><a href="case_matching.csv">局部计数CSV</a> · <a href="nms_gpu_review.json">真实GPU核验</a> · <a href="raw_cache/">原始缓存（符号链接）</a></p>'+''.join(image_rows)
        (archive/'index.html').write_text(page)
    print(json.dumps({'cases': len(cases), 'lost_vehicle_GT_records': sum(len(v['lost_baseline_matched_GT']) for c in cases for v in c['strict_vehicle_match_losses']), 'matching': str(output/'case_matching.json'), 'archive': str(archive)}))


def Run(args):
    import torch
    sys.path.insert(0, str(ROOT))
    import CenterHead
    from easydict import EasyDict
    from LYF_train_ODSS import CfgFromYamlFile, SetRandomSeed
    from run_options import CheckRunConfig
    from tools.visualize_three_task import (SingleModel, TemporalModel, WaymoDataset,
        WaymoDatasetMulti, ToDevice, PlotComparison)

    torch.set_num_threads(4)
    original_nms = CenterHead.rotate_nms_pcdet
    rows = []
    for stage_name in args.stages:
        b_temporal = stage_name == 'temporal'
        cfg_path = args.run_root/stage_name/'config.yaml'
        cfg = CfgFromYamlFile(str(cfg_path), EasyDict())
        CheckRunConfig(cfg)
        assert cfg.model.in_channels == 3
        assert cfg.model.post_processing.nms.nms_iou_threshold == [[.7], [.2, .25]]
        for split in ('train', 'val'):
            assert not cfg.data[f'{split}_dataset'].get('b_use_time_feature', False)
            assert not cfg.data[f'{split}_dataset'].get('b_use_completed_detection_labels', False)
        SetRandomSeed(0, 0)
        for key in ('tasks', 'out_size_factor', 'voxel_size'):
            cfg.model[key] = cfg.common_variables[key]
        cfg.model.point_cloud_range = cfg.common_variables.pc_range
        model = (TemporalModel if b_temporal else SingleModel)(cfg.model).cuda().eval()
        checkpoint_path = args.run_root/stage_name/'checkpoints/epoch_48.pth'
        checkpoint_bytes = checkpoint_path.read_bytes()
        checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        provenance = {'config': str(cfg_path.resolve()), 'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                      'checkpoint': str(checkpoint_path.resolve()), 'checkpoint_sha256': hashlib.sha256(checkpoint_bytes).hexdigest(),
                      'meta_epoch': checkpoint['meta']['epoch'], 'meta_iter': checkpoint['meta']['iter'],
                      'operator': CenterHead.iou3d_nms_cuda.__file__}
        del checkpoint, checkpoint_bytes
        for sample_spec in args.samples:
            split, index_text = sample_spec.split(':');index = int(index_text)
            destination = args.output/f'{stage_name}_{split}_{index:03}'
            if destination.exists():
                raise FileExistsError(destination)
            destination.mkdir(parents=True)
            SetRandomSeed(0, 0)
            dataset = (WaymoDatasetMulti if b_temporal else WaymoDataset)(cfg.data[f'{split}_dataset'], cfg.common_variables)
            sample = dataset[index];batch = dataset.CollateBatch([sample])
            target = batch['frames'][-1] if b_temporal else batch
            token = target['token'][0]
            info = next(v for v in dataset.infos if v['token'] == token)
            buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
            records = [];post_inputs = [];raw_heads = [];context = {}
            original_post = model.detection_head.post_processing

            def CapturePost(task_id, boxes, hm, test_cfg, center_range, iou):
                context.update(task_id=task_id, class_id=0, grid_indices=[])
                scores, labels = hm[0].max(-1)
                mask = (scores > test_cfg.score_threshold) & (boxes[0,:,:3] >= center_range[:3]).all(1) & (boxes[0,:,:3] <= center_range[3:]).all(1)
                for class_id in range(hm.shape[-1]):
                    context['grid_indices'].append(torch.where(mask & (labels == class_id))[0].cpu())
                post_inputs.append({'task_id': task_id, 'boxes': boxes.cpu(), 'hm': hm.cpu(), 'iou': iou.cpu()})
                return original_post(task_id, boxes, hm, test_cfg, center_range, iou)

            def CaptureNMS(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
                selected = original_nms(boxes, scores, thresh, pre_maxsize, post_max_size)
                order = scores.sort(descending=True)[1][:pre_maxsize]
                class_id = context['class_id'];global_label = class_id if context['task_id'] == 0 else class_id+1
                grid_indices = context['grid_indices'][class_id]
                position = {int(value): i for i, value in enumerate(order.cpu().tolist())}
                records.append({'class': CLASS_NAMES[global_label], 'global_label': global_label,
                    'task_id': context['task_id'], 'local_class_id': class_id, 'boxes': boxes.cpu(), 'scores': scores.cpu(),
                    'threshold': float(thresh), 'pre_max_size': pre_maxsize, 'post_max_size': post_max_size,
                    'sort_order_pre': order.cpu(), 'selected_class_indices': selected.cpu(),
                    'selected_sorted_positions': [position[int(value)] for value in selected.cpu().tolist()],
                    'dense_grid_indices': grid_indices, 'selected_dense_grid_indices': grid_indices[selected.cpu()]})
                context['class_id'] += 1
                return selected

            def CaptureHead(module, inputs, output):
                raw_heads.extend([{key: value.detach().cpu().clone() for key, value in task.items()} for task in output])

            hook = model.detection_head.register_forward_hook(CaptureHead)
            model.detection_head.post_processing = CapturePost
            CenterHead.rotate_nms_pcdet = CaptureNMS
            start = time.monotonic()
            try:
                with torch.no_grad():
                    memory = None
                    if b_temporal:
                        for frame in batch['frames'][:-1]:
                            memory = model(ToDevice(frame, 'cuda'), memory, b_memory_only=True)
                        result = model(ToDevice(target, 'cuda'), memory)
                    else:
                        result = model(ToDevice(target, 'cuda'))
            finally:
                hook.remove();model.detection_head.post_processing = original_post
                CenterHead.rotate_nms_pcdet = original_nms
            assert len(records) == 3 and len(raw_heads) == 2
            assert all(torch.equal(value, buffers[name]) for name, value in model.named_buffers())
            details = {**provenance, 'token': token, 'sample': sample_spec, 'b_temporal': b_temporal,
                       'inference_seconds': time.monotonic()-start, 'b_eval_buffers_unchanged': True}
            if b_temporal:
                details['clip'] = {'tokens': sample['token_seq'], 'history_interval': int(sample['history_interval'])}
            torch.save({'metadata': details, 'raw_heads_NCHW': raw_heads, 'decoded_pre_inputs': post_inputs,
                        'classes': records, 'result': result}, destination/'cache.pt')
            json_records = [{k: v.tolist() if torch.is_tensor(v) else v for k,v in record.items()} for record in records]
            SaveJSON(destination/'nms_inputs.json', {**details, 'classes': json_records})
            with torch.no_grad():
                for vehicle_threshold in (.7, .6, .5):
                    detections = {'box3d_lidar': [], 'scores': [], 'label_preds': []}
                    for record in records:
                        threshold = vehicle_threshold if record['global_label'] == 0 else record['threshold']
                        selected = original_nms(record['boxes'].cuda(), record['scores'].cuda(), threshold,
                                                record['pre_max_size'], record['post_max_size']).cpu()
                        detections['box3d_lidar'].append(record['boxes'][selected])
                        detections['scores'].append(record['scores'][selected])
                        detections['label_preds'].append(torch.full((len(selected),), record['global_label'], dtype=torch.long))
                    detections = {key: torch.cat(value) for key,value in detections.items()}
                    if vehicle_threshold == .7:
                        assert all(torch.equal(detections[key], result[token]['det'][key]) for key in detections)
                    comparison = dict(result[token]);comparison['det'] = detections
                    PlotComparison(comparison, target, info, cfg, destination/f'veh_{vehicle_threshold:.1f}.png', split, b_temporal)
                    torch.save(detections, destination/f'veh_{vehicle_threshold:.1f}.pt')
                    counts = [int(((detections['label_preds']==label) & (detections['scores']>=.15)).sum()) for label in range(3)]
                    rows.append({'stage': stage_name, 'sample': sample_spec, 'token': token,
                                 'vehicle_threshold': vehicle_threshold, 'post_count': len(detections['scores']),
                                 **dict(zip(CLASS_NAMES, counts))})
            SaveJSON(destination/'summary.json', {**details, 'counts': rows[-3:]})
            print(json.dumps({'case': destination.name, 'token': token, 'seconds': details['inference_seconds'], 'counts': rows[-3:]}), flush=True)
            del dataset, batch, sample, result, memory, buffers, post_inputs, raw_heads, records
        del model
        torch.cuda.empty_cache()
    for sample_spec in args.samples:
        tokens = {row['token'] for row in rows if row['sample'] == sample_spec}
        assert len(tokens) == 1, (sample_spec, tokens)
    SaveJSON(args.output/'counts.json', rows)
    with (args.output/'counts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def Main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, default=ROOT/'log/server_runs_dynamic50')
    parser.add_argument('--stages', nargs='+', choices=['single', 'temporal'], default=['single', 'temporal'])
    parser.add_argument('--samples', nargs='+', default=['train:10', 'val:0', 'val:100', 'val:300', 'val:2720'])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--crosscheck', dest='b_crosscheck', action='store_true', help='CPU only; requires independent Shapely2 interpreter')
    modes.add_argument('--match', dest='b_match', action='store_true', help='CPU only; pair cached detections with original GT')
    parser.add_argument('--archive', type=Path, help='Small report/images copy and raw-cache symlink; only with --match')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.b_match:
        MatchCases(args.output, args.archive)
    elif args.b_crosscheck:
        Crosscheck(args.output)
    else:
        Run(args)


if __name__ == '__main__':
    Main()
