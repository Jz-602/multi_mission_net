import numpy as np
import tensorflow as tf
import torch
import torch.distributed as dist
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

from ..utils.progressbar import ProgressBar
from WaymoDatasetV2 import WAYMO_INT_TO_TYPE_DICT, label_to_type
from occ_metrics import SummarizeMetrics as SummarizeOCCMetrics
from terrain_head import TERRAIN_DISTANCE_BINS_M, TerrainLoss
from terrain_metrics import TERRAIN_UNWEIGHTED_STATISTICS, TERRAIN_VALIDITY_STATISTICS, SummarizeTerrainValidity

try:
    tf.config.set_visible_devices([], 'GPU')
except (RuntimeError, ValueError):
    pass


def BuildSegmentationConfusionMatrix(point_pred, point_label, num_classes, ignore_index):
    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.float64,
        device=point_pred.device,
    )

    b_valid = (point_label >= 0) & (point_label < num_classes)
    if ignore_index is not None:
        b_valid &= (point_label != ignore_index)

    if not torch.any(b_valid):
        return confusion_matrix

    point_pred = point_pred[b_valid].to(torch.long)
    point_label = point_label[b_valid].to(torch.long)
    linear_indices = point_label * num_classes + point_pred
    confusion_matrix += torch.bincount(
        linear_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    return confusion_matrix


def ComputeSegmentationMetrics(confusion_matrix, class_names, ignore_index):
    num_classes = confusion_matrix.shape[0]
    class_names = [] if class_names is None else list(class_names)
    if len(class_names) < num_classes:
        class_names.extend(
            [f'class_{class_idx:02d}' for class_idx in range(len(class_names), num_classes)]
        )

    true_positive = torch.diag(confusion_matrix)
    gt_count = confusion_matrix.sum(dim=1)
    pred_count = confusion_matrix.sum(dim=0)
    union = gt_count + pred_count - true_positive

    class_iou = torch.zeros(num_classes, dtype=torch.float64, device=confusion_matrix.device)
    class_acc = torch.zeros(num_classes, dtype=torch.float64, device=confusion_matrix.device)

    b_valid_iou = union > 0
    b_valid_acc = gt_count > 0
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        b_valid_iou[ignore_index] = False
        b_valid_acc[ignore_index] = False

    class_iou[b_valid_iou] = true_positive[b_valid_iou] / union[b_valid_iou]
    class_acc[b_valid_acc] = true_positive[b_valid_acc] / gt_count[b_valid_acc]

    miou = class_iou[b_valid_iou].mean().item() if torch.any(b_valid_iou) else 0.0
    macc = class_acc[b_valid_acc].mean().item() if torch.any(b_valid_acc) else 0.0
    all_acc_denominator = gt_count.sum().item()
    all_acc = true_positive.sum().item() / all_acc_denominator if all_acc_denominator > 0 else 0.0

    class_result_list = []
    for class_idx, class_name in enumerate(class_names[:num_classes]):
        class_result_list.append({
            'class_idx': class_idx,
            'class_name': class_name,
            'iou': class_iou[class_idx].item(),
            'acc': class_acc[class_idx].item(),
            'support': int(gt_count[class_idx].item()),
            'b_valid_iou': bool(b_valid_iou[class_idx].item()),
            'b_valid_acc': bool(b_valid_acc[class_idx].item()),
        })

    return {
        'miou': miou,
        'macc': macc,
        'all_acc': all_acc,
        'class_result_list': class_result_list,
    }


def FormatSegmentationMetrics(epoch, metric_dict):
    message_lines = [
        f'Validation Epoch {epoch} Segmentation',
        (
            f"mIoU: {metric_dict['miou'] * 100.0:.2f} | "
            f"mAcc: {metric_dict['macc'] * 100.0:.2f} | "
            f"allAcc: {metric_dict['all_acc'] * 100.0:.2f}"
        ),
        '| Class                | IoU (%) |',
        '|----------------------|---------|',
    ]
    for class_result in metric_dict['class_result_list']:
        if class_result['b_valid_iou']:
            message_lines.append(
                f"| {class_result['class_name']:<20} | {class_result['iou'] * 100.0:7.2f} |"
            )
    return '\n'.join(message_lines)


def FormatDetectionMetrics(epoch, metric_dict):
    message_lines = [
        f'Validation Epoch {epoch} Detection',
        (
            f"mAP_L2: {metric_dict['map_l2']:.4f} | "
            f"mAPH_L2: {metric_dict['maph_l2']:.4f} | "
            f"mAP_L1: {metric_dict['map_l1']:.4f} | "
            f"mAPH_L1: {metric_dict['maph_l1']:.4f}"
        ),
        '| Class        | AP_L2  | APH_L2 | AP_L1  | APH_L1 |',
        '|--------------|--------|--------|--------|--------|',
    ]
    for class_name in ['vehicle', 'pedestrian', 'cyclist']:
        message_lines.append(
            f"| {class_name:<12} | {metric_dict[f'{class_name}_ap_l2']:.4f} | "
            f"{metric_dict[f'{class_name}_aph_l2']:.4f} | "
            f"{metric_dict[f'{class_name}_ap_l1']:.4f} | "
            f"{metric_dict[f'{class_name}_aph_l1']:.4f} |"
        )
    return '\n'.join(message_lines)


def LimitPeriod(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


def SummarizeCompletedDetectionGroundTruth(gt_infos):
    """记录ROI内GT实际点数/来源；派生标签评价不称为原始Waymo benchmark。"""
    summary = {
        'protocol': 'completed_detection_l2_v1',
        'definition': ('Waymo 3D Hungarian AP/APH; original GT eligibility unchanged; '
                       'supplemental GT eligible with difficulty=2 including zero real LiDAR points; '
                       'existing Waymo matching/difficulty behavior retained at both levels; '
                       'derived-label evaluation, not original Waymo benchmark'),
        'frame_count': len(gt_infos),
        'original_count': 0,
        'supplemental_count': 0,
        'original_eligible_count': 0,
        'supplemental_eligible_count': 0,
        'original_zero_point_count': 0,
        'supplemental_zero_point_count': 0,
        'real_point_count_histogram': {'original': {}, 'supplemental': {}},
        'eligible_by_class': {name: 0 for name in ('vehicle', 'pedestrian', 'cyclist')},
    }
    for info in gt_infos:
        for obj in info['gt_objects']:
            if obj['type_name'] not in summary['eligible_by_class']:
                continue
            b_supplemental = bool(obj.get('b_is_supplemental', False))
            source = 'supplemental' if b_supplemental else 'original'
            point_count = int(obj['num_points_in_top_lidar'])
            if b_supplemental and int(obj['difficulty']) != 2:
                raise ValueError('completed_detection_l2_v1 requires supplemental difficulty=2')
            summary[f'{source}_count'] += 1
            summary[f'{source}_zero_point_count'] += int(point_count == 0)
            histogram = summary['real_point_count_histogram'][source]
            key = str(point_count)
            histogram[key] = histogram.get(key, 0) + 1
            if b_supplemental or point_count > 0:
                summary[f'{source}_eligible_count'] += 1
                summary['eligible_by_class'][obj['type_name']] += 1
    return summary


class WaymoDetectionMetricsEstimator(tf.test.TestCase):
    WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'truck', 'cyclist']

    def GenerateWaymoTypeResults(self, infos, class_names, b_is_gt=False,
                                b_use_completed_detection_labels=False):
        frame_id_list = []
        boxes_3d_list = []
        obj_type_list = []
        score_list = []
        overlap_nlz_list = []
        difficulty_list = []

        for frame_index, info in enumerate(infos):
            if b_is_gt:
                gt_objects = info['gt_objects']
                box_name = np.array([obj['type_name'] for obj in gt_objects])
                num_points_in_gt = np.array([obj['num_points_in_top_lidar'] for obj in gt_objects])
                difficulty = np.array([obj['difficulty'] for obj in gt_objects], dtype=np.int8)
                boxes_3d = np.array([obj['box'][:7] for obj in gt_objects], dtype=np.float32).reshape(-1, 7)

                box_mask = np.array([name in class_names for name in box_name], dtype=np.bool_)
                zero_difficulty_mask = difficulty == 0
                difficulty[(num_points_in_gt > 5) & zero_difficulty_mask] = 1
                difficulty[(num_points_in_gt <= 5) & zero_difficulty_mask] = 2
                b_eligible = num_points_in_gt > 0
                if b_use_completed_detection_labels:
                    # 补框是真实目标帧LiDAR坐标系下的派生GT；不改写实际点数。
                    # 只扩展补框的评价资格，原始零点框和难度推断规则不变。
                    b_eligible |= np.array(
                        [obj.get('b_is_supplemental', False) for obj in gt_objects],
                        dtype=np.bool_,
                    )
                box_mask &= b_eligible

                box_name = box_name[box_mask]
                boxes_3d = boxes_3d[box_mask][:, :7]
                difficulty = difficulty[box_mask]
                score = np.ones(len(box_name), dtype=np.float32)
            else:
                box_name = np.array(info['name'])
                boxes_3d = info['boxes_lidar'][:, :7].astype(np.float32)
                difficulty = np.zeros(len(box_name), dtype=np.int8)
                score = info['score'].astype(np.float32)

            num_boxes = len(box_name)
            if num_boxes == 0:
                continue

            obj_type_list.append(np.array([self.WAYMO_CLASSES.index(name) for name in box_name], dtype=np.uint8))
            frame_id_list.append(np.full(num_boxes, frame_index, dtype=np.int64))
            boxes_3d_list.append(boxes_3d)
            score_list.append(score)
            overlap_nlz_list.append(np.zeros(num_boxes, dtype=np.float32))
            difficulty_list.append(difficulty)

        if len(boxes_3d_list) == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.uint8),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int8),
            )

        frame_id = np.concatenate(frame_id_list).reshape(-1).astype(np.int64)
        boxes_3d = np.concatenate(boxes_3d_list, axis=0).astype(np.float32)
        obj_type = np.concatenate(obj_type_list).reshape(-1).astype(np.uint8)
        score = np.concatenate(score_list).reshape(-1).astype(np.float32)
        overlap_nlz = np.concatenate(overlap_nlz_list).reshape(-1).astype(np.float32)
        difficulty = np.concatenate(difficulty_list).reshape(-1).astype(np.int8)

        boxes_3d[:, -1] = LimitPeriod(boxes_3d[:, -1], offset=0.5, period=np.pi * 2)
        return frame_id, boxes_3d, obj_type, score, overlap_nlz, difficulty

    def BuildConfig(self, b_collect_detection_range_metrics=False):
        config = metrics_pb2.Config()
        config_text = '''
        breakdown_generator_ids: OBJECT_TYPE
        difficulties {
            levels: 1
            levels: 2
        }
        matcher_type: TYPE_HUNGARIAN
        iou_thresholds: 0.0
        iou_thresholds: 0.7
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        iou_thresholds: 0.5
        box_type: TYPE_3D
        '''
        for score_cutoff in range(100):
            config.score_cutoffs.append(score_cutoff * 0.01)
        config.score_cutoffs.append(1.0)
        text_format.Merge(config_text, config)
        if b_collect_detection_range_metrics:
            text_format.Merge('breakdown_generator_ids: RANGE difficulties { levels: 1 levels: 2 }', config)
        return config

    def BuildGraph(self, graph, b_collect_detection_range_metrics=False):
        with graph.as_default():
            self.prediction_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self.prediction_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self.prediction_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self.prediction_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self.prediction_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)
            self.gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self.gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self.gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self.gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)

            return detection_metrics.get_detection_metric_ops(
                config=self.BuildConfig(b_collect_detection_range_metrics),
                prediction_frame_id=self.prediction_frame_id,
                prediction_bbox=self.prediction_bbox,
                prediction_type=self.prediction_type,
                prediction_score=self.prediction_score,
                prediction_overlap_nlz=self.prediction_overlap_nlz,
                ground_truth_bbox=self.gt_bbox,
                ground_truth_type=self.gt_type,
                ground_truth_frame_id=self.gt_frame_id,
                ground_truth_difficulty=self.gt_difficulty,
            )

    def SummarizeDistanceMetrics(self, raw_metrics, gt_boxes, gt_types, gt_difficulty, pd_boxes, pd_types):
        """[N,7] target-LiDAR XYZ/LWH/heading; official RANGE uses 3D center norm.

        Support is descriptive only: all boxes reach the unchanged official op.
        Difficulty<=level counts describe unmatched-FN eligibility, not a manual
        GT prefilter (matched L2 GT can contribute to installed-op L1 metrics).
        """
        ranges = (('[0, 30)', 0., 30.), ('[30, 50)', 30., 50.), ('[50, +inf)', 50., None))
        gt_distance = np.linalg.norm(gt_boxes[:, :3].astype(np.float64), axis=1)
        pd_distance = np.linalg.norm(pd_boxes[:, :3].astype(np.float64), axis=1)
        output = {
            'protocol': 'installed_waymo_official_RANGE_TYPE_3D',
            'distance_definition': 'sqrt(center_x^2 + center_y^2 + center_z^2), meters, target-frame LiDAR origin; 3D radial distance, not BEV radius',
            'matching': 'OBJECT_TYPE and RANGE breakdowns evaluated together by the same official Hungarian 3D operator on the identical full prediction/GT arrays; no manual distance prefilter',
            'zero_support_rule': 'AP/APH=None when this class/range contains zero eligible GT across difficulties; raw operator values retained. Difficulty<=level support is descriptive, not a prefilter or claimed exact AP denominator.',
            'macro_rule': 'Range mAP/mAPH average only classes with nonzero GT support; empty range is None. Main OBJECT_TYPE mAP/mAPH unchanged.',
            'ranges': [],
        }
        for label, lower, upper in ranges:
            b_gt_range = (gt_distance >= lower) & ((gt_distance < upper) if upper is not None else True)
            b_pd_range = (pd_distance >= lower) & ((pd_distance < upper) if upper is not None else True)
            entry = {'range': label, 'lower_m_inclusive': lower, 'upper_m_exclusive': upper, 'classes': {}}
            for name in ('vehicle', 'pedestrian', 'cyclist'):
                type_id = self.WAYMO_CLASSES.index(name)
                b_gt_class = b_gt_range & (gt_types == type_id)
                support = int(b_gt_class.sum())
                values = {'gt_count': support,
                          'prediction_count': int((b_pd_range & (pd_types == type_id)).sum())}
                for level in (1, 2):
                    values[f'gt_difficulty_le_l{level}_count'] = int((b_gt_class & (gt_difficulty <= level)).sum())
                    for metric in ('AP', 'APH'):
                        key = f'RANGE_TYPE_{name.upper()}_{label}_LEVEL_{level}/{metric}'
                        raw = float(raw_metrics[key])
                        values[f'raw_operator_{metric.lower()}_l{level}'] = raw
                        values[f'{metric.lower()}_l{level}'] = raw if support else None
                entry['classes'][name] = values
            entry['supported_classes'] = [name for name, values in entry['classes'].items() if values['gt_count']]
            for level in (1, 2):
                for metric in ('ap', 'aph'):
                    values = [entry['classes'][name][f'{metric}_l{level}'] for name in entry['supported_classes']]
                    entry[f'm{metric}_l{level}'] = float(np.mean(values)) if values else None
            output['ranges'].append(entry)
        return output

    def Evaluate(self, prediction_infos, gt_infos, b_use_completed_detection_labels=False,
                 b_collect_detection_range_metrics=False):
        tf.compat.v1.disable_eager_execution()
        session_config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0},
            allow_soft_placement=True,
        )

        pd_frame_id, pd_boxes_3d, pd_type, pd_score, pd_overlap_nlz, _ = self.GenerateWaymoTypeResults(
            prediction_infos,
            class_names=['vehicle', 'pedestrian', 'cyclist'],
            b_is_gt=False,
        )
        gt_frame_id, gt_boxes_3d, gt_type, _, _, gt_difficulty = self.GenerateWaymoTypeResults(
            gt_infos,
            class_names=['vehicle', 'pedestrian', 'cyclist'],
            b_is_gt=True,
            b_use_completed_detection_labels=b_use_completed_detection_labels,
        )

        if pd_boxes_3d.shape[0] == 0 and not b_collect_detection_range_metrics:
            return {
                'vehicle_ap_l1': 0.0,
                'vehicle_aph_l1': 0.0,
                'pedestrian_ap_l1': 0.0,
                'pedestrian_aph_l1': 0.0,
                'cyclist_ap_l1': 0.0,
                'cyclist_aph_l1': 0.0,
                'map_l1': 0.0,
                'maph_l1': 0.0,
                'vehicle_ap_l2': 0.0,
                'vehicle_aph_l2': 0.0,
                'pedestrian_ap_l2': 0.0,
                'pedestrian_aph_l2': 0.0,
                'cyclist_ap_l2': 0.0,
                'cyclist_aph_l2': 0.0,
                'map_l2': 0.0,
                'maph_l2': 0.0,
            }

        if pd_score.size and pd_score.max() > 1:
            pd_score = 1 / (1 + np.exp(-pd_score))

        graph = tf.Graph()
        metrics = self.BuildGraph(graph, b_collect_detection_range_metrics)

        # 这个是利用gpu跑val
        # with self.test_session(graph=graph) as sess:
        # 这个是利用cpu跑val
        with self.test_session(graph=graph, config=session_config, use_gpu=False) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            sess.run(
                [tf.group([value[1] for value in metrics.values()])],
                feed_dict={
                    self.prediction_bbox: pd_boxes_3d,
                    self.prediction_frame_id: pd_frame_id,
                    self.prediction_type: pd_type,
                    self.prediction_score: pd_score,
                    self.prediction_overlap_nlz: pd_overlap_nlz,
                    self.gt_bbox: gt_boxes_3d,
                    self.gt_type: gt_type,
                    self.gt_frame_id: gt_frame_id,
                    self.gt_difficulty: gt_difficulty,
                },
            )
            with tf.compat.v1.variable_scope('detection_metrics', reuse=True):
                metric_name_list = list(metrics.keys())
                metric_value_list = sess.run([metrics[metric_name][0] for metric_name in metric_name_list])
                raw_metric_dict = dict(zip(metric_name_list, metric_value_list))

        vehicle_ap_l1 = float(raw_metric_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'])
        vehicle_aph_l1 = float(raw_metric_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH'])
        pedestrian_ap_l1 = float(raw_metric_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP'])
        pedestrian_aph_l1 = float(raw_metric_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH'])
        cyclist_ap_l1 = float(raw_metric_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP'])
        cyclist_aph_l1 = float(raw_metric_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH'])
        vehicle_ap_l2 = float(raw_metric_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'])
        vehicle_aph_l2 = float(raw_metric_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH'])
        pedestrian_ap_l2 = float(raw_metric_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP'])
        pedestrian_aph_l2 = float(raw_metric_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH'])
        cyclist_ap_l2 = float(raw_metric_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP'])
        cyclist_aph_l2 = float(raw_metric_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH'])

        result = {
            'vehicle_ap_l1': vehicle_ap_l1,
            'vehicle_aph_l1': vehicle_aph_l1,
            'pedestrian_ap_l1': pedestrian_ap_l1,
            'pedestrian_aph_l1': pedestrian_aph_l1,
            'cyclist_ap_l1': cyclist_ap_l1,
            'cyclist_aph_l1': cyclist_aph_l1,
            'map_l1': float(np.mean([vehicle_ap_l1, pedestrian_ap_l1, cyclist_ap_l1])),
            'maph_l1': float(np.mean([vehicle_aph_l1, pedestrian_aph_l1, cyclist_aph_l1])),
            'vehicle_ap_l2': vehicle_ap_l2,
            'vehicle_aph_l2': vehicle_aph_l2,
            'pedestrian_ap_l2': pedestrian_ap_l2,
            'pedestrian_aph_l2': pedestrian_aph_l2,
            'cyclist_ap_l2': cyclist_ap_l2,
            'cyclist_aph_l2': cyclist_aph_l2,
            'map_l2': float(np.mean([vehicle_ap_l2, pedestrian_ap_l2, cyclist_ap_l2])),
            'maph_l2': float(np.mean([vehicle_aph_l2, pedestrian_aph_l2, cyclist_aph_l2])),
        }
        if b_collect_detection_range_metrics:
            result['distance_metrics'] = self.SummarizeDistanceMetrics(
                raw_metric_dict, gt_boxes_3d, gt_type, gt_difficulty, pd_boxes_3d, pd_type)
        return result


def _ToNumber(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return value


def BuildTerrainFrameStatistics(terrain_result, batch_dict, batch_index, criterion=None):
    criterion = TerrainLoss() if criterion is None else criterion
    sample_batch = {
        field_name: batch_dict[field_name][batch_index:batch_index + 1]
        for field_name in (
            'terrain_height',
            'terrain_valid_mask',
            'terrain_train_weight',
            'terrain_inpaint_distance',
        )
    }
    if 'terrain_ignore_mask' in batch_dict:
        sample_batch['terrain_ignore_mask'] = batch_dict['terrain_ignore_mask'][batch_index:batch_index + 1]
    outputs = {'terrain_height_pred': terrain_result['height']}
    if 'validity_logits' in terrain_result:
        outputs['terrain_validity_logits'] = terrain_result['validity_logits']
    _, logs, _ = criterion.ComputeLoss(outputs, sample_batch)
    statistic_names = [
        'terrain_valid_count',
        'terrain_weight_sum',
        'terrain_abs_error_sum',
        'terrain_squared_error_sum',
    ]
    for bin_name, _, _ in TERRAIN_DISTANCE_BINS_M:
        prefix = f'terrain_distance_{bin_name}'
        statistic_names.extend([
            f'{prefix}_valid_count',
            f'{prefix}_weight_sum',
            f'{prefix}_abs_error_sum',
            f'{prefix}_squared_error_sum',
        ])
    statistic_names.extend(TERRAIN_UNWEIGHTED_STATISTICS)
    if 'terrain_classification_count' in logs:
        statistic_names.extend(TERRAIN_VALIDITY_STATISTICS)
    return {name: _ToNumber(logs[name]) for name in statistic_names}


def SummarizeTerrainMetrics(frame_statistics):
    original_frames = list(frame_statistics)

    def Summarize(frames, prefix):
        valid_count = int(sum(frame[f'{prefix}_valid_count'] for frame in frames))
        weight_sum = float(sum(frame[f'{prefix}_weight_sum'] for frame in frames))
        abs_error_sum = float(sum(frame[f'{prefix}_abs_error_sum'] for frame in frames))
        squared_error_sum = float(sum(frame[f'{prefix}_squared_error_sum'] for frame in frames))
        return {
            'valid_count': valid_count,
            'weight_sum': weight_sum,
            'abs_error_sum': abs_error_sum,
            'squared_error_sum': squared_error_sum,
            'mae_m': abs_error_sum / weight_sum if weight_sum > 0.0 else None,
            'rmse_m': float(np.sqrt(squared_error_sum / weight_sum)) if weight_sum > 0.0 else None,
        }

    summary = Summarize(original_frames, 'terrain')
    summary['frame_count'] = len(original_frames)

    distance_bins = {}
    for bin_name, _, _ in TERRAIN_DISTANCE_BINS_M:
        prefix = f'terrain_distance_{bin_name}'
        bin_frames = []
        for original_frame in original_frames:
            bin_frames.append({
                f'{prefix}_{suffix}': original_frame[f'{prefix}_{suffix}']
                for suffix in ('valid_count', 'weight_sum', 'abs_error_sum', 'squared_error_sum')
            })
        distance_bins[bin_name] = Summarize(bin_frames, prefix)
    summary['distance_bins'] = distance_bins
    summary['distance_definition'] = 'distance to original terrain observation used for inpainting, not distance to vehicle'
    summary.update(SummarizeTerrainValidity(original_frames))
    return summary


def AccumulateBatchRecords(token_records, batch_result, batch_dict, terrain_criterion=None):
    for batch_index, token in enumerate(batch_dict['token']):
        frame_result = batch_result[token]
        det_result = frame_result.get('det', {})
        token_records[token] = {
            'det': {
                'box3d_lidar': det_result.get('box3d_lidar', torch.zeros((0, 7), dtype=torch.float32)).cpu(),
                'scores': det_result.get('scores', torch.zeros((0,), dtype=torch.float32)).cpu(),
                'label_preds': det_result.get('label_preds', torch.zeros((0,), dtype=torch.long)).cpu(),
            },
            'terrain': BuildTerrainFrameStatistics(
                frame_result['terrain'], batch_dict, batch_index, terrain_criterion
            ),
            'occ': frame_result['occ'].get('metrics', {}),
        }


def GatherTokenRecords(token_records, rank, world_size):
    if world_size <= 1:
        return token_records if rank == 0 else None
    dist.barrier()
    gathered_records = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_records, token_records)
    if rank != 0:
        return None
    merged_records = {}
    for rank_records in gathered_records:
        for token, record in rank_records.items():
            merged_records.setdefault(token, record)
    return merged_records


def _BuildDetectionInfos(dataset, token_records, point_cloud_range):
    roi_min = np.asarray(point_cloud_range[:3], dtype=np.float32)
    roi_max = np.asarray(point_cloud_range[3:], dtype=np.float32)
    gt_infos = []
    pred_infos = []
    for info in dataset.infos:
        token = info['token']
        if token not in token_records:
            continue
        det_result = token_records[token]['det']
        box_3d_lidar = det_result['box3d_lidar'].numpy()
        scores = det_result['scores'].numpy()
        label_preds = det_result['label_preds'].numpy()
        gt_object_list = []
        for gt_object in info['objects']:
            box = np.asarray(gt_object['box'], dtype=np.float32)
            if box.shape[0] < 7 or not np.all(np.isfinite(box[:7])):
                continue
            if not np.all((box[:3] >= roi_min) & (box[:3] < roi_max)):
                continue
            difficulty = 0
            if 'difficulty_level' in gt_object and len(gt_object['difficulty_level']) > 0:
                difficulty_value = gt_object['difficulty_level'][0]
                if not np.isnan(difficulty_value):
                    difficulty = int(difficulty_value)
            gt_object_list.append({
                'box': box,
                'type_name': WAYMO_INT_TO_TYPE_DICT[int(gt_object['type'])],
                'num_points_in_top_lidar': int(gt_object['num_points_in_top_lidar']),
                'difficulty': difficulty,
                'b_is_supplemental': bool(gt_object.get('b_is_supplemental', False)),
            })
        gt_infos.append({'token': token, 'gt_objects': gt_object_list})
        pred_infos.append({
            'name': [WAYMO_INT_TO_TYPE_DICT[label_to_type(int(label))] for label in label_preds],
            'score': scores,
            'boxes_lidar': box_3d_lidar,
            'sequence_name': token.split('-')[0],
            'frame_timestamp': int(token.split('-')[1]),
        })
    return pred_infos, gt_infos


def FinalizeThreeTaskMetrics(token_records, dataset, model_config, epoch, logger):
    pred_infos, gt_infos = _BuildDetectionInfos(
        dataset,
        token_records,
        list(model_config.point_cloud_range),
    )
    logger.info(
        'Validation Epoch %d: start CPU Waymo metric, frames=%d, pred_boxes=%d, gt_boxes=%d',
        epoch,
        len(pred_infos),
        int(sum(len(info['score']) for info in pred_infos)),
        int(sum(len(info['gt_objects']) for info in gt_infos)),
    )
    b_use_completed_detection_labels = bool(
        getattr(dataset, 'b_use_completed_detection_labels', False)
    )
    gt_audit = (SummarizeCompletedDetectionGroundTruth(gt_infos)
                if b_use_completed_detection_labels else None)
    det_metrics = WaymoDetectionMetricsEstimator().Evaluate(
        pred_infos, gt_infos,
        b_use_completed_detection_labels=b_use_completed_detection_labels,
        b_collect_detection_range_metrics=bool(model_config.get('b_collect_detection_range_metrics', False)),
    )
    if b_use_completed_detection_labels:
        det_metrics['ground_truth_audit'] = gt_audit
        logger.info('Completed detection GT protocol: %s', det_metrics['ground_truth_audit'])
    terrain_metrics = SummarizeTerrainMetrics(
        record['terrain'] for record in token_records.values()
    )
    occ_metrics = SummarizeOCCMetrics(
        record['occ'] for record in token_records.values() if record['occ']
    )
    logger.info(FormatDetectionMetrics(epoch, det_metrics))
    logger.info(
        'Validation Epoch %d Terrain: MAE=%s RMSE=%s valid=%d weight=%.3f',
        epoch,
        terrain_metrics['mae_m'],
        terrain_metrics['rmse_m'],
        terrain_metrics['valid_count'],
        terrain_metrics['weight_sum'],
    )
    logger.info(
        'Validation Epoch %d OCC: occupancy_iou=%s semantic_miou=%s frames=%d',
        epoch,
        occ_metrics.get('occupancy_iou'),
        occ_metrics.get('semantic_miou'),
        occ_metrics.get('frame_count', 0),
    )
    # 与manifest相同的dataset顺序，DDP补齐目标只统计一次。
    import hashlib
    expected_tokens = [entry['token'] for entry in dataset.infos]
    actual_tokens = [token for token in expected_tokens if token in token_records]
    unknown_tokens = sorted(set(token_records) - set(expected_tokens))
    audit = dict(unique_target_count=len(token_records), expected_target_count=len(expected_tokens),
                 token_sha256=hashlib.sha256('\n'.join(actual_tokens).encode()).hexdigest(),
                 expected_token_sha256=hashlib.sha256('\n'.join(expected_tokens).encode()).hexdigest(),
                 missing_count=len(set(expected_tokens) - set(token_records)), unknown_count=len(unknown_tokens),
                 b_complete=(len(set(expected_tokens)) == len(expected_tokens)
                             and set(token_records) == set(expected_tokens)))
    # 新论文协议保留逐帧可重算计数，旧入口默认不额外导出。
    artifact_dir = model_config.get('validation_artifact_dir')
    if artifact_dir:
        import gzip
        import json
        from pathlib import Path
        directory = Path(artifact_dir)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f'epoch_{epoch}_counts.jsonl.gz'
        temporary = destination.with_suffix('.tmp')
        with gzip.open(temporary, 'wt', encoding='utf-8') as stream:
            for token in actual_tokens:
                record = token_records[token]
                stream.write(json.dumps(dict(token=token, terrain=record['terrain'], occ=record['occ']),
                                        allow_nan=False) + '\n')
        temporary.replace(destination)
        audit['per_frame_counts'] = str(destination)
    return {'det': det_metrics, 'terrain': terrain_metrics, 'occ': occ_metrics, 'validation_audit': audit}


@torch.no_grad()
def ValidateMultiTask(model, val_dataloader, device, rank, world_size, epoch, logger):
    if val_dataloader is None:
        return None
    model.eval()
    model_module = getattr(model, 'module', model)
    terrain_criterion = getattr(model_module, 'terrain_criterion', TerrainLoss())
    token_records = {}
    progress_bar = ProgressBar(len(val_dataloader)) if rank == 0 else None
    for data_batch in val_dataloader:
        batch_dict = {
            key: (
                value if key == 'token'
                else [item.to(device, non_blocking=False) for item in value]
                if isinstance(value, list)
                else value.to(device, non_blocking=False)
            )
            for key, value in data_batch.items()
        }
        batch_result = model(batch_dict)
        AccumulateBatchRecords(token_records, batch_result, batch_dict, terrain_criterion)
        if progress_bar is not None:
            progress_bar.update()
    token_records = GatherTokenRecords(token_records, rank, world_size)
    if rank != 0:
        return None
    return FinalizeThreeTaskMetrics(
        token_records,
        val_dataloader.dataset,
        model_module.config,
        epoch,
        logger,
    )
