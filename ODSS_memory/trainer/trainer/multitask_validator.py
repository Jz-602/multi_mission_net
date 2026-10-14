import numpy as np
import tensorflow as tf
import torch
import torch.distributed as dist
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2

from ..utils.progressbar import ProgressBar
from WaymoDatasetV2 import WAYMO_INT_TO_TYPE_DICT, label_to_type


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


class WaymoDetectionMetricsEstimator(tf.test.TestCase):
    WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'truck', 'cyclist']

    def GenerateWaymoTypeResults(self, infos, class_names, b_is_gt=False):
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
                boxes_3d = np.array([obj['box'] for obj in gt_objects], dtype=np.float32)

                box_mask = np.array([name in class_names for name in box_name], dtype=np.bool_)
                zero_difficulty_mask = difficulty == 0
                difficulty[(num_points_in_gt > 5) & zero_difficulty_mask] = 1
                difficulty[(num_points_in_gt <= 5) & zero_difficulty_mask] = 2
                box_mask &= (num_points_in_gt > 0)

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

    def BuildConfig(self):
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
        return config

    def BuildGraph(self, graph):
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
                config=self.BuildConfig(),
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

    def Evaluate(self, prediction_infos, gt_infos):
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
        )

        if pd_boxes_3d.shape[0] == 0:
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

        if pd_score.max() > 1:
            pd_score = 1 / (1 + np.exp(-pd_score))

        graph = tf.Graph()
        metrics = self.BuildGraph(graph)

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

        return {
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


def ValidateMultiTask(model, val_dataloader, device, rank, world_size, epoch, logger):
    if val_dataloader is None:
        return None

    model.eval()
    model_module = getattr(model, 'module', model)
    model_config = getattr(model_module, 'config', None)
    num_classes = int(model_config.num_classes)
    ignore_index = getattr(model_config, 'ignore_index', None)
    class_names = getattr(model_config, 'class_names', None)

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.float64,
        device=device,
    )
    det_prediction_dict = {}

    if rank == 0:
        progress_bar = ProgressBar(len(val_dataloader))

    for batch_index, data_batch in enumerate(val_dataloader):
        # if batch_index >= 2:
        #     break
        batch_dict = {}
        for key, val in data_batch.items():
            if key == 'token':
                batch_dict[key] = val
            elif isinstance(val, list):
                batch_dict[key] = [item.to(device, non_blocking=False) for item in val]
            else:
                batch_dict[key] = val.to(device, non_blocking=False)

        batch_result = model(batch_dict)
        point_pred = torch.cat(
            [batch_result[token]['seg']['point_pred'].to(device) for token in batch_dict['token']],
            dim=0,
        )
        point_coords_xyz_all = batch_dict['points'][:, 1:4]
        point_cloud_range = point_coords_xyz_all.new_tensor(model_config.point_cloud_range)
        point_cloud_min_xyz = point_cloud_range[:3]
        point_cloud_max_xyz = point_cloud_range[3:]
        b_valid_mask = (
            (point_coords_xyz_all >= point_cloud_min_xyz)
            & (point_coords_xyz_all < point_cloud_max_xyz)
        ).all(dim=1)
        point_label = batch_dict['semantic_label'][b_valid_mask].to(device).long()
        confusion_matrix += BuildSegmentationConfusionMatrix(
            point_pred=point_pred,
            point_label=point_label,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        for token in batch_dict['token']:
            det_result = batch_result[token].get('det', {})
            det_prediction_dict[token] = {
                'box3d_lidar': det_result.get('box3d_lidar', torch.zeros((0, 7), dtype=torch.float32)),
                'scores': det_result.get('scores', torch.zeros((0,), dtype=torch.float32)),
                'label_preds': det_result.get('label_preds', torch.zeros((0,), dtype=torch.long)),
            }

        if rank == 0:
            progress_bar.update()

    if world_size > 1:
        dist.barrier()
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
        all_det_predictions = [None for _ in range(world_size)]
        dist.all_gather_object(all_det_predictions, det_prediction_dict)

    if rank != 0:
        return None

    if world_size > 1:
        merged_det_prediction_dict = {}
        for det_prediction_part in all_det_predictions:
            merged_det_prediction_dict.update(det_prediction_part)
    else:
        merged_det_prediction_dict = det_prediction_dict

    seg_metric_dict = ComputeSegmentationMetrics(
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        ignore_index=ignore_index,
    )

    dataset = val_dataloader.dataset
    gt_infos = []
    pred_infos = []
    for info in dataset.infos:
        token = info['token']
        if token not in merged_det_prediction_dict:
            continue

        det_result = merged_det_prediction_dict[token]
        box_3d_lidar = det_result['box3d_lidar'].detach().cpu().numpy()
        scores = det_result['scores'].detach().cpu().numpy()
        label_preds = det_result['label_preds'].detach().cpu().numpy()

        gt_object_list = []
        for gt_object in info['objects']:
            difficulty = 0
            if 'difficulty_level' in gt_object and len(gt_object['difficulty_level']) > 0:
                difficulty_val = gt_object['difficulty_level'][0]
                if not np.isnan(difficulty_val):
                    difficulty = int(difficulty_val)

            gt_object_list.append({
                'box': np.asarray(gt_object['box'], dtype=np.float32),
                'type_name': WAYMO_INT_TO_TYPE_DICT[int(gt_object['type'])],
                'num_points_in_top_lidar': int(gt_object['num_points_in_top_lidar']),
                'difficulty': difficulty,
            })

        gt_infos.append({
            'token': token,
            'gt_objects': gt_object_list,
        })
        pred_infos.append({
            'name': [WAYMO_INT_TO_TYPE_DICT[label_to_type(int(label))] for label in label_preds],
            'score': scores,
            'boxes_lidar': box_3d_lidar,
            'sequence_name': token.split('-')[0],
            'frame_timestamp': int(token.split('-')[1]),
        })

    num_pred_boxes = int(sum(len(info['score']) for info in pred_infos))
    num_gt_boxes = int(sum(len(info['gt_objects']) for info in gt_infos))
    metric_start_message = (
        f'Validation Epoch {epoch}: start Waymo detection metric, '
        f'frames={len(pred_infos)}, pred_boxes={num_pred_boxes}, gt_boxes={num_gt_boxes}'
    )
    logger.info(metric_start_message)

    det_metric_dict = WaymoDetectionMetricsEstimator().Evaluate(pred_infos, gt_infos)

    logger.info(FormatSegmentationMetrics(epoch, seg_metric_dict))
    logger.info(FormatDetectionMetrics(epoch, det_metric_dict))

    return {
        'seg': seg_metric_dict,
        'det': det_metric_dict,
    }
