import numpy as np
import tensorflow as tf
import torch
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2


def FormatDetectionMetrics(epoch, metric_dict):
    message_lines = [
        f"Validation Epoch {epoch} Detection",
        (
            f"mAP_L2: {metric_dict['map_l2']:.4f} | "
            f"mAPH_L2: {metric_dict['maph_l2']:.4f} | "
            f"mAP_L1: {metric_dict['map_l1']:.4f} | "
            f"mAPH_L1: {metric_dict['maph_l1']:.4f}"
        ),
        "| Class        | AP_L2  | APH_L2 | AP_L1  | APH_L1 |",
        "|--------------|--------|--------|--------|--------|",
    ]
    for class_name in ["vehicle", "pedestrian", "cyclist"]:
        message_lines.append(
            f"| {class_name:<12} | {metric_dict[f'{class_name}_ap_l2']:.4f} | "
            f"{metric_dict[f'{class_name}_aph_l2']:.4f} | "
            f"{metric_dict[f'{class_name}_ap_l1']:.4f} | "
            f"{metric_dict[f'{class_name}_aph_l1']:.4f} |"
        )
    return "\n".join(message_lines)


def BuildWaymoDetectionInputs(dataset_infos, prediction_dict, type_name_dict, label_to_type_function):
    expected_tokens = [info["token"] for info in dataset_infos]
    missing_tokens = [token for token in expected_tokens if token not in prediction_dict]
    if len(missing_tokens) > 0:
        raise RuntimeError(
            f"missing predictions for {len(missing_tokens)} validation frames; "
            f"first missing token: {missing_tokens[0]}"
        )

    gt_infos = []
    pred_infos = []
    for info in dataset_infos:
        token = info["token"]
        prediction = prediction_dict[token]
        boxes_lidar = prediction["box3d_lidar"]
        scores = prediction["scores"]
        label_preds = prediction["label_preds"]
        if isinstance(boxes_lidar, torch.Tensor):
            boxes_lidar = boxes_lidar.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(label_preds, torch.Tensor):
            label_preds = label_preds.detach().cpu().numpy()

        gt_objects = []
        for gt_object in info["objects"]:
            difficulty = 0
            difficulty_level = gt_object.get("difficulty_level", [])
            if len(difficulty_level) > 0 and not np.isnan(difficulty_level[0]):
                difficulty = int(difficulty_level[0])
            gt_objects.append({
                "box": np.asarray(gt_object["box"], dtype=np.float32),
                "type_name": type_name_dict[int(gt_object["type"])],
                "num_points_in_top_lidar": int(gt_object["num_points_in_top_lidar"]),
                "difficulty": difficulty,
            })

        gt_infos.append({"token": token, "gt_objects": gt_objects})
        pred_infos.append({
            "name": [type_name_dict[label_to_type_function(int(label))] for label in label_preds],
            "score": np.asarray(scores),
            "boxes_lidar": np.asarray(boxes_lidar),
            "sequence_name": token.split("-")[0],
            "frame_timestamp": int(token.split("-")[1]),
        })
    return pred_infos, gt_infos


def LimitPeriod(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period


class WaymoDetectionMetricsEstimator(tf.test.TestCase):
    WAYMO_CLASSES = ["unknown", "vehicle", "pedestrian", "truck", "cyclist"]

    def GenerateWaymoTypeResults(self, infos, class_names, b_is_gt=False):
        frame_id_list = []
        boxes_3d_list = []
        obj_type_list = []
        score_list = []
        overlap_nlz_list = []
        difficulty_list = []

        for frame_index, info in enumerate(infos):
            if b_is_gt:
                gt_objects = info["gt_objects"]
                box_name = np.array([obj["type_name"] for obj in gt_objects])
                num_points_in_gt = np.array([obj["num_points_in_top_lidar"] for obj in gt_objects])
                difficulty = np.array([obj["difficulty"] for obj in gt_objects], dtype=np.int8)
                boxes_3d = np.array([obj["box"] for obj in gt_objects], dtype=np.float32).reshape(-1, 7)
                box_mask = np.array([name in class_names for name in box_name], dtype=np.bool_)
                zero_difficulty_mask = difficulty == 0
                difficulty[(num_points_in_gt > 5) & zero_difficulty_mask] = 1
                difficulty[(num_points_in_gt <= 5) & zero_difficulty_mask] = 2
                box_mask &= num_points_in_gt > 0
                box_name = box_name[box_mask]
                boxes_3d = boxes_3d[box_mask][:, :7]
                difficulty = difficulty[box_mask]
                score = np.ones(len(box_name), dtype=np.float32)
            else:
                box_name = np.array(info["name"])
                boxes_3d = np.asarray(info["boxes_lidar"], dtype=np.float32).reshape(-1, 7)[:, :7]
                difficulty = np.zeros(len(box_name), dtype=np.int8)
                score = np.asarray(info["score"], dtype=np.float32).reshape(-1)

            num_boxes = len(box_name)
            if num_boxes == 0:
                continue
            obj_type_list.append(
                np.array([self.WAYMO_CLASSES.index(name) for name in box_name], dtype=np.uint8)
            )
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
        config_text = """
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
        """
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

    def EmptyMetrics(self):
        metric_dict = {}
        for class_name in ["vehicle", "pedestrian", "cyclist"]:
            for level in ["l1", "l2"]:
                metric_dict[f"{class_name}_ap_{level}"] = 0.0
                metric_dict[f"{class_name}_aph_{level}"] = 0.0
        for level in ["l1", "l2"]:
            metric_dict[f"map_{level}"] = 0.0
            metric_dict[f"maph_{level}"] = 0.0
        return metric_dict

    def Evaluate(self, prediction_infos, gt_infos):
        tf.compat.v1.disable_eager_execution()
        session_config = tf.compat.v1.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True)
        pd_results = self.GenerateWaymoTypeResults(
            prediction_infos,
            class_names=["vehicle", "pedestrian", "cyclist"],
            b_is_gt=False,
        )
        gt_results = self.GenerateWaymoTypeResults(
            gt_infos,
            class_names=["vehicle", "pedestrian", "cyclist"],
            b_is_gt=True,
        )
        pd_frame_id, pd_boxes_3d, pd_type, pd_score, pd_overlap_nlz, _ = pd_results
        gt_frame_id, gt_boxes_3d, gt_type, _, _, gt_difficulty = gt_results
        if pd_boxes_3d.shape[0] == 0:
            return self.EmptyMetrics()
        if pd_score.max() > 1:
            pd_score = 1 / (1 + np.exp(-pd_score))

        graph = tf.Graph()
        metrics = self.BuildGraph(graph)
        with self.test_session(graph=graph, config=session_config, use_gpu=False) as session:
            session.run(tf.compat.v1.initializers.local_variables())
            session.run(
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
            with tf.compat.v1.variable_scope("detection_metrics", reuse=True):
                metric_names = list(metrics.keys())
                metric_values = session.run([metrics[name][0] for name in metric_names])
                raw_metric_dict = dict(zip(metric_names, metric_values))

        metric_dict = {}
        waymo_class_names = {
            "vehicle": "VEHICLE",
            "pedestrian": "PEDESTRIAN",
            "cyclist": "CYCLIST",
        }
        for class_name, waymo_class_name in waymo_class_names.items():
            for level_number in [1, 2]:
                prefix = f"OBJECT_TYPE_TYPE_{waymo_class_name}_LEVEL_{level_number}"
                level_name = f"l{level_number}"
                metric_dict[f"{class_name}_ap_{level_name}"] = float(raw_metric_dict[f"{prefix}/AP"])
                metric_dict[f"{class_name}_aph_{level_name}"] = float(raw_metric_dict[f"{prefix}/APH"])
        for level_name in ["l1", "l2"]:
            metric_dict[f"map_{level_name}"] = float(np.mean([
                metric_dict[f"{class_name}_ap_{level_name}"]
                for class_name in waymo_class_names
            ]))
            metric_dict[f"maph_{level_name}"] = float(np.mean([
                metric_dict[f"{class_name}_aph_{level_name}"]
                for class_name in waymo_class_names
            ]))
        return metric_dict
