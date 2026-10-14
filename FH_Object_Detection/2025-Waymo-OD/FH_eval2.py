import argparse
import pickle

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from waymo_open_dataset.metrics.python import detection_metrics
from waymo_open_dataset.protos import metrics_pb2


import math

tf.get_logger().setLevel('INFO')


def limit_period(val, offset=0.5, period=np.pi):
    return val - np.floor(val / period + offset) * period

TYPE_LIST = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

class WaymoDetectionMetricsEstimator(tf.test.TestCase):
    WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'truck', 'cyclist']

    def generate_waymo_type_results(self, infos, class_names, is_gt=False, fake_gt_infos=True, bev=False):
        def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
            """
            Args:
                boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

            Returns:
                boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
            """
            w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
            boxes3d_lidar[:, 2] += h[:, 0] / 2
            return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)

        frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty = [], [], [], [], [], []
        for frame_index, info in enumerate(infos):
            if is_gt:
                box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)
                if 'num_points_in_gt' in info:
                    zero_difficulty_mask = info['difficulty'] == 0
                    info['difficulty'][(info['num_points_in_gt'] > 5) & zero_difficulty_mask] = 1
                    info['difficulty'][(info['num_points_in_gt'] <= 5) & zero_difficulty_mask] = 2
                    nonzero_mask = info['num_points_in_gt'] > 0
                    box_mask = box_mask & nonzero_mask
                else:
                    print('Please provide the num_points_in_gt for evaluating on Waymo Dataset '
                          '(If you create Waymo Infos before 20201126, please re-create the validation infos '
                          'with version 1.2 Waymo dataset to get this attribute).')
                    raise NotImplementedError

                num_boxes = box_mask.sum()
                box_name = info['name'][box_mask]

                difficulty.append(info['difficulty'][box_mask])
                score.append(np.ones(num_boxes))
                if fake_gt_infos:
                    info['gt_boxes_lidar'] = boxes3d_kitti_fakelidar_to_lidar(info['gt_boxes_lidar'])

                if not bev:
                    boxes3d.append(info['gt_boxes_lidar'][box_mask][:, :7])
                else:
                    boxes3d.append(info['gt_boxes_lidar'][box_mask][:, [0, 1, 3, 4, 6]])
            else:
                num_boxes = len(info['boxes_lidar'])
                difficulty.append([0] * num_boxes)
                score.append(info['score'])
                if not bev:
                    boxes3d.append(info['boxes_lidar'][:, :7])
                else:
                    boxes3d.append(info['boxes_lidar'][:, [0, 1, 3, 4, 6]])
                box_name = info['name']

            obj_type += [self.WAYMO_CLASSES.index(name) for i, name in enumerate(box_name)]
            frame_id.append(np.array([frame_index] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.array(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = limit_period(boxes3d[:, -1], offset=0.5, period=np.pi * 2)

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty


    def FH_generate_waymo_type_results_V2(self, infos, class_names, is_gt=False, fake_gt_infos=True, bev=False):
        frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty = [], [], [], [], [], []
        for frame_index, tmp_info in enumerate(infos):
            info = {}
            info['name'] = np.array([TYPE_LIST[object['type']] for object in tmp_info['objects']])
            info['num_points_in_gt'] = np.array([object['num_points_in_top_lidar'] for object in tmp_info['objects']])
            info['difficulty'] = np.zeros(len(tmp_info['objects']))
            for i in range(len(tmp_info['objects'])):
                object = tmp_info['objects'][i]
                if not math.isnan(object['difficulty_level'][0]):
                    info['difficulty'][i] = object['difficulty_level'][0]

            info['gt_boxes_lidar'] = np.array([object['box'] for object in tmp_info['objects']])

            box_mask = np.array([n in class_names for n in info['name']], dtype=np.bool_)

            zero_difficulty_mask = info['difficulty'] == 0
            info['difficulty'][(info['num_points_in_gt'] > 5) & zero_difficulty_mask] = 1
            info['difficulty'][(info['num_points_in_gt'] <= 5) & zero_difficulty_mask] = 2
            nonzero_mask = info['num_points_in_gt'] > 0
            box_mask = box_mask & nonzero_mask


            num_boxes = box_mask.sum()
            box_name = info['name'][box_mask]

            difficulty.append(info['difficulty'][box_mask])
            score.append(np.ones(num_boxes))

            if not bev:
                boxes3d.append(info['gt_boxes_lidar'][box_mask][:, :7])
            else:
                boxes3d.append(info['gt_boxes_lidar'][box_mask][:, [0, 1, 3, 4, 6]])

            obj_type += [self.WAYMO_CLASSES.index(name) for i, name in enumerate(box_name)]
            frame_id.append(np.array([frame_index] * num_boxes))
            overlap_nlz.append(np.zeros(num_boxes))  # set zero currently

        frame_id = np.concatenate(frame_id).reshape(-1).astype(np.int64)
        boxes3d = np.concatenate(boxes3d, axis=0)
        obj_type = np.array(obj_type).reshape(-1)
        score = np.concatenate(score).reshape(-1)
        overlap_nlz = np.concatenate(overlap_nlz).reshape(-1)
        difficulty = np.concatenate(difficulty).reshape(-1).astype(np.int8)

        boxes3d[:, -1] = limit_period(boxes3d[:, -1], offset=0.5, period=np.pi * 2)

        return frame_id, boxes3d, obj_type, score, overlap_nlz, difficulty


    def build_config(self, config_type):
        config = metrics_pb2.Config()
        config_text = ''
        if 'object' in config_type:
            config_text += '''
            breakdown_generator_ids: OBJECT_TYPE
            difficulties {
                levels:1
                levels:2
            }
            '''

        if 'range' in config_type:
            config_text += '''
            breakdown_generator_ids: RANGE
            difficulties {
                levels:1
                levels:2
            }
            '''

        if '3d' in config_type:
            config_text += '''
            matcher_type: TYPE_HUNGARIAN
            iou_thresholds: 0.0
            iou_thresholds: 0.7
            iou_thresholds: 0.5
            iou_thresholds: 0.5
            iou_thresholds: 0.5
            box_type: TYPE_3D
            '''
        elif 'bev' in config_type:
            config_text += '''
            matcher_type: TYPE_HUNGARIAN
            iou_thresholds: 0.0
            iou_thresholds: 0.7
            iou_thresholds: 0.5
            iou_thresholds: 0.5
            iou_thresholds: 0.5
            box_type: TYPE_2D
            '''

        for x in range(0, 100):
            config.score_cutoffs.append(x * 0.01)
        config.score_cutoffs.append(1.0)

        text_format.Merge(config_text, config)
        return config

    def build_graph(self, graph, config_type):
        with graph.as_default():
            self._pd_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._pd_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._pd_score = tf.compat.v1.placeholder(dtype=tf.float32)
            self._pd_overlap_nlz = tf.compat.v1.placeholder(dtype=tf.bool)

            self._gt_frame_id = tf.compat.v1.placeholder(dtype=tf.int64)
            self._gt_bbox = tf.compat.v1.placeholder(dtype=tf.float32)
            self._gt_type = tf.compat.v1.placeholder(dtype=tf.uint8)
            self._gt_difficulty = tf.compat.v1.placeholder(dtype=tf.uint8)
            metrics = detection_metrics.get_detection_metric_ops(
                config=self.build_config(config_type),
                prediction_frame_id=self._pd_frame_id,
                prediction_bbox=self._pd_bbox,
                prediction_type=self._pd_type,
                prediction_score=self._pd_score,
                prediction_overlap_nlz=self._pd_overlap_nlz,
                ground_truth_bbox=self._gt_bbox,
                ground_truth_type=self._gt_type,
                ground_truth_frame_id=self._gt_frame_id,
                ground_truth_difficulty=self._gt_difficulty,
            )
            return metrics

    def run_eval_ops(
            self,
            sess,
            graph,
            metrics,
            prediction_frame_id,
            prediction_bbox,
            prediction_type,
            prediction_score,
            prediction_overlap_nlz,
            ground_truth_frame_id,
            ground_truth_bbox,
            ground_truth_type,
            ground_truth_difficulty,
    ):
        sess.run(
            [tf.group([value[1] for value in metrics.values()])],
            feed_dict={
                self._pd_bbox: prediction_bbox,
                self._pd_frame_id: prediction_frame_id,
                self._pd_type: prediction_type,
                self._pd_score: prediction_score,
                self._pd_overlap_nlz: prediction_overlap_nlz,
                self._gt_bbox: ground_truth_bbox,
                self._gt_type: ground_truth_type,
                self._gt_frame_id: ground_truth_frame_id,
                self._gt_difficulty: ground_truth_difficulty,
            },
        )

    def eval_value_ops(self, sess, graph, metrics):
        return {item[0]: sess.run([item[1][0]]) for item in metrics.items()}

    def mask_by_distance(self, distance_thresh, boxes_3d, *args):
        mask = np.linalg.norm(boxes_3d[:, 0:2], axis=1) < distance_thresh + 0.5
        boxes_3d = boxes_3d[mask]
        ret_ans = [boxes_3d]
        for arg in args:
            ret_ans.append(arg[mask])

        return tuple(ret_ans)

    def waymo_evaluation(self, prediction_infos, gt_infos, class_name, config_type=['object', '3d'],
                         distance_thresh=100, fake_gt_infos=True):
        print('Start the waymo evaluation...')
        assert len(prediction_infos) == len(gt_infos), '%d vs %d' % (prediction_infos.__len__(), gt_infos.__len__())

        tf.compat.v1.disable_eager_execution()
        pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz, _ = self.generate_waymo_type_results(
            prediction_infos, class_name, is_gt=False, bev='bev' in config_type
        )
        gt_frameid, gt_boxes3d, gt_type, gt_score, gt_overlap_nlz, gt_difficulty = self.FH_generate_waymo_type_results_V2(
            gt_infos, class_name, is_gt=True, fake_gt_infos=fake_gt_infos, bev='bev' in config_type
        )

        pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz = self.mask_by_distance(
            distance_thresh, pd_boxes3d, pd_frameid, pd_type, pd_score, pd_overlap_nlz
        )
        gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty = self.mask_by_distance(
            distance_thresh, gt_boxes3d, gt_frameid, gt_type, gt_score, gt_difficulty
        )

        print('Number: (pd, %d) VS. (gt, %d)' % (len(pd_boxes3d), len(gt_boxes3d)))
        print('Level 1: %d, Level2: %d)' % ((gt_difficulty == 1).sum(), (gt_difficulty == 2).sum()))

        if pd_score.max() > 1:
            # assert pd_score.max() <= 1.0, 'Waymo evaluation only supports normalized scores'
            pd_score = 1 / (1 + np.exp(-pd_score))
            print('Warning: Waymo evaluation only supports normalized scores')

        graph = tf.Graph()
        metrics = self.build_graph(graph, config_type)
        with self.test_session(graph=graph) as sess:
            sess.run(tf.compat.v1.initializers.local_variables())
            self.run_eval_ops(
                sess, graph, metrics, pd_frameid, pd_boxes3d, pd_type, pd_score, pd_overlap_nlz,
                gt_frameid, gt_boxes3d, gt_type, gt_difficulty,
            )
            with tf.compat.v1.variable_scope('detection_metrics', reuse=True):
                aps = self.eval_value_ops(sess, graph, metrics)
        return aps


if __name__ == '__main__':

    pred_infos = pickle.load(open('./results/epoch_36/waymo_preds.pkl', 'rb'))

    # gt_infos might contain more elements than pred_infos, the following code is to extract the corresponding gt_info for the pred_infos
    # gt_infos = pickle.load(open('/home/hao/dataset/waymo-small/FH/waymo_infos_train.pkl', 'rb'))
    gt_infos = pickle.load(open('/home/hao/dataset/waymo/FH/waymo_infos_validation.pkl', 'rb'))

    gt_sequence_names = [gt_info['sequence_name'] for gt_info in gt_infos]
    gt_frame_timestamps = [gt_info['frame_timestamp_micros'] for gt_info in gt_infos]
    idxes = []
    for k in range(len(pred_infos)):
        pred_info = pred_infos[k]
        idx = next((i for i, (x, y) in enumerate(zip(gt_sequence_names, gt_frame_timestamps)) if x == pred_info['sequence_name'] and y == pred_info['frame_timestamp']), None)
        if idx is None:
            print('Errors occurred!\n')
        idxes.append(idx)
    gt_infos = [gt_infos[idx] for idx in idxes]


    if len(pred_infos) > 50:
        pred_infos = pred_infos[:50]
        gt_infos = gt_infos[:50]

    print('Start to evaluate the waymo format results...')
    eval = WaymoDetectionMetricsEstimator()

    waymo_AP = eval.waymo_evaluation(
        pred_infos, gt_infos, class_name=['vehicle', 'pedestrian', 'cyclist'],
        # config_type=['object', '3d', 'range'],
        config_type=['object', '3d'],
        # config_type=['object', 'range', 'bev'],
        distance_thresh=1000, fake_gt_infos=True
    )


    if 0:
        for item in waymo_AP:
            print(f'{item}: {waymo_AP[item]}')
    else:
        item = 'OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'
        print(f'{item}: {waymo_AP[item]}')
        item = 'OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP'
        print(f'{item}: {waymo_AP[item]}')
        item = 'OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP'
        print(f'{item}: {waymo_AP[item]}')

