_base_ = [
    './bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py'
]

# Tiny startup config for checking that BEVFusion LiDAR+Camera can build,
# read NuScenes-mini data, run forward, and produce losses on one GPU.
data_root = 'data/nuscenes/'
ann_train_file = 'nuscenes_mini_infos_train.pkl'
ann_val_file = 'nuscenes_mini_infos_val.pkl'
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None

model = dict(
    img_backbone=dict(init_cfg=None),
    data_preprocessor=dict(voxelize_cfg=dict(max_voxels=[30000, 40000])))

train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=1,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[1.0, 1.0],
        rot_range=[0.0, 0.0],
        translation_std=0.0),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow',
            'pcd_rotation', 'pcd_scale_factor', 'pcd_trans',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=1,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=True,
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[256, 704],
        resize_lim=[0.48, 0.48],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='Pack3DDetInputs',
        keys=['img', 'points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=ann_train_file,
        pipeline=train_pipeline,
        metainfo=dict(
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ]),
        modality=input_modality,
        test_mode=False,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            sweeps='sweeps/LIDAR_TOP'),
        use_valid_flag=True,
        box_type_3d='LiDAR',
        indices=8,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NuScenesDataset',
        data_root=data_root,
        ann_file=ann_val_file,
        pipeline=test_pipeline,
        metainfo=dict(
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ]),
        modality=input_modality,
        data_prefix=dict(
            pts='samples/LIDAR_TOP',
            CAM_FRONT='samples/CAM_FRONT',
            CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
            CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
            CAM_BACK='samples/CAM_BACK',
            CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
            CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
            sweeps='sweeps/LIDAR_TOP'),
        test_mode=True,
        box_type_3d='LiDAR',
        indices=2,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + ann_val_file,
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=1, val_interval=1000)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00002, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0,
        by_epoch=False,
        begin=0,
        end=1)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=1))
custom_hooks = []
auto_scale_lr = dict(enable=False, base_batch_size=1)
log_processor = dict(window_size=1)
