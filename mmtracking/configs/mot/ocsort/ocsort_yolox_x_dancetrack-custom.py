_base_ = [
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/dancetrack_img800.py', 
    '../../_base_/adamw_runtime.py'
]

## augment
# num_frames_retain 30 to 20
# add mosaic
# add random affine
# add border clip
# add rgb norm

img_scale = (648, 864) # ⭐ 600x800 to 648x864
samples_per_gpu = 4
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

model = dict(
    type='OCSORT',
    detector=dict(
        input_size=img_scale,
        random_size_range=(18, 32),
        bbox_head=dict(num_classes=1),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa: E501
        )
        ),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='OCSORTTracker',
        obj_score_thr=0.3, # default 0.3
        init_track_thr=0.7, # default 0.7 # exp 0.5, 0.9
        weight_iou_with_det_scores=True, # defalut True # exp False
        match_iou_thr=0.3, # default 0.3 # exp 0.6 
        num_tentatives=3, # default 3 # exp 1, 18 ⬅️
        vel_consist_weight=0.2, # default 0.2 # exp 0.1
        vel_delta_t=3, # default 3 # exp 1 9
        num_frames_retain=30, # default 30 # exp 15 60 75
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/opt/ml/final-project-level3-cv-04/pretrained_weight/ocsort_yolox_x_crowdhuman_mot17-private-half.pth'  # noqa: E501
        )
        ))

train_pipeline = [
    dict(
        type='Mosaic', # ⭐ add mosaic
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=True
        ),
    # dict(
    #     type='MinIoURandomCrop', # ⭐ add MinIoURandomCrop
    #     min_ious=(0.7, 0.8, 0.9),
    #     min_crop_size=0.8,
    #     bbox_clip_border=True),
    dict(
        type='RandomAffine', # ⭐ add random affine
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(type='YOLOXHSVRandomAug'),
    dict(
        type='Resize',
        img_scale=img_scale,
        keep_ratio=True,
        bbox_clip_border=True), # ⭐ change border clip
    dict(type='Normalize', **img_norm_cfg), # ⭐ add rgb norm
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', flip_ratio = 0.0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True), # ⭐ add rgb norm
            dict(
                type='Pad',
                size_divisor=32,
                pad_val=dict(img=(114.0, 114.0, 114.0))),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        _delete_=True,
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=[
                'data/dancetrack/annotations/train_cocoformat.json',
                'data/dancetrack/annotations/val_cocoformat.json'
            ],
            img_prefix=[
                'data/dancetrack/train',
                'data/dancetrack/val'
            ],
            classes=('pedestrian', ),
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False),
        pipeline=train_pipeline),
    val=dict(
        pipeline=test_pipeline,
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
    test=dict(
        pipeline=test_pipeline,
        interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)))


# some hyper parameters
total_epochs = 50
num_last_epochs = 50
resume_from = None
interval = 1

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=1,
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

# hooks
custom_hooks = [
    # dict(
    #     type='YOLOXModeSwitchHook',
    #     num_last_epochs=num_last_epochs,
    #     priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

# save config
checkpoint_config = dict(interval=interval)
evaluation = dict(metric=['bbox', 'track'], interval=100)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))
