_base_ = [
    # '../../_base_/models/faster_rcnn_r50_fpn.py',
    '../../_base_/models/yolox_x_8x8.py',
    '../../_base_/datasets/ht21_challenge.py',
    '../../_base_/default_runtime.py'
]

img_scale = (1080, 1920)
samples_per_gpu = 4

model = dict(
    type='DeepSORT',
    detector=dict(
        type='YOLOX',
        input_size=img_scale,
        random_size_range=(18, 32),
        random_size_interval=10,
        backbone=dict(
            type='CSPDarknet', deepen_factor=1.33, widen_factor=1.25),
        neck=dict(
            type='YOLOXPAFPN',
            in_channels=[320, 640, 1280],
            out_channels=320,
            num_csp_blocks=4),
        bbox_head=dict(
            type='YOLOXHead',
            num_classes=1,
            in_channels=320,
            feat_channels=320),
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.7)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
        )),
    motion=dict(type='KalmanFilter', center_only=False),
    reid=dict(
        type='BaseReID',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(3, ),
            style='pytorch'),
        neck=dict(type='GlobalAveragePooling', kernel_size=(8, 4), stride=1),
        head=dict(
            type='LinearReIDHead',
            num_fcs=1,
            in_channels=2048,
            fc_channels=1024,
            out_channels=128,
            num_classes=380,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            loss_pairwise=dict(
                type='TripletLoss', margin=0.3, loss_weight=1.0),
            norm_cfg=dict(type='BN1d'),
            act_cfg=dict(type='ReLU')),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/reid/tracktor_reid_r50_iter25245-a452f51f.pth'  # noqa: E501
        )),
    tracker=dict(
        type='SortTracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))

train_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        bbox_clip_border=False),
    dict(
        type='RandomAffine',
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        bbox_clip_border=False),
    dict(
        type='MixUp',
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        bbox_clip_border=False),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomCrop', crop_size=(108, 192)), #(1080, 1920)
    # dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=img_scale,
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
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
                '/opt/ml/final-project-level3-cv-04/mmtracking/data/HT21/annotations/half-train_cocoformat.json',
                # 'data/crowdhuman/annotations/crowdhuman_train.json',
                # 'data/crowdhuman/annotations/crowdhuman_val.json'
            ],
            img_prefix=[
                '/opt/ml/final-project-level3-cv-04/mmtracking/data/HT21/train', 
                # 'data/crowdhuman/train',
                # 'data/crowdhuman/val'
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

# optimizer
# default 8 gpu
optimizer = dict(
    type='AdamW',
    lr=0.0005,
    weight_decay=0.0005,
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=None)


# some hyper parameters
total_epochs = 50
resume_from = None
num_last_epochs = 1
interval = 10

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

custom_hooks = [
    dict(type='YOLOXModeSwitchHook', 
         num_last_epochs=num_last_epochs, 
         priority=48),
    dict(type='SyncNormHook', 
         num_last_epochs=num_last_epochs, 
         interval=interval, 
         priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]

evaluation = dict(metric=['bbox', 'track'], interval=10)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

fp16 = dict(loss_scale=dict(init_scale=512.0))

