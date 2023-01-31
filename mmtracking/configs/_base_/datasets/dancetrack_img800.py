# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/dancetrack/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(600, 800),
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomCrop', crop_size=(600, 800), bbox_clip_border=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        classes=('pedestrian', ),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_cocoformat.json',
        img_prefix=data_root + 'train',
        classes=('pedestrian', ),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_cocoformat.json',
        img_prefix=data_root + 'train',
        classes=('pedestrian', ),
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])