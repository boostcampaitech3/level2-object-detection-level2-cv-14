# dataset settings

dataset_type = 'CocoDataset'
data_root = 'dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

multi_scale = [(w,w) for w in range(512, 1024+1, 32)]

albu_train_transforms = [
    dict(
        type='OneOf',
        transforms=[
            dict(type='Flip',p=1.0),
            dict(type='RandomRotate90',p=1.0)
        ],
        p = 0.1
    ),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(type='RandomResizedCrop',height=1024, width=1024, scale=(0.5, 1.0), p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='ChannelShuffle',
                p=1.0),
            dict(
                type='RandomGamma',
                p=1.0),
            dict(
                type='RGBShift',
                p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
    ]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', 
        img_scale=multi_scale,
        multiscale_mode='value',
        keep_ratio=True,
        ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
    type='Albu',
    transforms=albu_train_transforms,
    bbox_params=dict(
        type='BboxParams',
        format='pascal_voc',
        label_fields=['gt_labels'],
        min_visibility=0.0,
        filter_lost_elements=True),
    keymap={
        'img': 'image',
        'gt_bboxes': 'bboxes'
    },
    update_pad_shape=False,
    skip_img_without_anno=True
    ),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(w,w) for w in range(512, 1024+1, 128)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_fold0.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes,
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_fold0.json',
        img_prefix=data_root,
        pipeline=val_pipeline,
        classes=classes,
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes,
        ))
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
