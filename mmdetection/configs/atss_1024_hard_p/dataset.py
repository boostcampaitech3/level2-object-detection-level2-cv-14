# dataset settings
# import path
dataset_type = 'CocoDataset'
data_root = 'dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type="RandomRotate90", p=1),
    dict(
        type="OneOf",
        transforms=[
            dict(type="HueSaturationValue", hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
            dict(type="RandomGamma"),
            dict(type="CLAHE"),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RandomBrightnessContrast", brightness_limit=0.25, contrast_limit=0.25),
            dict(type="RGBShift", r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="MotionBlur"),
            dict(type="GaussNoise"),
            dict(type="ImageCompression", quality_lower=75),
        ],
        p=0.4,
    ),
    dict(
        type="Cutout",
        num_holes = 15,
        max_h_size = 30,
        max_w_size = 30,
        fill_value = 0,
        p=0.4,
    ),

    dict(type="RandomSizedBBoxSafeCrop", height = 1024, width = 1024, erosion_rate=0.1, p = 0.1)
]    
img_scale=(1024, 1024)

custom_imports = dict(imports=['mmdet.datasets.pipelines.transforms'], allow_failed_imports=False)
train_pipeline = [
    dict(type='Mosaic_p', img_scale=img_scale, pad_val=0, p = 0.3),
    dict(
        type='MixUp_p',
        img_scale=img_scale,
        pad_val=0,
        p = 0.3
        ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
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

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
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
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train_fold0.json',
        img_prefix=data_root,
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=train_dataset,
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
