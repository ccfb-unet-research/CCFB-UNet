import os

norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

dataset_type = 'POLYPDataset'
crop_size = (256, 256)
img_scale = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            # Force exact scale so encoder strides stay even (avoid odd HxW crashes).
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = os.environ.get('DATA_ROOT', './polyp_datasets')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='train/images_albu',
            ann_dir='train/labels_albu',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/Kvasir/images',
        ann_dir='test/Kvasir/labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/Kvasir/images',
        ann_dir='test/Kvasir/labels',
        pipeline=test_pipeline))
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TensorboardLoggerHook'),
        dict(type='TextLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))
lr_config = dict(
    policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
    warmup='linear', warmup_iters=1500, warmup_ratio=1e-3)
runner = dict(
    type='IterBasedRunner',
    max_iters=28000)
checkpoint_config = dict(
    by_epoch=False,
    save_optimizer=False,
    interval=40000)
evaluation = dict(
    interval=2000,
    metric=['mIoU', 'mFscore', 'mDice'],
    pre_eval=False)
