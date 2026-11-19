"""Common, controlled protocol for DG/pretraining experiments.

Every child config inherits the same data augmentation, optimizer, loss,
batch size, evaluation cadence and 14k-iteration budget as unet_all.py.
Only the explicitly named model switch changes between experiments.
"""

_base_ = ['../_base_unet.py']

crop_size = (256, 256)
img_scale = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        img_scale=[(256, 256), (480, 480)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomRotate', prob=0.5, degree=30,
         pad_val=0, seg_pad_val=255),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='RandomFlip', flip_ratio=0.5, direction='vertical'),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=40,
        contrast_range=(0.4, 1.6),
        saturation_range=(0.4, 1.6),
        hue_delta=20),
    dict(
        type='RandomCutOut',
        prob=0.3,
        n_holes=(1, 3),
        cutout_shape=[(16, 16), (32, 32), (48, 48), (64, 64)],
        seg_fill_in=255),
    dict(type='Normalize', mean=[0, 0, 0], std=[1, 1, 1], to_rgb=True),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

model = dict(
    type='EncoderDecoderFull',
    pretrained=None,
    decode_head=dict(
        type='UnetOrig',
        in_channels=3,
        num_classes=2,
        base_channels=24,
        decoder_channels=48,
        use_fd_kan=False,
        use_mgd_encoder=False,
        use_gls_cams=False,
        use_ibn=False,
        use_mixstyle=False,
        use_pvt_control=False,
        init_fg_bias=True,
        ccfb_full_light_bridge=False,
        freeze_base_cnn=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce',
                 use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.8),
        ]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
)

optimizer = dict(
    type='AdamW', lr=2.0e-4, betas=(0.9, 0.999), weight_decay=5e-2)
lr_config = dict(
    policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
    warmup='linear', warmup_iters=300, warmup_ratio=1e-3)

runner = dict(type='IterBasedRunner', max_iters=14000)
checkpoint_config = dict(
    by_epoch=False, save_optimizer=False, interval=1000, max_keep_ckpts=20)
evaluation = dict(
    interval=1000,
    metric=['mIoU', 'mFscore', 'mDice'])

