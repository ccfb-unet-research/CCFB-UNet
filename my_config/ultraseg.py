_base_ = ['./_base_unet.py']

model = dict(
    type='EncoderDecoderFull',
    pretrained=None,
    decode_head=dict(
        type='UltraSeg130K',
        in_channels=3,
        num_classes=2,
        key=3,
        width_mult=2.5,  # ~[20,40,120,160,240] stronger than 130K
        use_deep_supervision=True,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=2.0)],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

work_dir = './work_dirs_polyp_v2/ultraseg'

data = dict(samples_per_gpu=16, workers_per_gpu=4)
optimizer = dict(type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=1e-4)
lr_config = dict(
    policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
    warmup='linear', warmup_iters=1000, warmup_ratio=1e-3)

runner = dict(type='IterBasedRunner', max_iters=28000)
checkpoint_config = dict(by_epoch=False, save_optimizer=False, interval=4000)
evaluation = dict(interval=2000, metric=['mIoU', 'mFscore', 'mDice'])
