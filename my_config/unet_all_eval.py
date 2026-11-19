_base_ = ['./_base_unet.py']
model = dict(
    type='EncoderDecoderFull',
    pretrained=None,
    decode_head=dict(
        type='UnetOrig',
        in_channels=3,
        num_classes=2,
        base_channels=24,
        decoder_channels=48,
        use_fd_kan=True,
        use_mgd_encoder=True,
        use_gls_cams=True,
        init_fg_bias=True,
        ccfb_full_light_bridge=False,
        freeze_base_cnn=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.1)],
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = './work_dirs_polyp_v2/unet_all'
data = dict(samples_per_gpu=1, workers_per_gpu=2)
