_base_ = ['../dg/_fair_common.py']
model = dict(decode_head=dict(
    _delete_=True, type='DDCNet', num_classes=2, base_channels=64,
    pretrained_backbone=False,
    loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.8)]))
work_dir = './work_dirs_polyp_v2/reviewer/fair_ddcnet_scratch'

