_base_ = ['../dg/_fair_common.py']
model = dict(decode_head=dict(
    _delete_=True, type='UltraSeg130K', in_channels=3, num_classes=2,
    key=3, width_mult=2.5, use_deep_supervision=True,
    loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.8)]))
work_dir = './work_dirs_polyp_v2/reviewer/fair_ultraseg_scratch'

