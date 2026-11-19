_base_ = ['../dg/_fair_common.py']
model = dict(decode_head=dict(
    _delete_=True, type='ResUNetPlusPlus', num_c=2, num_classes=2, base_c=48,
    loss_decode=[
        dict(type='CrossEntropyLoss', loss_name='loss_ce', use_sigmoid=False, loss_weight=1.0),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.8)]))
work_dir = './work_dirs_polyp_v2/reviewer/fair_resunetpp_scratch'

