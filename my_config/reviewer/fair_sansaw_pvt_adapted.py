_base_ = ['./_reviewer_dg_common.py']
model = dict(decode_head=dict(dg_method='san_saw', dg_loss_weight=0.05))
work_dir = './work_dirs_polyp_v2/reviewer/fair_sansaw_pvt_adapted'

