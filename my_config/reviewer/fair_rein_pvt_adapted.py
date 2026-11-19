_base_ = ['./_reviewer_dg_common.py']
model = dict(decode_head=dict(dg_method='rein', dg_loss_weight=0.0))
work_dir = './work_dirs_polyp_v2/reviewer/fair_rein_pvt_adapted'

