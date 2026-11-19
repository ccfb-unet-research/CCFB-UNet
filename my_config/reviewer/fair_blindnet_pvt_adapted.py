_base_ = ['./_reviewer_dg_common.py']
model = dict(decode_head=dict(dg_method='blindnet', dg_loss_weight=0.10))
work_dir = './work_dirs_polyp_v2/reviewer/fair_blindnet_pvt_adapted'

