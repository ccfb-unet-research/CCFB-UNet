_base_ = ['./_reviewer_dg_common.py']
model = dict(decode_head=dict(dg_method='robustnet', dg_loss_weight=0.10))
work_dir = './work_dirs_polyp_v2/reviewer/fair_robustnet_pvt_adapted'

