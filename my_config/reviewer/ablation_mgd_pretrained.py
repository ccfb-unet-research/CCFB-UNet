_base_ = ['../dg/_fair_common.py']
model = dict(decode_head=dict(use_mgd_encoder=True, pretrained_pvt=None))
work_dir = './work_dirs_polyp_v2/reviewer/ablation_mgd_pretrained'

