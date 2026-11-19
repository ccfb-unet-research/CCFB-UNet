_base_ = ['../dg/_fair_common.py']
model = dict(decode_head=dict(use_gls_cams=True, pretrained_pvt=None))
work_dir = './work_dirs_polyp_v2/reviewer/ablation_gls_pretrained'

