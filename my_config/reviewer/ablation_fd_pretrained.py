_base_ = ['../dg/_fair_common.py']
model = dict(decode_head=dict(use_fd_kan=True, pretrained_pvt=None))
work_dir = './work_dirs_polyp_v2/reviewer/ablation_fd_pretrained'

