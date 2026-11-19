_base_ = ['./_fair_common.py']

# IBN-Net-style shallow encoder normalization baseline.
model = dict(decode_head=dict(use_ibn=True))
work_dir = './work_dirs_polyp_v2/dg/fair_ibn'

