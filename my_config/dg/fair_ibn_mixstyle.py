_base_ = ['./_fair_common.py']

# Optional combined classical DG baseline; report it separately from IBN/MixStyle.
model = dict(
    decode_head=dict(
        use_ibn=True,
        use_mixstyle=True,
        mixstyle_p=0.5,
        mixstyle_alpha=0.1))
work_dir = './work_dirs_polyp_v2/dg/fair_ibn_mixstyle'

