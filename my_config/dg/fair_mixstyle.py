_base_ = ['./_fair_common.py']

# Classical MixStyle defaults: p=0.5, alpha=0.1 at shallow encoder stages.
model = dict(
    decode_head=dict(
        use_mixstyle=True,
        mixstyle_p=0.5,
        mixstyle_alpha=0.1))
work_dir = './work_dirs_polyp_v2/dg/fair_mixstyle'

