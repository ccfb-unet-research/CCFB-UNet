_base_ = ['./_fair_common.py']

# Same architecture as fair_pvt_control_pretrained, but PVT starts from scratch.
model = dict(
    decode_head=dict(
        use_pvt_control=True,
        pretrained_pvt=''))
work_dir = './work_dirs_polyp_v2/dg/fair_pvt_control_scratch'

