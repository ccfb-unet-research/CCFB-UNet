_base_ = ['./_fair_common.py']

# Same full CCFB architecture, with PVT randomly initialized.
model = dict(
    decode_head=dict(
        use_fd_kan=True,
        use_mgd_encoder=True,
        use_gls_cams=True,
        pretrained_pvt=''))
work_dir = './work_dirs_polyp_v2/dg/fair_ccfb_scratch'

