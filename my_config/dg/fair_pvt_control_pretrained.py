_base_ = ['./_fair_common.py']

# Pretraining control: plain U-Net plus only a simple PVT bottleneck fusion.
# pretrained_pvt=None deliberately reuses the exact default path of unet_all.
model = dict(
    decode_head=dict(
        use_pvt_control=True,
        pretrained_pvt=None))
work_dir = './work_dirs_polyp_v2/dg/fair_pvt_control_pretrained'

