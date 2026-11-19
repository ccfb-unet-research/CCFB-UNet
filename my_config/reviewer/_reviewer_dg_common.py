_base_ = ['../dg/_fair_common.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.reviewer_dg'],
    allow_failed_imports=False)

# All five framework-native comparisons use the same pretrained PVT control.
# This is a matched-backbone adaptation track, distinct from official VFM runs.
model = dict(decode_head=dict(
    type='ReviewerDGUnet',
    use_pvt_control=True,
    pretrained_pvt=None,
    dg_method='robustnet',
    dg_loss_weight=0.1))

