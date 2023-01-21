_base_ = ['./tracktor_faster-rcnn_r50_fpn_4e_mot17-private-half.py']

model = dict(
    detector=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/opt/ml/pretrained_weight__mmtracking/faster-rcnn_r50_fpn_fp16_4e.pth'  # noqa: E501
        )),
    reid=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            '/opt/ml/pretrained_weight__mmtracking/reid_r50_fp16_8x32_6e.pth'  # noqa: E501
        )))
fp16 = dict(loss_scale=512.)
