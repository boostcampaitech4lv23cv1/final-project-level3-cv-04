#!/usr/bin/env bash
DIRNAME2=pred-by-video

# 입력을 비디오
python custom_demo_mot_vis.py /opt/ml/mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py \
--checkpoint /opt/ml/pretrained_weight__mmtracking/ocsort_yolox_x_crowdhuman_mot17-private-half.pth \
--input /opt/ml/download_video/aespa_full_1440_1080.mp4 \
--output /opt/ml/output_mmtracking/${DIRNAME2}/result_by_video.mp4
