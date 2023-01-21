#!/usr/bin/env bash
DIRNAME1=test-track

# 입력을 이미지
python custom_demo_mot_vis.py /opt/ml/mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half-custom.py \
--checkpoint /opt/ml/pretrained_weight__mmtracking/ocsort_yolox_x_crowdhuman_mot17-private-half.pth \
--input /opt/ml/data/aespa-hard-fps30 \
--output /opt/ml/output_mmtracking/${DIRNAME1}/result.mp4 \
--fps 30

