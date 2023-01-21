#!/usr/bin/env bash
DIRNAME1=module_test3

# 입력 이미지, 출력 dir path
python basecode.py /opt/ml/mmtracking/configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_fp16_4e_mot17-private-half_custom.py \
--input /opt/ml/data/aespa_h264_960x720.mp4 \
--output /opt/ml/output_mmtracking/${DIRNAME1}/temp.mp4 \
--fps 24
