#!/usr/bin/env bash
# python custom_demo_mot_vis.py /opt/ml/mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py \
# --input /opt/ml/download_video/test_video_1_h264.mp4 \
# --output /opt/ml/output_mmtracking/ocsort_yolox_x_crowdhuman_mot17-private-half/result.mp4 \
# --checkpoint /opt/ml/pretrained_weight__mmtracking/ocsort_yolox_x_crowdhuman_mot17-private-half.pth

DIRNAME1=pred-by-img
DIRNAME2=pred-by-video

# 입력을 이미지
python custom_demo_mot_vis.py /opt/ml/mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py \
--checkpoint /opt/ml/pretrained_weight__mmtracking/ocsort_yolox_x_crowdhuman_mot17-private-half.pth \
--input /opt/ml/data \
--output /opt/ml/output_mmtracking/${DIRNAME1}/result_by_img.mp4 \
--fps 23

# 입력을 비디오
python custom_demo_mot_vis.py /opt/ml/mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half.py \
--checkpoint /opt/ml/pretrained_weight__mmtracking/ocsort_yolox_x_crowdhuman_mot17-private-half.pth \
--input /opt/ml/download_video/aespa_full_1440_1080.mp4 \
--output /opt/ml/output_mmtracking/${DIRNAME2}/result_by_video.mp4

# --backend cv2
# --show 넣으면 오류가 발생함
# --fps 23.976 이것도 넣으면 오류가 발생함...

