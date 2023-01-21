#!/usr/bin/env bash
TRACKING_NAME=20230121_1428
OUTDIR_NAME=video_${TRACKING_NAME}

echo ${TRACKING_NAME}
echo ${OUTDIR_NAME}

python3 Make_individual_video.py --target_dir /opt/ml/output_mmtracking/${TRACKING_NAME}/crop_imgs/track/per_id \
--output_dir /opt/ml/output_mmtracking/${TRACKING_NAME}/${OUTDIR_NAME}