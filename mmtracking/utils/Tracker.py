# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser
import shutil
import pandas as pd
import mmcv
import cv2
from mmtrack.apis import inference_mot, init_model
import torch
import argparse
import json
import numpy as np

# tag of ananlysis, if ANLYSIS=True save clip image and tracking video

# ignore this is just for testing
WEIGHT_PTH = "./pretrained_weight/ocsort_yolox_x_crowdhuman_mot17-private-half.pth"
CONFIG_PTH = "./mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half-custom.py"

# WIDERFACE_CONFIG_PTH = "./mmtracking/configs/mot/ocsort/ocsort_yolox_x_dancetrack-custom.py"
# WIDERFACE_TRAIN_PTH = "./pretrained_weight/ocsort_yolox_x_widerface.pth"


## 클립을 하기위해서 만든 툴
def clip(num, min_value, max_value):
   return max(min(num, max_value), min_value)


def tracking(meta_info, 
             output, 
             config=CONFIG_PTH,
             score_thr=0.,
             ANALYSIS=False): # mata_info:dict, output:str
    
    # for return raw data 
    raw_data = {
                'frame':[], 
                'filename':[], 
                'det_body_xmin':[], 
                'det_body_ymin':[], 
                'det_body_xmax':[], 
                'det_body_ymax':[],
                'det_conf':[],
                'track_id':[], 
                'track_body_xmin':[],
                'track_body_ymin':[],
                'track_body_xmax':[],
                'track_body_ymax':[],
                'track_conf':[]}
    
    # for return clip data
    clipped_data = {
                    'frame':[], 
                    'filename':[], 
                    'det_body_xmin':[], 
                    'det_body_ymin':[], 
                    'det_body_xmax':[], 
                    'det_body_ymax':[],
                    'det_conf':[],
                    'track_id':[], 
                    'track_body_xmin':[],
                    'track_body_ymin':[],
                    'track_body_xmax':[],
                    'track_body_ymax':[],
                    'track_conf':[]}
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load images
    if osp.isdir(meta_info['image_root']):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(meta_info['image_root'])),
            key=lambda x: int(x.split('.')[0]))
        filenames = [osp.basename(i) for i in imgs]
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(meta_info['image_root'])
        IN_VIDEO = True
    
    out_dir = tempfile.TemporaryDirectory()
    out_path = out_dir.name

    
    # making dirs for pred result
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(osp.join(output, "crop_imgs", "det", "per_frame"), exist_ok=True) # make dir for anlysis
    os.makedirs(osp.join(output, "crop_imgs", "track", "per_frame"), exist_ok=True) # make dir for anlysis
    os.makedirs(osp.join(output, "crop_imgs", "track", "per_id"), exist_ok=True) # make dir for anlysis
    det_img_save_path = osp.join(output, "crop_imgs", "det", "per_frame")
    track_img_save_path_per_frame = osp.join(output, "crop_imgs", "track", "per_frame")
    track_img_save_path_per_id = osp.join(output, "crop_imgs", "track", "per_id")

    fps = int(meta_info["fps"])

    # build the model from a config file and a checkpoint file
    model = init_model(config, WEIGHT_PTH, device=device)

    unmatching_cnt = 0 # unmatching counter
    prog_bar = mmcv.ProgressBar(len(imgs))
    for i, img in enumerate(imgs):
        frame_idx = i+1 # frame_idx
        if isinstance(img, str): # img is loaded by path,
            img_path = osp.join(meta_info['image_root'], img) # filename to path
            # img = cv2.imread(img_path) # read img for get clip size
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert BGR2RGB
        input_img_height = img.shape[0] # for clip
        input_img_width = img.shape[1] # for clip
        result = inference_mot(model, img_path, frame_id=i) # inference one img

        frame_max_row = max(result["det_bboxes"][0].shape[0], result["track_bboxes"][0].shape[0]) # number of bboxes is basis

        # make maximum bboxes
        raw_data["frame"].extend([frame_idx] * frame_max_row)
        clipped_data["frame"].extend([frame_idx] * frame_max_row)
        raw_data["filename"].extend([filenames[i]]*frame_max_row)
        clipped_data["filename"].extend([filenames[i]]*frame_max_row)
        
        # if det_bboxes != track_bboxes count
        if result["det_bboxes"][0].shape[0] != result["track_bboxes"][0].shape[0]:
            unmatching_cnt+=1     

        if output is not None:
            out_file = osp.join(out_path, f'{i:06d}.jpg')
        else:
            out_file = None

        for img_order, detected_info in enumerate(result["det_bboxes"][0]): # one iter is one det_bbox
            raw_xmin_d = detected_info[0]
            raw_ymin_d = detected_info[1]
            raw_xmax_d = detected_info[2]
            raw_ymax_d = detected_info[3]

            xmin_d = clip(raw_xmin_d, 0, input_img_width)
            ymin_d = clip(raw_ymin_d, 0, input_img_height)
            xmax_d = clip(raw_xmax_d, 0, input_img_width)
            ymax_d = clip(raw_ymax_d, 0, input_img_height)
            det_confidence_score = detected_info[4]

            raw_data["det_conf"].append(det_confidence_score)
            clipped_data["det_conf"].append(det_confidence_score)

            bbox_width = xmax_d - xmin_d
            bbox_height = ymax_d - ymin_d
            
            # img crop [H,W,C]
            cropped_det_img = img[int(ymin_d):int(ymin_d+bbox_height), int(xmin_d):int(xmin_d+bbox_width), : ]
            
            # raw coordinate
            raw_data["det_body_xmin"].append(raw_xmin_d)
            raw_data["det_body_ymin"].append(raw_ymin_d)
            raw_data["det_body_xmax"].append(raw_xmax_d)
            raw_data["det_body_ymax"].append(raw_ymax_d)

            # check inference result. if width=0 or height=0, we not append csv
            if 0 not in cropped_det_img.shape:
                clipped_data["det_body_xmin"].append(xmin_d)
                clipped_data["det_body_ymin"].append(ymin_d)
                clipped_data["det_body_xmax"].append(xmax_d)
                clipped_data["det_body_ymax"].append(ymax_d)
                
                # if ANAYSIS is True, save crop_img
                if ANALYSIS is True:
                    cv2.imwrite(osp.join(det_img_save_path, f"{frame_idx}_{img_order}.jpg"), cropped_det_img)
            else:
                # error case
                clipped_data["det_body_xmin"].append(None)
                clipped_data["det_body_ymin"].append(None)
                clipped_data["det_body_xmax"].append(None)
                clipped_data["det_body_ymax"].append(None)

        n_missbox = frame_max_row - result["det_bboxes"][0].shape[0] # num of lack bboxes
        if n_missbox != 0:
            print(f"num of miss track box: {n_missbox} so append empty row")
            for i in range(n_missbox):
                raw_data["det_conf"].append(None)
                clipped_data["det_conf"].append(None)
                raw_data["det_body_xmin"].append(None)
                raw_data["det_body_ymin"].append(None)
                raw_data["det_body_xmax"].append(None)
                raw_data["det_body_ymax"].append(None)
                clipped_data["det_body_xmin"].append(None)
                clipped_data["det_body_ymin"].append(None)
                clipped_data["det_body_xmax"].append(None)
                clipped_data["det_body_ymax"].append(None)
        
        for tracked_info in result["track_bboxes"][0]:
            trk_id = tracked_info[0] # for mkdirs

            raw_data["track_id"].append(trk_id)
            clipped_data["track_id"].append(trk_id)

            raw_xmin_t = tracked_info[1]
            raw_ymin_t = tracked_info[2]
            raw_xmax_t = tracked_info[3]
            raw_ymax_t = tracked_info[4]

            xmin_t = clip(raw_xmin_t, 0, input_img_width)
            ymin_t = clip(raw_ymin_t, 0, input_img_height)
            xmax_t = clip(raw_xmax_t, 0, input_img_width)
            ymax_t = clip(raw_ymax_t, 0, input_img_height)
            track_confidence_score = tracked_info[5]
            raw_data["track_conf"].append(track_confidence_score)
            clipped_data["track_conf"].append(track_confidence_score)

            bbox_width = xmax_t-xmin_t
            bbox_height = ymax_t-ymin_t

            # crop
            cropped_track_img = img[int(ymin_t):int(ymin_t+bbox_height), int(xmin_t):int(xmin_t+bbox_width), :]
            
            # img save path
            per_frame_save_path = osp.join(track_img_save_path_per_frame, f"{frame_idx}_{int(trk_id)}.jpg")

            # not clip data
            raw_data["track_body_xmin"].append(raw_xmin_t)
            raw_data["track_body_ymin"].append(raw_ymin_t)
            raw_data["track_body_xmax"].append(raw_xmax_t)
            raw_data["track_body_ymax"].append(raw_ymax_t)

            # make dir per each ids
            per_id_save_path = osp.join(track_img_save_path_per_id, f'{int(trk_id)}', f'{int(trk_id)}'+"_"+f'{frame_idx}'+".jpg")

            # generate id dirs
            os.makedirs(osp.dirname(per_id_save_path), exist_ok=True)

            # if width, height is acceptable
            if 0 not in cropped_track_img.shape:
                if ANALYSIS is True: # save
                    cv2.imwrite(per_frame_save_path, cropped_track_img)
                    cv2.imwrite(per_id_save_path, cropped_track_img)
                clipped_data["track_body_xmin"].append(xmin_t)
                clipped_data["track_body_ymin"].append(ymin_t)
                clipped_data["track_body_xmax"].append(xmax_t)
                clipped_data["track_body_ymax"].append(ymax_t)
            else:
                clipped_data["track_body_xmin"].append(None)
                clipped_data["track_body_ymin"].append(None)
                clipped_data["track_body_xmax"].append(None)
                clipped_data["track_body_ymax"].append(None)

        n_missbox = frame_max_row - result["track_bboxes"][0].shape[0] # num of unmatching box each iter

        if n_missbox != 0:
            for i in range(n_missbox):
                raw_data["track_id"].append(None)
                clipped_data["track_id"].append(None)
                raw_data["track_conf"].append(None)
                clipped_data["track_conf"].append(None)
                raw_data["track_body_xmin"].append(None)
                raw_data["track_body_ymin"].append(None)
                raw_data["track_body_xmax"].append(None)
                raw_data["track_body_ymax"].append(None)
                clipped_data["track_body_xmin"].append(None)
                clipped_data["track_body_ymin"].append(None)
                clipped_data["track_body_xmax"].append(None)
                clipped_data["track_body_ymax"].append(None)
        
        model.show_result(
            img,
            result,
            score_thr=score_thr,
            show=False,  
            thickness=5, 
            font_scale=1.0, 
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend="cv2") # default plt or cv2
        prog_bar.update()
    
    if output != None:
        img_dir_path = osp.join(output,"tracked_imgs")
        if ANALYSIS: # if ANALYSIS is True, make tracking video
            print()
            print(f'making the output video 📺 at {output} with a FPS of {fps}')
            print(f"out_path:{out_path}")
            print(f"osp.join(args.output,'tracking_video.mp4'):{osp.join(output,'tracking_video.mp4')}")
            mmcv.frames2video(out_path, osp.join(output,"tracking_video.mp4"), fps=fps, fourcc='mp4v')
        
            if osp.isdir(img_dir_path):
                print(f"delete exist dirs for overwritting")
                shutil.rmtree(img_dir_path)

            os.makedirs(img_dir_path, exist_ok=True) # make save dir
        
            for file_name in os.listdir(out_path):
                new_file_name = str(int(file_name.split(".")[0])+1).zfill(10) + ".jpg"
                shutil.copy(osp.join(out_path, file_name), osp.join(img_dir_path,new_file_name))

        out_dir.cleanup() # delete default mmtracking dir

    raw_df1 = pd.DataFrame(raw_data) # raw_df
    raw_df1.to_csv(osp.join(output,'csv/df1_raw.csv'))

    clipped_df1 = pd.DataFrame(clipped_data) # clipped_df
    clipped_df1.to_csv(osp.join(output,'csv/df1_clipped_no_postprecessing.csv')) # modify file name
    
    # if no analysis delete crop_img dirs
    if not ANALYSIS:
        imgs_paths = osp.join(output, "crop_imgs")
        shutil.rmtree(imgs_paths)

    return clipped_df1, raw_df1

def get_args_parser():
    parser = argparse.ArgumentParser('Hello world!', add_help=False)
    parser.add_argument('--meta_path', type=str, default='/opt/ml/final-project-level3-cv-04/data/20230122_1745.json',help='input video file or folder')
    parser.add_argument('--output_path', type=str, default='./test', help='output video file (mp4 format) or folder')
    parser.add_argument('--config', type=str, default='/opt/ml/mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half-custom.py')
    parser.add_argument('--score_thr', type=float, default=0.0, help='The threshold of score to filter bboxes.')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('download from Youtube', parents=[get_args_parser()])
    args = parser.parse_args()

    for arg in vars(args):
        print("--"+arg, getattr(args, arg))
    
    with open(args.meta_path) as f:
        meta_info = json.load(f)

    tracking(meta_info, args.output_path, config=args.config, score_thr = args.score_thr)
