import os
import os.path as osp
import pandas as pd
import numpy as np
import json
import pickle
from collections import Counter
import shutil
import cv2
from tqdm import tqdm
import natsort
import argparse


def load_pred(csv_dir:str)->json:
    with open(osp.join(csv_dir,'pred.pickle'),mode='rb') as f:
        pred_json = pickle.load(f)
    return pred_json

def load_meta(result_path:str) -> dict:
    with open(osp.join(result_path, result_path.split('/')[-3] + '.json'),'r') as f:
        meta_info = json.load(f)
    return meta_info

def load_df1(csv_dir:str)->pd.DataFrame:
    with open(osp.join(csv_dir, 'df1_face.pickle'), mode='rb') as f:
        df1 = pickle.load(f)
    return df1

def load_df2(csv_dir:str)->pd.DataFrame:
    with open(osp.join(csv_dir, 'df2_out_of_body_embedding.pickle'), mode='rb') as f:
        df2_sampled = pickle.load(f)
    return df2_sampled

def get_sampled_filename(RESULT_DIR_PATH:str)->list:
    return os.listdir(osp.join(RESULT_DIR_PATH, 'sampled_images'))

def draw_face_bbox(per_id_frames:pd.DataFrame, image_path:str, color:tuple):
    
    # xmin, ymin xmax, ymax filename
    for filename, pred, face_bbox, face_pred in zip(per_id_frames['filename'],
                                                    per_id_frames['pred'], 
                                                    per_id_frames['face_bbox'], 
                                                    per_id_frames['face_pred']):
        # check pred member in face_pred
        try:
            bbox_idx = face_pred.index(pred)
            bbox = face_bbox[bbox_idx,:]
        except Exception as e:
            bbox = None

        # if bbox none(= not face in detected bbox) non draw
        if bbox is None:
            continue

        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        img_full_path = osp.join(image_path,filename)
        img = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
        cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, 5)
        cv2.imwrite(img_full_path, img)

    return per_id_frames


def make_video(captured_imgs_path:str, output_dir_path:str, meta_info:dict):
    # make dir
    JUST_ONE_TIME = True
    output_file_path = osp.join(output_dir_path, 'torch_kpop_face_tracking_video.mp4')
    out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), meta_info['fps'], (meta_info['width'], meta_info['height']))
    
    for img_file_name in tqdm(natsort.natsorted(os.listdir(captured_imgs_path))):
        img_path = osp.join(captured_imgs_path, img_file_name)

        if JUST_ONE_TIME:
            print(img_path)
            JUST_ONE_TIME = False
        
        img = cv2.imread(img_path)
        out.write(img)
    
    out.release()

    # 제대로 열렸는지 확인
    if not out.isOpened():
        print('video open failed!')
    
    return None

def assign_color(member_names:list)->dict:
    color_dict = {}
    color_code = [
        (0,0,255), # red
        (255,51,0), # blue
        (0,204,51), # green
        (51,255,255), # yellow
        (255,255,0), # emerald
        (255,51,153), # purple
        (0,153,255), # orange
        ]
    for idx, member in enumerate(member_names):
        # print('assign func', type(member), member) # nan exists in the member's name.
        if type(member) is not str:
            continue
        color_dict[member] = color_code[idx]
    return color_dict

# func, make full video by final result
def make_facecame_full_resolution(RESULT_DIR_PATH:str) -> None:
    # copy from all capture frame to full_resolution_face_video dir
    captured_imgs_path = osp.join(RESULT_DIR_PATH, 'full_resolution_face_imgs')

    try:
        shutil.copytree(osp.join(RESULT_DIR_PATH, 'captures'), captured_imgs_path)
    except OSError as e:
        return print('Created folder already exists. The folder must be deleted to function normally. 🙅‍♂️')

    meta_info = load_meta(RESULT_DIR_PATH)

    csv_dir = osp.join(RESULT_DIR_PATH, 'csv')
    df1 = load_df1(csv_dir)
    
    df2_out_of_body_embedding = load_df2(csv_dir)
    pred = load_pred(csv_dir)

    # drop useless columns
    df1 = df1.drop(['det_body_xmin', 
                    'det_body_ymin', 
                    'det_body_xmax', 
                    'det_body_ymax',
                    'det_conf',
                    'track_body_xmin',
                    'track_body_ymin',
                    'track_body_xmax',
                    'track_body_ymax',
                    'track_conf',
                    'face_keypoint',
                    'face_confidence',
                    # 'face_pred', # ⭐ need this column
                    'num_overlap_bboxes',
                    'intercept_iou',
                    'isfront',
                    'face_embedding'
                    ], axis=1)
    
    # track_id type change float to int
    df1['track_id'] = df1['track_id'].astype('int')
    
    # assing pred
    for k,v in pred.items():
        df1.loc[df1['track_id'].astype('int') == k, 'pred'] = v
    
    # assing member color
    color_dict = assign_color(df1['pred'].unique())
    
    print(color_dict)

    print('drawing rectangle process...')
    for id in tqdm(df1['track_id'].unique()):
        per_id_frames = df1[df1['track_id'] == id]
        assigned_name = per_id_frames['pred'].unique()[0]
        if type(assigned_name) is not str:
            continue
        color = color_dict[assigned_name]
        print(f'    drawing bbox {assigned_name} face (id: {id})')
        per_id_frames = draw_face_bbox(per_id_frames, captured_imgs_path, color) # draw bboxes

    print('making video process...')
    make_video(captured_imgs_path, RESULT_DIR_PATH, meta_info)

    # delete captured_img_path
    shutil.rmtree(captured_imgs_path)

    print(f'video path: {osp.join(RESULT_DIR_PATH,"torch_kpop_face_tracking_video.mp4")}')

    return print('finish')


def get_args_parser():
    parser = argparse.ArgumentParser('Hello world!', add_help=False)
    parser.add_argument('--result_dir', type=str, default='./result/0lXwMdnpoFQ/60/',help='path of main.py')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('make face track video', parents=[get_args_parser()])
    args = parser.parse_args()

    for arg in vars(args):
        print("--"+arg, getattr(args, arg))

    make_facecame_full_resolution(args.result_dir)

