import os
import os.path as osp
from glob import glob
import cv2
import csv
import pandas as pd
import natsort
import numpy as np
from itertools import chain
import json
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter   
from PIL import Image
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import os
import argparse




#### clansing
def clansing(df:pd.DataFrame, pred:dict) -> pd.DataFrame:
    # drop useless column
    df.drop(['det_body_xmin', 
             'det_body_ymin', 
             'det_body_xmax',
             'det_body_ymax',
             'det_conf',
             'track_body_xmin',
             'track_body_ymin',
             'track_body_xmax',
             'track_body_ymax',
             'track_conf',
             'num_overlap_bboxes',
             'intercept_iou',
             'isfront',
             'face_bbox',
             'face_embedding',
             'face_confidence'], axis=1, inplace=True)
    # track_id type change
    df['track_id'] = df['track_id'].astype('int32')

    # assign name
    for k,v in pred.items():
        df.loc[df['track_id']==k,'name'] = v

    return df


#### get face order
def get_order(element, lst):
    try:
        return lst.index(element)
    except ValueError:
        return -1


#### for key point assign
def get_keypoint(element, idx):
    if idx != -1:
        left_eye_point = element[idx][0]
        right_eye_point = element[idx][1]
        center_point = [int((x + y)/2) for x, y in zip(left_eye_point, right_eye_point)]
        return center_point
    else: # if un detected keypoints return [-1,-1]
        return [-1, -1]


#### add bbox column
def keypoint_center_bounding_box(coord, crop_width, crop_height):
    x, y = coord
    if x==-1 or y==-1: # if undetected keypoints [-1,-1]
        return [-1,-1,-1,-1]
    x_min = x - crop_width / 2
    y_min = y - crop_height / 2
    x_max = x + crop_width / 2
    y_max = y + crop_height / 2
    coordinate = [int(i) for i in [x_min, y_min, x_max, y_max]] # int coordinate
    return coordinate


#### add shift bbox column
def shift_bounding_box(bbox, shift_ratio, meta_info):
    x_min, y_min, x_max, y_max = bbox
    if x_min == -1 or y_min == -1 or x_max == -1 or y_max == -1:
        return [0,0,meta_info['width'],meta_info['height']]
    height = y_max - y_min
    y_min += height * shift_ratio
    y_max += height * shift_ratio
    coordinate = [int(i) for i in [x_min, y_min, x_max, y_max]]
    return coordinate


#### add missing rows
def add_missing_files(df, all_files, name, meta_info):
    missing_files = set(all_files) - set(df['filename'])
    if missing_files:
        missing_df = pd.DataFrame({
                                    'frame':[int(filename.split('.')[0]) for filename in list(missing_files)],
                                    'filename': list(missing_files), 
                                    'name': [name for i in list(missing_files)],
                                    'shift_bbox': [[0,0,meta_info['width'],meta_info['height']] for i in list(missing_files)],
                                    })
        df = df.append(missing_df, ignore_index=True)
    
    df = df.sort_values(by='frame' ,ascending=True)
    df.reset_index(inplace=True, drop=True)
    return df


def replace_values(row, target_col, width, height):
    xmin, ymin, xmax, ymax = row[target_col]
    if xmin == 0 and ymin == 0 and xmax >= width:
        row[target_col] = [0, 0, width, ymax]
    if xmin == 0 and ymin == 0 and ymax >= height:
        row[target_col] = [0, 0, xmax, height]
    return row


def update_overange_bounding_box(df, target_col,width, height):
    df = df.apply(lambda row: replace_values(row, target_col, width, height), axis=1)
    return df


#### Define a function to collect the values of columns A, B, C, and D as a list
def collect_values(row):
    return [row['xmin'], row['ymin'], row['xmax'], row['ymax']]


#### tagging untrack frame
def tagging_untrack_frame(row, meta_info):
    if row == [0, 0, meta_info['width'], meta_info['height']]:
        return False
    else:
        return True


# for plotting time series
def plot_time_series(df, target_col, prefix, filename, col=2):
    colname = "xmax" if col==2 else "ymax"
    # Extract xmax from each row of A and store it as a new column xmax
    temp = pd.DataFrame()
    temp['max'] = df[target_col].apply(lambda x: x[col])
    # Plot xmax as a time series
    plt.plot(temp['max'])
    plt.xlabel(colname)
    plt.ylabel('coorinate')
    plt.title('Time Series Plot')
    # Save the plot
    filename = f'{colname}_plot_{prefix}_{filename}.png'
    filepath = os.path.join(os.getcwd(), filename)
    plt.savefig(filepath)
    return filepath


# savitzky_golay smoothing
def savitzky_golay(df, target_column, output_column):
    # split coordinates
    df['xmin'] = [coordinate[0] for coordinate in df[target_column]]
    df['ymin'] = [coordinate[1] for coordinate in df[target_column]]
    df['xmax'] = [coordinate[2] for coordinate in df[target_column]]
    df['ymax'] = [coordinate[3] for coordinate in df[target_column]]
    
    df['xmin'] = df[['xmin']].apply(savgol_filter,  window_length=45, polyorder=2)
    df['ymin'] = df[['ymin']].apply(savgol_filter,  window_length=45, polyorder=2)
    df['xmax'] = df[['xmax']].apply(savgol_filter,  window_length=45, polyorder=2)
    df['ymax'] = df[['ymax']].apply(savgol_filter,  window_length=45, polyorder=2)

    df['xmin'] = df['xmin'].astype('int')
    df['ymin'] = df['ymin'].astype('int')
    df['xmax'] = df['xmax'].astype('int')
    df['ymax'] = df['ymax'].astype('int')

    df[output_column] = df.apply(collect_values, axis=1)

    return df


def moving_average(df, window_size, target_column,  output_column):
    # split coordinates
    df['xmin'] = [coordinate[0] for coordinate in df[target_column]]
    df['ymin'] = [coordinate[1] for coordinate in df[target_column]]
    df['xmax'] = [coordinate[2] for coordinate in df[target_column]]
    df['ymax'] = [coordinate[3] for coordinate in df[target_column]]

    def smoothing(row, window):
        window = row.rolling(window)
        mean = window.mean()
        result = np.where(np.isnan(mean), row, mean)
        return result
    
    df['xmin'] = df[['xmin']].apply(smoothing, args=(window_size,))
    df['ymin'] = df[['ymin']].apply(smoothing, args=(window_size,))
    df['xmax'] = df[['xmax']].apply(smoothing, args=(window_size,))
    df['ymax'] = df[['ymax']].apply(smoothing, args=(window_size,))

    df['xmin'] = df['xmin'].astype('int')
    df['ymin'] = df['ymin'].astype('int')
    df['xmax'] = df['xmax'].astype('int')
    df['ymax'] = df['ymax'].astype('int')

    df[output_column] = df.apply(collect_values, axis=1)

    df = df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    return df


def moving_median(df, window_size, target_column,  output_column):
    # split coordinates
    df['xmin'] = [coordinate[0] for coordinate in df[target_column]]
    df['ymin'] = [coordinate[1] for coordinate in df[target_column]]
    df['xmax'] = [coordinate[2] for coordinate in df[target_column]]
    df['ymax'] = [coordinate[3] for coordinate in df[target_column]]

    def smoothing(row, window):
        window = row.rolling(window)
        mean = window.median()
        result = np.where(np.isnan(mean), row, mean)
        return result
    
    df['xmin'] = df[['xmin']].apply(smoothing, args=(window_size,))
    df['ymin'] = df[['ymin']].apply(smoothing, args=(window_size,))
    df['xmax'] = df[['xmax']].apply(smoothing, args=(window_size,))
    df['ymax'] = df[['ymax']].apply(smoothing, args=(window_size,))

    df['xmin'] = df['xmin'].astype('int')
    df['ymin'] = df['ymin'].astype('int')
    df['xmax'] = df['xmax'].astype('int')
    df['ymax'] = df['ymax'].astype('int')

    df[output_column] = df.apply(collect_values, axis=1)

    df = df.drop(['xmin', 'ymin', 'xmax', 'ymax'], axis=1)
    return df


def trim(df):
    # drop useless column
    df.drop(['face_keypoint', 
             'face_pred', 
             'key_point_order',
             'center_point',
            ], axis=1, inplace=True)
    return df


# top, bottom, left, right
def img_padding(img, xmin, ymin, xmax, ymax, w, h):
    if xmax>w:
        img = cv2.copyMakeBorder(img, 0, 0, 0, xmax-w, cv2.BORDER_CONSTANT)
    if xmin<0: 
        img = cv2.copyMakeBorder(img, 0, 0,-xmin, 0, cv2.BORDER_CONSTANT)
        xmax = xmax - xmin 
        xmin = 0
    if ymax>h:
        img = cv2.copyMakeBorder(img, 0, ymax-h, 0, 0, cv2.BORDER_CONSTANT)
    if ymin<0:
        img = cv2.copyMakeBorder(img, -ymin, 0, 0, 0, cv2.BORDER_CONSTANT)
        py = ymax - ymin
        ymin = 0
    return img, xmin, ymin, xmax, ymax


def short_untrack_bbox_update(df, target_col_name, new_col_name,keep_threshold, meta_info):
    df[new_col_name] = [None] * len(df)
    df.at[0, new_col_name] = df.at[0, target_col_name]
    cnt=0
    last_state=None
    df['is_track_update'] = [False]*len(df)
    for i in range(1, len(df)):
        # if untrack case
        if df.at[i, target_col_name] == [0,0,meta_info['width'],meta_info['height']] and df.at[i, 'is_track'] == False:
            # increase cnt
            cnt+=1
            if i > 0 and i < len(df) - 1:
                # move avg
                '''
                # calculate the average of the previous and next values for each component of shift_bbox
                xmin = int((df.at[i-1, 'shift_bbox'][0] + df.at[i+1, 'shift_bbox'][0]) / 2)
                ymin = int((df.at[i-1, 'shift_bbox'][1] + df.at[i+1, 'shift_bbox'][1]) / 2)
                xmax = int((df.at[i-1, 'shift_bbox'][2] + df.at[i+1, 'shift_bbox'][2]) / 2)
                ymax = int((df.at[i-1, 'shift_bbox'][3] + df.at[i+1, 'shift_bbox'][3]) / 2)
                # update the values of shift_bbox in the current row
                df.at[i, 'increase_decrease_bbox'] = [xmin, ymin, xmax, ymax]
                # update state
                '''
                # keep value
                df.at[i, new_col_name] = [0,0,meta_info['width'],meta_info['height']]
            elif i == 0:
                # if the current row is the first & undetect row, take the value of the next row
                df.at[i, new_col_name] = df.at[i+1, target_col_name]
            else:
                # if the current row is the last & undetect row, take the value of the previous row
                df.at[i, new_col_name] = df.at[i-1, target_col_name]
        # if tracked appeared
        else:
            # short_lange untrack case, fill last_state
            if 0<cnt<keep_threshold:
                for j in range(cnt):
                    target_idx = i-(j+1)
                    df.at[target_idx, new_col_name] = last_state
                    df.at[target_idx, 'is_track_update'] = True # if short time state update
                
            # long_lange untrack case, don't care
            elif cnt>=keep_threshold:
                pass
            
            cnt=0
            # untracked case just keep [0, 0, width, height] value
            df.at[i, new_col_name] = df.at[i, target_col_name]
            last_state=df.at[i, target_col_name]
        
    df['need_padding'] = df['is_track'] | df['is_track_update']
    return df


def crop_resize_save_img(df, img_dir_path, save_dir, name, meta_info, target_col, window_height, window_width, mode, fullscreen):
    mini_df = df[['filename', target_col, 'need_padding']]
    cnt=0
    width = meta_info['width']
    height = meta_info['height']
    padded_box = []
    # make dir
    dir_path = osp.join(save_dir, name)
    os.makedirs(dir_path, exist_ok=True)
    print(f"start cropping {len(df)} images.")
    for order, (index, row) in tqdm(enumerate(mini_df.iterrows())):
        # load
        img = cv2.imread(osp.join(img_dir_path, row['filename']))
        
        # get bbox coordinate
        xmin, ymin, xmax, ymax = row[target_col][0], row[target_col][1], row[target_col][2], row[target_col][3]

        if row['need_padding']: # tracked
            # get center-point
            center_y, center_x = (ymin+ymax)//2, (xmin+xmax)//2
            # get new xmin, ymin, xmax, ymax
            xmin, ymin, xmax, ymax = center_x-window_width//2, center_y-window_height//2, center_x+window_width//2, center_y+window_height//2
            # if over range, padding around
            if xmax>width or xmin<0 or ymax>height or ymin<0:
                img, xmin, ymin, xmax, ymax = img_padding(img, xmin, ymin, xmax, ymax, width, height)
            # crop
            img = img[ymin:ymax, xmin:xmax]
        
        else: # untracked
            if fullscreen==True: # mode == 'normal' 
                img = cv2.resize(img, dsize=(meta_info['width'], meta_info['height']), interpolation=cv2.INTER_CUBIC)
                xmin, ymin, xmax, ymax = 0, 0, window_width, window_height
            else: # fullscreen==False:
                center_y, center_x = (ymin+ymax)//2, (xmin+xmax)//2
                # get new xmin, ymin, xmax, ymax
                xmin, ymin, xmax, ymax = center_x-window_width//2, center_y-window_height//2, center_x+window_width//2, center_y+window_height//2
                img = img[ymin:ymax, xmin:xmax]
        
        # append
        padded_box.append([xmin, ymin, xmax, ymax])
        # if unmatching window img print size
        if img.shape[0] != window_height and img.shape[1] != window_width:
            # print(img.shape[0], window_height, img.shape[1], window_width)
            pass
        # write img
        cv2.imwrite(osp.join(dir_path, row['filename']), img)
    df['cropped_box'] = padded_box
    return df, dir_path


def enlarge_with_padding(img, crop_width, crop_height, window_width, window_height, padding_tag, path):
    if padding_tag == True:
        padded_img = np.zeros((window_height, window_width, 3), dtype=np.uint8)
        x_offset = int((window_width - crop_width) / 2)
        y_offset = int((window_height - crop_height) / 2)
        try:
           padded_img[y_offset:y_offset+crop_height, x_offset:x_offset+crop_width, :] = img
        except:
            print(f"error occur in enlarge_with_padding, file: {path}")
            print("input img size:", img.shape)
            print(f"window size: ({window_height}, {window_width})" )
            print("so just resize")
            padded_img = cv2.resize(img, (window_width, window_height), interpolation=cv2.INTER_CUBIC)
    else:
        padded_img = cv2.resize(img, (window_width, window_height), interpolation=cv2.INTER_CUBIC)
    return padded_img


def make_video(dir_path, df, meta_info, save_dir, window_height, window_width, name):
    files = os.listdir(dir_path)
    padding_tags = df['need_padding'].to_list()
    img_list = natsort.natsorted(files)
    img_paths = [osp.join(dir_path, i) for i in img_list]
    video_path = osp.join(save_dir, f'{name}_output.mp4')
    out = cv2.VideoWriter(video_path,
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          meta_info['fps'], 
                          (window_width, window_height))
    img = cv2.imread(img_paths[0])
    crop_h,crop_w,c = img.shape

    tag = 'normal'
    if window_height == crop_h and window_width == crop_w:
        tag = 'normal'
    elif window_height == crop_h and window_width != crop_w:
        tag = 'vertical'
    elif window_height != crop_h and window_width == crop_w:
        tag = 'horizontal'
    else:
        raise ValueError("view type error")

    print(f"{name}'s video recoding...")
    for path, padding_tag in tqdm(zip(img_paths, padding_tags)):
        img = cv2.imread(path)
        # enlarge_with_padding(img, crop_width, crop_height, window_width, window_height)
        if tag == 'vertical' or tag=='horizontal':
            # def enlarge_with_padding(img, crop_width, crop_height, window_width, window_height, padding_tag):
            img = enlarge_with_padding(img, crop_w, crop_h, window_width, window_height, padding_tag, path)
        # if normal don't need post processing
        out.write(img)
    out.release()
    h,w,c = img.shape
    return video_path





def video_gen(df:pd.DataFrame, meta_info:dict, member:str, pred:dict,  save_dir:str,
              window_ratio:float, aespect_ratio:float, shift_bb:float, fullscreen=True):
    view_type = None
    # calc window size
    window_height = int(meta_info['height']*window_ratio)
    window_width = int(meta_info['width']*window_ratio)
    
    print(f"start making {member}'s facecam")
    # calc crop_size and assign mode
    if aespect_ratio >= 1:
        print('horizontal video mode')
        mode = 'horizontal'
        crop_width = window_width # width is Criteria
        crop_height = int(crop_width * 1/aespect_ratio)
        if fullscreen==False:
            ValueError(f"{mode} must need full screen")
    elif aespect_ratio==0:
        print('normal video mode')
        mode = 'normal'
        crop_height = window_height # hold the original video asepect
        crop_width = window_width
    else:
        print('vertical video mode')
        mode = 'vertical'
        crop_height = window_height # height is Criteria
        crop_width = int(crop_height*aespect_ratio)
        if fullscreen==False:
            ValueError(f"{mode} must need full screen")

    print(f'window size: hegiht: {window_height}, width: {window_width}')
    print(f'video mode: hegiht: {mode}')
    print(f'show full screen: {fullscreen}')
    # 1. clansing df
    df = clansing(df, pred)
    
    # 2. assign key_point order
    df['key_point_order'] = df.apply(lambda x: get_order(x['name'], x['face_pred']), axis=1)

    # 3. get center point(int type)
    df['center_point'] = df.apply(lambda x: get_keypoint(x['face_keypoint'], x['key_point_order']), axis=1)

    # 4. get center_bbox(bbox center is key_point)
    df['center_bbox'] = df['center_point'].apply(lambda x: keypoint_center_bounding_box(x, crop_height, crop_width))

    # 5. shifting bbox, if undetected keypoints or unmatching preds bbox is [0, 0, meta_info['width'], meta_info['height']]
    df['shift_bbox'] = df['center_bbox'].apply(lambda x: shift_bounding_box(x, shift_bb, meta_info))

    # 6. extract selected member
    df = df[df['name'] == member]
    # print(f'after member select {len(df)} rows left.') # for debugging

    # 7. get all captures img filenames
    img_dir_path = meta_info['image_root'].replace('.','..') # change for current path
    img_files = os.listdir(img_dir_path)

    # 8. add missing rows
    df = add_missing_files(df, img_files, member, meta_info)
    # img_path = plot_time_series(df, 'shift_bbox', "0", 'add_missing_files', col=2) # for debugging
    # print(f'img_path: {img_path}')

    # 9. tagging untracked frame
    df['is_track'] = df['shift_bbox'].apply(lambda x: tagging_untrack_frame(x, meta_info))
    
    # 10. trim, drop useless column
    df = trim(df)

    # 11. moving avg untrack row, ignore threshold
    df = short_untrack_bbox_update(df, 'shift_bbox', 'ignore_short_untrack',meta_info['fps']*5, meta_info) # df add column increase_decrease_bbox
    # img_path = plot_time_series(df, 'ignore_short_untrack', 1, 'ignore_short_untrack', col=2) # for debugging
    # print(f'img_path: {img_path}') # for debugging
    
    # 12. smoothing ⭐
    df = moving_median(df, meta_info['fps']*1, 'ignore_short_untrack', 'median_bbox')
    df = moving_average(df, int(meta_info['fps']*.5), 'median_bbox', 'smoothed_bbox')
    # df = savitzky_golay(df, 'median_bbox', 'smoothed_bbox') # df add column smoothed_bbox
    # img_path = plot_time_series(df, 'smoothed_bbox', 2, 'moving_median_avg', col=2) # for debugging
    # print(f'img_path: {img_path}') # for debugging
    
    # 13. drop_duplicates row
    df = df.drop_duplicates(subset='filename', keep='first')
    # img_path = plot_time_series(df, 'smoothed_bbox', 3, 'drop_duplicates', col=2) # for debugging
    # print(f'img_path: {img_path}') # for debugging
    
    
    # 14. clip img for over bbox, if long untracked frame made by resume
    df, crop_img_path = crop_resize_save_img(df, 
                                             img_dir_path, 
                                             save_dir, 
                                             member, 
                                             meta_info, 
                                             'smoothed_bbox', 
                                             crop_height, 
                                             crop_width, 
                                             mode,
                                             fullscreen)

    
    # 15. making video
    video_path = make_video(crop_img_path, df, meta_info, save_dir, window_height, window_width, member)
    
    # 16. delete crop imgs
    shutil.rmtree(crop_img_path)
    
    print('video generation finish')
    return video_path



def get_args_parser():
    parser = argparse.ArgumentParser('SmoothingVideoGenerator', add_help=False)
    parser.add_argument('--df_root', default='/opt/ml/final-project-level3-cv-04/result/0lXwMdnpoFQ/100/csv/df1_face.pickle', type=str)
    parser.add_argument('--meta_info_root', default='/opt/ml/final-project-level3-cv-04/result/0lXwMdnpoFQ/100/0lXwMdnpoFQ.json', type=str)
    parser.add_argument('--member_name', default='aespa_winter', type=str)
    parser.add_argument('--pred_root', default='/opt/ml/final-project-level3-cv-04/result/0lXwMdnpoFQ/100/csv/pred.pickle', type=str)
    parser.add_argument('--save_dir', default='./', type=str)
    parser.add_argument('--window_ratio', default=0.5, type=float)
    parser.add_argument('--aespect_ratio', default=0., type=float)
    parser.add_argument('--shift_bb', default=0.2, type=float)

    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('from to mp4', parents=[get_args_parser()])
    args = parser.parse_args()
    
    for arg in vars(args):
        print("--"+arg, getattr(args, arg))

    with open(args.df_root, mode='rb') as f:
        df1 = pickle.load(f)
    
    with open(args.meta_info_root, mode='r') as f:
        meta_info = json.load(f)

    with open(args.pred_root, mode='rb') as f:
        pred = pickle.load(f)

    path = video_gen(
                     df1, 
                     meta_info, 
                     args.member_name, 
                     pred, 
                     args.save_dir, 
                     args.window_ratio, 
                     args.aespect_ratio, 
                     args.shift_bb,
                     True
                     )

    print(f'video path is {path}')