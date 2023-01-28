import pandas as pd
import numpy as np
import mmtracking.utils.libs.iou as iou
import json
from tqdm import tqdm
from itertools import combinations

# default just keeping ids which over 5sec 
def delete_short_id(df:pd.DataFrame, fps:int, sec:int=5)-> pd.DataFrame:
    df = df.copy()
    for trk_id in df['track_id'].unique():
        # nan id pass
        if np.isnan(trk_id): 
            continue

        # if num of frames over fps*second keep track_id
        if len(df[df['track_id'] == trk_id]) > fps*sec:
            pass
        else:
            df.loc[df['track_id'] == trk_id,'track_id'] = np.NaN
    return df

def clip(df:pd.DataFrame, meta_info:dict)-> pd.DataFrame:
    df = df.copy()
    df['track_body_xmin'] = df['track_body_xmin'].clip(0, meta_info['width'])
    df['track_body_xmax'] = df['track_body_xmax'].clip(0, meta_info['width'])
    df['track_body_ymin'] = df['track_body_ymin'].clip(0, meta_info['height'])
    df['track_body_ymax'] = df['track_body_ymax'].clip(0, meta_info['height'])
    return df



## add column is_front, overlap people, sum_iou
def count_overlap(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()

    overlap_array = []
    n_peoples_array = []
    isfront_array = []

    frame_list = [] # for each frame df

    for frame_num in sorted(df['frame'].unique()): #iter each frame
        overlap_cnt = 0 # counting per id
        one_frame_df = df[ df['frame'] == frame_num] # get one frame
        frame_list.append(one_frame_df) # frame_list = [frame1_list, frame2_list, frame3_list, ...]
    
    # calculate each frame
    for frame in frame_list:
        frame = frame[["track_body_xmin", "track_body_ymin", "track_body_xmax", "track_body_ymax"]].to_numpy()
        
        # per bbox
        for i in range(frame.shape[0]):
            is_front = True
            iou_of_frame = []
            one = np.expand_dims(frame[i], axis=0)
            others = np.delete(frame, i , axis = 0)

            for other in others:
                other = np.expand_dims(other, axis=0)
                iou_ = iou.iou(one, other)
                iou_of_frame.append(iou_)

            iou_of_frame = np.array(iou_of_frame)
            sum_of_iou = np.nansum(iou_of_frame, axis=0) # sum of intercept iou, ⭐nansum
            
            n_people_idx = np.where(iou_of_frame>0, True, False) 
            n_people = np.nansum(n_people_idx, axis=0) # number of overlap people ⭐nansum

            if n_people > 0: # if overlap people exist     
                n_people_idx = n_people_idx.flatten() # flatten
                others_y_max = np.max(others[n_people_idx], axis=0)[3] # overlaped others's y_max
                one_y_max = one.flatten()[3] # one's ymax
                if others_y_max >= one_y_max: # if other's ymax is bigger than one, is is_front False
                    is_front = False

            # append list
            overlap_array.append(sum_of_iou.item())
            n_peoples_array.append(n_people.item())
            isfront_array.append(is_front)

    # append df
    df["num_overlap_bboxes"] = n_peoples_array
    df["intercept_iou"] = overlap_array
    df["isfront"] = isfront_array
    return df


def compute_IOU(bbox1: list, 
                bbox2: list):
    bbox1_area = (bbox1[2]-bbox1[0]) * (bbox1[3]-bbox1[1])
    bbox2_area = (bbox2[2]-bbox2[0]) * (bbox2[3]-bbox2[1])
    
    # obtain x1, y1, x2, y2 of the intersection
    xlen = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
    ylen = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    
    overlapped = xlen * ylen
    
    return overlapped / (bbox1_area + bbox2_area - overlapped)


def assign_new_id(starting_frame: int,
                  df: pd.DataFrame,
                  track_id: int,
                  new_track_id: int,
                  verbose = False):       # starting_frame (포함)이후의 df에서, track_id 들을 new_track_id 로 바꾼 df를 반환함
    if verbose:
        print('originally, id {} count: {}'.format(track_id, df[df['track_id'] == track_id]['track_id'].count()))
    df.loc[(df['frame'] >= starting_frame) & (df['track_id'] == track_id), 'track_id'] = new_track_id
    if verbose:
        print('newly assinged id {} by count: {}'.format(new_track_id, df[df['track_id'] == new_track_id]['track_id'].count()))
    
    return df


def divide_ids_by_IOU(df, IOU_THRESHOLD = 0.6):
    maxframenum = df['frame'].values[-1]
    next_track_id = max(df['track_id'].values) + 1
    to_divide_ids = set()

    for i in tqdm(range(1, maxframenum+1)):
        now_df = df[df['frame']==i].reset_index(drop=True)
        now_overlapped_ids = set()
        for a, b in combinations(range(len(now_df)), 2):    #현재 프레임의 row 에서 두 개를 뽑은 조합을 모두 IoU 계산
            bbox1 = now_df.loc[a][['track_body_xmin', 'track_body_ymin', 'track_body_xmax', 'track_body_ymax']].values
            bbox2 = now_df.loc[b][['track_body_xmin', 'track_body_ymin', 'track_body_xmax', 'track_body_ymax']].values
            if compute_IOU(bbox1, bbox2) > IOU_THRESHOLD:
                overlapped_1 = now_df.loc[a]['track_id']
                overlapped_2 = now_df.loc[b]['track_id']
                now_overlapped_ids.update([overlapped_1, overlapped_2])
        
        # 원래는 겹치고 있다가 이번 프레임부터 겹치지 않은 id 들에 대해, 이후 프레임들에서 새 id를 부여해서 분리
        to_remove = set()
        for d_id in to_divide_ids:
            if d_id not in now_overlapped_ids:
                df = assign_new_id(i, df, d_id, next_track_id)
                next_track_id += 1
                to_remove.add(d_id)
        for del_id in to_remove:
            to_divide_ids.remove(del_id)
        
        for d_id in now_overlapped_ids:
            if d_id not in to_divide_ids:
                to_divide_ids.add(d_id)
    
    return df


def postprocessing(df:pd.DataFrame, meta_info:dict, sec:int=1):
    # divide ids by IOU
    df = divide_ids_by_IOU(df)
    
    # del short length ids, threshold 3sec
    df = delete_short_id(df, meta_info['fps'], sec)

    # add overlap column, 
    df = count_overlap(df)

    # clip the dataframe
    df = clip(df, meta_info)
    return df


if __name__ == "__main__":
    RAW_DF1_PATH = "/opt/ml/final-project-level3-cv-04/test_threshold_05/df1_raw.csv"
    with open("/opt/ml/final-project-level3-cv-04/data/20230127_2242.json") as f:
        meta_info = json.load(f)
    raw_df1 = pd.read_csv(RAW_DF1_PATH, index_col=0)
    postprocessed_df1 = postprocessing(raw_df1, meta_info, sec=5)