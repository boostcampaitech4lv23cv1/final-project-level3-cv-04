# hello world!
import pandas as pd
import numpy as np
import json

def postprocessing(df1:pd.DataFrame, meta_info:dict, sec:int=5):
    # del short length ids, threshold 3sec
    df1 = df1.copy()
    print(f"before filtering id_cnt: {len(df1['track_id'].unique())}")


    # just keeping ids which over 5sec 
    high_rating_id_cnt = 0 # for counting remained ids
    for id in df1['track_id'].unique():
        if np.isnan(id): # nan id pass
            continue
        if len(df1[df1['track_id'] == id]) > meta_info['fps']*sec:
            high_rating_id_cnt+=1
        else:
            df1.loc[df1['track_id'] == id,'track_id'] = np.NaN
    
    # print number of remain ids
    print(f'after filtering high_rating_id_cnt: {high_rating_id_cnt}')

    # clip the track bboxes
    df1['track_body_xmin'] = df1['track_body_xmin'].clip(0, meta_info['width'])
    df1['track_body_xmax'] = df1['track_body_xmax'].clip(0, meta_info['width'])
    df1['track_body_ymin'] = df1['track_body_ymin'].clip(0, meta_info['height'])
    df1['track_body_ymax'] = df1['track_body_ymax'].clip(0, meta_info['height'])
    
    # scailing track bboxes
    # df1.loc[df1['track_id'].notna(), 'track_body_xmin'] = df1.loc[df1['track_id'].notna(), 'track_body_xmin']/meta_info['width']
    # df1.loc[df1['track_id'].notna(), 'track_body_xmax'] = df1.loc[df1['track_id'].notna(), 'track_body_xmin']/meta_info['width']
    # df1.loc[df1['track_id'].notna(), 'track_body_ymin'] = df1.loc[df1['track_id'].notna(), 'track_body_xmin']/meta_info['height']
    # df1.loc[df1['track_id'].notna(), 'track_body_ymax'] = df1.loc[df1['track_id'].notna(), 'track_body_xmin']/meta_info['height']
    return df1

if __name__ == "__main__":
    RAW_DF1_PATH = "/opt/ml/output_mmtracking/20230121_1428/df1_raw.csv"
    with open("/opt/ml/data/20230121_1428.json") as f:
        meta_info = json.load(f)
    raw_df1 = pd.read_csv(RAW_DF1_PATH, index_col=0)
    postprocessed_df1 = postprocessing(raw_df1, meta_info, sec=5)