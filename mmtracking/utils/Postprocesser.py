import pandas as pd
import numpy as np
import json



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

def calculate_iou(df:pd.DataFrame)->pd.DataFrame:
    return df

def count_overlap(df:pd.DataFrame)->pd.DataFrame:
    for trk_id in sorted(df['track_id'].unique()):
        if np.isnan(trk_id): 
            continue
        print(df[df['track_id'] == trk_id])
        # print(df[df['track_id'] == trk_id])
        
        break


    return df

def postprocessing(df:pd.DataFrame, meta_info:dict, sec:int=5):
    # del short length ids, threshold 3sec
    df = delete_short_id(df, meta_info['fps'], sec)

    # add overlap column
    # df = count_overlap(df) # 🐬 개발 중!

    # front check

    # re-assign id 

    # clip the dataframe
    df = clip(df, meta_info)
    return df

if __name__ == "__main__":
    RAW_DF1_PATH = "/opt/ml/final-project-level3-cv-04/test_RGB/df1_raw.csv"
    with open("/opt/ml/final-project-level3-cv-04/data/20230122_1446.json") as f:
        meta_info = json.load(f)
    raw_df1 = pd.read_csv(RAW_DF1_PATH, index_col=0)
    postprocessed_df1 = postprocessing(raw_df1, meta_info, sec=5)