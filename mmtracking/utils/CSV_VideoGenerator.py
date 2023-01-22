import pandas as pd
import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tqdm import tqdm

import Make_individual_video
# from Make_individual_video import make_mp4 # custom func

def clip_and_save(df1:pd.DataFrame, imgs_path:str, out_path:str) -> str:
    nan_cnt = 0
    plt.figure(figsize=(10,10))
    
    for idx, r in df1.iterrows():
        
        # np nan check
        if np.isnan(r["track_id"]): # np의 nan은 이런 방식으로 처리해야 함
            nan_cnt+=1
            continue
    
        # read img
        img = cv2.imread(osp.join(imgs_path, r['filename']), cv2.IMREAD_UNCHANGED)

        # clip bbox
        id = str(int(r["track_id"]))
        ymin = int(r["track_body_ymin"])
        ymax = int(r["track_body_ymax"])
        xmin = int(r["track_body_xmin"])
        xmax = int(r["track_body_xmax"])
        img = img[int(r["track_body_ymin"]):int(r["track_body_ymax"]), int(r["track_body_xmin"]):int(r["track_body_xmax"]), :] # h*w*c

        # make dir
        save_dir_path = osp.join(out_path, id)
        os.makedirs(save_dir_path, exist_ok=True)
        
        # save clip img
        clipped_img_save_path = osp.join(save_dir_path, f"{id}_{str(r['frame'])}.jpg")
        cv2.imwrite(clipped_img_save_path, img)

    # for check last img
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # BRG2RGB and show
    
    return out_path, os.listdir(out_path)


def gen_csv_to_video(df:pd.DataFrame, imgs_path:str, output_path:str):
    clip_path, dirs = clip_and_save(df, imgs_path, output_path) # clip_path is same out path
    Make_individual_video.make_mp4(clip_path, output_path)
    
    # delete temp foler
    for dir_name in dirs:
        shutil.rmtree(osp.join(clip_path,dir_name))
    return output_path

if __name__ == "__main__":
    CSV_PATH = "/opt/ml/final-project-level3-cv-04/test/postprocessed_df1.csv"
    FULL_IMG_PATH = "/opt/ml/final-project-level3-cv-04/data/20230122_0246"

    df = pd.read_csv(CSV_PATH, index_col=0)
    path = gen_csv_to_video(df, FULL_IMG_PATH, output_path="../../test/gen_videos")
    print(f"{path}에 비디오가 생성되었습니다")