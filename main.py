import download_youtube.download_by_url as ytdownload   #형훈
from mmtracking.utils.tracker import tracking           #형훈
import sampler                              # 영동
import face_embedding                       # 영동
import body_embedding_extractor             # 상헌

import generate_body_anchor_idx             # 동영
import body_representation_generator        # 상헌
import body_expector                        # 상헌
import predictor                            # 휘준
import video_generator                      # 휘준

import pandas as pd
import os.path as osp


def main(YOUTUBE_LINK):
    meta_info = ytdownload.download_and_capture(YOUTUBE_LINK, DOWNLOAD_PATH)

    df1, _ = tracking(meta_info, 
                      output='/opt/ml/output_mmtracking/main_test', 
                      config='/opt/ml/mmtracking/configs/mot/ocsort/ocsort_yolox_x_crowdhuman_mot17-private-half-custom.py', 
                      score_thr=0.)

    
    df2 = sampler.sampler(df1=df1, num_sample=3)

    df2 = face_embedding.face_embedding_extractor(
                                                    df1=df1, df2=df2
                                                )
    
    
    #body anchor 를 만들기 위한 부분
    body_anchors = generate_body_anchor_idx(df2, 'aespa')
    
    df2 = body_embedding_extractor(df1,
                                  df2,
                                  body_anchors,
                                  meta_info)
    
    pred = predictor(df2)
    # pred = {1: 'karina', 2: 'winter . . . . . .}
    
    
    result_video = video_generator(df1, pred, meta_info)
    
    return result_video


    

if __name__ == "__main__":
    DOWNLOAD_PATH = '/opt/ml/data' 
    YOUTUBE_LINK = "https://www.youtube.com/watch?v=0lXwMdnpoFQ"
    
    result = main(YOUTUBE_LINK)
