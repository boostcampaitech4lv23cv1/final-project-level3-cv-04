import download_youtube.YoutubeDownloader as ytdownload   #형훈
from mmtracking.utils.Tracker import tracking
from mmtracking.utils.Postprocesser import postprocessing # 형훈
import sampler                              # 영동
# import face_embedding                       # 영동
# import body_embedding_extractor             # 상헌

# import generate_body_anchor_idx             # 동영
# import body_representation_generator        # 상헌
# import body_expector                        # 상헌
# import predictor                            # 휘준
# import video_generator                      # 휘준

# sys.path.append(os.path.join(os.path.dirname(__file__), "face_embedding"))


import os
import sys
import json
import pandas as pd
import os.path as osp



def main(YOUTUBE_LINK):
    DOWNLOAD_PATH = './data' 
    meta_info = ytdownload.download_and_capture(YOUTUBE_LINK, DOWNLOAD_PATH) # download_path는 meta_info.json, mp4, jpg가 저장됩니다 
    clipped_df1, raw_df1 = tracking(meta_info, output='./test_PATH', ANALYSIS=False) # output은 inference를 저장할 dir입니다
    df1 = postprocessing(raw_df1, meta_info, sec=5)
    # df1.to_csv("./test_MINI/postprocessed_df1.csv")
    
    # df1.to_csv("./test/postprocessed_df1_2.csv")
    # df2 = sampler.sampler(df1, num_sample=3)
    
    # with open("/opt/ml/torchkpop/face_embedding/anchor_face_embedding.json", "r", encoding="utf-8") as f:
    #     anchor_face_embedding = json.load(f)

    # df2 = face_embedding.face_embedding_extractor(df1, df2, anchor_face_embedding)
    # face_embedding.detect_face()
    
    # #body anchor 를 만들기 위한 부분
    # body_anchors = generate_body_anchor_idx(df2, 'aespa')
    
    # df2 = body_embedding_extractor(df1,
    #                               df2,
    #                               body_anchors,
    #                               meta_info)
    
    # pred = predictor(df2)
    # # pred = {1: 'karina', 2: 'winter . . . . . .}
    
    
    # result_video = video_generator(df1, pred, meta_info)
    
    return None


    

if __name__ == "__main__":
    YOUTUBE_LINK = "https://www.youtube.com/watch?v=0lXwMdnpoFQ"
    result = main(YOUTUBE_LINK)
