import os
import os.path as osp
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "face_embedding"))
sys.path.append(os.path.join(os.path.dirname(__file__), "body_embedding"))

import download_youtube.YoutubeDownloader as ytdownload   #형훈
from mmtracking.utils.Tracker import tracking
from mmtracking.utils.Postprocesser import postprocessing # 형훈
import sampler                              # 영동
import face_embedding                       # 영동
import predictor                            # 영동


from body_embedding.BodyEmbed import body_embedding_extractor # 상헌
from body_embedding.BodyEmbed import generate_body_anchor # 상헌
import json
import pandas as pd
from video_generator.MakingVideo import video_generator


def main(YOUTUBE_LINK):
    DOWNLOAD_PATH = './data' 

    # mp4 download and frame capture
    meta_info = ytdownload.download_and_capture(YOUTUBE_LINK, DOWNLOAD_PATH)
    
    # tracking
    clipped_df1, raw_df1 = tracking(meta_info, output='./test', ANALYSIS=True) # output is save dir

    # postprocessing
    # raw_df1 = pd.read_csv("/opt/ml/final-project-level3-cv-04/test_threshold_07/df1_raw.csv", index_col=0)
    # with open("/opt/ml/final-project-level3-cv-04/data/20230127_2242.json") as f:
        # meta_info = json.load(f)
    df1 = postprocessing(raw_df1, meta_info, sec=5)
    df1.to_csv("./test/df1_postprocessed.csv")

    # sampling for extract body, face feature
    # df2 = sampler.sampler(df1, meta_info, seconds_per_frame=5)
    # df2.to_csv("./test_ENV/df2_sampled.csv")

    # load saved face feature vector
    # with open("./pretrained_weight/anchor_face_embedding.json", "r", encoding="utf-8") as f:
        # anchor_face_embedding = json.load(f)

    # query face similarity
    # df2 = face_embedding.face_embedding_extractor(df1, df2, anchor_face_embedding, meta_info)
    # df2.to_csv("./test_ENV/df2_out_of_face_embedding.csv")

    # make body representation
    # body_anchors = generate_body_anchor(df1, df2, group_name="aespa", meta_info=meta_info)
    # df2 = body_embedding_extractor(df1, df2, body_anchors, meta_info=meta_info)
    
    # predictor
    # pred = predictor.predictor(df2, 1, 1)
    # print(pred)
    
    # del df2
    # del clipped_df1
    # del raw_df1
    # video_generator(df1, meta_info, member='aespa_ningning', pred=pred, full_video = True)
    
    return None


    

if __name__ == "__main__":
    YOUTUBE_LINK = "https://www.youtube.com/watch?v=0lXwMdnpoFQ" # target video
    # YOUTUBE_LINK = "https://youtu.be/fPpbfQiisA0" # hard sample
    result = main(YOUTUBE_LINK)
