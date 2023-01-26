import os
import os.path as osp
import sys
import pickle
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "face_embedding"))
sys.path.append(os.path.join(os.path.dirname(__file__), "body_embedding"))

import download_youtube.YoutubeDownloader as ytdownload   #형훈
from mmtracking.utils.Tracker import tracking
from mmtracking.utils.Postprocesser import postprocessing # 형훈
from timeline.TimeLineMaker import make_timeline
import sampler                              # 영동
import face_embedding                       # 영동
import predictor                            # 영동
from body_embedding.BodyEmbed import body_embedding_extractor # 상헌
from body_embedding.BodyEmbed import generate_body_anchor # 상헌


def app_timeline_maker(YOUTUBE_LINK):
    DOWNLOAD_PATH = './streamlit_output' 

    # mp4 download and frame capture
    meta_info = ytdownload.download_and_capture(YOUTUBE_LINK, DOWNLOAD_PATH)
    
    # tracking
    clipped_df1, raw_df1 = tracking(meta_info, output='./streamlit_output', ANALYSIS=False) # output is save dir

    # postprocessing
    df1 = postprocessing(raw_df1, meta_info, sec=5)
    df1.to_csv("./streamlit_output/df1_postprocessed.csv")

    # sampling for extract body, face feature
    df2 = sampler.sampler(df1, meta_info, seconds_per_frame=5)
    df2.to_csv("./streamlit_output/df2_sampled.csv")

    # load saved face feature vector
    with open("./pretrained_weight/anchor_face_embedding.json", "r", encoding="utf-8") as f:
        anchor_face_embedding = json.load(f)

    # query face similarity
    df2 = face_embedding.face_embedding_extractor(df1, df2, anchor_face_embedding, meta_info)
    df2.to_csv("./streamlit_output/df2_out_of_face_embedding.csv")

    # make body representation
    body_anchors = generate_body_anchor(df1, df2, group_name="aespa", meta_info=meta_info)
    df2 = body_embedding_extractor(df1, df2, body_anchors, meta_info=meta_info)
    
    # predictor
    pred = predictor.predictor(df2, 1, 1)
    with open("./streamlit_output/pred.pickle", "wb") as pred_pickle_file:
        pickle.dump(pred, pred_pickle_file)

    # timeline maker
    df1_name_tagged, timeline_info = make_timeline(df1, pred)

    df1_name_tagged.to_csv("./test_full/df1_name_tagged.csv")
    with open("./streamlit_output/e2e_timeline.pickle", "w") as df1_pickel_file:
        pickle.dump(df1_name_tagged, df1_pickel_file)
    
    return df1_name_tagged, timeline_info, meta_info, pred
