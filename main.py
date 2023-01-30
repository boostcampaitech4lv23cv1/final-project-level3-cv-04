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
from visualization.sampling_visualization import visualize_sample
import pickle

def main(YOUTUBE_LINK):
    DOWNLOAD_PATH = './data' 

    # mp4 download and frame capture
    meta_info = ytdownload.download_and_capture(YOUTUBE_LINK, DOWNLOAD_PATH)
    
    video_name = meta_info['filename']
    exp_name = os.path.splitext(video_name)[0]
    save_dir = os.path.join('./result', exp_name)
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, 'csv'))
    
    # tracking
    clipped_df1, raw_df1 = tracking(meta_info, output=save_dir, ANALYSIS=False) # output is save dir

    # postprocessing
    df1 = postprocessing(raw_df1, meta_info, sec=5)
    df1.to_csv(os.path.join(save_dir, "csv/df1_postprocessed.csv"))

    # sampling for extract body, face feature
    df2 = sampler.sampler(df1, meta_info, seconds_per_frame=5)
    df2.to_csv(os.path.join(save_dir, "csv/df2_sampled.csv"))
    

    # load saved face feature vector
    with open("./pretrained_weight/anchor_face_embedding.json", "r", encoding="utf-8") as f:
        anchor_face_embedding = json.load(f)

    # query face similarity
    df2 = face_embedding.face_embedding_extractor(df1, df2, anchor_face_embedding, meta_info)
    df2.to_csv(os.path.join(save_dir, "csv/df2_out_of_face_embedding.csv"))

    # make body representation
    body_anchors = generate_body_anchor(df1, df2, save_dir, group_name="aespa", meta_info=meta_info)
    df2 = body_embedding_extractor(df1, df2, body_anchors, meta_info=meta_info)
    df2.to_csv(os.path.join(save_dir, "csv/df2_out_of_body_embedding.csv"))
    
    # sampling df2 visualization
    visualize_sample(df1, df2, save_dir, meta_info=meta_info,)
    
    # predictor
    pred = predictor.predictor(df2, 1, 1)
    with open(os.path.join(save_dir, 'csv/pred.pickle'),'wb') as pred_pickle:
        pickle.dump(pred, pred_pickle)
    
    video_generator(df1, meta_info, member='aespa_winter', pred=pred, full_video = True)
    
    return None


    

if __name__ == "__main__":
    YOUTUBE_LINK = "https://www.youtube.com/watch?v=0lXwMdnpoFQ" # target video
    # YOUTUBE_LINK = "https://youtu.be/fPpbfQiisA0" # hard sample
    result = main(YOUTUBE_LINK)
