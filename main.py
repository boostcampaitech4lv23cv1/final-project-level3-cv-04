import os
import os.path as osp
import sys
import pickle

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
import json
import pandas as pd
from video_generator.NewVideo import video_generator
from visualization.sampling_visualization import visualize_sample
from video_generator.AudioMixer import mix_audio_video
import pickle


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main(YOUTUBE_LINK):
    DOWNLOAD_PATH = './data' 

    # 🍑 0. mp4 download and frame capture
    meta_info = ytdownload.download_and_capture(YOUTUBE_LINK, DOWNLOAD_PATH)
    
    video_name = meta_info['filename']
    exp_name = os.path.splitext(video_name)[0]
    save_dir = os.path.join('./result', exp_name)
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir, 'csv'))
    
    # 🍑 1. tracking ✔️
    clipped_df1, raw_df1 = tracking(meta_info, output=save_dir, ANALYSIS=False) # output is save dir

    # 🍑 2. postprocessing ✔️
    df1 = postprocessing(raw_df1, meta_info, sec=1)
    save_pickle(os.path.join(save_dir, "csv/df1_postprocessed.pickle"), df1) ## save
    
    # 🍑 3. sampling for extract body, face feature ✔️
    df2 = sampler.sampler(df1, meta_info, seconds_per_frame=5)
    save_pickle(os.path.join(save_dir, "csv/df2_sampled.pickle"), df2) ## save

    ## load pretraine face embedding ✔️
    with open("./pretrained_weight/anchor_face_embedding.json", "r", encoding="utf-8") as f:
        anchor_face_embedding = json.load(f)
    
    # 🍑 4. sampling for extract body, face feature ✔️
    df1 = face_embedding.face_embedding_extractor_all(df1, anchor_face_embedding, meta_info)
    save_pickle(os.path.join(save_dir, "csv/df1_face.pickle"), df1) ## save

    # 🍑 5. query face similarity ✔️
    df2 = face_embedding.face_embedding_extractor(df1, df2, anchor_face_embedding, meta_info)
    save_pickle(os.path.join(save_dir, "csv/df2_out_of_face_embedding.pickle"), df2) ## save

    # 🍑 6. make body representation 
    body_anchors = generate_body_anchor(df1, df2, save_dir, group_name="aespa", meta_info=meta_info)
    df2 = body_embedding_extractor(df1, df2, body_anchors, meta_info=meta_info)
    save_pickle(os.path.join(save_dir, "csv/df2_out_of_body_embedding.pickle"), df2) ## save
    
    # 🐛 extra. sampling df2 visualization
    visualize_sample(df1, df2, save_dir, meta_info=meta_info)
    
    # 🍑 7. predictor
    pred = predictor.predictor(df1, df2, face_coefficient=1, body_coefficient=1, no_duplicate=True)
    with open(os.path.join(save_dir, 'csv/pred.pickle'),'wb') as pred_pickle: ## save
        pickle.dump(pred, pred_pickle)

    # 🍑 8. video gen
    video_path = video_generator(df1, meta_info, member='aespa_winter', pred=pred)

    # 🍑 9. audio mix
    mix_audio_video(video_path, meta_info, save_dir)

    return None


    

if __name__ == "__main__":
    YOUTUBE_LINK = "https://www.youtube.com/watch?v=0lXwMdnpoFQ" # target video
    # YOUTUBE_LINK = "https://youtu.be/fPpbfQiisA0" # hard sample
    result = main(YOUTUBE_LINK)
