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
import group_recognizer
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

def main(YOUTUBE_LINK, start_sec:int, end_sec:int=60, member='aespa_karina'):
    # DOWNLOAD_PATH = './data' 
    youtube_id = YOUTUBE_LINK.split('=')[-1]
    save_dir = osp.join('./result', youtube_id, f"{start_sec}_{end_sec}") # save_dir_name chage from endtime to starttime_endtime
    print(f'save_dir : {save_dir}') # ex) if you assign start time 30 and end time 60sec so save_dir: "./result/0lXwMdnpoFQ/30_60"
    
    #  0. mp4 download and frame capture
    meta_info_path = osp.join(save_dir, f'{youtube_id}.json') 
    if osp.exists(meta_info_path):
        with open(meta_info_path) as meta_info_file:
            meta_info = json.load(meta_info_file)
        print(f'🎉 download_and_capture 함수 skip')
        print(f'load 경로 : {meta_info_path}')
    else: # mp4 download and frame capture
        os.makedirs(save_dir, exist_ok=True) # make dir if not exist
        os.makedirs(osp.join(save_dir, 'csv'), exist_ok=True) # create dir : save_dir/csv 
        meta_info = ytdownload.download_and_capture(YOUTUBE_LINK, start_sec, end_sec, save_dir)

    #  1. tracking 
    raw_df1_path = osp.join(save_dir, 'csv/df1_raw.csv')
    if osp.exists(raw_df1_path):
        print(f'🎉 tracking 함수 skip')
        print(f'load 경로 : {raw_df1_path}')
        raw_df1 = pd.read_csv(raw_df1_path)
    else:
        clipped_df1, raw_df1 = tracking(meta_info, output=save_dir, ANALYSIS=False) # output is save dir

    #  2. postprocessing 
    df1_postprocessed_path = osp.join(save_dir, "csv/df1_postprocessed.pickle")
    if os.path.exists(df1_postprocessed_path):
        with open(df1_postprocessed_path, 'rb') as df1_postprocessed_pickle:
            df1 = pickle.load(df1_postprocessed_pickle)
        print(f'🎉 postprocessing 함수 skip')
        print(f'load 경로 : {df1_postprocessed_path}')
    else:
        df1 = postprocessing(raw_df1, meta_info, sec=1)
        save_pickle(df1_postprocessed_path, df1) ## save
    
    #  3. sampling for extract body, face feature 
    df2_sampled_path = osp.join(save_dir, "csv/df2_sampled.pickle")
    if os.path.exists(df2_sampled_path):
        with open(df2_sampled_path, 'rb') as df2_sampled_pickle:
            df2 = pickle.load(df2_sampled_pickle)
        print(f'🎉 sampler 함수 skip')
        print(f'load 경로 : {df2_sampled_path}')
    else:
        df2 = sampler.sampler(df1, meta_info, seconds_per_frame=5)
        save_pickle(df2_sampled_path, df2) ## save

    ## load pretrained face embedding 
    with open("./pretrained_weight/integrated_face_embedding.json", "r", encoding="utf-8") as f:
        anchor_face_embedding = json.load(f)


    #  3-1. Group Recognizer
    GR = group_recognizer.GroupRecognizer(meta_info = meta_info, anchors = anchor_face_embedding)
    GR.register_dataframes(df1 = df1, df2 = df2)
    meta_info = GR.guess_group()
    
    # 3-2. Make new anchor face dict containing current group members
    current_face_anchors = dict()
    for k, v in anchor_face_embedding.items():
        if k in meta_info['member_list']:
            current_face_anchors[k] = v
    
    
    #  4. sampling for extract body, face feature 
    df1_face_path = osp.join(save_dir, "csv/df1_face.pickle")
    if osp.exists(df1_face_path):
        with open(df1_face_path, 'rb') as df1_face_pickle:
            df1 = pickle.load(df1_face_pickle)
        print(f'🎉 face_embedding_extractor_all 함수 skip')
        print(f'load 경로 : {df1_face_path}')
    else:
        df1 = face_embedding.face_embedding_extractor_all(df1, df2, current_face_anchors, meta_info)
        save_pickle(df1_face_path, df1) ## save    

    #  5. query face similarity 
    df2_out_of_face_embedding_path = osp.join(save_dir, 'csv/df2_out_of_face_embedding.pickle')
    if osp.exists(df2_out_of_face_embedding_path):
        with open(df2_out_of_face_embedding_path, 'rb') as df2_out_of_face_embedding_pickle:
            df2 = pickle.load(df2_out_of_face_embedding_pickle)
        print(f'🎉 face_embedding_extractor 함수 skip')
        print(f'load 경로 : {df2_out_of_face_embedding_path}')
    else:
        df2 = face_embedding.face_embedding_extractor(df1, df2, current_face_anchors, meta_info)
        save_pickle(df2_out_of_face_embedding_path, df2) ## save



    #  6. make body representation 
    df2_out_of_body_embedding_path = osp.join(save_dir, 'csv/df2_out_of_body_embedding.pickle')
    if osp.exists(df2_out_of_body_embedding_path):
        with open(df2_out_of_body_embedding_path, 'rb') as df2_out_of_body_embedding_pickle:
            df2 = pickle.load(df2_out_of_body_embedding_pickle)
        print(f'🎉 generate_body_anchor, body_embedding_extractor 함수 skip')
        print(f'load 경로 : {df2_out_of_body_embedding_path}')
    else:
        body_anchors = generate_body_anchor(df1, df2, save_dir, meta_info=meta_info) #, group_name="aespa"
        df2 = body_embedding_extractor(df1, df2, body_anchors, meta_info=meta_info)
        save_pickle(df2_out_of_body_embedding_path, df2) ## save
            
    
    #  extra. sampling df2 visualization
    visualize_sample(df1, df2, save_dir, meta_info=meta_info)
    
    #  7. predictor
    pred_path = osp.join(save_dir, 'csv/pred.pickle')
    if osp.exists(pred_path):
        with open(pred_path, 'rb') as pred_pickle:
            pred = pickle.load(pred_pickle)
        print(f'🎉 predictor 함수 skip')
        print(f'load 경로 : {pred_path}')
    else:
        pred = predictor.predictor(df1, df2, face_coefficient=1, body_coefficient=1, no_duplicate=True)
        save_pickle(pred_path, pred)

    #  8. video gen
    video_path = osp.join(save_dir, f'make_video_video_{member}', f'{member}_output.mp4')
    if not osp.exists(video_path):
        print('🔥 video_generator 함수 실행')
        video_path = video_generator(df1, meta_info, member=member, pred=pred, save_dir=save_dir)
        print(f'🔥 저장 경로 : {video_path}')

    #  9. audio mix
    file_name = osp.basename(video_path).split('.')[0] + "_mixed_audio.mp4" # final name
    mix_audio_video_path = osp.join(save_dir, file_name)
    if not osp.exists(mix_audio_video_path):
        print('🔥 mix_audio_video 함수 실행')
        mix_audio_video(video_path, meta_info, save_dir)
        print(f'저장 경로 : {video_path}')

    return mix_audio_video_path


    

if __name__ == "__main__":
    YOUTUBE_LINK = "https://www.youtube.com/watch?v=0lXwMdnpoFQ" # aespa baseline video(illusion)
    # YOUTUBE_LINK = "https://youtu.be/fPpbfQiisA0" # aespa hardcore video(illusion)
    # YOUTUBE_LINK = "https://www.youtube.com/watch?v=13aW5zJ832U" # newjeans cherry pick video(cookie)
    # YOUTUBE_LINK = "https://www.youtube.com/watch?v=rpyjbG6DC4g" # newjeans hypeboy video
    
    start_sec = 20
    end_sec = 40
    
    result = main(YOUTUBE_LINK, start_sec, end_sec, member='aespa_karina')
    # result = main(YOUTUBE_LINK, start_sec, end_sec, member='aespa_winter')
    # result = main(YOUTUBE_LINK, start_sec, end_sec, member='newjeans_hyein')
    # result = main(YOUTUBE_LINK, start_sec, end_sec, member='newjeans_danielle')
    # result = main(YOUTUBE_LINK, start_sec, end_sec, member='newjeans_haerin')
    # result = main(YOUTUBE_LINK, start_sec, end_sec, member='newjeans_hanni')
