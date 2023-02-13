import os
import os.path as osp
import sys
import pickle
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "face_embedding"))
sys.path.append(os.path.join(os.path.dirname(__file__), "body_embedding"))

import download_youtube.YoutubeDownloader as ytdownload
from mmtracking.utils.Tracker import tracking
from mmtracking.utils.Postprocesser import postprocessing
from timeline.TimeLineMaker import make_timeline
import sampler
import face_embedding
import group_recognizer
import predictor


from body_embedding.BodyEmbed import body_embedding_extractor
from body_embedding.BodyEmbed import generate_body_anchor
import json
import pandas as pd
from video_generator.VideoGenerator import video_generator
from visualization.sampling_visualization import visualize_sample
from video_generator.AudioMixer import mix_audio_video
from video_generator.SmoothingVideoGenerator import smooth_video_generator
import pickle


def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main(YOUTUBE_LINK, 
         start_sec:int=0, 
         end_sec:int=60, 
         member='aespa_karina', 
         smoothing=False, 
         window_ratio=0.5, 
         aspect_ratio=0., 
         shift_bb=0.2):

    youtube_id = YOUTUBE_LINK.split('=')[-1]
    save_dir = osp.join('./result', youtube_id, f"{start_sec}_{end_sec}") # save_dir_name chage from endtime to starttime_endtime
    print(f'save_dir : {save_dir}') # ex) if you assign start time 30 and end time 60sec so save_dir: "./result/0lXwMdnpoFQ/30_60"
    

    #  0. mp4 download and frame capture
    meta_info_path = osp.join(save_dir, f'{youtube_id}.json') 
    if osp.exists(meta_info_path):
        with open(meta_info_path) as meta_info_file:
            meta_info = json.load(meta_info_file)
        print(f'download_and_capture function skip...')
        print(f'load from {meta_info_path}')
    else: # mp4 download and frame capture
        os.makedirs(save_dir, exist_ok=True) # make dir if not exist
        os.makedirs(osp.join(save_dir, 'csv'), exist_ok=True) # create dir : save_dir/csv 
        meta_info = ytdownload.download_and_capture(YOUTUBE_LINK, start_sec, end_sec, save_dir)


    #  1. tracking 
    raw_df1_path = osp.join(save_dir, 'csv/df1_raw.csv')
    if osp.exists(raw_df1_path):
        print(f'tracking function skip...')
        print(f'load from {raw_df1_path}')
        raw_df1 = pd.read_csv(raw_df1_path)
    else:
        clipped_df1, raw_df1 = tracking(meta_info, output=save_dir, ANALYSIS=False) # output is save dir


    #  2. postprocessing 
    df1_postprocessed_path = osp.join(save_dir, "csv/df1_postprocessed.pickle")
    if os.path.exists(df1_postprocessed_path):
        with open(df1_postprocessed_path, 'rb') as df1_postprocessed_pickle:
            df1 = pickle.load(df1_postprocessed_pickle)
        print(f'postprocessing function skip...')
        print(f'load from {df1_postprocessed_path}')
    else:
        df1 = postprocessing(raw_df1, meta_info, sec=1)
        save_pickle(df1_postprocessed_path, df1) ## save
    

    #  3. sampling for extract body, face feature 
    df2_sampled_path = osp.join(save_dir, "csv/df2_sampled.pickle")
    if os.path.exists(df2_sampled_path):
        with open(df2_sampled_path, 'rb') as df2_sampled_pickle:
            df2 = pickle.load(df2_sampled_pickle)
        print(f'sampler function skip...')
        print(f'load from {df2_sampled_path}')
    else:
        df2 = sampler.sampler(df1, meta_info, seconds_per_frame=1)
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
        print(f'face_embedding_extractor_all function skip...')
        print(f'load from {df1_face_path}')
    else:
        df1 = face_embedding.face_embedding_extractor_all(df1, df2, current_face_anchors, meta_info)
        save_pickle(df1_face_path, df1) ## save    


    #  5. query face similarity 
    df2_out_of_face_embedding_path = osp.join(save_dir, 'csv/df2_out_of_face_embedding.pickle')
    if osp.exists(df2_out_of_face_embedding_path):
        with open(df2_out_of_face_embedding_path, 'rb') as df2_out_of_face_embedding_pickle:
            df2 = pickle.load(df2_out_of_face_embedding_pickle)
        print(f'face_embedding_extractor function skip...')
        print(f'load from {df2_out_of_face_embedding_path}')
    else:
        df2 = face_embedding.face_embedding_extractor(df1, df2, current_face_anchors, meta_info)
        save_pickle(df2_out_of_face_embedding_path, df2) ## save


    #  6. make body representation 
    df2_out_of_body_embedding_path = osp.join(save_dir, 'csv/df2_out_of_body_embedding.pickle')
    if osp.exists(df2_out_of_body_embedding_path):
        with open(df2_out_of_body_embedding_path, 'rb') as df2_out_of_body_embedding_pickle:
            df2 = pickle.load(df2_out_of_body_embedding_pickle)
        print(f'generate_body_anchor, body_embedding_extractor function skip...')
        print(f'load from {df2_out_of_body_embedding_path}')
    else:
        body_anchors = generate_body_anchor(df1, df2, save_dir, meta_info=meta_info) #, group_name="aespa"
        df2 = body_embedding_extractor(df1, df2, body_anchors, meta_info=meta_info)
        save_pickle(df2_out_of_body_embedding_path, df2) ## save
            
    #  extra. sampling df2 visualization for debugging
    # visualize_sample(df1, df2, save_dir, meta_info=meta_info)
    

    #  7. predictor
    pred_path = osp.join(save_dir, 'csv/pred.pickle')
    if osp.exists(pred_path):
        with open(pred_path, 'rb') as pred_pickle:
            pred = pickle.load(pred_pickle)
        print(f'predictor function skip...')
        print(f'load from {pred_path}')
    else:
        pred = predictor.predictor(df1, df2, face_coefficient=1, body_coefficient=1, no_duplicate=True)
        save_pickle(pred_path, pred)


    # 8. video generator
    video_type = "smoothing" if smoothing else "normal"
    print(f'generating {video_type} video...')



    if smoothing:
        # smooth normal view example
        video_path = smooth_video_generator(df1, meta_info, member, pred, save_dir, 
                                            window_ratio, aspect_ratio, shift_bb, fullscreen=True)

        # smooth vertical view example
        # video_path = smooth_video_generator(df1,  meta_info, member, pred, save_dir, 
        #                                     window_ratio=0.5, aespect_ratio=0.5, shift_bb=0.5, fullscreen=True)

        # smooth horizontal view example
        # video_path = smooth_video_generator(df1,  meta_info, member, pred, save_dir, 
        #                                     window_ratio=0.5, aespect_ratio=2.5, shift_bb=0.01, fullscreen=True)
    
    else:
        video_path = video_generator(df1, meta_info, member, pred, save_dir)


    #  9. audio mix
    file_name = osp.basename(video_path).split('.')[0] + "_mixed_audio.mp4" # final name
    mix_audio_video_path = osp.join(save_dir, file_name)
    mix_audio_video(video_path, meta_info, save_dir)

    return mix_audio_video_path


'''
sample video
"https://www.youtube.com/watch?v=0lXwMdnpoFQ" # aespa, illusion baseline
"https://www.youtube.com/watch?v=fPpbfQiisA0" # aespa, illusion stage mix
"https://www.youtube.com/watch?v=13aW5zJ832U" # newjeans, cookie
"https://www.youtube.com/watch?v=rpyjbG6DC4g" # newjeans, hype boy
'''

def get_args_parser():
    parser = argparse.ArgumentParser('torchkpop', add_help=False)
    parser.add_argument('--url', default='https://www.youtube.com/watch?v=0lXwMdnpoFQ', type=str)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=30, type=int)
    parser.add_argument('--member', default='aespa_karina', type=str)
    parser.add_argument('--smoothing', action='store_true') # default false
    parser.add_argument('--window_ratio', default=0.5, type=float) # smoothing generator using this params
    parser.add_argument('--aspect_ratio', default=0., type=float) # smoothing generator using this params
    parser.add_argument('--shift_bb', default=0.2, type=float) # smoothing generator using this params
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser('torck kpop', parents=[get_args_parser()])
    args = parser.parse_args()

    for arg in vars(args):
        print("--"+arg, getattr(args, arg))
    
    result = main(args.url, 
                  args.start, 
                  args.end, 
                  args.member, 
                  args.smoothing, 
                  args.window_ratio, 
                  args.aspect_ratio, 
                  args.shift_bb)