import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
import os.path as osp
import shutil

from streamlit_timeline_maker import app_timeline_maker
from streamlit_video_maker import app_video_maker

import sys
import pickle

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
from visualization.sampling_visualization import visualize_sample



### 1️. module code startline ###

def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

### 1. module code endline ###

### 2. streamlit code startline ###

# for on_click
def session_change_to_timeline():
    st.session_state.page = 1

# for on_click
def session_change_to_video():
    st.session_state.page = 2

# main page
def main_page():
    st.title("Torch-kpop")
    st.title("AI makes personal videos for you 😍")
    url = st.text_input(label="Input youtube URL 🔻", placeholder="https://www.youtube.com/watch?v=KXX3F4j1xjo")    
    youtube_id = url.split('=')[-1]
    start_sec = 0  # ⭐
    end_sec = 40  # ⭐
    save_dir = osp.join('./streamlit_output', youtube_id, str(end_sec))
    # if input btn clicked
    if st.button("SHOW TARGET VIDEO"):
        # check youtube url  # regex reference from https://stackoverflow.com/questions/19377262/regex-for-youtube-url
        if re.match("^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$", url):
            # if matching youtube url, show input video, and if you click SHOW TIMELINE button, session_change_to_timeline
            st.session_state.url = url
            st.session_state.youtube_id = youtube_id
            st.session_state.start_sec = start_sec # ⭐
            st.session_state.end_sec = end_sec # ⭐
            st.session_state.save_dir = save_dir
            st.video(url)
            if st.button("SHOW TIMELINE", on_click=session_change_to_timeline): # 
                pass
        else:
            st.write("Input is not youtube URL, check URL")        

# make timeline
def get_timeline_fig(timeline, meta_info):
    member_list = meta_info['member_list']
    df_timeline_list = []
    for i, member in enumerate(member_list): # member 예시 : 'aespa_karina'
        member_timeline = list(set(timeline[member]))
        df_member_timeline = pd.DataFrame(np.ones_like(member_timeline) * int(i+1), columns=[member], index=member_timeline)
        df_timeline_list.append(df_member_timeline)
        
    df_group = pd.concat(df_timeline_list, axis=1)
    print(df_group)
    for i, member in enumerate(member_list):
        df_group.replace(i+1, member)
    print(df_group)
    fig = px.scatter(df_group, width=1000, labels={'index':'\nframe', 'value':'member'})
    return fig

def get_meta_info():
    start_sec = st.session_state.start_sec # ⭐
    end_sec = st.session_state.end_sec # ⭐
    YOUTUBE_LINK = st.session_state.url
    save_dir = st.session_state.save_dir
    youtube_id = YOUTUBE_LINK.split('=')[-1]
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
    return meta_info

def get_raw_df1(meta_info):
    save_dir = st.session_state.save_dir
    
    raw_df1_path = osp.join(save_dir, 'csv/df1_raw.csv')
    if osp.exists(raw_df1_path):
        print(f'🎉 tracking 함수 skip')
        print(f'load 경로 : {raw_df1_path}')
        raw_df1 = pd.read_csv(raw_df1_path)
    else:
        clipped_df1, raw_df1 = tracking(meta_info, output=save_dir, ANALYSIS=False) # output is save dir
    return raw_df1

def get_df1_postprocessed(raw_df1, meta_info, sec=1):
    save_dir = st.session_state.save_dir
    df1_postprocessed_path = osp.join(save_dir, "csv/df1_postprocessed.pickle")
    if os.path.exists(df1_postprocessed_path):
        with open(df1_postprocessed_path, 'rb') as df1_postprocessed_pickle:
            df1 = pickle.load(df1_postprocessed_pickle)
        print(f'🎉 Postprocessing 함수 skip')
        print(f'load 경로 : {df1_postprocessed_path}')
    else:
        df1 = postprocessing(raw_df1, meta_info, sec=1)
        save_pickle(df1_postprocessed_path, df1) ## save
    return df1

def get_df2_sampled(df1, meta_info, seconds_per_frame=1):
    save_dir = st.session_state.save_dir
    df2_sampled_path = osp.join(save_dir, "csv/df2_sampled.pickle")
    if os.path.exists(df2_sampled_path):
        with open(df2_sampled_path, 'rb') as df2_sampled_pickle:
            df2 = pickle.load(df2_sampled_pickle)
        print(f'🎉 sampler 함수 skip')
        print(f'load 경로 : {df2_sampled_path}')
    else:
        df2 = sampler.sampler(df1, meta_info, seconds_per_frame=1)
        save_pickle(df2_sampled_path, df2) ## save
    return df2
    
def get_group_recognized_meta_info(meta_info, anchor_face_embedding, df1, df2):
    GR = group_recognizer.GroupRecognizer(meta_info = meta_info, anchors = anchor_face_embedding)
    GR.register_dataframes(df1 = df1, df2 = df2)
    meta_info = GR.guess_group()
    return meta_info

def get_current_face_anchors(meta_info, anchor_face_embedding):
    current_face_anchors = dict()
    for k, v in anchor_face_embedding.items():
        if k in meta_info['member_list']:
            current_face_anchors[k] = v

def get_df1_face(df1, df2, current_face_anchors, meta_info):
    save_dir = st.session_state.save_dir
    df1_face_path = osp.join(save_dir, "csv/df1_face.pickle")
    if osp.exists(df1_face_path):
        with open(df1_face_path, 'rb') as df1_face_pickle:
            df1 = pickle.load(df1_face_pickle)
        print(f'🎉 face_embedding_extractor_all 함수 skip')
        print(f'load 경로 : {df1_face_path}')
    else:
        df1 = face_embedding.face_embedding_extractor_all(df1, df2, current_face_anchors, meta_info)
        save_pickle(df1_face_path, df1) ## save
    return df1

def get_df2_out_of_face_embedding(df1, df2, current_face_anchors, meta_info):
    df2_out_of_face_embedding_path = osp.join(save_dir, 'csv/df2_out_of_face_embedding.pickle')
    if osp.exists(df2_out_of_face_embedding_path):
        with open(df2_out_of_face_embedding_path, 'rb') as df2_out_of_face_embedding_pickle:
            df2 = pickle.load(df2_out_of_face_embedding_pickle)
        print(f'🎉 face_embedding_extractor 함수 skip')
        print(f'load 경로 : {df2_out_of_face_embedding_path}')
    else:
        df2 = face_embedding.face_embedding_extractor(df1, df2, current_face_anchors, meta_info)
        save_pickle(df2_out_of_face_embedding_path, df2) ## save
    return df2

def get_df2_out_of_body_embedding(df1, df2, save_dir, meta_info):
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
    return df2

def get_pred(df1, df2, face_coefficient=1, body_coefficient=1, no_duplicate=True):
    save_dir = st.session_state.save_dir
    pred_path = osp.join(save_dir, 'csv/pred.pickle')
    if osp.exists(pred_path):
        with open(pred_path, 'rb') as pred_pickle:
            pred = pickle.load(pred_pickle)
        print(f'🎉 predictor 함수 skip')
        print(f'load 경로 : {pred_path}')
    else:
        pred = predictor.predictor(df1, df2, face_coefficient=1, body_coefficient=1, no_duplicate=True)
        save_pickle(pred_path, pred)


# timeline page
def timeline_page():
    # show text
    st.title("Timeline 🎥")
    
    # get timeline by inference
    with st.spinner('please wait...'):
        start_sec = st.session_state.start_sec # ⭐
        end_sec = st.session_state.end_sec # ⭐
        YOUTUBE_LINK = st.session_state.url
        save_dir = st.session_state.save_dir
        
        # DOWNLOAD_PATH = './data' 
        youtube_id = YOUTUBE_LINK.split('=')[-1]

        #  0. mp4 download and frame capture
        meta_info = get_meta_info()
        st.info('🎉 Download and Capture complete')

        #  1. tracking 
        raw_df1 = get_raw_df1(meta_info)
        st.info('🎉 Tracking complete')
        
        #  2. postprocessing 
        df1 = get_df1_postprocessed(raw_df1, meta_info, sec=1)
        st.info('🎉 Postprocessing complete')
        
        #  3. sampling for extract body, face feature 
        df2 = get_df2_sampled(df1, meta_info, seconds_per_frame=1)
        st.info('🎉 Sampler complete')

        ## load pretrained face embedding 
        with open("./pretrained_weight/integrated_face_embedding.json", "r", encoding="utf-8") as f:
            anchor_face_embedding = json.load(f)

        # 3-1. Group Recognizer
        meta_info = get_group_recognized_meta_info(meta_info, anchor_face_embedding, df1, df2)
        # 3-2. Make new anchor face dict containing current group members
        current_face_anchors = get_current_face_anchors(meta_info, anchor_face_embedding)
        st.info('🎉 Group Recognizer complete')
        
        #  4. sampling for extract body, face feature 
        df1 = get_df1_face(df1, df2, current_face_anchors, meta_info)
        st.info('🎉 Face Embedding Extractor All complete')

        #  5. query face similarity
        df2 = get_df2_out_of_face_embedding(df1, df2, current_face_anchors, meta_info)
        st.info('🎉 Face Embedding Extractor complete')


        #  6. make body representation 
        df2 = get_df2_out_of_body_embedding(df1, df2, save_dir, meta_info=meta_info)
        st.info('🎉 Body Embedding Extractor complete')
        
        #  extra. sampling df2 visualization
        visualize_sample(df1, df2, save_dir, meta_info=meta_info)
        
        #  7. predictor
        pred = get_pred(df1, df2, face_coefficient=1, body_coefficient=1, no_duplicate=True)
        st.info('🎉 Predictor complete')
        
        # timeline maker
        df1_name_tagged, timeline = make_timeline(df1, pred)
        
        df1_name_tagged_path = osp.join(save_dir, 'csv/df1_name_tagged.csv')
        df1_name_tagged.to_csv(df1_name_tagged_path) ## save
        timeline_path = osp.join(save_dir, 'csv/timeline.pickle')
        save_pickle(timeline_path, timeline) ## save
    
    # show group name
    st.info(f'{meta_info["group"]} tilem 영상입니다!')
        
    # get timeline figure
    timeline_fig = get_timeline_fig(timeline, meta_info)
    st.plotly_chart(timeline_fig, use_container_width=False)
    st.session_state.df1_name_tagged_GT = df1_name_tagged
    st.session_state.meta_info = meta_info
    st.session_state.pred = pred
    
    if st.button("MAKE PERSONAL VIDEO", on_click=session_change_to_video):
        pass

# video page
def video_page():
    st.title("show all members Video 🎵")

    with st.spinner('please wait...'):
        df1 = st.session_state.df1_name_tagged_GT
        meta_info = st.session_state.meta_info
        pred = st.session_state.pred
        save_dir = st.session_state.save_dir
        members_video_paths = app_video_maker(df1, meta_info, pred, save_dir)
    member_list = meta_info['member_list']
    
    for video_path in members_video_paths:
        # encoding h264(dst)
        src = video_path
        dst = "./temp.mp4"
        os.system("ffmpeg " + 
            f"-i {src} " +
                    f"-c:v h264 " +
                        f"-c:a copy {dst}")
        # delete mp4v(src)
        os.remove(src)
        # move (dst) to (src)
        shutil.move(dst, src)

    for i, member_video_path in enumerate(members_video_paths):
        member_name = member_list[i]
        st.subheader(f'{member_name} 영상')
        video_file_per_member = open(member_video_path, 'rb')
        video_bytes_per_member = video_file_per_member.read()
        st.video(video_bytes_per_member)
        
    st.text("!🎉 End 🎉!")
    
# init session_state
if "page" not in st.session_state:
    st.session_state.page = 0

# print cli current page number
if st.session_state.page == 0:
    main_page()
elif st.session_state.page == 1:
    timeline_page()
elif st.session_state.page == 2:
    video_page()