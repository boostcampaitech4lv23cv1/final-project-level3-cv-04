import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os
import os.path as osp

from new_app_timeline_maker import app_timeline_maker
from new_app_video_maker import app_video_maker

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
    video_sec = 60 # 🛠 나중에는 None으로 들어갈거임
    save_dir = osp.join('./result', youtube_id, str(video_sec))
    # if input btn clicked
    if st.button("SHOW TARGET VIDEO"):
        # check youtube url  # regex reference from https://stackoverflow.com/questions/19377262/regex-for-youtube-url
        if re.match("^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$", url):
            # if matching youtube url, show input video, and if you click SHOW TIMELINE button, session_change_to_timeline
            st.session_state.url = url
            st.session_state.youtube_id = youtube_id
            st.session_state.video_sec = video_sec
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
        df_member_timeline = pd.DataFrame(np.ones_like(member_timeline) * (i+1), columns=[member], index=member_timeline)
        df_timeline_list.append(df_member_timeline)
        
    df_group = pd.concat(df_timeline_list, axis=1)
    for i, member in enumerate(member_list):
        df_group.replace(i+1, member)
    fig = px.scatter(df_group)
    return fig

# timeline page
def timeline_page():
    # show text
    st.title("Timeline 🎥")
    
    # get timeline by inference
    with st.spinner('please wait...'):
        video_sec = st.session_state.video_sec # 🛠 지금은 이렇게 해놓고 나중엔 none을 보내주는걸로 하자.
        url = st.session_state.url
        save_dir = st.session_state.save_dir
        df1_name_tagged, timeline, meta_info, pred = app_timeline_maker(url, save_dir, video_sec=video_sec)

    # timeline
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
    # print(members_video_paths)

    for member_video_path in members_video_paths:
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