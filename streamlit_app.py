import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.express as px
import json

from app_timeline_maker import app_timeline_maker
from app_video_maker import app_video_maker

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
    
    
    # if input btn clicked
    if st.button("SHOW TARGET VIDEO"):
        # check youtube url  # regex reference from https://stackoverflow.com/questions/19377262/regex-for-youtube-url
        if re.match("^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$", url):
            # if matching youtube url, show input video, and if you click SHOW TIMELINE button, session_change_to_timeline
            st.session_state.url = url
            st.video(url)
            if st.button("SHOW TIMELINE", on_click=session_change_to_timeline): # 
                pass
        else:
            st.write("Input is not youtube URL, check URL")        

# make timeline
def get_timeline_fig(timeline):
    karina_timeline = list(set(timeline['aespa_karina']))
    winter_timeline = list(set(timeline['aespa_winter']))
    ningning_timeline = list(set(timeline['aespa_ningning']))
    giselle_timeline = list(set(timeline['aespa_giselle']))

    df_karina = pd.DataFrame(np.ones_like(karina_timeline)*1, columns=['karina'], index=karina_timeline)
    df_winter = pd.DataFrame(np.ones_like(winter_timeline)*2, columns=['winter'], index=winter_timeline)
    df_ningning = pd.DataFrame(np.ones_like(ningning_timeline)*3, columns=['ningning'], index=ningning_timeline)
    df_giselle = pd.DataFrame(np.ones_like(giselle_timeline)*4, columns=['giselle'], index=giselle_timeline)

    df_aespa = pd.concat([df_karina, df_winter, df_ningning, df_giselle], axis=1)
    df_aespa = df_aespa.replace(1, 'karina').replace(2, 'winter').replace(3, 'ningning').replace(4, 'giselle')
    
    fig = px.scatter(df_aespa)
    
    return fig

# timeline page
def timeline_page():

    # show text
    st.title("Timeline 🎥")
    
    # get timeline by inference
    with st.spinner('please wait...'):
        df1_name_tagged, timeline, meta_info, pred = app_timeline_maker(st.session_state.url)

    # timeline
    timeline_fig = get_timeline_fig(timeline)
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
        members_video_paths = app_video_maker(st.session_state.df1_name_tagged_GT, st.session_state.meta_info, st.session_state.pred)
    
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