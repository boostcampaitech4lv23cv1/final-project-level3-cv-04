# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.express as px
import json


# from app_timeline_maker import app_timeline_maker

# for on_click
def session_change_to_timeline(url:str):
    print("pushed button [TIMELINE]")
    st.session_state.page = 1
    st.session_state.url = url

# for on_click
def session_change_to_video(names:list):
    print("pushed button [VIDEO]")
    st.session_state.page = 2
    st.session_state.names = names


# main page
def main_page():
    # st.sidebar.markdown("# Main page 🎈")
    st.title("Torch-kpop")
    st.title("AI makes personal videos for you 😍")
    url = st.text_input(label="Input youtube URL 🔻", placeholder="https://www.youtube.com/watch?v=KXX3F4j1xjo")
    
    # if input btn clicked
    if st.button("SHOW TARGET VIDEO"):
        # check youtube url  # regex reference from https://stackoverflow.com/questions/19377262/regex-for-youtube-url
        if re.match("^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$", url):
            # if matching youtube url, show input video, and if you click SHOW TIMELINE button, session_change_to_timeline
            st.video(url)
            if st.button("SHOW TIMELINE", on_click=session_change_to_timeline, args=(url,)): # 
             pass
        else:
            st.write("Input is not youtube URL, check URL")        

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
    # timeline, df1_name_tagged_GT = app_timeline_maker(st.session_state.url)
    ############## 🛠 임시방편 ##############
    with open("timeline_GT.json", "r") as timeline_json: # 
        timeline = json.load(timeline_json)
    df1_name_tagged_GT = pd.read_csv('/opt/ml/torchkpop/df1_name_tagged_GT.csv')
    ############## 🛠 임시방편 ##############
    members_list = list(df1_name_tagged_GT['name'].unique())
    members_list.remove('background') # ['aespa_giselle', 'aespa_ningning', 'aespa_winter', 'aespa_karina']
        
    # show timeline 🔻
    timeline_fig = get_timeline_fig(timeline)
    st.plotly_chart(timeline_fig, use_container_width=False)
    
    if st.button("MAKE PERSONAL VIDEO", on_click=session_change_to_video): # 
        video_page()

# video page
def video_page():
    st.title("Individual Video 🎵")
    st.text(st.session_state.names)
    st.text("If you want make another video, press F5 🔁")

# init session_state
if "page" not in st.session_state:
    st.session_state.page = 0
    print("not assinged")

# hide the side bar
st.set_page_config(initial_sidebar_state="collapsed")

# print cli current page number
if st.session_state.page == 0:
    main_page()
elif st.session_state.page == 1:
    timeline_page()
elif st.session_state.page == 2:
    video_page()