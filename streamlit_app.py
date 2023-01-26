# Contents of ~/my_app/streamlit_app.py
import streamlit as st
import re

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
    url = st.text_input(label="Input youtube URL 🔻", placeholder="https://www.youtube.com/watch?v=0lXwMdnpoFQ")
    
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

# timeline page
def timeline_page():
    # show text
    st.title("Timeline 🎥")
    st.title(st.session_state.url) # use session_state.url
    # show timeline 🔻
    # 
    # st.session_state.url
    # 
    info = {"names":["karina", "winter", "ningning"]}
    if st.button("MAKE PERSONAL VIDEO", on_click=session_change_to_video, kwargs=info):
        pass

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