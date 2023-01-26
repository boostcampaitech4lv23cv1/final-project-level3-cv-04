import streamlit as st
import time
import numpy as np
import re
import streamlit as st

st.set_page_config(initial_sidebar_state="collapsed")



def entry_page():
    st.title("Torch-kpop")
    st.title("AI makes personal videos for you 😍")
    url = st.text_input(label="Input youtube URL 🔻", placeholder="https://www.youtube.com/watch?v=0lXwMdnpoFQ")
        
    if st.button("INPUT"):

        # check youtube regex reference from https://stackoverflow.com/questions/19377262/regex-for-youtube-url
        if re.match("^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$", url):
            
            # if matching youtube url, show input video
            video = st.video(url)

            # show button moe_running_btnng
            if st.button("MAKE VIDEO"):
                pass
            
        else:
            st.write("This is not a valid youtube url")

entry_page()


    


