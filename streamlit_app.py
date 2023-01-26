import streamlit as st
import time
import numpy as np
import re
import streamlit as st
# from streamlit_terran_timeline import generate_timeline, terran_timeline

def start_entry_page():
    st.title("Torch-kpop")
    st.title("AI makes personal videos for you 😍")
    url = st.text_input(label="Input youtube URL 🔻", placeholder="https://www.youtube.com/watch?v=0lXwMdnpoFQ")
    
    # if click ok btn
    if st.button("ok"):
        # check youtube regex reference from https://stackoverflow.com/questions/19377262/regex-for-youtube-url
        if re.match("^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube(-nocookie)?\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$", url):
            # if matching youtube url, show video
            video = st.video(url)
            st.button("make video")
            
        else:
            st.write("This is not a valid youtube url")

    


if __name__ == "__main__":
    start_entry_page()