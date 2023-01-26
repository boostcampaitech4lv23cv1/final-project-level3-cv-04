# Contents of ~/my_app/streamlit_app.py
import streamlit as st


def move_entry_page():
    st.session_state.page = 1

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")
    st.button(label="move page2", on_click=page_change)

def timeline_page():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def video_page():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "main_page": main_page,
    "timeline_page": timeline_page,
    "video_page": video_page,
}

if "page" not in st.session_state:
    st.session_state.page = 0

st.set_page_config(initial_sidebar_state="collapsed")
if st.session_state.page == 0:
    main_page()
elif st.session_state.page == 1:
    page2()
elif st.session_state.page == 2:
    page3()