import streamlit as st 
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
from about import About
from project import projects
from Contact import contact

st.set_page_config(
    page_title="portfolio",
    layout="wide",
)

def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)


            
selected = option_menu(
    menu_title=None,
    options=["About", "Projects", "Contact"],
    icons=["house", "book", "envelope"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal"
)

if selected == "About":
    About()
elif selected == "Projects":
    projects()
elif selected == "Contact":
    contact()
