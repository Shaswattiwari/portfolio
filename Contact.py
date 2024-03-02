import streamlit as st
from streamlit_lottie import st_lottie
import json
import base64 
from chat import chat

def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

def contact():
    col1, col2 = st.columns(2)
    with col1:
        st_lottie(load_lottie('messs2.json'))
                
                
    with col2:
        chat()
        