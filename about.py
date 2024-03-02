import streamlit as st
from streamlit_lottie import st_lottie
import json

def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

def About():
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.write('##')
            st.title('')
            st.title('')
            st.title("Hello, I'm Shaswat!")
            st.subheader("""I'm an aspiring data scientist passionate about AI and machine learning. I leverage data-driven solutions to solve complex problems and explore the intersection of technology and human intelligence.""")
      
        with col2:
            st_lottie(load_lottie('home page gif1.json'))

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st_lottie(load_lottie('about 2.json'))
        with col2:
            st.write('##')
            st.title('')
            st.title('')
            st.title('')
            st.subheader("""I leverage my expertise in Python libraries like Pandas, NumPy, TensorFlow, scikit-learn, and more to uncover data trends, train predictive models, and craft insightful visualizations. This allows me to deliver impactful solutions to complex data challenges.  In addition, I'm skilled in Streamlit for interactive web apps, Telebot for chatbots, and SQL for database management. My experience with Power BI and Excel ensures I can communicate these complex insights clearly and compellingly.""")
