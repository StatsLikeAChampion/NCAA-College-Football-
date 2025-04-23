import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from defense import defense_analysis
import requests
from streamlit_lottie import st_lottie  

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://lottie.host/a68af006-4ddf-42d6-8131-a76e93b0ffd3/gn2CSeTVC6.json")


selected = st.sidebar.radio('Pick a criteria to analyze:', ['Home Page','Defense Analysis','Field Goal','Offense Analysis'],index=0)

if selected == 'Defense Analysis':
    defense_analysis()

elif selected == 'Home Page':
    st.title("NCAA College Football Analytics")
    st_lottie(
        lottie_hello,
        speed=1,
        reverse=False,
        loop=True,
        quality="low",
        height = 200,
        width=300,
        key=None,
    )

    st.markdown("""
    ### Project Overview

    This web app explores NCAA College Football data by analyzing **team defensive performance**.

    ####  Goals:
    - Analyze defensive performance of teams across weeks.
    - Compute and compare **Stop Rates** using logistic regression and raw stats.

    ####  Methods:
    - Logistic Regression
    - API-based data retrieval
    - Data visualization (top teams, weekly trends)

    """)
