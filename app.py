import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import json
from difflib import get_close_matches

st.set_page_config(page_title="Medicaid Drug Spending NLP Analytics", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.markdown("""
    <style>
    .chat-box-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #ccc;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .user-msg {
        background-color: #007bff;
        color: white;
        padding: 12px;
        border-radius: 18px 18px 0 18px;
        margin: 10px 0;
        text-align: right;
        font-size: 1.05rem;
        width: fit-content;
        margin-left: auto;
    }
    .assistant-msg {
        background-color: #f1f1f1;
        color: black;
        padding: 12px;
        border-radius: 18px 18px 18px 0;
        margin: 10px 0;
        text-align: left;
        font-size: 1.05rem;
        width: fit-content;
        margin-right: auto;
    }
    .credit {
        margin-top: 30px;
        font-size: 0.9rem;
        color: #888;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# REMAINING CODE BLOCK HERE -- UNCHANGED FROM CURRENT CANVAS
# Only style and formatting was changed
