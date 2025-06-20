import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from io import StringIO
import os

st.set_page_config(page_title="Medicaid Data Analyzer", layout="wide")

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("💊 Medicaid Data Analyzer")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Load and cache data
@st.cache_data(show_spinner=False)
def load_data(upload):
    df = pd.read_csv(upload)
    return df

# Analysis: Top reimbursed products
def analyze_top_products_by_reimbursed(df):
    df.columns = df.columns.str.lower()
    product_col = [col for col in df.columns if 'product' in col or 'drug' in col][0]
    amount_col = [col for col in df.columns if 'reimbursed' in col or 'amount' in col][0]
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
    top_products = df.groupby(product_col)[amount_col].sum().sort_values(ascending=False).head(3)
    return top_products

# Analysis: Total reimbursed
def get_total_reimbursed(df):
    df.columns = df.columns.str.lower()
    amount_col = [col for col in df.columns if 'reimbursed' in col or 'amount' in col][0]
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
    return df[amount_col].sum()

# GPT Explanation
def explain_with_gpt(summary):
    messages = [
        {"role": "system", "content": "You are a helpful medical data analyst."},
        {"role": "user", "content": f"Please explain the following summary:
{summary}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(50), use_container_width=True)

    question = st.text_input("Ask a question about the dataset:")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Get Answer"):
            if "top" in question and "reimbursed" in question:
                try:
                    top_products = analyze_top_products_by_reimbursed(df)
                    st.write("Top 3 most reimbursed products:")
                    for name, amount in top_products.items():
                        st.markdown(f"- **{name}**: ${amount:,.2f}")
                except Exception as e:
                    st.error(f"Failed to analyze data: {e}")
            elif "total" in question and "reimbursed" in question:
                total = get_total_reimbursed(df)
                st.success(f"Total amount reimbursed: ${total:,.2f}")
            else:
                st.warning("This app currently supports questions about top reimbursed drugs and total reimbursements.")

    with col2:
        if st.button("Explain with GPT"):
            if "top" in question and "reimbursed" in question:
                top_products = analyze_top_products_by_reimbursed(df)
                summary = top_products.to_string()
                explanation = explain_with_gpt(summary)
                st.markdown(explanation)
            elif "total" in question and "reimbursed" in question:
                total = get_total_reimbursed(df)
                explanation = explain_with_gpt(f"The total amount reimbursed is ${total:,.2f}")
                st.markdown(explanation)
            else:
                st.warning("Only supported for reimbursement questions right now.")
else:
    st.info("Please upload a CSV file to begin.")
