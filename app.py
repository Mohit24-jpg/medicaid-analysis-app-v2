import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# Set your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Medicaid Drug NLP", layout="wide")
st.title("💊 Medicaid Drug Spending NLP Analytics")

uploaded_file = st.file_uploader("Upload your Medicaid CSV file", type="csv")

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file is None:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

df = load_data(uploaded_file)

# Show preview
st.subheader("🔍 Data Preview")
st.dataframe(df.head(10), use_container_width=True)

# Fuzzy match to correct column names
def match_column(user_input, df_columns):
    match, score = process.extractOne(user_input, df_columns)
    return match if score > 70 else None

# Function to generate answer using OpenAI
def generate_text_answer(df, question):
    summary = df.describe(include="all", datetime_is_numeric=True).fillna("").to_string()
    columns = ", ".join(df.columns.tolist())
    messages = [
        {"role": "system", "content": "You are a helpful data analyst."},
        {
            "role": "user",
            "content": f"""Here is a summary of the dataset:
{summary}

Columns available: {columns}

Answer the question concisely and directly, without showing any code or suggesting external tools:
{question}
""",
        },
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Function to generate chart using GPT-suggested logic
def generate_chart(df, question):
    # Match key columns
    group_col = match_column("product name", df.columns)
    value_col = match_column("total amount reimbursed", df.columns)

    if group_col and value_col:
        chart_data = df.groupby(group_col)[value_col].sum().sort_values(ascending=False).head(5)
        fig, ax = plt.subplots(figsize=(8, 4))
        chart_data.plot(kind="bar", ax=ax, color="skyblue")
        ax.set_title(f"Top 5 {group_col} by {value_col}", fontsize=14)
        ax.set_ylabel(value_col)
        ax.set_xlabel(group_col)
        st.pyplot(fig)
    else:
        st.error("Could not find appropriate columns to build a chart.")

# User input
st.subheader("❓ Ask a Question")
question = st.text_input("Type a question about the dataset:")

col1, col2 = st.columns(2)
with col1:
    if st.button("💬 Get Text Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Asking OpenAI..."):
                answer = generate_text_answer(df, question)
                st.markdown(f"**Answer:** {answer}")

with col2:
    if st.button("📊 Generate Chart"):
        if not question.strip():
            st.warning("Please enter a question for the chart.")
        else:
            with st.spinner("Creating chart..."):
                generate_chart(df, question)
