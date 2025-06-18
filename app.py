import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import io

# Set your OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📊 AI-Powered Data Analysis")
st.write("Upload a CSV file and ask questions about your data using natural language.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

# Load and cache data
@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

df = None
if uploaded_file:
    try:
        df = load_data(uploaded_file)
        st.success("File uploaded successfully.")
        st.dataframe(df.head(10))  # Preview
    except Exception as e:
        st.error(f"Error loading CSV: {e}")

# Input and buttons
if df is not None:
    question = st.text_input("Ask a question about your dataset:")
    col1, col2 = st.columns([1, 1])

    with col1:
        text_btn = st.button("Text Answer")

    with col2:
        chart_btn = st.button("Chart Answer")

    def generate_text_answer(df: pd.DataFrame, question: str) -> str:
        col_info = "\n".join([f"- {col}: {str(dtype)}" for col, dtype in df.dtypes.items()])
        try:
            stats = df.describe(include="all", datetime_is_numeric=True).fillna("").to_string()
        except TypeError:
            stats = df.describe(include="all").fillna("").to_string()

        prompt = f"""
You are a data analyst. Answer concisely and directly using the dataset below.

Dataset columns:
{col_info}

Summary statistics:
{stats}

Answer this question briefly, no explanation unless asked:
Question: {question}
"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def generate_chart(df: pd.DataFrame, question: str):
        sample_csv = df.head(200).to_csv(index=False)
        prompt = f"""
You are a Python data analyst. The user has uploaded a dataset and wants to generate a chart. 
They asked: "{question}"

Generate a single Python matplotlib or pandas plot using the dataframe called df.
Only return the Python code between triple backticks. Don't include explanations or text.

Here is a sample of the data (CSV format):
{sample_csv}
"""

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        code = response.choices[0].message.content
        import re
        match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL)
        code = match.group(1) if match else code

        # Execute code safely
        try:
            exec_globals = {"df": df, "plt": plt}
            exec(code, exec_globals)
            st.pyplot(plt)
            plt.clf()
        except Exception as e:
            st.error(f"Failed to render chart: {e}")
            st.code(code)

    if text_btn and question:
        with st.spinner("Generating answer..."):
            try:
                answer = generate_text_answer(df, question)
                st.success("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")

    if chart_btn and question:
        with st.spinner("Generating chart..."):
            try:
                generate_chart(df, question)
            except Exception as e:
                st.error(f"OpenAI chart generation failed: {e}")
