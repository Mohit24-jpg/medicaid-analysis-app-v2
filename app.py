import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io

# Initialize OpenAI API key from secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]

def ask_openai_chat(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def generate_text_answer(df, question):
    columns = df.columns.tolist()
    system_msg = (
        f"You are an expert data analyst. The data columns are: {columns}. "
        "Answer the user question directly in clear, simple natural language, no code."
    )
    user_msg = f"Question: {question}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    return ask_openai_chat(messages)

def generate_chart_code(df, question):
    columns = df.columns.tolist()
    system_msg = (
        f"You are a Python expert who analyzes pandas DataFrames. Columns: {columns}. "
        "Write Python matplotlib code (no imports, no print statements) "
        "to produce a chart answering the user's question from this data. "
        "Assume 'df' is the DataFrame. Return only the code snippet."
    )
    user_msg = f"Generate matplotlib code for: {question}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    return ask_openai_chat(messages)

def run_chart_code(code):
    # Run the matplotlib code safely in a limited namespace
    local_vars = {"df": df, "plt": plt}
    exec(code, {}, local_vars)
    st.pyplot(plt.gcf())
    plt.clf()

st.title("🩺 Medicaid Drug Data NLP Explorer")

uploaded_file = st.file_uploader("Upload a CSV file with your data", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    st.write("Data preview:", df.head())

    question = st.text_input("Ask a question about your data")

    if st.button("Text Answer"):
        if question.strip():
            with st.spinner("Getting answer..."):
                answer = generate_text_answer(df, question)
            st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")

    if st.button("Chart"):
        if question.strip():
            with st.spinner("Generating chart..."):
                code = generate_chart_code(df, question)
                try:
                    run_chart_code(code)
                except Exception as e:
                    st.error(f"Error running generated code: {e}")
                    st.code(code)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Upload a CSV file to get started.")
