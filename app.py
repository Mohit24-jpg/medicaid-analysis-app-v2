import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io

# Set OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Medicaid Drug Spending NLP Analytics")

with st.sidebar:
    st.header("Filters & Data Upload")
    state = st.text_input("Enter state abbreviation (e.g. OH)", max_chars=2)
    year = st.text_input("Enter year (e.g. 2024)", max_chars=4)
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if not (state and year and uploaded_file):
    st.info("Please enter state abbreviation, year, and upload a CSV file to proceed.")
    st.stop()

# Load CSV from upload
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

df = load_data(uploaded_file)
if df is None:
    st.stop()

st.subheader(f"Data preview: {state.upper()} {year}")
st.dataframe(df.head(10))

# Helper: call OpenAI ChatCompletion with conversation
def ask_openai_chat(messages):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Generate answer (text) by sending question + CSV summary
def generate_text_answer(df, question):
    # Provide summary and column list
    summary = df.describe(include="all", datetime_is_numeric=True).fillna("").to_string()
    columns = ", ".join(df.columns.tolist())
    messages = [
        {"role": "system", "content": "You are a helpful data analyst assistant."},
        {
            "role": "user",
            "content": (
                f"Here is a summary of the dataset:\n{summary}\n"
                f"Columns available: {columns}\n"
                f"Answer the question concisely: {question}"
            ),
        },
    ]
    return ask_openai_chat(messages)

# Generate chart (bar chart example)
def generate_chart(df, question):
    # Let GPT suggest columns and chart type, here simplified
    # For demo, just show top 5 by a numeric column named vaguely "amount"
    # In production, use GPT to parse question and suggest chart type & columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found to create chart.")
        return
    col = numeric_cols[0]
    top5 = df.groupby(df.columns[0])[col].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots()
    top5.plot(kind='bar', ax=ax)
    ax.set_title(f"Top 5 {df.columns[0]} by {col}")
    ax.set_ylabel(col)
    st.pyplot(fig)

st.subheader("Ask a question about the dataset")
question = st.text_input("Enter your question here")

col1, col2 = st.columns(2)

with col1:
    if st.button("Get Text Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_text_answer(df, question)
                st.markdown(f"**Answer:** {answer}")

with col2:
    if st.button("Create Chart"):
        if question.strip() == "":
            st.warning("Please enter a question for the chart.")
        else:
            with st.spinner("Creating chart..."):
                generate_chart(df, question)
