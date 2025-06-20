import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Medicaid Drug Spending NLP Analytics")

# File uploader and inputs
uploaded_file = st.file_uploader("Upload your Medicaid CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

df = load_data(uploaded_file)
if df is None:
    st.stop()

st.subheader("Data preview (first 10 rows)")
st.dataframe(df.head(10))

# Helper: OpenAI chat completion call
def ask_openai_chat(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def find_best_column(question, columns):
    # Use fuzzy matching to find best matching column names in the dataset
    matches = []
    for col in columns:
        score = process.extractOne(question, [col])[1]
        matches.append((col, score))
    matches.sort(key=lambda x: x[1], reverse=True)
    best_cols = [col for col, score in matches if score > 50]  # threshold
    return best_cols if best_cols else columns[:3]  # fallback to first 3 columns

# Generate text answer from the full dataframe
def generate_text_answer(df, question):
    # Summary without datetime_is_numeric (fix for older pandas)
    summary = df.describe(include="all").fillna("").to_string()
    columns = df.columns.tolist()
    best_cols = find_best_column(question, columns)
    columns_str = ", ".join(best_cols)

    messages = [
        {"role": "system", "content": "You are a helpful data analyst."},
        {
            "role": "user",
            "content": (
                f"Here is a summary of the dataset:\n{summary}\n\n"
                f"Columns relevant to the question: {columns_str}\n\n"
                f"Answer the question concisely and directly: {question}"
            ),
        },
    ]

    return ask_openai_chat(messages)

# Generate chart (bar chart of top 5 by a numeric column guessed)
def generate_chart(df, question):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found to create chart.")
        return
    best_cols = find_best_column(question, numeric_cols)
    col = best_cols[0]

    # Group by product name or first string column, sum numeric col, get top 5
    group_col_candidates = df.select_dtypes(include='object').columns.tolist()
    group_col = "Product Name" if "Product Name" in group_col_candidates else group_col_candidates[0]

    top5 = df.groupby(group_col)[col].sum().sort_values(ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(6, 4))  # smaller figure size
    top5.plot(kind='bar', ax=ax)
    ax.set_title(f"Top 5 {group_col} by {col}")
    ax.set_ylabel(col)
    ax.set_xlabel(group_col)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    st.pyplot(fig)

st.subheader("Ask a question about the dataset")

question = st.text_input("Enter your question here:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Get Text Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_text_answer(df, question)
                st.markdown(f"**Answer:** {answer}")

with col2:
    if st.button("Create Chart"):
        if not question.strip():
            st.warning("Please enter a question for the chart.")
        else:
            with st.spinner("Creating chart..."):
                generate_chart(df, question)
