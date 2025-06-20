import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from io import StringIO
from fuzzywuzzy import process

# Securely set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# App layout
st.set_page_config(page_title="Medicaid Drug Spending NLP", layout="wide")
st.title("💊 Medicaid Drug Spending NLP Analytics")

# Sidebar inputs
with st.sidebar:
    st.header("1️⃣ Upload Medicaid CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    st.markdown("---")
    st.header("2️⃣ Ask a Question")
    question = st.text_input("Enter your question about the data")

    col1, col2 = st.columns(2)
    text_btn = col1.button("💬 Get Text Answer")
    chart_btn = col2.button("📊 Generate Chart")

# Stop if no file
if not uploaded_file:
    st.info("Please upload a CSV file to get started.")
    st.stop()

# Load CSV (entire dataset)
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

df = load_csv(uploaded_file)

# Show preview
st.subheader("🔍 Data Preview")
st.dataframe(df.head(10))

# Helper to fuzzy-match user queries to column names
def match_column(user_input, column_names):
    match, score = process.extractOne(user_input.lower(), column_names)
    return match if score > 70 else None

# OpenAI text reasoning (with actual summary stats from full dataset)
def ask_openai(question, df):
    # Derive real top items using fuzzy logic
    col_product = match_column("product name", df.columns)
    col_amount = match_column("total amount reimbursed", df.columns)

    answer_context = ""
    if col_product and col_amount:
        grouped = df.groupby(col_product)[col_amount].sum().sort_values(ascending=False).head(10)
        answer_context = "\n".join([f"{k}: ${v:,.2f}" for k, v in grouped.items()])

    prompt = (
        f"You are a helpful data analyst. Answer questions using this dataset summary:
"
        f"{answer_context}
"
        f"Now answer this concisely: {question}"
    )

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Generate chart

def generate_chart(df, question):
    num_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(exclude='number').columns.tolist()

    y_col = match_column("amount reimbursed", num_cols)
    x_col = match_column("product name", cat_cols)

    if not x_col or not y_col:
        st.error("Could not determine suitable columns to chart. Try rephrasing the question.")
        return

    chart_data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))  # smaller figure
    chart_data.plot(kind="bar", ax=ax)
    ax.set_title(f"Top 10 {x_col} by {y_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

# Buttons logic
if text_btn:
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Asking OpenAI..."):
            answer = ask_openai(question, df)
            st.success("Answer:")
            st.markdown(answer)

if chart_btn:
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating chart..."):
            generate_chart(df, question)
