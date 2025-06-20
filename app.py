import streamlit as st
import requests
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io
import base64

# ========== CONFIG ==========

openai.api_key = st.secrets["OPENAI_API_KEY"]

API_BASE = "https://data.cms.gov/resource/srj6-uykx.json"  # Medicaid Drug Spending API endpoint

# ========== FUNCTIONS ==========

@st.cache_data(show_spinner=False)
def load_medicaid_data(state: str, year: str) -> pd.DataFrame:
    """
    Load Medicaid drug spending data from CMS API filtered by state and year.
    Fetches all pages via pagination.
    """
    state = state.upper()
    limit = 1000
    offset = 0
    all_rows = []

    while True:
        params = {
            "$limit": limit,
            "$offset": offset,
            "state": state,
            "year": year
        }
        response = requests.get(API_BASE, params=params)
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break
        all_rows.extend(rows)
        offset += limit

    if len(all_rows) == 0:
        return pd.DataFrame()  # empty

    df = pd.DataFrame(all_rows)

    # Convert numeric columns as needed
    for col in ["total_amount_reimbursed", "number_of_prescriptions"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def summarize_data(df: pd.DataFrame) -> str:
    """
    Create a concise summary of the dataset for GPT context.
    """
    if df.empty:
        return "The dataset is empty."

    num_rows = len(df)
    columns = list(df.columns)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    text = f"The dataset has {num_rows} rows and the following columns: {', '.join(columns)}.\n"
    text += f"Numeric columns include: {', '.join(numeric_cols)}.\n"

    # Add some simple stats
    for col in numeric_cols:
        col_sum = df[col].sum()
        col_mean = df[col].mean()
        text += f"Sum of {col} is {col_sum:,.2f}, average is {col_mean:,.2f}.\n"

    # Mention top 3 products by total amount reimbursed if column exists
    if "total_amount_reimbursed" in df.columns and "product_name" in df.columns:
        top3 = (
            df.groupby("product_name")["total_amount_reimbursed"]
            .sum()
            .sort_values(ascending=False)
            .head(3)
        )
        text += "Top 3 products by total amount reimbursed are:\n"
        for prod, amt in top3.items():
            text += f"- {prod}: ${amt:,.2f}\n"

    return text

def ask_openai_chat(messages):
    """
    Query OpenAI chat completion with messages.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def generate_text_answer(df: pd.DataFrame, question: str) -> str:
    """
    Generate a text answer by sending data summary + question to GPT.
    """
    summary = summarize_data(df)
    prompt = f"Dataset summary:\n{summary}\n\nQuestion: {question}\nAnswer concisely:"
    messages = [{"role": "user", "content": prompt}]
    return ask_openai_chat(messages)

def generate_chart(df: pd.DataFrame, question: str):
    """
    Generate a chart based on GPT instructions.
    GPT should respond with JSON specifying chart type and data columns.
    """
    summary = summarize_data(df)
    prompt = (
        f"Dataset summary:\n{summary}\n\n"
        "Based on this data, generate JSON instructions for a matplotlib chart "
        f"to answer this question: {question}\n"
        "JSON format example: {\"chart\": \"bar\", \"x\": \"product_name\", \"y\": \"total_amount_reimbursed\", \"top_n\": 5}"
    )
    messages = [{"role": "user", "content": prompt}]
    json_resp = ask_openai_chat(messages)

    import json
    try:
        instr = json.loads(json_resp)
    except Exception:
        st.error("Failed to parse chart instructions from GPT.")
        st.write("GPT response:", json_resp)
        return

    chart_type = instr.get("chart")
    x_col = instr.get("x")
    y_col = instr.get("y")
    top_n = instr.get("top_n", 5)

    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"Columns {x_col} or {y_col} not found in data.")
        return

    df_plot = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots()
    if chart_type == "bar":
        df_plot.plot(kind="bar", ax=ax)
        ax.set_ylabel(y_col.replace("_", " ").title())
        ax.set_xlabel(x_col.replace("_", " ").title())
        ax.set_title(f"Top {top_n} {x_col.replace('_', ' ').title()} by {y_col.replace('_', ' ').title()}")
    else:
        st.error(f"Unsupported chart type: {chart_type}")
        return

    st.pyplot(fig)

# ========== STREAMLIT UI ==========

st.title("Medicaid Drug Spending NLP Analytics")

with st.sidebar:
    st.header("Data filters")
    state = st.text_input("Enter state abbreviation (e.g. OH)", max_chars=2)
    year = st.text_input("Enter year (e.g. 2024)", max_chars=4)

uploaded_file = None
if state and year:
    with st.spinner("Loading Medicaid data..."):
        df = load_medicaid_data(state, year)
        if df.empty:
            st.error("No data found for given state and year.")
        else:
            st.write(f"Preview of data for {state}, {year}:")
            st.dataframe(df.head())

            question = st.text_input("Ask a question about the dataset:")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Get Answer (Text)"):
                    if not question.strip():
                        st.warning("Please enter a question.")
                    else:
                        with st.spinner("Getting answer from OpenAI..."):
                            answer = generate_text_answer(df, question)
                            st.markdown("**Answer:**")
                            st.write(answer)

            with col2:
                if st.button("Create Chart"):
                    if not question.strip():
                        st.warning("Please enter a question to generate a chart.")
                    else:
                        with st.spinner("Generating chart from OpenAI..."):
                            generate_chart(df, question)
else:
    st.info("Please enter state abbreviation and year to load data.")

