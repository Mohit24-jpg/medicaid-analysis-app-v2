import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import json

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Medicaid Drug Analytics", layout="wide")

# --- UI Header ---
st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/main/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")
st.markdown("#### Ask any question about the dataset below and generate insights or charts without manual coding.")

# --- Data Loading and Cleaning ---
CSV_URL = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"

@st.cache_data(show_spinner=True)
def load_and_clean():
    df = pd.read_csv(CSV_URL)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    for col in ['units_reimbursed', 'number_of_prescriptions', 'total_amount_reimbursed',
                'medicaid_amount_reimbursed', 'non_medicaid_amount_reimbursed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

df = load_and_clean()
if df.empty:
    st.error("Failed to load dataset. Please check the CSV URL.")
    st.stop()

# --- Define Functions for OpenAI Function Calling ---

def count_unique(column: str) -> int:
    if column in df.columns:
        return int(df[column].nunique())
    raise ValueError(f"Column '{column}' not found")


def sum_column(column: str) -> float:
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        return float(df[column].sum())
    raise ValueError(f"Column '{column}' missing or not numeric")


def top_n(column: str, n: int) -> dict:
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        series = df.groupby('product_name')[column].sum().sort_values(ascending=False).head(n)
        return series.to_dict()
    raise ValueError(f"Cannot compute top_n for '{column}'")


def bottom_n(column: str, n: int) -> dict:
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        series = df.groupby('product_name')[column].sum().sort_values(ascending=True).head(n)
        return series.to_dict()
    raise ValueError(f"Cannot compute bottom_n for '{column}'")

# Define function metadata
functions = [
    {
        "name": "count_unique",
        "description": "Count unique values in a column",
        "parameters": {
            "type": "object",
            "properties": {"column": {"type": "string", "description": "Column name to count unique values"}},
            "required": ["column"]
        }
    },
    {
        "name": "sum_column",
        "description": "Sum all values in a numeric column",
        "parameters": {
            "type": "object",
            "properties": {"column": {"type": "string", "description": "Numeric column to sum"}},
            "required": ["column"]
        }
    },
    {
        "name": "top_n",
        "description": "Get top N products by a numeric column",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Numeric column to rank"},
                "n": {"type": "integer", "description": "Number of top items to return"}
            },
            "required": ["column", "n"]
        }
    },
    {
        "name": "bottom_n",
        "description": "Get bottom N products by a numeric column",
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "Numeric column to rank"},
                "n": {"type": "integer", "description": "Number of bottom items to return"}
            },
            "required": ["column", "n"]
        }
    }
]

# --- UI: Interaction ---
st.subheader("📄 Data Preview")
st.dataframe(df.head(10))

st.subheader("❓ Ask a Question")
question = st.text_input("Enter question (e.g. 'Count unique utilization type', 'Top 5 by units reimbursed')")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧠 Get Text Answer"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                # Call GPT with function calling
                chat_resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an analytic assistant."},
                        {"role": "user", "content": question}
                    ],
                    functions=functions,
                    function_call="auto"
                )
                message = chat_resp.choices[0].message
                if message.get("function_call"):
                    fname = message["function_call"]["name"]
                    args = json.loads(message["function_call"]["arguments"])
                    # Dispatch
                    result = globals()[fname](**args)
                    st.markdown(f"**{fname} result:** {result}")
                else:
                    st.markdown(message.content)
with col2:
    if st.button("📊 Create Chart"):
        if not question:
            st.warning("Enter a question first.")
        else:
            with st.spinner("Charting..."):
                # Use function calling concept for chart
                chat_resp = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an analytic assistant."},
                        {"role": "user", "content": question}
                    ],
                    functions=functions,
                    function_call="auto"
                )
                message = chat_resp.choices[0].message
                if message.get("function_call"):
                    fname = message["function_call"]["name"]
                    args = json.loads(message["function_call"]["arguments"])
                    data = globals()[fname](**args)
                    # Plot
                    series = pd.Series(data)
                    fig, ax = plt.subplots(figsize=(8,4))
                    series.plot(kind='bar', ax=ax)
                    ax.set_title(f"{fname} on {args.get('column')}")
                    plt.xticks(rotation=30, ha='right')
                    st.pyplot(fig)
                else:
                    st.write(message.content)
