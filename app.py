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
st.markdown("#### Ask any question about the dataset below and generate precise figures or charts without hard-coding.")

# --- Data Loading and Cleaning ---
CSV_URL = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"
@st.cache_data(show_spinner=True)
def load_and_clean():
    df = pd.read_csv(CSV_URL)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Convert numeric columns
    for col in ['units_reimbursed', 'number_of_prescriptions', 'total_amount_reimbursed',
                'medicaid_amount_reimbursed', 'non_medicaid_amount_reimbursed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

df = load_and_clean()
if df.empty:
    st.error("Failed to load dataset. Please check the CSV URL.")
    st.stop()

# --- Synonyms for column mapping ---
COLUMN_SYNONYMS = {
    'total_reimbursed': 'total_amount_reimbursed',
    'reimbursed': 'total_amount_reimbursed',
    'total_amount': 'total_amount_reimbursed',
    'units': 'units_reimbursed',
    'prescriptions': 'number_of_prescriptions',
    'medicaid': 'medicaid_amount_reimbursed',
    'non_medicaid': 'non_medicaid_amount_reimbursed',
    'utilization': 'utilization_type'
}

def canonical_col(col_name: str) -> str:
    name = col_name.strip().lower().replace(' ', '_')
    return COLUMN_SYNONYMS.get(name, name)

# --- Define Functions for OpenAI Function Calling ---

def count_unique(column: str) -> int:
    col = canonical_col(column)
    if col in df.columns:
        return int(df[col].nunique())
    raise ValueError(f"Column '{column}' not found")


def sum_column(column: str) -> float:
    col = canonical_col(column)
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        return float(df[col].sum())
    raise ValueError(f"Column '{column}' missing or not numeric")


def top_n(column: str, n: int) -> dict:
    col = canonical_col(column)
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        series = df.groupby('product_name')[col].sum().sort_values(ascending=False).head(n)
        return series.to_dict()
    raise ValueError(f"Cannot compute top_n for '{column}'")


def bottom_n(column: str, n: int) -> dict:
    col = canonical_col(column)
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        series = df.groupby('product_name')[col].sum().sort_values(ascending=True).head(n)
        return series.to_dict()
    raise ValueError(f"Cannot compute bottom_n for '{column}'")


def sum_by_product(column: str) -> dict:
    col = canonical_col(column)
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        series = df.groupby('product_name')[col].sum().sort_values(ascending=False)
        return series.to_dict()
    raise ValueError(f"Cannot group-sum for '{column}'")

# Function metadata
functions = [
    {"name": "count_unique", "description": "Count unique values in a column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "sum_column",  "description": "Sum values in a numeric column",  "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "top_n",       "description": "Get top N products by a numeric column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "bottom_n",    "description": "Get bottom N products by a numeric column","parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "sum_by_product","description": "Sum numeric column for each product","parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}}
]

# --- UI: Interaction ---
st.subheader("📄 Data Preview")
st.dataframe(df.head(10))

st.subheader("❓ Ask a Question")
question = st.text_input("Enter question (e.g. 'Top 4 by total reimbursed')")

col1, col2 = st.columns(2)
with col1:
    if st.button("Get Text Answer"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                resp = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an analytic assistant. Use function calling for precise results."},
                        {"role": "user",   "content": question}
                    ],
                    functions=functions,
                    function_call="auto"
                )
                msg = resp.choices[0].message
                if msg.function_call:
                    fname = msg.function_call.name
                    args = json.loads(msg.function_call.arguments)
                    try:
                        result = globals()[fname](**args)
                        st.markdown(f"**{fname} result:** {result}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.markdown(msg.content)
with col2:
    if st.button("Create Chart"):
        if not question:
            st.warning("Enter a question first.")
        else:
            with st.spinner("Charting..."):
                resp = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an analytic assistant. Use function calling for chart data."},
                        {"role": "user",   "content": question}
                    ],
                    functions=functions,
                    function_call="auto"
                )
                msg = resp.choices[0].message
                if msg.function_call:
                    fname = msg.function_call.name
                    args = json.loads(msg.function_call.arguments)
                    try:
                        data = globals()[fname](**args)
                        series = pd.Series(data)
                        fig, ax = plt.subplots(figsize=(8,4))
                        series.plot(kind='bar', ax=ax)
                        ax.set_title(f"{fname} on {canonical_col(args.get('column'))}")
                        plt.xticks(rotation=30, ha='right')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.write(msg.content)
