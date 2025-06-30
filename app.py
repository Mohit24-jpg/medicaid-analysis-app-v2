import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import matplotlib.pyplot as plt
import json
from difflib import get_close_matches

# --- OpenAI Client Initialization ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- UI Header and Data Preview ---
st.image(
    "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/cd6be561d335a58ec5ca855ba3065a9e05eadfac/assets/logo.png",
    width=150
)
st.title("💊 Medicaid Drug Spending NLP Analytics")
st.markdown("#### Ask questions about drug spending, reimbursement, and utilization.")

CSV_URL = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"
df = pd.read_csv(CSV_URL)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
for col in ['units_reimbursed', 'number_of_prescriptions', 'total_amount_reimbursed', 'medicaid_amount_reimbursed', 'non_medicaid_amount_reimbursed']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
COLUMN_LIST = df.columns.tolist()
st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10))

# --- Session Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a Medicaid data analyst assistant. Use function calls where needed to return correct results."}
    ]
if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

# --- Smart Column Mapping ---
SMART_COLUMN_MAP = {
    "spending": "total_amount_reimbursed",
    "cost": "total_amount_reimbursed",
    "reimbursement": "medicaid_amount_reimbursed",
    "prescriptions": "number_of_prescriptions",
    "units": "units_reimbursed"
}

def resolve_column(col_name: str) -> str:
    col_name = col_name.lower().strip()
    if col_name in SMART_COLUMN_MAP:
        return SMART_COLUMN_MAP[col_name]
    matches = get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6)
    return matches[0] if matches else col_name

# --- Drug Name Normalization + Functions ---
def count_unique(column: str) -> int:
    column = resolve_column(column)
    return int(df[column].nunique())

def sum_column(column: str) -> float:
    column = resolve_column(column)
    return float(df[column].sum())

def top_n(column: str, n: int) -> dict:
    column = resolve_column(column)
    df_copy = df.copy()
    df_copy["product_name"] = df_copy["product_name"].astype(str)
    unique_names = df["product_name"].unique().tolist()
    name_map = {}
    for name in unique_names:
        match = get_close_matches(name, unique_names, n=1, cutoff=0.85)
        name_map[name] = match[0] if match else name
    df_copy["product_name"] = df_copy["product_name"].map(name_map)
    return df_copy.groupby("product_name")[column].sum().sort_values(ascending=False).head(n).to_dict()

def bottom_n(column: str, n: int) -> dict:
    column = resolve_column(column)
    return df.groupby("product_name")[column].sum().sort_values(ascending=True).head(n).to_dict()

def sum_by_product(column: str) -> dict:
    column = resolve_column(column)
    return df.groupby("product_name")[column].sum().sort_values(ascending=False).to_dict()

def average_by_product(column: str) -> dict:
    column = resolve_column(column)
    return df.groupby("product_name")[column].mean().sort_values(ascending=False).to_dict()

# --- Function Definitions for OpenAI ---
functions = [
    {"name": "count_unique", "description": "Count unique values in a column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "sum_column", "description": "Sum values in a numeric column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "top_n", "description": "Get top N products by a numeric column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "bottom_n", "description": "Get bottom N products by a numeric column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "sum_by_product", "description": "Sum a numeric column for each product", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "average_by_product", "description": "Calculate average of a numeric column for each product", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}}
]

# --- Chat Interface ---
st.subheader("💬 Chat Interface")
display_mode = st.radio("Select display mode:", ["Both", "Chart only", "Text only"], horizontal=True)

for msg in st.session_state.chat_history:
    if isinstance(msg, dict) and "role" in msg and "content" in msg:
        st.chat_message(msg["role"]).markdown(msg["content"])

user_input = st.chat_input("Ask a question like 'Top 5 drugs by spending' or follow up with context")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    with st.spinner("Analyzing..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.chat_history,
            functions=functions,
            function_call="auto"
        )
        msg = response.choices[0].message
        st.session_state.chat_history.append(msg)

        if hasattr(msg, "function_call") and msg.function_call:
            fname = msg.function_call.name
            args = json.loads(msg.function_call.arguments)
            try:
                result = globals()[fname](**args)
                if isinstance(result, dict):
                    if display_mode in ["Both", "Chart only"]:
                        series = pd.Series(result)
                        if len(series) > 1:
                            fig, ax = plt.subplots(figsize=(8, 4))
                            series.plot(kind='bar', ax=ax)
                            ax.set_title(f"{fname} on {args.get('column', '')}")
                            plt.xticks(rotation=30, ha='right')
                            st.pyplot(fig)

                    if display_mode in ["Both", "Text only"]:
                        output_lines = []
                        for k, v in result.items():
                            if isinstance(v, (int, float)) and v > 1000:
                                output_lines.append(f"{k.strip()}: ${v:,.2f}")
                            else:
                                output_lines.append(f"{k.strip()}: {v}")
                        st.text("\n".join(output_lines))
                else:
                    st.write(result)
                st.session_state.conversation_log.append({
                    "question": user_input,
                    "function": fname,
                    "args": args,
                    "result": result
                })
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            if msg.content:
                st.chat_message("assistant").markdown(msg.content)
                st.session_state.conversation_log.append({
                    "question": user_input,
                    "answer": msg.content
                })
