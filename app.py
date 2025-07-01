import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import plotly.express as px
import json
from difflib import get_close_matches

st.set_page_config(page_title="Medicaid Drug Spending NLP Analytics", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.markdown("""
    <style>
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 0 1rem;
        border: 1px solid #ccc;
        border-radius: 10px;
        background-color: #f9f9f9;
    }
    .stChatInputContainer { margin-top: 1rem !important; }
    </style>
""", unsafe_allow_html=True)

st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/cd6be561d335a58ec5ca855ba3065a9e05eadfac/assets/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")
st.markdown("#### Ask questions about drug spending, reimbursement, and utilization.")

CSV_URL = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"
df = pd.read_csv(CSV_URL)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

SMART_COLUMN_MAP = {
    "spending": "total_amount_reimbursed",
    "cost": "total_amount_reimbursed",
    "reimbursement": "medicaid_amount_reimbursed",
    "prescriptions": "number_of_prescriptions",
    "prescription_count": "number_of_prescriptions",
    "units": "units_reimbursed"
}

for col in ['units_reimbursed', 'number_of_prescriptions', 'total_amount_reimbursed', 'medicaid_amount_reimbursed', 'non_medicaid_amount_reimbursed']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
COLUMN_LIST = df.columns.tolist()
st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10), use_container_width=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a Medicaid data analyst assistant. Use function calls where needed to return correct results."}
    ]

def resolve_column(col_name: str) -> str:
    col_name = col_name.lower().strip()
    if col_name in SMART_COLUMN_MAP:
        return SMART_COLUMN_MAP[col_name]
    matches = get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6)
    return matches[0] if matches else col_name

@st.cache_data(show_spinner=False)
def normalize_product_names():
    name_map = {}
    unique_names = df["product_name"].astype(str).unique().tolist()
    for name in unique_names:
        match = get_close_matches(name, unique_names, n=1, cutoff=0.85)
        name_map[name] = match[0] if match else name
    return name_map

NAME_MAP = normalize_product_names()

def count_unique(column: str) -> int:
    column = resolve_column(column)
    return int(df[column].nunique())

def sum_column(column: str) -> float:
    column = resolve_column(column)
    return float(df[column].sum())

def top_n(column: str, n: int) -> dict:
    column = resolve_column(column)
    df_copy = df.copy()
    df_copy["product_name"] = df_copy["product_name"].astype(str).map(NAME_MAP)
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

functions = [
    {"name": "count_unique", "description": "Count unique values in a column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "sum_column", "description": "Sum values in a numeric column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "top_n", "description": "Get top N products by a numeric column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "bottom_n", "description": "Get bottom N products by a numeric column", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "sum_by_product", "description": "Sum a numeric column for each product", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "average_by_product", "description": "Calculate average of a numeric column for each product", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}}
]

st.subheader("💬 Chat Interface")
user_input = st.chat_input("Ask a question like 'Top 5 drugs by spending'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("Analyzing..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=st.session_state.chat_history,
                functions=functions,
                function_call="auto",
                timeout=30
            )
            msg = response.choices[0].message

            if hasattr(msg, "function_call") and msg.function_call:
                fname = msg.function_call.name
                args = json.loads(msg.function_call.arguments)
                try:
                    result = globals()[fname](**args)
                    if isinstance(result, dict):
                        if any(word in user_input.lower() for word in ["chart", "visual", "bar"]):
                            chart_df = pd.DataFrame.from_dict(result, orient='index', columns=["Value"])
                            chart_df.reset_index(inplace=True)
                            chart_df.columns = ["Drug", "Value"]
                            fig = px.bar(chart_df, x="Drug", y="Value", text="Value", title=user_input.strip().capitalize())
                            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
                            fig.update_layout(xaxis_title="Drug", yaxis_title="Amount in USD")
                            st.session_state.chat_history.append({"role": "assistant", "content": "Here is the chart:"})
                            st.session_state.chat_history.append({"role": "chart", "content": fig})
                        else:
                            formatted = "\n".join([
                                f"{k.strip()}: ${v:,.2f}" if isinstance(v, (int, float)) and v > 1000 else f"{k.strip()}: {v}"
                                for k, v in result.items()
                            ])
                            st.session_state.chat_history.append({"role": "assistant", "content": formatted})
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": str(result)})
                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Function error: {e}"})
            elif msg.content:
                st.session_state.chat_history.append({"role": "assistant", "content": msg.content})
        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Chat request failed: {e}"})

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"] if msg["role"] in ["user", "assistant"] else "assistant"):
        if msg["role"] == "chart" and hasattr(msg["content"], 'show'):
            st.plotly_chart(msg["content"], use_container_width=True)
        else:
            st.markdown(msg["content"])
st.markdown('</div>', unsafe_allow_html=True)
