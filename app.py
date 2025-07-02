import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import plotly.graph_objects as go
import json
from difflib import get_close_matches
import re

# --- Page and Style Configuration ---
st.set_page_config(page_title="Medicaid Drug Spending NLP Analytics", layout="wide")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Custom CSS for the chat bubbles and container
st.markdown("""
    <style>
    .chat-box-container {
        max-height: 550px;
        overflow-y: scroll;
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #ccc;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column-reverse; /* To keep chat at the bottom */
    }
    .user-msg {
        background-color: #007bff;
        color: white;
        padding: 12px;
        border-radius: 18px 18px 0 18px;
        margin: 10px 0;
        text-align: right;
        font-size: 1.05rem;
        width: fit-content;
        margin-left: auto;
    }
    .assistant-msg {
        background-color: #f1f1f1;
        color: black;
        padding: 12px;
        border-radius: 18px 18px 18px 0;
        margin: 10px 0;
        text-align: left;
        font-size: 1.05rem;
        width: fit-content;
        margin-right: auto;
    }
    .credit {
        margin-top: 30px;
        font-size: 0.9rem;
        color: #888;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/cd6be561d335a58ec5ca855ba3065a9e05eadfac/assets/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")
st.markdown("#### Ask questions about drug spending, reimbursement, and utilization.")

# --- Data Loading and Preparation ---
@st.cache_data(show_spinner="Loading and preparing data...")
def load_data():
    csv_url = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"
    df = pd.read_csv(csv_url)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # Clean numeric columns
    for col in ["units_reimbursed", "number_of_prescriptions", "total_amount_reimbursed", "medicaid_amount_reimbursed", "non_medicaid_amount_reimbursed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # Normalize product names
    unique_names = df["product_name"].astype(str).unique().tolist()
    name_map = {name: get_close_matches(name, unique_names, n=1, cutoff=0.85)[0] if get_close_matches(name, unique_names, n=1, cutoff=0.85) else name for name in unique_names}
    df["product_name"] = df["product_name"].map(name_map)
    
    return df

df = load_data()
COLUMN_LIST = df.columns.tolist()

SMART_COLUMN_MAP = {
    "spending": "total_amount_reimbursed", "cost": "total_amount_reimbursed",
    "reimbursement": "medicaid_amount_reimbursed", "prescriptions": "number_of_prescriptions",
    "prescription_count": "number_of_prescriptions", "script_count": "number_of_prescriptions",
    "total_reimbursement": "total_amount_reimbursed", "units": "units_reimbursed"
}

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": "You are a Medicaid data analyst assistant. Use function calls and generate charts when asked."}
    ]

# --- Core Functions ---
def resolve_column(col_name: str) -> str:
    col_name = col_name.lower().strip()
    if col_name in SMART_COLUMN_MAP:
        return SMART_COLUMN_MAP[col_name]
    matches = get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6)
    return matches[0] if matches else col_name

# --- Data Analysis Functions for OpenAI ---
def count_unique(column: str) -> int:
    return int(df[resolve_column(column)].nunique())

def sum_column(column: str) -> float:
    return float(df[resolve_column(column)].sum())

def top_n(column: str, n: int) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().nlargest(n).to_dict()

def bottom_n(column: str, n: int) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().nsmallest(n).to_dict()

def sum_by_product(column: str) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().sort_values(ascending=False).to_dict()

def average_by_product(column: str) -> dict:
    return df.groupby("product_name")[resolve_column(column)].mean().sort_values(ascending=False).to_dict()

functions = [
    {"name": "count_unique", "description": "Count unique values in a column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "sum_column", "description": "Sum values in a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "top_n", "description": "Get top N products by a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "bottom_n", "description": "Get bottom N products by a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "sum_by_product", "description": "Sum a numeric column for each product.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "average_by_product", "description": "Calculate average of a numeric column for each product.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}}
]

# --- ✨ New Versatile Charting Function ✨ ---
def create_chart(data: dict, user_prompt: str, column_name: str) -> go.Figure:
    chart_df = pd.DataFrame.from_dict(data, orient='index', columns=["Value"])
    chart_df.reset_index(inplace=True)
    chart_df.columns = ["Entity", "Value"]
    
    prompt_lower = user_prompt.lower()
    
    # Detect chart type from prompt
    if "pie" in prompt_lower:
        fig = px.pie(chart_df, names='Entity', values='Value', title=user_prompt.strip().capitalize())
    elif "line" in prompt_lower:
        fig = px.line(chart_df, x='Entity', y='Value', title=user_prompt.strip().capitalize(), markers=True)
    elif "scatter" in prompt_lower:
        fig = px.scatter(chart_df, x='Entity', y='Value', title=user_prompt.strip().capitalize(), size='Value', hover_name='Entity')
    elif "area" in prompt_lower:
        fig = px.area(chart_df, x='Entity', y='Value', title=user_prompt.strip().capitalize(), markers=True)
    elif "funnel" in prompt_lower:
        fig = px.funnel(chart_df, x='Value', y='Entity', title=user_prompt.strip().capitalize())
    elif "treemap" in prompt_lower:
        fig = px.treemap(chart_df, path=['Entity'], values='Value', title=user_prompt.strip().capitalize())
    else: # Default to bar chart
        fig = px.bar(chart_df, x='Entity', y='Value', text='Value', title=user_prompt.strip().capitalize())
    
    # Detect color from prompt
    color_match = re.search(r'\b(red|green|blue|purple|orange|yellow|pink|black)\b', prompt_lower)
    if color_match:
        fig.update_traces(marker_color=color_match.group(1))

    # General styling
    fig.update_traces(texttemplate='$%{text:,.0f}', textposition='auto')
    fig.update_layout(
        title_font_size=22,
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        xaxis_title=chart_df.columns[0],
        yaxis_title=column_name.replace("_", " ").title(),
        showlegend=False
    )
    return fig

# --- UI Layout ---
st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10), use_container_width=True)

user_input = st.chat_input("Ask: 'Top 5 drugs by spending as a blue bar chart'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Prepare messages for OpenAI API
    messages_for_gpt = [msg for msg in st.session_state.chat_history if msg["role"] != "assistant" or isinstance(msg["content"], str)]

    with st.spinner("Analyzing..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages_for_gpt,
                functions=functions,
                function_call="auto",
                timeout=45
            )
            msg = response.choices[0].message

            if msg.function_call:
                fname = msg.function_call.name
                args = json.loads(msg.function_call.arguments)
                
                try:
                    result = globals()[fname](**args)
                    
                    if isinstance(result, dict):
                        # Check if a chart was requested
                        if any(word in user_input.lower() for word in ["chart", "visual", "bar", "graph", "pie", "line", "plot", "area", "funnel", "treemap"]):
                            fig = create_chart(result, user_input, args.get("column", "Value"))
                            st.session_state.chat_history.append({"role": "assistant", "content": fig})
                        else:
                            # Format as text if no chart is requested
                            formatted = "\n".join([f"{k.strip()}: ${v:,.2f}" if isinstance(v, (int, float)) and v > 1000 else f"{k.strip()}: {v}" for k, v in result.items()])
                            st.session_state.chat_history.append({"role": "assistant", "content": formatted})
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": str(result)})

                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Function error: {e}"})
            
            elif msg.content:
                st.session_state.chat_history.append({"role": "assistant", "content": msg.content})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {e}"})
    
    # Rerun to display the new message immediately
    st.rerun()

# --- Display Chat History (Restored Bubble Interface) ---
st.subheader("💬 Chat Interface")
st.markdown('<div class="chat-box-container">', unsafe_allow_html=True)
# Iterate in reverse to show latest messages at the bottom
for msg in reversed(st.session_state.chat_history):
    if msg["role"] == "system":
        continue # Don't display system messages
    if msg["role"] == "user":
        st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
    elif isinstance(msg["content"], go.Figure):
        st.plotly_chart(msg["content"], use_container_width=True)
    elif isinstance(msg["content"], str):
        st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="credit">Created by Mohit Vaid</div>', unsafe_allow_html=True)