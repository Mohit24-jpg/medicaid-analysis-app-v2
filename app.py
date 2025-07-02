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

# Custom CSS for the chat bubbles and container.
st.markdown("""
    <style>
    .chat-box-container {
        height: 550px;
        overflow-y: scroll;
        padding: 1.5rem;
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
    }
    .user-msg, .assistant-msg {
        padding: 12px;
        border-radius: 18px;
        margin: 10px 0;
        font-size: 1.05rem;
        width: fit-content;
        max-width: 80%;
    }
    .user-msg {
        background-color: #007bff;
        color: white;
        border-bottom-right-radius: 0;
        align-self: flex-end;
    }
    .assistant-msg {
        background-color: #f1f1f1;
        color: black;
        border-bottom-left-radius: 0;
        align-self: flex-start;
    }
    .credit { margin-top: 30px; font-size: 0.9rem; color: #888; text-align: center; }
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
    for col in ["units_reimbursed", "number_of_prescriptions", "total_amount_reimbursed", "medicaid_amount_reimbursed", "non_medicaid_amount_reimbursed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
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
        {"role": "system", "content": "You are a helpful Medicaid data analyst assistant. When asked to create a chart, you must call a function to get the data first. If the user asks to modify an existing chart (e.g., 'change to a pie chart'), do not call a function, just respond with the modification request."}
    ]

# --- Core Data & Charting Functions ---
def resolve_column(col_name: str) -> str:
    col_name = col_name.lower().strip()
    return SMART_COLUMN_MAP.get(col_name, get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6)[0] if get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6) else col_name)

def top_n(column: str, n: int) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().nlargest(n).to_dict()

def bottom_n(column: str, n: int) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().nsmallest(n).to_dict()

functions = [
    {"name": "top_n", "description": "Get top N products by a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "bottom_n", "description": "Get bottom N products by a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
]

def create_chart(data: dict, user_prompt: str, column_name: str) -> go.Figure:
    chart_df = pd.DataFrame.from_dict(data, orient='index', columns=["Value"]).reset_index()
    chart_df.columns = ["Entity", "Value"]
    
    prompt_lower = user_prompt.lower()
    chart_types = ["pie", "line", "scatter", "area", "funnel", "treemap"]
    chart_type = next((t for t in chart_types if t in prompt_lower), "bar")

    fig_map = {
        "pie": px.pie(chart_df, names='Entity', values='Value'),
        "line": px.line(chart_df, x='Entity', y='Value', markers=True),
        "bar": px.bar(chart_df, x='Entity', y='Value', text_auto='.2s')
    }
    fig = fig_map.get(chart_type, px.bar(chart_df, x='Entity', y='Value', text_auto='.2s'))
    
    # --- ✨ FORMATTING FIX & CONTEXT FIX ---
    # Apply number formatting and hover templates
    if chart_type == 'pie':
        fig.update_traces(textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>')
    else:
        fig.update_traces(texttemplate='$%{y:,.2s}', hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>')

    color_match = re.search(r'\b(red|green|blue|purple|orange|yellow|pink|black)\b', prompt_lower)
    if color_match:
        fig.update_traces(marker_color=color_match.group(1))

    fig.update_layout(
        title_text=user_prompt.strip().capitalize(),
        title_font_size=22,
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        xaxis_title="Entity",
        yaxis_title=column_name.replace("_", " ").title(),
        showlegend=(chart_type == "pie")
    )
    return fig

# --- UI Layout & Main Logic ---
st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10), use_container_width=True)

user_input = st.chat_input("Ask: 'Top 5 drugs by spending as a pie chart'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Prepare messages for API, ensuring full context is passed
    messages_for_gpt = [msg for msg in st.session_state.chat_history if msg["role"] != "assistant" or isinstance(msg.get("content"), str)]

    with st.spinner("Analyzing..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o", messages=messages_for_gpt, functions=functions, function_call="auto"
            )
            msg = response.choices[0].message

            # Case 1: AI wants to call a function to get new data
            if msg.function_call:
                fname = msg.function_call.name
                args = json.loads(msg.function_call.arguments)
                result = globals()[fname](**args)
                fig = create_chart(result, user_input, args.get("column", "Value"))
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Generated a chart for '{user_input}'.", # Context for AI
                    "figure": fig, "chart_data": result, "chart_args": args # Data for chart modification
                })

            # Case 2: AI response is text. Check if it's a chart modification request.
            else:
                chart_keywords = ["chart", "visual", "bar", "graph", "pie", "line", "plot", "area", "funnel", "treemap", "convert", "change"]
                is_chart_mod_request = any(word in user_input.lower() for word in chart_keywords)
                
                last_chart_data = None
                for hist_msg in reversed(st.session_state.chat_history):
                    if hist_msg["role"] == "assistant" and "chart_data" in hist_msg:
                        last_chart_data = hist_msg
                        break

                if is_chart_mod_request and last_chart_data:
                    # Found a chart to modify. Use OLD data with NEW prompt.
                    result = last_chart_data["chart_data"]
                    args = last_chart_data["chart_args"]
                    fig = create_chart(result, user_input, args.get("column", "Value"))
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Modified the chart based on '{user_input}'.",
                        "figure": fig, "chart_data": result, "chart_args": args
                    })
                else:
                    # It's just a regular text response
                    st.session_state.chat_history.append({"role": "assistant", "content": msg.content})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {e}"})
    
    st.rerun()

# --- Display Chat History ---
st.subheader("💬 Chat Interface")
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "system": continue
        
        div_class = "user-msg" if msg["role"] == "user" else "assistant-msg"
        
        if "figure" in msg:
            st.plotly_chart(msg["figure"], use_container_width=True)
        elif isinstance(msg.get("content"), str):
             # For user messages and text-only assistant messages
            st.markdown(f'<div class="{div_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# Auto-scroll script
st.components.v1.html("""
    <script>
        const chatContainer = window.parent.document.querySelector('.st-emotion-cache-1f1G2gn');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
""", height=0)

st.markdown('<div class="credit">Created by Mohit Vaid</div>', unsafe_allow_html=True)