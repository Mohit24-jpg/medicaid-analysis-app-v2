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

# --- CSS STYLING (FIXED) ---
# Styles for message bubbles. Alignment is now handled by the display logic.
st.markdown("""
    <style>
    .message-wrapper {
        display: flex;
        width: 100%;
        margin: 5px 0;
    }
    .user-wrapper {
        justify-content: flex-end;
    }
    .assistant-wrapper {
        justify-content: flex-start;
    }
    .user-msg, .assistant-msg {
        padding: 12px;
        border-radius: 18px;
        font-size: 1.05rem;
        width: fit-content;
        max-width: 75%;
    }
    .user-msg {
        background-color: #007bff;
        color: white;
        border-bottom-right-radius: 0;
    }
    .assistant-msg {
        background-color: #f1f1f1;
        color: black;
        border-bottom-left-radius: 0;
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
    st.session_state.chat_history = []

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
    
    if chart_type == 'pie':
        fig.update_traces(textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>(%{percent})<extra></extra>')
    else:
        fig.update_traces(texttemplate='$%{y:,.2s}', hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>')

    color_match = re.search(r'\b(red|green|blue|purple|orange|yellow|pink|black)\b', prompt_lower)
    if color_match:
        fig.update_traces(marker_color=color_match.group(1))

    fig.update_layout(
        title_text=user_prompt.strip().capitalize(), title_font_size=22,
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        xaxis_title="Entity", yaxis_title=column_name.replace("_", " ").title(),
        showlegend=(chart_type == "pie")
    )
    return fig

# --- UI Layout & Main Logic ---
st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10), use_container_width=True)

user_input = st.chat_input("Ask: 'Top 5 drugs by spending'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    messages_for_gpt = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_history]

    with st.spinner("Analyzing..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o", messages=messages_for_gpt, functions=functions, function_call="auto"
            )
            msg = response.choices[0].message

            if msg.function_call:
                fname = msg.function_call.name
                args = json.loads(msg.function_call.arguments)
                result = globals()[fname](**args)
                
                formatted_text = f"Here are the {args.get('n')} results for {args.get('column')}:\n\n"
                formatted_text += "\n".join([f"- {k.strip()}: ${v:,.2f}" for k, v in result.items()])
                
                st.session_state.chat_history.append({
                    "role": "assistant", "content": formatted_text,
                    "chart_data": result, "chart_args": args
                })
            else:
                # --- LOGIC FIX: Expanded keywords to better detect chart modification requests ---
                chart_keywords = [
                    "chart", "visual", "bar", "graph", "pie", "line", "plot", 
                    "convert", "change", "color", "blue", "red", "green", 
                    "purple", "orange", "yellow", "pink", "black"
                ]
                is_chart_mod_request = any(word in user_input.lower() for word in chart_keywords)
                
                last_assistant_msg_with_data = next((m for m in reversed(st.session_state.chat_history[:-1]) if m["role"] == "assistant" and "chart_data" in m), None)

                if is_chart_mod_request and last_assistant_msg_with_data:
                    result = last_assistant_msg_with_data["chart_data"]
                    args = last_assistant_msg_with_data["chart_args"]
                    fig = create_chart(result, user_input, args.get("column", "Value"))
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": f"Here is the chart for '{user_input}'.", "figure": fig
                    })
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": msg.content})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {e}"})
    
    st.rerun()

# --- DISPLAY FIX: Reworked display logic for proper alignment and scrolling ---
st.subheader("💬 Chat Interface")
# Use st.container with a fixed height to create a reliable scrollable area.
chat_container = st.container(height=600)

with chat_container:
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'''
                <div class="message-wrapper user-wrapper">
                    <div class="user-msg">{msg["content"]}</div>
                </div>
            ''', unsafe_allow_html=True)
        
        elif msg["role"] == "assistant":
            # Display assistant's text content in a bubble
            if msg.get("content"):
                st.markdown(f'''
                    <div class="message-wrapper assistant-wrapper">
                        <div class="assistant-msg">{msg["content"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            
            # Display the figure directly into the container (will be left-aligned)
            if msg.get("figure"):
                st.plotly_chart(msg["figure"], use_container_width=True)

st.markdown('<div class="credit">Created by Mohit Vaid</div>', unsafe_allow_html=True)
