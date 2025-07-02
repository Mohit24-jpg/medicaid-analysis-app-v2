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
# The key fix here is using `flex-direction: column` and having the display loop
# handle the ordering of messages.
st.markdown("""
    <style>
    .chat-box-container {
        height: 550px; /* Use height instead of max-height for a fixed size */
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
        align-self: flex-end;
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
        align-self: flex-start;
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
        {"role": "system", "content": "You are a helpful Medicaid data analyst assistant. You will be asked to retrieve data and visualize it. When asked to create a chart, you must call a function to get the data first, then create the visualization."}
    ]

# --- Core Functions ---
def resolve_column(col_name: str) -> str:
    col_name = col_name.lower().strip()
    return SMART_COLUMN_MAP.get(col_name, get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6)[0] if get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6) else col_name)

# --- Data Analysis Functions for OpenAI ---
def count_unique(column: str) -> int:
    return int(df[resolve_column(column)].nunique())

def sum_column(column: str) -> float:
    return float(df[resolve_column(column)].sum())

def top_n(column: str, n: int) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().nlargest(n).to_dict()

def bottom_n(column: str, n: int) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().nsmallest(n).to_dict()

functions = [
    {"name": "count_unique", "description": "Count unique values in a column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "sum_column", "description": "Sum values in a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "top_n", "description": "Get top N products by a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "bottom_n", "description": "Get bottom N products by a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
]

# --- Versatile Charting Function ---
def create_chart(data: dict, user_prompt: str, column_name: str) -> go.Figure:
    chart_df = pd.DataFrame.from_dict(data, orient='index', columns=["Value"]).reset_index()
    chart_df.columns = ["Entity", "Value"]
    
    prompt_lower = user_prompt.lower()
    chart_types = ["pie", "line", "scatter", "area", "funnel", "treemap"]
    chart_type = next((t for t in chart_types if t in prompt_lower), "bar")

    fig_map = {
        "pie": px.pie(chart_df, names='Entity', values='Value'),
        "line": px.line(chart_df, x='Entity', y='Value', markers=True),
        "scatter": px.scatter(chart_df, x='Entity', y='Value', size='Value', hover_name='Entity'),
        "area": px.area(chart_df, x='Entity', y='Value', markers=True),
        "funnel": px.funnel(chart_df, x='Value', y='Entity'),
        "treemap": px.treemap(chart_df, path=['Entity'], values='Value'),
        "bar": px.bar(chart_df, x='Entity', y='Value', text='Value')
    }
    fig = fig_map[chart_type]
    
    color_match = re.search(r'\b(red|green|blue|purple|orange|yellow|pink|black)\b', prompt_lower)
    if color_match:
        fig.update_traces(marker_color=color_match.group(1))

    fig.update_layout(
        title_text=user_prompt.strip().capitalize(),
        title_font_size=22,
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        xaxis_title=chart_df.columns[0],
        yaxis_title=column_name.replace("_", " ").title(),
        showlegend=(chart_type == "pie")
    )
    return fig

# --- UI Layout ---
st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10), use_container_width=True)

user_input = st.chat_input("Ask: 'Show a pie chart for the top 5 drugs by prescriptions'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ** CONTEXT FIX **: Prepare messages for API, ensuring chart context is preserved.
    messages_for_gpt = []
    for msg in st.session_state.chat_history:
        if msg["role"] in ["system", "user"]:
            messages_for_gpt.append(msg)
        elif msg["role"] == "assistant" and isinstance(msg.get("content"), str):
            # For assistant messages, only send the text content for context
            messages_for_gpt.append({"role": "assistant", "content": msg["content"]})

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
                        # Check context from history for chart keywords
                        chart_keywords = ["chart", "visual", "bar", "graph", "pie", "line", "plot", "area", "funnel", "treemap"]
                        chart_request_in_context = any(word in user_input.lower() for word in chart_keywords)
                        
                        if not chart_request_in_context:
                            # Look at previous user message if current one isn't a chart request
                            if len(st.session_state.chat_history) > 1:
                                prev_user_msg = next((m['content'] for m in reversed(st.session_state.chat_history[:-1]) if m['role'] == 'user'), None)
                                if prev_user_msg and any(word in prev_user_msg.lower() for word in chart_keywords):
                                    chart_request_in_context = True
                                    user_input = prev_user_msg # Use the original chart prompt for title/type

                        if chart_request_in_context:
                            fig = create_chart(result, user_input, args.get("column", "Value"))
                            # ** CONTEXT FIX **: Store a text summary for the AI and the figure for Streamlit
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": f"Generated a '{fig.layout.title.text}' chart.", # Context for the AI
                                "figure": fig # Object for Streamlit to render
                            })
                        else:
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
    
    st.rerun()

# --- Display Chat History ---
st.subheader("💬 Chat Interface")
chat_container = st.container()

with chat_container:
    # ** MESSAGE FLOW FIX **: Iterate through messages normally. CSS handles the visual order.
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            continue
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            # ** DISPLAY FIX **: Check for a figure to render, otherwise show text.
            if "figure" in msg and msg["figure"] is not None:
                st.plotly_chart(msg["figure"], use_container_width=True)
            elif isinstance(msg.get("content"), str):
                st.markdown(f'<div class="assistant-msg">{msg["content"]}</div>', unsafe_allow_html=True)

# A small script to auto-scroll to the bottom of the chat container
st.components.v1.html("""
    <script>
        const chatContainer = window.parent.document.querySelector('.st-emotion-cache-1f1G2gn');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
""", height=0)

st.markdown('<div class="credit">Created by Mohit Vaid</div>', unsafe_allow_html=True)