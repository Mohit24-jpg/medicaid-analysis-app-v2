import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import plotly.graph_objects as go
import json
from difflib import get_close_matches
import re
from io import StringIO

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
def load_data_and_definitions():
    # Load main dataset
    csv_url = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"
    df = pd.read_csv(csv_url)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    for col in ["units_reimbursed", "number_of_prescriptions", "total_amount_reimbursed", "medicaid_amount_reimbursed", "non_medicaid_amount_reimbursed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    unique_names = df["product_name"].astype(str).unique().tolist()
    name_map = {name: get_close_matches(name, unique_names, n=1, cutoff=0.85)[0] if get_close_matches(name, unique_names, n=1, cutoff=0.85) else name for name in unique_names}
    df["product_name"] = df["product_name"].map(name_map)
    
    # --- NEW: Load and parse the data dictionary ---
    data_dictionary_csv = """
    Variable Name,Label,Description,Data Type
    utilization_type,Utilization Type,"Indicates whether the state reported data is for 'FFS' (Fee-for-Service) or 'MCO' (Managed Care Organization).",string
    state,State,"The two-character abbreviation for the state in which the drug was dispensed.",string
    ndc,National Drug Code,"The 11-digit National Drug Code (NDC) of the drug that was dispensed.",string
    product_name,Product Name,"The proprietary name, if any, and the strength of the drug product.",string
    units_reimbursed,Units Reimbursed,"The number of units of the drug dispensed.",numeric
    number_of_prescriptions,Number of Prescriptions,"The number of prescriptions dispensed.",numeric
    total_amount_reimbursed,Total Amount Reimbursed,"The total amount reimbursed from all sources for the drug.",numeric
    medicaid_amount_reimbursed,Medicaid Amount Reimbursed,"The amount reimbursed by the Medicaid program for the drug.",numeric
    non_medicaid_amount_reimbursed,Non-Medicaid Amount Reimbursed,"The amount reimbursed from other sources for the drug.",numeric
    quarter,Quarter,"The calendar quarter and year of the report.",date
    """
    
    dd_df = pd.read_csv(StringIO(data_dictionary_csv))
    column_definitions = {row['Variable Name']: row['Description'] for index, row in dd_df.iterrows()}
    
    return df, column_definitions

df, COLUMN_DEFINITIONS = load_data_and_definitions()
COLUMN_LIST = df.columns.tolist()

SMART_COLUMN_MAP = {
    "spending": "total_amount_reimbursed", "cost": "total_amount_reimbursed",
    "reimbursement": "medicaid_amount_reimbursed", "prescriptions": "number_of_prescriptions",
    "prescription_count": "number_of_prescriptions", "script_count": "number_of_prescriptions",
    "total_reimbursement": "total_amount_reimbursed", "units": "units_reimbursed"
}

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    # --- NEW: Create an enhanced system prompt using the data dictionary ---
    intro = "You are a helpful Medicaid data analyst assistant. You have access to a dataset with the following columns:\n\n"
    definitions_text = "\n".join([f"- `{col}`: {desc}" for col, desc in COLUMN_DEFINITIONS.items()])
    outro = "\n\nUse your functions to answer questions about this data. When asked for a specific drug, use the `get_product_stat` function. For calculations like 'cost per unit', use the `get_calculated_stat` function."
    system_prompt = intro + definitions_text + outro
    st.session_state.chat_history = [{"role": "system", "content": system_prompt}]

# --- Core Data & Charting Functions ---
def resolve_column(col_name: str) -> str:
    col_name = col_name.lower().strip().replace(" ", "_")
    return SMART_COLUMN_MAP.get(col_name, get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6)[0] if get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6) else col_name)

def top_n(column: str, n: int) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().nlargest(n).to_dict()

def bottom_n(column: str, n: int) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().nsmallest(n).to_dict()

def get_product_stat(product_name: str, column: str, stat: str) -> str:
    product_names = df['product_name'].unique()
    match = get_close_matches(product_name.upper(), product_names, n=1, cutoff=0.8)
    
    if not match:
        return f"I could not find data for a product matching '{product_name}'."
        
    actual_product_name = match[0]
    resolved_column = resolve_column(column)
    product_df = df[df['product_name'] == actual_product_name]
    
    if product_df.empty:
         return f"I found the product '{actual_product_name}' but it has no data for the column '{resolved_column}'."

    if stat in ['average', 'mean', 'avg']:
        value = product_df[resolved_column].mean()
        stat_name = "Average"
    elif stat in ['total', 'sum', 'total amount']:
        value = product_df[resolved_column].sum()
        stat_name = "Total"
    elif stat in ['count', 'number']:
        value = product_df[resolved_column].count()
        return f"Based on the data, there are {int(value):,} entries for {actual_product_name}."
    else:
        return f"Sorry, I can't calculate the '{stat}' for a product."

    # --- FIX: Make response more descriptive ---
    if "amount" in resolved_column or "spending" in column:
        return f"Based on the data, the {stat_name.lower()} {column} for {actual_product_name} is ${value:,.2f}."
    else:
        return f"Based on the data, the {stat_name.lower()} {column} for {actual_product_name} is {value:,.2f}."

# --- NEW FUNCTION for calculated stats like 'cost per prescription' ---
def get_calculated_stat(product_name: str, numerator: str, denominator: str) -> str:
    product_names = df['product_name'].unique()
    match = get_close_matches(product_name.upper(), product_names, n=1, cutoff=0.8)
    
    if not match:
        return f"I could not find data for a product matching '{product_name}'."
        
    actual_product_name = match[0]
    num_col = resolve_column(numerator)
    den_col = resolve_column(denominator)
    product_df = df[df['product_name'] == actual_product_name]

    if product_df.empty:
        return f"Could not find data for {actual_product_name}."

    num_sum = product_df[num_col].sum()
    den_sum = product_df[den_col].sum()

    if den_sum == 0:
        return f"Cannot calculate as the denominator ({denominator}) is zero for {actual_product_name}."

    value = num_sum / den_sum
    
    # --- FIX: Make response more descriptive ---
    return (f"The calculated average {numerator} per {denominator} for {actual_product_name} is ${value:,.2f}. "
            f"(Calculated as Total {num_col.replace('_', ' ').title()} / Total {den_col.replace('_', ' ').title()})")


functions = [
    {"name": "top_n", "description": "Get top N products by a numeric column. Use for questions like 'What are the top 5 drugs?'", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "bottom_n", "description": "Get bottom N products by a numeric column. Use for questions like 'What are the bottom 5 drugs?'", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "get_product_stat", "description": "Get a specific statistic (like average, total, or count) for a single named product. Use for questions about one specific drug.", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "column": {"type": "string"}, "stat": {"type": "string"}}, "required": ["product_name", "column", "stat"]}},
    {"name": "get_calculated_stat", "description": "Calculate a 'per unit' statistic for a single product, like 'cost per prescription'. Use when a question involves division or a rate.", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "numerator": {"type": "string"}, "denominator": {"type": "string"}}, "required": ["product_name", "numerator", "denominator"]}}
]

def create_chart(data: dict, user_prompt: str, chart_args: dict, prev_chart_type: str = 'bar', prev_title: str = None) -> tuple[go.Figure, str]:
    chart_df = pd.DataFrame.from_dict(data, orient='index', columns=["Value"]).reset_index()
    chart_df.columns = ["Entity", "Value"]
    
    prompt_lower = user_prompt.lower()
    
    chart_types = ["pie", "donut", "line", "scatter", "area", "funnel", "treemap"]
    chart_type = next((t for t in chart_types if t in prompt_lower), prev_chart_type)

    color_palette = None
    if 'professional' in prompt_lower or 'corporate' in prompt_lower:
        color_palette = px.colors.qualitative.G10
    elif 'vibrant' in prompt_lower or 'colorful' in prompt_lower:
        color_palette = px.colors.qualitative.Vivid
    elif 'pastel' in prompt_lower:
        color_palette = px.colors.qualitative.Pastel

    is_donut = 'donut' in prompt_lower or (prev_chart_type == 'donut' and 'pie' not in prompt_lower)
    
    # --- COLOR FIX: Ensure color palette is passed to pie charts ---
    fig_map = {
        "pie": px.pie(chart_df, names='Entity', values='Value', color_discrete_sequence=color_palette, hole=0.4 if is_donut else 0),
        "donut": px.pie(chart_df, names='Entity', values='Value', color_discrete_sequence=color_palette, hole=0.4),
        "line": px.line(chart_df, x='Entity', y='Value', markers=True, color_discrete_sequence=color_palette),
        "bar": px.bar(chart_df, x='Entity', y='Value', text_auto='.2s', color='Entity', color_discrete_sequence=color_palette)
    }
    fig = fig_map.get(chart_type, px.bar(chart_df, x='Entity', y='Value', text_auto='.2s', color='Entity', color_discrete_sequence=color_palette))
    
    color_match = re.search(r'\b(red|green|blue|purple|orange|yellow|pink|black)\b', prompt_lower)
    if color_match:
        fig.update_traces(marker_color=color_match.group(1))

    if chart_type in ['pie', 'donut']:
        fig.update_traces(textinfo='percent+label', hovertemplate='<b>%{label}</b><br>Value: $%{value:,.2f}<br>(%{percent})<extra></extra>')
    else:
        fig.update_traces(texttemplate='$%{y:,.2s}', hovertemplate='<b>%{x}</b><br>Value: $%{y:,.2f}<extra></extra>')

    is_simple_command = len(user_prompt.split()) < 6 and any(w in prompt_lower for w in chart_types + ['color', 'professional', 'remove', 'add'])
    if is_simple_command and prev_title:
        title = prev_title
    else:
        func_name = chart_args.get("func_name", "top_n")
        n = chart_args.get("n", len(data))
        column_name = chart_args.get("column", "value").replace('_', ' ').title()
        direction = "Top" if func_name == "top_n" else "Bottom"
        title = f"{direction} {n} Products by {column_name}"

    fig.update_layout(
        title_text=title, title_font_size=22,
        font=dict(family="Arial, sans-serif", size=14, color="black"),
        xaxis_title="Entity", yaxis_title=chart_args.get("column", "value").replace('_', ' ').title(),
        showlegend=(chart_type in ["pie", "donut"]) or (color_palette is not None and chart_type == 'bar')
    )
    final_chart_type = 'donut' if is_donut else chart_type
    return fig, final_chart_type

# --- UI Layout & Main Logic ---
st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10), use_container_width=True)

user_input = st.chat_input("Ask: 'Top 5 drugs by spending'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    messages_for_gpt = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_history if msg.get("content") is not None]

    with st.spinner("Analyzing..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o", messages=messages_for_gpt, functions=functions, function_call="auto"
            )
            msg = response.choices[0].message

            if msg.function_call:
                fname = msg.function_call.name
                args = json.loads(msg.function_call.arguments)
                
                if fname == "get_product_stat":
                    result_string = get_product_stat(**args)
                    st.session_state.chat_history.append({"role": "assistant", "content": result_string})
                
                elif fname == "get_calculated_stat":
                    result_string = get_calculated_stat(**args)
                    st.session_state.chat_history.append({"role": "assistant", "content": result_string})

                elif fname in ["top_n", "bottom_n"]:
                    result = globals()[fname](**args)
                    args["func_name"] = fname
                    
                    formatted_text = f"Here are the {args.get('n', 'top')} results for {args.get('column')}:\n\n"
                    formatted_text += "\n".join([f"- {k.strip()}: ${v:,.2f}" for k, v in result.items()])
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": formatted_text,
                        "chart_data": result, "chart_args": args
                    })
            else:
                style_keywords = ["chart", "visual", "bar", "graph", "pie", "donut", "line", "plot", "convert", "change", "color", "professional", "vibrant", "pastel", "blue", "red", "green", "purple"]
                data_keywords = ["remove", "add", "without", "exclude", "include"]
                
                is_style_mod_request = any(word in user_input.lower() for word in style_keywords)
                is_data_mod_request = any(word in user_input.lower() for word in data_keywords)
                
                last_assistant_msg_with_data = next((m for m in reversed(st.session_state.chat_history[:-1]) if m.get("chart_data")), None)

                if (is_style_mod_request or is_data_mod_request) and last_assistant_msg_with_data:
                    current_data = last_assistant_msg_with_data["chart_data"].copy()
                    args = last_assistant_msg_with_data["chart_args"]
                    
                    if is_data_mod_request:
                        prompt_words = set(re.findall(r'\b\w+\b', user_input.lower()))
                        keys_to_remove = []
                        for key in current_data.keys():
                            if get_close_matches(key.lower(), prompt_words, n=1, cutoff=0.8):
                                keys_to_remove.append(key)
                        for key in keys_to_remove:
                            del current_data[key]
                    
                    last_fig_msg = next((m for m in reversed(st.session_state.chat_history) if m.get("figure")), None)
                    prev_title = last_fig_msg['figure'].layout.title.text if last_fig_msg else None
                    prev_chart_type = last_assistant_msg_with_data.get("chart_type", "bar")

                    fig, new_chart_type = create_chart(
                        data=current_data, user_prompt=user_input, chart_args=args,
                        prev_chart_type=prev_chart_type, prev_title=prev_title
                    )
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"[Chart updated based on user request: '{user_input}']",
                        "figure": fig,
                        "chart_data": current_data, "chart_args": args, "chart_type": new_chart_type
                    })
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": msg.content})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {e}"})
    
    st.rerun()

# --- DISPLAY LOGIC ---
st.subheader("💬 Chat Interface")
chat_container = st.container(height=600)

with chat_container:
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="message-wrapper user-wrapper"><div class="user-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)
        
        elif msg["role"] == "assistant":
            if msg.get("figure"):
                st.plotly_chart(msg["figure"], use_container_width=True, key=f"chart_{i}")
            elif msg.get("content"):
                if not msg["content"].startswith("[Chart generated") and not msg["content"].startswith("[Chart updated"):
                    st.markdown(f'<div class="message-wrapper assistant-wrapper"><div class="assistant-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="credit">Created by Mohit Vaid</div>', unsafe_allow_html=True)
