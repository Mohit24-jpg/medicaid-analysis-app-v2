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

# --- CSS Styling ---
st.markdown("""
    <style>
    /* Reduce top padding of the main app container */
    .main .block-container {
        padding-top: 0.2rem;
    }
    /* --- FIX: Center align the main title --- */
    h1 {
        text-align: center;
    }
    /* Center align the subtitle */
    h4 {
        text-align: center;
    }
    .message-wrapper { display: flex; width: 100%; margin: 5px 0; }
    .user-wrapper { justify-content: flex-end; }
    .assistant-wrapper { justify-content: flex-start; }
    .user-msg, .assistant-msg { padding: 12px; border-radius: 18px; font-size: 1.05rem; width: fit-content; max-width: 75%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .user-msg { background-color: #007bff; color: white; border-bottom-right-radius: 0; }
    .assistant-msg { background-color: #f1f1f1; color: black; border-bottom-left-radius: 0; }
    .credit { margin-top: 30px; font-size: 0.9rem; color: #555; text-align: center; }
    .styled-table { border-collapse: collapse; margin: 25px 0; font-size: 0.9em; font-family: sans-serif; min-width: 400px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.15); }
    .styled-table thead tr { background-color: #007bff; color: #ffffff; text-align: left; }
    .styled-table th, .styled-table td { padding: 12px 15px; }
    .styled-table tbody tr { border-bottom: 1px solid #dddddd; }
    .styled-table tbody tr:nth-of-type(even) { background-color: #f3f3f3; }
    .styled-table tbody tr:last-of-type { border-bottom: 2px solid #007bff; }
</style>
"""
, unsafe_allow_html=True)


# --- App Header ---
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
    
    data_dictionary_csv = """Variable Name,Label,Description,Data Type
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
    intro = "You are a helpful Medicaid data analyst assistant. You have access to a dataset with the following columns:\n\n"
    definitions_text = "\n".join([f"- `{col}`: {desc}" for col, desc in COLUMN_DEFINITIONS.items()])
    # --- FIX: Updated instructions to include get_unique_values ---
    outro = "\n\nUse your functions to answer questions. For charting, use `create_chart_figure`. For tables, use `create_table_html`. To get a list of unique values from a column (e.g., all states), use `get_unique_values`."
    system_prompt = intro + definitions_text + outro
    st.session_state.chat_history = [{"role": "system", "content": system_prompt}]
    st.session_state.initial_load = True

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
    if not match: return f"I could not find data for a product matching '{product_name}'."
    actual_product_name = match[0]
    resolved_column = resolve_column(column)
    product_df = df[df['product_name'] == actual_product_name]
    if product_df.empty: return f"I found the product '{actual_product_name}' but it has no data for the column '{resolved_column}'."
    if stat in ['average', 'mean', 'avg']: value, stat_name = product_df[resolved_column].mean(), "Average"
    elif stat in ['total', 'sum']: value, stat_name = product_df[resolved_column].sum(), "Total"
    elif stat in ['count', 'number']: return f"Based on the data, there are {int(product_df[resolved_column].count()):,} entries for {actual_product_name}."
    else: return f"Sorry, I can't calculate the '{stat}' for a product."
    if "amount" in resolved_column or "spending" in column: return f"Based on the data, the {stat_name.lower()} {column} for {actual_product_name} is ${value:,.2f}."
    else: return f"Based on the data, the {stat_name.lower()} {column} for {actual_product_name} is {value:,.2f}."

def get_calculated_stat(product_name: str, numerator: str, denominator: str) -> str:
    product_names = df['product_name'].unique()
    match = get_close_matches(product_name.upper(), product_names, n=1, cutoff=0.8)
    if not match: return f"I could not find data for a product matching '{product_name}'."
    actual_product_name = match[0]
    num_col, den_col = resolve_column(numerator), resolve_column(denominator)
    product_df = df[df['product_name'] == actual_product_name]
    if product_df.empty: return f"Could not find data for {actual_product_name}."
    num_sum, den_sum = product_df[num_col].sum(), product_df[den_col].sum()
    if den_sum == 0: return f"Cannot calculate as the denominator ({denominator}) is zero for {actual_product_name}."
    value = num_sum / den_sum
    return (f"The calculated average {numerator} per {denominator} for {actual_product_name} is ${value:,.2f}. "
            f"(Calculated as Total {num_col.replace('_', ' ').title()} / Total {den_col.replace('_', ' ').title()})")

def get_top_n_by_calculated_metric(numerator: str, denominator: str, n: int) -> dict:
    num_col, den_col = resolve_column(numerator), resolve_column(denominator)
    grouped = df.groupby('product_name').agg({ num_col: 'sum', den_col: 'sum' }).reset_index()
    ratio_col_name = f"{numerator}_per_{denominator}"
    grouped[ratio_col_name] = grouped.apply(lambda row: row[num_col] / row[den_col] if row[den_col] != 0 else 0, axis=1)
    top_n_df = grouped.nlargest(n, ratio_col_name)
    return pd.Series(top_n_df[ratio_col_name].values, index=top_n_df['product_name']).to_dict()

# --- NEW: Function to get unique values from a column ---
def get_unique_values(column: str) -> list:
    """Gets a list of unique values from a specified column."""
    resolved_column = resolve_column(column)
    if resolved_column not in df.columns:
        return [f"Error: Column '{resolved_column}' not found."]
    return df[resolved_column].unique().tolist()

def create_table_html(data: dict, chart_args: dict) -> str:
    """Creates a styled HTML table from the data."""
    table_df = pd.DataFrame.from_dict(data, orient='index', columns=["Value"]).reset_index()
    
    func_name = chart_args.get("func_name", "top_n")
    if func_name == 'get_top_n_by_calculated_metric':
        value_col_name = f"{chart_args.get('numerator', 'value')} Per {chart_args.get('denominator', 'value')}".replace('_', ' ').title()
    else:
        value_col_name = chart_args.get("column", "value").replace('_', ' ').title()
        
    table_df.columns = ["Product Name", value_col_name]
    
    table_df[value_col_name] = table_df[value_col_name].apply(lambda x: f"${x:,.2f}")
    
    return table_df.to_html(classes='styled-table', index=False)

def create_chart_figure(data: dict, customization_prompt: str, chart_args: dict) -> go.Figure:
    """
    Uses an LLM to generate Plotly Python code for a chart based on data, a prompt, and original context.
    """
    chart_df = pd.DataFrame.from_dict(data, orient='index', columns=["Value"]).reset_index()
    chart_df.columns = ["Entity", "Value"]

    func_name = chart_args.get("func_name", "top_n")
    if func_name == 'get_top_n_by_calculated_metric':
        y_axis_label = f"{chart_args.get('numerator', 'value')} Per {chart_args.get('denominator', 'value')}".replace('_', ' ').title()
    else:
        y_axis_label = chart_args.get("column", "value").replace('_', ' ').title()

    data_context = f"""
The data in the 'Value' column represents: '{y_axis_label}'.
The data in the 'Entity' column represents: 'Product Name'.
The data was generated by ranking the '{func_name}' based on the user's original request.
"""
    code_generation_prompt = f"""
You are an expert Python data visualization assistant who specializes in the Plotly library.
Your task is to write Python code to generate a chart based on a user's request and the provided data context.
You will be given a pandas DataFrame named 'chart_df'. The DataFrame has two columns: 'Entity' and 'Value'.
The user has selected the 'Light' theme for the app.

{data_context}

Your code MUST:
1. Be a single, executable block of Python.
2. Use the `plotly.graph_objects` library, imported as `go`, or `plotly.express` as `px`.
3. Create a figure object named `fig`.
4. Set the chart's template to 'plotly_white'.
5. Do NOT include any code to display the chart (e.g., `fig.show()`).
6. Do NOT include the DataFrame creation code; assume `chart_df` already exists.
7. Set a descriptive title and axis labels for the chart based on the user's request and the data context provided above.
8. IMPORTANT: When adding annotations or text labels to bars, format the numbers as currency (e.g., '$1.2M', '$456.7k', '$789.00').

User's customization request: '{customization_prompt}'
"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": code_generation_prompt}],
            temperature=0,
        )
        chart_code = response.choices[0].message.content.strip()
        
        if chart_code.startswith("```python"):
            chart_code = chart_code[9:]
        if chart_code.endswith("```"):
            chart_code = chart_code[:-3]

        local_vars = {"pd": pd, "px": px, "go": go, "chart_df": chart_df}
        exec(chart_code, {}, local_vars)
        
        return local_vars.get("fig", go.Figure())
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title_text=f"Chart Generation Error: {e}")
        return fig

# --- FIX: Added get_unique_values to the list of functions ---
functions = [
    {"name": "top_n", "description": "Get top N products by a single numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "bottom_n", "description": "Get bottom N products by a single numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}}, "required": ["column", "n"]}},
    {"name": "get_product_stat", "description": "Get a simple statistic (average, total, or count) for one specific, named product.", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "column": {"type": "string"}, "stat": {"type": "string"}}, "required": ["product_name", "column", "stat"]}},
    {"name": "get_calculated_stat", "description": "Calculate a ratio for a single named product (e.g., cost per prescription for Trulicity).", "parameters": {"type": "object", "properties": {"product_name": {"type": "string"}, "numerator": {"type": "string"}, "denominator": {"type": "string"}}, "required": ["product_name", "numerator", "denominator"]}},
    {"name": "get_top_n_by_calculated_metric", "description": "Ranks products by a calculated ratio and returns the top N.", "parameters": {"type": "object", "properties": {"numerator": {"type": "string"}, "denominator": {"type": "string"}, "n": {"type": "integer"}}, "required": ["numerator", "denominator", "n"]}},
    {"name": "get_unique_values", "description": "Get a list of unique values from a specified column in the dataset.", "parameters": {"type": "object", "properties": {"column": {"type": "string", "description": "The name of the column to get unique values from."}}, "required": ["column"]}}
]

# --- UI Layout & Main Logic ---
st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10), use_container_width=True)

# --- DISPLAY LOGIC ---
st.subheader("💬 Chat Interface")
chat_container = st.container(height=600)

with chat_container:
    for i, msg in enumerate(st.session_state.get("chat_history", [])):
        if msg["role"] == "user":
            st.markdown(f'<div class="message-wrapper user-wrapper"><div class="user-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)
        
        elif msg["role"] == "assistant":
            if msg.get("figure"):
                st.plotly_chart(msg["figure"], use_container_width=True, key=f"chart_{i}")
            elif msg.get("table_html"):
                st.markdown(msg["table_html"], unsafe_allow_html=True)
            elif msg.get("content"):
                if not msg["content"].startswith("[Chart generated") and not msg["content"].startswith("[Table generated"):
                    st.markdown(f'<div class="message-wrapper assistant-wrapper"><div class="assistant-msg">{msg["content"]}</div></div>', unsafe_allow_html=True)

user_input = st.chat_input("Ask: 'Top 5 drugs by spending'")

if user_input:
    st.session_state.initial_load = False
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    messages_for_gpt = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.chat_history if msg.get("content") is not None]

    with st.spinner("Analyzing..."):
        try:
            viz_keywords = ["chart", "graph", "plot", "visualize", "bar", "pie", "donut", "line", "background", "color", "axis", "labels", "call out", "table"]
            question_words = ["what", "which", "how", "why", "recommend", "is there", "can you explain"]
            
            prompt_lower = user_input.lower().strip()
            is_question = any(prompt_lower.startswith(word) for word in question_words)

            is_viz_request = any(word in prompt_lower for word in viz_keywords) and not is_question
            last_assistant_msg_with_data = next((m for m in reversed(st.session_state.chat_history[:-1]) if m.get("chart_data")), None)

            if is_viz_request and last_assistant_msg_with_data:
                data = last_assistant_msg_with_data["chart_data"]
                chart_args = last_assistant_msg_with_data.get("chart_args", {})
                
                if 'table' in prompt_lower:
                    table_html = create_table_html(data, chart_args)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": f"[Table generated for '{user_input}']", "table_html": table_html, "chart_data": data, "chart_args": chart_args
                    })
                else: 
                    fig = create_chart_figure(data, user_input, chart_args)
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": f"[Chart generated for '{user_input}']", "figure": fig, "chart_data": data, "chart_args": chart_args
                    })
            else: 
                response = openai.chat.completions.create(
                    model="gpt-4o", messages=messages_for_gpt, functions=functions, function_call="auto"
                )
                msg = response.choices[0].message

                if msg.function_call:
                    fname = msg.function_call.name
                    args = json.loads(msg.function_call.arguments)
                    
                    # --- FIX: Handle the new get_unique_values function ---
                    if fname == "get_unique_values":
                        unique_list = get_unique_values(**args)
                        result_string = f"Here are the unique values for the '{args.get('column')}' column:\n\n"
                        result_string += ", ".join(sorted(map(str, unique_list)))
                        st.session_state.chat_history.append({"role": "assistant", "content": result_string})

                    elif fname in ["get_product_stat", "get_calculated_stat"]:
                        result_string = globals()[fname](**args)
                        st.session_state.chat_history.append({"role": "assistant", "content": result_string})
                    
                    else: 
                        result = globals()[fname](**args)
                        args["func_name"] = fname
                        
                        if fname == "get_top_n_by_calculated_metric":
                            result_string = f"Here are the top {args.get('n')} results for {args.get('numerator')} per {args.get('denominator')}:\n\n" + "\n".join([f"- {k.strip()}: ${v:,.2f}" for k, v in result.items()])
                        else:
                            result_string = f"Here are the {args.get('n', 'top')} results for {args.get('column')}:\n\n" + "\n".join([f"- {k.strip()}: ${v:,.2f}" for k, v in result.items()])
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": result_string, "chart_data": result, "chart_args": args})

                else: 
                    st.session_state.chat_history.append({"role": "assistant", "content": msg.content})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"An error occurred: {e}"})
    
    st.rerun()


st.markdown('<div class="credit">Created by Mohit Vaid</div>', unsafe_allow_html=True)

if not st.session_state.get('initial_load', True):
    st.components.v1.html("""
        <script>
            var chatContainer = window.parent.document.querySelector('.st-emotion-cache-1f1G2gn');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
    """, height=0)
