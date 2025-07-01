import streamlit as st
import pandas as pd
import openai
import plotly.express as px
import json
from difflib import get_close_matches

# --- Page and API Configuration ---
st.set_page_config(page_title="Medicaid Drug Spending NLP Analytics", layout="wide")
# Make sure to set your OpenAI API key in Streamlit's secrets management
# (e.g., in secrets.toml: OPENAI_API_KEY = "sk-...")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Custom CSS Styling ---
# FIX: Added styles for the clipboard functionality.
st.markdown("""
    <style>
    .user-msg {
        background-color: #007bff;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        text-align: right;
        font-size: 1.0rem;
        width: fit-content;
        margin-left: auto;
        max-width: 75%;
    }
    .assistant-msg-container {
        position: relative;
        width: fit-content;
        margin-right: auto;
        max-width: 95%;
    }
    .assistant-msg {
        background-color: #f1f1f1;
        color: black;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        text-align: left;
        font-size: 1.0rem;
    }
    .clipboard-icon {
        position: absolute;
        top: 10px;
        right: -30px; /* Position to the right of the bubble */
        cursor: pointer;
        opacity: 0; /* Hidden by default */
        transition: opacity 0.2s;
        font-size: 16px;
    }
    .assistant-msg-container:hover .clipboard-icon {
        opacity: 0.7; /* Show on hover */
    }
    .clipboard-icon:hover {
        opacity: 1;
    }
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        margin-top: 10px;
    }
    .dataframe th, .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    .dataframe th {
        background-color: #f8f9fa;
    }
    .credit {
        margin-top: 30px;
        font-size: 0.9rem;
        color: #888;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/cd6be561d335a58ec5ca855ba3065a9e05eadfac/assets/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")
st.markdown("#### Ask questions and customize visualizations with natural language.")

# --- Data Loading and Caching ---
@st.cache_data
def load_data():
    """Loads, cleans, and caches the dataset."""
    CSV_URL = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"
    df = pd.read_csv(CSV_URL)
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    numeric_cols = [
        "units_reimbursed", "number_of_prescriptions", "total_amount_reimbursed",
        "medicaid_amount_reimbursed", "non_medicaid_amount_reimbursed"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

df = load_data()

# --- Column Name Mapping and Normalization ---
SMART_COLUMN_MAP = {
    "spending": "total_amount_reimbursed",
    "cost": "total_amount_reimbursed",
    "reimbursement": "medicaid_amount_reimbursed",
    "prescriptions": "number_of_prescriptions",
    "prescription_count": "number_of_prescriptions",
    "units": "units_reimbursed"
}
COLUMN_LIST = df.columns.tolist()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "system", "content": """
        You are a helpful Medicaid data analyst assistant.
        - You MUST use the provided functions to answer any questions about the data. Do not invent answers.
        - Default to providing a text-based summary.
        - Only set 'display_as_chart' to true if the user explicitly asks for a 'chart', 'graph', 'plot', or 'visualize'. This includes donut, scatter, and area charts.
        - Only set 'display_as_table' to true if the user explicitly asks for a 'table'.
        - Listen for customization requests. If the user asks for a color scheme or palette (e.g., 'vibrant colors', 'use the plasma palette'), pass a valid Plotly sequential color scale name (e.g., 'Viridis', 'Plasma', 'Blues') to the `color_palette` parameter.
        """}
    ]

def resolve_column(col_name: str) -> str:
    """Finds the correct column name from user input."""
    col_name = col_name.lower().strip()
    if col_name in SMART_COLUMN_MAP:
        return SMART_COLUMN_MAP[col_name]
    matches = get_close_matches(col_name, COLUMN_LIST, n=1, cutoff=0.6)
    return matches[0] if matches else col_name

@st.cache_data(show_spinner=False)
def normalize_product_names():
    """Groups similar product names together."""
    name_map = {}
    unique_names = df["product_name"].astype(str).unique().tolist()
    for name in unique_names:
        match = get_close_matches(name, unique_names, n=1, cutoff=0.85)
        name_map[name] = match[0] if match else name
    return name_map

NAME_MAP = normalize_product_names()

# --- Data Analysis Functions (for AI) ---
def count_unique(column: str, **kwargs) -> int:
    return int(df[resolve_column(column)].nunique())

def sum_column(column: str, **kwargs) -> float:
    return float(df[resolve_column(column)].sum())

def top_n(column: str, n: int, **kwargs) -> dict:
    column = resolve_column(column)
    df_copy = df.copy()
    df_copy["product_name"] = df_copy["product_name"].astype(str).map(NAME_MAP)
    return df_copy.groupby("product_name")[column].sum().sort_values(ascending=False).head(n).to_dict()

def bottom_n(column: str, n: int, **kwargs) -> dict:
    column = resolve_column(column)
    return df.groupby("product_name")[column].sum().sort_values(ascending=True).head(n).to_dict()

def sum_by_product(column: str, **kwargs) -> dict:
    return df.groupby("product_name")[resolve_column(column)].sum().sort_values(ascending=False).to_dict()

def average_by_product(column: str, **kwargs) -> dict:
    return df.groupby("product_name")[resolve_column(column)].mean().sort_values(ascending=False).to_dict()

# --- AI Function Definitions ---
OUTPUT_FORMAT_PROPERTIES = {
    "display_as_chart": {"type": "boolean", "description": "Set to true if the user explicitly asks for a visual chart."},
    "display_as_table": {"type": "boolean", "description": "Set to true if the user explicitly asks for a data table."},
    "chart_type": {"type": "string", "enum": ["bar", "pie", "line", "donut", "scatter", "area"], "description": "The type of chart to display."},
    "title": {"type": "string", "description": "A custom title for the chart or table."},
    "color_palette": {"type": "string", "enum": ["Plasma", "Viridis", "Blues", "Greens", "Reds", "YlOrRd", "Spectral"], "description": "A named color palette for the chart."},
    "column_labels": {"type": "array", "items": {"type": "string"}, "description": "Custom labels for table columns, e.g., ['Drug Name', 'Total Sales']"}
}

functions = [
    {"name": "count_unique", "description": "Count unique values in a column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "sum_column", "description": "Sum values in a numeric column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}}, "required": ["column"]}},
    {"name": "top_n", "description": "Get top N products by a column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}, **OUTPUT_FORMAT_PROPERTIES}, "required": ["column", "n"]}},
    {"name": "bottom_n", "description": "Get bottom N products by a column.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, "n": {"type": "integer"}, **OUTPUT_FORMAT_PROPERTIES}, "required": ["column", "n"]}},
    {"name": "sum_by_product", "description": "Sum a column for each product.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, **OUTPUT_FORMAT_PROPERTIES}, "required": ["column"]}},
    {"name": "average_by_product", "description": "Calculate the average of a column for each product.", "parameters": {"type": "object", "properties": {"column": {"type": "string"}, **OUTPUT_FORMAT_PROPERTIES}, "required": ["column"]}}
]

st.subheader("📊 Sample of the dataset")
st.dataframe(df.head(10), use_container_width=True)

# --- Chat Interface ---
st.subheader("💬 Chat Interface")

chat_container = st.container(height=550)

for i, msg in enumerate(st.session_state.chat_history):
    if msg["role"] == "system":
        continue
    
    content = msg["content"]
    
    if msg["role"] == "user":
        chat_container.markdown(f'<div class="user-msg">{content}</div>', unsafe_allow_html=True)
    else: # Assistant
        if hasattr(content, 'to_plotly_json'):
            chat_container.plotly_chart(content, use_container_width=True, key=f"chart_{i}")
        else:
            # FIX: Wrap text-based messages in a container to position the clipboard icon.
            html_content = ""
            text_to_copy = ""
            if isinstance(content, pd.DataFrame):
                html_content = content.to_html(classes='dataframe', border=0, index=False)
                text_to_copy = content.to_csv(index=False)
            elif isinstance(content, str):
                html_content = content.replace("\n", "<br>")
                text_to_copy = content

            # Using a unique ID for each message for the clipboard functionality
            message_id = f"msg-{i}"
            
            # JavaScript to handle the copy action
            js_code = f"""
            <script>
            function copyToClipboard_{message_id}() {{
                const textToCopy = `{text_to_copy.replace('`', '\\`')}`; // Escape backticks in the text
                const tempTextArea = document.createElement('textarea');
                tempTextArea.value = textToCopy;
                document.body.appendChild(tempTextArea);
                tempTextArea.select();
                document.execCommand('copy');
                document.body.removeChild(tempTextArea);
                
                // Optional: Show a "Copied!" message
                const icon = document.getElementById('icon_{message_id}');
                const original_text = icon.innerHTML;
                icon.innerHTML = '✅';
                setTimeout(() => {{ icon.innerHTML = original_text; }}, 1000);
            }}
            </script>
            """
            
            chat_container.markdown(f"""
            <div class="assistant-msg-container">
                <div class="assistant-msg">{html_content}</div>
                <span id="icon_{message_id}" class="clipboard-icon" onclick="copyToClipboard_{message_id}()">📋</span>
                {js_code}
            </div>
            """, unsafe_allow_html=True)


# The chat input is defined here, and Streamlit docks it to the bottom of the page.
user_input = st.chat_input("Ask a question, e.g., 'Show me a table of the top 5 drugs by spending'")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    messages_for_gpt = [msg for msg in st.session_state.chat_history if isinstance(msg.get("content"), str)]

    with st.spinner("Analyzing..."):
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=messages_for_gpt,
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
                    result_is_dict = isinstance(result, dict)

                    if result_is_dict and args.get("display_as_table"):
                        custom_labels = args.get("column_labels", ["Product", "Value"])
                        table_df = pd.DataFrame(list(result.items()), columns=custom_labels)
                        st.session_state.chat_history.append({"role": "assistant", "content": table_df})

                    elif result_is_dict and args.get("display_as_chart"):
                        chart_df = pd.DataFrame.from_dict(result, orient='index', columns=["Value"]).reset_index()
                        chart_df.columns = ["Product", "Value"]
                        
                        chart_type = args.get("chart_type", "bar")
                        y_axis_title = args.get("column", "Value").replace("_", " ").title()
                        default_title = f"{fname.replace('_', ' ').title()} of {y_axis_title}"
                        chart_title = args.get("title", default_title)
                        
                        color_palette_name = args.get("color_palette")
                        plotly_kwargs = {}
                        
                        if color_palette_name:
                            try:
                                color_sequence = getattr(px.colors.sequential, color_palette_name)
                                plotly_kwargs['color_discrete_sequence'] = color_sequence
                            except AttributeError:
                                pass
                        
                        fig = None
                        if chart_type == 'pie':
                            fig = px.pie(chart_df, names="Product", values="Value", **plotly_kwargs)
                        elif chart_type == 'donut':
                            fig = px.pie(chart_df, names="Product", values="Value", hole=0.4, **plotly_kwargs)
                        elif chart_type == 'line':
                            fig = px.line(chart_df, x="Product", y="Value", markers=True, **plotly_kwargs)
                        elif chart_type == 'scatter':
                            fig = px.scatter(chart_df, x="Product", y="Value", size="Value", **plotly_kwargs)
                        elif chart_type == 'area':
                             fig = px.area(chart_df, x="Product", y="Value", **plotly_kwargs)
                        else: # Default to bar
                            if 'color_discrete_sequence' not in plotly_kwargs:
                                plotly_kwargs['color_discrete_sequence'] = ['blue'] * len(chart_df)
                            fig = px.bar(chart_df, x="Product", y="Value", text_auto='.2s', **plotly_kwargs)
                            fig.update_traces(textangle=0, textposition='outside')

                        if fig:
                            fig.update_layout(
                                title={
                                    'text': chart_title,
                                    'y':0.9,
                                    'x':0.5,
                                    'xanchor': 'center',
                                    'yanchor': 'top'
                                },
                                xaxis_title="Product", 
                                yaxis_title=y_axis_title
                            )
                            st.session_state.chat_history.append({"role": "assistant", "content": fig})

                    elif result_is_dict:
                        formatted_str = "\n".join([f"- **{k.strip()}**: ${v:,.2f}" for k, v in result.items()])
                        st.session_state.chat_history.append({"role": "assistant", "content": formatted_str})
                    else:
                        st.session_state.chat_history.append({"role": "assistant", "content": str(result)})

                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Function error: {e}"})
            elif msg.content:
                st.session_state.chat_history.append({"role": "assistant", "content": msg.content})

        except Exception as e:
            st.session_state.chat_history.append({"role": "assistant", "content": f"Chat request failed: {e}"})
    
    # Rerun the script to display the new message
    st.rerun()

st.markdown('<div class="credit">Created by Mohit Vaid</div>', unsafe_allow_html=True)
