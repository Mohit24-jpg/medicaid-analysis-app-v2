import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import json

🛠️ Configuration
openai.api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Medicaid Drug Analytics", layout="wide")

🚩 UI Header
st.image(
"https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/main/logo.png",
width=150
)
st.title("💊 Medicaid Drug Spending NLP Analytics")
st.markdown(
"#### Ask natural language questions about drug spending, reimbursement, and utilization directly from the dataset."
)

📥 Load Data
CSV_URL = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"

@st.cache_data(show_spinner=True)
def load_and_clean():
df = pd.read_csv(CSV_URL)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
for col in [
'units_reimbursed', 'number_of_prescriptions',
'total_amount_reimbursed', 'medicaid_amount_reimbursed',
'non_medicaid_amount_reimbursed'
]:
if col in df.columns:
df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
return df

df = load_and_clean()
if df.empty:
st.error("Failed to load dataset. Please check the CSV URL.")
st.stop()

COLUMN_LIST = df.columns.tolist()

🔧 App Functions
def count_unique(column: str) -> int:
return int(df[column].nunique())

def sum_column(column: str) -> float:
return float(df[column].sum())

def top_n(column: str, n: int) -> dict:
if column not in df.columns:
raise ValueError(f"Column '{column}' not found.")
return (
df.groupby("product_name")[column]
.sum()
.sort_values(ascending=False)
.head(n)
.to_dict()
)

def bottom_n(column: str, n: int) -> dict:
if column not in df.columns:
raise ValueError(f"Column '{column}' not found.")
return (
df.groupby("product_name")[column]
.sum()
.sort_values(ascending=True)
.head(n)
.to_dict()
)

def sum_by_product(column: str) -> dict:
return df.groupby("product_name")[column].sum().sort_values(ascending=False).to_dict()

def average_by_product(column: str) -> dict:
return df.groupby("product_name")[column].mean().sort_values(ascending=False).to_dict()

🔍 OpenAI Function Definitions
functions = [
{
"name": "count_unique",
"description": "Count unique values in a column",
"parameters": {
"type": "object",
"properties": {
"column": {"type": "string", "enum": COLUMN_LIST}
},
"required": ["column"]
}
},
{
"name": "sum_column",
"description": "Sum values in a numeric column",
"parameters": {
"type": "object",
"properties": {
"column": {"type": "string", "enum": COLUMN_LIST}
},
"required": ["column"]
}
},
{
"name": "top_n",
"description": "Get top N products by a numeric column",
"parameters": {
"type": "object",
"properties": {
"column": {"type": "string", "enum": COLUMN_LIST},
"n": {"type": "integer"}
},
"required": ["column", "n"]
}
},
{
"name": "bottom_n",
"description": "Get bottom N products by a numeric column",
"parameters": {
"type": "object",
"properties": {
"column": {"type": "string", "enum": COLUMN_LIST},
"n": {"type": "integer"}
},
"required": ["column", "n"]
}
},
{
"name": "sum_by_product",
"description": "Sum a numeric column for each product",
"parameters": {
"type": "object",
"properties": {
"column": {"type": "string", "enum": COLUMN_LIST}
},
"required": ["column"]
}
},
{
"name": "average_by_product",
"description": "Calculate average of a numeric column for each product",
"parameters": {
"type": "object",
"properties": {
"column": {"type": "string", "enum": COLUMN_LIST}
},
"required": ["column"]
}
}
]

📊 UI Interface
st.subheader("📄 Data Preview")
st.dataframe(df.head(10))

st.subheader("❓ Ask a Question")
question = st.text_input("Ask a question like 'Top 5 products by total_amount_reimbursed'")

col1, col2 = st.columns(2)

with col1:
if st.button("Get Text Answer"):
if not question:
st.warning("Please enter a question.")
else:
with st.spinner("Thinking..."):
response = openai.chat.completions.create(
model="gpt-4o",
messages=[
{"role": "system", "content": "You're a data analyst. Use function calling to return correct structured results."},
{"role": "user", "content": question}
],
functions=functions,
function_call="auto"
)
msg = response.choices[0].message
if hasattr(msg, "function_call") and msg.function_call is not None:
fname = msg.function_call.name
args = json.loads(msg.function_call.arguments)
try:
result = globals()fname
st.markdown(f"📌 Result from {fname}:")
st.json(result)
except Exception as e:
st.error(f"Error: {e}")
else:
st.markdown(f"💬 {msg.content}")

with col2:
if st.button("Create Chart"):
if not question:
st.warning("Enter a question first.")
else:
with st.spinner("Generating chart..."):
response = openai.chat.completions.create(
model="gpt-4o",
messages=[
{"role": "system", "content": "You're a chart analyst. Use function calling to fetch numerical chart data."},
{"role": "user", "content": question}
],
functions=functions,
function_call="auto"
)
msg = response.choices[0].message
if hasattr(msg, "function_call") and msg.function_call is not None:
fname = msg.function_call.name
args = json.loads(msg.function_call.arguments)
try:
data = globals()fname
series = pd.Series(data)
fig, ax = plt.subplots(figsize=(8, 4))
series.plot(kind='bar', ax=ax)
ax.set_title(f"{fname} on {args.get('column')}")
plt.xticks(rotation=30, ha='right')
st.pyplot(fig)
except Exception as e:
st.error(f"Chart Error: {e}")
else:
st.markdown(f"📝 {msg.content}")