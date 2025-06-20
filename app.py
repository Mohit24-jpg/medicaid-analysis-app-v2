import streamlit as st
import pandas as pd
import openai
import requests

# Set OpenAI API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Medicaid Drug Spending NLP", layout="wide")
st.title("💊 Medicaid Drug Spending NLP Analytics")

# Load Medicaid JSON API
@st.cache_data
def load_data():
    url = "https://data.medicaid.gov/resource/61729e5a-7aa8-448c-8903-ba3e0cd0ea3c.json?$limit=50000"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("Failed to load data from Medicaid API.")
        return None
    return pd.DataFrame(response.json())

# Load data
df = load_data()
if df is None:
    st.stop()

# Normalize column names
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# Clean and convert numeric columns
for col in ['total_amount_reimbursed', 'number_of_prescriptions', 'units_reimbursed']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Sidebar filters
with st.sidebar:
    st.header("📌 Filter Data")
    states = sorted(df['state'].dropna().unique())
    selected_state = st.selectbox("Select State", states)

    years = sorted(df['year'].dropna().unique())
    selected_year = st.selectbox("Select Year", years)

    st.markdown("---")
    st.markdown("🔍 Ask a question in natural language about the filtered data.")

# Filter dataset
filtered_df = df[(df['state'] == selected_state) & (df['year'] == selected_year)]

st.subheader(f"📄 Preview: {selected_state} - {selected_year}")
st.dataframe(filtered_df.head(10))

# GPT-based question answering
def ask_gpt(question, df):
    # Create a summary of the dataset
    summary = df.describe(include='all', datetime_is_numeric=True).fillna("").to_string()
    columns = ", ".join(df.columns.tolist())

    messages = [
        {"role": "system", "content": "You are a helpful data analyst assistant. Be concise and only output the answer."},
        {
            "role": "user",
            "content": (
                f"Here is a summary of the Medicaid dataset:\n\n"
                f"{summary}\n\n"
                f"Columns available: {columns}\n\n"
                f"Answer the following question: {question}"
            )
        }
    ]

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Question input
st.subheader("💬 Ask Your Question")
question = st.text_input("Example: What are the top 3 most prescribed drugs by total amount reimbursed?")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    elif filtered_df.empty:
        st.warning("No data for the selected filters.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_gpt(question, filtered_df)
            st.markdown(f"**Answer:** {answer}")
