import streamlit as st
import pandas as pd
import openai
import requests

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Medicaid Spending NLP", layout="wide")
st.title("💊 Medicaid Drug Spending NLP Analytics")

# Load API data
@st.cache_data
def load_data():
    url = "https://data.medicaid.gov/resource/srj6-uykx.json?$limit=50000"
    resp = requests.get(url)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())

df = load_data()
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# Convert numeric columns
for num_col in ["total_amount_reimbursed", "number_of_prescriptions", "units_reimbursed", "medicaid_amount_reimbursed"]:
    df[num_col] = pd.to_numeric(df.get(num_col, None), errors="coerce")

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    states = sorted(df["state"].dropna().unique())
    state = st.selectbox("State", states)

    years = sorted(df["year"].dropna().unique())
    year = st.selectbox("Year", years)

filtered = df[(df["state"] == state) & (df["year"] == year)]
if filtered.empty:
    st.warning(f"No data found for {state} in {year}.")
    st.stop()

st.subheader(f"Preview: {state} — {year}")
st.dataframe(filtered.head(10))

# GPT Question Answering
def ask_gpt(question, data_df):
    summary = data_df.describe(include="all", datetime_is_numeric=True).fillna("").to_string()
    columns = ", ".join(data_df.columns)
    sample = data_df.sample(n=min(100, len(data_df))).to_dict(orient="records")

    messages = [
        {"role": "system", "content": "You are a data analyst. Answer only based on the provided data."},
        {"role": "user", "content": f"Columns: {columns}"},
        {"role": "user", "content": f"Sample: {sample}"},
        {"role": "user", "content": f"Answer concisely: {question}"}
    ]

    resp = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
    return resp.choices[0].message.content.strip()

st.subheader("Ask a question")
question = st.text_input("E.g., top 3 drugs by total amount reimbursed")

if st.button("Get Answer"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = ask_gpt(question, filtered)
            st.markdown(f"**Answer:** {answer}")
