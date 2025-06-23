import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import requests

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Medicaid Drug Spending App", layout="wide")
st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/main/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")

st.markdown("""
#### Ask questions and generate charts using live Medicaid drug data
""")

# Sidebar filters
with st.sidebar:
    st.header("📊 Filter Dataset")
    state = st.text_input("Enter state abbreviation (e.g. OH)", max_chars=2).upper()
    year = st.text_input("Enter year (e.g. 2023)", max_chars=4)
    quarter = st.selectbox("Select quarter", options=["", "1", "2", "3", "4"])

if not (state and year):
    st.info("Please enter state and year to load data.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_data(state, year, quarter):
    base_url = "https://data.medicaid.gov/resource/26vd-nv8x.json"
    params = {
        "$limit": 50000,
        "$where": f"state='{state}' AND year='{year}'" + (f" AND quarter='{quarter}'" if quarter else "")
    }
    try:
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        # Convert important columns to numeric
        num_cols = ["total_amount_reimbursed", "number_of_prescriptions"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_data(state, year, quarter)
if df.empty:
    st.warning("No data returned. Check filters or try another state/year.")
    st.stop()

# Preview
st.subheader(f"📄 Preview for {state} - {year} Q{quarter or 'All'}")
st.dataframe(df.head(10))

# Column mapping helper
def fuzzy_column_match(question, columns):
    matches = {}
    for col in columns:
        score = process.extractOne(col, [question])[1]
        matches[col] = score
    good_matches = [col for col, score in matches.items() if score > 60]
    return good_matches if good_matches else columns

def ask_openai_chat(messages):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_text_answer(df, question):
    top3 = df.groupby("product_name")["total_amount_reimbursed"].sum().sort_values(ascending=False).head(3)
    total_reimbursed = df["total_amount_reimbursed"].sum()
    matched_cols = fuzzy_column_match(question, df.columns)

    context_text = (
        f"Dataset info for {state} {year}:\n"
        f"- Total reimbursed: ${total_reimbursed:,.2f}\n"
        f"- Top 3 by reimbursed amount:\n" +
        "\n".join([f"  {i+1}. {k}: ${v:,.2f}" for i, (k, v) in enumerate(top3.items())]) +
        f"\n\nColumns matched to your question: {', '.join(matched_cols)}\n\n"
        f"Answer the following question concisely:\n{question}"
    )

    messages = [
        {"role": "system", "content": "You are a helpful Medicaid drug data analyst."},
        {"role": "user", "content": context_text}
    ]
    return ask_openai_chat(messages)

def generate_chart(df, question):
    top5 = df.groupby("product_name")["total_amount_reimbursed"].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 4))
    top5.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top 5 Products by Total Amount Reimbursed")
    ax.set_ylabel("Reimbursed ($)")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig)

# Input and output layout
st.subheader("💬 Ask a question about this dataset")
question = st.text_input("Ask anything like 'Top 5 drugs by prescriptions'")
col1, col2 = st.columns(2)

with col1:
    if st.button("🧠 Get Text Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                answer = generate_text_answer(df, question)
                st.markdown(
                    """
                    <div style="border:1px solid #ccc; padding:12px; border-radius:8px; background-color:#f9f9f9">
                    <strong>💡 Answer:</strong><br>
                    {}</div>
                    """.format(answer), unsafe_allow_html=True)

with col2:
    if st.button("📊 Create Chart"):
        if not question.strip():
            st.warning("Please enter a question for the chart.")
        else:
            with st.spinner("Generating chart..."):
                generate_chart(df, question)
