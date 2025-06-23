import streamlit as st
import pandas as pd
import requests
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# === Set API key from Streamlit secrets ===
openai.api_key = st.secrets["OPENAI_API_KEY"]

# === Page config and logo ===
st.set_page_config(page_title="Medicaid Drug Spending NLP Analytics", layout="wide")
st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/main/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")

st.markdown("#### Ask questions and generate charts using live Medicaid drug data")

# === Sidebar filters ===
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
    base_url = "https://data.medicaid.gov/resource/ynj2-r877.json"
    # Build query parameters with filters
    where_clause = f"state='{state}' AND year='{year}'"
    if quarter:
        where_clause += f" AND quarter='{quarter}'"

    params = {
        "$limit": 50000,
        "$where": where_clause
    }

    try:
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)

        # Convert important numeric columns
        numeric_cols = [
            "total_reimbursement_amt",
            "number_of_prescriptions",
            "units_reimbursed"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

df = load_data(state, year, quarter)
if df.empty:
    st.warning("No data found for the selected filters. Try different state/year/quarter.")
    st.stop()

# Preview data
st.subheader(f"📄 Preview for {state} - {year} Q{quarter or 'All'}")
st.dataframe(df.head(10))

# Fuzzy column matching helper
def fuzzy_column_match(question, columns):
    matches = {}
    for col in columns:
        score = process.extractOne(col, [question])[1]
        matches[col] = score
    good_matches = [col for col, score in matches.items() if score > 60]
    return good_matches if good_matches else columns

# OpenAI chat call helper
def ask_openai_chat(messages):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# Generate text answer
def generate_text_answer(df, question):
    # Top 3 products by total reimbursement amount
    if "product_name" not in df.columns or "total_reimbursement_amt" not in df.columns:
        return "Dataset missing required columns for analysis."

    top3 = df.groupby("product_name")["total_reimbursement_amt"].sum().sort_values(ascending=False).head(3)
    total_reimbursed = df["total_reimbursement_amt"].sum()
    matched_cols = fuzzy_column_match(question, df.columns)

    context_text = (
        f"Dataset info for {state} {year} Q{quarter or 'All'}:\n"
        f"- Total Reimbursement Amount: ${total_reimbursed:,.2f}\n"
        f"- Top 3 Products by Total Reimbursement Amount:\n" +
        "\n".join([f"  {i+1}. {k}: ${v:,.2f}" for i, (k, v) in enumerate(top3.items())]) +
        f"\n\nColumns matched to your question: {', '.join(matched_cols)}\n\n"
        f"Answer the following question concisely:\n{question}"
    )

    messages = [
        {"role": "system", "content": "You are a helpful Medicaid drug data analyst."},
        {"role": "user", "content": context_text}
    ]
    return ask_openai_chat(messages)

# Generate chart (top 5 products by total reimbursement)
def generate_chart(df, question):
    if "product_name" not in df.columns or "total_reimbursement_amt" not in df.columns:
        st.error("Dataset missing required columns for charting.")
        return
    top5 = df.groupby("product_name")["total_reimbursement_amt"].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 4))
    top5.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top 5 Products by Total Reimbursement Amount")
    ax.set_ylabel("Reimbursement Amount ($)")
    ax.set_xlabel("Product Name")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# UI: Question input and buttons
st.subheader("💬 Ask a question about this dataset")
question = st.text_input("Ask anything like 'Top 5 drugs by prescriptions'")

col1, col2 = st.columns(2)

with col1:
    if st.button("🧠 Get Text Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_text_answer(df, question)
                st.markdown(
                    f"""
                    <div style="
                        border:2px solid #4CAF50; 
                        padding:15px; 
                        border-radius:10px; 
                        background-color:#e8f5e9;
                        font-size:16px;
                        line-height:1.4;
                    ">
                    <strong>💡 Answer:</strong><br>{answer}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with col2:
    if st.button("📊 Create Chart"):
        if not question.strip():
            st.warning("Please enter a question to create a chart.")
        else:
            with st.spinner("Generating chart..."):
                generate_chart(df, question)
