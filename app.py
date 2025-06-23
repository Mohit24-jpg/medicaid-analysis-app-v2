import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import requests

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Medicaid Drug Spending App", layout="wide")

# Show company logo (you can change URL to your own)
st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/main/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")
st.markdown("#### Ask questions and generate charts using live Medicaid drug data from data.medicaid.gov")

# Sidebar filters
with st.sidebar:
    st.header("📊 Filter Dataset")
    state = st.text_input("Enter state abbreviation (e.g. OH)", max_chars=2).upper()
    year = st.text_input("Enter year (e.g. 2023)", max_chars=4)
    quarter = st.selectbox("Select quarter (optional)", options=["", "1", "2", "3", "4"])

if not (state and year):
    st.info("Please enter both state and year to load data.")
    st.stop()

@st.cache_data(show_spinner=True)
def load_data(state, year, quarter):
    base_url = "https://data.medicaid.gov/resource/26vd-nv8x.json"
    # Build the where clause dynamically
    where_clause = f"state='{state}' AND year='{year}'"
    if quarter:
        where_clause += f" AND quarter='{quarter}'"

    params = {
        "$limit": 50000,
        "$where": where_clause
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            return pd.DataFrame()  # empty
        df = pd.DataFrame(data)
        # Convert important columns to numeric (some may be missing, handle gracefully)
        numeric_cols = [
            "total_amount_reimbursed",
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
    st.warning("No data found for your filters. Please try different state/year/quarter.")
    st.stop()

# Preview data
st.subheader(f"📄 Data preview for {state} - {year} Q{quarter or 'All'}")
st.dataframe(df.head(10))

# Helper: fuzzy column match
def fuzzy_column_match(question, columns):
    matches = {}
    for col in columns:
        score = process.extractOne(col, [question])[1]
        matches[col] = score
    good_matches = [col for col, score in matches.items() if score > 60]
    return good_matches if good_matches else columns

# OpenAI Chat helper
def ask_openai_chat(messages):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Generate text answer based on dataframe and question
def generate_text_answer(df, question):
    # Example: top 3 products by total reimbursed
    if "product_name" not in df.columns or "total_amount_reimbursed" not in df.columns:
        return "Sorry, the necessary data columns are missing."

    top3 = df.groupby("product_name")["total_amount_reimbursed"].sum().sort_values(ascending=False).head(3)
    total_reimbursed = df["total_amount_reimbursed"].sum()

    matched_cols = fuzzy_column_match(question, df.columns)

    context_text = (
        f"Dataset info for {state} {year} quarter {quarter or 'All'}:\n"
        f"- Total Amount Reimbursed: ${total_reimbursed:,.2f}\n"
        f"- Top 3 Products by Total Amount Reimbursed:\n" +
        "\n".join([f"  {i+1}. {k}: ${v:,.2f}" for i, (k, v) in enumerate(top3.items())]) +
        f"\n\nColumns matched to your question: {', '.join(matched_cols)}\n\n"
        f"Answer the following question concisely:\n{question}"
    )

    messages = [
        {"role": "system", "content": "You are a helpful Medicaid drug data analyst."},
        {"role": "user", "content": context_text}
    ]
    return ask_openai_chat(messages)

# Generate bar chart for top 5 products by reimbursed amount
def generate_chart(df, question):
    if "product_name" not in df.columns or "total_amount_reimbursed" not in df.columns:
        st.error("Necessary columns missing for chart.")
        return

    top5 = df.groupby("product_name")["total_amount_reimbursed"].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 4))
    top5.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top 5 Products by Total Amount Reimbursed")
    ax.set_ylabel("Total Amount Reimbursed ($)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# Question input and buttons
st.subheader("💬 Ask a question about this dataset")
question = st.text_input("Enter your question here (e.g. 'Top 5 drugs by prescriptions')")

col1, col2 = st.columns(2)

with col1:
    if st.button("🧠 Get Text Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_text_answer(df, question)
                st.markdown(
                    """
                    <div style="border:1px solid #ccc; padding:12px; border-radius:8px; background-color:#f9f9f9; font-size:16px;">
                    <strong>💡 Answer:</strong><br>
                    {}</div>
                    """.format(answer), unsafe_allow_html=True)

with col2:
    if st.button("📊 Create Chart"):
        if not question.strip():
            st.warning("Please enter a question to generate a chart.")
        else:
            with st.spinner("Generating chart..."):
                generate_chart(df, question)
