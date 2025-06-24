import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Medicaid Drug Spending App", layout="wide")
st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/main/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")

st.markdown("""
#### Ask questions and generate charts using Medicaid drug data from GitHub
""")

# GitHub CSV file URL
CSV_URL = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(CSV_URL)
    num_cols = [
        "Total Amount Reimbursed",
        "Number of Prescriptions",
        "Units Reimbursed"
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

df = load_data()
if df.empty:
    st.warning("No data returned from the CSV file.")
    st.stop()

# Preview
display_cols = [col for col in df.columns if len(col) < 60]
st.subheader("📄 Preview of Dataset")
st.dataframe(df[display_cols].head(10))

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
    total_reimbursed = df["Total Amount Reimbursed"].sum()
    top3 = df.groupby("Product Name")["Total Amount Reimbursed"].sum().sort_values(ascending=False).head(3)
    bottom3 = df.groupby("Product Name")["Number of Prescriptions"].sum().sort_values().head(3)
    matched_cols = fuzzy_column_match(question, df.columns)

    if "least prescribed" in question.lower():
        table = "\n".join([f"  {i+1}. {k}: {v:,.0f} prescriptions" for i, (k, v) in enumerate(bottom3.items())])
        context_text = f"The least prescribed drugs based on the dataset are:\n{table}"
    else:
        context_text = (
            f"Dataset info:\n"
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
    top5 = df.groupby("Product Name")["Total Amount Reimbursed"].sum().sort_values(ascending=False).head(5)
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
