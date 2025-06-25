import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# --- Configuration ---
openai.api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="Medicaid Drug Analytics", layout="wide")

# --- UI Header ---
st.image("https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/main/logo.png", width=150)
st.title("💊 Medicaid Drug Spending NLP Analytics")
st.markdown("#### Ask any question about the dataset below and generate insights or charts without manual coding.")

# --- Data Loading and Cleaning ---
CSV_URL = "https://raw.githubusercontent.com/Mohit24-jpg/medicaid-analysis-app-v2/master/data-06-17-2025-2_01pm.csv"

@st.cache_data(show_spinner=True)
def load_and_clean():
    df = pd.read_csv(CSV_URL)
    # Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    # Numeric conversion and fill
    for col in ['units_reimbursed', 'number_of_prescriptions', 'total_amount_reimbursed',
                'medicaid_amount_reimbursed', 'non_medicaid_amount_reimbursed']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

df = load_and_clean()
if df.empty:
    st.error("Failed to load dataset. Please check the CSV URL.")
    st.stop()

# --- Precompute Summaries ---
top_by_units = df.groupby('product_name')['units_reimbursed'].sum().sort_values(ascending=False).head(10)
top_by_prescriptions = df.groupby('product_name')['number_of_prescriptions'].sum().sort_values(ascending=False).head(10)
top_by_reimbursement = df.groupby('product_name')['total_amount_reimbursed'].sum().sort_values(ascending=False).head(10)
bottom_by_prescriptions = df.groupby('product_name')['number_of_prescriptions'].sum().sort_values().head(10)

# --- Helper Functions ---
SYNONYMS = {
    'units': 'units_reimbursed',
    'prescriptions': 'number_of_prescriptions',
    'reimbursed': 'total_amount_reimbursed',
    'medicaid': 'medicaid_amount_reimbursed',
    'non_medicaid': 'non_medicaid_amount_reimbursed'
}

def resolve_column(question):
    q = question.lower()
    for key, col in SYNONYMS.items():
        if key in q:
            return col
    # Fallback: fuzzy match
    matches = {col: process.extractOne(col, [question])[1] for col in df.columns}
    best = max(matches, key=matches.get)
    return best

# Build a small data digest for GPT context
digest = {
    'top_units': top_by_units.head(5).to_dict(),
    'bottom_prescriptions': bottom_by_prescriptions.head(5).to_dict(),
    'top_reimbursement': top_by_reimbursement.head(5).to_dict()
}

# GPT call
def ask_gpt(prompt):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data analyst assistant specializing in Medicaid drug spending."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return resp.choices[0].message.content.strip()

# Build dynamic prompt based on user question
def build_prompt(question):
    col = resolve_column(question)
    # Craft prompt including digest and directive
    prompt = (
        f"Dataset digest:\n"
        f"Top 5 by units reimbursed: {digest['top_units']}\n"
        f"Bottom 5 by prescriptions: {digest['bottom_prescriptions']}\n"
        f"Top 5 by reimbursement: {digest['top_reimbursement']}\n"
        f"When asked about '{question}', use column '{col}' and the full dataset to provide an accurate answer."
        " Do not explain your reasoning, just give the result." 
    )
    return prompt

# --- Charting Utility ---
def plot_metric(question):
    col = resolve_column(question)
    series = df.groupby('product_name')[col].sum().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(kind='bar', ax=ax)
    ax.set_title(f"Top 10 Products by {col.replace('_', ' ').title()}")
    ax.set_ylabel(col.replace('_', ' ').title())
    plt.xticks(rotation=30, ha='right')
    st.pyplot(fig)

# --- UI: Display Preview and Interaction ---
st.subheader("📄 Data Preview")
st.dataframe(df.head(10))

st.subheader("❓ Ask a Question")
question = st.text_input("Enter your question (e.g. 'Show top drugs by units reimbursed')")

col1, col2 = st.columns(2)
with col1:
    if st.button("🧠 Get Text Answer"):
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing..."):
                prompt = build_prompt(question)
                answer = ask_gpt(prompt)
                st.markdown(
                    f"<div style='padding:15px; background:#f9f9f9; border-radius:8px;'>" 
                    f"<strong>Answer:</strong><br>{answer}</div>",
                    unsafe_allow_html=True
                )
with col2:
    if st.button("📊 Create Chart"):
        if not question:
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Generating chart..."):
                plot_metric(question)
