import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Medicaid Drug Spending NLP Analytics")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your Medicaid CSV file", type=["csv"])
if not uploaded_file:
    st.info("Please upload a CSV file to continue.")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)

# Show preview
st.subheader("Data preview (first 10 rows)")
st.dataframe(df.head(10))

# Known important columns (your list)
EXPECTED_COLUMNS = [
    "Utilization Type", "State", "NDC", "Labeler Code", "Product Code", "Package Size",
    "Year", "Quarter", "Suppression Used", "Product Name", "Units Reimbursed",
    "Number of Prescriptions", "Total Amount Reimbursed", "Medicaid Amount Reimbursed",
    "Non Medicaid Amount Reimbursed"
]

# Map user question terms to closest column names for better matching
def fuzzy_column_match(question, columns):
    # Extract key nouns or terms? (simple: match any column roughly)
    matches = {}
    for col in columns:
        score = process.extractOne(col, [question])[1]
        matches[col] = score
    # Return top matches over threshold
    good_matches = [col for col, score in matches.items() if score > 60]
    return good_matches if good_matches else columns  # fallback all columns

def ask_openai_chat(messages):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def generate_text_answer(df, question):
    # Aggregate top 3 by total reimbursed amount
    top3 = df.groupby("Product Name")["Total Amount Reimbursed"].sum().sort_values(ascending=False).head(3)
    top3_text = "\n".join([f"{i+1}. {name} - ${amount:,.2f}" for i, (name, amount) in enumerate(top3.items())])

    # You can also provide other useful aggregates if needed (e.g. total reimbursed)
    total_reimbursed = df["Total Amount Reimbursed"].sum()

    # Fuzzy match columns for user question context (optional advanced)
    matched_cols = fuzzy_column_match(question, df.columns)

    context_text = (
        f"Dataset info:\n"
        f"- Total Amount Reimbursed: ${total_reimbursed:,.2f}\n"
        f"- Top 3 Products by Total Amount Reimbursed:\n{top3_text}\n"
        f"Columns matched to your question: {', '.join(matched_cols)}\n\n"
        f"Answer the following question concisely without extra explanation:\n{question}"
    )

    messages = [
        {"role": "system", "content": "You are a helpful data analyst assistant."},
        {"role": "user", "content": context_text}
    ]

    return ask_openai_chat(messages)

def generate_chart(df, question):
    # For demo: always plot top 5 products by total reimbursed (can extend to parse question)
    top5 = df.groupby("Product Name")["Total Amount Reimbursed"].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 4))  # smaller size
    top5.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Top 5 Products by Total Amount Reimbursed")
    ax.set_ylabel("Total Amount Reimbursed ($)")
    ax.set_xlabel("Product Name")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Ask a question about the dataset")

question = st.text_input("Enter your question here:")

col1, col2 = st.columns(2)

with col1:
    if st.button("Get Text Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                answer = generate_text_answer(df, question)
                st.markdown(f"**Answer:** {answer}")

with col2:
    if st.button("Create Chart"):
        if not question.strip():
            st.warning("Please enter a question to create a chart.")
        else:
            with st.spinner("Creating chart..."):
                generate_chart(df, question)
