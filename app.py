import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
from fuzzywuzzy import process

# 🎯 Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 🖼️ Logo and Title
st.image("assets/logo.png", width=150)
st.markdown("## 💊 Medicaid Drug Spending NLP Analytics")
st.markdown("---")

# 📤 CSV Upload
uploaded_file = st.file_uploader("📎 Upload your Medicaid CSV file", type=["csv"])
if not uploaded_file:
    st.info("⬆️ Please upload a CSV file to continue.")
    st.stop()

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(uploaded_file)

# 🔍 Preview
st.subheader("📄 Data preview (first 10 rows)")
st.dataframe(df.head(10))

# 🧠 Known columns
EXPECTED_COLUMNS = [
    "Utilization Type", "State", "NDC", "Labeler Code", "Product Code", "Package Size",
    "Year", "Quarter", "Suppression Used", "Product Name", "Units Reimbursed",
    "Number of Prescriptions", "Total Amount Reimbursed", "Medicaid Amount Reimbursed",
    "Non Medicaid Amount Reimbursed"
]

# 🔍 Fuzzy match for smarter column lookup
def fuzzy_column_match(question, columns):
    matches = {col: process.extractOne(col, [question])[1] for col in columns}
    return [col for col, score in matches.items() if score > 60] or columns

# 🤖 GPT call
def ask_openai_chat(messages):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# 📜 Smart answer generator
def generate_text_answer(df, question):
    top3 = df.groupby("Product Name")["Total Amount Reimbursed"].sum().sort_values(ascending=False).head(3)
    top3_text = "\n".join([f"{i+1}. {name} - ${amount:,.2f}" for i, (name, amount) in enumerate(top3.items())])
    total_reimbursed = df["Total Amount Reimbursed"].sum()
    matched_cols = fuzzy_column_match(question, df.columns)

    context_text = (
        f"Dataset info:\n"
        f"- Total Amount Reimbursed: ${total_reimbursed:,.2f}\n"
        f"- Top 3 Products by Total Amount Reimbursed:\n{top3_text}\n"
        f"Columns matched to your question: {', '.join(matched_cols)}\n\n"
        f"Answer the following question concisely:\n{question}"
    )

    messages = [
        {"role": "system", "content": "You are a helpful data analyst assistant."},
        {"role": "user", "content": context_text}
    ]

    return ask_openai_chat(messages)

# 📈 Chart visualizer
def generate_chart(df, question):
    top5 = df.groupby("Product Name")["Total Amount Reimbursed"].sum().sort_values(ascending=False).head(5)
    fig, ax = plt.subplots(figsize=(8, 4))
    top5.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("💵 Top 5 Products by Total Amount Reimbursed")
    ax.set_ylabel("Amount ($)")
    ax.set_xlabel("Product")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# 🔍 Ask and Answer section
st.subheader("❓ Ask a question about the dataset")
question = st.text_input("💬 Enter your question here:")

col1, col2 = st.columns(2)

with col1:
    if st.button("📄 Get Text Answer"):
        if not question.strip():
            st.warning("⚠️ Please enter a question.")
        else:
            with st.spinner("Thinking like a data analyst..."):
                answer = generate_text_answer(df, question)
                st.markdown(
                    f"""
                    <div style="border: 1px solid #ccc; background-color: #f9f9f9; padding: 1rem; border-radius: 8px;">
                    <strong>📌 Answer:</strong><br><br>
                    {answer.replace('\n', '<br>')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

with col2:
    if st.button("📊 Create Chart"):
        if not question.strip():
            st.warning("⚠️ Please enter a question to create a chart.")
        else:
            with st.spinner("Creating visual..."):
                generate_chart(df, question)
