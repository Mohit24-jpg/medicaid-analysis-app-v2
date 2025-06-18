import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import io

# Initialize OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🩺 Medicaid Data NLP Analysis")

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return None

def create_prompt(question, df):
    prompt = (
        f"You are a precise data analyst assistant.\n"
        f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
        f"Columns: {', '.join(df.columns)}.\n"
        f"Answer the question below concisely and directly, no explanation unless asked:\n\n"
        f"Question: {question}\n"
        f"If you cannot answer based on the columns, say 'Sorry, that information is not available.'"
    )
    return prompt

def ask_openai_chat(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300,
    )
    return response.choices[0].message.content.strip()

def generate_chart(df, question):
    # Simple heuristic chart generation based on question keywords
    # You can improve this with better NLP or more logic
    question = question.lower()
    if "top" in question and ("drug" in question or "product" in question):
        # For example: top drugs by reimbursed amount or prescriptions
        if "reimbursed" in question and "amount" in question:
            if "amount reimbursed" in df.columns and "drug name" in df.columns:
                grouped = df.groupby("drug name")["amount reimbursed"].sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots()
                grouped.plot(kind="bar", ax=ax)
                ax.set_title("Top 10 Drugs by Amount Reimbursed")
                ax.set_ylabel("Amount Reimbursed")
                ax.set_xlabel("Drug Name")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                return fig
        if "number of prescriptions" in question or "prescriptions" in question:
            if "drug name" in df.columns and "number of prescriptions" in df.columns:
                grouped = df.groupby("drug name")["number of prescriptions"].sum().sort_values(ascending=False).head(10)
                fig, ax = plt.subplots()
                grouped.plot(kind="bar", ax=ax)
                ax.set_title("Top 10 Drugs by Number of Prescriptions")
                ax.set_ylabel("Number of Prescriptions")
                ax.set_xlabel("Drug Name")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                return fig

    # Default fallback chart: show distribution of numeric columns count
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        col = numeric_cols[0]
        fig, ax = plt.subplots()
        df[col].hist(ax=ax, bins=20)
        ax.set_title(f"Histogram of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        return fig

    return None

# --- Streamlit UI ---

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file)

    st.write(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    question = st.text_input("Ask a question about your data:")

    if question:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Get Text Answer"):
                with st.spinner("Getting answer..."):
                    prompt = create_prompt(question, df)
                    answer = ask_openai_chat(prompt)
                    st.markdown(f"**Answer:** {answer}")

        with col2:
            if st.button("Get Chart"):
                with st.spinner("Generating chart..."):
                    fig = generate_chart(df, question)
                    if fig:
                        st.pyplot(fig)
                    else:
                        st.info("Sorry, I couldn't generate a chart for that question.")

else:
    st.info("Please upload a CSV file to get started.")
