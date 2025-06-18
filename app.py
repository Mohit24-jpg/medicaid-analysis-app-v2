import streamlit as st
import pandas as pd
from fuzzywuzzy import process
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Medicaid Dataset NLP Analyzer")

uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"])

def find_best_numeric_column(df, question):
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        return None
    best_match, score = process.extractOne(question.lower(), [col.lower() for col in numeric_cols])
    if score > 80:
        # Return original case-sensitive column name matching the lowercase best_match
        for col in numeric_cols:
            if col.lower() == best_match:
                return col
    return None

def generate_answer(df, question):
    # Normalize columns to lowercase stripped for matching
    df.columns = df.columns.str.lower().str.strip()
    question_lower = question.lower()

    # Try to find a numeric column that best matches question keywords
    matched_col = find_best_numeric_column(df, question_lower)

    if matched_col and ('total' in question_lower or 'sum' in question_lower or 'amount' in question_lower):
        # Provide the real sum answer from full dataset
        total = df[matched_col].sum()
        return f"The total of '{matched_col}' is: ${total:,.2f}"

    # If no numeric column match or question not about totals, fallback to GPT

    # Sample up to 300 rows for prompt size limits
    sample_df = df.sample(min(300, len(df))).copy()
    sample_csv = sample_df.to_csv(index=False)

    prompt = f"""You are a data analyst. Given the following CSV data preview, answer the user's question briefly and exactly without extra explanation.

CSV Data Preview:
{sample_csv}

User Question: {question}
Answer:"""

    messages = [
        {"role": "system", "content": "You are a helpful data analyst assistant."},
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=300,
    )
    answer = response.choices[0].message.content.strip()
    return answer

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    question = st.text_input("Ask a question about your dataset")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Get Text Answer"):
            if not question:
                st.warning("Please enter a question first.")
            else:
                with st.spinner("Analyzing..."):
                    answer = generate_answer(df, question)
                st.success("Answer:")
                st.write(answer)
    with col2:
        if st.button("Get Chart (Coming Soon)"):
            st.info("Chart generation will be implemented soon.")

else:
    st.info("Please upload a CSV file to start.")
