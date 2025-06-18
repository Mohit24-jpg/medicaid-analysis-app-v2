import streamlit as st
import pandas as pd
from fuzzywuzzy import process

st.title("CSV Chatbot — Ask Questions About Your Data")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def match_column(query_terms, columns):
    # Returns best matching column in columns for any of the query_terms using fuzzy matching
    for term in query_terms:
        match, score = process.extractOne(term, columns)
        if score > 70:  # Threshold can be tuned
            return match
    return None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data")

    if st.button("Get Answer") and question:
        cols = df.columns.tolist()

        # Example: detect keywords to columns
        # This can be expanded or replaced with embedding similarity + LLM for better understanding
        drug_col = match_column(["drug", "medication", "medicine", "name"], cols)
        cost_col = match_column(["cost", "amount", "price", "reimbursed"], cols)
        count_col = match_column(["count", "prescription", "units", "quantity"], cols)

        # Simple intent detection examples:
        question_lower = question.lower()

        try:
            if "top" in question_lower and "cost" in question_lower and drug_col and cost_col:
                # Find top N drugs by cost
                import re
                n = int(re.search(r"top (\d+)", question_lower).group(1)) if re.search(r"top (\d+)", question_lower) else 3
                result = df.groupby(drug_col)[cost_col].sum().sort_values(ascending=False).head(n)
                st.write(f"Top {n} {drug_col} by total {cost_col}:")
                st.table(result)

            elif ("total" in question_lower or "sum" in question_lower) and cost_col:
                total = df[cost_col].sum()
                st.write(f"Total {cost_col}: {total}")

            elif ("unique" in question_lower or "distinct" in question_lower) and drug_col:
                unique_count = df[drug_col].nunique()
                st.write(f"Number of unique {drug_col}: {unique_count}")

            else:
                st.write("Sorry, I couldn't understand the question or find matching data columns.")
        except Exception as e:
            st.error(f"Error processing your question: {e}")

else:
    st.info("Please upload a CSV file to start.")
