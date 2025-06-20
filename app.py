import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import io

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="AI Medicaid CSV Analyzer", layout="wide")
st.title("📊 AI-Powered Medicaid CSV Analyzer")

uploaded_file = st.file_uploader("Upload your Medicaid CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(50))

    question = st.text_input("Ask a question about your dataset:")

    def ask_openai(messages):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def generate_summary_preview(df):
        preview = df.head(200)
        text = preview.to_csv(index=False)
        return text

    def generate_text_answer(df, question):
        try:
            preview_csv = generate_summary_preview(df)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful data analyst. Answer the user's question based on the uploaded dataset. Be concise and do not show code unless specifically asked."
                },
                {"role": "user", "content": f"Here is a sample of the dataset (first 200 rows):\n\n{preview_csv}"},
                {"role": "user", "content": f"Question: {question}"}
            ]
            return ask_openai(messages)
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def generate_chart_code(df, question):
        preview = df.head(200).to_csv(index=False)
        messages = [
            {"role": "system", "content": "You are a Python chart assistant. Generate a matplotlib or pandas chart to answer the question using the sample data."},
            {"role": "user", "content": f"Here is a sample of the dataset:\n{preview}"},
            {"role": "user", "content": f"Question: {question}"}
        ]
        return ask_openai(messages)

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🧠 Get Answer"):
            with st.spinner("Thinking..."):
                result = generate_text_answer(df, question)
            st.success("Answer:")
            st.write(result)

    with col2:
        if st.button("📈 Generate Chart"):
            with st.spinner("Generating chart..."):
                code = generate_chart_code(df, question)
                try:
                    local_vars = {"df": df, "plt": plt}
                    exec(code, {"__builtins__": {}}, local_vars)
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Chart failed: {str(e)}")
                    st.code(code)

else:
    st.info("Please upload a CSV file to begin.")
