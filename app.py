import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import openai
import io

# Load API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="AI Medicaid CSV Analyzer", layout="wide")
st.title("📊 AI-Powered Medicaid CSV Analyzer")
st.markdown("Upload a CSV file and ask any question about the data. Use the buttons to get an answer or generate a chart.")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

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

    def generate_text_answer(df, question):
        try:
            # Removed 'datetime_is_numeric' for compatibility
            stats = df.describe(include="all").fillna("").to_string()
            messages = [
                {"role": "system", "content": "You are a helpful data analyst. Answer the question based on the dataset summary without showing code."},
                {"role": "user", "content": f"Here is a summary of the dataset:\n{stats}"},
                {"role": "user", "content": f"Question: {question}"}
            ]
            return ask_openai(messages)
        except Exception as e:
            return f"Error analyzing the data: {str(e)}"

    def generate_chart_code(df, question):
        try:
            preview = df.head(20).to_csv(index=False)
            messages = [
                {"role": "system", "content": "You are a Python data visualization assistant. Generate Python code using matplotlib or pandas to create a chart for the following question."},
                {"role": "user", "content": f"Here is a sample of the dataset:\n{preview}"},
                {"role": "user", "content": f"Generate a matplotlib chart in Python to answer this question: {question}"}
            ]
            code = ask_openai(messages)
            return code
        except Exception as e:
            return f"Error generating chart: {str(e)}"

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
                    # Secure execution
                    local_vars = {'df': df, 'plt': plt}
                    exec(code, {"__builtins__": {}}, local_vars)
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Chart generation failed: {str(e)}")
                    st.code(code)

else:
    st.warning("Please upload a CSV file to get started.")
