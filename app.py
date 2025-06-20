import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Set up API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🧠 AI-Powered CSV Analysis App")

# Upload CSV file
if "df" not in st.session_state:
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("CSV uploaded and loaded successfully.")
else:
    df = st.session_state["df"]

if "df" in st.session_state:
    st.subheader("Preview of your data")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data")

    def ask_gpt(question, df, mode="text"):
        sample_data = df.sample(min(500, len(df))).to_csv(index=False)
        prompt = f"""
You are a smart data analyst.

Here is a preview of the uploaded CSV (first 500 rows):

{sample_data}

The user asked:
"{question}"

If mode is 'text': Provide only a concise, relevant answer from the data.
If mode is 'chart': Provide only valid Python code that defines a function `generate_chart(df)` using matplotlib or seaborn to visualize the answer.
Do not explain the code, just output the code only.

Mode: {mode}
"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    if question:
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💬 Get Answer"):
                with st.spinner("Thinking..."):
                    try:
                        answer = ask_gpt(question, df, mode="text")
                        st.markdown(f"**Answer:** {answer}")
                    except Exception as e:
                        st.error(f"Failed to generate answer: {e}")

        with col2:
            if st.button("📊 Generate Chart"):
                with st.spinner("Generating chart..."):
                    try:
                        code = ask_gpt(question, df, mode="chart")
                        st.code(code, language="python")

                        # Execute chart code safely
                        local_vars = {}
                        exec(code, {"df": df, "plt": plt, "sns": sns}, local_vars)
                        chart_func = local_vars.get("generate_chart")

                        if chart_func:
                            chart_func(df)
                            st.pyplot()
                        else:
                            st.warning("No chart function generated.")
                    except Exception as e:
                        st.error(f"Chart execution failed: {e}")
