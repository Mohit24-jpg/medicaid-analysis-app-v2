import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt

# Initialize OpenAI client with new SDK syntax
client = openai.OpenAI()

st.title("Natural Language CSV Data Analyzer")

# Upload CSV file
uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV into DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.write("### Sample Data")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    question = st.text_input("Ask a question about your data")

    col1, col2 = st.columns(2)

    with col1:
        text_btn = st.button("Get Text Answer")

    with col2:
        chart_btn = st.button("Get Chart")

    def df_to_text_summary(df):
        # Basic summary info for context
        summary = f"Data has {df.shape[0]} rows and {df.shape[1]} columns. Columns are: {', '.join(df.columns)}."
        return summary

    def create_prompt(question, df):
        # We give the model a summary of data columns and few sample rows, plus the question
        sample_rows = df.head(5).to_csv(index=False)
        prompt = (
            f"You are a data analyst assistant.\n"
            f"The dataset columns are: {', '.join(df.columns)}.\n"
            f"Here are 5 sample rows:\n{sample_rows}\n\n"
            f"Answer the question based on this data:\n{question}\n"
            f"If you can't answer, say 'Sorry, I don't see that information in the data.'"
        )
        return prompt

    def ask_openai_chat(prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant for data analysis."},
            {"role": "user", "content": prompt},
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    if text_btn:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            prompt = create_prompt(question, df)
            with st.spinner("Getting answer from OpenAI..."):
                answer = ask_openai_chat(prompt)
            st.markdown("### Answer")
            st.write(answer)

    if chart_btn:
        if not question.strip():
            st.warning("Please enter a question first.")
        else:
            # For charts, ask the model what columns to plot (expect a JSON with x and y columns)
            prompt_chart = (
                f"You are a data analyst assistant.\n"
                f"Based on this dataset columns: {', '.join(df.columns)}\n"
                f"Suggest the best columns to plot as x and y axis to answer this question:\n{question}\n"
                f"Reply ONLY in JSON format like {{'x':'column_name', 'y':'column_name', 'kind':'bar' or 'line'}}\n"
                f"If no good plot possible, reply with {{}}"
            )
            with st.spinner("Asking OpenAI for chart suggestion..."):
                chart_resp = ask_openai_chat(prompt_chart)

            import json
            try:
                chart_info = json.loads(chart_resp.replace("'", '"'))
            except Exception:
                chart_info = {}

            if not chart_info:
                st.info("Sorry, no suitable chart could be suggested.")
            else:
                x_col = chart_info.get("x")
                y_col = chart_info.get("y")
                kind = chart_info.get("kind", "bar")

                if x_col not in df.columns or y_col not in df.columns:
                    st.error("Suggested columns not found in data.")
                else:
                    st.markdown(f"### Chart: {kind.title()} plot of {y_col} vs {x_col}")
                    fig, ax = plt.subplots()
                    if kind == "bar":
                        df.plot(kind="bar", x=x_col, y=y_col, ax=ax)
                    elif kind == "line":
                        df.plot(kind="line", x=x_col, y=y_col, ax=ax)
                    else:
                        st.error(f"Unsupported chart kind: {kind}")
                        st.stop()
                    st.pyplot(fig)
