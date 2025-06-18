import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("NLP Data Analysis on Any CSV")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

def ask_openai_chat(messages, max_tokens=700, temp=0):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
    )
    return response.choices[0].message.content.strip()

def generate_answer(df, question):
    # Send the question + column names for context to OpenAI
    columns = df.columns.tolist()
    system_msg = f"You are a helpful assistant analyzing a pandas DataFrame with columns: {columns}. "\
                 "Answer the user question based on this data."
    user_msg = f"Data columns: {columns}\nUser question: {question}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    return ask_openai_chat(messages)

def generate_chart_code(df, question):
    # Ask OpenAI to generate matplotlib python code snippet to plot chart answering question
    columns = df.columns.tolist()
    system_msg = f"You are a helpful assistant. Given a pandas DataFrame with columns: {columns}, "\
                 "write python matplotlib code to plot a chart answering the user question. "\
                 "Only return python code inside a code block. Do not include any explanations."
    user_msg = f"Data columns: {columns}\nUser question: {question}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    code = ask_openai_chat(messages)
    return code

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    question = st.text_input("Ask a question about your data")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Text Answer"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(df, question)
                    st.write("Answer:")
                    st.write(answer)

    with col2:
        if st.button("Chart"):
            if question.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Generating chart..."):
                    code = generate_chart_code(df, question)
                    st.code(code, language="python")
                    # Execute generated code safely
                    try:
                        # Define local environment with df and plt for exec
                        local_env = {"df": df, "plt": plt}
                        exec(code, {}, local_env)
                        st.pyplot()
                    except Exception as e:
                        st.error(f"Failed to execute chart code: {e}")
else:
    st.info("Please upload a CSV file to get started.")
