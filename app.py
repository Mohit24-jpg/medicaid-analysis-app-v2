import streamlit as st
import pandas as pd
import openai

# Load your OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Medicaid Drug Dataset - GPT Query Assistant")

@st.cache_data
def load_data():
    return pd.read_csv("SDUD-2024.csv")

df = load_data()

st.write("Dataset preview:")
st.dataframe(df.head())

user_question = st.text_input("Ask a question about the dataset:")

def ask_gpt(question, columns):
    prompt = f"""
You are a Python data analyst assistant. You have a pandas DataFrame named 'df' with columns: {', '.join(columns)}.
Write Python pandas code to answer the user's question based on this DataFrame.
Assign the final answer to a variable named 'result'.

User question: {question}

Only output the Python code. No explanations.
"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=250,
        temperature=0,
        stop=None,
    )
    return response.choices[0].text.strip()

if user_question:
    with st.spinner("Generating answer..."):
        code = ask_gpt(user_question, df.columns.tolist())
        st.code(code, language="python")

        # Safe execution environment
        local_env = {"df": df, "pd": pd}

        try:
            exec(code, {}, local_env)
            if "result" in local_env:
                st.write("Answer:")
                res = local_env["result"]
                # Show nicely if it's a DataFrame or Series
                if isinstance(res, (pd.DataFrame, pd.Series)):
                    st.dataframe(res)
                else:
                    st.write(res)
            else:
                st.error("The code did not define a variable named 'result'.")
        except Exception as e:
            st.error(f"Error executing code: {e}")
