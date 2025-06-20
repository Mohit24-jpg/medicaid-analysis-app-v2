import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Medicaid Drug Spending NLP Analytics")

with st.sidebar:
    st.header("Filters & Data Upload")
    state = st.text_input("Enter state abbreviation (e.g. OH)", max_chars=2)
    year = st.text_input("Enter year (e.g. 2024)", max_chars=4)
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if not (state and year and uploaded_file):
    st.info("Please enter state abbreviation, year, and upload a CSV file to proceed.")
    st.stop()

@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

df = load_data(uploaded_file)
if df is None:
    st.stop()

st.subheader(f"Data preview: {state.upper()} {year}")
st.dataframe(df.head(10))

# ---------------- GPT Logic ----------------

def ask_openai(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_text_answer(df, question):
    # Sample raw data if too large
    sample_data = df.sample(n=min(100, len(df))).to_dict(orient="records")

    messages = [
        {"role": "system", "content": "You are a data analyst. Answer only based on the data provided."},
        {"role": "user", "content": f"The dataset has the following columns:\n{', '.join(df.columns)}"},
        {"role": "user", "content": f"Here is a sample of the dataset:\n{sample_data}"},
        {"role": "user", "content": f"Answer concisely: {question}"}
    ]
    return ask_openai(messages)

def generate_chart(df, question):
    # Let GPT decide which chart to create
    sample_data = df.sample(n=min(100, len(df))).to_dict(orient="records")
    messages = [
        {"role": "system", "content": "You're a Python data visualization assistant. Use matplotlib and pandas. Don't explain, only give code."},
        {"role": "user", "content": f"Sample data:\n{sample_data}"},
        {"role": "user", "content": f"Write Python code to create a chart answering: {question}"}
    ]

    code = ask_openai(messages)

    try:
        # Execute the charting code in a safe context
        local_vars = {"df": df, "plt": plt}
        exec(code, {}, local_vars)
        fig = plt.gcf()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to create chart: {e}")
        st.code(code)

# ---------------- UI ----------------

st.subheader("Ask a question about your dataset")
question = st.text_input("Enter your question")

col1, col2 = st.columns(2)

with col1:
    if st.button("Get Text Answer"):
        if question:
            with st.spinner("Analyzing..."):
                answer = generate_text_answer(df, question)
                st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Enter a question first.")

with col2:
    if st.button("Create Chart"):
        if question:
            with st.spinner("Generating chart..."):
                generate_chart(df, question)
        else:
            st.warning("Enter a question to generate a chart.")
