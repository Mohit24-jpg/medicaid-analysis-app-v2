import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import requests
import io

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Medicaid Drug Spending NLP Analytics")

with st.sidebar:
    st.header("Filters")
    state = st.selectbox("Select state abbreviation", ["OH", "NY", "CA", "TX", "FL"], index=0)
    year = st.selectbox("Select year", ["2020", "2021", "2022", "2023", "2024"], index=4)

if not (state and year):
    st.info("Please select state and year to proceed.")
    st.stop()

# Load Medicaid API data
data_url = f"https://data.medicaid.gov/api/views/61729e5a-7aa8-448c-8903-ba3e0cd0ea3c/rows.csv?accessType=DOWNLOAD&bom=true&format=true&delimiter=%2C"

@st.cache_data

def load_data():
    response = requests.get(data_url)
    if response.status_code == 200:
        df = pd.read_csv(io.StringIO(response.text))
        return df
    else:
        st.error("Failed to load Medicaid data from API.")
        return None

df = load_data()
if df is None:
    st.stop()

# Filter data
filtered_df = df[(df['state'] == state.upper()) & (df['year'] == int(year))]
if filtered_df.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

st.subheader(f"Data preview: {state.upper()} {year}")
st.dataframe(filtered_df.head(10))

# GPT Interface
def ask_openai(messages):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()

def generate_text_answer(df, question):
    sample_data = df.sample(n=min(100, len(df))).to_dict(orient="records")
    messages = [
        {"role": "system", "content": "You are a data analyst. Answer only based on the data provided."},
        {"role": "user", "content": f"The dataset has the following columns:\n{', '.join(df.columns)}"},
        {"role": "user", "content": f"Here is a sample of the dataset:\n{sample_data}"},
        {"role": "user", "content": f"Answer concisely: {question}"}
    ]
    return ask_openai(messages)

def generate_chart(df, question):
    sample_data = df.sample(n=min(100, len(df))).to_dict(orient="records")
    messages = [
        {"role": "system", "content": "You're a Python data visualization assistant. Use matplotlib and pandas. Don't explain, only give code."},
        {"role": "user", "content": f"Sample data:\n{sample_data}"},
        {"role": "user", "content": f"Write Python code to create a chart answering: {question}"}
    ]

    code = ask_openai(messages)

    try:
        local_vars = {"df": df, "plt": plt}
        exec(code, {}, local_vars)
        fig = plt.gcf()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to create chart: {e}")
        st.code(code)

# Question Input and Buttons
st.subheader("Ask a question about the filtered dataset")
question = st.text_input("Enter your question")

col1, col2 = st.columns(2)

with col1:
    if st.button("Get Text Answer"):
        if question:
            with st.spinner("Analyzing..."):
                answer = generate_text_answer(filtered_df, question)
                st.markdown(f"**Answer:** {answer}")
        else:
            st.warning("Enter a question first.")

with col2:
    if st.button("Create Chart"):
        if question:
            with st.spinner("Generating chart..."):
                generate_chart(filtered_df, question)
        else:
            st.warning("Enter a question to generate a chart.")
