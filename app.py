import streamlit as st
import pandas as pd
import requests
import openai

st.title("Medicaid Drug Spending NLP Analytics")

# Set API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

@st.cache_data
def load_data():
    url = "https://data.medicaid.gov/resource/srj6-uykx.json?$limit=50000"
    resp = requests.get(url)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())

df = load_data()

st.success(f"Loaded {len(df)} rows from Medicaid API.")

st.dataframe(df.head())

# NLP Question
question = st.text_input("Ask a question about the data")

def ask_openai(question, df):
    sample = df.head(50).to_csv(index=False)
    prompt = f"""You are a data analyst. Answer concisely.
    
This is a sample of a Medicaid dataset:
{sample}

Now answer this question using the data: {question}
Only answer with facts that can be inferred from the dataset.
"""

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

if question:
    with st.spinner("Analyzing with OpenAI..."):
        answer = ask_openai(question, df)
        st.markdown("### Answer:")
        st.write(answer)
