import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import tempfile

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("📊 AI-Powered CSV Analysis App")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    # Load and cache the entire dataset
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    df = load_data(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(10))

    question = st.text_input("Ask a question about this dataset:")

    col1, col2 = st.columns(2)

    def generate_text_answer(df: pd.DataFrame, question: str) -> str:
        # Use pandas to answer basic structured questions
        if "top" in question and "reimbursed" in question:
            cols = df.columns.str.lower()
            prod_col = next((col for col in df.columns if "product" in col.lower()), None)
            cost_col = next((col for col in df.columns if "reimbursed" in col.lower()), None)

            if prod_col and cost_col:
                top_items = (
                    df.groupby(prod_col)[cost_col]
                    .sum()
                    .sort_values(ascending=False)
                    .head(3)
                    .reset_index()
                )
                formatted = "\n".join(
                    f"{row[prod_col]} — ${row[cost_col]:,.2f}"
                    for _, row in top_items.iterrows()
                )
                return f"Top 3 most reimbursed products:\n{formatted}"

        # Otherwise fallback to GPT on sample
        sample = df.head(300).to_csv(index=False)
        prompt = f"""
The user uploaded this dataset (first 300 rows shown):

{sample}

Answer this question directly and concisely:

{question}
"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def generate_chart(df: pd.DataFrame, question: str):
        sample = df.head(300).to_csv(index=False)
        prompt = f"""
Given this CSV sample (first 300 rows):

{sample}

Generate Python matplotlib code to answer the question: "{question}".
Only include the code. Do not print explanation or use plt.show().
Assume necessary libraries are already imported.
"""
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        code = response.choices[0].message.content

        try:
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as temp_py:
                temp_py.write("import matplotlib.pyplot as plt\n")
                temp_py.write("import pandas as pd\n")
                temp_py.write("from io import StringIO\n")
                temp_py.write(f"df = pd.read_csv(StringIO('''{sample}'''))\n")
                temp_py.write(code + "\nplt.tight_layout()\n")
                temp_path = temp_py.name

            exec_globals = {}
            with open(temp_path) as f:
                code_contents = f.read()
                exec(code_contents, exec_globals)

            st.pyplot(plt)

        except Exception as e:
            st.error(f"Chart generation failed: {e}")

    if question:
        if col1.button("Get Text Answer"):
            answer = generate_text_answer(df, question)
            st.markdown(f"**Answer:** {answer}")

        if col2.button("Get Chart"):
            generate_chart(df, question)
