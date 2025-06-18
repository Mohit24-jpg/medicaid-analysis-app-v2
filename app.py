import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt

# Load API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Data Analyst GPT", layout="wide")
st.title("📊 Smart CSV Data Analyzer")
st.write("Upload any CSV and ask questions — choose a text response or a chart.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    return pd.read_csv(file)

if uploaded_file:
    df = load_data(uploaded_file)

    # Preview
    st.subheader("🔍 Data Preview (first 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

    # User question
    question = st.text_input("Ask a question about this data:")

    # Answer generator
    def generate_answer(df: pd.DataFrame, question: str) -> str:
        # Use schema summary, not sample rows (more reliable for full dataset)
        col_info = "\n".join(
            [f"- {col}: {str(dtype)}" for col, dtype in df.dtypes.items()]
        )
        stats = df.describe(include="all", datetime_is_numeric=True).fillna("").to_string()

        prompt = f"""
You are a data analyst. Given a dataset, answer questions clearly, concisely, and only using the data.

Dataset column summary:
{col_info}

Basic stats:
{stats}

Now answer this question (just the answer, no code or reasoning):

Question: {question}
"""

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    # Button for text answer
    if st.button("🧠 Text Answer") and question:
        with st.spinner("Analyzing..."):
            result = generate_answer(df, question)
        st.markdown(f"**Answer:** {result}")

    # Button for chart
    if st.button("📈 Show Chart"):
        st.subheader("Chart Builder")

        # Automatically find usable columns
        num_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()

        if not num_cols or not cat_cols:
            st.warning("Need at least one numeric and one categorical column.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                cat_col = st.selectbox("Categorical Column", cat_cols)
            with col2:
                num_col = st.selectbox("Numeric Column", num_cols)

            chart_data = (
                df.groupby(cat_col)[num_col]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            fig, ax = plt.subplots()
            chart_data.plot(kind="bar", ax=ax, legend=False)
            ax.set_ylabel(num_col)
            ax.set_title(f"{num_col} by {cat_col}")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
