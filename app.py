import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
from typing import List, Dict

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Smart CSV Analyst", layout="wide")
st.title("Smart CSV Analyst with OpenAI - Text or Chart")

@st.cache_data(show_spinner=True)
def load_data(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

@st.cache_data(show_spinner=True)
def ask_openai_chat(messages: List[Dict[str, str]], max_tokens=700, temp=0):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temp,
    )
    return response['choices'][0]['message']['content']

def generate_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None, hue_col: str = None):
    plt.figure(figsize=(10,6))
    sns.set_theme(style="whitegrid")

    if chart_type == 'bar':
        if y_col:
            sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col)
        else:
            sns.countplot(data=df, x=x_col, hue=hue_col)
    elif chart_type == 'line':
        if y_col:
            sns.lineplot(data=df, x=x_col, y=y_col, hue=hue_col)
        else:
            st.error("Line chart requires y-axis column.")
            return None
    elif chart_type == 'pie':
        if y_col:
            data = df.groupby(x_col)[y_col].sum()
            plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')
        else:
            st.error("Pie chart requires y-axis column.")
            return None
    else:
        st.error(f"Unsupported chart type: {chart_type}")
        return None

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Upload any CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.session_state.df = df

    st.write("### Dataset Preview (first 10 rows)")
    st.dataframe(df.head(10))

    st.write("### Columns detected:")
    st.write(list(df.columns))

    question = st.text_input("Ask your question about the data:")

    # Buttons for Text or Chart answer
    col1, col2 = st.columns(2)
    with col1:
        btn_text = st.button("Get Text Answer")
    with col2:
        btn_chart = st.button("Get Chart")

    if question:
        system_msg = (
            "You are a helpful data analyst assistant. "
            "The user has uploaded a CSV dataset with unknown columns and data. "
            "Answer their questions about this dataset. "
            "If asked for charts, specify chart type (bar, line, pie) and columns to plot in format: "
            "CHART:{chart_type}, X:{x_column}, Y:{y_column} (Y can be optional)."
        )

        sample_data = df.head(5).to_dict(orient='records')
        columns = list(df.columns)

        messages = [{"role": "system", "content": system_msg}]
        user_msg = (
            f"Dataset columns:\n{columns}\n\n"
            f"Sample rows:\n{sample_data}\n\n"
            f"User question:\n{question}\n"
        )
        messages.append({"role": "user", "content": user_msg})

        if btn_text:
            with st.spinner("Getting text answer..."):
                try:
                    answer = ask_openai_chat(messages)
                    # Remove any chart instructions if present
                    answer = re.sub(r"CHART:.*", "", answer, flags=re.IGNORECASE).strip()
                    st.markdown("### Text Answer:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"OpenAI API error: {e}")

        if btn_chart:
            with st.spinner("Getting chart answer..."):
                try:
                    answer = ask_openai_chat(messages)
                    st.markdown("### Model Answer (may include chart instructions):")
                    st.write(answer)

                    # Parse chart instructions
                    chart_pattern = r"CHART:(\w+),\s*X:(\w+)(?:,\s*Y:(\w+))?"
                    match = re.search(chart_pattern, answer, flags=re.IGNORECASE)

                    if match:
                        chart_type = match.group(1).lower()
                        x_col = match.group(2)
                        y_col = match.group(3) if match.group(3) else None

                        if x_col not in df.columns:
                            st.error(f"Column '{x_col}' not found in data.")
                        elif y_col and y_col not in df.columns:
                            st.error(f"Column '{y_col}' not found in data.")
                        else:
                            chart_img = generate_chart(df, chart_type, x_col, y_col)
                            if chart_img:
                                st.image(chart_img)
                    else:
                        st.info("No chart instructions detected in the answer. Try asking for a chart explicitly.")

                except Exception as e:
                    st.error(f"OpenAI API error: {e}")

else:
    st.info("Upload a CSV file to start analyzing your data using natural language.")
