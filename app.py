import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
models = client.models.list()

for m in models.data:
    print(m.id)
st.set_page_config(layout="wide")
st.markdown("##  AI Data Analyst Agent")
st.markdown("---")

st.write("Welcome! Upload your dataset to begin.")

# File Upload 

uploaded_file = st.file_uploader("Upload your CSV File ", type=["csv"])

# If file is uploaded 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = df.convert_dtypes()
    # 🔥 ADD THIS LINE (important)
    df = df.astype(str).replace('<NA>', None)

   # Convert numeric columns safely
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # Dataset Insights 

    col1, col2 , col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing values", df.isnull().sum().sum())

    # Data types 
    st.subheader("Column Data Types")
    st.write(df.dtypes)

    # Charts 
    st.subheader("Data Visualization")

    # selecting numeric columns 
    numeric_col = df.select_dtypes(include=['int64','float64']).columns

    if len(numeric_col) > 0 :
        selected_col = st.selectbox("Select column for histogram", numeric_col)

        st.write(f"Histogram of {selected_col}")

        fig, ax = plt.subplots()
        ax.hist(df[selected_col].dropna(), bins=30)
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")

        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        st.subheader("Ask Questions About your Data 🤖")
        st.subheader("📊 Auto Insights Generator")

if st.button("Generate Insights 🚀"):

        insights_prompt = f"""
    You are a data analyst.

    ONLY use the dataset provided.

    Give:
    - 5–8 key insights
    - Include numbers/percentages
    - Highlight trends and correlations

    DO NOT give business recommendations.

    Dataset:
    {df.head(30).to_string()}
    """

        response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": insights_prompt}]
    )

        st.write(response.choices[0].message.content)

        st.subheader("💡 Business Recommendations")

        if st.button("Get Recommendations 📈"):

         rec_prompt = f"""
You are a business analyst.

Based on this dataset, give:
- 3 actionable recommendations
- 2 pricing strategies
- 2 risks

Be practical and business-focused.

Dataset:
{df.head(30).to_string()}
"""

         response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": rec_prompt}]
    )

        st.write(response.choices[0].message.content)

user_question = st.text_input("Ask anything about your dataset:")

if user_question:
        prompt = f"""
    You are a data analyst.

    ONLY use the dataset provided below.
    DO NOT suggest generic ML methods or code.

    Give clear, business-style insights based on actual data.

    Dataset Columns:
    {list(df.columns)}

    Sample Data:
    {df.head(20).to_string()}

    Question: {user_question}

    Answer in simple bullet points with real observations.
    """
        response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
)

        st.write(response.choices[0].message.content)
  
            
