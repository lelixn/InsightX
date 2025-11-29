import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocess import load_and_preprocess
from ml.train import get_models
from ml.evaluate import evaluate_models
from ml.explain import explain_model
from ml.insights import generate_insights
from ml.pdfgen import generate_pdf


st.set_page_config(page_title="InsightX Dashboard", layout="wide")

st.title("📊 InsightX – ML Visualization Dashboard")
st.write("Upload a dataset → Explore → Train ML models → Get Insights")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
    st.dataframe(df.head())
else:
    st.warning("Please upload a CSV file to begin.")
    st.stop()

st.subheader("🔎 Exploratory Data Analysis")

# Show basic stats
st.write("### Dataset Info")
st.write(df.describe())

# Correlation heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)


st.subheader("📈 Interactive Visualizations")

numeric_cols = df.select_dtypes(include=['int64','float64']).columns

if len(numeric_cols) >= 2:
    x = st.selectbox("Select X-axis", numeric_cols)
    y = st.selectbox("Select Y-axis", numeric_cols)

    fig = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
    st.plotly_chart(fig, use_container_width=True)



st.subheader("🤖 Train ML Models")

if st.button("Run ML Pipeline"):
    st.info("Training models...")

    (X_train, X_test, y_train, y_test), le = load_and_preprocess(uploaded_file)

    models = get_models()
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    st.write("### 📊 Model Performance")
    st.dataframe(results)

    # save best
    best_name = results.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
    best_model = models[best_name]

    joblib.dump(best_model, "models/best_streamlit_model.pkl")
    st.success(f"Best model saved: {best_name}")


st.subheader("🔮 Make Predictions")

if st.button("Predict on New Sample"):
    sample = X_test.iloc[0:1]  # example
    pred = best_model.predict(sample)[0]
    label = le.inverse_transform([pred])[0]

    st.info(f"Prediction: **{label}**")
    st.write("Input Features:")
    st.dataframe(sample)


    tab1, tab2, tab3, tab4 = st.tabs(["EDA", "Visualizations", "ML Models", "Explainability"])
    with tab1:
        st.subheader("🔎 Exploratory Data Analysis")
        st.write(df.describe())

    with tab2:
        st.subheader("📈 Interactive Visualizations")
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=x, y=y, title=f"{x} vs {y}")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("🤖 Trained ML Models")
        st.dataframe(results)


    with tab4:
        st.subheader("🔍 Model Explainability (SHAP)")

    if "best_model" not in st.session_state:
        st.warning("Train a model first in the ML Models tab.")
    else:
        model = st.session_state["best_model"]
        
        # random sample from training
        sample = X_test.iloc[:100]  # first 100 rows

        st.info("Generating SHAP Summary Plot...")

        from ml.explain import explain_model
        fig = explain_model(model, sample)

        st.pyplot(fig)
        st.success("SHAP plot generated.")

        from ml.pdfgen import generate_pdf

if st.button("Download Report"):
    path = generate_pdf(auto_report)
    with open(path, "rb") as f:
        st.download_button("⬇ Download PDF", data=f, file_name="InsightX_Report.pdf")
