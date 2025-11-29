import shap
import matplotlib.pyplot as plt

def explain_model(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    
    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values, X_sample, show=False)
    return fig

from ml.insights import generate_insights
auto_report = generate_insights(df, results)

st.text_area("📜 AI-Generated Insights", auto_report, height=200)
