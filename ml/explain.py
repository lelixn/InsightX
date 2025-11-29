import shap
import matplotlib.pyplot as plt

def explain_model(model, X_sample):
    """
    Generate a SHAP summary plot for a trained model.
    Only call this from the Streamlit front-end.
    """
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        fig = plt.figure(figsize=(10, 5))
        shap.summary_plot(shap_values, X_sample, show=False)
        return fig

    except Exception as e:
        fig = plt.figure(figsize=(6, 4))
        plt.text(0.1, 0.5, f"SHAP Explanation Error:\n{str(e)}", fontsize=12)
        return fig
