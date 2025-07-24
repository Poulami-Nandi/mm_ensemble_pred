import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def explain_with_shap(model, X, model_name="Model"):
    """
    Generate SHAP summary plot and bar plot for model interpretability.
    
    Args:
        model: Trained model (must be tree-based like XGBoost or support SHAP)
        X: Input features (DataFrame)
        model_name: Label for the model being explained
    """
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
    except Exception as e:
        st.warning(f"SHAP explainer could not be initialized: {e}")
        return

    st.subheader(f"🔍 SHAP Summary Plot for {model_name}")
    fig_summary = plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot(fig_summary)

    st.subheader(f"📊 SHAP Feature Importance (mean |SHAP value|)")
    fig_bar = plt.figure()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig_bar)

    st.caption("These plots help you understand which features contributed most to the model's predictions.")

def explain_instance(model, X, instance_index=0, model_name="Model"):
    """
    Explain prediction for a single instance.

    Args:
        model: Trained model
        X: Feature matrix (DataFrame)
        instance_index: Index of row to explain
        model_name: Name of model
    """
    st.subheader(f"🔍 SHAP Explanation for One Prediction - {model_name}")
    try:
        explainer = shap.Explainer(model, X)
        shap_value = explainer(X.iloc[instance_index:instance_index+1])
        fig = shap.plots.waterfall(shap_value[0], show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"SHAP instance explanation failed: {e}")
