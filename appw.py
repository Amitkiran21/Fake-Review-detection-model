import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Required for joblib to unpickle your custom Naive Bayes model
class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / X.shape[0])
            feature_counts = np.asarray(X_c.sum(axis=0)).ravel()
            smoothed_fc = feature_counts + self.alpha
            smoothed_total = smoothed_fc.sum()
            self.feature_log_prob_[idx] = np.log(smoothed_fc) - np.log(smoothed_total)

        return self

    def predict(self, X):
        joint_log_likelihood = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(joint_log_likelihood, axis=1)]


st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("Fake Product Review Detection — Model Comparison")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    tfidf_path = os.path.join(BASE_DIR, "tfidf.joblib")
    models_path = os.path.join(BASE_DIR, "models.joblib")
    results_path = os.path.join(BASE_DIR, "comparison_results.csv")

    missing = [p for p in (tfidf_path, models_path, results_path) if not os.path.exists(p)]
    if missing:
        return None, None, None, [os.path.basename(p) for p in missing]

    tfidf = joblib.load(tfidf_path)
    models = joblib.load(models_path)
    results_df = pd.read_csv(results_path)
    return tfidf, models, results_df, []

tfidf, models, results_df, missing_files = load_artifacts()

if missing_files:
    st.error(f"Missing required files in `{BASE_DIR}`: {', '.join(missing_files)}")
    st.info("Export `tfidf.joblib`, `models.joblib`, and `comparison_results.csv` from your notebook into this folder, then press R to reload.")
    st.stop()

# 1. Performance Table
st.subheader("1. Model Performance Metrics")
st.dataframe(results_df, use_container_width=True, hide_index=True)

# 2. Charts (Uses saved PNGs if present, otherwise renders a native chart)
st.subheader("2. Visual Comparison")
f1_img = os.path.join(BASE_DIR, "f1_comparison.png")
cm_img = os.path.join(BASE_DIR, "confusion_matrices.png")

if os.path.exists(f1_img) and os.path.exists(cm_img):
    col1, col2 = st.columns(2)
    with col1:
        st.image(f1_img, caption="F1-Score Comparison")
    with col2:
        st.image(cm_img, caption="Confusion Matrices (Test Set)")
else:
    metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1-Score"] if c in results_df.columns]
    st.bar_chart(results_df.set_index("Model")[metric_cols])

# 3. Live Prediction
st.subheader("3. Test All Models Live")
user_review = st.text_area(
    "Enter a product review:",
    placeholder="Best product ever!!! Highly recommend buying this right now...",
    height=120,
)

if st.button("Analyze Review", type="primary"):
    if not user_review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned = re.sub(r"[^a-z\s]", "", user_review.lower())
        vec = tfidf.transform([cleaned])

        cols = st.columns(len(models))
        for col, (name, model) in zip(cols, models.items()):
            pred = model.predict(vec)[0]
            label = "Fake (1)" if pred == 1 else "Real (0)"
            col.metric(label=name, value=label)