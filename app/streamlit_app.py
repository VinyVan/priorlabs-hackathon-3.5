"""Sahel Agri Predictor — Streamlit demo.

- Upload a CSV (same features as train, target optional) OR use the built-in sample.
- Predict with TabPFN-3.5 (API if TABPFN_API_KEY set, else local fallback) or baseline.
- Explain (top features) + recommend in FR/EN.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.agent import run_agent
from src.baseline import train_baseline
from src.data_loader import PROJECT_ROOT, align_to_features, load_split
from src.tabpfn_model import train_tabpfn, which_backend


@st.cache_resource(show_spinner="Training models (TabPFN-3.5 + baseline)...")
def get_models():
    X_train, X_test, y_train, y_test, features = load_split()
    tabpfn_model, tabpfn_backend = train_tabpfn(X_train, y_train)
    base_model, base_backend = train_baseline(X_train, y_train)
    return {
        "tabpfn": tabpfn_model,
        "tabpfn_backend": tabpfn_backend,
        "baseline": base_model,
        "baseline_backend": base_backend,
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
    }


st.set_page_config(page_title="Sahel Agri Predictor", page_icon="🌾")
st.title("🌾 Sahel Agri Predictor (TabPFN-3.5)")
st.caption("Cropland prediction + plain-language explanation + recommendation — FR/EN.")

bundle = get_models()
features = bundle["features"]

col_a, col_b, col_c = st.columns(3)
with col_a:
    model_choice = st.selectbox("Model", ["tabpfn", "baseline"])
with col_b:
    lang = st.selectbox("Langue / Language", ["fr", "en"])
with col_c:
    st.metric("TabPFN backend", f"{bundle['tabpfn_backend']} ({which_backend()} configured)")

st.divider()
st.subheader("1. Input — upload CSV or use sample")
uploaded = st.file_uploader("Upload CSV with training features", type=["csv"])
use_sample = st.button("Use built-in sample row")

input_df: pd.DataFrame | None = None
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        input_df = align_to_features(raw, features)
        st.success(f"CSV loaded: {raw.shape} -> aligned {input_df.shape}")
        st.dataframe(input_df.head(3))
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
elif use_sample:
    sample_path = PROJECT_ROOT / "data" / "processed" / "sample_example.csv"
    if sample_path.exists():
        input_df = pd.read_csv(sample_path)
        input_df = align_to_features(input_df, features)
    else:
        input_df = bundle["X_test"].iloc[[0]].reset_index(drop=True)
    st.info("Using sample row (first test row).")
    st.dataframe(input_df)

if input_df is not None and len(input_df):
    st.divider()
    st.subheader("2. Predict + explain + recommend")
    idx = st.number_input("Row to explain", 0, len(input_df) - 1, 0, 1) if len(input_df) > 1 else 0
    row = input_df.iloc[[int(idx)]].reset_index(drop=True)
    model = bundle[model_choice]
    with st.spinner("Running agent..."):
        res = run_agent(model, row, bundle["X_test"], bundle["y_test"], lang=lang)
    verdict = "Cultivée ✅" if (res["label"] == 1 and lang == "fr") else (
        "Non cultivée ❌" if lang == "fr" else ("Cropland ✅" if res["label"] == 1 else "Not cropland ❌")
    )
    st.metric("Prediction", verdict, f"p = {res['proba']:.2f}")
    st.progress(min(max(res["proba"], 0.0), 1.0))
    st.write("**Top signals**")
    st.dataframe(pd.DataFrame(res["explanation"]))
    st.write("**Recommendation**")
    st.info(res["recommendation"])
    with st.expander("Full agent text"):
        st.text(res["text"])
else:
    st.warning("Upload a CSV or click 'Use built-in sample row' to run a prediction.")

st.divider()
st.caption("Demo placeholder video: see demo/ (link added before submission). V1 scope: docs/CDC.md + docs/ARCHITECTURE.md.")
