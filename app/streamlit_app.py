"""Sahel Agri Predictor — Streamlit demo (polished Sahel/agri edition).

- Upload a CSV (same features as train, target optional) OR use the built-in sample.
- Predict with TabPFN-3.5 (API if TABPFN_API_KEY set, else local fallback) or baseline.
- Explain (top features) + recommend in FR/EN.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src.agent import run_agent
from src.baseline import train_baseline
from src.data_loader import PROJECT_ROOT, align_to_features, load_split
from src.tabpfn_model import train_tabpfn, which_backend


st.set_page_config(
    page_title="Sahel Agri Predictor — TabPFN-3.5",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------- i18n ---
T = {
    "fr": {
        "pitch": "Prédiction de parcelles cultivées au Sahel + explication simple + recommandation agronomique.",
        "model": "Modèle",
        "backend": "Moteur TabPFN",
        "input_title": "1. Entrée — téléversez un CSV ou utilisez l'exemple",
        "upload": "Téléverser un CSV avec les variables d'entraînement",
        "sample_btn": "Utiliser l'exemple intégré",
        "sample_info": "Exemple intégré utilisé (première ligne du jeu de test).",
        "csv_ok": "CSV chargé",
        "csv_aligned": "aligné",
        "need_input": "Téléversez un CSV ou cliquez sur « Utiliser l'exemple intégré » pour lancer une prédiction.",
        "result_title": "2. Résultat — prédiction + explication + recommandation",
        "row": "Ligne à expliquer (index)",
        "proba_label": "Probabilité culture",
        "signals": "Signaux principaux (importance combinée)",
        "reco": "Recommandation agronomique",
        "full_text": "Texte complet de l'agent",
        "compare_title": "3. Comparaison des modèles — TabPFN-3.5 vs baseline",
        "compare_hint": "Métriques sur le même split stratifié 80/20 (seed 42), cible « Cropland ».",
        "no_metrics": "Métriques introuvables — lancez `.venv/bin/python scripts/evaluate.py`.",
        "footer": "Démo Sahel Agri Predictor — PriorLabs Hackathon 3.5",
    },
    "en": {
        "pitch": "Sahel cropland prediction + plain-language explanation + agronomic recommendation.",
        "model": "Model",
        "backend": "TabPFN backend",
        "input_title": "1. Input — upload a CSV or use the built-in sample",
        "upload": "Upload a CSV with the training features",
        "sample_btn": "Use built-in sample row",
        "sample_info": "Using built-in sample (first test row).",
        "csv_ok": "CSV loaded",
        "csv_aligned": "aligned",
        "need_input": "Upload a CSV or click 'Use built-in sample row' to run a prediction.",
        "result_title": "2. Results — prediction + explanation + recommendation",
        "row": "Row to explain (index)",
        "proba_label": "Crop probability",
        "signals": "Top signals (combined importance)",
        "reco": "Agronomic recommendation",
        "full_text": "Full agent text",
        "compare_title": "3. Model comparison — TabPFN-3.5 vs baseline",
        "compare_hint": "Metrics on the same stratified 80/20 split (seed 42), target 'Cropland'.",
        "no_metrics": "Metrics file not found — run `.venv/bin/python scripts/evaluate.py`.",
        "footer": "Sahel Agri Predictor demo — PriorLabs Hackathon 3.5",
    },
}


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


def load_metrics() -> dict | None:
    path = PROJECT_ROOT / "data" / "processed" / "metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def winner_sentence(metrics: dict, lang: str) -> str:
    """Short FR/EN sentence stating the TabPFN vs baseline winner (F1 first, accuracy tie-break)."""
    try:
        tf1 = float(metrics["tabpfn"]["f1"])
        bf1 = float(metrics["baseline"]["f1"])
        tacc = float(metrics["tabpfn"]["accuracy"])
        bacc = float(metrics["baseline"]["accuracy"])
    except Exception:
        return (
            "Comparaison indisponible. / Comparison unavailable."
            if lang == "fr"
            else "Comparison unavailable. / Comparaison indisponible."
        )
    tabpfn_wins = (tf1, tacc) >= (bf1, bacc)
    if lang == "fr":
        if tabpfn_wins:
            return (
                f"🏆 TabPFN-3.5 gagne (F1 {tf1:.3f} vs {bf1:.3f}, "
                f"accuracy {tacc:.3f} vs {bacc:.3f}). / "
                f"TabPFN-3.5 wins (F1 {tf1:.3f} vs {bf1:.3f}, "
                f"accuracy {tacc:.3f} vs {bacc:.3f})."
            )
        return (
            f"🏆 La baseline gagne (F1 {bf1:.3f} vs {tf1:.3f}, "
            f"accuracy {bacc:.3f} vs {tacc:.3f}). / "
            f"Baseline wins (F1 {bf1:.3f} vs {tf1:.3f}, "
            f"accuracy {bacc:.3f} vs {tacc:.3f})."
        )
    if tabpfn_wins:
        return (
            f"🏆 TabPFN-3.5 wins (F1 {tf1:.3f} vs {bf1:.3f}, "
            f"accuracy {tacc:.3f} vs {bacc:.3f}). / "
            f"TabPFN-3.5 gagne (F1 {tf1:.3f} vs {bf1:.3f}, "
            f"accuracy {tacc:.3f} vs {bacc:.3f})."
        )
    return (
        f"🏆 Baseline wins (F1 {bf1:.3f} vs {tf1:.3f}, "
        f"accuracy {bacc:.3f} vs {tacc:.3f}). / "
        f"La baseline gagne (F1 {bf1:.3f} vs {tf1:.3f}, "
        f"accuracy {bacc:.3f} vs {tacc:.3f})."
    )


# ------------------------------------------------------------------ hero ---
st.markdown(
    "🌾 **Sahel Agri Predictor** · *TabPFN-3.5 au service de l'agriculture sahélienne*",
)
st.title("🌾 Sahel Agri Predictor — TabPFN-3.5")
st.markdown(
    "Prédiction de parcelles cultivées au Sahel + explication simple + recommandation agronomique.  \n"
    "*Sahel cropland prediction + plain-language explanation + agronomic recommendation.*"
)
st.caption("🌍 Sahel · 🛰️ satellite + climat · 🤖 TabPFN-3.5 + baseline CatBoost/LightGBM · FR/EN")

bundle = get_models()
features = bundle["features"]

# ------------------------------------------------------- model + language ---
col_a, col_b, col_c = st.columns(3)
with col_a:
    model_choice = st.selectbox("Modèle / Model", ["tabpfn", "baseline"])
with col_b:
    lang = st.selectbox("Langue / Language", ["fr", "en"])
with col_c:
    st.metric(
        f"{T[lang]['backend']} / {T['en']['backend']}",
        f"{bundle['tabpfn_backend']} ({which_backend()} configured)",
    )

txt = T[lang]

# ------------------------------------------------------------------ input ---
st.divider()
st.subheader(txt["input_title"])
uploaded = st.file_uploader(txt["upload"] + " / " + T["en" if lang == "fr" else "fr"]["upload"], type=["csv"])
if st.button(txt["sample_btn"] + " / " + T["en" if lang == "fr" else "fr"]["sample_btn"]):
    st.session_state["_use_sample"] = True

input_df: pd.DataFrame | None = None
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        input_df = align_to_features(raw, features)
        st.success(f"✅ {txt['csv_ok']} / {T['en' if lang == 'fr' else 'fr']['csv_ok']}: {raw.shape} → {txt['csv_aligned']} {input_df.shape}")
        st.dataframe(input_df.head(3), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV / Impossible de lire le CSV : {e}")
elif st.session_state.get("_use_sample"):
    sample_path = PROJECT_ROOT / "data" / "processed" / "sample_example.csv"
    if sample_path.exists():
        input_df = pd.read_csv(sample_path)
        input_df = align_to_features(input_df, features)
    else:
        input_df = bundle["X_test"].iloc[[0]].reset_index(drop=True)
    st.info(f"ℹ️ {txt['sample_info']} / {T['en' if lang == 'fr' else 'fr']['sample_info']}")
    st.dataframe(input_df, use_container_width=True)

# ----------------------------------------------------------------- results ---
if input_df is not None and len(input_df):
    st.divider()
    st.subheader(txt["result_title"])
    if len(input_df) > 1:
        idx = st.number_input(
            txt["row"] + " / " + T["en" if lang == "fr" else "fr"]["row"],
            0,
            len(input_df) - 1,
            0,
            1,
        )
    else:
        idx = 0
    row = input_df.iloc[[int(idx)]].reset_index(drop=True)
    model = bundle[model_choice]
    with st.spinner("Running agent... / Exécution de l'agent..."):
        res = run_agent(model, row, bundle["X_test"], bundle["y_test"], lang=lang)

    proba = float(min(max(res["proba"], 0.0), 1.0))
    is_crop = res["label"] == 1
    verdict_fr = "Cultivée ✅" if is_crop else "Non cultivée ❌"
    verdict_en = "Cropland ✅" if is_crop else "Not cropland ❌"
    verdict = f"{verdict_fr} / {verdict_en}"

    # Styled verdict card
    with st.container(border=True):
        st.markdown(f"### {'🌱' if is_crop else '🏜️'} {verdict}")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric(txt["proba_label"] + " / Crop probability" if lang == "fr" else "Crop probability / Probabilité culture", f"{proba:.2f}")
        with c2:
            st.write(f"**{txt['proba_label']} / Probability** : {proba:.0%}")
            st.progress(proba)

    # Top signals as native bar chart (no new dependencies)
    st.write(f"**{txt['signals']}**")
    try:
        expl = res.get("explanation", [])
        chart_df = pd.DataFrame(
            {
                row_e["feature"]: abs(row_e.get("z", 0.0))
                * (abs(row_e.get("global_score", 0.0)) + 1e-9)
                for row_e in expl
            },
            index=["importance"],
        ).T.sort_values("importance", ascending=False)
        if len(chart_df):
            st.bar_chart(chart_df, use_container_width=True)
        st.dataframe(pd.DataFrame(expl), use_container_width=True)
    except Exception:
        st.dataframe(pd.DataFrame(res.get("explanation", [])), use_container_width=True)

    # Recommendation callout
    st.write(f"**{txt['reco']}**")
    reco_text = res["recommendation"]
    if is_crop and proba >= 0.8:
        st.success(f"🌱 {reco_text}")
    elif is_crop:
        st.info(f"🌿 {reco_text}")
    elif proba >= 0.35:
        st.warning(f"⚠️ {reco_text}")
    else:
        st.error(f"🏜️ {reco_text}")

    with st.expander(txt["full_text"] + " / Full text" if lang == "fr" else "Full agent text / Texte complet"):
        st.text(res["text"])
else:
    st.warning(f"⚠️ {txt['need_input']}")

# -------------------------------------------------------- model comparison ---
st.divider()
st.subheader(txt["compare_title"] + (" / Model comparison" if lang == "fr" else " / Comparaison des modèles"))
st.caption(txt["compare_hint"])
metrics = load_metrics()
if metrics is None:
    st.warning(f"⚠️ {txt['no_metrics']}")
else:
    try:
        table = pd.DataFrame(
            [
                {
                    "model": "tabpfn",
                    "backend": metrics["tabpfn"].get("backend", ""),
                    "accuracy": round(float(metrics["tabpfn"]["accuracy"]), 4),
                    "f1": round(float(metrics["tabpfn"]["f1"]), 4),
                    "auc": round(float(metrics["tabpfn"]["auc"]), 4),
                },
                {
                    "model": "baseline",
                    "backend": metrics["baseline"].get("backend", ""),
                    "accuracy": round(float(metrics["baseline"]["accuracy"]), 4),
                    "f1": round(float(metrics["baseline"]["f1"]), 4),
                    "auc": round(float(metrics["baseline"]["auc"]), 4),
                },
            ]
        )
        st.table(table)
    except Exception:
        st.json(metrics)
    st.success(winner_sentence(metrics, lang))

# ------------------------------------------------------------------ footer ---
st.divider()
st.caption(
    f"🌾 {txt['footer']} · "
    "[Repo GitHub / GitHub repo](https://github.com/VinyVan/priorlabs-hackathon-3.5) · "
    "Built with TabPFN-3.5 · PriorLabs Hackathon 3.5"
)
