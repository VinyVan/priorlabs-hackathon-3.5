"""Agent: predict + explain (top features) + recommend, in FR/EN plain language.

Explain strategy (model-agnostic, no SHAP dependency):
- Global importance: permutation importance on a reference set (validation split),
  fallback to built-in feature_importances_ / coef_ if permutation fails.
- Per-sample explanation: rank global top features by per-row |z-score|,
  so the user sees which of the globally important signals stand out for THIS plot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def predict_single(model, X_row: pd.DataFrame):
    """Return (label:int, proba:float) for a single-row DataFrame."""
    proba = float(model.predict_proba(X_row)[0, 1])
    label = int(proba >= 0.5)
    return label, proba


def global_importance(model, X_ref: pd.DataFrame, y_ref: pd.Series, top_k: int = 10, random_state: int = 42):
    """Return list of (feature, score) sorted desc. Fast: zero extra model calls.

    1. Built-in `feature_importances_` / `coef_` when available (trees, linear).
    2. Otherwise (e.g. TabPFN API where each inference is a network round-trip):
       univariate ROC-AUC per numeric feature on the reference set.
    Permutation importance is deliberately NOT used: with N features x R repeats
    it would cost hundreds of TabPFN API calls for a single explanation.
    """
    # 1. Built-ins (instant, no inference).
    try:
        if hasattr(model, "feature_importances_"):
            scores = np.asarray(model.feature_importances_, dtype=float)
            order = np.argsort(scores)[::-1][:top_k]
            return [(str(X_ref.columns[i]), float(scores[i])) for i in order]
        if hasattr(model, "coef_"):
            scores = np.abs(np.asarray(model.coef_).ravel())
            order = np.argsort(scores)[::-1][:top_k]
            return [(str(X_ref.columns[i]), float(scores[i])) for i in order]
    except Exception:
        pass
    # 2. Univariate AUC proxy (vectorized, no model calls).
    try:
        from sklearn.metrics import roc_auc_score

        y = y_ref.values
        aucs: dict[str, float] = {}
        for c in X_ref.columns:
            try:
                v = pd.to_numeric(X_ref[c], errors="coerce").fillna(0.0).values
                if np.std(v) == 0:
                    continue
                a = roc_auc_score(y, v)
                aucs[str(c)] = float(max(a, 1 - a))
            except Exception:
                continue
        if aucs:
            top = sorted(aucs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return [(k, v) for k, v in top]
    except Exception:
        pass
    # 3. Last resort: variance ranking.
    scores = X_ref.var(numeric_only=True).fillna(0.0)
    top = scores.sort_values(ascending=False).head(top_k)
    return [(str(k), float(v)) for k, v in top.items()]


def explain_sample(
    model,
    X_row: pd.DataFrame,
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
    top_k: int = 5,
):
    """Explain one sample -> list of dicts {feature, value, z, global_score}."""
    glob = global_importance(model, X_ref, y_ref, top_k=max(top_k * 2, 10))
    gscore = dict(glob)
    means = X_ref.mean(numeric_only=True)
    stds = X_ref.std(numeric_only=True).replace(0, 1e-9).fillna(1.0)
    row = X_row.iloc[0]
    ranked = []
    for feat, _ in glob:
        if feat not in X_row.columns:
            continue
        try:
            v = float(row[feat])
            m = float(means.get(feat, 0.0))
            s = float(stds.get(feat, 1.0))
            z = (v - m) / s
        except Exception:
            v, z = 0.0, 0.0
        ranked.append(
            {"feature": feat, "value": v, "z": float(z), "global_score": float(gscore.get(feat, 0.0))}
        )
    ranked.sort(key=lambda d: abs(d["z"]) * (abs(d["global_score"]) + 1e-9), reverse=True)
    return ranked[:top_k]


def recommend(label: int, proba: float, lang: str = "fr") -> str:
    """Rule-based agronomic recommendation in plain language."""
    lang = (lang or "fr").lower()[:2]
    if lang == "fr":
        if label == 1 and proba >= 0.8:
            return (
                "Parcelle très probablement cultivée — priorisez-la pour le suivi rendement, "
                "vérifiez l'accès à l'eau et planifiez la récolte. Risque alimentaire local faible sur ce point."
            )
        if label == 1:
            return (
                "Parcelle probablement cultivée — à vérifier sur le terrain si possible, "
                "surveillez la pluie et l'état des cultures dans les prochaines semaines."
            )
        if proba >= 0.35:
            return (
                "Cas incertain — refaites une observation (autre date satellite ou visite terrain) "
                "avant de décider ; ne pas exclure un usage agricole."
            )
        return (
            "Parcelle probablement non cultivée — pas d'action agricole immédiate ; "
            "réévaluez après les prochaines pluies si la sécurité alimentaire locale est fragile."
        )
    else:
        if label == 1 and proba >= 0.8:
            return (
                "Very likely cropland — prioritize it for yield monitoring, "
                "check water access and plan harvest. Local food risk low at this point."
            )
        if label == 1:
            return (
                "Likely cropland — verify on the ground if possible, "
                "monitor rainfall and crop condition in the coming weeks."
            )
        if proba >= 0.35:
            return (
                "Uncertain case — take another observation (different satellite date or field visit) "
                "before deciding; do not rule out agricultural use."
            )
        return (
            "Likely not cropland — no immediate farm action; "
            "re-assess after the next rains if local food security is fragile."
        )


def format_explanation(expl: list[dict], lang: str = "fr") -> str:
    lang = (lang or "fr").lower()[:2]
    lines = []
    for i, e in enumerate(expl, 1):
        direction = "au-dessus" if e["z"] >= 0 else "en-dessous"
        if lang == "fr":
            lines.append(
                f"{i}. {e['feature']} = {e['value']:.3f} ({direction} de la moyenne, z={e['z']:+.2f})"
            )
        else:
            direction_en = "above" if e["z"] >= 0 else "below"
            lines.append(
                f"{i}. {e['feature']} = {e['value']:.3f} ({direction_en} average, z={e['z']:+.2f})"
            )
    return "\n".join(lines)


def run_agent(model, X_row: pd.DataFrame, X_ref: pd.DataFrame, y_ref: pd.Series, lang: str = "fr"):
    """Full agent pipeline -> dict with prediction, explanation, recommendation."""
    label, proba = predict_single(model, X_row)
    expl = explain_sample(model, X_row, X_ref, y_ref)
    reco = recommend(label, proba, lang)
    if lang.lower().startswith("fr"):
        verdict = "Cultivée" if label == 1 else "Non cultivée"
        text = (
            f"Prédiction : {verdict} (probabilité culture = {proba:.2f}).\n"
            f"Signaux principaux :\n{format_explanation(expl, 'fr')}\n"
            f"Recommandation : {reco}"
        )
    else:
        verdict = "Cropland" if label == 1 else "Not cropland"
        text = (
            f"Prediction: {verdict} (crop probability = {proba:.2f}).\n"
            f"Top signals:\n{format_explanation(expl, 'en')}\n"
            f"Recommendation: {reco}"
        )
    return {
        "label": label,
        "proba": proba,
        "explanation": expl,
        "recommendation": reco,
        "text": text,
    }
