"""Evaluate TabPFN-3.5 vs baseline on the SAME stratified 80/20 split (seed 42).

Writes data/processed/metrics.json with {tabpfn, baseline} accuracy/F1/AUC,
saves sample_example.csv + submissions/sample_preds.csv,
and prints one sample prediction (FINISH CONDITION).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.agent import run_agent
from src.baseline import train_baseline
from src.data_loader import PROJECT_ROOT, load_split
from src.tabpfn_model import train_tabpfn, which_backend


def score(y_true, proba):
    pred = (proba >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, proba))
    except Exception:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "auc": auc,
    }


def main():
    X_train, X_test, y_train, y_test, features = load_split()
    print(f"[evaluate] train={X_train.shape} test={X_test.shape} features={len(features)}")
    print(f"[evaluate] TABPFN backend hint: {which_backend()}")

    # --- TabPFN-3.5 ---
    tabpfn_model, tabpfn_backend = train_tabpfn(X_train, y_train)
    tabpfn_proba = tabpfn_model.predict_proba(X_test)[:, 1]
    tabpfn_metrics = {"backend": tabpfn_backend, **score(y_test, tabpfn_proba)}
    print(f"[evaluate] tabpfn ({tabpfn_backend}): {tabpfn_metrics}")

    # --- Baseline ---
    base_model, base_backend = train_baseline(X_train, y_train)
    base_proba = base_model.predict_proba(X_test)[:, 1]
    baseline_metrics = {"backend": base_backend, **score(y_test, base_proba)}
    print(f"[evaluate] baseline ({base_backend}): {baseline_metrics}")

    metrics = {
        "split": {"test_size": 0.2, "random_state": 42, "stratified": True, "target": "Cropland"},
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(features)),
        "tabpfn": tabpfn_metrics,
        "baseline": baseline_metrics,
    }

    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] wrote {out_dir / 'metrics.json'}")

    # --- Sample artifacts + one printed sample prediction (FINISH CONDITION) ---
    sample_row = X_test.iloc[[0]]
    sample_true = int(y_test.iloc[0])
    X_ref = X_test
    res = run_agent(tabpfn_model, sample_row, X_ref, y_test, lang="fr")
    print(
        f"[evaluate] SAMPLE prediction: label={res['label']} proba={res['proba']:.4f} "
        f"true={sample_true} features={len(features)}"
    )
    print(f"[evaluate] SAMPLE explanation:\n{res['text']}")

    sample_row.to_csv(out_dir / "sample_example.csv", index=False)
    import pandas as pd

    pd.DataFrame(
        {
            "true": y_test.values,
            "tabpfn_proba": tabpfn_proba,
            "baseline_proba": base_proba,
        }
    ).to_csv(PROJECT_ROOT / "submissions" / "sample_preds.csv", index=False)
    print("[evaluate] wrote data/processed/sample_example.csv + submissions/sample_preds.csv")
    print("[evaluate] DONE exit 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
