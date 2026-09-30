"""Train CatBoost/LightGBM baseline on same stratified 80/20 split (seed 42)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data_loader import PROJECT_ROOT, load_split
from src.baseline import train_baseline


def main():
    X_train, X_test, y_train, y_test, features = load_split()
    print(f"[baseline] train={X_train.shape} test={X_test.shape} n_features={len(features)}")
    model, backend = train_baseline(X_train, y_train)
    print(f"[baseline] backend={backend}")
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")
    print(f"[baseline] accuracy={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
    result = {"backend": backend, "accuracy": float(acc), "f1": float(f1), "auc": float(auc)}
    # Merge into data/processed/metrics.json (preserving any tabpfn entry).
    metrics_path = PROJECT_ROOT / "data" / "processed" / "metrics.json"
    merged = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    merged["baseline"] = result
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(merged, indent=2))
    print(f"[baseline] updated {metrics_path}")
    return result


if __name__ == "__main__":
    main()
