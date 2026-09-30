"""Train TabPFN-3.5 on stratified 80/20 split (seed 42), print accuracy/F1/AUC."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.data_loader import PROJECT_ROOT, load_split
from src.tabpfn_model import train_tabpfn


def main():
    X_train, X_test, y_train, y_test, features = load_split()
    print(f"[train_tabpfn] train={X_train.shape} test={X_test.shape} n_features={len(features)}")
    model, backend = train_tabpfn(X_train, y_train)
    print(f"[train_tabpfn] backend={backend}")
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")
    print(f"[train_tabpfn] accuracy={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
    result = {"backend": backend, "accuracy": float(acc), "f1": float(f1), "auc": float(auc)}
    # Merge into data/processed/metrics.json (preserving any baseline entry).
    metrics_path = PROJECT_ROOT / "data" / "processed" / "metrics.json"
    merged = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    merged["tabpfn"] = result
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(merged, indent=2))
    print(f"[train_tabpfn] updated {metrics_path}")
    return result


if __name__ == "__main__":
    main()
