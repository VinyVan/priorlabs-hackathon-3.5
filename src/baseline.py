"""Baseline: CatBoost preferred, LightGBM fallback. Same split/seed as TabPFN.

Wraps the booster in BaselineModel, which ordinal-encodes object/string
columns with categories fitted on the TRAIN set (unseen -> -1), so both
training and later predict/predict_proba (incl. uploaded CSVs) work.
"""
from __future__ import annotations

import pandas as pd


def fit_cat_maps(X_train: pd.DataFrame) -> dict[str, pd.Index]:
    """Categories per object column, fitted on train."""
    maps: dict[str, pd.Index] = {}
    for c in X_train.columns:
        if X_train[c].dtype == object or str(X_train[c].dtype) == "category":
            maps[c] = pd.Index(X_train[c].dropna().unique())
    return maps


def apply_cat_maps(X: pd.DataFrame, maps: dict[str, pd.Index]) -> pd.DataFrame:
    """Encode object columns to int codes with train-fitted categories (unseen -> -1)."""
    X = X.copy()
    for c, cats in maps.items():
        if c in X.columns:
            X[c] = pd.Categorical(X[c], categories=cats).codes.astype("int64")
    return X


class BaselineModel:
    """Booster + categorical encoder with a sklearn-like interface."""

    def __init__(self, model, cat_maps: dict[str, pd.Index]):
        self.model = model
        self.cat_maps = cat_maps

    def _prep(self, X: pd.DataFrame) -> pd.DataFrame:
        return apply_cat_maps(X, self.cat_maps)

    def fit(self, X, y):
        self.model.fit(self._prep(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self._prep(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))

    def __getattr__(self, name):
        # Delegate feature_importances_, classes_, etc. to the inner booster.
        return getattr(self.__dict__["model"], name)


def build_baseline(prefer: str = "catboost"):
    """Return (model, backend_name) with backend in {"catboost", "lightgbm"}.

    Tries `prefer` first, falls back to the other library.
    """
    order = ["catboost", "lightgbm"] if prefer == "catboost" else ["lightgbm", "catboost"]
    last_err: Exception | None = None
    for name in order:
        try:
            if name == "catboost":
                from catboost import CatBoostClassifier  # type: ignore

                model = CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                )
                return model, "catboost"
            else:
                from lightgbm import LGBMClassifier  # type: ignore

                model = LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=-1,
                    random_state=42,
                    verbose=-1,
                )
                return model, "lightgbm"
        except Exception as e:  # pragma: no cover - env dependent
            last_err = e
            continue
    raise ImportError(
        "Neither catboost nor lightgbm is available. pip install catboost lightgbm"
    ) from last_err


def train_baseline(X_train, y_train, prefer: str = "catboost"):
    """Fit baseline and return (BaselineModel, backend)."""
    raw_model, backend = build_baseline(prefer)
    maps = fit_cat_maps(X_train)
    model = BaselineModel(raw_model, maps)
    model.fit(X_train, y_train)
    return model, backend
