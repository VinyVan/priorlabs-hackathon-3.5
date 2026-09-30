"""Data loading + stratified 80/20 split (seed 42) for Sahel Agri Predictor V1.

Target: binary `Cropland` column in data/raw/train_feat.parquet.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train_feat.parquet"
TARGET_COL = "Cropland"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_train(path: str | Path = DEFAULT_TRAIN_PATH, target: str = TARGET_COL):
    """Load train parquet -> (X, y, feature_names)."""
    df = pd.read_parquet(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}. Cols: {list(df.columns[:5])}...")
    feature_names = [c for c in df.columns if c != target]
    X = df[feature_names]
    y = df[target].astype(int)
    return X, y, feature_names


def get_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified 80/20 split, seed 42."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def load_split(
    path: str | Path = DEFAULT_TRAIN_PATH,
    target: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Convenience: load parquet + return X_train, X_test, y_train, y_test, feature_names."""
    X, y, feature_names = load_train(path, target)
    X_train, X_test, y_train, y_test = get_split(X, y, test_size, random_state)
    return X_train, X_test, y_train, y_test, feature_names


def load_test_features(path: str | Path | None = None) -> pd.DataFrame:
    """Load test_feat.parquet (no target expected)."""
    p = Path(path) if path else PROJECT_ROOT / "data" / "raw" / "test_feat.parquet"
    return pd.read_parquet(p)


def align_to_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Align an uploaded CSV/DataFrame to training feature order.

    - Drops extra cols (incl. Cropland if present), adds missing cols as 0.0.
    - Returns columns in training order.
    """
    df = df.copy()
    df = df.drop(columns=[TARGET_COL], errors="ignore")
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0.0
    return df[feature_names]
