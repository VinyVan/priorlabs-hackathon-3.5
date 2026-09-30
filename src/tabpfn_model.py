"""TabPFN-3.5 wrapper: tabpfn_client API first, local tabpfn fallback.

- Uses env TABPFN_API_KEY only (never reads .env files).
- API: `from tabpfn_client import TabPFNClassifier`
- Fallback local: `from tabpfn import TabPFNClassifier`
"""
from __future__ import annotations

import os


def which_backend() -> str:
    """Return 'api' if a key is configured, else 'local'."""
    return "api" if os.environ.get("TABPFN_API_KEY") else "local"


def build_tabpfn():
    """Build a TabPFNClassifier, preferring the 3.5 API client.

    Returns (model, backend_name). backend_name in {"api", "local"}.
    Falls back to local `tabpfn` if the client is missing or no key is set.
    """
    key = os.environ.get("TABPFN_API_KEY")
    if key:
        try:
            import tabpfn_client  # type: ignore
            from tabpfn_client import TabPFNClassifier  # type: ignore

            # Authenticate the client from env (never from .env files).
            try:
                tabpfn_client.set_access_token(key)
            except Exception:
                pass
            return TabPFNClassifier(), "api"
        except Exception as e:  # pragma: no cover - env dependent
            print(f"[tabpfn_model] tabpfn_client unavailable ({e}), trying local fallback.")
    try:
        from tabpfn import TabPFNClassifier as LocalTabPFNClassifier  # type: ignore

        return LocalTabPFNClassifier(), "local"
    except Exception as e:
        raise ImportError(
            "Neither tabpfn_client nor local tabpfn is available. "
            "Install with: pip install tabpfn-client tabpfn"
        ) from e


def train_tabpfn(X_train, y_train):
    """Fit a TabPFN classifier (in-context fit) and return (model, backend)."""
    model, backend = build_tabpfn()
    model.fit(X_train, y_train)
    return model, backend
