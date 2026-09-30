"""FastAPI backend for the Next.js flagship demo (V2).

Endpoints:
  GET  /health   -> {"status": "ok", ...}
  GET  /metrics  -> serves data/processed/metrics.json verbatim
  GET  /sample   -> one example row from data/processed/sample_example.csv
  POST /predict  -> JSON row (or list of rows) or multipart CSV file
                    -> TabPFN-3.5 (TABPFN_API_KEY) with local/catboost fallback,
                       plus explain + recommend via src.agent (sys.path import).

Notes:
  - TABPFN_API_KEY comes from the preloaded env only; never read from .env files.
  - Seed 42 / stratified 80/20 split identical to training (via src.data_loader).
  - Figures are served from metrics.json, never recomputed by hand.
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import agent  # noqa: E402
from src import data_loader  # noqa: E402

METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "metrics.json"
SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "sample_example.csv"

app = FastAPI(title="Sahel Agri Predictor API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()
_DATA = None  # (X_train, X_test, y_train, y_test, feature_names)
_BASELINE = None  # (model, backend)
_TABPFN = None  # (model, backend) or ("fallback-error", ...)
_TABPFN_ERROR: str | None = None


def get_data():
    """Load stratified split once (seed 42). Fast: single parquet read."""
    global _DATA
    if _DATA is None:
        with _lock:
            if _DATA is None:
                X_train, X_test, y_train, y_test, feature_names = data_loader.load_split()
                _DATA = (X_train, X_test, y_train, y_test, feature_names)
    return _DATA


def get_baseline():
    """Train (once) CatBoost baseline, LightGBM fallback. Local, fast."""
    global _BASELINE
    if _BASELINE is None:
        with _lock:
            if _BASELINE is None:
                from src.baseline import train_baseline

                X_train, _, y_train, _, _ = get_data()
                model, backend = train_baseline(X_train, y_train)
                _BASELINE = (model, backend)
    return _BASELINE


def get_tabpfn():
    """Build + fit TabPFN-3.5 once. Falls back to local tabpfn, then baseline.

    Raises the last error only if even the baseline is unavailable.
    Returns (model, backend) where backend in {"api", "local", "catboost", ...}.
    """
    global _TABPFN, _TABPFN_ERROR
    if _TABPFN is None:
        with _lock:
            if _TABPFN is None:
                X_train, _, y_train, _, _ = get_data()
                # 1. Try TabPFN-3.5 API / local fallback via wrapper.
                try:
                    from src.tabpfn_model import train_tabpfn

                    model, backend = train_tabpfn(X_train, y_train)
                    _TABPFN = (model, backend)
                except Exception as e:  # API down / no key / no local pkg
                    _TABPFN_ERROR = f"{type(e).__name__}: {e}"
                    # 2. Fall back to the already-trained baseline.
                    model, backend = get_baseline()
                    _TABPFN = (model, f"{backend}-fallback")
    return _TABPFN


def _pick_model(name: str):
    """Return (model, backend, used_fallback: bool)."""
    name = (name or "tabpfn").lower()
    if name == "baseline":
        model, backend = get_baseline()
        return model, backend, False
    model, backend = get_tabpfn()
    return model, backend, backend.endswith("-fallback")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "sahel-agri-api",
        "tabpfn_key_configured": bool(os.environ.get("TABPFN_API_KEY")),
        "metrics_available": METRICS_PATH.exists(),
        "sample_available": SAMPLE_PATH.exists(),
    }


@app.get("/")
def root():
    return {
        "service": "sahel-agri-api",
        "endpoints": ["GET /health", "GET /metrics", "GET /sample", "POST /predict"],
    }


@app.get("/metrics")
def metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.get("/sample")
def sample():
    df = pd.read_csv(SAMPLE_PATH)
    if len(df) == 0:
        return JSONResponse(status_code=404, content={"error": "sample_example.csv is empty"})
    row = df.iloc[0].to_dict()
    # JSON-safe: cast numpy scalars.
    clean = {k: (float(v) if isinstance(v, float) else v) for k, v in row.items()}
    for k, v in clean.items():
        try:
            import math

            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = 0.0
        except Exception:
            pass
    return {"row": clean, "columns": list(df.columns), "n_columns": len(df.columns)}


@app.post("/predict")
async def predict_endpoint(request: Request):
    ctype = request.headers.get("content-type", "")
    model_name = "tabpfn"
    lang = "fr"
    df: pd.DataFrame | None = None

    if "multipart" in ctype:
        form = await request.form()
        model_name = str(form.get("model", "tabpfn") or "tabpfn")
        lang = str(form.get("lang", form.get("language", "fr")) or "fr")
        upload = form.get("file")
        if upload is None:
            return JSONResponse(status_code=400, content={"error": "multipart: missing 'file' field"})
        content = await upload.read()  # type: ignore[union-attr]
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"cannot parse CSV: {e}"})
    else:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "invalid JSON body"})
        if isinstance(body, list):
            df = pd.DataFrame(body)
        elif isinstance(body, dict):
            model_name = str(body.get("model", "tabpfn") or "tabpfn")
            lang = str(body.get("lang", body.get("language", "fr")) or "fr")
            if isinstance(body.get("row"), dict):
                df = pd.DataFrame([body["row"]])
            elif isinstance(body.get("rows"), list):
                df = pd.DataFrame(body["rows"])
            elif isinstance(body.get("data"), list):
                df = pd.DataFrame(body["data"])
            else:
                # Raw single row: remaining keys are features.
                row = {k: v for k, v in body.items() if k not in ("model", "lang", "language")}
                if not row:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "empty row: send {'row': {...}} or raw feature dict"},
                    )
                df = pd.DataFrame([row])
        else:
            return JSONResponse(status_code=400, content={"error": "body must be a JSON object or array"})

    if df is None or len(df) == 0:
        return JSONResponse(status_code=400, content={"error": "no rows to predict"})

    try:
        _, _, _, _, feature_names = get_data()
        Xq = data_loader.align_to_features(df, feature_names)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"feature alignment failed: {e}"})

    lang = (lang or "fr").lower()[:2]
    if lang not in ("fr", "en"):
        lang = "fr"

    try:
        model, backend, used_fallback = _pick_model(model_name)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"model unavailable: {e}"})

    try:
        X_train, _, y_train, _, _ = get_data()
        # Full agent pipeline on the first row.
        result = agent.run_agent(model, Xq.iloc[[0]], X_train, y_train, lang=lang)
        # Extra rows (CSV batch): cheap label+proba each.
        predictions = None
        if len(Xq) > 1:
            predictions = []
            for i in range(len(Xq)):
                try:
                    lbl, pr = agent.predict_single(model, Xq.iloc[[i]])
                except Exception:
                    lbl, pr = 0, 0.0
                predictions.append({"index": i, "label": int(lbl), "proba": float(pr)})
        payload = {
            "label": int(result["label"]),
            "proba": float(result["proba"]),
            "model": model_name if not used_fallback else "tabpfn",
            "backend": backend,
            "fallback": bool(used_fallback),
            "recommendation": result["recommendation"],
            "explanation": result["explanation"],
            "text": result["text"],
            "n_rows": int(len(Xq)),
            "tabpfn_error": _TABPFN_ERROR if used_fallback else None,
        }
        if predictions is not None:
            payload["predictions"] = predictions
        return payload
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"prediction failed: {e}"})
