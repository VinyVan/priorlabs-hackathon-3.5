# ARCHITECTURE — Sahel Agri Predictor

## Stack
Python 3.11, `tabpfn-client` (TabPFN-3.5 API), `tabpfn` (fallback), scikit-learn, pandas, pyarrow, catboost, lightgbm, streamlit, shap (optionnel).

## Arbre
```
docs/CDC.md, ARCHITECTURE.md, VERSIONING.md
data/raw/          # copie train_feat.parquet, test_feat.parquet, test.csv
data/processed/    # metrics.json, sample_example.csv
src/
  data_loader.py   # charge parquets, split stratifié seed=42
  tabpfn_model.py  # wrapper TabPFNClassifier (API + fallback)
  baseline.py      # CatBoost/LightGBM
  agent.py         # predict+explain+recommend (FR/EN)
app/streamlit_app.py
scripts/train_tabpfn.py, baseline.py, evaluate.py
notebooks/01_eda.ipynb (optionnel)
submissions/       # preds exemple
demo/              # script vidéo + placeholder
requirements.txt, .gitignore, README.md
```

## Non-négociables
- Clé via env `TABPFN_API_KEY` uniquement, jamais lue depuis `.env*` par l'agent (le framework charge l'env).
- Seed 42 partout, split stratifié 80/20.
- L'agent ne fait ni `git`, ni `pip install` global, ni lecture de fichiers secrets.
- Preuve exigée : build vert (`python scripts/evaluate.py`) + smoke (`streamlit` import OK + 1 prédiction).
