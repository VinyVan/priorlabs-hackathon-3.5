# CDC — TabPFN-3.5 Sahel Agri (Hackathon PriorLabs 3.5)

## Objectif
Gagner le hackathon : projet open-ended avec TabPFN-3.5, jury Prior Labs (créativité + perf). Deadline 6 octobre 2026.

## Concept
`Sahel Agri Predictor` : prédiction cropland / risque alimentaire à partir de features tabulaires satellite + climat, avec TabPFN-3.5 au cœur, derrière un agent qui prédit + explique + recommande, + mini-app démo.

Catégories cochées : hard problem (santé/climat/sécurité alimentaire), agent, app.

## Données
- Source : `~/Documents/projects/cropland-mapping/data/train_feat.parquet` (1000 lignes, 189 cols, cible `Cropland`), `test_feat.parquet`, `test.csv`.
- Copie locale dans `data/raw/` (ne jamais modifier l'original).
- Cible : classification binaire `Cropland` (0/1).

## Exigences
1. Utiliser TabPFN-3.5 via `tabpfn_client` (`TabPFNClassifier`) avec `TABPFN_API_KEY` (env, jamais commitée). Fallback local `tabpfn` si pas de clé.
2. Baseline comparatif : CatBoost ou LightGBM (déjà dans requirements cropland) vs TabPFN-3.5, même split, métriques accuracy/F1/AUC.
3. Agent : script `src/agent.py` — `predict + explain (top features) + recommend` en langage simple (FR/EN).
4. App démo : Streamlit `app/streamlit_app.py` — upload CSV ou exemple, prédiction, explication, reco. Doit tourner en local avec `streamlit run`.
5. Repo soumettable : README avec setup en 5 min, résultats chiffrés, lien vidéo démo (placeholder `demo/`), `requirements.txt`, `.gitignore` sans secrets.
6. Reproductibilité : `scripts/train_tabpfn.py`, `scripts/baseline.py`, `scripts/evaluate.py`, seed fixe, outputs dans `submissions/` + `data/processed/metrics.json`.

## Hors scope
- Pas de vidéo finale (Hermes la fera après), pas de compte Prior Labs dans le code, pas de données Sahel-Mali réelles V1 (extension documentée seulement).

## Critères d'acceptation
- `pip install -r requirements.txt` OK, `python scripts/train_tabpfn.py` produit `metrics.json` avec TabPFN ≥ baseline ou écart expliqué.
- `streamlit run app/streamlit_app.py` répond 200, prédit sur 1 exemple.
- Aucun secret commité, aucun `git` exécuté par l'agent.
