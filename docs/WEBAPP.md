# WEBAPP — Next.js flagship demo (V2, après polish Streamlit)

## Pourquoi
Le Streamlit prouve le ML, le Next.js gagne le jury : vraie app soumettable (catégorie `app`), démo vidéo premium, URL publique.

## Stack
- `web/` : Next.js 14 App Router + TypeScript + Tailwind, FR/EN (sous-dossier `web/messages/`), thème clair/sombre via `data-theme`.
- `api/` : FastAPI Python (même `.venv`, scikit-learn/pandas/tabpfn-client/catboost) — endpoints `POST /predict` (TabPFN-3.5 + baseline, CSV ou JSON), `GET /metrics` (lit `data/processed/metrics.json`), `GET /sample`.
- Next.js proxifie vers FastAPI (rewrite `/api/py/*`), une seule URL à tunnéliser.

## Pages (une par sujet)
1. `/` — hero Sahel Agri + verdict live sur exemple + CTA.
2. `/predict` — upload CSV ou exemple, jauge proba, top signaux (bar chart), reco FR/EN, switch modèle TabPFN/baseline.
3. `/compare` — tableau metrics.json + phrase gagnant + courbes sample_preds.
4. `/about` — méthode, dataset, lien repo + hackathon.

## Non-négociables
- L'agent ne touche ni `app/`, ni `src/`, ni `scripts/`, ni `.streamlit/` (territoire du polish) ; nouveau code seulement dans `web/` + `api/` + ce doc.
- Jamais de `git`, jamais de lecture `.env*` (env préchargé, `TABPFN_API_KEY` + `API_PORT`).
- Seed 42, split identique, chiffres = ceux de `metrics.json`, jamais recalculés à la main.
- Preuve : `tsc --noEmit` vert + `next build` vert + FastAPI `/health` 200 + 1 prédiction via curl + capture du hero.

## Ordre de lancement
Uniquement quand `feature/polish-app` est mergée (jamais deux agents dans le même worktree).
Branche : `feature/webapp-nextjs` depuis `develop`.
