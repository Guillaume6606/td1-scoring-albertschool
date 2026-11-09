#!/usr/bin/env bash
# Prépare l'environnement : uv sync, données synthétiques (déterministes), vérification.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "uv manquant : https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
if [[ "$(uname)" == "Darwin" ]] && ! brew list libomp >/dev/null 2>&1; then
  echo "macOS : XGBoost a besoin d'OpenMP -> brew install libomp"; exit 1
fi
uv sync
[[ -f data/history.parquet ]] || uv run prepare.py --data-only
uv run prepare.py --verify || echo "ATTENTION : données différentes de la référence (versions numpy/pandas ?). Le TD reste utilisable."
if [[ ! -d .git ]]; then  # un dépôt local pour voir (git diff) ou annuler (git checkout) vos modifications
  git init -q && git add -A && git -c user.name=td1 -c user.email=td1@localhost commit -qm "TD1 : version livrée" && echo "dépôt git local créé"
fi
uv run python -c "import xgboost, fastapi, pandas; print('Setup OK -', 'xgboost', xgboost.__version__)"
