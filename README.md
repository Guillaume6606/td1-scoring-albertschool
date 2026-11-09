# TD 1 — Mesurer et dimensionner un service de scoring

Un modèle XGBoost **déjà entraîné** score des transactions bancaires, une par demande. Chaque transaction
est complétée par des agrégats calculés sur l'historique du client. Le service tourne avec FastAPI +
Uvicorn, sur CPU. Il est livré **sans conteneur** : vous écrivez l'image Docker et le script de
déploiement. Le code fonctionne mais ne tient pas le pic de charge prévu par le contrat de service.

Le TD se fait dans **`notebooks/td1_experiences.ipynb`**, environ 1 h 30. Vous packagez et déployez le
service, puis vous écrivez les mesures : chronométrage, profil, tests de charge, correction du goulot
et dimensionnement.

| Étape | Ce que vous mesurez | Ce que vous codez |
| --- | --- | --- |
| préliminaire | l'image se construit, le service répond avec le bon quota | `Dockerfile`, `.dockerignore`, `scripts/deploy.sh` |
| 0 | latence d'une demande, part de chaque étape | `mesurer` : chronométrer une demande HTTP |
| 1 | le goulot, et la croissance du coût avec l'historique H ; qualité selon H | profil `cProfile`, `cout_enrich` |
| 2 | effet d'un index client | `client_index` et `client_rows` dans `features.py` |
| 3 | capacité, file d'attente au pic, configuration la moins chère | test à K clients, générateur à débit imposé, `en_attente`, `respecte_sla` |
| 4 | montée en charge par paliers | vos paliers, `dimensionner` (capacité utile, réplicas) |

## À faire AVANT la séance (≈ 10 min, plusieurs centaines de Mo à télécharger)

- Python 3.13 via [uv](https://docs.astral.sh/uv/) ; **macOS : `brew install libomp`** (OpenMP pour XGBoost).
- Docker (Docker Desktop, OrbStack ou Colima). La VM Docker doit avoir au moins 2 CPU et 4 Go :
  `colima start --cpu 4 --memory 6`.

```bash
./scripts/setup.sh          # uv sync, données déterministes (~4 s), contrôle, dépôt git local
uv run jupyter lab notebooks/td1_experiences.ipynb
```

Plateformes testées : macOS 26 (Apple M5, OrbStack). Linux : même image Docker, non chronométré. Windows : non pris en charge.

## Fichiers

| Fichier | Rôle |
| --- | --- |
| `app.py` | service FastAPI : `GET /health`, `POST /score`, durées par étape et en-têtes `X-Server-Total-Ms` |
| `features.py` | historique client, agrégats (`query`, `client_aggregates`), enrichissement, matrice ; **étape 2 à compléter** |
| `scoring.py` | chemin métier unique `score()` (appel direct et API) |
| `bench.py` | plomberie : relancer le service via votre `scripts/deploy.sh` (`restart_server`), demandes prêtes à envoyer (`corps`), générateur de secours (`charge`), `check_golden` |
| `plots.py` | graphiques (dont `timeline` pour les tests de charge) |
| `settings.py` | variables `FEATURES`, `HISTORY_ROWS`, `XGB_NTHREAD`, `WEB_CONCURRENCY` |
| `prepare.py` | données synthétiques déterministes (lancé par `setup.sh`) |

Chaque mesure ajoute une ligne à `results/results.csv` (configuration, durées, percentiles, mémoire,
CPU, variante du code) et les latences unitaires dans `results/raw/<run_id>.csv`.

## Vérifier une correction

`uv run bench.py --check-golden` : les 500 transactions de référence (historique de 100 000 lignes,
3 largeurs de schéma) gardent les mêmes identifiants, le même ordre et les mêmes scores (tolérance 1e-6).
Reconstruire l'image avant de remesurer en HTTP ; `restart_server` prévient si l'image est plus
ancienne que le code.

## Dépannage

- `XGBoost Library (libxgboost.dylib) could not be loaded` → `brew install libomp`.
- Le service ne démarre pas → `docker logs td1-api`.
- Port 8765 déjà utilisé → `URL = "http://localhost:8766"` dans le notebook.
- `WrongService` → un autre programme répond sur ce port : même remède.
- Une mesure annonce `ATTENTION : le client de charge est lui-même saturé` → réduire les débits demandés.
- Annuler vos modifications de code : `git checkout features.py` (dépôt créé par `setup.sh`).
