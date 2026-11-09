"""Chemin métier unique : enrichissement -> matrice -> prédiction, utilisé par l'API et en appel direct."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb

import features as ft


@dataclass
class Timings:
    enrich: float
    matrix: float
    predict: float
    total: float


def load_model(f: int, models_dir: Path, nthread: int) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(models_dir / f"model_f{f}.ubj"))
    booster.set_param({"nthread": nthread})
    return booster


def score(
    transactions: pd.DataFrame,
    history: pd.DataFrame,
    model: xgb.Booster,
    schema: list[ft.FeatureSpec],
    t0: pd.Timestamp,
) -> tuple[list[int], np.ndarray, Timings]:
    """Scores dans l'ordre d'entrée, avec les durées par étape (secondes)."""
    t_start = perf_counter()
    rows = ft.enrich(transactions, history, schema, t0)
    t_enrich = perf_counter()
    X = ft.build_matrix(rows, schema)
    t_matrix = perf_counter()
    scores = model.inplace_predict(X, missing=np.nan) if len(X) else np.empty(0, dtype=np.float32)
    t_predict = perf_counter()
    ids = [r["transaction_id"] for r in rows]
    return (
        ids,
        scores,
        Timings(
            enrich=t_enrich - t_start,
            matrix=t_matrix - t_enrich,
            predict=t_predict - t_matrix,
            total=t_predict - t_start,
        ),
    )
