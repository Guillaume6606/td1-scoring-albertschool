"""Configuration d'exécution lue dans l'environnement (une seule source pour l'API et bench.py)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SUPPORTED_FEATURES = (16, 64, 256)
T0 = pd.Timestamp("2025-07-01", tz="UTC")  # date de l'instantané historique


@dataclass(frozen=True)
class Settings:
    features: int = 64
    history_rows: int = 100_000
    xgb_nthread: int = 8
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"
    port: int = 8000
    t0: pd.Timestamp = T0


def from_env() -> Settings:
    features = int(os.environ.get("FEATURES", 64))
    if features not in SUPPORTED_FEATURES:
        raise ValueError(f"FEATURES={features}; valeurs possibles : 16, 64, 256")
    return Settings(
        features=features,
        history_rows=int(os.environ.get("HISTORY_ROWS", 100_000)),
        xgb_nthread=int(os.environ.get("XGB_NTHREAD", 8)),
        data_dir=Path(os.environ.get("DATA_DIR", ROOT / "data")),
        models_dir=Path(os.environ.get("MODELS_DIR", ROOT / "models")),
        port=int(os.environ.get("PORT", 8000)),
    )
