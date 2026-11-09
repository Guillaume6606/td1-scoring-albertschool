"""Préparation des features : historique client, enrichissement, matrice.

Le schéma (noms, ordre, types) est généré par make_schema(F) et figé dans models/schema_fF.json.
Deux fonctions portent les défauts volontaires du TD :
- enrich()        : une requête sur tout l'historique par agrégat et par transaction (Q × H) ;
                    client_index() / client_rows() sont à compléter à l'étape 2
- build_matrix()  : matrice reconstruite par concaténations successives
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FEATURES = (
    "amount",
    "log_amount",
    "hour",
    "weekday",
    "is_weekend",
    "channel",
    "merchant_category",
    "country_match",
)
WINDOWS = (7, 30, 90, 365)
N_CATEGORIES = 20
_WHERE_RE = re.compile(r"^(\w+)(?://(\d+))?(==|<|>=)(\d+)$")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str  # raw | agg | derived
    window_days: int | None = None
    where: str | None = None
    stat: str | None = None


def _agg(name, stat, window=None, where=None):
    return FeatureSpec(name, "agg", window, where, stat)


def make_schema(f: int) -> list[FeatureSpec]:
    """Schémas imbriqués : les 16 premières colonnes de F=64 sont celles de F=16, etc."""
    specs = [FeatureSpec(n, "raw") for n in RAW_FEATURES]
    specs += [
        _agg("hist_count", "count"),
        _agg("hist_amount_mean", "mean"),
        _agg("hist_amount_std", "std"),
        _agg("hist_amount_max", "max"),
        _agg("hist_days_since_last", "days_since_last"),
        _agg("hist_n_categories", "nunique_category"),
        _agg("hist_share_web", "share", None, "channel==0"),
        FeatureSpec("amount_ratio_to_mean", "derived"),
    ]
    if f > 16:
        for w in WINDOWS:
            for st in ("count", "sum", "mean", "std", "max", "min"):
                specs.append(_agg(f"hist_{st}_{w}d", st, w))
        for w in WINDOWS:
            specs += [
                _agg(f"hist_share_web_{w}d", "share", w, "channel==0"),
                _agg(f"hist_share_foreign_{w}d", "share", w, "country_match==0"),
                _agg(f"hist_n_categories_{w}d", "nunique_category", w),
                _agg(f"hist_days_active_{w}d", "days_active", w),
            ]
        for w in WINDOWS:
            specs += [
                _agg(f"hist_share_night_{w}d", "share", w, "hour<6"),
                _agg(f"hist_share_weekend_{w}d", "share", w, "weekday>=5"),
            ]
    if f > 64:
        for c in range(N_CATEGORIES):
            for st in ("count", "mean", "sum"):
                specs.append(_agg(f"cat{c}_{st}", st, None, f"merchant_category=={c}"))
        for w in (30, 90):
            for c in range(N_CATEGORIES):
                specs.append(_agg(f"cat{c}_count_{w}d", "count", w, f"merchant_category=={c}"))
        for ch in range(3):
            for w in WINDOWS:
                for st in ("count", "mean", "max"):
                    specs.append(_agg(f"ch{ch}_{st}_{w}d", st, w, f"channel=={ch}"))
        for b in range(6):
            for st in ("count", "mean"):
                specs.append(_agg(f"hour{b}_{st}", st, None, f"hour//4=={b}"))
        for d in range(7):
            for st in ("count", "mean"):
                specs.append(_agg(f"wd{d}_{st}", st, None, f"weekday=={d}"))
        for q in (10, 25, 50, 75, 90):
            specs.append(_agg(f"hist_amount_p{q}", f"p{q}"))
        for w in (7, 365):
            for c in range(N_CATEGORIES):
                specs.append(_agg(f"cat{c}_count_{w}d", "count", w, f"merchant_category=={c}"))
        for c in range(N_CATEGORIES):
            for st in ("std", "max"):
                specs.append(_agg(f"cat{c}_{st}", st, None, f"merchant_category=={c}"))
    if len(specs) < f:
        raise ValueError(f"schema generator produces {len(specs)} < {f} features")
    return specs[:f]


def schema_to_json(specs: list[FeatureSpec], f: int, t0: pd.Timestamp) -> dict:
    cols = []
    for s in specs:
        d = asdict(s)
        d["dtype"] = "float32"
        d["missing"] = (
            "never" if s.kind == "raw" else "NaN si le client n'a aucune ligne dans la fenêtre"
        )
        cols.append(d)
    return {"features": f, "t0": t0.isoformat(), "model": f"model_f{f}.ubj", "columns": cols}


def load_schema(f: int, models_dir: Path) -> list[FeatureSpec]:
    d = json.loads((models_dir / f"schema_f{f}.json").read_text())
    keys = ("name", "kind", "window_days", "where", "stat")
    return [FeatureSpec(**{k: c[k] for k in keys}) for c in d["columns"]]


def load_history(h_rows: int, data_dir: Path) -> pd.DataFrame:
    """Les h_rows transactions les plus récentes avant T0 (instantané figé)."""
    hist = pd.read_parquet(data_dir / "history.parquet").tail(h_rows).reset_index(drop=True)
    hist["hour"] = hist["timestamp"].dt.hour.astype(np.int8)
    hist["weekday"] = hist["timestamp"].dt.weekday.astype(np.int8)
    hist["ts_ns"] = hist["timestamp"].dt.as_unit("ns").astype(np.int64)
    return hist


def raw_features(tx: pd.DataFrame) -> pd.DataFrame:
    ts = tx["timestamp"]
    return pd.DataFrame(
        {
            "amount": tx["amount"].to_numpy(dtype=np.float64),
            "log_amount": np.log1p(tx["amount"].to_numpy(dtype=np.float64)),
            "hour": ts.dt.hour.to_numpy(),
            "weekday": ts.dt.weekday.to_numpy(),
            "is_weekend": (ts.dt.weekday >= 5).to_numpy(dtype=np.int8),
            "channel": tx["channel"].to_numpy(dtype=np.int8),
            "merchant_category": tx["merchant_category"].to_numpy(dtype=np.int8),
            "country_match": tx["country_match"].to_numpy(dtype=np.int8),
        }
    )


def where_mask(df: pd.DataFrame, where: str) -> np.ndarray:
    """Masque booléen d'une condition du schéma (ex. "merchant_category==3", "hour//4==2")."""
    col, div, op, val = _WHERE_RE.match(where).groups()
    s = df[col].to_numpy()
    if div:
        s = s // int(div)
    v = int(val)
    return {"==": s == v, "<": s < v, ">=": s >= v}[op]


DAY_NS = 86_400 * 10**9


def query(
    history: pd.DataFrame, cid: int, window_days: int | None, where: str | None, t0: pd.Timestamp
) -> np.ndarray:
    """Lignes du client `cid` dans la fenêtre, filtrées par `where` : comme une requête SQL sans
    index, elle examine toutes les lignes de `history`."""
    m = history["client_id"].to_numpy() == cid
    if window_days is not None:
        m &= history["ts_ns"].to_numpy() >= t0.value - window_days * DAY_NS
    if where is not None:
        m &= where_mask(history, where)
    return m


def _stat(history: pd.DataFrame, m: np.ndarray, stat: str, t0: pd.Timestamp) -> float:
    if stat == "count":
        return float(m.sum())
    amt = history["amount"].to_numpy()[m]
    if len(amt) == 0:
        return np.nan
    if stat in ("sum", "mean", "max", "min"):
        return float(getattr(amt, stat)())
    if stat == "std":
        return float(amt.std(ddof=1)) if len(amt) > 1 else np.nan
    if stat == "days_since_last":
        return float((t0.value - history["ts_ns"].to_numpy()[m].max()) // DAY_NS)
    if stat == "nunique_category":
        return float(len(np.unique(history["merchant_category"].to_numpy()[m])))
    if stat == "days_active":
        return float(len(np.unique(history["ts_ns"].to_numpy()[m] // DAY_NS)))
    if stat.startswith("p"):
        return float(np.percentile(amt, int(stat[1:])))
    raise ValueError(stat)


def client_aggregates(
    history: pd.DataFrame, cid: int, specs: list[FeatureSpec], t0: pd.Timestamp
) -> dict:
    """Agrégats historiques du client `cid` : une requête sur `history` par agrégat."""
    if not (history["client_id"].to_numpy() == cid).any():
        return {s.name: np.nan for s in specs}
    out = {}
    for s in specs:
        if s.stat == "share":
            m = query(history, cid, s.window_days, None, t0)
            n = m.sum()
            out[s.name] = float((m & where_mask(history, s.where)).sum() / n) if n else np.nan
        else:
            m = query(history, cid, s.window_days, s.where, t0)
            out[s.name] = _stat(history, m, s.stat, t0)
    return out


# ------------------------------------------------------------- étape 2 du TD : correction du défaut A
_INDEX_CACHE: dict[tuple, dict] = {}


def client_index(history: pd.DataFrame) -> dict | None:
    """{client_id: positions de ses lignes dans history}, calculé UNE seule fois par historique.

    Tant que cette fonction renvoie None, enrich() interroge tout l'historique (version livrée).
    Indices : history.groupby("client_id").indices donne exactement ce dictionnaire ;
    le garder dans _INDEX_CACHE (clé : (id(history), len(history))) pour ne pas le recalculer
    à chaque lot.
    """
    # À COMPLÉTER (étape 2)
    return None


def client_rows(history: pd.DataFrame, index: dict, cid: int) -> pd.DataFrame:
    """Les seules lignes du client cid, via l'index (DataFrame vide s'il n'a pas d'historique).
    Indice : history.iloc[positions] ; history.iloc[:0] est un DataFrame vide."""
    # À COMPLÉTER (étape 2)
    raise NotImplementedError("client_rows : à compléter (étape 2)")


def enrich(
    transactions: pd.DataFrame, history: pd.DataFrame, schema: list[FeatureSpec], t0: pd.Timestamp
) -> list[dict]:
    """Une ligne (dict) par transaction, dans l'ordre d'entrée, clés = schéma + transaction_id."""
    agg_specs = [s for s in schema if s.kind == "agg"]
    raw = raw_features(transactions)
    index = client_index(history)
    rows = []
    for tid, cid, r in zip(
        transactions["transaction_id"],
        transactions["client_id"],
        raw.itertuples(index=False),
        strict=True,
    ):
        feats = r._asdict()
        # sans index : tout l'historique, pour chaque transaction et chaque agrégat
        source = history if index is None else client_rows(history, index, cid)
        feats.update(client_aggregates(source, cid, agg_specs, t0))
        feats["amount_ratio_to_mean"] = feats["amount"] / feats["hist_amount_mean"]
        feats["transaction_id"] = int(tid)
        rows.append(feats)
    return rows


def build_matrix(rows: list[dict], schema: list[FeatureSpec]) -> np.ndarray:
    """Matrice (B, F) float32 dans l'ordre du schéma."""
    names = [s.name for s in schema]
    if not rows:
        return np.empty((0, len(names)), dtype=np.float32)
    df = pd.DataFrame([rows[0]])
    for row in rows[1:]:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)  # une copie par ligne
    return df[names].astype("float32").to_numpy()
