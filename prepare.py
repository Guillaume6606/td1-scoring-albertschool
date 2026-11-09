"""Données synthétiques déterministes.

    uv run prepare.py --data-only   # génère data/*.parquet (déterministe) + CHECKSUMS
    uv run prepare.py --verify      # vérifie les checksums

Découpage temporel : historique (< T0) ; 30 jours d'apprentissage/validation (jamais scorés en TD) ;
puis 60 jours de réserve à scorer (jamais vus à l'entraînement).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import features as ft
from settings import ROOT, T0

SEED = 20251103
N_CLIENTS = 50_000
HISTORY_ROWS = 1_000_000
TRAIN_ROWS = 200_000
RESERVE_ROWS = 1_000_000
FRAUD_RATE = 0.02
DATA_FILES = ("history.parquet", "train.parquet", "transactions.parquet")


# ----------------------------------------------------------------------------- génération
def _profiles(rng: np.random.Generator, n: int) -> dict:
    activity = rng.gamma(2.0, 1.0, n)
    return {
        "mu": rng.normal(3.5, 0.6, n),
        "sigma": rng.uniform(0.3, 0.8, n),
        "activity": activity / activity.sum(),
        "cat_cum": np.cumsum(rng.dirichlet(np.full(ft.N_CATEGORIES, 0.3), n), axis=1),
        "cat_pref": None,
        "ch_cum": np.cumsum(rng.dirichlet([3.0, 5.0, 1.0], n), axis=1),
        "foreign_rate": rng.beta(1, 30, n),
        "night_rate": rng.beta(1, 12, n),
    }


def _draw(rng, prof, n_rows, t_start, t_end, first_id, labelled):
    cid = rng.choice(len(prof["mu"]), size=n_rows, p=prof["activity"])
    day = t_start + pd.to_timedelta(rng.random(n_rows) * (t_end - t_start).days, unit="D").floor(
        "D"
    )
    night = rng.random(n_rows) < prof["night_rate"][cid]
    hour = np.where(night, rng.integers(0, 6, n_rows), rng.integers(7, 23, n_rows))
    ts = day + pd.to_timedelta(hour * 3600 + rng.integers(0, 3600, n_rows), unit="s")
    z = rng.normal(0, 1, n_rows)
    amount = np.round(np.exp(prof["mu"][cid] + prof["sigma"][cid] * z), 2)
    u = rng.random(n_rows)
    cat = (u[:, None] > prof["cat_cum"][cid]).sum(axis=1).clip(max=ft.N_CATEGORIES - 1)
    channel = (rng.random(n_rows)[:, None] > prof["ch_cum"][cid]).sum(axis=1).clip(max=2)
    foreign = rng.random(n_rows) < prof["foreign_rate"][cid]
    df = pd.DataFrame(
        {
            "transaction_id": np.arange(first_id, first_id + n_rows, dtype=np.int64),
            "client_id": cid.astype(np.int32),
            "timestamp": ts,
            "amount": amount.clip(min=0.5),
            "merchant_category": cat.astype(np.int8),
            "channel": channel.astype(np.int8),
            "country_match": ~foreign,
        }
    )
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["transaction_id"] = np.arange(first_id, first_id + n_rows, dtype=np.int64)
    if labelled:
        order = np.argsort(ts.values, kind="stable")
        z, foreign, night, cat, channel, cid = (
            a[order] for a in (z, foreign, night, cat, channel, cid)
        )
        cat_share = np.diff(
            np.concatenate([np.zeros((len(prof["mu"]), 1)), prof["cat_cum"]], axis=1), axis=1
        )
        rare_cat = cat_share[cid, cat] < 0.02
        logit = (
            1.2 * np.clip(z, 0, None)
            + 2.5 * foreign
            + 1.0 * night
            + 1.5 * rare_cat
            + 0.8 * (channel == 0)
        ) + rng.normal(0, 0.5, n_rows)
        lo, hi = -15.0, 5.0
        for _ in range(60):  # ordonnée à l'origine ajustée pour obtenir FRAUD_RATE
            c = (lo + hi) / 2
            if (1 / (1 + np.exp(-(logit + c)))).mean() > FRAUD_RATE:
                hi = c
            else:
                lo = c
        p = 1 / (1 + np.exp(-(logit + c)))
        df["label"] = (rng.random(n_rows) < p).astype(np.int8)
    return df


def generate(
    n_clients=N_CLIENTS,
    history_rows=HISTORY_ROWS,
    train_rows=TRAIN_ROWS,
    reserve_rows=RESERVE_ROWS,
    seed=SEED,
    t0=T0,
):
    rng = np.random.default_rng(seed)
    prof = _profiles(rng, n_clients)
    hist = _draw(rng, prof, history_rows, t0 - pd.Timedelta(days=365), t0, 1, labelled=False)
    train = _draw(
        rng, prof, train_rows, t0, t0 + pd.Timedelta(days=30), history_rows + 1, labelled=True
    )
    reserve = _draw(
        rng,
        prof,
        reserve_rows,
        t0 + pd.Timedelta(days=30),
        t0 + pd.Timedelta(days=90),
        history_rows + train_rows + 1,
        labelled=True,
    )
    return hist, train, reserve


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_data(data_dir: Path, hist, train, reserve) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, df in zip(DATA_FILES, (hist, train, reserve), strict=True):
        df.to_parquet(data_dir / name, index=False)
    lines = [f"{_sha256(data_dir / n)}  {n}" for n in DATA_FILES]
    (data_dir / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")


def verify_checksums(data_dir: Path) -> list[str]:
    bad = []
    for line in (data_dir / "CHECKSUMS.sha256").read_text().splitlines():
        digest, name = line.split()
        p = data_dir / name
        if not p.exists() or _sha256(p) != digest:
            bad.append(name)
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data-only", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    a = ap.parse_args(argv)
    if a.data_only:
        t = time.perf_counter()
        write_data(a.data_dir, *generate())
        print(f"data written to {a.data_dir} in {time.perf_counter() - t:.1f}s")
    if a.verify:
        bad = verify_checksums(a.data_dir)
        print("checksums OK" if not bad else f"ATTENTION checksums différents : {bad}")
        if bad and not a.data_only:
            sys.exit(1)


if __name__ == "__main__":
    main()
