"""Graphiques simples à partir de results/results.csv (utilisés par le notebook)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# balayage -> (abscisse, ordonnées, colonne de couleur)
AXES: dict[str, tuple[str, list[str], str | None]] = {
    "volume": ("n", ["wall_s", "rows_per_s", "rss_peak_mb"], "variant"),
    "history": ("h", ["srv_enrich_ms", "wall_s", "rss_peak_mb"], "variant"),
    "features": (
        "f",
        ["srv_enrich_ms", "srv_matrix_ms", "srv_predict_ms", "rss_peak_mb"],
        "variant",
    ),
    "batch": ("batch", ["rows_per_s", "req_per_s", "p50_ms", "bytes_sent"], "variant"),
    "concurrency": (
        "concurrency",
        ["req_per_s", "p95_ms", "srv_predict_ms", "container_mem_peak_mb"],
        "wt",
    ),
    "resources": ("cpu_quota", ["req_per_s", "p95_ms", "cpu_pct_mean"], "variant"),
    "memory": ("mem_limit", ["container_mem_peak_mb", "p95_ms"], "variant"),
    "matrix": ("batch", ["matrix_ms_per_row", "srv_matrix_ms", "rss_delta_mb"], "variant"),
}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["mode"] != "startup"].copy()
    df["wt"] = "W=" + df["workers"].astype(str) + " T=" + df["nthread"].astype(str)
    df["matrix_ms_per_row"] = df["srv_matrix_ms"] / df["batch"].replace(0, pd.NA)
    if "rss_start_mb" in df:
        df["rss_delta_mb"] = df["rss_peak_mb"] - df["rss_start_mb"]
    return df


def plot_axis(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    logx: bool = False,
    logy: bool = False,
    title: str = "",
    out: Path | None = None,
    ax=None,
):
    """Courbe y = f(x), une ligne par valeur de `hue`, médiane des répétitions."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.4))
    groups = [(None, df)] if hue is None else df.groupby(hue)
    for key, g in groups:
        agg = g.groupby(x)[y].median().reset_index()
        xs = pd.to_numeric(agg[x], errors="coerce")
        if xs.isna().any():
            ax.plot(agg[x].astype(str), agg[y], marker="o", label=None if key is None else str(key))
        else:
            ax.plot(xs, agg[y], marker="o", label=None if key is None else str(key))
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"{y} selon {x}")
    ax.grid(alpha=0.3)
    if hue is not None:
        ax.legend(fontsize=8)
    if out is not None:
        ax.figure.tight_layout()
        ax.figure.savefig(out, dpi=110)
    return ax.figure


def plot_sweep(df: pd.DataFrame, name: str, out_dir: Path | None = None) -> list[Path]:
    """Une figure par ordonnée du balayage `name`; renvoie les PNG écrits."""
    x, ys, hue = AXES[name]
    d = _prep(df)
    d = d[d["label"].str.endswith(f":{name}", na=False)] if "label" in d else d
    if d.empty:
        return []
    logx = x in ("n", "h", "batch", "concurrency")
    d = d[d[x].notna()] if x in d else d
    paths = []
    for y in ys:
        if y not in d or d[y].isna().all():
            continue
        fig = plot_axis(
            d, x, y, hue=hue if d[hue].nunique() > 1 else None, logx=logx, title=f"{name} : {y}"
        )
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            p = out_dir / f"{name}_{y}.png"
            fig.savefig(p, dpi=110, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)
    return paths


def timeline(df: pd.DataFrame, sla: dict | None = None, en_attente=None, title: str = ""):
    """Test de charge dans le temps : débits demandé / réussi par seconde, requêtes en attente
    (si la fonction en_attente(df, t) est fournie), latence de chaque requête et seuil du SLA."""
    import numpy as np

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    t_end = float(max(df["t_prevu"].max(), df["t_fin"].max())) + 1
    bins = np.arange(0, t_end + 1, 1.0)
    a1.hist(df["t_prevu"], bins=bins, histtype="step", lw=2, label="demandées / s")
    ok = df[df["statut"] == "ok"]
    a1.hist(ok["t_fin"], bins=bins, histtype="stepfilled", alpha=0.4, label="réussies / s")
    if en_attente is not None:
        grid = np.arange(0, t_end, 0.5)
        vals = [en_attente(df, t) for t in grid]
        if all(v is not None and v is not ... for v in vals):
            ax = a1.twinx()
            ax.plot(grid, vals, color="crimson", lw=1.5, label="en attente")
            ax.set_ylabel("requêtes en attente", color="crimson")
    a1.set_xlabel("temps (s)")
    a1.set_ylabel("requêtes par seconde")
    a1.legend(loc="upper left", fontsize=8)
    a1.set_title(title or "débit demandé et débit réalisé")
    a2.scatter(ok["t_prevu"], ok["latence_ms"], s=3, label="réussie")
    ko = df[df["statut"] != "ok"]
    if len(ko):
        top = max(ok["latence_ms"].max() if len(ok) else 1, 1)
        a2.scatter(
            ko["t_prevu"], np.full(len(ko), top * 1.15), s=3, c="red", marker="x", label="échec"
        )
    if sla:
        a2.axhline(sla["p95_max_ms"], color="k", ls="--", lw=1, label="SLA p95")
    a2.set_yscale("log")
    a2.set_xlabel("heure d'envoi prévue (s)")
    a2.set_ylabel("latence (ms)")
    a2.legend(loc="upper left", fontsize=8)
    a2.set_title("latence de chaque requête")
    fig.tight_layout()
    return fig
