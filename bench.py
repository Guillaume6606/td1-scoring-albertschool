"""Client de mesure du TD : appel direct ou HTTP, durées par étape, percentiles, ressources.

    uv run bench.py --mode direct --n 1000 --h 100000 --f 64 --batch 256 --label essai
    uv run bench.py --mode http --n 10000 --h 100000 --f 64 --batch 256 --concurrency 4
    uv run bench.py --profile --n 1000 --h 100000 --f 64 --batch 256
    uv run bench.py --sweep history            # balayage nommé (voir SWEEPS)
    uv run bench.py --check-golden             # les scores sont-ils inchangés après correction ?

Le client HTTP est en boucle fermée : chacun des K clients attend sa réponse avant d'envoyer la
requête suivante ; le débit d'arrivée s'adapte donc au service. K n'est pas un débit imposé.
Chaque exécution ajoute une ligne à results/results.csv.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import hashlib
import inspect
import io
import json
import os
import platform
import pstats
import socket
import subprocess
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
import numpy as np
import pandas as pd
import psutil

import features as ft
import scoring
import settings
from settings import ROOT, T0

RESULTS_CSV = ROOT / "results" / "results.csv"
PIDFILE = ROOT / "results" / ".api.pid"
CONTAINER = "td1-api"
DEPLOY = ROOT / "scripts" / "deploy.sh"
RESULT_COLUMNS = [
    "run_id",
    "ts_utc",
    "label",
    "host",
    "platform",
    "cpu_count",
    "cpu_quota",
    "mem_limit",
    "git_sha",
    "git_dirty",
    "variant",
    "mode",
    "n",
    "h",
    "f",
    "batch",
    "concurrency",
    "workers",
    "nthread",
    "reps",
    "warmup",
    "status",
    "model_sha",
    "wall_s",
    "rows_per_s",
    "req_per_s",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "p95_reliable",
    "n_requests",
    "errors",
    "timeouts",
    "bytes_sent",
    "bytes_recv",
    "client_encode_ms",
    "client_decode_ms",
    "srv_enrich_ms",
    "srv_matrix_ms",
    "srv_predict_ms",
    "srv_handler_ms",
    "srv_total_ms",
    "client_cpu_pct",
    "rss_start_mb",
    "rss_peak_mb",
    "rss_mean_mb",
    "container_mem_peak_mb",
    "cpu_pct_mean",
    "logical_matrix_mb",
    "startup_s",
    "notes",
]
_cache: dict = {}
_server: dict | None = None  # dernière configuration lancée par restart_server


class ConfigMismatch(Exception):
    pass


# ----------------------------------------------------------------------------- contexte machine
def _sh(*args) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def detect_variant() -> str:
    """Corrections appliquées : fichier VARIANT s'il existe, sinon détection dans le code
    (A : étape 2 complétée, B : build_matrix modifiée)."""
    variant_file = ROOT / "VARIANT"
    if variant_file.exists():
        return variant_file.read_text().strip()
    flags = ""
    if index_enabled():
        flags += "A"
    if "concat" not in inspect.getsource(ft.build_matrix):
        flags += "B"
    return flags or "shipped"


def server_variant() -> str:
    """Variante du code qui tourne dans le service : l'image :livre est toujours le code livré."""
    if _server and str(_server.get("image", "")).endswith(":livre"):
        return "shipped"
    return detect_variant()


def index_enabled() -> bool:
    """La correction A (index des clients) est-elle en place ? Testée sur un mini-historique."""
    try:
        h = pd.DataFrame({"client_id": np.array([7, 3, 7], dtype=np.int32)})
        return ft.client_index(h) is not None
    except Exception:
        return False


def code_sha() -> str:
    """Empreinte du code métier : distingue deux versions même sans dépôt git."""
    src = b"".join((ROOT / p).read_bytes() for p in ("features.py", "scoring.py", "app.py"))
    return hashlib.sha256(src).hexdigest()[:8]


def model_sha(f: int) -> str:
    return hashlib.sha256((ROOT / "models" / f"model_f{f}.ubj").read_bytes()).hexdigest()[:8]


def machine_info() -> dict:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "git_sha": code_sha(),
        "git_dirty": bool(_sh("git", "status", "--porcelain", "--", "*.py")),
        "variant": detect_variant(),
    }


def container_running() -> bool:
    return _sh("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER) == "true"


def container_limits() -> tuple[str, str]:
    out = _sh(
        "docker", "inspect", "-f", "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}", CONTAINER
    )
    if not out:
        return "none", "none"
    nano, mem = (int(x) for x in out.split())
    return (f"{nano / 1e9:g}" if nano else "none"), (f"{mem / 2**30:g}g" if mem else "none")


def container_env(name: str) -> str:
    return _sh("docker", "exec", CONTAINER, "sh", "-c", f"echo ${name}")


def server_workers(health: dict) -> int:
    if container_running():
        return int(container_env("WEB_CONCURRENCY") or 1)
    if PIDFILE.exists():
        try:
            kids = psutil.Process(int(PIDFILE.read_text())).children()
            workers = [p for p in kids if "resource_tracker" not in " ".join(p.cmdline())]
            return max(1, len(workers))
        except psutil.Error:
            pass
    return 1


# ----------------------------------------------------------------------------- ressources
class ResourceSampler:
    """Échantillonne RSS/CPU d'un arbre de processus (psutil) ou d'un conteneur (cgroup v2)."""

    def __init__(
        self, pids: list[int] | None = None, container: str | None = None, interval_s: float = 0.25
    ):
        self.pids, self.container, self.interval = pids, container, interval_s
        self.rss: list[float] = []
        self.cmem: list[float] = []
        self.cpu_start = self.cpu_end = 0.0
        self._stop = threading.Event()

    def _procs(self):
        procs = []
        for pid in self.pids or []:
            try:
                p = psutil.Process(pid)
                procs += [p] + p.children(recursive=True)
            except psutil.Error:
                pass
        return procs

    def _cpu_seconds(self) -> float:
        if self.container:
            out = _sh("docker", "exec", self.container, "cat", "/sys/fs/cgroup/cpu.stat")
            for line in out.splitlines():
                if line.startswith("usage_usec"):
                    return int(line.split()[1]) / 1e6
            return 0.0
        total = 0.0
        for p in self._procs():
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.Error:
                pass
        return total

    def _sample(self):
        if self.container:
            out = _sh("docker", "exec", self.container, "cat", "/sys/fs/cgroup/memory.current")
            if out:
                self.cmem.append(int(out) / 2**20)
        rss = 0.0
        for p in self._procs():
            try:
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        if rss:
            self.rss.append(rss / 2**20)

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self):
        self.cpu_start = self._cpu_seconds()
        self.t_start = perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()
        self.cpu_end = self._cpu_seconds()
        self.wall = perf_counter() - self.t_start

    def summary(self) -> dict:
        return {
            "rss_start_mb": round(self.rss[0], 1) if self.rss else np.nan,
            "rss_peak_mb": round(max(self.rss), 1) if self.rss else np.nan,
            "rss_mean_mb": round(float(np.mean(self.rss)), 1) if self.rss else np.nan,
            "container_mem_peak_mb": round(max(self.cmem), 1) if self.cmem else np.nan,
            "cpu_pct_mean": round(100 * (self.cpu_end - self.cpu_start) / max(self.wall, 1e-9), 1),
        }


def _server_sampler() -> ResourceSampler:
    if container_running():
        return ResourceSampler(container=CONTAINER)
    if PIDFILE.exists():
        return ResourceSampler(pids=[int(PIDFILE.read_text())])
    return ResourceSampler(pids=[])


# ----------------------------------------------------------------------------- données / modèle
def reserve(n: int) -> pd.DataFrame:
    if "reserve" not in _cache:
        _cache["reserve"] = pd.read_parquet(ROOT / "data" / "transactions.parquet").drop(
            columns="label"
        )
    return _cache["reserve"].head(n)


def load_direct(h: int, f: int, nthread: int):
    key = ("hist", h)
    if key not in _cache:
        _cache[key] = ft.load_history(h, ROOT / "data")
    model = scoring.load_model(f, ROOT / "models", nthread)
    return _cache[key], model, ft.load_schema(f, ROOT / "models")


def corps(n: int, batch: int = 1) -> list[bytes]:
    """n corps de requête JSON prêts à envoyer à POST /score, `batch` transactions de la réserve chacun."""
    tx = reserve(n * batch)
    recs = tx.assign(timestamp=tx["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict("records")
    return [json.dumps({"transactions": recs[i * batch : (i + 1) * batch]}).encode() for i in range(n)]


def payload(batch: pd.DataFrame) -> bytes:
    recs = batch.assign(timestamp=batch["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")).to_dict(
        "records"
    )
    return json.dumps({"transactions": recs}).encode()


class WrongService(Exception):
    pass


def wait_healthy(url: str, timeout_s: float = 90) -> dict:
    """Attend le service du TD. Échoue tout de suite si un AUTRE service répond sur ce port."""
    deadline = perf_counter() + timeout_s
    last = None
    while perf_counter() < deadline:
        try:
            r = httpx.get(url + "/health", timeout=2)
            js = (
                r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            )
            if r.status_code == 200 and "history_rows" in js:
                return js
            raise WrongService(
                f"{url} répond, mais ce n'est pas le service du TD (HTTP {r.status_code} : "
                f"{r.text[:120]}). Port déjà pris : URL = "
                f"'http://localhost:8766' dans le notebook."
            )
        except httpx.HTTPError as e:
            last = repr(e)
        time.sleep(0.5)
    raise TimeoutError(f"service non disponible sur {url} : {last} (docker logs td1-api)")


# ----------------------------------------------------------------------------- mesure
def _write_row(row: dict, results_csv: Path) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    new = not results_csv.exists()
    with open(results_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_COLUMNS})


def _base_row(mode, n, h, f, batch, concurrency, workers, nthread, reps, warmup, label) -> dict:
    quota, mem = ("host", "host") if mode == "direct" else container_limits()
    info = machine_info()
    if mode != "direct":
        info["variant"] = server_variant()
    return {
        "run_id": uuid.uuid4().hex[:8],
        "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        **info,
        "cpu_quota": quota,
        "mem_limit": mem,
        "mode": mode,
        "n": n,
        "h": h,
        "f": f,
        "batch": batch,
        "concurrency": concurrency,
        "workers": workers,
        "nthread": nthread,
        "reps": reps,
        "warmup": warmup,
        "logical_matrix_mb": round(batch * f * 4 / 2**20, 3),
    }


def run(
    mode: str,
    n: int,
    h: int,
    f: int,
    batch: int,
    concurrency: int = 1,
    reps: int = 3,
    warmup: int = 1,
    label: str = "",
    url: str = "http://localhost:8765",
    timeout_s: float = 60.0,
    cap_s: float = 120.0,
    results_csv: Path = RESULTS_CSV,
    nthread: int | None = None,
    notes: str = "",
) -> dict:
    """Une configuration : `reps` passages sur les n premières transactions de la réserve, par lots de
    `batch`. Retourne la ligne de résultats (et l'ajoute au CSV)."""
    tx = reserve(n)
    batches = [tx.iloc[i : i + batch] for i in range(0, len(tx), batch)]
    records: list[dict] = []
    lock = threading.Lock()
    stop = threading.Event()
    t_begin = perf_counter()
    enc_ms: list[float] = []

    if mode == "direct":
        nthread = nthread or settings.from_env().xgb_nthread
        hist, model, schema = load_direct(h, f, nthread)
        workers = 1
        sampler = ResourceSampler(pids=[os.getpid()])

        def do(i: int, client=None) -> dict:
            t = perf_counter()
            _, s, tm = scoring.score(batches[i], hist, model, schema, T0)
            return {
                "lat": perf_counter() - t,
                "ok": True,
                "rows": len(s),
                "enrich": tm.enrich * 1e3,
                "matrix": tm.matrix * 1e3,
                "predict": tm.predict * 1e3,
                "handler": tm.total * 1e3,
                "total": tm.total * 1e3,
                "sent": 0,
                "recv": 0,
                "dec": 0.0,
            }
    elif mode == "http":
        health = wait_healthy(url, 10)
        if health["features"] != f or health["history_rows"] != h:
            raise ConfigMismatch(
                f"serveur : features={health['features']} history_rows={health['history_rows']} ; "
                f"demandé : f={f} h={h}. Relancer le service (bench.restart_server) avec la bonne configuration."
            )
        nthread = health["xgb_nthread"]
        workers = server_workers(health)
        bodies = []
        for b in batches:
            t = perf_counter()
            bodies.append(payload(b))
            enc_ms.append((perf_counter() - t) * 1e3)
        sampler = _server_sampler()

        def do(i: int, client: httpx.Client) -> dict:
            body = bodies[i]
            t = perf_counter()
            try:
                r = client.post(
                    url + "/score", content=body, headers={"content-type": "application/json"}
                )
                lat = perf_counter() - t
                if r.status_code != 200:
                    return {
                        "lat": lat,
                        "ok": False,
                        "timeout": False,
                        "rows": 0,
                        "sent": len(body),
                        "recv": len(r.content),
                    }
                t2 = perf_counter()
                js = r.json()
                tm = js["timings_ms"]
                return {
                    "lat": lat,
                    "ok": True,
                    "rows": len(js["scores"]),
                    "enrich": tm["enrich"],
                    "matrix": tm["matrix"],
                    "predict": tm["predict"],
                    "handler": tm["handler"],
                    "total": float(r.headers.get("x-server-total-ms", "nan")),
                    "sent": len(body),
                    "recv": len(r.content),
                    "dec": (perf_counter() - t2) * 1e3,
                }
            except httpx.TimeoutException:
                return {
                    "lat": perf_counter() - t,
                    "ok": False,
                    "timeout": True,
                    "rows": 0,
                    "sent": len(body),
                    "recv": 0,
                }
            except httpx.HTTPError:
                return {
                    "lat": perf_counter() - t,
                    "ok": False,
                    "timeout": False,
                    "rows": 0,
                    "sent": len(body),
                    "recv": 0,
                }
    else:
        raise ValueError(mode)

    def one_rep(record: bool, limit: int | None = None, rep: int = 0) -> bool:
        """Un passage sur les lots ; False si le plafond de durée est atteint."""
        idx = list(range(len(batches) if limit is None else min(limit, len(batches))))
        counter = iter(idx)
        counter_lock = threading.Lock()

        def worker():
            client = httpx.Client(timeout=timeout_s) if mode == "http" else None
            while not stop.is_set():
                with counter_lock:
                    i = next(counter, None)
                if i is None:
                    break
                rec = do(i, client)
                if record:
                    rec["rep"], rec["batch_idx"] = rep, i
                    with lock:
                        records.append(rec)
                if perf_counter() - t_begin > cap_s:
                    stop.set()
            if client:
                client.close()

        threads = [
            threading.Thread(target=worker) for _ in range(concurrency if mode == "http" else 1)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        return not stop.is_set()

    proc = psutil.Process()
    status = "ok"
    with sampler:
        for _ in range(warmup):
            if not one_rep(False, limit=10):
                status = "timeout"
                break
        cpu0 = proc.cpu_times()
        t_measure = perf_counter()
        if status == "ok":
            for r in range(reps):
                if not one_rep(True, rep=r):
                    status = "timeout"
                    break
        wall = perf_counter() - t_measure
    cpu1 = proc.cpu_times()

    if mode == "direct" and hasattr(ft, "TABLE_BUILD_SECONDS"):
        # coût de construction éventuel déclaré par features.py pour cette configuration (h, f)
        key = (id(hist), len(hist), tuple(x.name for x in schema if x.kind == "agg"))
        b_s = ft.TABLE_BUILD_SECONDS.get(key)
        b_mb = getattr(ft, "TABLE_MB_BY_KEY", {}).get(key)
        if b_s is not None:
            notes = f"{notes} agg_table_build_s={b_s:.3f} agg_table_mb={b_mb or 0:.1f}".strip()
    if mode == "http" and "agg_table_build_s" in health:
        notes = (
            f"{notes} server_agg_table_build_s={health['agg_table_build_s']} "
            f"server_agg_table_mb={health['agg_table_mb']}"
        ).strip()
    oks = [r for r in records if r["ok"]]
    lat_ms = np.array([r["lat"] for r in oks]) * 1e3
    timeouts = sum(1 for r in records if not r["ok"] and r.get("timeout"))
    errors = sum(1 for r in records if not r["ok"] and not r.get("timeout"))
    if records and not oks:
        status = "error"
    med = lambda k: round(float(np.median([r[k] for r in oks])), 3) if oks else np.nan  # noqa: E731
    row = _base_row(
        mode,
        n,
        h,
        f,
        batch,
        concurrency if mode == "http" else 1,
        workers,
        nthread,
        reps,
        warmup,
        label,
    )
    row.update(
        {
            "status": status,
            "model_sha": model_sha(f),
            "wall_s": round(wall, 3),
            "rows_per_s": round(sum(r["rows"] for r in oks) / wall, 1),
            "req_per_s": round(len(oks) / wall, 2),
            "p50_ms": round(float(np.percentile(lat_ms, 50)), 2) if oks else np.nan,
            "p95_ms": round(float(np.percentile(lat_ms, 95)), 2) if oks else np.nan,
            "p99_ms": round(float(np.percentile(lat_ms, 99)), 2) if oks else np.nan,
            "p95_reliable": len(oks) >= 200,
            "n_requests": len(records),
            "errors": errors,
            "timeouts": timeouts,
            "bytes_sent": sum(r["sent"] for r in records),
            "bytes_recv": sum(r["recv"] for r in records),
            "client_encode_ms": round(float(np.median(enc_ms)), 3) if enc_ms else 0.0,
            "client_decode_ms": med("dec"),
            "srv_enrich_ms": med("enrich"),
            "srv_matrix_ms": med("matrix"),
            "srv_predict_ms": med("predict"),
            "srv_handler_ms": med("handler"),
            "srv_total_ms": med("total"),
            "client_cpu_pct": round(
                100 * ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) / max(wall, 1e-9), 1
            ),
            **sampler.summary(),
            "startup_s": "",
            "notes": notes,
        }
    )
    _write_row(row, results_csv)
    raw_dir = results_csv.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).assign(run_id=row["run_id"]).to_csv(
        raw_dir / f"{row['run_id']}.csv", index=False
    )
    print(
        f"[{row['status']}] {mode} n={n} h={h} f={f} b={batch} k={row['concurrency']} w={workers} t={nthread} | "
        f"{row['wall_s']}s {row['rows_per_s']} rows/s p50={row['p50_ms']}ms p95={row['p95_ms']}ms "
        f"err={errors} to={timeouts} | srv enrich/matrix/predict={row['srv_enrich_ms']}/{row['srv_matrix_ms']}/"
        f"{row['srv_predict_ms']}ms | rss_peak={row['rss_peak_mb']}MB cmem={row['container_mem_peak_mb']}MB",
        flush=True,
    )
    return row


# ----------------------------------------------------------------------------- cycle de vie du service
def stop_server() -> None:
    subprocess.run(["bash", str(DEPLOY), "stop"], check=False)


def warn_stale_image() -> None:
    created = _sh("docker", "image", "inspect", "-f", "{{.Created}}", "td1-scoring")
    if not created:
        print("ATTENTION : image td1-scoring absente : construisez-la (docker build)")
        return
    img = datetime.fromisoformat(created.split(".")[0].replace("Z", "")).replace(tzinfo=UTC)
    newest = max((ROOT / p).stat().st_mtime for p in ("features.py", "scoring.py", "app.py"))
    if datetime.fromtimestamp(newest, UTC) > img:
        print(
            "ATTENTION : le code a changé depuis la construction de l'image : le service tourne "
            "avec l'ANCIEN code. Reconstruisez l'image."
        )


def restart_server(
    *,
    features: int,
    history_rows: int,
    workers: int,
    nthread: int,
    docker: bool = True,
    cpus: float | None = None,
    memory: str | None = None,
    url: str = "http://localhost:8765",
    results_csv: Path = RESULTS_CSV,
    label: str = "",
    limit_concurrency: int | None = None,
    image: str = "td1-scoring",
) -> dict:
    """Relance le service avec cette configuration et enregistre le coût de démarrage (ligne mode=startup).
    limit_concurrency : au-delà de ce nombre de requêtes en cours, le service répond 503 au lieu
    de les mettre en file."""
    global _server
    stop_server()
    env = dict(
        os.environ,
        WEB_CONCURRENCY=str(workers),
        XGB_NTHREAD=str(nthread),
        FEATURES=str(features),
        HISTORY_ROWS=str(history_rows),
        PORT=url.rsplit(":", 1)[-1],
        LIMIT=str(limit_concurrency or ""),
        IMAGE=image,
        CPUS=str(cpus or os.cpu_count()),
        MEMORY=memory or "8g",
        ROOT=str(ROOT),
    )
    if not DEPLOY.exists():
        raise FileNotFoundError(f"{DEPLOY} absent : écrivez-le (étape préliminaire)")
    if image == "td1-scoring":
        warn_stale_image()
    t = perf_counter()
    subprocess.run(["bash", str(DEPLOY)], env=env, check=True)
    health = wait_healthy(url, 120)
    startup = perf_counter() - t
    time.sleep(1.0)
    with _server_sampler() as s:
        time.sleep(0.6)
    row = _base_row("startup", 0, history_rows, features, 0, 0, workers, nthread, 0, 0, label)
    row.update(
        {
            "status": "ok",
            "startup_s": round(startup, 2),
            "rss_peak_mb": health["rss_mb"],
            **{k: v for k, v in s.summary().items() if k != "rss_peak_mb"},
            "notes": f"docker={docker} cpus={cpus} memory={memory} limit={limit_concurrency}"
            + (
                f" server_agg_table_build_s={health['agg_table_build_s']}"
                f" server_agg_table_mb={health['agg_table_mb']}"
                if "agg_table_build_s" in health
                else ""
            ),
        }
    )
    _write_row(row, results_csv)
    _server = dict(
        features=features,
        history_rows=history_rows,
        workers=workers,
        nthread=nthread,
        docker=docker,
        cpus=cpus,
        memory=memory,
        image=image if docker else "local",
    )
    print(
        f"[startup] {_server['image']} W={workers} T={nthread} F={features} H={history_rows} cpus={cpus} mem={memory} "
        f"limit={limit_concurrency} variante={server_variant()} : "
        f"{startup:.1f}s, rss(1 worker)={health['rss_mb']}MB, conteneur={row['container_mem_peak_mb']}MB",
        flush=True,
    )
    return row


# ----------------------------------------------------------------------------- profil
def profile(
    n: int,
    h: int,
    f: int,
    batch: int,
    out: Path = ROOT / "results" / "profile",
    label: str = "",
    nthread: int | None = None,
) -> Path:
    """cProfile d'un passage en appel direct ; écrit .prof et .txt (top 30 cumulé)."""
    hist, model, schema = load_direct(h, f, nthread or settings.from_env().xgb_nthread)
    tx = reserve(n)
    pr = cProfile.Profile()
    pr.enable()
    for i in range(0, len(tx), batch):
        scoring.score(tx.iloc[i : i + batch], hist, model, schema, T0)
    pr.disable()
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"profile_{label or f'n{n}_h{h}_f{f}_b{batch}'}"
    pr.dump_stats(stem.with_suffix(".prof"))
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(30)
    stem.with_suffix(".txt").write_text(s.getvalue())
    print(s.getvalue()[:3000])
    return stem.with_suffix(".txt")


# ----------------------------------------------------------------------------- charge imposée
def charge(
    paliers: list[tuple[float, float]],
    batch: int = 1,
    timeout_s: float = 2.0,
    url: str = "http://localhost:8765",
    label: str = "",
    results_csv: Path = RESULTS_CSV,
) -> pd.DataFrame:
    """Charge en boucle OUVERTE : les requêtes partent à heure fixe, que le service ait répondu ou non.

    paliers : [(débit demandé en req/s, durée en s), ...], ex. [(20, 10), (100, 30), (20, 15)].
    Latence = réponse - heure PRÉVUE d'envoi (une requête retardée par le client compte aussi).
    Une requête sans réponse après timeout_s est abandonnée par le client (statut "timeout").
    Retourne une ligne par requête : palier, debit_demande, t_prevu, t_fin (s depuis le début),
    latence_ms, statut (ok | timeout | http503 | ...)."""
    import asyncio

    health = wait_healthy(url, 10)
    f, h = health["features"], health["history_rows"]
    tx = reserve(max(int(sum(r * d for r, d in paliers) * batch) + batch, batch))
    tx = tx.assign(timestamp=tx["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    recs = tx.to_dict("records")
    plan, t = [], 0.0
    for k, (rate, dur) in enumerate(paliers):
        end = t + dur
        while t < end - 1e-9:
            plan.append((k, rate, t))
            t += 1.0 / rate
    bodies = [
        json.dumps({"transactions": recs[i * batch : (i + 1) * batch]}).encode()
        for i in range(len(plan))
    ]
    out: list[dict] = []
    with httpx.Client(
        timeout=10
    ) as warm:  # échauffement hors mesure : service et connexions chauds
        for body in bodies[:20]:
            warm.post(url + "/score", content=body, headers={"content-type": "application/json"})

    async def main():
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=200)
        async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as cli:
            t0 = perf_counter() + 0.3

            async def one(i, k, rate, ts):
                await asyncio.sleep(max(0.0, t0 + ts - perf_counter()))
                statut = "ok"
                try:
                    r = await cli.post(
                        url + "/score",
                        content=bodies[i],
                        headers={"content-type": "application/json"},
                    )
                    if r.status_code != 200:
                        statut = f"http{r.status_code}"
                except httpx.TimeoutException:
                    statut = "timeout"
                except httpx.HTTPError as e:
                    statut = type(e).__name__
                fin = perf_counter() - t0
                out.append(
                    dict(
                        palier=k,
                        debit_demande=rate,
                        t_prevu=round(ts, 4),
                        t_fin=round(fin, 4),
                        latence_ms=round((fin - ts) * 1e3, 2),
                        statut=statut,
                    )
                )

            await asyncio.gather(*(one(i, k, r, ts) for i, (k, r, ts) in enumerate(plan)))

    proc = psutil.Process()
    cpu0, w0 = proc.cpu_times(), perf_counter()
    with _server_sampler() as sampler:
        th = threading.Thread(target=lambda: asyncio.run(main()))  # marche aussi dans Jupyter
        th.start()
        th.join()
    cpu1, wall = proc.cpu_times(), perf_counter() - w0
    df = pd.DataFrame(out).sort_values("t_prevu", ignore_index=True)
    client_cpu = 100 * ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) / wall
    ok = df[df["statut"] == "ok"]
    row = _base_row(
        "charge",
        len(df) * batch,
        h,
        f,
        batch,
        0,
        server_workers(health),
        health["xgb_nthread"],
        1,
        0,
        label,
    )
    row.update(
        {
            "status": "ok",
            "model_sha": model_sha(f),
            "wall_s": round(wall, 3),
            "req_per_s": round(len(ok) / wall, 2),
            "rows_per_s": round(len(ok) * batch / wall, 1),
            "p50_ms": round(float(ok["latence_ms"].quantile(0.5)), 2) if len(ok) else np.nan,
            "p95_ms": round(float(ok["latence_ms"].quantile(0.95)), 2) if len(ok) else np.nan,
            "p99_ms": round(float(ok["latence_ms"].quantile(0.99)), 2) if len(ok) else np.nan,
            "p95_reliable": len(ok) >= 200,
            "n_requests": len(df),
            "errors": int((~df["statut"].isin(["ok", "timeout"])).sum()),
            "timeouts": int((df["statut"] == "timeout").sum()),
            "client_cpu_pct": round(client_cpu, 1),
            **sampler.summary(),
            "notes": "paliers="
            + ";".join(f"{r:g}x{d:g}s" for r, d in paliers)
            + f" timeout={timeout_s}",
        }
    )
    _write_row(row, results_csv)
    raw_dir = results_csv.parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.assign(run_id=row["run_id"]).to_csv(raw_dir / f"{row['run_id']}.csv", index=False)
    print(
        f"[charge] {len(df)} requêtes, {len(ok)} réussies, {row['timeouts']} timeouts, "
        f"{row['errors']} rejets/erreurs | CPU client {row['client_cpu_pct']} %, "
        f"CPU service {row['cpu_pct_mean']} %",
        flush=True,
    )
    if client_cpu > 80:
        print("ATTENTION : le client de charge est lui-même saturé ; réduire les débits demandés.")
    return df


def par_palier(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par palier : débit demandé, débit réalisé (réponses réussies par seconde), nombre de demandes."""
    rows = []
    for k, g in df.groupby("palier"):
        dur = g["t_prevu"].max() - g["t_prevu"].min() + 1 / g["debit_demande"].iloc[0]
        rows.append(
            dict(
                palier=k,
                debit_demande=g["debit_demande"].iloc[0],
                debit_realise=round(int((g["statut"] == "ok").sum()) / dur, 1),
                n=len(g),
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- balayages
def _srv(f, h, w=4, t=8, cpus=2, memory="4g"):
    return dict(features=f, history_rows=h, workers=w, nthread=t, cpus=cpus, memory=memory)


SWEEPS: dict[str, list[dict]] = {
    "volume": [
        dict(mode="direct", n=n, h=100_000, f=64, batch=256) for n in (1_000, 10_000, 100_000)
    ],
    "history": [
        dict(mode="direct", n=1_000, h=h, f=64, batch=256) for h in (10_000, 100_000, 1_000_000)
    ]
    + [
        dict(
            mode="http",
            n=1_000,
            h=100_000,
            f=64,
            batch=256,
            concurrency=1,
            server=_srv(64, 100_000),
        )
    ],
    "features": [dict(mode="direct", n=2_000, h=10_000, f=f, batch=1_024) for f in (16, 64, 256)]
    + [
        dict(
            mode="http",
            n=2_000,
            h=10_000,
            f=64,
            batch=1_024,
            concurrency=1,
            server=_srv(64, 10_000),
        )
    ],
    "batch": [dict(mode="direct", n=10_000, h=10_000, f=64, batch=b) for b in (256, 10_000)]
    + [
        dict(mode="http", n=10_000, h=10_000, f=64, batch=b, concurrency=1, server=_srv(64, 10_000))
        for b in (1, 32, 256, 1_024)
    ],
    "concurrency": [
        dict(
            mode="http",
            n=20_480,
            h=10_000,
            f=16,
            batch=256,
            concurrency=k,
            server=_srv(16, 10_000, w, t),
        )
        for (w, t) in ((4, 8), (2, 1), (1, 2))
        for k in (1, 4, 16)
    ],
    "resources": [
        dict(
            mode="http",
            n=20_480,
            h=10_000,
            f=16,
            batch=256,
            concurrency=4,
            server=_srv(16, 10_000, 2, 1, cpus=c),
        )
        for c in (1, 2, 4)
    ],
    "matrix": [
        dict(mode="direct", n=16_384, h=10_000, f=256, batch=b) for b in (256, 1_024, 4_096, 16_384)
    ],
    "memory": [
        dict(
            mode="http",
            n=1_024,
            h=1_000_000,
            f=64,
            batch=256,
            concurrency=1,
            server=_srv(64, 1_000_000, 4, 8, memory=m),
        )
        for m in ("2g", "4g")
    ],
}


def run_sweep(
    name: str,
    docker: bool = True,
    label: str = "",
    reps: int = 3,
    warmup: int = 1,
    overrides: dict | None = None,
    points: list[dict] | None = None,
    results_csv: Path = RESULTS_CSV,
) -> pd.DataFrame:
    """Exécute les points d'un balayage nommé (ou la liste `points` fournie), en relançant le service
    quand sa configuration change."""
    global _server
    rows = []
    for point in points if points is not None else SWEEPS[name]:
        p = {**point, **(overrides or {})}
        srv = p.pop("server", None)
        if os.environ.get("TD1_SMOKE"):
            p["n"], p["h"], reps, warmup = min(p["n"], 300), min(p["h"], 10_000), 1, 0
            if srv:
                srv = {**srv, "history_rows": p["h"]}
        if srv:
            srv = {**srv, "docker": docker}
            if not docker:
                srv.update(cpus=None, memory=None)
            if {**srv, "image": "td1-scoring" if docker else "local"} != _server:
                rows.append(restart_server(**srv, label=f"{label}:{name}", results_csv=results_csv))
        rows.append(
            run(**p, reps=reps, warmup=warmup, label=f"{label}:{name}", results_csv=results_csv)
        )
    return pd.DataFrame(rows)


def check_golden(models_dir: Path = ROOT / "models", atol: float = 1e-6) -> bool:
    """Compare les scores actuels aux scores de référence (mêmes données, même modèle)."""
    g = json.loads((models_dir / "golden.json").read_text())
    ok = True
    for f, ref in g["widths"].items():
        hist, model, schema = load_direct(g["h"], int(f), 1)
        ids, scores, _ = scoring.score(reserve(g["n"]), hist, model, schema, T0)
        same_ids = ids == ref["ids"]
        diff = (
            float(np.max(np.abs(np.asarray(scores) - np.asarray(ref["scores"]))))
            if same_ids
            else np.inf
        )
        good = same_ids and diff <= atol
        ok &= good
        print(
            f"F={f}: ids/ordre {'identiques' if same_ids else 'DIFFERENTS'}, max|Δscore|={diff:.2e} -> {'OK' if good else 'ECHEC'}"
        )
    return ok


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["direct", "http"], default="direct")
    ap.add_argument("--n", type=int, default=1_000)
    ap.add_argument("--h", type=int, default=100_000)
    ap.add_argument("--f", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--cap", type=float, default=120.0)
    ap.add_argument("--nthread", type=int, default=None)
    ap.add_argument("--results", type=Path, default=RESULTS_CSV)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--sweep", choices=sorted(SWEEPS))
    ap.add_argument(
        "--no-docker", action="store_true", help="relancer le service avec uvicorn local"
    )
    ap.add_argument("--check-golden", action="store_true")
    ap.add_argument(
        "--restart",
        action="store_true",
        help="relancer le service avec --workers/--nthread/--f/--h",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpus", type=float, default=None)
    ap.add_argument("--memory", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.check_golden:
        raise SystemExit(0 if check_golden() else 1)
    if a.profile:
        profile(a.n, a.h, a.f, a.batch, label=a.label, nthread=a.nthread)
    elif a.sweep:
        run_sweep(
            a.sweep,
            docker=not a.no_docker,
            label=a.label,
            reps=a.reps,
            warmup=a.warmup,
            results_csv=a.results,
        )
    elif a.restart:
        restart_server(
            features=a.f,
            history_rows=a.h,
            workers=a.workers,
            nthread=a.nthread or 8,
            docker=not a.no_docker,
            cpus=a.cpus,
            memory=a.memory,
            url=a.url,
            results_csv=a.results,
        )
    else:
        run(
            a.mode,
            a.n,
            a.h,
            a.f,
            a.batch,
            a.concurrency,
            a.reps,
            a.warmup,
            a.label,
            a.url,
            a.timeout,
            a.cap,
            a.results,
            a.nthread,
        )


if __name__ == "__main__":
    main()
