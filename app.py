"""Service de scoring : GET /health, POST /score. Modèle et historique chargés une fois par worker."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime
from time import perf_counter

import pandas as pd
import psutil
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from starlette.datastructures import MutableHeaders

import features as ft
import scoring
import settings


class Transaction(BaseModel):
    transaction_id: int
    client_id: int
    timestamp: datetime
    amount: float = Field(gt=0)
    merchant_category: int = Field(ge=0, le=19)
    channel: int = Field(ge=0, le=2)
    country_match: bool


class ScoreRequest(BaseModel):
    transactions: list[Transaction]


class ScoredTransaction(BaseModel):
    transaction_id: int
    score: float


class ScoreResponse(BaseModel):
    scores: list[ScoredTransaction]
    timings_ms: dict[str, float]
    worker_pid: int


class TimingMiddleware:
    """Mesure côté serveur du traitement complet (réception -> envoi de la réponse),
    au-delà du seul handler : décodage JSON, validation pydantic, encodage inclus."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start = perf_counter()
        scope.setdefault("state", {})

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.append("X-Server-Total-Ms", f"{(perf_counter() - start) * 1000:.3f}")
                headers.append(
                    "X-Server-Handler-Ms", f"{scope['state'].get('handler_ms', 0.0):.3f}"
                )
            await send(message)

        await self.app(scope, receive, send_wrapper)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = settings.from_env()
    app.state.cfg = cfg
    app.state.schema = ft.load_schema(cfg.features, cfg.models_dir)
    app.state.model = scoring.load_model(cfg.features, cfg.models_dir, cfg.xgb_nthread)
    app.state.history = ft.load_history(cfg.history_rows, cfg.data_dir)
    yield


app = FastAPI(title="TD1 scoring", lifespan=lifespan)
app.add_middleware(TimingMiddleware)


@app.get("/health")
def health(request: Request):
    cfg = request.app.state.cfg
    return {
        "status": "ok",
        "pid": os.getpid(),
        "features": cfg.features,
        "history_rows": int(len(request.app.state.history)),
        "xgb_nthread": cfg.xgb_nthread,
        "rss_mb": round(psutil.Process().memory_info().rss / 2**20, 1),
        "model_file": f"model_f{cfg.features}.ubj",
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    t = perf_counter()
    st = request.app.state
    tx = pd.DataFrame([x.model_dump() for x in req.transactions])
    if len(tx):
        tx["timestamp"] = pd.to_datetime(tx["timestamp"], utc=True)
        ids, scores, tm = scoring.score(tx, st.history, st.model, st.schema, st.cfg.t0)
    else:
        ids, scores, tm = [], [], scoring.Timings(0, 0, 0, 0)
    handler_ms = (perf_counter() - t) * 1000
    request.scope["state"]["handler_ms"] = handler_ms
    return ScoreResponse(
        scores=[
            ScoredTransaction(transaction_id=i, score=float(s))
            for i, s in zip(ids, scores, strict=True)
        ],
        timings_ms={
            "enrich": tm.enrich * 1000,
            "matrix": tm.matrix * 1000,
            "predict": tm.predict * 1000,
            "handler": handler_ms,
        },
        worker_pid=os.getpid(),
    )
