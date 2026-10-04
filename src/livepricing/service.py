"""HTTP surface for traders and monitoring.

    GET  /health                    liveness
    GET  /markets                   every market's latest quote
    GET  /markets/{game_id}         one market
    POST /overrides                 trader view on a game, validated + audited
    POST /params                    hot-swap live parameters (audited; the shadow keeps
                                    the reviewed config, so an unreviewed change alerts)
    GET  /audit                     override / parameter audit log
    GET  /metrics                   Prometheus text format

With KAFKA_BOOTSTRAP set, the engine consumes pbp.events in a background thread.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .config import Automation, ModelConfig
from .engine import PricingEngine
from .live import TotalParams, WinProbParams


class OverrideIn(BaseModel):
    game_id: int
    points: float
    trader: str
    reason: str
    decays: bool = True


class ParamsIn(BaseModel):
    who: str
    version: str
    win: dict
    total: dict


def create_app(engine: PricingEngine) -> FastAPI:
    app = FastAPI(title="Live pricing engine")

    @app.get("/health")
    def health():
        return {"ok": True, "config_version": engine.cfg.version}

    @app.get("/markets")
    def markets():
        return [m.last_quote.to_dict() for m in engine.markets.values() if m.last_quote]

    @app.get("/markets/{game_id}")
    def market(game_id: int):
        m = engine.markets.get(game_id)
        if m is None or m.last_quote is None:
            raise HTTPException(404, "no market for that game")
        return m.last_quote.to_dict()

    @app.post("/overrides")
    def override(o: OverrideIn):
        ok, why = engine.apply_override(o.game_id, o.points, o.trader, o.reason, o.decays)
        if not ok:
            raise HTTPException(422, why)
        return {"accepted": True}

    @app.post("/params")
    def params(p: ParamsIn):
        cfg = ModelConfig(p.version, WinProbParams(**p.win), TotalParams(**p.total),
                          engine.cfg.automation)
        engine.set_params(cfg, p.who)
        return {"live_version": cfg.version, "shadow_version": engine.shadow.version}

    @app.get("/audit")
    def audit():
        return engine.audit

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        lines = []
        for k, v in engine.metrics().items():
            lines.append(f"livepricing_{k} {v}")
        return "\n".join(lines) + "\n"

    return app


def _from_env() -> FastAPI:
    pregame_path = Path(os.environ.get("PREGAME_JSON", "artifacts/pregame_slate.json"))
    pregame = {int(k): v for k, v in json.loads(pregame_path.read_text()).items()}
    cfg = ModelConfig.load()
    engine = PricingEngine(cfg, pregame)
    app = create_app(engine)
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP")
    if bootstrap:
        from .bus import KafkaBus
        from .replay import run_engine

        bus = KafkaBus(bootstrap, idle_timeout=float("inf"))
        threading.Thread(target=run_engine, args=(bus, engine), daemon=True).start()
    return app


app = None
if os.environ.get("LIVEPRICING_SERVE"):
    app = _from_env()

__all__ = ["create_app", "Automation"]
