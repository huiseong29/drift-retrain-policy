"""
FastAPI 서비스.

  POST /predict          예측 요청. 예측값과 예측 id 를 돌려준다.
  POST /label            정답 도착 (지연되어 들어온다)
  POST /bootstrap        초기 모델 학습
  GET  /labeling-policy  어느 구간의 레이블을 먼저 확보해야 하는지
  GET  /status           현재 지표, 모델 버전, 최근 이벤트
  GET  /history          대시보드용 시계열
  GET  /                 대시보드 페이지
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import store
from .config import settings
from .model import registry
from .monitor import monitor

HERE = os.path.dirname(os.path.abspath(__file__))


class PredictIn(BaseModel):
    features: list[float] = Field(..., min_length=1)


class LabelIn(BaseModel):
    pred_id: int
    y: int
    expedited: bool = False


class BootstrapIn(BaseModel):
    X: list[list[float]]
    y: list[int]


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    monitor.start()
    yield
    monitor.stop()


app = FastAPI(title="Drift-aware retraining pipeline", lifespan=lifespan)


# ----------------------------------------------------------------------
@app.post("/bootstrap")
def bootstrap(body: BootstrapIn):
    meta = registry.fit(body.X, body.y, reason="initial")
    store.add_event("deploy", store.count_predictions(), meta)
    return meta


@app.post("/predict")
def predict(body: PredictIn):
    if not registry.ready:
        raise HTTPException(409, "모델이 아직 배포되지 않음 (/bootstrap 먼저)")
    pred, proba, ver = registry.predict(body.features)
    pid = store.add_prediction(body.features, pred, proba, ver)
    return {"pred_id": pid, "pred": pred, "proba": round(proba, 4),
            "model_version": ver}


@app.post("/label")
def label(body: LabelIn):
    store.add_label(body.pred_id, body.y, body.expedited)
    return {"ok": True}


@app.get("/labeling-policy")
def labeling_policy():
    """입력 지표가 지목한 구간. 레이블 예산을 여기에 먼저 쓴다."""
    r = monitor.expedite_range()
    return {"expedite": r}


@app.get("/status")
def status():
    return {
        "model": registry.meta,
        "metrics": monitor.last,
        "event_counts": store.event_counts(),
        "events": store.recent_events(15),
        "settings": {
            "psi_threshold": settings.psi_threshold,
            "perf_tol": settings.perf_tol,
            "expedite": settings.expedite,
            "window": settings.window,
        },
    }


@app.get("/history")
def history(limit: int = 400):
    return {"metrics": store.recent_metrics(limit),
            "events": store.recent_events(60)}


@app.get("/health")
def health():
    return {"ok": True, "model_ready": registry.ready,
            "n_seen": store.count_predictions()}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(os.path.join(HERE, "dashboard.html"), encoding="utf-8") as f:
        return f.read()