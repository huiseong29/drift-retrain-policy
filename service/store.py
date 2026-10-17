"""
SQLite 저장소.

운영 환경을 모사하는 데 필요한 최소 테이블만 둔다.

  predictions  - 들어온 요청과 그때 모델이 낸 예측
  labels       - 나중에 도착하는 정답 (지연을 재현하기 위해 별도 테이블)
  metrics      - 감시 루프가 매 주기 기록하는 지표
  events       - 경보 접수 / 확인 / 기각 / 재학습 이력
"""
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

import numpy as np

from .config import settings

_LOCK = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    features   TEXT NOT NULL,
    pred       INTEGER NOT NULL,
    proba      REAL NOT NULL,
    model_ver  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    pred_id    INTEGER PRIMARY KEY,
    ts         REAL NOT NULL,
    y          INTEGER NOT NULL,
    expedited  INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS metrics (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    n_seen     INTEGER NOT NULL,
    psi        REAL,
    ks         REAL,
    acc_lagged REAL,
    pending    INTEGER NOT NULL,
    model_ver  INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    kind       TEXT NOT NULL,
    n_seen     INTEGER NOT NULL,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(ts);
CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts);
"""


@contextmanager
def conn():
    os.makedirs(os.path.dirname(settings.db_path) or ".", exist_ok=True)
    c = sqlite3.connect(settings.db_path, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        with _LOCK:
            yield c
            c.commit()
    finally:
        c.close()


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def reset():
    """데모를 다시 돌릴 때 사용."""
    with conn() as c:
        for t in ("predictions", "labels", "metrics", "events"):
            c.execute(f"DROP TABLE IF EXISTS {t}")
        c.executescript(SCHEMA)


# ----------------------------------------------------------------------
def add_prediction(features, pred, proba, model_ver):
    with conn() as c:
        cur = c.execute(
            "INSERT INTO predictions (ts, features, pred, proba, model_ver) "
            "VALUES (?,?,?,?,?)",
            (time.time(), json.dumps(list(map(float, features))),
             int(pred), float(proba), int(model_ver)),
        )
        return cur.lastrowid


def add_label(pred_id, y, expedited=False):
    with conn() as c:
        c.execute(
            "INSERT OR REPLACE INTO labels (pred_id, ts, y, expedited) "
            "VALUES (?,?,?,?)",
            (int(pred_id), time.time(), int(y), int(expedited)),
        )


def add_metric(n_seen, psi, ks, acc_lagged, pending, model_ver):
    with conn() as c:
        c.execute(
            "INSERT INTO metrics (ts, n_seen, psi, ks, acc_lagged, pending, "
            "model_ver) VALUES (?,?,?,?,?,?,?)",
            (time.time(), int(n_seen), psi, ks, acc_lagged,
             int(pending), int(model_ver)),
        )


def add_event(kind, n_seen, detail=None):
    with conn() as c:
        c.execute(
            "INSERT INTO events (ts, kind, n_seen, detail) VALUES (?,?,?,?)",
            (time.time(), kind, int(n_seen), json.dumps(detail or {})),
        )


# ----------------------------------------------------------------------
def count_predictions():
    with conn() as c:
        return c.execute("SELECT COUNT(*) n FROM predictions").fetchone()["n"]


def recent_features(limit):
    """최근 요청의 입력값만 (레이블 불필요 — 지표 산출용)."""
    with conn() as c:
        rows = c.execute(
            "SELECT features FROM predictions ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    if not rows:
        return np.empty((0, settings.n_features))
    return np.array([json.loads(r["features"]) for r in rows][::-1])


def labeled_range(lo_id, hi_id):
    """[lo_id, hi_id) 구간에서 정답이 도착한 것만."""
    with conn() as c:
        rows = c.execute(
            "SELECT p.id, p.features, p.pred, l.y FROM predictions p "
            "JOIN labels l ON l.pred_id = p.id "
            "WHERE p.id >= ? AND p.id < ? ORDER BY p.id",
            (int(lo_id), int(hi_id)),
        ).fetchall()
    return rows


def latest_labeled(limit):
    """정답이 도착한 가장 최근 구간 (재학습용)."""
    with conn() as c:
        rows = c.execute(
            "SELECT p.features, l.y FROM predictions p "
            "JOIN labels l ON l.pred_id = p.id "
            "ORDER BY p.id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    if not rows:
        return np.empty((0, settings.n_features)), np.empty(0, dtype=int)
    X = np.array([json.loads(r["features"]) for r in rows][::-1])
    y = np.array([r["y"] for r in rows][::-1], dtype=int)
    return X, y


def max_labeled_id():
    with conn() as c:
        r = c.execute("SELECT MAX(pred_id) m FROM labels").fetchone()
    return r["m"] or 0


def recent_metrics(limit=400):
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM metrics ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows][::-1]


def recent_events(limit=50):
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(r) for r in rows]


def event_counts():
    with conn() as c:
        rows = c.execute(
            "SELECT kind, COUNT(*) n FROM events GROUP BY kind"
        ).fetchall()
    return {r["kind"]: r["n"] for r in rows}