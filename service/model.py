"""
배포 모델 래퍼.

재학습할 때마다 버전을 올리고 디스크에 남긴다. 운영에서 중요한 것은
"지금 어떤 버전이 응답하고 있는가"와 "그 버전이 어떤 데이터로 학습됐는가"이므로
두 가지를 함께 기록한다.
"""
import json
import os
import pickle
import threading
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import settings


class ModelRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self._model = None
        self._version = 0
        self._ref_X = None          # PSI 비교 기준 = 현재 모델의 학습 입력
        self._meta = {}
        os.makedirs(settings.model_dir, exist_ok=True)

    # ------------------------------------------------------------------
    @property
    def version(self):
        return self._version

    @property
    def ready(self):
        return self._model is not None

    @property
    def ref_X(self):
        with self._lock:
            return None if self._ref_X is None else self._ref_X.copy()

    @property
    def meta(self):
        with self._lock:
            return dict(self._meta)

    # ------------------------------------------------------------------
    def fit(self, X, y, reason="initial"):
        """새 버전을 학습하고 즉시 서빙 대상으로 교체한다."""
        X, y = np.asarray(X, float), np.asarray(y, int)
        if len(np.unique(y)) < 2:
            raise ValueError("한쪽 클래스만 있어 학습할 수 없음")

        m = LogisticRegression(max_iter=1000).fit(X, y)

        with self._lock:
            self._model = m
            self._version += 1
            self._ref_X = X.copy()
            self._meta = {
                "version": self._version,
                "reason": reason,
                "n_train": int(len(X)),
                "trained_at": time.time(),
                "train_acc": float((m.predict(X) == y).mean()),
            }
            self._persist()
            return dict(self._meta)

    def predict(self, x):
        with self._lock:
            if self._model is None:
                raise RuntimeError("모델이 아직 학습되지 않음")
            x = np.asarray(x, float).reshape(1, -1)
            proba = float(self._model.predict_proba(x)[0, 1])
            return int(proba >= 0.5), proba, self._version

    def predict_batch(self, X):
        with self._lock:
            if self._model is None:
                raise RuntimeError("모델이 아직 학습되지 않음")
            return self._model.predict(np.asarray(X, float))

    # ------------------------------------------------------------------
    def _persist(self):
        path = os.path.join(settings.model_dir, f"v{self._version:03d}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "meta": self._meta}, f)
        with open(os.path.join(settings.model_dir, "latest.json"), "w") as f:
            json.dump(self._meta, f, indent=2)


registry = ModelRegistry()