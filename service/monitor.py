"""
감시 루프 — 하이브리드 정책의 온라인 구현.

실험 코드(policy.py 의 hybrid)와 같은 규칙을 스트리밍 환경에서 돌린다.

    1) 입력 지표(PSI)가 임계를 넘으면 재학습하지 않고 '확인 대기'에 올린다.
    2) 그 구간에 한해 레이블을 우선 확보하도록 요청한다.
    3) 레이블이 도착하면 그 구간의 실제 정확도를 확인한다.
         떨어졌으면  -> 재학습   (확인된 경보)
         멀쩡하면    -> 기각     (기준 구간만 갱신)
    4) 지표가 끝내 울리지 않는 컨셉 드리프트를 위해, 통상 지연의 성능 감시를
       안전망으로 함께 돌린다.

재학습 결정은 항상 레이블이 내린다. 입력 지표는 레이블을 어디에 먼저 쓸지만
정한다.
"""
import threading
import time

import numpy as np

from . import store
from .config import settings
from .model import registry

# 실험 코드와 같은 지표 구현을 그대로 쓴다 (두 곳에서 다르게 구현하지 않기 위해)
from drift import psi_multi, ks_multi


class Monitor:
    def __init__(self):
        self._lock = threading.RLock()
        self._thread = None
        self._stop = threading.Event()

        self.base_acc = None        # 배포 직후 기준 정확도
        self.base_sd = None         # 그 구간의 윈도우 간 변동 (노이즈 바닥)
        self._warmup = []           # 기준 산출용 관측치
        self.pending = None         # {"lo":…, "hi":…, "raised_at":…}
        self.last_retrain_at = 0    # 재학습 쿨다운용 (본 샘플 수 기준)
        self.last = {}              # 최근 지표 스냅샷

    # ------------------------------------------------------------------
    @property
    def drop_threshold(self):
        """
        성능 저하 판정선.

        고정 허용오차만 쓰면 윈도우 간 자연 변동에 걸려 드리프트가 없는
        구간에서도 재학습이 발생한다. 드리프트 없는 워밍업 구간에서 변동폭을
        측정해 두고, 고정값과 변동폭의 2배 중 큰 쪽을 쓴다.
        탐지기 임계값을 무드리프트 스트림으로 보정한 것과 같은 원리다.
        """
        if self.base_acc is None:
            return None
        margin = settings.perf_tol
        if self.base_sd is not None:
            margin = max(margin, 2.0 * self.base_sd)
        return self.base_acc - margin

    # ------------------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:                      # 루프는 죽지 않는다
                store.add_event("monitor_error", store.count_predictions(),
                                {"error": repr(e)})
            self._stop.wait(settings.monitor_interval_s)

    # ------------------------------------------------------------------
    def expedite_range(self):
        """시뮬레이터/라벨러가 조회하는 '레이블 우선 확보' 구간."""
        with self._lock:
            if self.pending is None:
                return None
            return {"lo": self.pending["lo"], "hi": self.pending["hi"],
                    "factor": settings.expedite}

    # ------------------------------------------------------------------
    def _accuracy(self, lo_id, hi_id):
        rows = store.labeled_range(lo_id, hi_id)
        if len(rows) < settings.min_window // 3:
            return None, len(rows)
        correct = sum(int(r["pred"] == r["y"]) for r in rows)
        return correct / len(rows), len(rows)

    def _retrain(self, reason, n_seen, detail=None):
        X, y = store.latest_labeled(settings.train_window)
        if len(X) < settings.min_train_rows or len(np.unique(y)) < 2:
            store.add_event("retrain_skipped", n_seen,
                            {"reason": reason, "n_labeled": int(len(X))})
            return False
        meta = registry.fit(X, y, reason=reason)
        self.last_retrain_at = n_seen
        store.add_event("retrain", n_seen, {**(detail or {}), **meta})
        return True

    # ------------------------------------------------------------------
    def tick(self):
        n_seen = store.count_predictions()
        if not registry.ready or n_seen < settings.min_window:
            return

        cur_X = store.recent_features(settings.window)
        ref_X = registry.ref_X
        if ref_X is None or len(cur_X) < settings.min_window:
            return

        psi = psi_multi(ref_X, cur_X)
        ks = ks_multi(ref_X, cur_X)

        max_lab = store.max_labeled_id()
        acc_lagged, n_lab = self._accuracy(
            max(1, max_lab - settings.window), max_lab + 1)

        # 기준 정확도: 최초 배포 직후 구간에서 변동폭까지 함께 측정한다
        with self._lock:
            if self.base_acc is None and acc_lagged is not None \
                    and registry.version == 1:
                self._warmup.append(float(acc_lagged))
                if len(self._warmup) >= settings.warmup_ticks:
                    self.base_acc = float(np.mean(self._warmup))
                    self.base_sd = float(np.std(self._warmup, ddof=1))
                    store.add_event("baseline_set", n_seen, {
                        "base_acc": round(self.base_acc, 4),
                        "base_sd": round(self.base_sd, 4),
                        "drop_threshold": round(self.drop_threshold, 4),
                        "n_obs": len(self._warmup),
                    })

        base = self.base_acc
        thresh = self.drop_threshold
        cooled = (n_seen - self.last_retrain_at) >= settings.retrain_cooldown
        retrained = False

        with self._lock:
            # (1)(2) 경보 접수 → 해당 구간 레이블 우선 확보 요청
            if self.pending is None and psi >= settings.psi_threshold and cooled:
                lo = max(1, n_seen - settings.window)
                self.pending = {"lo": lo, "hi": n_seen, "raised_at": n_seen,
                                "psi": float(psi)}
                store.add_event("alarm_raised", n_seen,
                                {"psi": round(float(psi), 4),
                                 "range": [lo, n_seen]})

            # (3) 우선 확보된 레이블로 확인
            if self.pending is not None and base is not None:
                acc_win, n_win = self._accuracy(self.pending["lo"],
                                                self.pending["hi"])
                if acc_win is not None:
                    if acc_win < thresh:
                        store.add_event("alarm_confirmed", n_seen,
                                        {"window_acc": round(acc_win, 4),
                                         "threshold": round(thresh, 4)})
                        retrained = self._retrain(
                            "confirmed_alarm", n_seen,
                            {"window_acc": round(acc_win, 4)})
                    else:
                        store.add_event("alarm_dismissed", n_seen,
                                        {"window_acc": round(acc_win, 4),
                                         "threshold": round(thresh, 4),
                                         "psi": round(float(psi), 4)})
                        # 같은 변화로 계속 울리지 않도록 기준 구간을 갱신
                        registry._ref_X = cur_X.copy()
                    self.pending = None

            # (4) 안전망: 지표가 침묵하는 경우를 위한 지연 성능 감시
            if not retrained and thresh is not None and acc_lagged is not None \
                    and cooled and acc_lagged < thresh:
                store.add_event("safety_net_triggered", n_seen,
                                {"acc_lagged": round(acc_lagged, 4),
                                 "threshold": round(thresh, 4)})
                retrained = self._retrain("safety_net", n_seen,
                                          {"acc_lagged": round(acc_lagged, 4)})

            self.last = {
                "n_seen": n_seen,
                "psi": float(psi),
                "ks": float(ks),
                "acc_lagged": acc_lagged,
                "base_acc": base,
                "drop_threshold": thresh,
                "pending": self.pending is not None,
                "model_version": registry.version,
                "n_labeled_window": n_lab,
            }

        store.add_metric(n_seen, float(psi), float(ks), acc_lagged,
                         self.pending is not None, registry.version)


monitor = Monitor()