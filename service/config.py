"""파이프라인 설정값. 실험 코드(policy.py)의 상수와 의미를 맞춰 둔다."""
import os
from dataclasses import dataclass


def _env(name, default, cast=float):
    v = os.environ.get(name)
    return default if v is None else cast(v)


@dataclass(frozen=True)
class Settings:
    # 저장소
    db_path: str = os.environ.get("DRIFT_DB", "service_data/pipeline.db")
    model_dir: str = os.environ.get("DRIFT_MODELS", "service_data/models")

    # 감시 주기
    monitor_interval_s: float = _env("MONITOR_INTERVAL", 1.0)
    window: int = _env("WINDOW", 300, int)        # 지표 산출 윈도우
    min_window: int = _env("MIN_WINDOW", 150, int)

    # 하이브리드 정책 (policy.py 의 hybrid 와 같은 규칙)
    psi_threshold: float = _env("PSI_TH", 0.2)
    perf_tol: float = _env("PERF_TOL", 0.05)
    expedite: float = _env("EXPEDITE", 4.0)       # 지목 구간의 레이블 가속 배수
    warmup_ticks: int = _env("WARMUP_TICKS", 8, int)   # 기준선·변동폭 측정 횟수

    # 재학습
    train_window: int = _env("TRAIN_WINDOW", 1000, int)
    min_train_rows: int = _env("MIN_TRAIN_ROWS", 300, int)
    retrain_cooldown: int = _env("RETRAIN_COOLDOWN", 600, int)   # 샘플 수

    n_features: int = _env("N_FEATURES", 5, int)


settings = Settings()