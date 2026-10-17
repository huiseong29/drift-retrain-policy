"""
Drift detection vs. actual performance degradation
--------------------------------------------------
포스터용 예비 실험 (1-file 버전)

핵심 질문:
  드리프트 탐지 지표(PSI, KS)가 "경보"를 울린 시점과
  모델 성능이 실제로 떨어진 시점이 일치하는가?

실행:
  python drift.py
출력:
  results/drift_data.png     - 데이터 드리프트 시나리오
  results/drift_concept.png  - 컨셉 드리프트 시나리오
  results/summary.csv        - 수치 요약
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SEED = 42
OUT = "results"
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------------
# 1. 합성 데이터 생성기
# ----------------------------------------------------------------------
def make_stream(
    n_ref=2000,          # 학습용(참조) 구간 크기
    n_stream=6000,       # 운영 구간 크기
    n_features=5,
    drift_start=3000,    # 운영 구간 내에서 드리프트가 시작되는 지점
    drift_kind="data",   # "data" | "concept" | "none"
    magnitude=1.5,       # 변화 강도
    gradual=False,       # True면 서서히 변화
    strength=1.0,        # 컨셉 드리프트 강도 (0=변화 없음, 1=완전 반전)
    seed=SEED,
):
    """
    drift_kind:
      "data"    - 입력 X의 분포가 이동 (covariate shift)
                  → PSI/KS 같은 입력 기반 지표가 잡을 수 있음
      "concept" - X 분포는 그대로, X→y 관계(가중치)만 변화
                  → 입력 기반 지표는 구조적으로 못 잡음

    strength (concept 전용):
      가중치 벡터를 각도 pi*strength 만큼 회전시킨다. 크기는 보존되므로
      "신호가 약해져서" 성능이 떨어지는 것이 아니라 순수하게 관계의 방향만
      바뀐다. strength=1.0 이면 w_new = -w (완전 반전).
    """
    rng = np.random.default_rng(seed)

    w = rng.normal(0, 1, n_features)          # 원래 관계

    # w 와 직교하면서 크기가 같은 벡터를 만들어 회전축으로 쓴다.
    # 별도 난수원을 쓰는 이유: 이 벡터를 뽑느라 주 난수열이 밀리면
    # strength 를 추가하기 전에 얻은 결과가 재현되지 않는다.
    rng_rot = np.random.default_rng(seed + 10_000)
    r = rng_rot.normal(0, 1, n_features)
    r -= (r @ w) / (w @ w) * w
    w_perp = r / np.linalg.norm(r) * np.linalg.norm(w)

    theta = np.pi * strength
    w_new = np.cos(theta) * w + np.sin(theta) * w_perp

    def label(X, weights):
        logit = X @ weights
        p = 1 / (1 + np.exp(-logit))
        return (rng.random(len(X)) < p).astype(int)

    # 참조 구간 (학습용)
    X_ref = rng.normal(0, 1, (n_ref, n_features))
    y_ref = label(X_ref, w)

    # 운영 구간
    X_st = rng.normal(0, 1, (n_stream, n_features))
    y_st = np.zeros(n_stream, dtype=int)

    # 드리프트 진행 비율 (0 → 1)
    ramp = np.zeros(n_stream)
    if drift_kind != "none":
        if gradual:
            tail = n_stream - drift_start
            ramp[drift_start:] = np.linspace(0, 1, tail)
        else:
            ramp[drift_start:] = 1.0

    if drift_kind == "data":
        # 첫 두 피처의 평균을 이동
        shift = np.zeros((n_stream, n_features))
        shift[:, 0] = ramp * magnitude
        shift[:, 1] = ramp * magnitude * 0.8
        X_st = X_st + shift
        y_st = label(X_st, w)

    elif drift_kind == "concept":
        # X는 건드리지 않고 관계만 바꿈
        for i in range(n_stream):
            w_i = (1 - ramp[i]) * w + ramp[i] * w_new
            logit = X_st[i] @ w_i
            p = 1 / (1 + np.exp(-logit))
            y_st[i] = int(rng.random() < p)

    else:
        y_st = label(X_st, w)

    return X_ref, y_ref, X_st, y_st


# ----------------------------------------------------------------------
# 2. 드리프트 탐지 지표
# ----------------------------------------------------------------------
def psi(ref, cur, bins=10):
    """Population Stability Index. 참조 구간 기준으로 구간을 나눠 비율 변화를 측정."""
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)

    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_multi(X_ref, X_cur):
    """피처별 PSI의 최댓값 (가장 많이 변한 피처 기준)."""
    return max(psi(X_ref[:, j], X_cur[:, j]) for j in range(X_ref.shape[1]))


def ks_multi(X_ref, X_cur):
    """피처별 KS 통계량의 최댓값."""
    return max(ks_2samp(X_ref[:, j], X_cur[:, j]).statistic
               for j in range(X_ref.shape[1]))


# ----------------------------------------------------------------------
# 3. 실험: 윈도우를 밀면서 지표와 성능을 동시에 측정
# ----------------------------------------------------------------------
def run(drift_kind, gradual=False, window=500, step=100, magnitude=1.5,
        strength=1.0, seed=SEED):
    X_ref, y_ref, X_st, y_st = make_stream(
        drift_kind=drift_kind, gradual=gradual, magnitude=magnitude,
        strength=strength, seed=seed
    )

    model = LogisticRegression(max_iter=1000).fit(X_ref, y_ref)

    rows = []
    for start in range(0, len(X_st) - window + 1, step):
        end = start + window
        Xw, yw = X_st[start:end], y_st[start:end]

        rows.append({
            "pos": end,                                   # 윈도우 끝 위치
            "accuracy": accuracy_score(yw, model.predict(Xw)),
            "psi": psi_multi(X_ref, Xw),
            "ks": ks_multi(X_ref, Xw),
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. 경보 시점 / 성능 하락 시점 탐지
# ----------------------------------------------------------------------
def first_cross(df, col, threshold):
    """지표가 임계를 처음 넘은 위치. 없으면 None."""
    hit = df[df[col] >= threshold]
    return None if hit.empty else int(hit.iloc[0]["pos"])


def perf_drop_point(df, baseline_n=5, tolerance=0.05):
    """
    초기 구간 평균 정확도를 기준으로,
    그보다 tolerance 이상 떨어진 첫 지점.
    """
    base = df["accuracy"].iloc[:baseline_n].mean()
    hit = df[df["accuracy"] < base - tolerance]
    return (None if hit.empty else int(hit.iloc[0]["pos"])), base


# ----------------------------------------------------------------------
# 5. 그래프
# ----------------------------------------------------------------------
def plot(df, title, fname, drift_start=3000,
         psi_th=0.2, ks_th=0.1):

    drop_pos, base = perf_drop_point(df)
    psi_pos = first_cross(df, "psi", psi_th)
    ks_pos = first_cross(df, "ks", ks_th)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # 위: 성능
    ax1.plot(df["pos"], df["accuracy"], color="#1f77b4", lw=2, label="Accuracy")
    ax1.axhline(base - 0.05, color="#1f77b4", ls=":", alpha=0.6,
                label="Degradation threshold")
    ax1.set_ylabel("Accuracy")
    ax1.set_title(title)

    # 아래: 드리프트 지표
    ax2.plot(df["pos"], df["psi"], color="#d62728", lw=2, label="PSI")
    ax2.plot(df["pos"], df["ks"], color="#ff7f0e", lw=2, label="KS statistic")
    ax2.axhline(psi_th, color="#d62728", ls=":", alpha=0.6)
    ax2.axhline(ks_th, color="#ff7f0e", ls=":", alpha=0.6)
    ax2.set_ylabel("Drift metric")
    ax2.set_xlabel("Stream position")

    # 시점 표시
    for ax in (ax1, ax2):
        ax.axvline(drift_start, color="gray", ls="--", alpha=0.7)
        if drop_pos:
            ax.axvline(drop_pos, color="#1f77b4", alpha=0.35, lw=6)
        if psi_pos:
            ax.axvline(psi_pos, color="#d62728", alpha=0.35, lw=6)
        if ks_pos:
            ax.axvline(ks_pos, color="#ff7f0e", alpha=0.35, lw=6)
        ax.grid(alpha=0.25)

    ax1.legend(loc="lower left", fontsize=9)
    ax2.legend(loc="upper left", fontsize=9)

    # 주석
    note = []
    note.append(f"true drift onset: {drift_start}")
    note.append(f"perf drop: {drop_pos if drop_pos else 'not detected'}")
    note.append(f"PSI alarm: {psi_pos if psi_pos else 'not detected'}")
    note.append(f"KS alarm:  {ks_pos if ks_pos else 'not detected'}")
    fig.text(0.99, 0.01, "   |   ".join(note),
             ha="right", fontsize=9, color="#444")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(f"{OUT}/{fname}", dpi=150)
    plt.close()

    return {"perf_drop": drop_pos, "psi_alarm": psi_pos, "ks_alarm": ks_pos}


# ----------------------------------------------------------------------
# 6. 메인
# ----------------------------------------------------------------------
if __name__ == "__main__":
    summary = []

    scenarios = [
        ("data",    False, "Data drift (sudden)",    "drift_data.png"),
        ("concept", False, "Concept drift (sudden)", "drift_concept.png"),
        ("data",    True,  "Data drift (gradual)",   "drift_data_gradual.png"),
        ("concept", True,  "Concept drift (gradual)","drift_concept_gradual.png"),
    ]

    for kind, gradual, title, fname in scenarios:
        print(f"running: {title} ...", flush=True)
        df = run(kind, gradual=gradual)
        res = plot(df, title, fname)
        res.update({"scenario": title})
        summary.append(res)
        df.to_csv(f"{OUT}/raw_{fname.replace('.png', '.csv')}", index=False)

    s = pd.DataFrame(summary)[
        ["scenario", "perf_drop", "psi_alarm", "ks_alarm"]
    ]
    s["psi_lead"] = s["perf_drop"] - s["psi_alarm"]   # 양수면 성능 하락보다 먼저 경보
    s["ks_lead"] = s["perf_drop"] - s["ks_alarm"]
    s.to_csv(f"{OUT}/summary.csv", index=False)

    print()
    print(s.to_string(index=False))
    print(f"\n-> {OUT}/ 에 그래프와 csv 저장됨")