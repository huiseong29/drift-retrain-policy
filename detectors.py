"""
탐지기 계열 확장

지금까지는 PSI와 KS 두 개만 보고 "지표가 성능 저하를 못 맞춘다"고 주장했다.
"더 좋은 탐지기를 쓰면 되지 않느냐"는 반론을 막으려면, 탐지기를 계열별로
넓게 깔고 계열 단위로 결과가 갈리는지 확인해야 한다.

가설:
    입력만 보는 계열은 컨셉 드리프트를 원리적으로 놓친다.
    오차 기반 계열만 잡지만, 그것은 레이블을 요구한다.
    즉 한계는 특정 지표의 성능 문제가 아니라 정보 접근의 문제다.

계열:
    A. 입력 분포      PSI, KS, Wasserstein, MMD          (레이블 불필요)
    B. 판별 기반      domain discriminator AUC            (레이블 불필요)
    C. 모델 출력      예측확률 평균/엔트로피/분포          (레이블 불필요)
    D. 오차 기반      DDM, ADWIN, Page-Hinkley            (레이블 필요)

공정성:
    A~C의 임계값은 드리프트가 없는 스트림에서 경보가 한 번도 울리지 않도록
    보정한다. 따라서 어떤 경보든 "드리프트 없는 상태에서는 보이지 않던 변화"다.
    D는 river 기본 설정을 쓰고, 같은 무드리프트 스트림에서 오탐을 함께 보고한다.

실행:
    python detectors.py
출력:
    results/detector_summary.csv
    results/fig_detector_family.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from river.drift import ADWIN, PageHinkley
from river.drift.binary import DDM

from drift import make_stream, psi_multi, perf_drop_point

warnings.filterwarnings("ignore")

OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 10
N_CAL = 5              # 임계값 보정용 무드리프트 시드 수
WINDOW, STEP = 500, 100
DRIFT_START = 3000
RNG = np.random.default_rng(0)

SCENARIOS = [
    ("data",    False, "Data / sudden"),
    ("concept", False, "Concept / sudden"),
    ("data",    True,  "Data / gradual"),
    ("concept", True,  "Concept / gradual"),
]

# 윈도우 단위로 숫자 하나를 내는 탐지기 (A~C 계열)
WINDOW_STATS = [
    ("PSI",           "A. Input distribution"),
    ("KS",            "A. Input distribution"),
    ("Wasserstein",   "A. Input distribution"),
    ("MMD",           "A. Input distribution"),
    ("Discriminator", "B. Discriminative"),
    ("PredMean",      "C. Model output"),
    ("PredEntropy",   "C. Model output"),
    ("PredKS",        "C. Model output"),
]

# 샘플 단위 스트림을 받는 탐지기 (D 계열, 레이블 필요)
STREAM_DETECTORS = ["DDM", "ADWIN", "PageHinkley"]
FAMILY_D = "D. Error-based (needs labels)"


# ----------------------------------------------------------------------
# 윈도우 통계량
# ----------------------------------------------------------------------
def _ks_multi(A, B):
    return max(ks_2samp(A[:, j], B[:, j]).statistic for j in range(A.shape[1]))


def _wasserstein_multi(A, B):
    return max(wasserstein_distance(A[:, j], B[:, j]) for j in range(A.shape[1]))


def _mmd(A, B, n=300, gamma=None):
    """RBF 커널 MMD^2. 계산량을 줄이려고 부분표본을 쓴다."""
    a = A[RNG.choice(len(A), min(n, len(A)), replace=False)]
    b = B[RNG.choice(len(B), min(n, len(B)), replace=False)]
    if gamma is None:
        gamma = 1.0 / a.shape[1]

    def k(P, Q):
        d = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d)

    return float(k(a, a).mean() + k(b, b).mean() - 2 * k(a, b).mean())


def _discriminator(A, B):
    """참조 구간과 현재 구간을 구분하는 분류기의 AUC. 0.5면 구분 불가."""
    n = min(len(A), len(B), 500)
    a = A[RNG.choice(len(A), n, replace=False)]
    b = B[RNG.choice(len(B), n, replace=False)]

    X = np.vstack([a, b])
    y = np.r_[np.zeros(n), np.ones(n)]
    idx = RNG.permutation(len(X))
    X, y = X[idx], y[idx]

    half = len(X) // 2
    clf = LogisticRegression(max_iter=500).fit(X[:half], y[:half])
    try:
        return float(roc_auc_score(y[half:], clf.predict_proba(X[half:])[:, 1]))
    except ValueError:
        return 0.5


def _entropy(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))


def window_stats(X_ref, Xw, model, ref_proba):
    """A~C 계열의 모든 통계량을 한 번에 계산."""
    proba = model.predict_proba(Xw)[:, 1]
    return {
        "PSI":           psi_multi(X_ref, Xw),
        "KS":            _ks_multi(X_ref, Xw),
        "Wasserstein":   _wasserstein_multi(X_ref, Xw),
        "MMD":           _mmd(X_ref, Xw),
        # AUC는 0.5가 "변화 없음"이므로 거리로 변환
        "Discriminator": abs(_discriminator(X_ref, Xw) - 0.5),
        "PredMean":      abs(float(proba.mean()) - float(ref_proba.mean())),
        "PredEntropy":   abs(_entropy(proba) - _entropy(ref_proba)),
        "PredKS":        float(ks_2samp(ref_proba, proba).statistic),
    }


# ----------------------------------------------------------------------
# 스트림 전체를 한 번 훑으며 모든 탐지기를 동시에 돌린다
# ----------------------------------------------------------------------
def analyze(kind, gradual, seed, thresholds=None):
    X_ref, y_ref, X_st, y_st = make_stream(
        drift_kind=kind, gradual=gradual, seed=seed
    )
    model = LogisticRegression(max_iter=1000).fit(X_ref, y_ref)
    ref_proba = model.predict_proba(X_ref)[:, 1]

    # --- A~C: 윈도우 통계량 ---
    rows = []
    for s in range(0, len(X_st) - WINDOW + 1, STEP):
        e = s + WINDOW
        Xw, yw = X_st[s:e], y_st[s:e]
        r = window_stats(X_ref, Xw, model, ref_proba)
        r["pos"] = e
        r["accuracy"] = float((model.predict(Xw) == yw).mean())
        rows.append(r)
    win = pd.DataFrame(rows)

    # --- D: 샘플 단위 오차 스트림 ---
    pred_all = model.predict(X_st)
    wrong = (pred_all != y_st).astype(int)

    stream_alarm = {}
    dets = {"DDM": DDM(), "ADWIN": ADWIN(), "PageHinkley": PageHinkley()}
    for name, det in dets.items():
        alarm = None
        for i, w in enumerate(wrong):
            det.update(int(w) if name == "DDM" else float(w))
            if det.drift_detected:
                alarm = i
                break
        stream_alarm[name] = alarm

    # --- 경보 시점 산출 ---
    drop, _ = perf_drop_point(win)
    out = {"perf_drop": drop}

    if thresholds is not None:
        for name, _family in WINDOW_STATS:
            hit = win[win[name] >= thresholds[name]]
            out[name] = None if hit.empty else int(hit.iloc[0]["pos"])
    out.update(stream_alarm)

    return out, win


# ----------------------------------------------------------------------
# 1. 무드리프트 스트림으로 임계값 보정
# ----------------------------------------------------------------------
def calibrate():
    """드리프트가 없을 때 관측되는 최댓값을 임계값으로 삼는다 (여유 5%)."""
    peaks = {name: [] for name, _ in WINDOW_STATS}
    fp = {d: 0 for d in STREAM_DETECTORS}

    for seed in range(100, 100 + N_CAL):
        res, win = analyze("none", False, seed, thresholds=None)
        for name, _ in WINDOW_STATS:
            peaks[name].append(win[name].max())
        for d in STREAM_DETECTORS:
            fp[d] += int(res[d] is not None)

    th = {name: float(np.max(v) * 1.05) for name, v in peaks.items()}
    fp_rate = {d: fp[d] / N_CAL for d in STREAM_DETECTORS}
    return th, fp_rate


# ----------------------------------------------------------------------
# 2. 본 실험
# ----------------------------------------------------------------------
def collect(thresholds):
    rows = []
    for kind, gradual, label in SCENARIOS:
        for seed in range(N_SEEDS):
            res, _ = analyze(kind, gradual, seed, thresholds)
            dropped = res["perf_drop"] is not None
            for name, family in WINDOW_STATS + [(d, FAMILY_D)
                                                for d in STREAM_DETECTORS]:
                fired = res[name] is not None
                rows.append({
                    "scenario": label,
                    "detector": name,
                    "family": family,
                    "seed": seed,
                    "fired": fired,
                    "perf_dropped": dropped,
                    "false_alarm": fired and not dropped,
                    "miss": dropped and not fired,
                })
        print(f"  done: {label}", flush=True)
    return pd.DataFrame(rows)


def summarize(raw):
    g = (raw.groupby(["family", "detector", "scenario"])
            .agg(fire_rate=("fired", "mean"),
                 perf_drop_rate=("perf_dropped", "mean"),
                 false_alarm=("false_alarm", "mean"),
                 miss=("miss", "mean"))
            .reset_index())
    return g


# ----------------------------------------------------------------------
# 3. 그림 — 계열별 발화율 히트맵
# ----------------------------------------------------------------------
def fig_family(summary):
    order = [n for n, _ in WINDOW_STATS] + STREAM_DETECTORS
    fam_of = dict(WINDOW_STATS)
    fam_of.update({d: FAMILY_D for d in STREAM_DETECTORS})

    cols = [s[2] for s in SCENARIOS]
    mat = (summary.pivot(index="detector", columns="scenario",
                         values="fire_rate")
                  .reindex(index=order, columns=cols))

    # 참조 행: 실제로 성능이 떨어졌는가
    truth = (summary.groupby("scenario")["perf_drop_rate"].first()
                    .reindex(cols))

    fig, (ax_t, ax) = plt.subplots(
        2, 1, figsize=(9.5, 8.2),
        gridspec_kw={"height_ratios": [1, 11], "hspace": 0.06},
    )

    # 위: 정답 행
    ax_t.imshow(truth.values[None, :], cmap="Blues", vmin=0, vmax=1,
                aspect="auto")
    for j, v in enumerate(truth.values):
        ax_t.text(j, 0, f"{v:.0%}", ha="center", va="center", fontsize=10,
                  color="white" if v > 0.5 else "#333", fontweight="bold")
    ax_t.set_yticks([0])
    ax_t.set_yticklabels(["performance actually degraded"], fontsize=10,
                         fontweight="bold")
    ax_t.set_xticks([])
    for sp in ax_t.spines.values():
        sp.set_visible(False)

    # 아래: 탐지기별 발화율
    im = ax.imshow(mat.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=10,
                    color="white" if v > 0.55 else "#333")

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=10)

    # 계열 경계선
    bounds, prev = [], None
    for i, d in enumerate(order):
        if prev is not None and fam_of[d] != prev:
            bounds.append(i - 0.5)
        prev = fam_of[d]
    for b in bounds:
        ax.axhline(b, color="#222", lw=2)

    # 계열 이름 (축 바깥 왼쪽에)
    seen = {}
    for i, d in enumerate(order):
        seen.setdefault(fam_of[d], []).append(i)
    for fam, idxs in seen.items():
        ax.text(-0.22, float(np.mean(idxs)), fam.split(". ")[1],
                transform=ax.get_yaxis_transform(),
                rotation=90, va="center", ha="center",
                fontsize=9.5, color="#555", fontweight="bold",
                clip_on=False)

    for sp in ax.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, ax=[ax_t, ax], fraction=0.035, pad=0.03)
    cb.set_label("Alarm rate across seeds", fontsize=10)

    fig.suptitle(
        "Only error-based detectors see concept drift —\n"
        "and those are the ones that need labels",
        fontsize=13.5, fontweight="bold", y=0.975,
    )
    plt.savefig(f"{OUT}/fig_detector_family.png", dpi=200,
                bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[1/2] calibrating thresholds on drift-free streams "
          f"({N_CAL} seeds) ...")
    th, fp_rate = calibrate()
    print("  thresholds:", {k: round(v, 3) for k, v in th.items()})
    print("  error-based false alarm on drift-free stream:", fp_rate)

    print(f"\n[2/2] running {N_SEEDS} seeds x {len(SCENARIOS)} scenarios ...")
    raw = collect(th)
    raw.to_csv(f"{OUT}/detector_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(f"{OUT}/detector_summary.csv", index=False)
    fig_family(summary)

    pivot = (summary.pivot(index="detector", columns="scenario",
                           values="fire_rate")
                    .reindex([n for n, _ in WINDOW_STATS] + STREAM_DETECTORS))
    pd.set_option("display.width", 160)
    print("\n=== alarm rate by detector ===")
    print(pivot.to_string(float_format=lambda v: f"{v:6.0%}"))
    print(f"\n-> {OUT}/ 에 csv 2개, 그림 1개 저장됨")