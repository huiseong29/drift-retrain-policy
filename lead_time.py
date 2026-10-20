"""
예고 정확도 분석 (lead time analysis)

앞선 실험들은 각 시나리오에서 '경보 시점'과 '성능 하락 시점'을 한 번씩
비교했고, 그 결과 두 시점이 함께 관측된 사례가 없었다. 그러나 이는
시나리오를 data / concept 두 극단으로만 잡았기 때문이다.

현실의 드리프트는 입력 변화와 관계 변화가 섞여서 온다. 두 성분의 크기를
각각 독립적으로 바꾸면 '지표도 울리고 성능도 떨어지는' 조합이 생기고,
비로소 예고 시간(lead time)을 측정할 수 있다.

  lead time = (성능 하락 시점) − (지표 경보 시점)
      양수 : 지표가 성능 하락보다 먼저 울림 → 예고 성공
      0 이하: 동시 또는 뒤늦게 울림        → 예고 실패

측정:
  1. 입력 변화량 × 관계 변화량 격자에서 lead time 분포
  2. 예고 성공률과 평균 선행 시간
  3. 조기 탐지 ROC — 임계값을 바꿔가며 '하락 전 경보' 능력을 AUC 로 요약
  4. 오차 기반 탐지기(ADWIN)와의 비교

실행:
    python lead_time.py
출력:
    results/lead_raw.csv / lead_summary.csv / lead_roc.csv
    results/fig_lead_time.png / fig_lead_roc.png
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from river.drift import ADWIN

from drift import run, make_stream, perf_drop_point, first_cross

OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 10
DRIFT_START = 3000
PSI_TH, KS_TH = 0.2, 0.1

# 입력 변화량 (PSI 가 반응하는 축) × 관계 변화량 (성능이 반응하는 축)
MAGNITUDES = [0.0, 0.3, 0.6, 1.0, 1.5]
STRENGTHS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

BLUE, RED, ORANGE, GREEN = "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"


# ----------------------------------------------------------------------
def adwin_alarm(magnitude, strength, seed):
    """오차 기반 탐지기의 경보 시점 (레이블 사용)."""
    X_ref, y_ref, X_st, y_st = make_stream(
        drift_kind="mixed", magnitude=magnitude, strength=strength, seed=seed)
    model = LogisticRegression(max_iter=1000).fit(X_ref, y_ref)
    wrong = (model.predict(X_st) != y_st).astype(float)

    det = ADWIN()
    for i, w in enumerate(wrong):
        det.update(float(w))
        if det.drift_detected:
            return i
    return None


def collect():
    rows = []
    for mag in MAGNITUDES:
        for st in STRENGTHS:
            for seed in range(N_SEEDS):
                df = run("mixed", magnitude=mag, strength=st, seed=seed)
                drop, _ = perf_drop_point(df)
                rows.append({
                    "magnitude": mag,
                    "strength": st,
                    "seed": seed,
                    "perf_drop": drop,
                    "psi_alarm": first_cross(df, "psi", PSI_TH),
                    "ks_alarm": first_cross(df, "ks", KS_TH),
                    "adwin_alarm": adwin_alarm(mag, st, seed),
                    "psi_max": float(df["psi"].max()),
                })
        print(f"  magnitude {mag} done", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def add_lead(raw, detector):
    """두 시점이 모두 관측된 경우에만 lead time 을 계산한다."""
    col = f"{detector}_alarm"
    both = raw["perf_drop"].notna() & raw[col].notna()
    lead = pd.Series(np.nan, index=raw.index)
    lead[both] = raw.loc[both, "perf_drop"] - raw.loc[both, col]
    return lead, both


def summarize(raw):
    rows = []
    for detector in ["psi", "ks", "adwin"]:
        lead, both = add_lead(raw, detector)
        col = f"{detector}_alarm"

        degraded = raw["perf_drop"].notna()
        fired = raw[col].notna()

        # 예고 성공: 성능이 실제로 떨어졌고, 지표가 그보다 먼저 울린 경우
        warned = both & (lead > 0)

        rows.append({
            "detector": detector,
            "n_runs": len(raw),
            "n_degraded": int(degraded.sum()),
            "n_fired": int(fired.sum()),
            "n_both": int(both.sum()),
            # 성능이 떨어진 경우 중 몇 %를 미리 알렸는가
            "warning_rate": float(warned.sum() / max(degraded.sum(), 1)),
            "mean_lead": float(lead[warned].mean()) if warned.any() else np.nan,
            "median_lead": float(lead[warned].median()) if warned.any() else np.nan,
            "p25_lead": float(lead[warned].quantile(.25)) if warned.any() else np.nan,
            "p75_lead": float(lead[warned].quantile(.75)) if warned.any() else np.nan,
            # 울렸으나 성능은 멀쩡 (헛경보)
            "false_alarm_rate": float((fired & ~degraded).sum()
                                      / max(fired.sum(), 1)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 조기 탐지 ROC
# ----------------------------------------------------------------------
def early_detection_roc(raw_detail):
    """
    임계값을 바꿔가며 '성능 하락 전에 울렸는가'를 ROC 로 요약한다.

    양성 : 성능이 떨어진 실행에서, 하락 시점 이전에 경보가 울림
    음성 : 성능이 떨어지지 않은 실행에서 경보가 울림 (헛경보)

    AUC 0.5 는 무작위와 같다는 뜻이다.
    """
    thresholds = np.concatenate([np.arange(0.01, 0.30, 0.01),
                                 np.arange(0.30, 3.01, 0.05)])
    rows = []
    for th in thresholds:
        tp = fp = pos = neg = 0
        for rec in raw_detail:
            alarm = None
            hit = rec["df"][rec["df"]["psi"] >= th]
            if not hit.empty:
                alarm = int(hit.iloc[0]["pos"])

            if rec["drop"] is not None:
                pos += 1
                if alarm is not None and alarm < rec["drop"]:
                    tp += 1
            else:
                neg += 1
                if alarm is not None:
                    fp += 1

        rows.append({"threshold": round(float(th), 3),
                     "tpr": tp / max(pos, 1), "fpr": fp / max(neg, 1)})

    df = pd.DataFrame(rows).sort_values("fpr")
    auc = float(np.trapezoid(df["tpr"], df["fpr"]))
    # 곡선 양끝을 (0,0),(1,1) 로 닫아 면적을 보정
    x = np.concatenate([[0.0], df["fpr"].values, [1.0]])
    y = np.concatenate([[0.0], df["tpr"].values, [1.0]])
    auc = float(np.trapezoid(y, x))
    return df, auc


def collect_detail():
    """ROC 계산용으로 윈도우 결과를 보관한다."""
    detail = []
    for mag in MAGNITUDES:
        for st in STRENGTHS:
            for seed in range(N_SEEDS):
                df = run("mixed", magnitude=mag, strength=st, seed=seed)
                detail.append({"df": df, "drop": perf_drop_point(df)[0],
                               "magnitude": mag, "strength": st})
    return detail


# ----------------------------------------------------------------------
def fig_lead(raw, summary):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # (1) lead time 분포
    ax = axes[0]
    data, labels, colors = [], [], []
    for det, c in [("psi", RED), ("ks", ORANGE), ("adwin", GREEN)]:
        lead, both = add_lead(raw, det)
        v = lead[both].dropna().values
        if len(v):
            data.append(v)
            labels.append(det)
            colors.append(c)

    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for med in bp["medians"]:
            med.set_color("#222")
            med.set_linewidth(2)
    ax.axhline(0, color="#333", lw=1.6, ls="--")
    ax.set_ylabel("Lead time  (perf drop − alarm)", fontsize=10)
    ax.set_title("Positive = warned before degradation", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # (2) 예고 성공률
    ax = axes[1]
    cols = {"psi": RED, "ks": ORANGE, "adwin": GREEN}
    ax.bar(summary["detector"], summary["warning_rate"],
           color=[cols[d] for d in summary["detector"]], width=0.55)
    for i, r in summary.iterrows():
        ax.text(i, r["warning_rate"] + 0.02, f"{r['warning_rate']:.0%}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Warning rate", fontsize=10)
    ax.set_title("Share of degradations warned in advance", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # (3) 입력변화량 × 관계변화량 격자에서의 예고 성공률 (PSI)
    ax = axes[2]
    lead, both = add_lead(raw, "psi")
    raw2 = raw.assign(_warned=(both & (lead > 0)),
                      _degraded=raw["perf_drop"].notna())
    piv = (raw2.groupby(["strength", "magnitude"])
               .apply(lambda g: g["_warned"].sum() / max(g["_degraded"].sum(), 1),
                      include_groups=False)
               .unstack())
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0, vmax=1,
                   aspect="auto", origin="lower")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="#222")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=9)
    ax.set_xlabel("input shift magnitude", fontsize=10)
    ax.set_ylabel("concept drift strength", fontsize=10)
    ax.set_title("PSI warning rate by drift composition", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.045)

    fig.suptitle("Warning accuracy: input metrics warn only when the input "
                 "itself moves",
                 fontsize=13.5, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{OUT}/fig_lead_time.png", dpi=200)
    plt.close()


def fig_roc(roc, auc):
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], ls="--", color="#888", lw=1.4,
            label="random (AUC 0.50)")
    ax.plot(roc["fpr"], roc["tpr"], color=RED, lw=2.6,
            label=f"PSI (AUC {auc:.2f})")
    ax.fill_between(roc["fpr"], roc["tpr"], alpha=0.10, color=RED)

    ax.set_xlabel("False alarm rate\n(fired while performance was fine)",
                  fontsize=10)
    ax.set_ylabel("Early warning rate\n(fired before degradation)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title("Early-detection ROC across all PSI thresholds",
                 fontsize=12.5, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_lead_roc.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[1/2] lead time over {len(MAGNITUDES)}x{len(STRENGTHS)} grid "
          f"x {N_SEEDS} seeds ...")
    raw = collect()
    raw.to_csv(f"{OUT}/lead_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(f"{OUT}/lead_summary.csv", index=False)
    fig_lead(raw, summary)

    print("\n[2/2] early-detection ROC ...")
    detail = collect_detail()
    roc, auc = early_detection_roc(detail)
    roc.to_csv(f"{OUT}/lead_roc.csv", index=False)
    fig_roc(roc, auc)

    pd.set_option("display.width", 180)
    print("\n=== 예고 정확도 ===")
    show = summary.copy()
    for c in ["warning_rate", "false_alarm_rate"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    for c in ["mean_lead", "median_lead", "p25_lead", "p75_lead"]:
        show[c] = show[c].round(0)
    print(show.to_string(index=False))

    print(f"\n조기 탐지 AUC (PSI): {auc:.3f}")
    print(f"\n-> {OUT}/ 에 csv 3개, 그림 2개 저장됨")