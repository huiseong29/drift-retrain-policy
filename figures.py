"""
포스터용 핵심 그림 (2x2)
왼쪽: Data drift  -> 지표는 울리는데 성능은 멀쩡 (False Alarm)
오른쪽: Concept drift -> 성능은 무너지는데 지표는 침묵 (Missed Detection)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from drift import run, perf_drop_point, first_cross

OUT = "results"
os.makedirs(OUT, exist_ok=True)

PSI_TH, KS_TH = 0.2, 0.1
DRIFT_START = 3000

BLUE, RED, ORANGE = "#1f77b4", "#d62728", "#ff7f0e"

cases = [
    # kind, title, verdict, 배지 y 위치 (정확도 곡선과 겹치지 않게)
    ("data",    "Data drift\n(input distribution shifts)",  "FALSE ALARM",      0.08),
    ("concept", "Concept drift\n(X-y relationship changes)", "MISSED DETECTION", 0.62),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 7.8), sharex=True)

for col, (kind, title, verdict, badge_y) in enumerate(cases):
    df = run(kind)
    drop_pos, base = perf_drop_point(df)
    psi_pos = first_cross(df, "psi", PSI_TH)
    ks_pos = first_cross(df, "ks", KS_TH)

    ax_top, ax_bot = axes[0, col], axes[1, col]

    # --- 성능 ---
    ax_top.plot(df["pos"], df["accuracy"], color=BLUE, lw=2.2)
    ax_top.axhline(base - 0.05, color=BLUE, ls=":", alpha=0.7)
    ax_top.set_ylim(0, 1)
    ax_top.set_title(title, fontsize=12, pad=10)
    if col == 0:
        ax_top.set_ylabel("Model accuracy", fontsize=11)

    # --- 지표 ---
    ax_bot.plot(df["pos"], df["psi"], color=RED, lw=2.2, label="PSI")
    ax_bot.plot(df["pos"], df["ks"], color=ORANGE, lw=2.2, label="KS statistic")
    ax_bot.axhline(PSI_TH, color=RED, ls=":", alpha=0.7)
    ax_bot.axhline(KS_TH, color=ORANGE, ls=":", alpha=0.7)
    ax_bot.set_ylim(0, 3.2)
    ax_bot.set_xlabel("Stream position", fontsize=11)
    if col == 0:
        ax_bot.set_ylabel("Drift metric", fontsize=11)
        ax_bot.legend(loc="upper left", fontsize=10)

    for ax in (ax_top, ax_bot):
        ax.axvline(DRIFT_START, color="gray", ls="--", alpha=0.8)
        ax.grid(alpha=0.2)

    # 드리프트 발생 구간 음영
    for ax in (ax_top, ax_bot):
        ax.axvspan(DRIFT_START, df["pos"].max(), color="gray", alpha=0.06)

    # 판정 라벨
    ax_top.text(
        0.97, badge_y, verdict,
        transform=ax_top.transAxes, ha="right", va="bottom",
        fontsize=12, fontweight="bold", color="#b00020",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff0f0", ec="#b00020", lw=1.2),
    )

    # 설명 한 줄 (곡선과 겹치지 않는 높이에)
    if kind == "data":
        msg = f"metrics fire at {min(psi_pos, ks_pos)}   |   accuracy never degrades"
        msg_y = 0.32          # PSI(≈2.1)와 KS(≈0.55) 사이
    else:
        msg = f"accuracy collapses at {drop_pos}   |   metrics never fire"
        msg_y = 0.90          # 두 곡선 모두 바닥에 깔려 있음
    ax_bot.text(
        0.97, msg_y, msg,
        transform=ax_bot.transAxes, ha="right", va="top",
        fontsize=10, color="#333",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )

fig.suptitle(
    "Distribution shift is not performance degradation",
    fontsize=16, fontweight="bold", y=0.978,
)
fig.text(
    0.5, 0.934,
    "Dashed vertical line = true drift onset (t=3000).   "
    "Dotted horizontal lines = alarm thresholds (PSI 0.2, KS 0.1).",
    ha="center", fontsize=10, color="#555",
)

plt.tight_layout(rect=[0, 0, 1, 0.915])
plt.savefig(f"{OUT}/poster_main.png", dpi=200)
print(f"saved -> {OUT}/poster_main.png")