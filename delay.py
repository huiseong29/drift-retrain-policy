"""
레이블 지연 민감도

앞선 실험의 결론은 "입력 기반 탐지기는 컨셉 드리프트를 못 잡고,
오차 기반만 잡는다"였다. 그러면 곧바로 이런 반론이 온다.

    "그럼 오차 기반을 쓰면 되지 않나?"

이 실험이 그 반론에 답한다. 오차 기반 탐지와 성능 기반 재학습은
정답 레이블을 필요로 하는데, 실제 운영에서 레이블은 한참 뒤에 도착한다.
지연이 커질수록 이 계열의 이점이 어떻게 사라지는지 측정한다.

측정:
    1) 정책 관점 - 지연별 누적 오분류 (성능 기반 vs 입력 기반 vs 주기적)
    2) 손익분기 - 성능 기반 정책이 주기적 재학습보다 나은 지연 구간은 어디까지인가

실행:
    python delay.py
출력:
    results/delay_raw.csv
    results/delay_summary.csv
    results/fig_delay.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policy import simulate, COLORS

OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 10
DELAYS = [0, 100, 200, 400, 800, 1200, 1600, 2000]
POLICIES = ["never", "periodic", "psi", "perf"]

SCENARIOS = [
    ("concept", False, "Concept / sudden"),
    ("concept", True,  "Concept / gradual"),
    ("data",    False, "Data / sudden"),
]


# ----------------------------------------------------------------------
def collect():
    rows = []
    for kind, gradual, label in SCENARIOS:
        for delay in DELAYS:
            for policy in POLICIES:
                for seed in range(N_SEEDS):
                    r = simulate(policy, kind, gradual, seed,
                                 label_delay=delay)
                    r.update({"scenario": label, "policy": policy,
                              "delay": delay, "seed": seed})
                    rows.append(r)
        print(f"  done: {label}", flush=True)
    return pd.DataFrame(rows)


def summarize(raw):
    return (raw.groupby(["scenario", "policy", "delay"])
               .agg(errors=("errors", "mean"),
                    errors_sd=("errors", "std"),
                    retrains=("retrains", "mean"))
               .reset_index())


def benefit_curve(summary, label, policy="perf"):
    """
    레이블을 쓰는 정책이 '아무것도 안 함' 대비 줄여준 오분류 수.
    지연 0에서의 값을 1.0으로 두고 비율로 환산한다.
    이 비율이 레이블 한 건의 실질 가치다.
    """
    g = summary[summary["scenario"] == label]
    never = float(g[g["policy"] == "never"]["errors"].iloc[0])
    p = g[g["policy"] == policy].set_index("delay")["errors"]

    benefit = pd.Series({d: never - p[d] for d in DELAYS})
    base = benefit[0]
    ratio = benefit / base if base > 0 else benefit * 0
    return benefit, ratio


def half_life(ratio):
    """이점이 절반으로 줄어드는 지연값 (선형 보간)."""
    ds = list(ratio.index)
    for a, b in zip(ds, ds[1:]):
        if ratio[a] >= 0.5 > ratio[b]:
            t = (ratio[a] - 0.5) / (ratio[a] - ratio[b])
            return a + t * (b - a)
    return None


# ----------------------------------------------------------------------
def fig_delay(summary):
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(14.5, 4.8),
                             sharey=False)

    for ax, (_, _, label) in zip(axes, SCENARIOS):
        g = summary[summary["scenario"] == label]

        for policy in POLICIES:
            p = g[g["policy"] == policy].sort_values("delay")
            ax.plot(p["delay"], p["errors"], lw=2.2, marker="o", ms=5,
                    color=COLORS[policy], label=policy)
            ax.fill_between(p["delay"],
                            p["errors"] - p["errors_sd"],
                            p["errors"] + p["errors_sd"],
                            color=COLORS[policy], alpha=0.10, lw=0)

        _, ratio = benefit_curve(summary, label)
        hl = half_life(ratio)
        if hl is not None:
            ax.axvline(hl, color="#555", ls=":", lw=1.6)
            ax.text(hl, ax.get_ylim()[1], f"  half of the\n  benefit gone\n  ({hl:.0f})",
                    va="top", ha="left", fontsize=9, color="#555")

        ax.set_title(label, fontsize=11.5)
        ax.set_xlabel("Label delay (samples)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Cumulative misclassifications", fontsize=10)
    axes[0].legend(fontsize=9.5, loc="best")

    fig.suptitle(
        "Error-based policies lose their advantage as labels arrive later",
        fontsize=13.5, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{OUT}/fig_delay.png", dpi=200)
    plt.close()


def fig_benefit(summary):
    """레이블의 실질 가치가 지연에 따라 어떻게 사라지는지 한 장으로."""
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    styles = {"Concept / sudden": "-", "Concept / gradual": "--",
              "Data / sudden": ":"}
    tones = {"Concept / sudden": "#1f77b4", "Concept / gradual": "#2ca02c",
             "Data / sudden": "#9467bd"}

    for _, _, label in SCENARIOS:
        benefit, ratio = benefit_curve(summary, label)
        if benefit[0] < 50:          # 애초에 이점이 없던 시나리오는 표시만
            ax.plot(DELAYS, [0] * len(DELAYS), styles[label],
                    color=tones[label], lw=2,
                    label=f"{label}  (no benefit to begin with)")
            continue

        ax.plot(DELAYS, ratio.values, styles[label], color=tones[label],
                lw=2.4, marker="o", ms=5, label=label)

        hl = half_life(ratio)
        if hl is not None:
            ax.plot([hl], [0.5], marker="v", ms=11, color=tones[label],
                    zorder=5)
            ax.annotate(f"{hl:.0f}", (hl, 0.5),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=10, color=tones[label],
                        fontweight="bold")

    ax.axhline(0.5, color="#888", ls=":", lw=1.4)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xlabel("Label delay (samples)", fontsize=11)
    ax.set_ylabel("Benefit retained\n(vs. no retraining, relative to zero delay)",
                  fontsize=11)
    ax.set_ylim(-0.08, 1.08)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.set_title("The value of a label decays with how late it arrives",
                 fontsize=13.5, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_delay_benefit.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"running {len(DELAYS)} delays x {len(POLICIES)} policies "
          f"x {len(SCENARIOS)} scenarios x {N_SEEDS} seeds ...")
    raw = collect()
    raw.to_csv(f"{OUT}/delay_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(f"{OUT}/delay_summary.csv", index=False)
    fig_delay(summary)
    fig_benefit(summary)

    pd.set_option("display.width", 160)
    print("\n=== cumulative misclassifications by label delay ===")
    for _, _, label in SCENARIOS:
        g = summary[summary["scenario"] == label]
        piv = (g.pivot(index="policy", columns="delay", values="errors")
                .reindex(POLICIES))
        print(f"\n[{label}]")
        print(piv.to_string(float_format=lambda v: f"{v:7.0f}"))

        benefit, ratio = benefit_curve(summary, label)
        hl = half_life(ratio)
        print(f"  지연 0에서의 이점   : {benefit[0]:.0f} 건 감소")
        print(f"  지연 {max(DELAYS)}에서의 이점: {benefit[max(DELAYS)]:.0f} 건 감소 "
              f"({ratio[max(DELAYS)]:.0%} 잔존)")
        print(f"  이점 반감 지연      : "
              f"{f'{hl:.0f}' if hl is not None else '구간 내 없음'}")

    print(f"\n-> {OUT}/ 에 csv 2개, 그림 1개 저장됨")