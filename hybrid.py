"""
제안 정책 평가: 입력 지표를 트리거가 아니라 '확인 요청'으로 쓰기

앞선 실험들의 결론은 모두 부정형이었다.
  - 입력 기반 탐지기는 컨셉 드리프트를 놓친다
  - 임계값으로는 해결되지 않는다
  - 오차 기반은 잡지만 레이블이 늦게 온다

그렇다면 입력 지표는 버려야 하는가? 이 실험은 그렇지 않다고 주장한다.
입력 지표는 "재학습하라"는 명령으로 쓰면 틀리지만, "이 구간의 레이블을
먼저 확보하라"는 요청으로 쓰면 가치가 있다.

제안 정책(hybrid):
  PSI 발화 → 재학습 보류, 해당 구간 레이블 우선 확보
           → 레이블로 확인
                실제 저하 O → 재학습 (우선 확보분을 학습에도 사용)
                실제 저하 X → 기각, 기준 구간만 갱신
  PSI 침묵 → 통상 지연의 성능 감시가 안전망으로 동작

재학습 결정은 언제나 레이블이 내린다. 입력 지표는 레이블 예산을 어디에
먼저 쓸지만 정한다.

실행:
    python hybrid.py
출력:
    results/hybrid_summary.csv / hybrid_stats.csv / hybrid_delay.csv
    results/fig_hybrid.png / fig_hybrid_delay.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from policy import simulate, COLORS

OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 20
POLICIES = ["never", "periodic", "psi", "perf", "hybrid"]
DELAYS = [100, 200, 400, 800, 1200, 1600, 2000]

SCENARIOS = [
    ("data",    False, "Data / sudden"),
    ("concept", False, "Concept / sudden"),
    ("mixed",   False, "Mixed / sudden"),
    ("mixed",   True,  "Mixed / gradual"),
]

COLORS = dict(COLORS)
COLORS["hybrid"] = "#17becf"


# ----------------------------------------------------------------------
def collect():
    rows = []
    for kind, gradual, label in SCENARIOS:
        for policy in POLICIES:
            for seed in range(N_SEEDS):
                r = simulate(policy, kind, gradual, seed)
                r.update({"scenario": label, "policy": policy, "seed": seed})
                rows.append(r)
        print(f"  done: {label}", flush=True)
    return pd.DataFrame(rows)


def summarize(raw):
    g = (raw.groupby(["scenario", "policy"])
            .agg(errors=("errors", "mean"),
                 errors_sd=("errors", "std"),
                 retrains=("retrains", "mean"),
                 confirmed=("confirmed", "mean"),
                 dismissed=("dismissed", "mean"))
            .reset_index())
    g["policy"] = pd.Categorical(g["policy"], POLICIES, ordered=True)
    return g.sort_values(["scenario", "policy"]).reset_index(drop=True)


def boot_ci(x, n=5000, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), *np.percentile(means, [2.5, 97.5])


def stats(raw):
    rng = np.random.default_rng(0)
    rows = []
    for _, _, label in SCENARIOS:
        g = raw[raw["scenario"] == label]
        get = lambda p: (g[g["policy"] == p]
                         .sort_values("seed")["errors"].to_numpy(float))

        for base in ["psi", "perf"]:
            d = get(base) - get("hybrid")      # 양수면 hybrid 가 더 적게 틀림
            m, lo, hi = boot_ci(d, rng=rng)
            try:
                p = float(wilcoxon(get(base), get("hybrid")).pvalue)
            except ValueError:
                p = np.nan
            rows.append({"scenario": label,
                         "comparison": f"{base} - hybrid",
                         "mean_diff": m, "ci_low": lo, "ci_high": hi,
                         "p_value": p})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def delay_sweep():
    """지연이 길수록 '레이블 우선 확보'의 가치가 커지는지 확인."""
    rows = []
    for kind, gradual, label in [("mixed", False, "Mixed / sudden"),
                                 ("mixed", True,  "Mixed / gradual")]:
        for delay in DELAYS:
            for policy in ["perf", "hybrid", "psi"]:
                for seed in range(N_SEEDS):
                    r = simulate(policy, kind, gradual, seed,
                                 label_delay=delay)
                    rows.append({"scenario": label, "policy": policy,
                                 "delay": delay, "seed": seed,
                                 "errors": r["errors"]})
        print(f"  done: {label}", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
def fig_hybrid(summary):
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(16.5, 4.6),
                             sharey=False)

    for ax, (_, _, label) in zip(axes, SCENARIOS):
        g = summary[summary["scenario"] == label]
        for _, row in g.iterrows():
            is_ours = row["policy"] == "hybrid"
            ax.scatter(row["retrains"], row["errors"],
                       s=320 if is_ours else 190,
                       color=COLORS[row["policy"]],
                       edgecolor="#111" if is_ours else "white",
                       linewidth=2.2 if is_ours else 1.6,
                       marker="*" if is_ours else "o",
                       zorder=4 if is_ours else 3)

        ax.set_title(label, fontsize=11.5)
        ax.set_xlabel("Retrains", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.margins(x=0.22, y=0.20)

    axes[0].set_ylabel("Cumulative misclassifications", fontsize=10)

    handles = [plt.Line2D([], [], ls="", markersize=11 if p != "hybrid" else 15,
                          marker="o" if p != "hybrid" else "*",
                          markerfacecolor=COLORS[p],
                          markeredgecolor="white" if p != "hybrid" else "#111",
                          label=p + (" (proposed)" if p == "hybrid" else ""))
               for p in POLICIES]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.93))

    fig.suptitle("Confirming alarms with labels keeps the cost of 'never' "
                 "and the benefit of 'perf'",
                 fontsize=13.5, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(f"{OUT}/fig_hybrid.png", dpi=200)
    plt.close()


def fig_hybrid_delay(raw):
    g = (raw.groupby(["scenario", "policy", "delay"])["errors"]
            .mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, label in zip(axes, ["Mixed / sudden", "Mixed / gradual"]):
        s = g[g["scenario"] == label]
        for policy in ["psi", "perf", "hybrid"]:
            p = s[s["policy"] == policy].sort_values("delay")
            ax.plot(p["delay"], p["errors"], lw=2.4, marker="o", ms=5,
                    color=COLORS[policy],
                    label=policy + (" (proposed)" if policy == "hybrid" else ""))
        ax.set_title(label, fontsize=11.5)
        ax.set_xlabel("Label delay (samples)", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Cumulative misclassifications", fontsize=10)
    axes[0].legend(fontsize=9.5, loc="upper left")
    fig.suptitle("The longer labels take, the more it pays to spend them "
                 "where the input signal points",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(f"{OUT}/fig_hybrid_delay.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[1/2] policy comparison ({N_SEEDS} seeds) ...")
    raw = collect()
    raw.to_csv(f"{OUT}/hybrid_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(f"{OUT}/hybrid_summary.csv", index=False)
    fig_hybrid(summary)

    st = stats(raw)
    st.to_csv(f"{OUT}/hybrid_stats.csv", index=False)

    print(f"\n[2/2] delay sweep ...")
    d_raw = delay_sweep()
    d_raw.to_csv(f"{OUT}/hybrid_delay.csv", index=False)
    fig_hybrid_delay(d_raw)

    pd.set_option("display.width", 170)

    print("\n=== policy comparison ===")
    show = summary.copy()
    show["errors"] = show["errors"].round(0).astype(int)
    for c in ["retrains", "confirmed", "dismissed"]:
        show[c] = show[c].round(1)
    print(show[["scenario", "policy", "retrains", "errors",
                "confirmed", "dismissed"]].to_string(index=False))

    print("\n=== hybrid vs baselines (positive = hybrid makes fewer errors) ===")
    s = st.copy()
    for c in ["mean_diff", "ci_low", "ci_high"]:
        s[c] = s[c].round(1)
    s["p_value"] = s["p_value"].apply(
        lambda v: "n/a" if pd.isna(v) else f"{v:.2e}")
    print(s.to_string(index=False))

    print(f"\n-> {OUT}/ 에 csv 4개, 그림 2개 저장됨")