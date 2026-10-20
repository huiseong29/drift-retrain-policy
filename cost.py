"""
비용 기반 재학습 시점 결정

앞선 실험은 재학습 횟수와 누적 오분류를 따로 보고했다. 운영자가 실제로
답해야 하는 질문은 둘을 하나의 척도로 합친 것이다.

    총비용 = c_error × (누적 오분류) + c_retrain × (재학습 횟수)

여기서 중요한 것은 두 비용의 절대값이 아니라 **비율** 이다.

    r = c_retrain / c_error
      = 재학습 1회가 오분류 몇 건에 해당하는가

r 이 작으면 (재학습이 싸면) 자주 재학습하는 편이 낫고, r 이 크면 (재학습이
비싸면) 꼭 필요할 때만 해야 한다. 이 실험은 r 을 넓게 스윕하면서 다음에
답한다.

  1. 주어진 r 에서 최적 재학습 주기는 얼마인가
  2. 주어진 r 에서 어떤 정책을 써야 하는가
  3. 정책이 바뀌는 손익분기 r 은 어디인가

실행:
    python cost.py
출력:
    results/cost_period.csv / cost_policy.csv / cost_breakeven.csv
    results/fig_cost_period.png / fig_cost_policy.png
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

# 재학습 1회 = 오분류 몇 건인가. 로그 스케일로 넓게 훑는다.
RATIOS = np.unique(np.round(np.logspace(0, 3.4, 60)).astype(int))

PERIODS = [200, 300, 400, 600, 800, 1000, 1500, 2000, 3000, 6000]
POLICIES = ["never", "periodic", "psi", "perf", "hybrid"]

SCENARIOS = [
    ("data",    False, "Data / sudden"),
    ("concept", False, "Concept / sudden"),
    ("mixed",   False, "Mixed / sudden"),
    ("mixed",   True,  "Mixed / gradual"),
]

COLORS = dict(COLORS)
COLORS["hybrid"] = "#17becf"


# ----------------------------------------------------------------------
def total_cost(errors, retrains, ratio):
    """오분류 1건을 1단위로 환산한 총비용."""
    return errors + ratio * retrains


# ----------------------------------------------------------------------
# 1. 최적 재학습 주기
# ----------------------------------------------------------------------
def sweep_periods():
    rows = []
    for kind, gradual, label in SCENARIOS:
        for period in PERIODS:
            res = [simulate("periodic", kind, gradual, s, period=period)
                   for s in range(N_SEEDS)]
            rows.append({
                "scenario": label,
                "period": period,
                "errors": float(np.mean([r["errors"] for r in res])),
                "retrains": float(np.mean([r["retrains"] for r in res])),
            })
        print(f"  periods done: {label}", flush=True)
    return pd.DataFrame(rows)


def optimal_periods(period_df):
    """각 비용비에서 총비용을 최소화하는 주기."""
    rows = []
    for _, _, label in SCENARIOS:
        g = period_df[period_df.scenario == label]
        for r in RATIOS:
            cost = total_cost(g["errors"].values, g["retrains"].values, r)
            i = int(np.argmin(cost))
            rows.append({
                "scenario": label,
                "ratio": int(r),
                "best_period": int(g["period"].values[i]),
                "best_cost": float(cost[i]),
                "retrains": float(g["retrains"].values[i]),
                "errors": float(g["errors"].values[i]),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. 비용비별 최적 정책
# ----------------------------------------------------------------------
def sweep_policies():
    rows = []
    for kind, gradual, label in SCENARIOS:
        for policy in POLICIES:
            res = [simulate(policy, kind, gradual, s) for s in range(N_SEEDS)]
            rows.append({
                "scenario": label,
                "policy": policy,
                "errors": float(np.mean([r["errors"] for r in res])),
                "retrains": float(np.mean([r["retrains"] for r in res])),
                "errors_sd": float(np.std([r["errors"] for r in res], ddof=1)),
            })
        print(f"  policies done: {label}", flush=True)
    return pd.DataFrame(rows)


def best_policy_by_ratio(policy_df):
    rows = []
    for _, _, label in SCENARIOS:
        g = policy_df[policy_df.scenario == label].reset_index(drop=True)
        for r in RATIOS:
            cost = total_cost(g["errors"].values, g["retrains"].values, r)
            i = int(np.argmin(cost))
            rows.append({
                "scenario": label, "ratio": int(r),
                "best_policy": g["policy"].values[i],
                "best_cost": float(cost[i]),
                # 2위와의 격차 — 작으면 선택이 애매하다는 뜻
                "margin_vs_second": float(np.sort(cost)[1] - cost[i]),
            })
    return pd.DataFrame(rows)


def breakeven_points(best_df):
    """최적 정책이 바뀌는 비용비."""
    rows = []
    for _, _, label in SCENARIOS:
        g = best_df[best_df.scenario == label].sort_values("ratio")
        prev = None
        for _, r in g.iterrows():
            if prev is not None and r["best_policy"] != prev:
                rows.append({"scenario": label,
                             "ratio": int(r["ratio"]),
                             "from_policy": prev,
                             "to_policy": r["best_policy"]})
            prev = r["best_policy"]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------
def fig_period(period_df, opt_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # (1) 비용비별 총비용 곡선 (Mixed / sudden)
    ax = axes[0]
    label = "Mixed / sudden"
    g = period_df[period_df.scenario == label]
    for r, c in [(10, "#9467bd"), (100, "#1f77b4"), (500, "#ff7f0e"),
                 (2000, "#d62728")]:
        cost = total_cost(g["errors"].values, g["retrains"].values, r)
        ax.plot(g["period"], cost, marker="o", ms=5, lw=2.2, color=c,
                label=f"r = {r}")
        i = int(np.argmin(cost))
        ax.plot([g["period"].values[i]], [cost[i]], marker="*", ms=17,
                color=c, markeredgecolor="#111", markeredgewidth=1.1,
                zorder=5)

    ax.set_xscale("log")
    ax.set_xticks(PERIODS)
    ax.set_xticklabels([str(p) if p in (200, 400, 1000, 2000, 6000) else ""
                        for p in PERIODS], fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("Retraining period (samples)", fontsize=10)
    ax.set_ylabel("Total cost  (errors + r × retrains)", fontsize=10)
    ax.set_title(f"{label}\n★ = cost-minimising period", fontsize=11)
    ax.legend(fontsize=9, title="cost ratio", title_fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)

    # (2) 최적 주기가 비용비에 따라 어떻게 이동하는가
    ax = axes[1]
    for (_, _, label), c in zip(SCENARIOS,
                                ["#9467bd", "#1f77b4", "#d62728", "#ff7f0e"]):
        g = opt_df[opt_df.scenario == label].sort_values("ratio")
        ax.step(g["ratio"], g["best_period"], where="post", lw=2.4,
                color=c, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks(PERIODS)
    ax.set_yticklabels([str(p) for p in PERIODS], fontsize=8.5)
    ax.set_xlabel("Cost ratio  r = c_retrain / c_error", fontsize=10)
    ax.set_ylabel("Optimal retraining period", fontsize=10)
    ax.set_title("Cheaper retraining → retrain more often", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    ax.set_axisbelow(True)

    fig.suptitle("Optimal retraining interval follows directly from the "
                 "cost ratio",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(f"{OUT}/fig_cost_period.png", dpi=200)
    plt.close()


def fig_policy(policy_df, best_df, breakeven):
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(16.5, 4.8),
                             sharex=True)

    for ax, (_, _, label) in zip(axes, SCENARIOS):
        g = policy_df[policy_df.scenario == label]
        for _, row in g.iterrows():
            cost = total_cost(row["errors"], row["retrains"], RATIOS)
            ax.plot(RATIOS, cost, lw=2.4, color=COLORS[row["policy"]],
                    label=row["policy"])

        # 최적 정책이 바뀌는 지점
        be = breakeven[breakeven.scenario == label]
        for _, b in be.iterrows():
            ax.axvline(b["ratio"], color="#555", ls=":", lw=1.4)
            ax.text(b["ratio"], ax.get_ylim()[1] * 0.97,
                    f" r={b['ratio']}", fontsize=8.5, color="#555",
                    va="top", rotation=90)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cost ratio  r", fontsize=10)
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.25, which="both")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Total cost", fontsize=10)
    axes[0].legend(fontsize=9, loc="upper left")

    fig.suptitle("Which policy wins depends on how expensive retraining is",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(f"{OUT}/fig_cost_policy.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[1/2] retraining period sweep ({len(PERIODS)} periods) ...")
    period_df = sweep_periods()
    opt_df = optimal_periods(period_df)
    period_df.to_csv(f"{OUT}/cost_period_raw.csv", index=False)
    opt_df.to_csv(f"{OUT}/cost_period.csv", index=False)
    fig_period(period_df, opt_df)

    print(f"\n[2/2] policy comparison across cost ratios ...")
    policy_df = sweep_policies()
    best_df = best_policy_by_ratio(policy_df)
    be = breakeven_points(best_df)
    policy_df.to_csv(f"{OUT}/cost_policy.csv", index=False)
    best_df.to_csv(f"{OUT}/cost_best_policy.csv", index=False)
    be.to_csv(f"{OUT}/cost_breakeven.csv", index=False)
    fig_policy(policy_df, best_df, be)

    pd.set_option("display.width", 180)

    # RATIOS 는 로그 격자이므로 원하는 값에 가장 가까운 지점을 고른다
    want = [1, 10, 50, 100, 500, 1000, 2500]
    sel = sorted({int(RATIOS[np.argmin(np.abs(RATIOS - w))]) for w in want})

    print("\n=== 비용비별 최적 재학습 주기 ===")
    piv = (opt_df[opt_df.ratio.isin(sel)]
           .pivot(index="scenario", columns="ratio", values="best_period"))
    print(piv.to_string())

    print("\n=== 비용비별 최적 정책 ===")
    piv2 = (best_df[best_df.ratio.isin(sel)]
            .pivot(index="scenario", columns="ratio", values="best_policy"))
    print(piv2.to_string())

    print("\n=== 정책이 바뀌는 손익분기점 ===")
    print(be.to_string(index=False) if len(be) else "  (구간 내 전환 없음)")

    print(f"\n-> {OUT}/ 에 csv 5개, 그림 2개 저장됨")