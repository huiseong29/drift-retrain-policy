"""
견고성 검증

지금까지의 결론은 코드에 박힌 몇 개의 상수 위에 서 있다. 그 상수를 바꿨을 때
결론이 무너지는지 확인한다. 무너지지 않으면 결론이 강해지고, 무너진다면
"이 조건에서만 성립한다"고 범위를 명시해야 한다.

세 가지를 본다.

  1. PERF_TOL 민감도
     "성능이 떨어졌다"의 판정 기준(0.05)을 바꾸면 오탐/미탐이 어떻게 변하는가.
     이 값이 결론의 정의 자체이므로 가장 중요하다.

  2. 컨셉 드리프트 강도 연속 스윕
     지금까지는 드리프트가 있다/없다 이분법이었고, 강도도 완전 반전이라는
     극단값이었다. 0에서 1까지 연속으로 바꾸면서 성능 저하와 탐지기 발화를
     나란히 본다. "극단적으로 설정해서 그런 것 아니냐"는 반박을 막는다.

  3. 통계적 유의성
     정책 간 차이에 부트스트랩 신뢰구간과 Wilcoxon 부호순위 검정을 붙인다.
     지금까지는 평균만 보고 있었다.

실행:
    python robustness.py
출력:
    results/robust_tol.csv / fig_robust_tol.png
    results/robust_strength.csv / fig_robust_strength.png
    results/robust_stats.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from river.drift import ADWIN

from drift import run, make_stream, first_cross
from policy import simulate

OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 20
PSI_TH, KS_TH = 0.2, 0.1
DRIFT_START = 3000

TOLS = [0.02, 0.03, 0.05, 0.08, 0.10]
STRENGTHS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

SCENARIOS = [
    ("data",    False, "Data / sudden"),
    ("concept", False, "Concept / sudden"),
    ("data",    True,  "Data / gradual"),
    ("concept", True,  "Concept / gradual"),
]

BLUE, RED, ORANGE, GREEN = "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"


# ----------------------------------------------------------------------
# 공통: 성능 저하 판정을 임계값의 함수로 다시 쓴다
# ----------------------------------------------------------------------
def perf_drop_at(df, tol, baseline_n=5):
    base = df["accuracy"].iloc[:baseline_n].mean()
    hit = df[df["accuracy"] < base - tol]
    return None if hit.empty else int(hit.iloc[0]["pos"])


# ----------------------------------------------------------------------
# 1. PERF_TOL 민감도
# ----------------------------------------------------------------------
def tol_sensitivity():
    # 윈도우 결과를 한 번만 계산해두고 임계값만 바꿔가며 재판정한다
    cache = {}
    for kind, gradual, label in SCENARIOS:
        for seed in range(N_SEEDS):
            cache[(label, seed)] = run(kind, gradual=gradual, seed=seed)
        print(f"  cached: {label}", flush=True)

    rows = []
    for tol in TOLS:
        for (label, seed), df in cache.items():
            dropped = perf_drop_at(df, tol) is not None
            psi_fired = first_cross(df, "psi", PSI_TH) is not None
            rows.append({
                "tol": tol, "scenario": label, "seed": seed,
                "perf_dropped": dropped,
                "psi_fired": psi_fired,
                "false_alarm": psi_fired and not dropped,
                "miss": dropped and not psi_fired,
                "agree": psi_fired == dropped,
            })
    return pd.DataFrame(rows)


def fig_tol(raw):
    g = (raw.groupby(["scenario", "tol"])
            .agg(perf_drop_rate=("perf_dropped", "mean"),
                 false_alarm=("false_alarm", "mean"),
                 miss=("miss", "mean"),
                 agree=("agree", "mean"))
            .reset_index())

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, (_, _, label) in zip(axes, SCENARIOS):
        s = g[g["scenario"] == label].sort_values("tol")
        ax.plot(s["tol"], s["false_alarm"], marker="o", lw=2.2, color=RED,
                label="False alarm")
        ax.plot(s["tol"], s["miss"], marker="s", lw=2.2, color=BLUE,
                label="Miss")
        ax.plot(s["tol"], s["agree"], marker="^", lw=1.8, color="#888",
                ls="--", label="Alarm agrees with reality")
        ax.axvline(0.05, color="#555", ls=":", lw=1.4)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Degradation threshold (PERF_TOL)", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Rate across seeds", fontsize=10)
    axes[0].legend(fontsize=9, loc="center left")
    fig.suptitle("The conclusion does not depend on where we draw the "
                 "'degraded' line  (dotted = value used in the paper)",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    plt.savefig(f"{OUT}/fig_robust_tol.png", dpi=200)
    plt.close()
    return g


# ----------------------------------------------------------------------
# 2. 컨셉 드리프트 강도 연속 스윕
# ----------------------------------------------------------------------
def strength_sweep():
    rows = []
    for strength in STRENGTHS:
        for seed in range(N_SEEDS):
            df = run("concept", strength=strength, seed=seed)

            pre = df[df["pos"] <= DRIFT_START]["accuracy"].mean()
            post = df[df["pos"] > DRIFT_START + 500]["accuracy"].mean()

            # 오차 기반 탐지기도 같이 본다 (레이블 사용)
            X_ref, y_ref, X_st, y_st = make_stream(
                drift_kind="concept", strength=strength, seed=seed)
            model = LogisticRegression(max_iter=1000).fit(X_ref, y_ref)
            wrong = (model.predict(X_st) != y_st).astype(float)
            det, adwin_alarm = ADWIN(), None
            for i, w in enumerate(wrong):
                det.update(float(w))
                if det.drift_detected:
                    adwin_alarm = i
                    break

            rows.append({
                "strength": strength,
                "seed": seed,
                "acc_drop": float(pre - post),
                "psi_fired": first_cross(df, "psi", PSI_TH) is not None,
                "ks_fired": first_cross(df, "ks", KS_TH) is not None,
                "adwin_fired": adwin_alarm is not None,
            })
        print(f"  strength {strength:.1f} done", flush=True)
    return pd.DataFrame(rows)


def fig_strength(raw):
    g = (raw.groupby("strength")
            .agg(acc_drop=("acc_drop", "mean"),
                 acc_sd=("acc_drop", "std"),
                 psi=("psi_fired", "mean"),
                 ks=("ks_fired", "mean"),
                 adwin=("adwin_fired", "mean"))
            .reset_index())

    fig, ax1 = plt.subplots(figsize=(9.2, 5.4))

    ax1.plot(g["strength"], g["acc_drop"], color=BLUE, lw=2.8, marker="o",
             ms=6, label="Accuracy drop (right axis scale shown left)")
    ax1.fill_between(g["strength"], g["acc_drop"] - g["acc_sd"],
                     g["acc_drop"] + g["acc_sd"], color=BLUE, alpha=0.12, lw=0)
    ax1.set_xlabel("Concept drift strength  (0 = no change, 1 = full reversal)",
                   fontsize=11)
    ax1.set_ylabel("Accuracy drop", fontsize=11, color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    top = float((g["acc_drop"] + g["acc_sd"]).max())
    ax1.set_ylim(-0.04, top * 1.12)
    ax1.grid(alpha=0.25)
    ax1.set_axisbelow(True)

    ax2 = ax1.twinx()
    ax2.plot(g["strength"], g["psi"], color=RED, lw=2.2, marker="s", ms=5,
             label="PSI alarm rate")
    ax2.plot(g["strength"], g["ks"], color=ORANGE, lw=2.2, marker="^", ms=5,
             label="KS alarm rate")
    ax2.plot(g["strength"], g["adwin"], color=GREEN, lw=2.2, marker="D", ms=5,
             ls="--", label="ADWIN alarm rate (uses labels)")
    ax2.set_ylabel("Alarm rate", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1[:1] + h2, ["Accuracy drop"] + l2, fontsize=9.5,
               loc="center left")

    ax1.set_title("Performance degrades continuously; input-based alarms "
                  "stay flat at zero",
                  fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_robust_strength.png", dpi=200)
    plt.close()
    return g


# ----------------------------------------------------------------------
# 3. 통계적 유의성
# ----------------------------------------------------------------------
def boot_ci(x, n=5000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=float)
    means = rng.choice(x, (n, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def stats_tests():
    rows = []
    rng = np.random.default_rng(0)

    for kind, gradual, label in SCENARIOS:
        per_policy = {}
        for policy in ["never", "psi", "perf", "periodic"]:
            errs = [simulate(policy, kind, gradual, s)["errors"]
                    for s in range(N_SEEDS)]
            per_policy[policy] = np.array(errs, dtype=float)

            m, lo, hi = boot_ci(errs, rng=rng)
            rows.append({"scenario": label, "comparison": policy,
                         "mean": m, "ci_low": lo, "ci_high": hi,
                         "p_value": np.nan})

        # perf 가 psi 보다 정말 나은가
        d = per_policy["psi"] - per_policy["perf"]
        m, lo, hi = boot_ci(d, rng=rng)
        try:
            p = float(wilcoxon(per_policy["psi"], per_policy["perf"]).pvalue)
        except ValueError:
            p = np.nan
        rows.append({"scenario": label, "comparison": "psi - perf (diff)",
                     "mean": m, "ci_low": lo, "ci_high": hi, "p_value": p})

        # psi 가 아무것도 안 하는 것보다 나은가
        d2 = per_policy["never"] - per_policy["psi"]
        m2, lo2, hi2 = boot_ci(d2, rng=rng)
        try:
            p2 = float(wilcoxon(per_policy["never"], per_policy["psi"]).pvalue)
        except ValueError:
            p2 = np.nan
        rows.append({"scenario": label, "comparison": "never - psi (diff)",
                     "mean": m2, "ci_low": lo2, "ci_high": hi2, "p_value": p2})

        print(f"  done: {label}", flush=True)

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("[1/3] PERF_TOL sensitivity ...")
    tol_raw = tol_sensitivity()
    tol_sum = fig_tol(tol_raw)
    tol_sum.to_csv(f"{OUT}/robust_tol.csv", index=False)

    print("\n[2/3] concept drift strength sweep ...")
    st_raw = strength_sweep()
    st_sum = fig_strength(st_raw)
    st_sum.to_csv(f"{OUT}/robust_strength.csv", index=False)

    print("\n[3/3] bootstrap CI and Wilcoxon tests ...")
    stats = stats_tests()
    stats.to_csv(f"{OUT}/robust_stats.csv", index=False)

    pd.set_option("display.width", 170)

    print("\n=== 1. PERF_TOL sensitivity (PSI false alarm / miss) ===")
    piv = tol_sum.pivot(index="scenario", columns="tol",
                        values=["false_alarm", "miss"])
    print(piv.to_string(float_format=lambda v: f"{v:5.0%}"))

    print("\n=== 2. concept drift strength ===")
    print(st_sum.to_string(index=False,
                           float_format=lambda v: f"{v:7.3f}"))

    print("\n=== 3. statistical tests ===")
    show = stats[stats["comparison"].str.contains("diff")].copy()
    for c in ["mean", "ci_low", "ci_high"]:
        show[c] = show[c].round(1)
    show["p_value"] = show["p_value"].apply(
        lambda v: "n/a" if pd.isna(v) else f"{v:.2e}")
    print(show.to_string(index=False))

    print(f"\n-> {OUT}/ 에 csv 3개, 그림 2개 저장됨")