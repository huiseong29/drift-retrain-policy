"""
다중 시드 반복 + 임계값 스윕

목적 1 (반복):  단일 시드 결과가 우연이 아님을 보인다.
목적 2 (스윕):  임계값을 어떻게 잡아도 오탐/미탐을 동시에 없앨 수 없음을 보인다.

실행:
    python sweep.py
출력:
    results/seeds_raw.csv        - 시드별 원자료
    results/seeds_summary.csv    - 시나리오별 집계
    results/threshold_sweep.csv  - 임계값별 오탐/미탐
    results/fig_seeds.png        - 반복 결과 그림
    results/fig_threshold.png    - 임계값 스윕 그림
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drift import run, perf_drop_point, first_cross

OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 20
PSI_TH, KS_TH = 0.2, 0.1

SCENARIOS = [
    ("data",    False, "Data / sudden"),
    ("concept", False, "Concept / sudden"),
    ("data",    True,  "Data / gradual"),
    ("concept", True,  "Concept / gradual"),
]

BLUE, RED, ORANGE = "#1f77b4", "#d62728", "#ff7f0e"


# ----------------------------------------------------------------------
# 1. 시드 반복
# ----------------------------------------------------------------------
def collect():
    rows = []
    for kind, gradual, label in SCENARIOS:
        for seed in range(N_SEEDS):
            df = run(kind, gradual=gradual, seed=seed)
            drop, base = perf_drop_point(df)
            rows.append({
                "scenario": label,
                "kind": kind,
                "seed": seed,
                "perf_drop": drop,
                "psi_alarm": first_cross(df, "psi", PSI_TH),
                "ks_alarm": first_cross(df, "ks", KS_TH),
                "psi_max": df["psi"].max(),
                "ks_max": df["ks"].max(),
                "acc_start": base,
                "acc_end": df["accuracy"].iloc[-5:].mean(),
            })
        print(f"  done: {label}", flush=True)
    return pd.DataFrame(rows)


def summarize(raw):
    out = []
    for label in [s[2] for s in SCENARIOS]:
        g = raw[raw["scenario"] == label]
        n = len(g)

        perf_dropped = g["perf_drop"].notna()
        psi_fired = g["psi_alarm"].notna()
        ks_fired = g["ks_alarm"].notna()

        out.append({
            "scenario": label,
            "n": n,
            "perf_drop_rate": perf_dropped.mean(),
            "psi_fire_rate": psi_fired.mean(),
            "ks_fire_rate": ks_fired.mean(),
            # 경보는 울렸으나 성능은 멀쩡
            "psi_false_alarm": (psi_fired & ~perf_dropped).mean(),
            # 성능은 떨어졌으나 경보 없음
            "psi_miss": (~psi_fired & perf_dropped).mean(),
            "ks_false_alarm": (ks_fired & ~perf_dropped).mean(),
            "ks_miss": (~ks_fired & perf_dropped).mean(),
            # 둘 다 발생한 경우에만 선행시간 계산
            "psi_lead_mean": (g.loc[psi_fired & perf_dropped, "perf_drop"]
                              - g.loc[psi_fired & perf_dropped, "psi_alarm"]).mean(),
            "acc_drop_mean": (g["acc_start"] - g["acc_end"]).mean(),
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------
# 2. 임계값 스윕
# ----------------------------------------------------------------------
def threshold_sweep():
    """PSI 임계값을 넓게 바꿔가며 오탐/미탐이 동시에 0이 되는 지점이 있는지 본다."""
    cached = {}
    for kind, gradual, label in SCENARIOS:
        for seed in range(N_SEEDS):
            cached[(label, seed)] = run(kind, gradual=gradual, seed=seed)
        print(f"  cached: {label}", flush=True)

    # PSI는 데이터 드리프트에서 3.0까지 올라가므로, 경보가 완전히
    # 멎는 지점까지 범위를 넓혀 트레이드오프 곡선 전체를 본다.
    thresholds = np.concatenate([
        np.arange(0.01, 0.20, 0.01),
        np.arange(0.20, 1.00, 0.05),
        np.arange(1.00, 4.01, 0.10),
    ])

    rows = []
    for th in thresholds:
        fa = miss = total = 0
        for (label, seed), df in cached.items():
            drop, _ = perf_drop_point(df)
            fired = first_cross(df, "psi", th) is not None
            dropped = drop is not None
            fa += int(fired and not dropped)
            miss += int(dropped and not fired)
            total += 1
        rows.append({
            "threshold": round(float(th), 3),
            "false_alarm_rate": fa / total,
            "miss_rate": miss / total,
            "total_error_rate": (fa + miss) / total,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. 그림
# ----------------------------------------------------------------------
def fig_seeds(summary):
    labels = summary["scenario"].tolist()
    x = np.arange(len(labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w, summary["perf_drop_rate"], w, label="Performance degraded",
           color=BLUE)
    ax.bar(x, summary["psi_fire_rate"], w, label="PSI alarm fired", color=RED)
    ax.bar(x + w, summary["ks_fire_rate"], w, label="KS alarm fired",
           color=ORANGE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Fraction of runs", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Alarms and actual degradation never coincide  "
                 f"({N_SEEDS} seeds per scenario)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10, ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_seeds.png", dpi=200)
    plt.close()


def fig_threshold(sweep):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep["threshold"], sweep["false_alarm_rate"], color=RED, lw=2.2,
            label="False alarm rate (metric fires, performance fine)")
    ax.plot(sweep["threshold"], sweep["miss_rate"], color=BLUE, lw=2.2,
            label="Miss rate (performance drops, metric silent)")
    ax.plot(sweep["threshold"], sweep["total_error_rate"], color="#444",
            lw=1.6, ls="--", label="Total error rate")

    best = sweep.loc[sweep["total_error_rate"].idxmin()]
    ax.axvline(best["threshold"], color="gray", ls=":", alpha=0.8)
    ax.annotate(
        f"best possible threshold = {best['threshold']}\n"
        f"still wrong on {best['total_error_rate']:.0%} of runs",
        xy=(best["threshold"], best["total_error_rate"]),
        xytext=(best["threshold"] + 0.55, best["total_error_rate"] + 0.20),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray"),
    )

    ax.set_xlabel("PSI alarm threshold", fontsize=11)
    ax.set_ylabel("Error rate across all runs", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("No threshold removes both failure modes",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig_threshold.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[1/2] running {N_SEEDS} seeds x {len(SCENARIOS)} scenarios ...")
    raw = collect()
    raw.to_csv(f"{OUT}/seeds_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(f"{OUT}/seeds_summary.csv", index=False)
    fig_seeds(summary)

    print(f"\n[2/2] threshold sweep ...")
    sweep = threshold_sweep()
    sweep.to_csv(f"{OUT}/threshold_sweep.csv", index=False)
    fig_threshold(sweep)

    pd.set_option("display.width", 160)
    print("\n=== scenario summary ===")
    print(summary[["scenario", "n", "perf_drop_rate", "psi_fire_rate",
                   "ks_fire_rate", "psi_false_alarm", "psi_miss",
                   "acc_drop_mean"]].to_string(index=False))

    best = sweep.loc[sweep["total_error_rate"].idxmin()]
    print("\n=== threshold sweep ===")
    print(f"best PSI threshold : {best['threshold']}")
    print(f"  false alarm rate : {best['false_alarm_rate']:.2%}")
    print(f"  miss rate        : {best['miss_rate']:.2%}")
    print(f"  total error rate : {best['total_error_rate']:.2%}")
    print(f"\n-> {OUT}/ 에 csv 4개, 그림 2개 저장됨")