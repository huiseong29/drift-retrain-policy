"""
통계 보강

앞선 실험들은 평균과 단일 검정만 보고했다. 여기서 네 가지를 보강한다.

  1. 다중비교 보정 (Holm–Bonferroni)
     정책 비교에서 여러 번 검정했으므로, 보정 없이는 우연히 유의해질 수 있다.

  2. 효과크기
     p 값은 "차이가 있다"만 말한다. 얼마나 큰 차이인지는 Cohen's dz 와
     rank-biserial 상관으로 함께 보고한다.

  3. 동등성 검정 (TOST)
     "PSI 재학습의 효과가 0과 구분되지 않는다"는 주장은 p > 0.05 만으로는
     성립하지 않는다. 검정력이 낮아서 못 잡았을 수도 있기 때문이다.
     실질적 동등 범위를 정해 두고 양측 단측검정으로 '차이가 없음'을 적극
     입증한다.

  4. 검정력 분석
     시드 20개로 탐지할 수 있는 최소 효과크기(MDE)를 계산해, 유의하지 않게
     나온 결과가 표본 부족 때문이 아님을 보인다.

  5. 발화율 비교에 Fisher 정확검정
     탐지기 계열 간 발화율 차이(0/20 vs 20/20)에 검정을 붙인다.

실행:
    python stats.py
출력:
    results/stats_tests.csv      - 보정된 p 값, 효과크기, 신뢰구간
    results/stats_tost.csv       - 동등성 검정
    results/stats_power.csv      - 검정력 분석
    results/stats_firerate.csv   - 발화율 Fisher 검정
"""
import os
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

from policy import simulate

OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 20
ALPHA = 0.05

SCENARIOS = [
    ("data",    False, "Data / sudden"),
    ("concept", False, "Concept / sudden"),
    ("mixed",   False, "Mixed / sudden"),
    ("mixed",   True,  "Mixed / gradual"),
]
POLICIES = ["never", "psi", "perf", "hybrid"]

# 실질적 동등 범위: '아무것도 안 함' 대비 이 비율 이내의 차이는
# 운영상 의미 없는 것으로 본다.
EQUIV_MARGIN_RATIO = 0.05


# ----------------------------------------------------------------------
def collect():
    """정책 × 시나리오 × 시드의 누적 오분류."""
    data = {}
    for kind, gradual, label in SCENARIOS:
        for policy in POLICIES:
            data[(label, policy)] = np.array(
                [simulate(policy, kind, gradual, s)["errors"]
                 for s in range(N_SEEDS)], dtype=float)
        print(f"  collected: {label}", flush=True)
    return data


# ----------------------------------------------------------------------
# 효과크기
# ----------------------------------------------------------------------
def cohens_dz(d):
    """대응표본 Cohen's dz. 차이의 평균을 차이의 표준편차로 나눈 값."""
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def rank_biserial(d):
    """
    Wilcoxon 에 대응하는 효과크기. -1 ~ +1.
    양수면 d > 0 인 쌍이 우세하다는 뜻.
    """
    nz = d[d != 0]
    if len(nz) == 0:
        return 0.0
    ranks = sps.rankdata(np.abs(nz))
    rpos = ranks[nz > 0].sum()
    rneg = ranks[nz < 0].sum()
    return float((rpos - rneg) / ranks.sum())


def interpret_dz(dz):
    a = abs(dz)
    return ("negligible" if a < 0.2 else
            "small" if a < 0.5 else
            "medium" if a < 0.8 else "large")


# ----------------------------------------------------------------------
# 다중비교 보정
# ----------------------------------------------------------------------
def holm(pvals):
    """Holm–Bonferroni. Bonferroni 보다 검정력을 덜 잃는다."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for i, idx in enumerate(order):
        val = (n - i) * p[idx]
        running = max(running, val)          # 단조성 보장
        adj[idx] = min(running, 1.0)
    return adj


# ----------------------------------------------------------------------
# 1. 쌍별 비교
# ----------------------------------------------------------------------
def pairwise(data):
    rows = []
    rng = np.random.default_rng(0)

    for _, _, label in SCENARIOS:
        for a, b in itertools.combinations(POLICIES, 2):
            x, y = data[(label, a)], data[(label, b)]
            d = x - y                        # 양수면 a 가 더 많이 틀림

            if np.allclose(d, 0):
                p = 1.0
            else:
                p = float(sps.wilcoxon(x, y).pvalue)

            # 부트스트랩 신뢰구간
            means = rng.choice(d, (5000, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])

            rows.append({
                "scenario": label,
                "comparison": f"{a} - {b}",
                "mean_diff": float(d.mean()),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "cohens_dz": cohens_dz(d),
                "rank_biserial": rank_biserial(d),
                "p_raw": p,
            })

    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_raw"].values)
    df["significant"] = df["p_holm"] < ALPHA
    df["effect"] = df["cohens_dz"].apply(interpret_dz)
    return df


# ----------------------------------------------------------------------
# 2. 동등성 검정 (TOST)
# ----------------------------------------------------------------------
def tost(x, y, margin):
    """
    두 단측검정. 차이가 (-margin, +margin) 안에 있음을 적극 입증한다.
    두 p 값 중 큰 쪽이 ALPHA 미만이면 '실질적으로 동등'하다고 결론한다.
    """
    d = x - y
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return 0.0, 0.0, True

    t_lo = (d.mean() + margin) / se          # H0: diff <= -margin
    t_hi = (d.mean() - margin) / se          # H0: diff >= +margin
    p_lo = float(sps.t.sf(t_lo, df=n - 1))
    p_hi = float(sps.t.cdf(t_hi, df=n - 1))
    p = max(p_lo, p_hi)
    return p_lo, p_hi, p < ALPHA


def equivalence(data):
    """'PSI 재학습은 아무것도 안 하는 것과 같다'를 적극 입증한다."""
    rows = []
    for _, _, label in SCENARIOS:
        never = data[(label, "never")]
        margin = EQUIV_MARGIN_RATIO * never.mean()

        for policy in ["psi", "perf", "hybrid"]:
            x = data[(label, policy)]
            d = never - x
            p_lo, p_hi, equiv = tost(never, x, margin)
            rows.append({
                "scenario": label,
                "comparison": f"never vs {policy}",
                "mean_diff": float(d.mean()),
                "margin": float(margin),
                "p_lower": p_lo,
                "p_upper": p_hi,
                "equivalent": equiv,
                "note": ("실질적 동등 — 재학습 효과 없음" if equiv
                         else "동등하지 않음"),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 3. 검정력 분석
# ----------------------------------------------------------------------
def min_detectable_effect(n, alpha=ALPHA, power=0.80):
    """대응표본 t 검정 기준, 주어진 n 에서 탐지 가능한 최소 Cohen's dz."""
    from scipy.optimize import brentq

    def achieved(dz):
        ncp = dz * np.sqrt(n)
        crit = sps.t.ppf(1 - alpha / 2, df=n - 1)
        return (sps.nct.sf(crit, df=n - 1, nc=ncp)
                + sps.nct.cdf(-crit, df=n - 1, nc=ncp)) - power

    return float(brentq(achieved, 1e-4, 5.0))


def achieved_power(dz, n, alpha=ALPHA):
    ncp = abs(dz) * np.sqrt(n)
    crit = sps.t.ppf(1 - alpha / 2, df=n - 1)
    # 비중심 t 는 ncp 가 크면 수치적으로 불안정하다. 그 구간의 검정력은
    # 사실상 1.0 이므로 명시적으로 처리한다.
    if ncp > 15:
        return 1.0
    p = (sps.nct.sf(crit, df=n - 1, nc=ncp)
         + sps.nct.cdf(-crit, df=n - 1, nc=ncp))
    return 1.0 if not np.isfinite(p) else float(np.clip(p, 0.0, 1.0))


def power_analysis(data):
    mde = min_detectable_effect(N_SEEDS)
    rows = []
    for _, _, label in SCENARIOS:
        never = data[(label, "never")]
        for policy in ["psi", "perf", "hybrid"]:
            d = never - data[(label, policy)]
            dz = cohens_dz(d)
            sd = d.std(ddof=1)
            rows.append({
                "scenario": label,
                "comparison": f"never - {policy}",
                "observed_dz": dz,
                "mde_dz_at_n20": mde,
                "achieved_power": achieved_power(dz, N_SEEDS),
                # 이 설계로 탐지 가능한 최소 오분류 차이 (건수)
                "mde_errors": float(mde * sd),
                "mde_pct_of_never": float(mde * sd / never.mean() * 100),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. 발화율 비교 (Fisher 정확검정)
# ----------------------------------------------------------------------
def fire_rate_tests():
    path = f"{OUT}/detector_summary.csv"
    if not os.path.exists(path):
        print("  detector_summary.csv 없음 — detectors.py 를 먼저 실행")
        return pd.DataFrame()

    d = pd.read_csv(path)
    n = 10          # detectors.py 의 N_SEEDS
    rows = []

    for scen in sorted(d[d.scenario.str.startswith("Concept")].scenario.unique()):
        s = d[d.scenario == scen].set_index("detector")["fire_rate"]
        for input_based in ["PSI", "KS", "Wasserstein", "MMD", "Discriminator"]:
            for error_based in ["ADWIN", "PageHinkley", "DDM"]:
                a = int(round(s[input_based] * n))
                b = int(round(s[error_based] * n))
                table = [[a, n - a], [b, n - b]]
                p = float(sps.fisher_exact(table).pvalue)
                rows.append({
                    "scenario": scen,
                    "input_based": input_based,
                    "error_based": error_based,
                    "fire_input": a, "fire_error": b, "n": n,
                    "p_raw": p,
                })

    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(df["p_raw"].values)
        df["significant"] = df["p_holm"] < ALPHA
    return df


# ----------------------------------------------------------------------
# 5. 동등성 검정 그림 (forest plot)
# ----------------------------------------------------------------------
def fig_equivalence(data, fname="fig_equivalence.png"):
    """
    평균 차이와 95% 신뢰구간을 동등범위와 함께 그린다.

    신뢰구간이 회색 띠(동등범위) 안에 완전히 들어가면 '실질적으로 동등'이고,
    띠 밖으로 벗어나면 실제 차이가 있다는 뜻이다. p 값 하나보다 이 그림이
    "차이가 없음"을 훨씬 분명하게 보여준다.
    """
    rng = np.random.default_rng(0)
    rows = []
    for _, _, label in SCENARIOS:
        never = data[(label, "never")]
        margin = EQUIV_MARGIN_RATIO * never.mean()
        for policy in ["psi", "perf", "hybrid"]:
            d = never - data[(label, policy)]
            boot = rng.choice(d, (5000, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append({
                "scenario": label, "policy": policy,
                "mean": d.mean(), "lo": lo, "hi": hi,
                "margin": margin,
                # 신뢰구간이 동등범위 안에 완전히 들어가는가
                "equivalent": (lo > -margin) and (hi < margin),
            })
    df = pd.DataFrame(rows)

    POL_COLOR = {"psi": "#d62728", "perf": "#1f77b4", "hybrid": "#17becf"}
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(16, 4.4))

    for ax, (_, _, label) in zip(axes, SCENARIOS):
        g = df[df.scenario == label].reset_index(drop=True)
        margin = g["margin"].iloc[0]

        # 동등범위 띠
        ax.axvspan(-margin, margin, color="#999", alpha=0.16, lw=0)
        ax.axvline(0, color="#333", lw=1.2)

        for i, r in g.iterrows():
            y = len(g) - 1 - i
            ax.plot([r.lo, r.hi], [y, y], lw=3,
                    color=POL_COLOR[r.policy], solid_capstyle="round")
            ax.plot([r["mean"]], [y], marker="D", ms=9,
                    color=POL_COLOR[r.policy], markeredgecolor="white",
                    markeredgewidth=1.4, zorder=3)
            ax.text(r.hi, y + 0.22,
                    "equivalent" if r.equivalent else "different",
                    fontsize=8.5, ha="left" if r.hi >= 0 else "right",
                    color="#555")

        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([r.policy for r in g.itertuples()][::-1],
                           fontsize=10)
        ax.set_ylim(-0.6, len(g) - 0.3)
        ax.set_title(f"{label}\nequivalence margin  ±{margin:.0f}", fontsize=10.5)
        ax.set_xlabel("errors avoided  (never − policy)", fontsize=9.5)
        ax.grid(axis="x", alpha=0.25)
        ax.set_axisbelow(True)

        span = max(abs(g.lo.min()), abs(g.hi.max()), margin) * 1.45
        ax.set_xlim(-span * 0.35, span)

    fig.suptitle("A confidence interval inside the grey band means the policy "
                 "is practically equivalent to doing nothing",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(f"{OUT}/{fname}", dpi=200)
    plt.close()
    return df


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"[1/4] simulating ({N_SEEDS} seeds) ...")
    data = collect()

    print("\n[2/4] pairwise comparisons with Holm correction ...")
    pw = pairwise(data)
    pw.to_csv(f"{OUT}/stats_tests.csv", index=False)

    print("[3/4] equivalence (TOST) and power analysis ...")
    eq = equivalence(data)
    eq.to_csv(f"{OUT}/stats_tost.csv", index=False)
    fig_equivalence(data)
    pw2 = power_analysis(data)
    pw2.to_csv(f"{OUT}/stats_power.csv", index=False)

    print("[4/4] fire-rate Fisher exact tests ...")
    fr = fire_rate_tests()
    if len(fr):
        fr.to_csv(f"{OUT}/stats_firerate.csv", index=False)

    pd.set_option("display.width", 200)

    print("\n=== 주요 비교 (Holm 보정) ===")
    key = pw[pw["comparison"].isin(["never - psi", "psi - hybrid",
                                    "perf - hybrid"])].copy()
    for c in ["mean_diff", "ci_low", "ci_high", "cohens_dz"]:
        key[c] = key[c].round(2)
    key["p_raw"] = key["p_raw"].apply(lambda v: f"{v:.2e}")
    key["p_holm"] = key["p_holm"].apply(lambda v: f"{v:.2e}")
    print(key[["scenario", "comparison", "mean_diff", "ci_low", "ci_high",
               "cohens_dz", "effect", "p_holm", "significant"]]
          .to_string(index=False))

    print("\n=== 동등성 검정 (never vs psi) ===")
    e = eq[eq["comparison"] == "never vs psi"].copy()
    for c in ["mean_diff", "margin"]:
        e[c] = e[c].round(1)
    e["p_upper"] = e["p_upper"].apply(lambda v: f"{v:.2e}")
    e["p_lower"] = e["p_lower"].apply(lambda v: f"{v:.2e}")
    print(e[["scenario", "mean_diff", "margin", "p_lower", "p_upper",
             "equivalent", "note"]].to_string(index=False))

    print("\n=== 검정력 ===")
    p = pw2[pw2["comparison"] == "never - psi"].copy()
    for c in ["observed_dz", "mde_dz_at_n20", "achieved_power",
              "mde_errors", "mde_pct_of_never"]:
        p[c] = p[c].round(3)
    print(p.to_string(index=False))
    print(f"\n  시드 {N_SEEDS}개로 검정력 80%를 확보하는 최소 효과크기: "
          f"dz = {min_detectable_effect(N_SEEDS):.3f}")

    if len(fr):
        sig = fr["significant"].mean()
        print(f"\n=== 발화율 Fisher 검정 ===")
        print(f"  입력기반 × 오차기반 쌍 {len(fr)}개 중 "
              f"Holm 보정 후 유의: {fr['significant'].sum()}개 ({sig:.0%})")

    print(f"\n-> {OUT}/ 에 csv 4개 저장됨")