"""
재학습 정책 비교

지표가 성능 저하를 못 맞춘다는 것이 실제 운영에서 어떤 손해로 이어지는지 본다.
스트림을 따라가며 각 정책이 언제 재학습할지 스스로 결정하게 하고,
누적 오분류 수와 재학습 횟수를 함께 측정한다.

정책:
  never      - 재학습하지 않음 (하한선)
  periodic   - N 샘플마다 무조건 재학습
  psi        - PSI가 임계를 넘으면 재학습  (현업에서 흔한 방식)
  perf       - 관측된 정확도가 떨어지면 재학습 (레이블 지연 반영)
  onset      - 드리프트 발생 시점을 안다고 가정하고, 이후 구간만으로 학습할 수
               있게 되는 가장 이른 시점에 단 한 번 재학습. 현실에서는 불가능한
               정보를 쓰지만 재학습은 1회뿐이므로 전역 최적은 아니다.
               "드리프트 시점을 완벽히 알아도 한 번만 고쳐서는 부족하다"는
               비교 기준으로 쓴다.

비용 모델:
  total_cost(c) = 누적 오분류 수 + c x 재학습 횟수
  c 는 "재학습 1회가 오분류 몇 건만큼의 비용인가". 이 값을 넓게 스윕한다.

실행:
    python policy.py
출력:
    results/policy_raw.csv      - 시드별 원자료
    results/policy_summary.csv  - 시나리오 x 정책 집계
    results/fig_policy.png      - 재학습 횟수 vs 누적 오분류
    results/fig_cost.png        - 비용 비율별 최적 정책
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from drift import make_stream, psi_multi

OUT = "results"
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 10
CHUNK = 200           # 한 번에 처리하는 샘플 수
LABEL_DELAY = 400     # 정답이 도착하기까지의 지연 (샘플 수)
TRAIN_WINDOW = 1000   # 재학습에 쓰는 최근 구간 크기
PSI_TH = 0.2
PERF_TOL = 0.05       # 초기 정확도 대비 이만큼 떨어지면 하락으로 판정
PERIOD = 1000         # periodic 정책의 주기
DRIFT_START = 3000

POLICIES = ["never", "periodic", "psi", "perf", "onset"]

SCENARIOS = [
    ("data",    False, "Data / sudden"),
    ("concept", False, "Concept / sudden"),
    ("data",    True,  "Data / gradual"),
    ("concept", True,  "Concept / gradual"),
]

COLORS = {
    "never":    "#9467bd",
    "periodic": "#2ca02c",
    "psi":      "#d62728",
    "perf":     "#1f77b4",
    "onset":   "#7f7f7f",
}


# ----------------------------------------------------------------------
def simulate(policy, kind, gradual, seed, label_delay=LABEL_DELAY,
             expedite=4):
    """스트림을 따라가며 정책이 재학습 시점을 스스로 결정하게 한다."""
    X_ref, y_ref, X_st, y_st = make_stream(
        drift_kind=kind, gradual=gradual, seed=seed
    )

    model = LogisticRegression(max_iter=1000).fit(X_ref, y_ref)
    ref_X = X_ref                 # PSI 비교 기준 (재학습하면 갱신)

    errors = 0
    retrains = 0
    chunk_acc = []                # 청크별 정확도 (perf 정책이 지연 후 참조)
    last_periodic = 0

    pending = None                # hybrid: 확인 대기 중인 트리거 청크
    dismissed = 0                 # hybrid: 레이블 확인 결과 기각된 경보 수
    confirmed = 0                 # hybrid: 확인되어 재학습으로 이어진 경보 수

    n_chunks = len(X_st) // CHUNK

    for i in range(n_chunks):
        s, e = i * CHUNK, (i + 1) * CHUNK
        Xw, yw = X_st[s:e], y_st[s:e]

        pred = model.predict(Xw)
        errors += int((pred != yw).sum())
        chunk_acc.append(float((pred == yw).mean()))

        # ---- 재학습 여부 결정 ----
        retrain = False
        fast_labels = False        # hybrid: 우선 확보한 레이블을 쓸 수 있는가

        if policy == "never":
            pass

        elif policy == "periodic":
            if e - last_periodic >= PERIOD:
                retrain = True
                last_periodic = e

        elif policy == "psi":
            if psi_multi(ref_X, Xw) >= PSI_TH:
                retrain = True

        elif policy == "perf":
            # 레이블 지연 때문에 지금 청크의 정확도는 아직 볼 수 없다.
            lag_chunks = label_delay // CHUNK
            idx = i - lag_chunks
            if idx >= 5:                                  # 기준선 확보 후
                base = float(np.mean(chunk_acc[:5]))
                if chunk_acc[idx] < base - PERF_TOL:
                    retrain = True

        elif policy == "hybrid":
            # 제안 정책: 입력 지표를 "재학습 트리거"가 아니라 "확인 요청"으로 쓴다.
            #
            #  1) PSI 가 울리면 즉시 재학습하지 않고 그 청크를 확인 대기에 올린다.
            #  2) 그 청크에 한해 레이블을 우선 확보한다. 전수 레이블링을
            #     앞당기는 것이 아니라 지목된 구간만 빨리 받는 것이므로,
            #     지연이 1/expedite 로 줄어든다. 이것이 비용이다.
            #  3) 확보된 레이블로 실제 정확도를 확인한다.
            #       - 떨어졌으면 재학습     (확인된 경보)
            #       - 멀쩡하면 기각하고 기준 구간만 갱신 (기각된 경보)
            #  4) PSI 가 끝내 울리지 않는 컨셉 드리프트를 위해, 통상 지연의
            #     성능 감시를 안전망으로 함께 돌린다.
            #
            # 입력 지표는 "어디에 레이블을 먼저 쓸지"를 정하는 신호로만 쓰이며,
            # 재학습 결정은 항상 레이블이 내린다.
            lag_chunks = label_delay // CHUNK
            lag_fast = max(1, int(round(label_delay / expedite / CHUNK)))
            base = float(np.mean(chunk_acc[:5])) if i >= 5 else None

            # (1)(2) 경보 접수 — 해당 구간 레이블을 우선 확보
            if pending is None and psi_multi(ref_X, Xw) >= PSI_TH:
                pending = i

            # (3) 우선 확보된 레이블로 확인
            if pending is not None and base is not None \
                    and i - pending >= lag_fast:
                if chunk_acc[pending] < base - PERF_TOL:
                    retrain = True
                    confirmed += 1
                    # 우선 확보한 레이블은 재학습에도 쓸 수 있다
                    fast_labels = True
                else:
                    dismissed += 1
                    ref_X = Xw          # 같은 변화로 계속 울리지 않게 기준 갱신
                pending = None

            # (4) 안전망: 입력 지표가 침묵하는 경우를 위한 통상 지연 감시
            if not retrain and base is not None:
                idx = i - lag_chunks
                if idx >= 5 and chunk_acc[idx] < base - PERF_TOL:
                    retrain = True

        elif policy == "onset":
            # 드리프트 이후 구간만으로 학습할 수 있게 되는 가장 이른 시점.
            # 레이블 지연 때문에 드리프트 발생 즉시 재학습해도 손에 들어온
            # 데이터는 아직 드리프트 이전 것이라 효과가 없다.
            if retrains == 0 and (e - label_delay - TRAIN_WINDOW) >= DRIFT_START:
                retrain = True

        # ---- 재학습 실행 ----
        if retrain:
            # 정답이 도착한 지점까지만 사용.
            # 확인된 경보로 재학습하는 경우에는 우선 확보한 구간까지 쓸 수 있다.
            eff_delay = (label_delay / expedite) if fast_labels else label_delay
            avail_end = int(e - eff_delay)
            if avail_end >= TRAIN_WINDOW:
                lo = avail_end - TRAIN_WINDOW
                Xt, yt = X_st[lo:avail_end], y_st[lo:avail_end]
            elif avail_end > 200:
                Xt, yt = X_st[:avail_end], y_st[:avail_end]
            else:
                Xt, yt = X_ref, y_ref          # 아직 쓸 데이터가 없음

            if len(np.unique(yt)) < 2:         # 한쪽 클래스만 있으면 학습 불가
                retrain = False
            else:
                model = LogisticRegression(max_iter=1000).fit(Xt, yt)
                ref_X = Xt
                retrains += 1

    total = n_chunks * CHUNK
    return {
        "errors": errors,
        "retrains": retrains,
        "error_rate": errors / total,
        "mean_acc": float(np.mean(chunk_acc)),
        "confirmed": confirmed,      # hybrid 전용
        "dismissed": dismissed,      # hybrid 전용
    }


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
                 retrains=("retrains", "mean"),
                 error_rate=("error_rate", "mean"),
                 mean_acc=("mean_acc", "mean"))
            .reset_index())
    # 정책 순서 고정
    g["policy"] = pd.Categorical(g["policy"], POLICIES, ordered=True)
    return g.sort_values(["scenario", "policy"]).reset_index(drop=True)


# ----------------------------------------------------------------------
def fig_policy(summary):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), sharey=True)

    for ax, (_, _, label) in zip(axes, SCENARIOS):
        g = summary[summary["scenario"] == label]
        for _, row in g.iterrows():
            ax.scatter(row["retrains"], row["errors"],
                       s=190, color=COLORS[row["policy"]],
                       edgecolor="white", linewidth=1.8, zorder=3,
                       label=row["policy"])

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Retrains", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.margins(x=0.22, y=0.18)

    axes[0].set_ylabel("Cumulative misclassifications", fontsize=10)

    handles = [plt.Line2D([], [], marker="o", ls="", markersize=11,
                          markerfacecolor=COLORS[p], markeredgecolor="white",
                          label=p)
               for p in POLICIES]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.93))

    fig.suptitle("Lower-left is better: fewer retrains, fewer errors",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    plt.savefig(f"{OUT}/fig_policy.png", dpi=200)
    plt.close()


def fig_cost(summary):
    """재학습 1회 비용(c)을 바꿔가며 각 정책의 총비용을 비교."""
    cs = np.arange(0, 2001, 25)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for ax, label in zip(axes, ["Data / sudden", "Concept / sudden"]):
        g = summary[summary["scenario"] == label]
        for _, row in g.iterrows():
            cost = row["errors"] + cs * row["retrains"]
            ax.plot(cs, cost, lw=2.2, color=COLORS[row["policy"]],
                    label=row["policy"])

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Cost of one retrain (in misclassification units)",
                      fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Total cost", fontsize=10)
    axes[0].legend(fontsize=9.5, loc="upper left")

    fig.suptitle("PSI-triggered retraining: pure cost under data drift, pure loss under concept drift",
                 fontsize=13, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f"{OUT}/fig_cost.png", dpi=200)
    plt.close()


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print(f"running {len(POLICIES)} policies x {len(SCENARIOS)} scenarios "
          f"x {N_SEEDS} seeds ...")
    raw = collect()
    raw.to_csv(f"{OUT}/policy_raw.csv", index=False)

    summary = summarize(raw)
    summary.to_csv(f"{OUT}/policy_summary.csv", index=False)

    fig_policy(summary)
    fig_cost(summary)

    pd.set_option("display.width", 160)
    print("\n=== policy summary (mean over seeds) ===")
    show = summary.copy()
    show["errors"] = show["errors"].round(0).astype(int)
    show["retrains"] = show["retrains"].round(1)
    show["mean_acc"] = show["mean_acc"].round(3)
    print(show[["scenario", "policy", "retrains", "errors", "mean_acc"]]
          .to_string(index=False))
    print(f"\n-> {OUT}/ 에 csv 2개, 그림 2개 저장됨")