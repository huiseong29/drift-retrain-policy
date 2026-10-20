"""
재학습 정책의 동작 검증.

핵심은 각 정책이 "어떤 신호로 결정하는가"가 코드에서 지켜지는지 확인하는 것이다.
  - psi    : 입력 지표만 본다 → 레이블 지연과 무관해야 한다
  - perf   : 레이블을 본다    → 지연이 길수록 나빠져야 한다
  - hybrid : 입력 지표로 지목하고 레이블로 결정한다 → 헛경보를 기각해야 한다
"""
import numpy as np
import pytest

from policy import simulate

SEEDS = range(5)


def _mean(policy, kind, key, gradual=False, **kw):
    return np.mean([simulate(policy, kind, gradual, s, **kw)[key]
                    for s in SEEDS])


# ----------------------------------------------------------------------
# 각 정책이 약속한 신호만 쓰는가
# ----------------------------------------------------------------------
def test_never_never_retrains():
    assert _mean("never", "mixed", "retrains") == 0


def test_psi_policy_ignores_label_delay():
    """입력 지표만 보므로 레이블이 언제 오든 결과가 같아야 한다."""
    a = _mean("psi", "concept", "errors", label_delay=100)
    b = _mean("psi", "concept", "errors", label_delay=2000)
    assert abs(a - b) / a < 0.02, f"지연에 반응하면 안 된다 ({a:.0f} vs {b:.0f})"


def test_perf_policy_degrades_with_label_delay():
    """레이블 기반이므로 지연이 길수록 손해가 커져야 한다."""
    short = _mean("perf", "concept", "errors", label_delay=100)
    long_ = _mean("perf", "concept", "errors", label_delay=2000)
    assert long_ > short * 1.1, f"지연에 둔감하다 ({short:.0f} → {long_:.0f})"


# ----------------------------------------------------------------------
# 논문의 핵심 주장
# ----------------------------------------------------------------------
def test_psi_policy_misses_concept_drift():
    """컨셉 드리프트에서 PSI 정책은 거의 재학습하지 않는다 (미탐)."""
    assert _mean("psi", "concept", "retrains") < 1.0


def test_psi_policy_wastes_retrains_on_data_drift():
    """데이터 드리프트에서는 성능이 멀쩡한데도 반복 재학습한다 (오탐)."""
    retrains = _mean("psi", "data", "retrains")
    psi_err = _mean("psi", "data", "errors")
    never_err = _mean("never", "data", "errors")

    assert retrains > 3, f"경보가 안 울렸다 ({retrains})"
    # 여러 번 재학습했는데 오분류는 거의 그대로 = 낭비
    assert abs(psi_err - never_err) / never_err < 0.02


def test_psi_policy_no_better_than_doing_nothing_on_concept_drift():
    """PSI 재학습의 효과가 '아무것도 안 함'과 구분되지 않는다."""
    psi_err = _mean("psi", "concept", "errors")
    never_err = _mean("never", "concept", "errors")
    assert abs(psi_err - never_err) / never_err < 0.03


# ----------------------------------------------------------------------
# 제안 정책
# ----------------------------------------------------------------------
def test_hybrid_dismisses_false_alarms_on_data_drift():
    """입력 지표가 울려도 레이블이 멀쩡하면 기각해야 한다."""
    dismissed = _mean("hybrid", "data", "dismissed")
    hybrid_rt = _mean("hybrid", "data", "retrains")
    psi_rt = _mean("psi", "data", "retrains")

    assert dismissed > 0, "기각이 한 번도 없다"
    assert hybrid_rt < psi_rt / 2, \
        f"낭비성 재학습을 못 줄였다 (hybrid {hybrid_rt} vs psi {psi_rt})"


def test_hybrid_still_catches_concept_drift_via_safety_net():
    """PSI 가 침묵하는 경우에도 안전망이 작동해야 한다."""
    assert _mean("hybrid", "concept", "retrains") > 3
    assert _mean("hybrid", "concept", "errors") < \
        _mean("never", "concept", "errors") * 0.8


def test_hybrid_beats_baselines_on_mixed_drift():
    """혼합 드리프트에서 제안 정책이 두 베이스라인보다 나아야 한다."""
    h = _mean("hybrid", "mixed", "errors")
    assert h < _mean("perf", "mixed", "errors")
    assert h < _mean("psi", "mixed", "errors")


@pytest.mark.parametrize("kind", ["data", "concept", "mixed"])
def test_hybrid_is_never_much_worse(kind):
    """어느 시나리오에서도 유의한 열세가 없어야 한다."""
    h = _mean("hybrid", kind, "errors")
    for base in ["never", "psi", "perf"]:
        assert h < _mean(base, kind, "errors") * 1.02, \
            f"{kind} 에서 {base} 보다 많이 나쁘다"


def test_hybrid_confirmations_lead_to_retrains():
    """확인된 경보 수는 재학습 수를 넘을 수 없다."""
    for seed in SEEDS:
        r = simulate("hybrid", "mixed", False, seed)
        assert r["confirmed"] <= r["retrains"]


# ----------------------------------------------------------------------
# 재현성
# ----------------------------------------------------------------------
def test_simulation_is_deterministic():
    a = simulate("hybrid", "mixed", False, 0)
    b = simulate("hybrid", "mixed", False, 0)
    assert a == b