"""
데이터 생성기와 지표의 성질 검증.

여기 있는 테스트는 구현 세부가 아니라 **논문의 전제**를 검사한다.
가장 중요한 것은 `test_concept_drift_leaves_input_distribution_intact` 로,
"컨셉 드리프트는 입력 분포를 바꾸지 않는다"는 전제가 깨지면 논문 전체의
결론이 무의미해진다.
"""
import numpy as np
import pytest
from scipy.stats import ks_2samp

from drift import make_stream, psi, psi_multi, ks_multi, run, perf_drop_point

DRIFT_START = 3000


def _split(X):
    """드리프트 전후 구간으로 나눈다."""
    return X[:DRIFT_START], X[DRIFT_START + 500:]


# ----------------------------------------------------------------------
# 논문의 전제
# ----------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_concept_drift_leaves_input_distribution_intact(seed):
    """
    컨셉 드리프트는 X→y 관계만 바꾸고 X 분포는 건드리지 않아야 한다.
    이것이 깨지면 "입력 지표가 구조적으로 못 잡는다"는 주장이 성립하지 않는다.
    """
    _, _, X_st, _ = make_stream(drift_kind="concept", seed=seed)
    pre, post = _split(X_st)

    for j in range(X_st.shape[1]):
        p = ks_2samp(pre[:, j], post[:, j]).pvalue
        assert p > 0.01, f"피처 {j} 의 분포가 바뀌었다 (KS p={p:.4f})"


@pytest.mark.parametrize("seed", range(5))
def test_concept_drift_does_degrade_accuracy(seed):
    """분포는 그대로지만 성능은 실제로 떨어져야 한다."""
    df = run("concept", seed=seed)
    pre = df[df["pos"] <= DRIFT_START]["accuracy"].mean()
    post = df[df["pos"] > DRIFT_START + 500]["accuracy"].mean()
    assert pre - post > 0.15, f"성능 저하가 너무 작다 ({pre - post:.3f})"


@pytest.mark.parametrize("seed", range(5))
def test_data_drift_does_shift_input_distribution(seed):
    """데이터 드리프트는 반대로 X 분포를 실제로 이동시켜야 한다."""
    _, _, X_st, _ = make_stream(drift_kind="data", seed=seed)
    pre, post = _split(X_st)
    assert psi_multi(pre, post) > 1.0


def test_no_drift_keeps_input_metrics_quiet():
    """드리프트가 없으면 입력 지표는 임계 아래에 머물러야 한다."""
    for seed in range(10):
        df = run("none", seed=seed)
        assert df["psi"].max() < 0.2, f"seed {seed} 에서 PSI 오탐"


def test_ground_truth_definition_has_a_noise_floor():
    """
    성능 저하 '정답' 정의 자체의 오탐률을 특성화한다.

    perf_drop_point 는 고정 허용오차(0.05)를 쓰므로, 드리프트가 없어도
    윈도우 간 자연 변동에 걸려 저하로 판정되는 경우가 있다. 이 바닥값을
    알고 있어야 시나리오별 '실제 저하율'을 올바르게 해석할 수 있다.

    데이터 드리프트 시나리오에서 관측되는 5–10% 의 저하율은 드리프트
    효과가 아니라 대부분 이 바닥값이다.
    """
    hits = [perf_drop_point(run("none", seed=s))[0] is not None
            for s in range(30)]
    rate = sum(hits) / len(hits)
    assert rate <= 0.20, f"정답 정의의 오탐률이 과도하다 ({rate:.0%})"
    assert rate > 0.0, "바닥값이 0 이면 이 테스트의 전제를 재검토해야 한다"


# ----------------------------------------------------------------------
# 강도 파라미터
# ----------------------------------------------------------------------
def test_strength_monotonically_increases_degradation():
    """강도를 올릴수록 성능 저하가 커져야 한다."""
    drops = []
    for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
        per_seed = []
        for seed in range(3):
            df = run("concept", strength=s, seed=seed)
            pre = df[df["pos"] <= DRIFT_START]["accuracy"].mean()
            post = df[df["pos"] > DRIFT_START + 500]["accuracy"].mean()
            per_seed.append(pre - post)
        drops.append(np.mean(per_seed))

    assert drops[0] < 0.02, "강도 0 에서 저하가 있으면 안 된다"
    assert all(a < b for a, b in zip(drops, drops[1:])), f"단조 증가 아님: {drops}"


def test_strength_preserves_weight_magnitude():
    """회전 방식이므로 가중치 크기가 보존되어야 한다 (신호 약화가 아님)."""
    rng = np.random.default_rng(0)
    w = rng.normal(0, 1, 5)
    rng_rot = np.random.default_rng(10_000)
    r = rng_rot.normal(0, 1, 5)
    r -= (r @ w) / (w @ w) * w
    w_perp = r / np.linalg.norm(r) * np.linalg.norm(w)

    for s in [0.0, 0.3, 0.7, 1.0]:
        th = np.pi * s
        w_new = np.cos(th) * w + np.sin(th) * w_perp
        assert np.isclose(np.linalg.norm(w_new), np.linalg.norm(w))


def test_strength_one_reproduces_full_reversal():
    """strength=1.0 은 기존 구현(w_new = -w)과 같아야 한다."""
    rng = np.random.default_rng(3)
    w = rng.normal(0, 1, 5)
    rng_rot = np.random.default_rng(3 + 10_000)
    r = rng_rot.normal(0, 1, 5)
    r -= (r @ w) / (w @ w) * w
    w_perp = r / np.linalg.norm(r) * np.linalg.norm(w)
    w_new = np.cos(np.pi) * w + np.sin(np.pi) * w_perp
    assert np.allclose(w_new, -w, atol=1e-12)


# ----------------------------------------------------------------------
# 지표
# ----------------------------------------------------------------------
def test_psi_is_zero_for_identical_samples():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert psi(x, x) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    vals = [psi(ref, rng.normal(m, 1, 5000)) for m in [0.0, 0.5, 1.0, 2.0]]
    assert all(a < b for a, b in zip(vals, vals[1:])), vals


def test_ks_is_bounded():
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (500, 3))
    B = rng.normal(5, 1, (500, 3))
    assert 0.0 <= ks_multi(A, A) <= 1.0
    assert 0.9 < ks_multi(A, B) <= 1.0


# ----------------------------------------------------------------------
# 재현성
# ----------------------------------------------------------------------
def test_same_seed_gives_same_stream():
    a = make_stream(drift_kind="mixed", seed=7)
    b = make_stream(drift_kind="mixed", seed=7)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_different_seeds_differ():
    _, _, A, _ = make_stream(drift_kind="mixed", seed=1)
    _, _, B, _ = make_stream(drift_kind="mixed", seed=2)
    assert not np.array_equal(A, B)


def test_mixed_drift_has_both_effects():
    """혼합은 분포 이동과 성능 저하를 동시에 보여야 한다."""
    df = run("mixed", seed=1)
    pre = df[df["pos"] <= DRIFT_START]["accuracy"].mean()
    post = df[df["pos"] > DRIFT_START + 500]["accuracy"].mean()
    assert df["psi"].max() > 1.0, "입력 분포가 안 움직였다"
    assert pre - post > 0.15, "성능이 안 떨어졌다"