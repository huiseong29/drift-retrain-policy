"""
운영 환경 시뮬레이터.

파이프라인에 실제로 트래픽을 흘려 보낸다.

  - 드리프트가 들어 있는 스트림을 만들어 /predict 로 보낸다
  - 정답은 일정 지연 뒤에 /label 로 도착한다
  - 매 주기 /labeling-policy 를 조회해, 서비스가 지목한 구간이 있으면
    그 구간의 레이블만 먼저 보낸다 (레이블 예산의 재배분)

실행:
    python -m service.simulator --drift mixed --rate 250
"""
import argparse
import sys
import time

import httpx
import numpy as np

sys.path.insert(0, ".")
from drift import make_stream           # noqa: E402


def run(base_url, drift, gradual, delay, rate, seed, n_stream):
    X_ref, y_ref, X_st, y_st = make_stream(
        drift_kind=drift, gradual=gradual, seed=seed, n_stream=n_stream
    )

    with httpx.Client(base_url=base_url, timeout=20.0) as cli:
        for _ in range(60):                      # 서버 기동 대기
            try:
                if cli.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            raise SystemExit("서버에 연결할 수 없습니다")

        meta = cli.post("/bootstrap",
                        json={"X": X_ref.tolist(),
                              "y": y_ref.tolist()}).json()
        print(f"배포 완료: v{meta['version']}  "
              f"train_acc={meta['train_acc']:.3f}", flush=True)

        pending = {}          # pred_id -> [정답, 도착 예정 index, 가속여부]
        id_of = {}            # stream index -> pred_id
        interval = 1.0 / max(rate, 1)
        expedited_total = 0

        for i, (x, y) in enumerate(zip(X_st, y_st)):
            r = cli.post("/predict", json={"features": x.tolist()}).json()
            pid = r["pred_id"]
            id_of[i] = pid
            pending[pid] = [int(y), i + delay, False]

            # 도착 예정인 정답 전달
            due = [p for p, v in pending.items() if v[1] <= i]
            for p in due:
                yy, _, fast = pending.pop(p)
                cli.post("/label", json={"pred_id": p, "y": yy,
                                         "expedited": fast})

            # 서비스가 지목한 구간이 있으면 그 구간만 먼저 확보
            if i % 25 == 0:
                pol = cli.get("/labeling-policy").json().get("expedite")
                if pol:
                    lo, hi, f = pol["lo"], pol["hi"], max(pol["factor"], 1.0)
                    for p, v in pending.items():
                        if lo <= p < hi and not v[2]:
                            v[1] = min(v[1], i + int(delay / f))
                            v[2] = True
                            expedited_total += 1

            if i % 500 == 0 and i:
                st = cli.get("/status").json()
                m = st.get("metrics") or {}
                print(f"  i={i:5d}  v{st['model'].get('version')}  "
                      f"psi={m.get('psi', 0):.3f}  "
                      f"acc={(m.get('acc_lagged') or 0):.3f}  "
                      f"expedited={expedited_total}", flush=True)

            time.sleep(interval)

        # 남은 정답 모두 전달
        for p, (yy, _, fast) in list(pending.items()):
            cli.post("/label", json={"pred_id": p, "y": yy,
                                     "expedited": fast})

        st = cli.get("/status").json()
        print("\n=== 종료 ===")
        print("최종 모델 버전 :", st["model"].get("version"))
        print("이벤트 집계     :", st.get("event_counts"))
        print("우선 확보 레이블:", expedited_total)
        return st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--drift", default="mixed",
                    choices=["none", "data", "concept", "mixed"])
    ap.add_argument("--gradual", action="store_true")
    ap.add_argument("--delay", type=int, default=400)
    ap.add_argument("--rate", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    run(a.url, a.drift, a.gradual, a.delay, a.rate, a.seed, a.n)