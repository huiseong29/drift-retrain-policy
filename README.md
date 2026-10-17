# drift-retrain-policy

Measuring how well drift detection metrics predict actual model
performance degradation, and when to retrain.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python drift.py       # 실험 실행 -> results/
python figures.py     # 포스터용 합본 그림 -> results/poster_main.png
```

## Files

- `drift.py` — 합성 데이터 생성, 드리프트 지표(PSI/KS), 실험 루프
- `figures.py` — 포스터용 2x2 그림
- `results/` — 그래프와 csv (gitignored)

## Preliminary result

네 가지 시나리오에서 지표 경보 시점과 실제 성능 저하 시점이 한 번도 일치하지 않음.

| scenario | perf drop | PSI alarm | KS alarm |
|---|---|---|---|
| Data drift (sudden) | — | 3200 | 3100 |
| Concept drift (sudden) | 3100 | — | — |
| Data drift (gradual) | — | 4000 | 3600 |
| Concept drift (gradual) | 4000 | — | — |

단일 시드 합성 데이터 기반 예비 실험. 다중 시드 반복과 실제 데이터 검증은 진행 중.