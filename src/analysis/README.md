# 📊 Analysis (통계적 가설 검증 모듈)

이 폴더는 `data/processed/`의 데이터셋을 바탕으로 통계적 가설을 검증하는 코드를 모아두는 곳입니다. `src/preprocessing/`이 "분석용 데이터셋을 만드는" 단계라면, 이곳은 "만들어진 데이터셋으로 실제 질문에 답하는" 단계입니다.

## ⚠️ 작업 규칙

* **순수 함수 + 오케스트레이션 분리:** 통계 계산(평균 차이, 신뢰구간, 검정 등)은 `*_stats.py`에 재사용 가능한 순수 함수로 작성하고, 실제 데이터 로드·로그 출력은 `run_*.py` 스크립트가 담당합니다. `src/preprocessing/data_pipeline.py`와 같은 패턴입니다.
* **재현성:** 무작위 표본추출(예: 대조군 층화추출)이 필요한 경우 고정된 시드를 사용해서, 스크립트를 다시 돌려도 같은 결과가 나오도록 합니다.
* **테스트:** `*_stats.py`의 함수들은 작은 합성 데이터로 단위 테스트를 작성합니다 (`tests/analysis/`).

## 📄 파일 구성

* `avoidance_stats.py`: 투수의 "장타 허용 구종 회피" 가설 검증에 쓰이는 통계 함수들
  * `compute_season_usage_rate()`: 투수-시즌-구종별 구사율 (경기 내 사전 비중보다 노이즈가 적은 기준선)
  * `build_stratified_placebo_candidates()`: 대조군(예: `field_out`)을 처치군(XBH)의 season×pitch_family 분포에 맞춰 층화추출
  * `summarize_paired_diff()` / `summarize_diff_in_diff()`: 짝지은 비교 / 대조군 대비 diff-in-diff 비교 (평균, 신뢰구간, t-test, Wilcoxon/Mann-Whitney)
* `run_pitch_avoidance_analysis.py`: 위 함수들로 3단계 검증(경기 내 기준선 → 시즌 기준선 → placebo diff-in-diff)을 순서대로 실행하고 요약 로그를 출력하는 진입점

## 🔄 실행 방법

`data/raw`와 `data/processed/pitch_reuse_after_xbh_events.parquet`가 이미 있어야 합니다 (없다면 `src/collection/statcast_scraper.py` → `src/preprocessing/data_pipeline.py` 순서로 먼저 실행).

```bash
python -m src.analysis.run_pitch_avoidance_analysis
```
