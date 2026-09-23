# 📊 Analysis (통계적 가설 검증 모듈)

이 폴더는 `data/processed/`의 데이터셋을 바탕으로 통계적 가설을 검증하는 코드를 모아두는 곳입니다. `src/preprocessing/`이 "분석용 데이터셋을 만드는" 단계라면, 이곳은 "만들어진 데이터셋으로 실제 질문에 답하는" 단계입니다.

## ⚠️ 작업 규칙

* **순수 함수 + 오케스트레이션 분리:** 통계 계산(평균 차이, 신뢰구간, 검정 등)은 `*_stats.py`에 재사용 가능한 순수 함수로 작성하고, 실제 데이터 로드·로그 출력은 `run_*.py` 스크립트가 담당합니다. `src/preprocessing/data_pipeline.py`와 같은 패턴입니다.
* **재현성:** 무작위 표본추출(예: 대조군 층화추출)이 필요한 경우 고정된 시드를 사용해서, 스크립트를 다시 돌려도 같은 결과가 나오도록 합니다.
* **테스트:** `*_stats.py`의 함수들은 작은 합성 데이터로 단위 테스트를 작성합니다 (`tests/analysis/`). R 연동 코드(`glmer_runner.py` 등)는 R이 없는 팀원도 있어 단위 테스트 대신 실제 데이터로 스크립트를 돌려 검증합니다.

## 🎯 메인 / 보조 분석

* **메인 — 같은 타자 재대결:** 장타를 맞은 타자를 같은 투수가 같은 경기에서 바로 다음 타석에 다시 상대한 경우 (`run_same_batter_rematch_analysis.py`)
* **보조 — 다음 타자:** 같은 투수가 바로 다음에 상대하는 타자(항상 다른 타자) 기준. 표본이 큰 조건에서 회피 효과의 존재를 먼저 확인한 분석 (`run_pitch_avoidance_analysis.py`, `run_mixed_effects_model*.py`)

두 분석은 표본과 질문이 달라 수치를 평균내거나 직접 합칠 수 없습니다. 이벤트 생성은 모두 `src/preprocessing/build_next_ab_dataset.py`의 `build_event_dataset_for_events(..., next_pa_mode=...)`를 공유해서 장타와 대조군이 같은 로직으로 계산되도록 합니다.

## 📄 파일 구성

* `avoidance_stats.py`: 투수의 "장타 허용 구종 회피" 가설 검증에 쓰이는 통계 함수들
  * `compute_season_usage_rate()`: 투수-시즌-구종별 구사율 (경기 내 사전 비중보다 노이즈가 적은 기준선, 구종 결측 투구는 제외)
  * `build_stratified_placebo_candidates()`: 대조군(예: `field_out`)을 처치군(XBH)의 season×pitch_family 분포에 맞춰 층화추출 (`candidate_filter`로 후보를 미리 좁히고, `strata_cols`에 `pitcher`를 넣으면 같은 투수 매칭)
  * `match_treatment_to_control()`: 같은 투수 매칭처럼 대조군이 모든 층을 못 채울 때, 장타 쪽을 대조군의 층별 건수에 맞춰 1:1로 줄임
  * `summarize_paired_diff()` / `summarize_diff_in_diff()`: 짝지은 비교 / 대조군 대비 diff-in-diff 비교 (평균, 신뢰구간, t-test, Wilcoxon/Mann-Whitney)
* `run_same_batter_rematch_analysis.py`: **[메인]** 같은 타자 재대결 이벤트를 만들고, 경기 내/시즌 baseline 대비 감소 → placebo 넷팅 diff-in-diff(전체·구종 계열·시즌·baseline>0) → 이진 재사용 지표 → 혼합효과 회귀 → 동타/이타(platoon) 모델·좌우/구종별 스플릿 → 같은 투수 매칭 robustness까지 실행하는 진입점 (R이 없으면 혼합효과 이후만 생략). 이벤트 파일을 `data/processed/`에 저장해 다른 스크립트가 재사용
* `intensive_margin.py`: **[메인]** Intensive margin용 순수 함수 — 투수·시즌·구종별 baseline(위치는 타자 타석별), 코스/무브먼트/구속 편차, pre/post 투구 관측 수집
* `run_intensive_margin_analysis.py`: **[메인]** 재사용한 이벤트에서 코스·무브먼트·구속 편차가 장타 후 달라지는지를 outcome별 선형 혼합모델 3종(요청 스펙 / 이벤트 랜덤효과 추가 / 대조군 포함 `group × time`)으로 검정하고, 투수별 랜덤효과를 사례연구용으로 저장하는 진입점 (R 필요)
* `run_pitch_avoidance_analysis.py`: **[보조]** 3단계 검증(경기 내 기준선 → 시즌 기준선 → placebo diff-in-diff)을 순서대로 실행하고 요약 로그를 출력하는 진입점
* `mixed_effects_model.py`: 혼합효과 로지스틱 회귀용 순수 함수와 모델 공식 (데이터 결합, ICC 계산, 투수 랭킹, calibration 테이블, `MODEL_FORMULA_ORIGINAL`/`MODEL_FORMULA_ROBUSTNESS`)
* `glmer_runner.py`: R `lme4::glmer`(로지스틱)와 `lmer`(선형)를 rpy2로 호출해 적합하고 고정효과(오즈비/Wald CI 포함)·분산성분·랜덤효과·예측값을 꺼내는 공용 코드
* `run_mixed_effects_model.py`: **[보조]** XBH+placebo 통합 데이터로 robustness 공식 `reused_same_type ~ group * baseline_usage + ... + factor(season) + (0 + group | pitcher)` 을 적합하고, 고정효과/투수별 랜덤효과/calibration을 출력하는 진입점
* `run_mixed_effects_model_comparison.py`: **[보조]** 원본 공식(`catcher_changed` 포함, `(1 + group | pitcher)`)과 robustness 공식을 같은 데이터로 나란히 적합해 고정효과와 투수별 랜덤효과 상관을 비교하는 진입점

## 🔄 실행 방법

`data/raw`(fielder_2 포함)와 `data/processed/pitch_reuse_after_xbh_events.parquet`가 이미 있어야 합니다 (없다면 `src/collection/statcast_scraper.py` → `src/preprocessing/data_pipeline.py` 순서로 먼저 실행).

```bash
python -m src.analysis.run_same_batter_rematch_analysis   # 메인
python -m src.analysis.run_pitch_avoidance_analysis       # 보조
```

혼합효과 모델은 추가로 R과 R 패키지 `lme4`, Python 패키지 `rpy2`가 필요합니다:

```bash
brew install r cmake   # cmake는 lme4의 의존 패키지 nloptr 빌드에 필요
Rscript -e 'install.packages("lme4", repos="https://cloud.r-project.org")'
pip install rpy2

python -m src.analysis.run_mixed_effects_model
python -m src.analysis.run_mixed_effects_model_comparison
```

결과 파일은 모두 `data/processed/`(git 제외)에 저장됩니다.

* 다음 타자 기준 투수별 랜덤효과: `pitch_avoidance_mixed_model_random_effects.csv` (`run_mixed_effects_model.py`, robustness 공식). 비교 스크립트는 원본 공식 결과를 `_original.csv`, robustness 결과를 `_robustness.csv`로 따로 저장합니다
* 재대결 기준 이벤트: `pitch_reuse_after_xbh_same_batter_rematch_events.parquet`, `pitch_reuse_placebo_same_batter_rematch_events.parquet`
* 재대결 기준 투수별 랜덤효과: `pitch_avoidance_same_batter_rematch_random_effects.csv`

**참고:** `catcher_changed`는 정의상 "같은 투수가 던지는 바로 다음 타석" 구간(`has_next_ab=True`)에서만 계산되는데, 포수는 하프이닝 도중 거의 교체되지 않아 전체 표본(123,044건) 중 1이 5건뿐입니다. 회귀계수가 사실상 추정 불가능해서(표준오차가 매우 큼) `MODEL_FORMULA`에서 제외했습니다 — 원본 이벤트 데이터셋에는 컬럼 자체는 여전히 남아있습니다.
