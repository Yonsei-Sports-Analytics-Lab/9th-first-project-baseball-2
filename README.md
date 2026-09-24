# ⚾ 투수의 장타 허용 후 다음 타석에서의 구종 사용 변화

투수가 장타(2루타·3루타·홈런)를 허용한 뒤, **같은 경기에서 그 타자와 다시 맞붙는 재대결 타석에서 방금 맞은 구종을 회피하는가?** 를 MLB Statcast 투구 데이터로 검증하는 프로젝트입니다. 같은 타자 재대결이 **메인 분석**이고, "바로 다음에 상대하는 타자" 기준 분석은 회피 효과의 존재를 먼저 확인한 **보조 분석**입니다.

---

## 🔎 연구 질문과 관찰 설계

* **장타 이벤트:** `events ∈ {double, triple, home_run}` 인 투구 (79,285건)
* **메인 — 같은 타자 재대결:** 장타를 맞은 타자의 같은 경기 바로 다음 타석을 같은 투수가 다시 상대한 경우(`next_pa_mode="same_batter"`). 장타 79,285건 중 28,765건(36.3%), 구종 결측을 뺀 28,745건
* **보조 — 다음 타자:** 같은 `game_pk`, 같은 투수, `at_bat_number + 1` 인 타석 — 항상 **다른 타자**이며, 투수 교체 등으로 없으면 `has_next_ab = False`(전체의 6.1%, 분석 제외). 74,437건
* **핵심 변수:** `baseline_usage`(그 경기에서 해당 투구 이전까지 그 구종 사용 비중), `same_type_share`(비교 타석에서 그 구종 비중), `reused_same_type`(비교 타석에 한 번이라도 다시 던졌는지, 0/1). 시즌 전체 구사율(`season_usage_rate`)을 더 안정적인 baseline으로도 사용
* **구종 계열(`pitch_family`):** fastball(FF·SI·FT·FA·FC) / breaking(SL·ST·CU·KC·CS·SV·SC) / offspeed(CH·FS·FO·EP·KN), 미분류는 `other`
* **대조군(placebo):** `field_out` 이벤트를 장타 그룹의 season × pitch_family 분포에 맞춰 층화추출(seed 고정)하고, 장타와 **완전히 동일한 코드 경로**로 같은 변수를 계산. 메인 분석의 대조군은 같은 타자 재대결이 가능한 `field_out`만 후보로 삼음

---

## 📌 핵심 결과 (2026-09-24 기준)

데이터: 2021 시즌 ~ 2026-07-14(올스타전 전날) 정규시즌, 투구 3,993,429건

### 메인: 같은 타자 재대결 (장타 28,745건 vs 대조군 `field_out` 28,745건)

| 단계 | 결과 |
|---|---|
| ① 시즌 전체 구사율 baseline 대비 (대조군 없음) | 사용 비중 29.7% → 18.7%, **−11.0%p** (p≈0) |
| ② 대조군 넷팅 diff-in-diff, 시즌 baseline | 장타 −11.0%p vs 대조군 +3.3%p → **순수 감소 14.3%p** (95% CI [13.9, 14.7], p≈0) |
| ③ 대조군 넷팅, 경기 내 baseline | 장타 −15.7%p vs 대조군 −0.5%p → **순수 감소 15.2%p** (95% CI [14.7, 15.7], p≈0) |
| ④ 재사용률 | 장타 45.6% vs 대조군 67.5% (χ² p≈0) |
| ⑤ 혼합효과 로지스틱 회귀 | `group` 오즈비 **0.278**(평균 baseline에서 약 0.37), `group × baseline_usage` 오즈비 **2.296** (p<0.001) |

* 구종 계열별 순수 감소(시즌 baseline)는 fastball·breaking·offspeed 모두 약 13~17%p입니다.
* 시즌별 감소폭(시즌 baseline, 대조군 없음)은 6개 시즌 모두 −10.4%p ~ −11.3%p로 일관됩니다.

  | Season | Events | Baseline | Post | Diff |
  |---|---|---|---|---|
  | 2021 | 4,951 | 32.49% | 21.19% | −11.30%p |
  | 2022 | 4,859 | 30.67% | 19.33% | −11.34%p |
  | 2023 | 5,525 | 30.58% | 19.47% | −11.11%p |
  | 2024 | 5,227 | 28.59% | 18.17% | −10.42%p |
  | 2025 | 5,165 | 28.27% | 17.29% | −10.98%p |
  | 2026 | 3,018 | 26.15% | 15.14% | −11.01%p |

* ⑤ 해석: 장타를 맞으면 재사용 오즈가 대조군보다 크게 낮아지고, 평소 그 구종 의존도가 높을수록 회피 효과가 약해집니다(대체할 구종이 마땅치 않으면 못 피함). 표본 56,432행(장타 28,174 + 대조군 28,258, 경기 내 baseline 결측 제외), 투수 1,009명, 투수별 기울기 분산 0.0701, 수렴 정상·isSingular 아님.
* 범석님 노트북(Python, 시즌 baseline)과 이 저장소의 독립 재구현이 일치합니다: 재대결 건수는 시즌별 1~7건 차이, 시즌 baseline은 0.02%p 이내, 시즌별 감소폭은 최대 0.10%p 차이.
* 재구현 스크립트: `src/analysis/run_same_batter_rematch_analysis.py` (투수별 랜덤효과는 `data/processed/pitch_avoidance_same_batter_rematch_random_effects.csv`, 로컬 전용).

**추가 검증 (2026-09-24, 같은 스크립트와 `run_intensive_margin_analysis.py`)**

| 검증 | 결과 |
|---|---|
| ⑥ 동타/이타(platoon) 혼합모델 | `platoon_match` 오즈비 1.145 [1.082, 1.212] (p<0.001), `group × platoon_match` 오즈비 **0.962** [0.894, 1.036] (p=0.30) — 회피 효과가 동타/이타에 따라 다르다는 증거 없음 |
| ⑥ 순수 감소 스플릿 (시즌 baseline) | 이타 13.7%p vs 동타 15.0%p, 타자 좌/우 14.3%p 대 14.3%p, 투수 좌/우 14.2%p 대 14.3%p. 구종별(표본 300건 이상)은 FF 11.7%p ~ FS 19.0%p |
| ⑦ 투수 단위 매칭 (같은 투수×시즌×구종계열의 `field_out`만 대조군) | 장타의 98.5%(28,328건, 839명)가 1:1 매칭. 순수 감소 14.4%p / 15.2%p로 층화 매칭(14.3 / 15.2)과 동일, `group` 오즈비 0.296 대 0.278, `group × baseline_usage` 1.86 대 2.30(둘 다 p<0.001), 투수별 랜덤 기울기 상관 Pearson 0.994 / Spearman 0.986 |
| ⑨ 구종별 회피 (9개 구종, `run_pitch_type_avoidance_analysis.py`) | 전 구종에서 회피(`group` 오즈비 0.22~0.50, 모두 CI가 1에서 멀음), 구종에 따라 강도가 다름(우도비 검정 χ²=142.8, df=8, p<0.001). FF가 가장 약하고(0.50) FS·CU·FC·CH가 가장 강함(0.22~0.25) |
| ⑩ 구종별 재구사율 감소 (7개 구종 FF·SI·SL·CH·FC·CU·ST, `run_pitch_type_reuse_rate_analysis.py`) | 재대결 구사율이 사전 대비 장타 후 크게 줄고(전체 30.0% → 19.0%), 대조군 보정 순수 감소는 포심 11.7%p ~ 체인지업 16.3%p(전체 14.2%p), 상대 감소율은 포심 30% ~ 커브 63%(전체 43%). 재사용률은 장타 46.3% vs 대조군 67.8% |
| ⑧ Intensive margin (재사용한 이벤트만: 장타 13,103건, 대조군 19,400건) | 대조군 대비 변화(`group × time`): 코스 **+0.019 ft** [0.009, 0.028] (p=0.0002, 약 0.2인치), 무브먼트 +0.0014 ft (p=0.24), 구속 −0.003 mph (p=0.78) |

* ⑧의 대조군 없는 pre→post 구속 변화는 −0.32 mph이지만 대조군도 −0.37 mph 떨어집니다. 경기가 진행되며(피로 등) 생기는 변화이고 장타에 대한 반응이 아닙니다. 그래서 Intensive margin은 대조군 대비 값을 기준으로 봅니다.
* 코스 변화는 통계적으로는 유의하지만 편차 표준편차(0.53 ft)의 약 3.5%로 실질 크기는 아주 작습니다. 무브먼트·구속에서는 장타에 특정한 변화가 관찰되지 않았습니다.
* 변수 정의와 결측 처리는 [docs/variable_spec.md](docs/variable_spec.md)에 정리했습니다.

### 보조: 다음 타자 기준 (장타 72,886건 vs 대조군 50,158건)

| 단계 | 결과 |
|---|---|
| ① 경기 내 baseline 대비 (noisy, 대조군 없음) | 사용 비중 36.1% → 26.0%, **−10.1%p** (p≈0) |
| ② 시즌 전체 구사율 baseline 대비 | 32.4% → 26.2%, **−6.2%p** (p≈0) — 평균회귀 일부 확인 |
| ③ 대조군(field_out) 넷팅 diff-in-diff | 장타 −10.1%p vs 대조군 −2.9%p → **순수 감소 7.25%p** (95% CI [6.90, 7.60], p≈0), ①의 71.6% |
| ④ 혼합효과 로지스틱 회귀 | `group` 오즈비 **0.592** (p≈7e-94), `group × baseline_usage` 오즈비 **1.146** (p=0.031) |
| ⑤ robustness (랜덤효과 구조 단순화) | 투수별 랜덤 기울기 상관 **1.0000**, isSingular 해소 → 단순화 모델을 메인 공식으로 채택 |

* 모델: `reused_same_type ~ group * baseline_usage + balls + strikes + outs_when_up + score_diff + stand + pitch_family + factor(season) + (0 + group | pitcher)`, R `lme4::glmer`(nAGQ=0, bobyqa). 표본 123,044행, 투수 1,769명. 예측확률 10분위 calibration은 전 구간에서 근접.
* 투수별 회피 성향(랜덤 기울기)은 `data/processed/pitch_avoidance_mixed_model_random_effects.csv`에 저장(로컬 전용).

### 두 분석의 관계

* 표본과 모집단이 달라 수치를 평균내거나 그대로 합칠 수 없습니다.
* 참고로 경기 내 baseline 기준 순수 감소는 재대결 15.2%p, 다음 타자 7.25%p이고 혼합모델 `group` 오즈비는 0.278 대 0.592입니다. 다만 재대결은 타자가 다시 타석에 설 때까지 투수가 남아 있는 경우로 한정돼 모집단(투수 유형, 경기 시점 등)이 다르므로, 이 차이가 "그 타자에 대한 맞춤 조정" 때문인지는 아직 분리되지 않았습니다.

---

## 📁 저장소 구조 (Repository Structure)

협업 시 충돌을 방지하고 코드의 가독성을 높이기 위해 아래의 디렉토리 구조를 엄격히 준수합니다.

* `data/raw/`: 원본 데이터 파일 보관 (수정 절대 금지) — `{year}/{month}.parquet` 월별 파일
* `data/processed/`: 전처리 및 정제가 완료된 데이터, 분석 결과 CSV 보관
* `notebooks/`: 탐색적 데이터 분석(EDA) 및 실험용 Jupyter Notebook
* `src/`: 프로젝트의 핵심 로직을 담당하는 파이썬 모듈
  * `collection/`: Statcast 월 단위 수집기 (재실행 시 이미 받은 달은 건너뜀, 일시 오류 시 재시도)
  * `preprocessing/`: 이벤트 데이터셋 생성(`build_next_ab_dataset.py`, 다음 타자/같은 타자 재대결 두 모드), 구종 계열 분류, 분포 리포트, 전체 파이프라인(`data_pipeline.py`)
  * `analysis/`: 같은 타자 재대결 분석(메인), 다음 타자 기준 회피 가설 검증(시즌 baseline, placebo diff-in-diff), 혼합효과 로지스틱 회귀(R `lme4` 연동)
  * `models/`, `utils/`, `visualization/`: 아직 사용 전 (폴더별 README 참고)
* `tests/`: 순수 함수 단위 테스트 (77개)
* `docs/`: 변수 스펙(`variable_spec.md`), 초기 구현 계획서(`superpowers/plans/`)
* `requirements.txt`: 프로젝트 실행에 필요한 파이썬 패키지 목록

---

## 🚀 진행 현황 (Progress)

- [x] 데이터 수집 (2021~2026, 45개월)
- [x] 회피 효과 실재 검증 — 다음 타자 기준(보조): 대조군 비교, diff-in-diff
- [x] Extensive margin 혼합효과모델 — 다음 타자 기준(보조) 및 robustness 체크
- [x] 메인 분석(같은 타자 재대결) 대조군 검증 + 혼합효과모델, 범석님 노트북과 교차검증
- [x] 대조군/변수 정리: 동타/이타(platoon) 검정, 투수 단위 매칭 robustness, 변수 스펙 문서 (2026-09-24)
- [x] Intensive margin (구종은 유지해도 코스·무브먼트·구속을 바꾸는지) (2026-09-24)
- [ ] 가치모델 (RL 프레임 또는 run value 기반 GBM)
- [ ] 사례연구 (회피 성향 상/하위 투수)
- [ ] 최종 산출물 / 발표자료

---

## 🧪 방법론 메모 및 한계

* `baseline_usage`는 한 경기 내 적은 투구 수로 계산돼 노이즈가 큽니다. 대조군 없이 나온 감소폭(−10.1%p, −15.7%p 등)을 그대로 인용하지 말고 대조군 보정치를 기준으로 사용하세요.
* 메인 분석의 대조군은 시즌 baseline에서 오히려 사용 비중이 늘어납니다(+3.3%p). 그 구종이 이 경기에서 실제로 던져졌다는 조건 때문에 같은 경기 안에서 쓰임이 이어지는 것으로 추정되며(검증 전), 이 효과는 장타 그룹에도 똑같이 작용하므로 대조군으로 걷어내는 게 맞습니다.
* 대조군은 season × pitch_family 층화추출이라 투수, 이닝, 카운트, 주자 상황까지 매칭된 것은 아닙니다. 장타는 위기 상황과 함께 나오는 경우가 많아 "장타를 맞아서"와 "위기라서"가 섞였을 가능성이 있습니다(주자 상황은 수집하지 않음).
* 다음 타자 분석에서 `has_next_ab`/`baseline_usage` 필터 후 표본 유지율이 장타(약 92%)보다 `field_out`(약 63%)이 훨씬 낮은데, 3아웃째로 이닝이 끝나는 경우 등이 원인으로 추정되며 선택 편향 가능성이 있습니다(`outs_when_up` 통제 등 추가 확인 필요).
* `pitch_type`이 없는 투구는 구사율(분자·분모)에서 제외하고, 이벤트 투구의 구종이 결측인 장타(39건)는 재사용 지표를 NaN으로 둡니다. 비교 타석 안의 결측 구종 투구는 `same_type_share` 분모에 포함되는데, 범석님 노트북은 이를 제외해 시즌별 감소폭이 최대 0.10%p 달라지는 것으로 보입니다.
* Intensive margin은 "다시 던진 이벤트"로만 조건을 걸기 때문에(재사용 자체가 결과의 일부) 선택 효과가 있을 수 있습니다. pre는 같은 경기 안의 이전 같은 구종 투구라 시간 경과(피로 등)와 섞이므로 대조군 대비 값(`group × time`)을 기준으로 해석하세요. 편차의 baseline에는 비교 대상 투구도 포함됩니다.
* `catcher_changed`(포수 교체 여부)는 모델에서 제외했습니다. `has_next_ab=True`는 하프이닝이 끊기지 않은 구간이라 포수 교체가 구조적으로 거의 없어(123,044건 중 5건) 계수를 추정할 수 없습니다. 컬럼 자체는 이벤트 데이터셋에 남아 있습니다.
* 초기 스펙의 구종 계열 표기가 2분류/3분류로 엇갈려 3분류(fastball/breaking/offspeed)로 확정했습니다.

---

## ⚠️ 주의사항 (Precautions)

안전하고 원활한 협업을 위해 아래 사항을 반드시 숙지해 주세요.

* **대용량 데이터 업로드 금지:** 용량이 큰 CSV 파일이나 수만 건의 로우 데이터는 절대 GitHub에 직접 Commit/Push 하지 마세요. 반드시 `.gitignore`를 확인하고, 데이터는 구글 드라이브 등 별도의 스토리지를 통해 공유해야 합니다.
* **보안 정보 노출 주의:** API Key, 데이터베이스 비밀번호, 개인 정보 등은 절대 코드에 직접 작성하지 마세요. `.env` 파일을 활용하고 해당 파일이 `.gitignore`에 포함되어 있는지 확인해야 합니다.
* **Main 브랜치 직접 Push 금지:** `main` 브랜치에 코드를 직접 올리는 것은 금지되어 있습니다. 반드시 각자의 작업 브랜치를 생성(`feat/데이터수집` 등)하여 작업한 후, Pull Request(PR)를 통해 코드 리뷰를 거쳐 병합(Merge)하세요.
* **환경 동기화:** 새로운 라이브러리를 설치한 경우, 반드시 `pip freeze > requirements.txt` 명령어를 통해 의존성 목록을 업데이트하고 커밋해 주세요.
* **이슈 트래킹:** 새로운 작업을 시작하거나 버그를 발견했을 때는 항상 템플릿에 맞추어 `Issue`를 먼저 등록해 주세요.

---

## 💻 시작하기 (Getting Started)

1. 저장소 클론: `git clone [저장소 URL]`
2. 디렉토리 이동: `cd [저장소 이름]`
3. 가상환경 생성 및 실행: (권장) `python -m venv .venv` 후 활성화
4. 패키지 설치: `pip install -r requirements.txt`

### 분석 재현 순서

`data/raw`, `data/processed`는 `.gitignore` 대상이라 저장소에 없습니다. 아래 순서로 로컬에서 다시 생성하세요 (모두 저장소 루트에서 실행).

```bash
# 1. Statcast 수집 (2021-03 ~ 2026-07-14, 네트워크에 따라 약 10~20분). 중단돼도 재실행하면 이어서 받음
python -m src.collection.statcast_scraper

# 2. 이벤트 데이터셋 생성 + 구종 분포 리포트 (약 2분)
python -m src.preprocessing.data_pipeline

# 3. [메인] 같은 타자 재대결 분석 (baseline 2종 → placebo 넷팅 → 이진 지표 → 혼합효과 → platoon → 투수 단위 매칭, 약 3분)
python -m src.analysis.run_same_batter_rematch_analysis

# 3-1. [메인] Intensive margin (코스·무브먼트·구속 편차, 3.의 이벤트 파일을 재사용, 약 2분, R 필요)
python -m src.analysis.run_intensive_margin_analysis

# 3-2. [메인] 구종별 회피 (3.의 이벤트 파일을 재사용, 약 10초, R 필요)
python -m src.analysis.run_pitch_type_avoidance_analysis

# 3-3. [메인] 7개 구종 재구사율 감소 요약 + 그림 3장 (약 10초, R 불필요, 그림은 data/processed/figures/)
python -m src.analysis.run_pitch_type_reuse_rate_analysis
python -m src.visualization.pitch_type_reuse_plots

# 4. [보조] 다음 타자 기준 회피 가설 검증 (경기 내 baseline → 시즌 baseline → placebo diff-in-diff)
python -m src.analysis.run_pitch_avoidance_analysis

# 5. [보조] 다음 타자 기준 혼합효과 로지스틱 회귀 / 원본 vs robustness 비교 (R 필요, 아래 참고)
python -m src.analysis.run_mixed_effects_model
python -m src.analysis.run_mixed_effects_model_comparison

# 단위 테스트
pytest
```

### R 환경 (혼합효과 모델용)

`statsmodels`의 `MixedLM`은 로지스틱 랜덤효과를 지원하지 않아 R `lme4::glmer`를 `rpy2`로 호출합니다. macOS(Homebrew) 기준:

```bash
brew install r cmake   # cmake는 lme4 의존 패키지(nloptr) 빌드에 필요
Rscript -e 'install.packages("lme4", repos="https://cloud.r-project.org")'
```

Homebrew R을 쓰면 실행 시 `Error importing in API mode ... Trying to import in ABI mode.` 메시지가 뜨지만 ABI 모드로 정상 동작하니 무시해도 됩니다. R이 없으면 재대결 분석 스크립트는 혼합효과 단계만 건너뜁니다.
