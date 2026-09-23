# ⚾ 투수의 장타 허용 후 다음 타석에서의 구종 사용 변화

투수가 장타(2루타·3루타·홈런)를 허용한 뒤, **같은 경기에서 바로 이어지는 다음 타석에서 방금 맞은 구종을 회피하는가?** 를 MLB Statcast 투구 데이터로 검증하는 프로젝트입니다.

---

## 🔎 연구 질문과 관찰 설계

* **장타 이벤트:** `events ∈ {double, triple, home_run}` 인 투구 (79,285건)
* **다음 타석:** 같은 `game_pk`, 같은 투수, `at_bat_number + 1` 인 타석 — 투수 교체 등으로 없으면 `has_next_ab = False` (전체의 6.1%, 메인 분석에서 제외)
* **핵심 변수:** `baseline_usage`(그 경기에서 해당 투구 이전까지 그 구종 사용 비중), `same_type_share`(다음 타석에서 그 구종 비중), `reused_same_type`(다음 타석에 한 번이라도 다시 던졌는지, 0/1)
* **구종 계열(`pitch_family`):** fastball(FF·SI·FT·FA·FC) / breaking(SL·ST·CU·KC·CS·SV·SC) / offspeed(CH·FS·FO·EP·KN), 미분류는 `other`
* **대조군(placebo):** `field_out` 이벤트를 장타 그룹의 season × pitch_family 분포에 맞춰 층화추출(seed 고정)하고, 장타와 **완전히 동일한 코드 경로**로 같은 변수를 계산

---

## 📌 핵심 결과 (2026-09-23 기준)

데이터: 2021 시즌 ~ 2026-07-14(올스타전 전날) 정규시즌, 투구 3,993,429건, 장타 이벤트 79,285건 (`has_next_ab=True` 74,437건, 93.9%)

| 단계 | 결과 |
|---|---|
| ① 경기 내 baseline 대비 (noisy, 대조군 없음) | 사용 비중 36.1% → 26.0%, **−10.1%p** (p≈0) |
| ② 시즌 전체 구사율 baseline 대비 | 32.4% → 26.2%, **−6.2%p** (p≈0) — 평균회귀 일부 확인 |
| ③ 대조군(field_out) 넷팅 diff-in-diff | 장타 −10.1%p vs 대조군 −2.9%p → **순수 감소 7.25%p** (95% CI [6.90, 7.60], p≈0), ①의 71.6% |
| ④ 혼합효과 로지스틱 회귀 | `group` 오즈비 **0.592** (p≈7e-94), `group × baseline_usage` 오즈비 **1.146** (p=0.031) |
| ⑤ robustness (랜덤효과 구조 단순화) | 투수별 랜덤 기울기 상관 **1.0000**, isSingular 해소 → 단순화 모델을 메인으로 채택 |

* ④ 해석: 장타를 맞으면 재사용 오즈가 약 41% 낮아지고, 평소 그 구종 의존도가 높을수록 회피 효과가 약해집니다(대체할 구종이 마땅치 않으면 못 피함).
* 모델: `reused_same_type ~ group * baseline_usage + balls + strikes + outs_when_up + score_diff + stand + pitch_family + factor(season) + (0 + group | pitcher)`, R `lme4::glmer`(nAGQ=0, bobyqa). 표본 123,044행(장타 72,886 + 대조군 50,158), 투수 1,769명.
* 예측확률 10분위 calibration: 예측과 실제 재사용률이 전 구간에서 근접.
* 투수별 회피 성향(랜덤 기울기)은 `data/processed/pitch_avoidance_mixed_model_random_effects.csv`에 저장(로컬 전용).

---

## 📁 저장소 구조 (Repository Structure)

협업 시 충돌을 방지하고 코드의 가독성을 높이기 위해 아래의 디렉토리 구조를 엄격히 준수합니다.

* `data/raw/`: 원본 데이터 파일 보관 (수정 절대 금지) — `{year}/{month}.parquet` 월별 파일
* `data/processed/`: 전처리 및 정제가 완료된 데이터, 분석 결과 CSV 보관
* `notebooks/`: 탐색적 데이터 분석(EDA) 및 실험용 Jupyter Notebook
* `src/`: 프로젝트의 핵심 로직을 담당하는 파이썬 모듈
  * `collection/`: Statcast 월 단위 수집기 (재실행 시 이미 받은 달은 건너뜀, 일시 오류 시 재시도)
  * `preprocessing/`: 이벤트 데이터셋 생성(`build_next_ab_dataset.py`), 구종 계열 분류, 분포 리포트, 전체 파이프라인(`data_pipeline.py`)
  * `analysis/`: 회피 가설 검증(시즌 baseline, placebo diff-in-diff), 혼합효과 로지스틱 회귀(R `lme4` 연동)
  * `models/`, `utils/`, `visualization/`: 아직 사용 전 (폴더별 README 참고)
* `tests/`: 순수 함수 단위 테스트 (47개)
* `docs/superpowers/plans/`: 초기 구현 계획서
* `requirements.txt`: 프로젝트 실행에 필요한 파이썬 패키지 목록

---

## 🚀 진행 현황 (Progress)

- [x] 데이터 수집 (2021~2026, 45개월)
- [x] 회피 효과 실재 검증 (대조군 비교, diff-in-diff)
- [x] Extensive margin 혼합효과모델 (다시 던졌는가 O/X) 및 robustness 체크
- [ ] Intensive margin (구종은 유지해도 코스·무브먼트·구속을 바꾸는지)
- [ ] 가치모델 (RL 프레임 또는 run value 기반 GBM)
- [ ] 사례연구 (회피 성향 상/하위 투수)
- [ ] 최종 산출물 / 발표자료

**보완 분석:** 같은 타자와의 바로 다음 재대결로 한정한 분석(팀원 노트북, 아직 repo에 미반영)에서는 2021~2026 전 시즌 −10.4%p ~ −11.3%p로 일관된 감소가 관찰됐습니다. 다만 메인 분석(다음 타석 아무 타자)과는 다른 질문이라 직접 비교할 수 없고, 대조군 검증은 아직 없습니다.

---

## 🧪 방법론 메모 및 한계

* `baseline_usage`는 한 경기 내 적은 투구 수로 계산돼 노이즈가 큽니다. ①의 −10.1%p를 그대로 인용하지 말고 대조군 보정치(③, 7.25%p)를 기준으로 사용하세요.
* `catcher_changed`(포수 교체 여부)는 모델에서 제외했습니다. `has_next_ab=True`는 하프이닝이 끊기지 않은 구간이라 포수 교체가 구조적으로 거의 없어(123,044건 중 5건) 계수를 추정할 수 없습니다. 컬럼 자체는 이벤트 데이터셋에 남아 있습니다.
* 대조군은 season × pitch_family 층화추출이라 투수 단위로 완전히 매칭된 것은 아닙니다. `has_next_ab`/`baseline_usage` 필터 후 표본 유지율이 장타(약 92%)보다 `field_out`(약 63%)이 훨씬 낮은데, 3아웃째로 이닝이 끝나는 경우 등이 원인으로 추정되며 선택 편향 가능성이 있습니다(`outs_when_up` 통제 등 추가 확인 필요).
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

# 3. 회피 가설 검증 (경기 내 baseline → 시즌 baseline → placebo diff-in-diff)
python -m src.analysis.run_pitch_avoidance_analysis

# 4. 혼합효과 로지스틱 회귀 (R 필요, 아래 참고) / 원본 vs robustness 비교
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

Homebrew R을 쓰면 실행 시 `Error importing in API mode ... Trying to import in ABI mode.` 메시지가 뜨지만 ABI 모드로 정상 동작하니 무시해도 됩니다.
