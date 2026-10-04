# ⚾ 투수의 장타 허용 후 구종 회피

MLB 투수가 장타(2루타·3루타·홈런)를 맞은 뒤, **그 공의 구종을 이후에 덜 던지는가**를 Statcast 투구 데이터(2021–2026, 359만 구)로 검증하는 프로젝트입니다.

> 📄 **전체 내용은 [분석 보고서](docs/보고서.md)를 먼저 읽어 보세요.** 왜 이 분석을 했는지부터 결과·한계까지 그림과 함께 정리했습니다.
>
> 📋 진행 순서는 [docs/작업계획.md](docs/작업계획.md), 파일 안내는 [docs/파일구조.md](docs/파일구조.md)를 보세요.

---

## ❓ 핵심 질문과 현재 답

| # | 질문 | 답 | 근거 |
|---|---|---|---|
| 1 | 장타를 맞은 구종을 **같은 경기에서 그 타자를 다시 만났을 때** 덜 던지는가? | **예.** 비슷한 상황의 아웃(`field_out`) 뒤보다 약 **14%p** 덜 던지고, 다시 던질 오즈는 대조군의 약 **0.29배** | [노트북 03](notebooks/03_CY_matching_ladder_by_mode.ipynb) |
| 2 | **경기 상황**(주자·아웃·이닝·카운트·점수차)과 **그날 그 구종의 상태**를 맞춰도 남는가? | **예.** 대조군을 단계별로 맞춰도(M0 → M3) 14.39 → 14.34%p로 거의 그대로. 그날 그 구종의 상태와 장타 맞은 공의 위치·구속까지 맞춰도 13.6%p 이상 남음 | [노트북 03](notebooks/03_CY_matching_ladder_by_mode.ipynb), [07](notebooks/07_CY_pitch_state_control.ipynb) |
| 3 | **바로 다음 타자**에게도 피하는가? | **약하게 예.** 약 6.7%p로 재대결의 절반 이하. 단, 2아웃 장타는 비교할 대조군이 없어 빠지고, 다음 타석에 주자가 많은 차이가 섞임 | [노트북 03](notebooks/03_CY_matching_ladder_by_mode.ipynb) |
| 4 | 회피는 **장타를 친 그 타자**를 향한 것인가, 그 구종을 전반적으로 덜 쓰게 된 것인가? | **주로 그 타자.** 같은 장타 안에서 중간 타자들에게는 4.0%p, 그 타자와의 재대결에서는 13.5%p 더 피함(타자 맞춤 9.5%p). 장타 전에는 그 타자에게 오히려 그 구종을 더 던졌음 | [노트북 05](notebooks/05_CY_batter_specific_avoidance.ipynb) |
| 5 | **결과**(장타) 때문인가, **타구 질**(잘 맞음) 때문인가? | **둘 다.** 잘 맞은 아웃도 피하고(배럴 아웃 +10.1%p), 타구 질이 같아도 결과가 나쁠수록 더 피함. 홈런은 타구 질과 무관하게 크게 피함 | [노트북 04](notebooks/04_CY_hit_quality_avoidance.ipynb) |
| 6 | 그렇게 피한 것이 **투수에게 좋은 선택**이었나? | **아직 모름.** 다음 단계(Run Value 기반 평가, 회피 성향 상·하위 투수 사례) | [작업계획](docs/작업계획.md) |

**한 줄 요약:** 투수는 장타를 맞은 구종을 **그 타자에게** 확실히 덜 던지고, 이 경향은 경기 상황을 맞춰도 남으며, 결과와 타구 질 모두에 반응합니다. 이것이 좋은 선택이었는지가 다음 질문입니다.

### 용어

* **재대결(메인):** 같은 경기에서 장타를 친 타자의 다음 타석을 같은 투수가 상대한 경우
* **다음 타자(보조):** 같은 투수가 바로 다음 타석(다른 타자)을 상대한 경우
* **대조군:** 장타 대신 `field_out`(타구가 나왔지만 아웃)으로 끝난 타석. 시즌·구종계열(M0)부터 주자·아웃(M1), 이닝·카운트(M2), 점수차(M3)까지 단계별로 장타와 맞춰 뽑음
* **순수 감소:** (평소 구사율 − 비교 타석에서 그 구종 비율)의 장타 평균 − 대조군 평균. 대조군도 같은 경기에서 이미 던진 구종을 다시 쓰는 경향이 있어서, 그만큼을 빼고 봄

> 초기 분석은 대조군 설정에 문제가 있어 [archive/01_초기분석/](archive/01_초기분석/README.md)에 시행착오로 보관했습니다.

---

## 📁 저장소 구조 (Repository Structure)

협업 시 충돌을 방지하고 코드의 가독성을 높이기 위해 아래의 디렉토리 구조를 엄격히 준수합니다.

* `data/raw/`: 원본 데이터 파일 보관 (수정 절대 금지) — 팀 공유 research CSV 6개(점수 컬럼 포함, [설명](data/raw/README.md)). `src/` 파이프라인용 `{year}/{month}.parquet` 월별 파일도 이 폴더에 둡니다
* `data/research/`: `data/raw`의 CSV를 가리키는 심볼릭 링크 (git 제외, [만드는 법](data/raw/README.md)). 분석 코드는 이 이름으로 CSV를 찾습니다
* `data/processed/`: 전처리 및 정제가 완료된 데이터, 분석 결과 CSV 보관
* `notebooks/`: 실험 노트북. `01`·`02` 상황 매칭 대조군·재대결 실행 변화, **`03` M0~M3 사다리(재대결·다음 타자 분리)**, **`04` 타구 질과 회피**, **`05` 타자 맞춤 회피 점검**, **`06` 신뢰구간 재계산(투수 cluster)**, **`07` 그날 구종 상태 통제**
* `src/`: 프로젝트의 핵심 로직을 담당하는 파이썬 모듈
  * `collection/`: Statcast 월 단위 수집기 (재실행 시 이미 받은 달은 건너뜀, 일시 오류 시 재시도)
  * `preprocessing/`: 이벤트 데이터셋 생성(`build_next_ab_dataset.py`, 다음 타자/같은 타자 재대결 두 모드), 구종 계열 분류, 분포 리포트, 전체 파이프라인(`data_pipeline.py`)
  * `analysis/`: 같은 타자 재대결 분석(메인), 다음 타자 기준 회피 가설 검증(시즌 baseline, placebo diff-in-diff), 혼합효과 로지스틱 회귀(R `lme4` 연동), 상황 매칭 사다리(`situation_matching.py`), 타구 질(`hit_quality.py`)
  * `models/`, `utils/`, `visualization/`: 아직 사용 전 (폴더별 README 참고)
* `tests/`: 순수 함수 단위 테스트 (107개)
* `docs/`: **분석 보고서(`보고서.md`, 그림 `figures/`)**, 작업 계획(`작업계획.md`), 파일별 역할(`파일구조.md`), 타구 질 추가 연구(`추가연구_타구질.md`), 상세 결과 보관본(`분석결과_상세.md`), 변수 스펙(`variable_spec.md`), 초기 구현 계획서(`superpowers/plans/`)
* `archive/`: 더 이상 결론에 쓰지 않는 이전 분석 (시행착오 기록)
* `requirements.txt`: 프로젝트 실행에 필요한 파이썬 패키지 목록


---

## 🚀 진행 현황 (Progress)

- [x] 데이터 고정: 팀 공유 research CSV (2021–2026-07-12, 투수-시즌 500구 이상, 2026은 300구 이상, 점수 컬럼 포함)
- [x] 회피 확인: 재대결(메인)·다음 타자(보조), 대조군 diff-in-diff, 혼합효과 모델
- [x] 검증: 동타/이타, 투수 단위 매칭, 구종별, 시즌별, intensive margin(코스·무브먼트·구속)
- [x] 상황 통제: M0~M3 매칭 사다리(점수차 포함), 재대결·다음 타자 분리 (2026-10-05)
- [x] 타구 질: xwOBA·배럴과 결과의 효과 분리 (2026-10-05)
- [x] 점검: 타자 맞춤 회피, 그날 구종 상태, 투수 cluster 신뢰구간 (2026-10-05)
- [x] 보고서: [docs/보고서.md](docs/보고서.md) (2026-10-05)
- [ ] 회피의 합리성 평가 (Run Value 기반 GBM)
- [ ] 사례연구 (회피 성향 상/하위 투수)

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

`data/raw`, `data/research`, `data/processed`는 `.gitignore` 대상이라 저장소에 없습니다. 팀 공유 research CSV 6개(`statcast_<연도>_min500_research_score.csv`)를 [공유 드라이브](https://drive.google.com/drive/folders/1nP5Z0f7q1cNftgKkVTsWe329ishi8iUM)에서 받아 `data/raw/`에 넣고 `data/research/`에 링크를 건 뒤([방법](data/raw/README.md)), 아래 순서로 로컬에서 다시 생성하세요 (모두 저장소 루트에서 실행). ⚠️ 1~5단계는 Statcast를 다시 수집한 `data/raw/{연도}/{월}.parquet`이 필요합니다. 고정 데이터만 쓰려면 CSV만 읽는 노트북(6단계)을 실행하세요. 이후 모든 단계는 `data/raw` 중 CSV에 있는 행만 씁니다(`src/preprocessing/research_sample.py`). CSV 위치를 바꾸려면 `RESEARCH_SAMPLE_DIR` 환경변수를 지정하세요. 파일명은 `statcast_<연도>_min<컷오프>_research.csv` 형식이어야 하며, 형식이 어긋난 `statcast_*.csv`(예: 다운로드하며 붙은 ` (1)`)가 있으면 로더가 에러로 알려 줍니다.

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

# 3-4. [진단] 대조군 사용 비중이 왜 늘어나는지(성공 강화 vs 같은 경기 내 구종 지속) 결과별 앵커로 확인 (약 3분)
python -m src.analysis.run_outcome_anchor_check

# 4. [보조] 다음 타자 기준 회피 가설 검증 (경기 내 baseline → 시즌 baseline → placebo diff-in-diff)
python -m src.analysis.run_pitch_avoidance_analysis

# 5. [보조] 다음 타자 기준 혼합효과 로지스틱 회귀 / 원본 vs robustness 비교 (R 필요, 아래 참고)
python -m src.analysis.run_mixed_effects_model
python -m src.analysis.run_mixed_effects_model_comparison

# 6. [3단계] 노트북 (CSV만 사용, R 필요). PATH에 R이 있어야 함
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 03_CY_matching_ladder_by_mode.ipynb   # 약 8분
jupyter nbconvert --to notebook --execute --inplace 04_CY_hit_quality_avoidance.ipynb     # 약 1분
jupyter nbconvert --to notebook --execute --inplace 05_CY_batter_specific_avoidance.ipynb # 약 3분
jupyter nbconvert --to notebook --execute --inplace 06_CY_robust_inference.ipynb          # 약 2분 (03 결과 CSV 필요)
jupyter nbconvert --to notebook --execute --inplace 07_CY_pitch_state_control.ipynb       # 약 1분
cd ..

# 7. 보고서 그림 (docs/figures/)
python -m src.visualization.report_figures

# 단위 테스트
pytest
```

### R 환경 (혼합효과 모델용)

`statsmodels`의 `MixedLM`은 로지스틱 랜덤효과를 지원하지 않아 R `lme4::glmer`를 `rpy2`로 호출합니다. macOS(Homebrew) 기준:

```bash
brew install r cmake   # cmake는 lme4 의존 패키지(nloptr) 빌드에 필요
Rscript -e 'install.packages("lme4", repos="https://cloud.r-project.org")'
```

Homebrew가 없으면 [CRAN](https://cloud.r-project.org/bin/macosx/)의 `.pkg`로 R을 설치하고 `Rscript -e 'install.packages("lme4", repos="https://cloud.r-project.org", type="binary")'`로 `lme4`를 설치해도 됩니다(cmake 불필요). 이 경우 `R`이 `PATH`에 없을 수 있으니 `export PATH=/Library/Frameworks/R.framework/Resources/bin:$PATH` 후 실행하세요.

Homebrew R을 쓰면 실행 시 `Error importing in API mode ... Trying to import in ABI mode.` 메시지가 뜨지만 ABI 모드로 정상 동작하니 무시해도 됩니다. R이 없으면 재대결 분석 스크립트는 혼합효과 단계만 건너뜁니다.
