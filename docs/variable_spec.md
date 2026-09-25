# 변수 테이블 스펙 (과제1·2 공용)

2026-09-25 기준, 모든 수치는 현재 `data/processed/` 파일에서 직접 센 값입니다. 파일은 `.gitignore` 대상이라 각자 아래 "생성 코드"로 다시 만들어야 합니다.

**표본:** 팀 공유 research CSV(투수-시즌 투구 수 ≥ 500, 2026은 ≥ 300, 2026-07-09까지)에 있는 투구만 씁니다. 자세한 정의는 [data/research/README.md](../data/research/README.md)를 보세요. 2026 전반기 표본은 팀에서 확인 중이라, 확정되면 아래 수치가 바뀔 수 있습니다.

## 1. 데이터 파일

| 파일 (`data/processed/`) | 행 수 | 1행의 의미 | 생성 코드 |
|---|---|---|---|
| `pitch_reuse_after_xbh_events.parquet` (`.csv`) | 70,068 | 장타 1건 (비교 타석 = 다음 타자) | `src/preprocessing/data_pipeline.py` |
| `pitch_reuse_after_xbh_same_batter_rematch_events.parquet` | 27,294 | 장타 중 같은 타자·같은 투수 재대결이 있는 1건 | `run_same_batter_rematch_analysis.py` |
| `pitch_reuse_placebo_same_batter_rematch_events.parquet` | 27,274 | 대조군 `field_out` 재대결 1건 (시즌×구종계열 층화) | 〃 |
| `pitch_reuse_placebo_same_batter_rematch_pitcher_matched_events.parquet` | 27,100 | 대조군 재대결 1건 (같은 투수×시즌×구종계열) | 〃 |
| `intensive_margin_pitch_observations.parquet` | 361,590 | 투구 1개 (pre/post 관측) | `run_intensive_margin_analysis.py` |

## 2. 이벤트 테이블 컬럼

장타·대조군이 같은 스키마입니다(29컬럼, 다음 타자 기준 파일은 `season_usage_rate`가 없는 28컬럼). 대조군에서도 `hit_pitch_type`은 "그 이벤트를 만든 구종"이라는 뜻으로 이름을 그대로 씁니다. 결측률은 장타 재대결 / 대조군 재대결입니다.

| 컬럼 | 정의 | 결측 처리 |
|---|---|---|
| `game_pk`, `game_date`, `season` | 경기 ID, 경기일, 시즌(경기일의 연도) | 없음 |
| `pitcher`, `pitcher_name`, `batter` | 투수 ID, 투수 이름, 타자 ID | 없음 |
| `stand`, `p_throws` | 타자 타석(L/R), 투수 투구팔(L/R) | 없음 |
| `platoon_match` | 동타=1(`stand == p_throws`), 이타=0 | 둘 중 하나가 없으면 NaN (실제 0건) |
| `at_bat_number` | 이벤트 타석의 경기 내 순번 | 없음 |
| `event_pitch_number` | 이벤트를 만든 투구(그 타석의 마지막 투구)의 `pitch_number` | 없음 |
| `next_at_bat_number` | 비교 타석의 순번. 다음 타자 모드는 `at_bat_number + 1`, 재대결 모드는 그 타자의 같은 경기 바로 다음 타석 | `has_next_ab=False`이면 NaN |
| `events` | `double`/`triple`/`home_run`(장타) 또는 `field_out`(대조군) | 없음 |
| `hit_pitch_type` | 이벤트 투구의 구종 코드 | 결측 0.07% / 0% (구종 없는 투구) |
| `pitch_family` | fastball / breaking / offspeed / other (FC는 fastball) | `hit_pitch_type`이 없으면 NaN |
| `is_cutter` | 구종이 FC이면 True | 없음 |
| `balls`, `strikes`, `outs_when_up`, `inning`, `inning_topbot` | 이벤트 투구 시점의 카운트·아웃·이닝·초말 | 없음 |
| `score_diff` | 투수팀 관점 점수차(수비팀 − 공격팀), 이벤트 투구 시점(결과 반영 전) | 없음 |
| `baseline_usage` | 그 경기에서 이벤트 투구 **이전**까지 같은 투수의 `hit_pitch_type` 비중 | 이전 투구가 없거나 구종이 없으면 NaN (1.99% / 1.47%) |
| `has_next_ab` | 비교 타석이 있고 같은 투수가 던졌는가 | 없음 (재대결 파일은 전부 True) |
| `next_ab_pitch_count` | 비교 타석에서 그 투수가 던진 투구 수 (구종 없는 투구 포함) | 비교 타석이 없으면 0 |
| `reused_same_type` | 비교 타석에 `hit_pitch_type`이 한 번이라도 있으면 1, 없으면 0 | 비교 타석이 없거나 구종이 없으면 NaN (0.07% / 0%) |
| `same_type_share` | 비교 타석 투구 중 `hit_pitch_type` 비중 (분모 = `next_ab_pitch_count`) | `reused_same_type`과 동일 |
| `catcher_changed` | 이벤트 투구와 비교 타석 첫 투구의 포수(`fielder_2`)가 다르면 1 | 비교 타석이 없으면 NaN |
| `season_usage_rate` | 그 투수의 그 시즌 `hit_pitch_type` 구사율 (구종이 있는 투구만 분모) | 구종이 없으면 NaN (0.07% / 0%). 재대결 파일에만 있음 |

* `catcher_changed`는 다음 타자 기준에서는 모델 표본 109,198건 중 4건만 1이라 모델에서 뺐습니다. 재대결에서는 장타 122건(0.4%), 대조군 98건(0.4%)로 늘어나지만 여전히 드뭅니다. 이 변수는 아직 어떤 모델에도 넣지 않았습니다.
* 구종이 없는 이벤트 투구(장타 37건)는 재사용 관련 값이 모두 NaN입니다. 0으로 채우지 않습니다.

## 3. 분석에서 만드는 파생 값

* `group`: 장타 = 1, 대조군 = 0
* 사용 표본: 경기 내 baseline 분석은 `has_next_ab & baseline_usage.notna()`, 시즌 baseline 분석은 `has_next_ab & season_usage_rate.notna()`
* 감소폭 `diff = baseline − same_type_share` (양수 = 사용 비중 감소). 순수 효과는 장타의 `diff` 평균 − 대조군의 `diff` 평균
* `expected_reuse_prob = 1 − (1 − baseline)^next_ab_pitch_count` (사전 성향을 유지했다면 기대되는 재사용 확률)
* 혼합효과 로지스틱: `REQUIRED_MODEL_COLUMNS`에 결측이 있는 행은 제외, platoon 모델은 `platoon_match`도 필수

## 4. 투구 관측 테이블 (Intensive margin)

**포함되는 이벤트:** `has_next_ab`이고 `reused_same_type == 1`인 이벤트(장타 12,421건, 대조군 18,502건). 재사용하지 않은 이벤트는 비교할 투구가 없어 제외합니다.

* **pre (`time=0`):** 같은 경기에서 그 투수가 던진 같은 구종 투구 중 이벤트 투구 **이전**의 것 (이벤트 투구 자체는 제외). 이벤트당 중앙값 9개
* **post (`time=1`):** 비교 타석에서 그 투수가 던진 같은 구종 투구. 이벤트당 중앙값 1개

| 컬럼 | 정의 | 결측 처리 |
|---|---|---|
| `event_id` | `{group}_{game_pk}_{이벤트 at_bat_number}` | 없음 |
| `group`, `time` | 장타=1/대조군=0, pre=0/post=1 | 없음 |
| `pitcher`(문자열), `season`, `pitch_type`, `stand`, `hit_pitch_type` | 그 투구의 식별 정보 | 없음 |
| `at_bat_number`, `pitch_number` | 그 투구의 타석·투구 순번 | 없음 |
| `balls`, `strikes` | **그 투구 시점**의 카운트 | 없음 |
| `score_diff` | **그 투구 시점**의 투수팀 관점 점수차 (장타 후 점수가 바뀌므로 pre/post가 다를 수 있음) | 없음 |
| `plate_x`, `plate_z` | 홈플레이트 통과 위치 (ft) | 0.01% 미만 |
| `pfx_x`, `pfx_z` | 무브먼트 (ft) | 0.01% 미만 |
| `release_speed` | 구속 (mph) | 0.01% 미만 |
| `baseline_usage`, `platoon_match` | 이벤트의 값을 복사 | `baseline_usage` 0.31% |
| `base_plate_x/z`, `base_pfx_x/z`, `base_release_speed` | 그 투수·시즌·구종의 평소 값(평균) | 표본 부족 시 NaN (위치 0.09% / 무브먼트·구속 0.01%) |
| `location_dev` | `sqrt((plate_x − base_plate_x)² + (plate_z − base_plate_z)²)`, ft | baseline이 없거나 위치가 없으면 NaN (0.09%) |
| `movement_dev` | `sqrt((pfx_x − base_pfx_x)² + (pfx_z − base_pfx_z)²)`, ft | NaN (0.01%) |
| `velocity_dev` | `release_speed − base_release_speed`, mph (부호 있음) | NaN (0.01%) |

* **baseline 단위:** 투수 × 시즌 × 구종. 위치만 **타자 타석(`stand`)별**로 나눕니다(타깃 방향이 좌우타에 따라 뒤집히므로). 셀에 유효 투구가 20개 미만이면 baseline을 만들지 않아 편차가 NaN입니다.
* baseline에는 비교 대상 투구도 포함됩니다(투수·시즌당 수백 개 중 몇 개라 영향은 작음).
* 모델 입력에서는 해당 outcome이나 공변량이 결측인 행을 제외합니다.
* 편차 분포: `location_dev` 평균 0.94 ft(최대 6.7), `movement_dev` 평균 0.22 ft(최대 2.9), `velocity_dev` 평균 0.09 mph(−26 ~ +9). 이상치는 그대로 두었습니다.
