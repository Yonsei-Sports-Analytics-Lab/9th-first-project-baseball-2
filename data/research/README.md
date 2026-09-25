# 📂 Research Sample (팀 공유 분석 표본 CSV)

팀 전체가 같은 표본으로 분석하기 위한 CSV를 두는 폴더입니다. CSV 파일은 용량이 커서(약 850MB) `.gitignore` 대상이며, 공유 드라이브(`drive-download-…zip`)에서 받아 이 폴더에 넣습니다.

## 파일

| 파일 | 투구 수 | 투수 | 기간 |
|---|---|---|---|
| `statcast_2021_min500_research.csv` | 634,829 | 486 | 2021-04-01 ~ 10-03 |
| `statcast_2022_min500_research.csv` | 637,264 | 473 | 2022-04-07 ~ 10-05 |
| `statcast_2023_min500_research.csv` | 648,341 | 479 | 2023-03-30 ~ 10-01 |
| `statcast_2024_min500_research.csv` | 641,086 | 474 | 2024-03-28 ~ 09-30 |
| `statcast_2025_min500_research.csv` | 639,744 | 479 | 2025-03-27 ~ 09-28 |
| `statcast_2026_min300_research.csv` | 391,038 | 452 | 2026-03-26 ~ 07-12 |

* 정규시즌(`game_type == R`)만, 투수-시즌 단위로 **시즌 투구 수 ≥ 500**(2026은 시즌 일부만 진행되어 **≥ 300**)인 투수의 모든 투구입니다.
* **2026 파일은 수정본(2026-09-25 수령)입니다.** 이전 버전(378,007행, 07-09까지)에 비해 13,031행이 늘었고(기존 행은 모두 그대로), 기간이 07-12까지로 늘었습니다. 새로 생긴 7월 행 외에 3~6월 행도 1,228개 추가되어 있습니다. 공유 드라이브에서 받은 파일명은 `statcast_2026_min300_research (1).csv`인데, 코드가 `statcast_*_research.csv`로 찾으므로 **`(1)`을 떼고 `statcast_2026_min300_research.csv`로 저장**해야 합니다.
* 해외 개막전(2024 서울, 2025 도쿄 시리즈)과 2026-07-13 이후 경기는 CSV에 없습니다.

## 코드에서 쓰는 방식

CSV에는 점수(`home_score` 등)와 `fielder_2`가 없어서 CSV만으로는 분석 변수를 만들 수 없습니다. 그래서 `data/raw`(전체 수집본)를 읽고 **CSV에 있는 행(`game_pk`, `at_bat_number`, `pitch_number`, `pitcher` 키 일치)만 남깁니다** (`src/preprocessing/research_sample.py`의 `load_research_pitches()`).

* 겹치는 열은 CSV와 raw가 같습니다. 예외로 `pitch_type` 12행만 다른데(Statcast 사후 재분류), 팀 표본과 똑같이 맞추려고 CSV 값을 씁니다.
* CSV의 행이 `data/raw`에 없으면 에러를 냅니다 (raw를 CSV 마지막 날짜까지 수집했는지 확인하세요).
* 폴더 위치는 기본 `data/research/`이고, `RESEARCH_SAMPLE_DIR` 환경변수로 바꿀 수 있습니다.
* 키 인덱스는 `data/processed/research_sample_index.parquet`에 캐시되며 CSV가 바뀌면 자동으로 다시 만듭니다.
