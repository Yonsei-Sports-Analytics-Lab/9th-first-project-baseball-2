# 📂 Raw Data (원본 데이터)

이 폴더는 분석에 사용될 **가공되지 않은 원본 데이터(Raw Data)**를 보관하는 곳입니다. 
데이터의 무결성을 보장하기 위해, **이 폴더에 위치한 파일은 어떠한 경우에도 직접 수정해서는 안 됩니다.** 데이터 정제 및 변환 결과물은 반드시 `data/processed/` 폴더에 저장해 주세요.

## 🔒 고정 데이터셋 (분석 기준)

**모든 분석은 아래 6개 파일만 사용합니다.** 파일을 새로 받거나 교체하지 마세요. Statcast는 사후에 구종 분류 등이 수정되므로, 다시 수집하면 결과가 달라질 수 있습니다.

- 출처: MLB Statcast (Baseball Savant), 투구 단위(pitch-level)
- 경기 유형: 정규시즌(`game_type == 'R'`)만
- 투수 필터: 해당 시즌 투구 수 **500구 이상** 투수 (2026은 시즌 진행 중이라 **300구 이상**)
- 공통 컬럼 58개 (첫 줄 헤더, UTF-8 BOM 포함 → `pd.read_csv(..., encoding='utf-8-sig')`)
- 2026-10-01판부터 점수 컬럼 10개(`home_score`, `away_score`, `bat_score`, `fld_score`, `post_*_score`, `score_diff`, `post_score_diff`)가 추가됐습니다. 행·투수 수는 이전판과 같습니다.
- **없는 컬럼:** run value(`delta_run_exp`), 포수(`fielder_2`)
- 2026 파일은 이름이 `min500`이지만 실제로는 **300구 컷오프**입니다(이전판 `min300`과 행 수 동일).

| 파일 | 기간 | 투구 수(행) | 투수 수 | SHA-256 |
|---|---|---:|---:|---|
| `statcast_2021_min500_research_score.csv` | 2021-04-01 ~ 2021-10-03 | 634,829 | 486 | `e7ba972ea6497792f01830df54ae516c13a5f952c1020587430cc1099d9c4834` |
| `statcast_2022_min500_research_score.csv` | 2022-04-07 ~ 2022-10-05 | 637,264 | 473 | `e358705da0960621d8977cbce87e0cf1850e7a48184afce290bab6c65632eeac` |
| `statcast_2023_min500_research_score.csv` | 2023-03-30 ~ 2023-10-01 | 648,341 | 479 | `2fa1580be1aecd0d810eb4a839949f59fc85991f9fb2dbf0fc40f1e132685d54` |
| `statcast_2024_min500_research_score.csv` | 2024-03-28 ~ 2024-09-30 | 641,086 | 474 | `5cecc21f8c843a3a1343d44cedd849a8d687d78b1f89ab4aeacbfa8bbe7f3414` |
| `statcast_2025_min500_research_score.csv` | 2025-03-27 ~ 2025-09-28 | 639,744 | 479 | `cc58992d18e1f5d4dbc445f2705410305fb00a18e7e7107dd1e5a1e9bbbc21d8` |
| `statcast_2026_min500_research_score.csv` | 2026-03-26 ~ 2026-07-12 | 391,038 | 452 | `f73963abe897d06a7a89094cc1b7bf97dfc0dfb9eb2ae61324c0c3b1e3992ba3` |

### 받은 파일 검증

```bash
cd data/raw && shasum -a 256 statcast_*_research_score.csv
```

출력된 해시가 위 표와 모두 일치해야 합니다.

### `data/research/` 링크

새 분석 코드(`src/`, `notebooks/`)는 `data/research/statcast_<연도>_min<컷오프>_research.csv`라는 이름으로 CSV를 찾습니다. 파일을 복사하지 말고 이 폴더의 파일에 심볼릭 링크를 거세요(저장소 루트에서 실행).

```bash
mkdir -p data/research
for y in 2021 2022 2023 2024 2025; do ln -sf ../raw/statcast_${y}_min500_research_score.csv data/research/statcast_${y}_min500_research.csv; done
ln -sf ../raw/statcast_2026_min500_research_score.csv data/research/statcast_2026_min300_research.csv
```

## 📥 데이터 수집 및 세팅 방법

전체 데이터셋은 용량이 크므로(파일당 약 90~145MB) GitHub에 업로드하지 않습니다. 팀원들은 로컬 환경에서 작업을 시작하기 전, 구글 드라이브에서 위 6개 파일을 받아 이 폴더(`data/raw/`)에 그대로 넣어 주세요.

### 📥 데이터 다운로드 링크
* **2021–2026 Statcast 투구 데이터 (고정본):** [구글 드라이브 링크(클릭)](#)
