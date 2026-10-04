"""Fit the pitch-avoidance mixed-effects logistic regression (current
recommended spec: MODEL_FORMULA_ROBUSTNESS, no catcher_changed, random
slope only) via R's lme4::glmer (rpy2 bridge) and report:
1. fixed-effects table (estimate, SE, p, odds ratio)
2. convergence status and ICC (pitcher-level variance share)
3. full per-pitcher random-effects table, saved to CSV
4. top/bottom 10 pitchers by their random group-slope
5. a predicted-vs-observed calibration table

See run_mixed_effects_model_comparison.py for a side-by-side robustness
check against the original (catcher_changed + random intercept & slope)
specification.

Requires: R (`brew install r`), the R package `lme4`
(`Rscript -e 'install.packages("lme4")'`), and `pip install rpy2`.

Requires data/raw (with fielder_2 -- re-collect via
src/collection/statcast_scraper.py if missing), the team's research CSVs in
data/research/ (see src/preprocessing/research_sample.py) and
data/processed/pitch_reuse_after_xbh_events.parquet (rebuild via
src/preprocessing/data_pipeline.py).
"""

import logging
from pathlib import Path

import pandas as pd

from src.analysis.avoidance_stats import build_stratified_placebo_candidates
from src.analysis.glmer_runner import (
    check_convergence,
    extract_fixed_effects_table,
    extract_pitcher_variance_components,
    extract_predictions,
    extract_random_effects_table,
    fit_glmer,
)
from src.analysis.mixed_effects_model import (
    MODEL_FORMULA,
    build_combined_model_dataset,
    compute_calibration_table,
    compute_icc,
    rank_pitchers_by_group_slope,
)
from src.preprocessing.build_next_ab_dataset import build_event_dataset_for_events
from src.preprocessing.research_sample import load_research_pitches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
XBH_EVENTS_PATH = PROCESSED_DATA_DIR / "pitch_reuse_after_xbh_events.parquet"
RANDOM_EFFECTS_OUTPUT_PATH = PROCESSED_DATA_DIR / "pitch_avoidance_mixed_model_random_effects.csv"
PLACEBO_OUTCOME_EVENTS = {"field_out"}
PLACEBO_SEED = 20260923


def build_filtered_groups() -> tuple[pd.DataFrame, pd.DataFrame]:
    pitches = load_research_pitches()
    xbh = pd.read_parquet(XBH_EVENTS_PATH)
    placebo_candidates = build_stratified_placebo_candidates(pitches, xbh, PLACEBO_OUTCOME_EVENTS, seed=PLACEBO_SEED)
    placebo = build_event_dataset_for_events(pitches, placebo_candidates)

    xbh_f = xbh[xbh["has_next_ab"] & xbh["baseline_usage"].notna()].copy()
    placebo_f = placebo[placebo["has_next_ab"] & placebo["baseline_usage"].notna()].copy()
    return xbh_f, placebo_f


def run() -> None:
    xbh_f, placebo_f = build_filtered_groups()
    combined_raw = pd.concat([xbh_f.assign(group=1), placebo_f.assign(group=0)], ignore_index=True)

    logger.info("=== 최종 통합 데이터셋 스키마 (모델링 전, 결측 제거 전) ===")
    logger.info("컬럼: %s", combined_raw.columns.tolist())
    logger.info(
        "행 수: XBH=%d, placebo=%d, 합계=%d",
        (combined_raw["group"] == 1).sum(), (combined_raw["group"] == 0).sum(), len(combined_raw),
    )
    key_cols = ["outs_when_up", "baseline_usage", "score_diff", "stand", "pitch_family"]
    logger.info("주요 컬럼 결측치 비율:\n%s", combined_raw[key_cols].isna().mean())
    logger.info(
        "참고: catcher_changed=1 비율 %.5f (%d / %d) -- 정의상 같은 하프이닝 내 연속 타석 비교라 매우 희귀해 "
        "모델 공식에서 제외함 (사용자 확인)",
        combined_raw["catcher_changed"].mean(), combined_raw["catcher_changed"].sum(), combined_raw["catcher_changed"].notna().sum(),
    )

    combined = build_combined_model_dataset(xbh_f, placebo_f)
    n_pitchers = combined["pitcher"].nunique()
    logger.info("모델링용 최종 표본: %d행 (결측 제거로 %d행 손실), 투수 %d명", len(combined), len(combined_raw) - len(combined), n_pitchers)

    pitcher_meta = (
        combined[combined["group"] == 1]
        .groupby("pitcher")
        .agg(pitcher_name=("pitcher_name", "first"), n_xbh_events=("pitcher_name", "size"))
        .reset_index()
    )

    logger.info("glmer 적합 중 (nAGQ=0, bobyqa)... 공식: %s", MODEL_FORMULA)
    fit_glmer(combined, MODEL_FORMULA)
    logger.info("=== 수렴 상태 ===\n%s", check_convergence())

    fixed_effects = extract_fixed_effects_table()
    pd.set_option("display.max_rows", None, "display.width", 160)
    logger.info("=== 고정효과 ===\n%s", fixed_effects.to_string(index=False))

    intercept_var, group_var, corr = extract_pitcher_variance_components()
    icc = compute_icc(intercept_var) if intercept_var is not None else None
    logger.info(
        "=== 랜덤효과 분산 성분 === 투수 절편 분산=%s, group 기울기 분산=%.4f, 상관=%s, ICC=%s",
        f"{intercept_var:.4f}" if intercept_var is not None else "N/A (랜덤 절편 없음)",
        group_var,
        f"{corr:.3f}" if corr == corr else "N/A",
        f"{icc:.4f}" if icc is not None else "N/A (랜덤 절편 없음)",
    )

    random_effects = extract_random_effects_table()
    random_effects.to_csv(RANDOM_EFFECTS_OUTPUT_PATH, index=False)
    logger.info("투수별 랜덤효과 전체 저장: %s (%d명)", RANDOM_EFFECTS_OUTPUT_PATH, len(random_effects))

    top10, bottom10 = rank_pitchers_by_group_slope(random_effects, pitcher_meta, n=10)
    # 전체 group 고정효과가 음수(-)이므로: 개인 총효과 = 고정효과 + re_group.
    # re_group이 클수록(양수에 가까울수록) 총효과가 0에 가까워져 "회피가 약함",
    # re_group이 작을수록(음수로 클수록) 총효과가 더 음수라 "회피가 강함".
    logger.info("=== group 랜덤 기울기 상위 10명 (re_group 높음 = 회피 경향이 평균보다 약함/없음) ===\n%s",
                 top10[["pitcher", "pitcher_name", "re_group", "n_xbh_events"]].to_string(index=False))
    logger.info("=== group 랜덤 기울기 하위 10명 (re_group 낮음 = 회피 경향이 평균보다 강함) ===\n%s",
                 bottom10[["pitcher", "pitcher_name", "re_group", "n_xbh_events"]].to_string(index=False))

    predicted = extract_predictions()
    calibration = compute_calibration_table(combined["reused_same_type"], pd.Series(predicted), n_bins=10)
    logger.info("=== calibration (예측확률 10분위 vs 실제 관측 재사용률) ===\n%s", calibration.to_string(index=False))


if __name__ == "__main__":
    run()
