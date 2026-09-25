"""Robustness comparison for the pitch-avoidance mixed-effects model:

- ORIGINAL: catcher_changed included, random intercept & slope
  (1 + group | pitcher) -- came out isSingular=TRUE on the full raw sample
  (pitcher intercept variance at the zero boundary). On the research sample
  it is not flagged singular, but the group-slope variance is ~0 (0.0007), so
  its per-pitcher slopes are not interpretable. catcher_changed was
  practically unidentifiable (only 4 of 109,536 rows have catcher_changed=1
  in the research sample).
- ROBUSTNESS: catcher_changed dropped, random slope only
  (0 + group | pitcher).

Both are fit on the exact same 109,536-row combined dataset, and this
script reports: a side-by-side fixed-effects table, both models'
convergence/variance components, the Pearson/Spearman correlation between
the two models' per-pitcher random group-slopes, the robustness model's
top/bottom 10 pitchers (restricted to >=100 XBH events), and its
calibration table.
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
    MODEL_FORMULA_ORIGINAL,
    MODEL_FORMULA_ROBUSTNESS,
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
ORIGINAL_RANDOM_EFFECTS_PATH = PROCESSED_DATA_DIR / "pitch_avoidance_mixed_model_random_effects_original.csv"
ROBUSTNESS_RANDOM_EFFECTS_PATH = PROCESSED_DATA_DIR / "pitch_avoidance_mixed_model_random_effects_robustness.csv"
PLACEBO_OUTCOME_EVENTS = {"field_out"}
PLACEBO_SEED = 20260923
MIN_XBH_SAMPLE_FOR_RANKING = 100


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    pitches = load_research_pitches()
    xbh = pd.read_parquet(XBH_EVENTS_PATH)
    placebo_candidates = build_stratified_placebo_candidates(pitches, xbh, PLACEBO_OUTCOME_EVENTS, seed=PLACEBO_SEED)
    placebo = build_event_dataset_for_events(pitches, placebo_candidates)

    xbh_f = xbh[xbh["has_next_ab"] & xbh["baseline_usage"].notna()].copy()
    placebo_f = placebo[placebo["has_next_ab"] & placebo["baseline_usage"].notna()].copy()
    combined = build_combined_model_dataset(xbh_f, placebo_f)

    pitcher_meta = (
        combined[combined["group"] == 1]
        .groupby("pitcher")
        .agg(pitcher_name=("pitcher_name", "first"), n_xbh_events=("pitcher_name", "size"))
        .reset_index()
    )
    return combined, pitcher_meta


def fit_and_report(label: str, formula: str, combined: pd.DataFrame, random_effects_path: Path) -> dict:
    logger.info("=== [%s] glmer 적합 중 ===\n공식: %s", label, formula)
    fit_glmer(combined, formula)
    convergence = check_convergence()
    logger.info("[%s] 수렴 상태: %s", label, convergence)

    fixed_effects = extract_fixed_effects_table()
    intercept_var, group_var, corr = extract_pitcher_variance_components()
    icc = compute_icc(intercept_var) if intercept_var is not None else None
    logger.info(
        "[%s] 분산성분: 투수 절편 분산=%s, group 기울기 분산=%.4f, 상관=%s, ICC=%s",
        label,
        f"{intercept_var:.4f}" if intercept_var is not None else "N/A (랜덤 절편 없음)",
        group_var,
        f"{corr:.3f}" if corr == corr else "N/A",
        f"{icc:.4f}" if icc is not None else "N/A (랜덤 절편 없음)",
    )

    random_effects = extract_random_effects_table()
    random_effects.to_csv(random_effects_path, index=False)
    logger.info("[%s] 랜덤효과 저장: %s (%d명)", label, random_effects_path, len(random_effects))

    predicted = extract_predictions()
    calibration = compute_calibration_table(combined["reused_same_type"], pd.Series(predicted), n_bins=10)

    return {
        "label": label,
        "convergence": convergence,
        "fixed_effects": fixed_effects,
        "intercept_var": intercept_var,
        "group_var": group_var,
        "corr": corr,
        "icc": icc,
        "random_effects": random_effects,
        "calibration": calibration,
    }


def run() -> None:
    combined, pitcher_meta = build_dataset()
    logger.info("통합 표본: %d행, 투수 %d명 (두 모델 모두 이 데이터로 적합)", len(combined), combined["pitcher"].nunique())

    original = fit_and_report("원본: catcher_changed + (1+group|pitcher)", MODEL_FORMULA_ORIGINAL, combined, ORIGINAL_RANDOM_EFFECTS_PATH)
    robust = fit_and_report("robustness: catcher_changed 제외 + (0+group|pitcher)", MODEL_FORMULA_ROBUSTNESS, combined, ROBUSTNESS_RANDOM_EFFECTS_PATH)

    pd.set_option("display.max_rows", None, "display.width", 220)

    # 1. 고정효과 비교표
    merged_fe = original["fixed_effects"].merge(
        robust["fixed_effects"], on="term", how="outer", suffixes=("_원본", "_robust")
    )
    logger.info(
        "=== 고정효과 비교 (원본 vs robustness) ===\n%s",
        merged_fe[
            ["term", "estimate_원본", "p_value_원본", "odds_ratio_원본", "estimate_robust", "p_value_robust", "odds_ratio_robust"]
        ].to_string(index=False),
    )
    for term in ["group", "group:baseline_usage"]:
        row = merged_fe[merged_fe["term"] == term]
        if len(row):
            r = row.iloc[0]
            logger.info(
                "핵심 계수 [%s]: 원본 OR=%.3f (p=%.2e)  vs  robust OR=%.3f (p=%.2e)",
                term, r["odds_ratio_원본"], r["p_value_원본"], r["odds_ratio_robust"], r["p_value_robust"],
            )

    # 4. 랜덤효과(re_group) 원본 vs robustness 상관
    re_compare = original["random_effects"][["pitcher", "re_group"]].merge(
        robust["random_effects"][["pitcher", "re_group"]], on="pitcher", suffixes=("_원본", "_robust")
    )
    pearson_corr = re_compare["re_group_원본"].corr(re_compare["re_group_robust"], method="pearson")
    spearman_corr = re_compare["re_group_원본"].corr(re_compare["re_group_robust"], method="spearman")
    logger.info(
        "=== 투수별 re_group: 원본 vs robustness 상관 (n=%d명) === Pearson=%.4f, Spearman(순위)=%.4f",
        len(re_compare), pearson_corr, spearman_corr,
    )

    # 5. robustness 모델 상/하위 10명 (XBH 표본 >= 100건만)
    eligible_meta = pitcher_meta[pitcher_meta["n_xbh_events"] >= MIN_XBH_SAMPLE_FOR_RANKING]
    eligible_re = robust["random_effects"][robust["random_effects"]["pitcher"].isin(eligible_meta["pitcher"])]
    logger.info("표본 %d건 이상 투수 수: %d명 (전체 %d명 중)", MIN_XBH_SAMPLE_FOR_RANKING, len(eligible_meta), len(pitcher_meta))
    top10, bottom10 = rank_pitchers_by_group_slope(eligible_re, eligible_meta, n=10)
    logger.info(
        "=== [robustness] group 기울기 상위 10명 (표본>=%d, 회피 경향 약함) ===\n%s",
        MIN_XBH_SAMPLE_FOR_RANKING, top10[["pitcher", "pitcher_name", "re_group", "n_xbh_events"]].to_string(index=False),
    )
    logger.info(
        "=== [robustness] group 기울기 하위 10명 (표본>=%d, 회피 경향 강함) ===\n%s",
        MIN_XBH_SAMPLE_FOR_RANKING, bottom10[["pitcher", "pitcher_name", "re_group", "n_xbh_events"]].to_string(index=False),
    )

    # 6. robustness 모델 calibration
    logger.info("=== [robustness] calibration ===\n%s", robust["calibration"].to_string(index=False))


if __name__ == "__main__":
    run()
