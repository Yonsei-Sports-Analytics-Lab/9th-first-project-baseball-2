"""Fit the pitch-avoidance mixed-effects logistic regression via R's
lme4::glmer (rpy2 bridge) and report every output requested:
1. fixed-effects table (estimate, SE, p, odds ratio)
2. convergence status and ICC (pitcher-level variance share)
3. full per-pitcher random-effects table, saved to CSV
4. top/bottom 10 pitchers by their random group-slope
5. a predicted-vs-observed calibration table

Requires: R (`brew install r`), the R package `lme4`
(`Rscript -e 'install.packages("lme4")'`), and `pip install rpy2`.

Requires data/raw (with fielder_2 -- re-collect via
src/collection/statcast_scraper.py if missing) and
data/processed/pitch_reuse_after_xbh_events.parquet (rebuild via
src/preprocessing/data_pipeline.py after re-collecting).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

R_CONVERTER = ro.default_converter + numpy2ri.converter + pandas2ri.converter

from src.analysis.avoidance_stats import build_stratified_placebo_candidates
from src.analysis.mixed_effects_model import (
    MODEL_FORMULA,
    build_combined_model_dataset,
    compute_calibration_table,
    compute_icc,
    rank_pitchers_by_group_slope,
)
from src.preprocessing.build_next_ab_dataset import build_event_dataset_for_events
from src.preprocessing.load_raw_pitches import load_all_pitches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
XBH_EVENTS_PATH = PROCESSED_DATA_DIR / "pitch_reuse_after_xbh_events.parquet"
RANDOM_EFFECTS_OUTPUT_PATH = PROCESSED_DATA_DIR / "pitch_avoidance_mixed_model_random_effects.csv"
PLACEBO_OUTCOME_EVENTS = {"field_out"}
PLACEBO_SEED = 20260923


def build_filtered_groups() -> tuple[pd.DataFrame, pd.DataFrame]:
    pitches = load_all_pitches()
    xbh = pd.read_parquet(XBH_EVENTS_PATH)
    placebo_candidates = build_stratified_placebo_candidates(pitches, xbh, PLACEBO_OUTCOME_EVENTS, seed=PLACEBO_SEED)
    placebo = build_event_dataset_for_events(pitches, placebo_candidates)

    xbh_f = xbh[xbh["has_next_ab"] & xbh["baseline_usage"].notna()].copy()
    placebo_f = placebo[placebo["has_next_ab"] & placebo["baseline_usage"].notna()].copy()
    return xbh_f, placebo_f


def fit_glmer(combined: pd.DataFrame):
    """Runs the fit as a plain R code string (rather than calling glmer()
    with Python kwargs) because passing an R glmerControl() object through
    rpy2's argument conversion strips its S3 class and breaks glmer's
    internal dispatch -- building the call in R avoids that entirely.
    """
    ro.r("library(lme4)")
    with R_CONVERTER.context():
        r_df = ro.conversion.get_conversion().py2rpy(combined)
        ro.globalenv["model_data"] = r_df
    ro.r(f"""
    model <- glmer(
        {MODEL_FORMULA},
        data = model_data,
        family = binomial,
        nAGQ = 0,
        control = glmerControl(optimizer = "bobyqa")
    )
    """)
    return ro.globalenv["model"]


def extract_fixed_effects_table(model) -> pd.DataFrame:
    base = importr("base")
    stats_r = importr("stats")
    coefs = stats_r.coef(base.summary(model))
    with R_CONVERTER.context():
        df = ro.conversion.get_conversion().rpy2py(base.as_data_frame(coefs))
    df = df.reset_index().rename(
        columns={
            "index": "term",
            "Estimate": "estimate",
            "Std. Error": "std_error",
            "z value": "z_value",
            "Pr(>|z|)": "p_value",
        }
    )
    df["odds_ratio"] = np.exp(df["estimate"])
    return df


def check_convergence() -> str:
    msg = ro.r("if (is.null(model@optinfo$conv$lme4$messages)) 'OK: no convergence warnings' "
               "else paste(model@optinfo$conv$lme4$messages, collapse='; ')")
    singular = bool(ro.r("isSingular(model)")[0])
    return f"{msg[0]}; isSingular={singular}" + (
        " (경계값 적합: 일부 분산 성분이 0에 매우 가까움 -- 아래 투수 절편 분산 참고)" if singular else ""
    )


def extract_pitcher_variance_components() -> tuple[float, float, float]:
    """Returns (intercept_variance, group_slope_variance, correlation)."""
    pitcher_vc = ro.r("as.data.frame(VarCorr(model))")
    with R_CONVERTER.context():
        vc_df = ro.conversion.get_conversion().rpy2py(pitcher_vc)
    intercept_var = vc_df.loc[(vc_df["grp"] == "pitcher") & (vc_df["var1"] == "(Intercept)") & vc_df["var2"].isna(), "vcov"].iloc[0]
    group_var = vc_df.loc[(vc_df["grp"] == "pitcher") & (vc_df["var1"] == "group") & vc_df["var2"].isna(), "vcov"].iloc[0]
    corr_row = vc_df.loc[(vc_df["grp"] == "pitcher") & (vc_df["var2"] == "group")]
    corr = float(corr_row["sdcor"].iloc[0]) if len(corr_row) else float("nan")
    return float(intercept_var), float(group_var), corr


def extract_random_effects_table() -> pd.DataFrame:
    pitcher_ranef = ro.r("as.data.frame(ranef(model)$pitcher)")
    with R_CONVERTER.context():
        df = ro.conversion.get_conversion().rpy2py(pitcher_ranef)
    df = df.reset_index().rename(columns={"index": "pitcher", "(Intercept)": "re_intercept", "group": "re_group"})
    return df


def extract_predictions() -> np.ndarray:
    preds = ro.r('predict(model, type="response")')
    with R_CONVERTER.context():
        arr = ro.conversion.get_conversion().rpy2py(preds)
    return np.asarray(arr)


def run() -> None:
    xbh_f, placebo_f = build_filtered_groups()
    combined_raw = pd.concat([xbh_f.assign(group=1), placebo_f.assign(group=0)], ignore_index=True)

    logger.info("=== 최종 통합 데이터셋 스키마 (모델링 전, 결측 제거 전) ===")
    logger.info("컬럼: %s", combined_raw.columns.tolist())
    logger.info(
        "행 수: XBH=%d, placebo=%d, 합계=%d",
        (combined_raw["group"] == 1).sum(), (combined_raw["group"] == 0).sum(), len(combined_raw),
    )
    key_cols = ["outs_when_up", "catcher_changed", "baseline_usage", "score_diff", "stand", "pitch_family"]
    logger.info("주요 컬럼 결측치 비율:\n%s", combined_raw[key_cols].isna().mean())
    logger.info(
        "catcher_changed=1 비율: %.5f (%d / %d) -- 정의상 같은 하프이닝 내 연속 타석 비교라 매우 희귀함",
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

    logger.info("glmer 적합 중 (nAGQ=0, bobyqa)... 수 분 걸릴 수 있음")
    model = fit_glmer(combined)
    logger.info("=== 수렴 상태 ===\n%s", check_convergence())

    fixed_effects = extract_fixed_effects_table(model)
    pd.set_option("display.max_rows", None, "display.width", 160)
    logger.info("=== 고정효과 ===\n%s", fixed_effects.to_string(index=False))

    intercept_var, group_var, corr = extract_pitcher_variance_components()
    icc = compute_icc(intercept_var)
    logger.info(
        "=== 랜덤효과 분산 성분 === 투수 절편 분산=%.4f, group 기울기 분산=%.4f, 상관=%.3f, ICC=%.4f",
        intercept_var, group_var, corr, icc,
    )

    random_effects = extract_random_effects_table()
    random_effects.to_csv(RANDOM_EFFECTS_OUTPUT_PATH, index=False)
    logger.info("투수별 랜덤효과 전체 저장: %s (%d명)", RANDOM_EFFECTS_OUTPUT_PATH, len(random_effects))

    top10, bottom10 = rank_pitchers_by_group_slope(random_effects, pitcher_meta, n=10)
    logger.info("=== group 랜덤 기울기 상위 10명 (회피 경향이 평균보다 더 강함) ===\n%s",
                 top10[["pitcher", "pitcher_name", "re_group", "n_xbh_events"]].to_string(index=False))
    logger.info("=== group 랜덤 기울기 하위 10명 (회피 경향이 평균보다 약하거나 반대) ===\n%s",
                 bottom10[["pitcher", "pitcher_name", "re_group", "n_xbh_events"]].to_string(index=False))

    predicted = extract_predictions()
    calibration = compute_calibration_table(combined["reused_same_type"], pd.Series(predicted), n_bins=10)
    logger.info("=== calibration (예측확률 10분위 vs 실제 관측 재사용률) ===\n%s", calibration.to_string(index=False))


if __name__ == "__main__":
    run()
