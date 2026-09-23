"""Run all three tiers of the pitch-avoidance analysis end to end and log
a consolidated summary:

1. Paired test against the within-game baseline_usage column (noisiest,
   most optimistic estimate).
2. Paired test against each pitcher's season-level usage rate (less noisy
   baseline, controls for within-game regression to the mean).
3. Placebo/diff-in-diff test against a stratified sample of field_out
   events, built through the identical next-at-bat logic -- nets out any
   generic post-pitch drift that isn't specific to allowing an XBH.

Requires data/raw and data/processed/pitch_reuse_after_xbh_events.parquet
to already exist (see src/collection/statcast_scraper.py and
src/preprocessing/data_pipeline.py).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from src.analysis.avoidance_stats import (
    build_stratified_placebo_candidates,
    compute_expected_reuse_prob,
    compute_season_usage_rate,
    summarize_diff_in_diff,
    summarize_paired_diff,
)
from src.preprocessing.build_next_ab_dataset import build_event_dataset_for_events
from src.preprocessing.load_raw_pitches import load_all_pitches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
XBH_EVENTS_PATH = PROCESSED_DATA_DIR / "pitch_reuse_after_xbh_events.parquet"
PLACEBO_OUTCOME_EVENTS = {"field_out"}
PLACEBO_SEED = 20260923


def run_ingame_baseline_test(xbh: pd.DataFrame) -> pd.DataFrame:
    d = xbh[xbh["has_next_ab"] & xbh["baseline_usage"].notna()].copy()
    diff = d["baseline_usage"] - d["same_type_share"]
    result = summarize_paired_diff(diff)
    logger.info(
        "[1/3] 경기 내 사전 기준선(baseline_usage) 대비: n=%d, 차이=%+.4f (95%% CI [%+.4f, %+.4f]), p=%.3e",
        result["n"], result["mean"], result["ci_low"], result["ci_high"], result["t_p"],
    )
    return d


def run_season_baseline_test(pitches: pd.DataFrame, xbh: pd.DataFrame) -> None:
    season_usage = compute_season_usage_rate(pitches)
    d = xbh.merge(
        season_usage[["pitcher", "season", "pitch_type", "season_usage_rate"]],
        left_on=["pitcher", "season", "hit_pitch_type"],
        right_on=["pitcher", "season", "pitch_type"],
        how="left",
    )
    d = d[d["has_next_ab"] & d["season_usage_rate"].notna()].copy()
    diff = d["season_usage_rate"] - d["same_type_share"]
    result = summarize_paired_diff(diff)
    logger.info(
        "[2/3] 시즌 전체 평균 구사율 대비: n=%d, 차이=%+.4f (95%% CI [%+.4f, %+.4f]), p=%.3e",
        result["n"], result["mean"], result["ci_low"], result["ci_high"], result["t_p"],
    )


def run_placebo_diff_in_diff(pitches: pd.DataFrame, xbh: pd.DataFrame) -> dict:
    placebo_candidates = build_stratified_placebo_candidates(pitches, xbh, PLACEBO_OUTCOME_EVENTS, seed=PLACEBO_SEED)
    logger.info("[3/3] placebo(%s) 층화추출: %d건", PLACEBO_OUTCOME_EVENTS, len(placebo_candidates))
    placebo = build_event_dataset_for_events(pitches, placebo_candidates)

    xbh_f = xbh[xbh["has_next_ab"] & xbh["baseline_usage"].notna()].copy()
    placebo_f = placebo[placebo["has_next_ab"] & placebo["baseline_usage"].notna()].copy()
    logger.info("필터 후 표본: XBH=%d, placebo=%d", len(xbh_f), len(placebo_f))

    xbh_f["diff"] = xbh_f["baseline_usage"] - xbh_f["same_type_share"]
    placebo_f["diff"] = placebo_f["baseline_usage"] - placebo_f["same_type_share"]
    overall = summarize_diff_in_diff(xbh_f["diff"], placebo_f["diff"])
    logger.info(
        "순수 효과(placebo 통제): XBH diff=%+.4f, placebo diff=%+.4f, net=%+.4f (95%% CI [%+.4f, %+.4f]), p=%.3e",
        overall["treatment_mean"], overall["control_mean"], overall["net_effect"],
        overall["ci_low"], overall["ci_high"], overall["t_p"],
    )

    for fam in ["fastball", "breaking", "offspeed"]:
        gx = xbh_f[xbh_f["pitch_family"] == fam]["diff"]
        gp = placebo_f[placebo_f["pitch_family"] == fam]["diff"]
        r = summarize_diff_in_diff(gx, gp)
        logger.info("  pitch_family=%s: net=%+.4f (95%% CI [%+.4f, %+.4f]), p=%.3e", fam, r["net_effect"], r["ci_low"], r["ci_high"], r["t_p"])

    for season in sorted(xbh_f["season"].unique()):
        gx = xbh_f[xbh_f["season"] == season]["diff"]
        gp = placebo_f[placebo_f["season"] == season]["diff"]
        r = summarize_diff_in_diff(gx, gp)
        logger.info("  season=%s: net=%+.4f (95%% CI [%+.4f, %+.4f])", season, r["net_effect"], r["ci_low"], r["ci_high"])

    gx_robust = xbh_f[xbh_f["baseline_usage"] > 0]["diff"]
    gp_robust = placebo_f[placebo_f["baseline_usage"] > 0]["diff"]
    robust = summarize_diff_in_diff(gx_robust, gp_robust)
    logger.info("  robustness(baseline_usage>0): net=%+.4f (95%% CI [%+.4f, %+.4f]), p=%.3e", robust["net_effect"], robust["ci_low"], robust["ci_high"], robust["t_p"])

    for d in (xbh_f, placebo_f):
        d["expected_reuse_prob"] = compute_expected_reuse_prob(d["baseline_usage"], d["next_ab_pitch_count"])
        d["reuse_gap"] = d["reused_same_type"] - d["expected_reuse_prob"]
    gap_result = summarize_diff_in_diff(xbh_f["reuse_gap"], placebo_f["reuse_gap"], alternative="less")
    logger.info(
        "이진 지표(reuse_gap, 기대 대비 실측): XBH=%+.4f, placebo=%+.4f, net=%+.4f, p=%.3e",
        gap_result["treatment_mean"], gap_result["control_mean"], gap_result["net_effect"], gap_result["t_p"],
    )

    contingency = pd.DataFrame({
        "reused=1": [xbh_f["reused_same_type"].sum(), placebo_f["reused_same_type"].sum()],
        "reused=0": [(xbh_f["reused_same_type"] == 0).sum(), (placebo_f["reused_same_type"] == 0).sum()],
    }, index=["XBH", "placebo"])
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    logger.info("카이제곱(raw 재사용률): XBH=%.4f placebo=%.4f, chi2=%.2f p=%.3e", xbh_f["reused_same_type"].mean(), placebo_f["reused_same_type"].mean(), chi2, chi_p)

    combined = pd.concat([
        xbh_f.assign(group=1)[["reused_same_type", "baseline_usage", "group"]],
        placebo_f.assign(group=0)[["reused_same_type", "baseline_usage", "group"]],
    ])
    logit = smf.logit("reused_same_type ~ group + baseline_usage", data=combined).fit(disp=False)
    logger.info(
        "로지스틱 회귀(baseline_usage 통제): group 계수=%+.4f p=%.3e odds ratio=%.4f",
        logit.params["group"], logit.pvalues["group"], np.exp(logit.params["group"]),
    )

    return overall


def run() -> None:
    if not XBH_EVENTS_PATH.exists():
        raise FileNotFoundError(f"{XBH_EVENTS_PATH} 이 없습니다. 먼저 src/preprocessing/data_pipeline.py를 실행하세요.")

    pitches = load_all_pitches()
    xbh = pd.read_parquet(XBH_EVENTS_PATH)
    logger.info("원본 투구 %d건, XBH 이벤트 %d건 로드 완료", len(pitches), len(xbh))

    ingame_filtered = run_ingame_baseline_test(xbh)
    run_season_baseline_test(pitches, xbh)
    placebo_overall = run_placebo_diff_in_diff(pitches, xbh)

    naive = summarize_paired_diff(ingame_filtered["baseline_usage"] - ingame_filtered["same_type_share"])
    logger.info(
        "=== 종합: 대조군 없는 naive 추정 %+.4f -> placebo 통제 순수 효과 %+.4f (naive 대비 %.1f%%) ===",
        naive["mean"], placebo_overall["net_effect"], placebo_overall["net_effect"] / naive["mean"] * 100,
    )


if __name__ == "__main__":
    run()
