"""Pitch-avoidance analysis restricted to SAME-BATTER REMATCHES.

The main analysis (run_pitch_avoidance_analysis.py) compares against the
very next batter (a different batter). This script instead keeps only
extra-base hits where the same pitcher faces the SAME batter again in that
batter's immediately following plate appearance of the game, and asks
whether the pitcher avoids the pitch type that batter just hit.

Everything is built with the exact same event builder as the main analysis
(build_event_dataset_for_events with next_pa_mode="same_batter"), and the
placebo group (field_out) is drawn only from field_outs that also have a
same-batter rematch, stratified to the XBH group's season x pitch_family
distribution. Reports, against two baselines (within-game baseline_usage and
the pitcher's season-wide usage rate): the paired drop, the placebo-netted
diff-in-diff (overall / by pitch family / by season / baseline>0), the binary
reuse comparison, and the mixed-effects model (needs R; skipped if absent).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.avoidance_stats import (
    build_stratified_placebo_candidates,
    compute_expected_reuse_prob,
    compute_season_usage_rate,
    summarize_diff_in_diff,
    summarize_paired_diff,
)
from src.analysis.mixed_effects_model import MODEL_FORMULA_ROBUSTNESS, build_combined_model_dataset
from src.preprocessing.build_next_ab_dataset import (
    build_event_dataset_for_events,
    filter_to_same_batter_rematch,
    identify_extra_base_hit_events,
)
from src.preprocessing.load_raw_pitches import load_all_pitches

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
XBH_REMATCH_PATH = PROCESSED_DATA_DIR / "pitch_reuse_after_xbh_same_batter_rematch_events.parquet"
PLACEBO_REMATCH_PATH = PROCESSED_DATA_DIR / "pitch_reuse_placebo_same_batter_rematch_events.parquet"
RANDOM_EFFECTS_PATH = PROCESSED_DATA_DIR / "pitch_avoidance_same_batter_rematch_random_effects.csv"
PLACEBO_OUTCOME_EVENTS = {"field_out"}
PLACEBO_SEED = 20260923
NEEDED_COLUMNS = [
    "game_pk", "game_date", "season", "pitcher", "player_name", "batter", "pitch_type", "events",
    "at_bat_number", "pitch_number", "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "stand", "p_throws", "home_score", "away_score", "fielder_2",
]


def attach_season_usage(events: pd.DataFrame, season_usage: pd.DataFrame) -> pd.DataFrame:
    return events.merge(
        season_usage[["pitcher", "season", "pitch_type", "season_usage_rate"]],
        left_on=["pitcher", "season", "hit_pitch_type"],
        right_on=["pitcher", "season", "pitch_type"],
        how="left",
    ).drop(columns=["pitch_type"])


def report_paired(label: str, events: pd.DataFrame, baseline_col: str) -> None:
    d = events[events["has_next_ab"] & events[baseline_col].notna()]
    r = summarize_paired_diff(d[baseline_col] - d["same_type_share"])
    logger.info(
        "[%s] 사전 %.4f -> 사후 %.4f, 감소 %.4f (95%% CI [%.4f, %.4f]), n=%d, p=%.2e",
        label, d[baseline_col].mean(), d["same_type_share"].mean(), r["mean"], r["ci_low"], r["ci_high"], r["n"], r["t_p"],
    )


def report_season_table(events: pd.DataFrame, baseline_col: str) -> None:
    d = events[events["has_next_ab"] & events[baseline_col].notna()]
    rows = []
    for season, g in d.groupby("season"):
        rows.append({
            "season": season, "events": len(g), "baseline": g[baseline_col].mean(),
            "post": g["same_type_share"].mean(), "diff(post-baseline)": g["same_type_share"].mean() - g[baseline_col].mean(),
        })
    logger.info("[XBH 재대결, %s 기준] 시즌별\n%s", baseline_col, pd.DataFrame(rows).round(4).to_string(index=False))


def report_diff_in_diff(label: str, xbh: pd.DataFrame, placebo: pd.DataFrame, baseline_col: str) -> None:
    def prep(events: pd.DataFrame) -> pd.DataFrame:
        d = events[events["has_next_ab"] & events[baseline_col].notna()].copy()
        d["diff"] = d[baseline_col] - d["same_type_share"]
        return d

    x, p = prep(xbh), prep(placebo)
    o = summarize_diff_in_diff(x["diff"], p["diff"])
    logger.info(
        "[%s / %s] n_XBH=%d n_placebo=%d | XBH 감소 %.4f, placebo 감소 %.4f, 순수효과 %.4f (95%% CI [%.4f, %.4f]) Welch p=%.2e MW p=%.2e",
        label, baseline_col, o["n_treatment"], o["n_control"], o["treatment_mean"], o["control_mean"],
        o["net_effect"], o["ci_low"], o["ci_high"], o["t_p"], o["u_p"],
    )
    for fam in ["fastball", "breaking", "offspeed"]:
        r = summarize_diff_in_diff(x[x["pitch_family"] == fam]["diff"], p[p["pitch_family"] == fam]["diff"])
        logger.info(
            "    family=%-8s n=%d/%d XBH %.4f placebo %.4f 순수 %.4f [%.4f, %.4f]",
            fam, r["n_treatment"], r["n_control"], r["treatment_mean"], r["control_mean"], r["net_effect"], r["ci_low"], r["ci_high"],
        )
    for season in sorted(x["season"].unique()):
        r = summarize_diff_in_diff(x[x["season"] == season]["diff"], p[p["season"] == season]["diff"])
        logger.info(
            "    season=%s n=%d/%d XBH %.4f placebo %.4f 순수 %.4f [%.4f, %.4f]",
            season, r["n_treatment"], r["n_control"], r["treatment_mean"], r["control_mean"], r["net_effect"], r["ci_low"], r["ci_high"],
        )
    r = summarize_diff_in_diff(x[x[baseline_col] > 0]["diff"], p[p[baseline_col] > 0]["diff"])
    logger.info(
        "    robustness(baseline>0) n=%d/%d XBH %.4f placebo %.4f 순수 %.4f [%.4f, %.4f]",
        r["n_treatment"], r["n_control"], r["treatment_mean"], r["control_mean"], r["net_effect"], r["ci_low"], r["ci_high"],
    )


def report_binary(xbh: pd.DataFrame, placebo: pd.DataFrame, baseline_col: str) -> None:
    def prep(events: pd.DataFrame) -> pd.DataFrame:
        d = events[events["has_next_ab"] & events[baseline_col].notna()].copy()
        d["reuse_gap"] = d["reused_same_type"] - compute_expected_reuse_prob(d[baseline_col], d["next_ab_pitch_count"])
        return d

    x, p = prep(xbh), prep(placebo)
    g = summarize_diff_in_diff(x["reuse_gap"], p["reuse_gap"], alternative="less")
    contingency = pd.DataFrame(
        {
            "reused=1": [x["reused_same_type"].sum(), p["reused_same_type"].sum()],
            "reused=0": [(x["reused_same_type"] == 0).sum(), (p["reused_same_type"] == 0).sum()],
        },
        index=["XBH", "placebo"],
    )
    chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
    logger.info(
        "[이진 재사용 / %s] raw 재사용률 XBH %.4f vs placebo %.4f (chi2=%.1f p=%.2e) | 기대 대비 gap XBH %.4f placebo %.4f 순수 %.4f (p=%.2e)",
        baseline_col, x["reused_same_type"].mean(), p["reused_same_type"].mean(), chi2, chi_p,
        g["treatment_mean"], g["control_mean"], g["net_effect"], g["t_p"],
    )


def report_mixed_model(xbh_f: pd.DataFrame, placebo_f: pd.DataFrame) -> None:
    try:
        from src.analysis.glmer_runner import (
            check_convergence,
            extract_fixed_effects_table,
            extract_pitcher_variance_components,
            extract_random_effects_table,
            fit_glmer,
        )
    except ImportError:
        logger.warning("rpy2/R 없음 -- 혼합효과 모델 생략")
        return

    combined = build_combined_model_dataset(xbh_f, placebo_f)
    logger.info(
        "[혼합효과] 표본 %d행 (XBH %d + placebo %d), 투수 %d명 | 공식: %s",
        len(combined), int((combined["group"] == 1).sum()), int((combined["group"] == 0).sum()),
        combined["pitcher"].nunique(), MODEL_FORMULA_ROBUSTNESS,
    )
    fit_glmer(combined, MODEL_FORMULA_ROBUSTNESS)
    logger.info("[혼합효과] 수렴: %s", check_convergence())
    fe = extract_fixed_effects_table()
    logger.info(
        "[혼합효과] 고정효과\n%s",
        fe[["term", "estimate", "std_error", "p_value", "odds_ratio"]].round(4).to_string(index=False),
    )
    _, group_var, _ = extract_pitcher_variance_components()
    logger.info("[혼합효과] 투수 group 기울기 분산=%.4f", group_var)
    extract_random_effects_table().to_csv(RANDOM_EFFECTS_PATH, index=False)
    logger.info("[혼합효과] 투수별 랜덤효과 저장: %s", RANDOM_EFFECTS_PATH)


def run() -> None:
    pitches = load_all_pitches()[NEEDED_COLUMNS].copy()

    xbh_all = identify_extra_base_hit_events(pitches)
    xbh_candidates = filter_to_same_batter_rematch(pitches, xbh_all)
    by_season = pd.DataFrame({
        "xbh_total": xbh_all.groupby("season").size(),
        "same_batter_rematch": xbh_candidates.groupby("season").size(),
    })
    by_season["rematch_share"] = by_season["same_batter_rematch"] / by_season["xbh_total"]
    logger.info(
        "장타 %d건 중 같은 타자·같은 투수 재대결 %d건 (%.1f%%)\n%s",
        len(xbh_all), len(xbh_candidates), len(xbh_candidates) / len(xbh_all) * 100, by_season.round(4).to_string(),
    )

    xbh = build_event_dataset_for_events(pitches, xbh_candidates, next_pa_mode="same_batter")
    placebo_candidates = build_stratified_placebo_candidates(
        pitches, xbh, PLACEBO_OUTCOME_EVENTS, seed=PLACEBO_SEED, candidate_filter=filter_to_same_batter_rematch
    )
    placebo = build_event_dataset_for_events(pitches, placebo_candidates, next_pa_mode="same_batter")
    logger.info("placebo(field_out 재대결) %d건 층화추출 (XBH 재대결 %d건에 맞춤)", len(placebo), len(xbh))

    season_usage = compute_season_usage_rate(pitches)
    xbh = attach_season_usage(xbh, season_usage)
    placebo = attach_season_usage(placebo, season_usage)
    xbh.to_parquet(XBH_REMATCH_PATH, index=False)
    placebo.to_parquet(PLACEBO_REMATCH_PATH, index=False)

    logger.info("=== 1. XBH 재대결만 (대조군 없음) ===")
    report_paired("경기 내 baseline", xbh, "baseline_usage")
    report_paired("시즌 baseline", xbh, "season_usage_rate")
    report_season_table(xbh, "season_usage_rate")

    logger.info("=== 2. placebo(field_out 재대결) 넷팅 diff-in-diff ===")
    report_diff_in_diff("재대결", xbh, placebo, "season_usage_rate")
    report_diff_in_diff("재대결", xbh, placebo, "baseline_usage")

    logger.info("=== 3. 이진 재사용 지표 ===")
    report_binary(xbh, placebo, "season_usage_rate")

    logger.info("=== 4. 혼합효과 로지스틱 회귀 (robustness 공식) ===")
    xbh_f = xbh[xbh["has_next_ab"] & xbh["baseline_usage"].notna()]
    placebo_f = placebo[placebo["has_next_ab"] & placebo["baseline_usage"].notna()]
    report_mixed_model(xbh_f, placebo_f)


if __name__ == "__main__":
    run()
