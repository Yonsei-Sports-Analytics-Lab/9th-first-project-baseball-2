"""Avoidance by the pitch type that was hit (main analysis: same-batter rematch).

For every pitch type with enough events in both groups (>= MIN_BREAKDOWN_N):
- placebo-netted usage drop (season baseline), reuse rates, and the reuse gap
  versus what the pitcher's baseline usage alone would predict (comparable
  across types whose baseline usage differs a lot);
- a per-type mixed logistic model (reused_same_type ~ group * baseline_c +
  covariates + (0 + group | pitcher)) whose `group` odds ratio is the
  avoidance at THAT type's mean baseline usage (< 1 = avoided);
and one joint test of whether avoidance differs across pitch types
(likelihood-ratio test of group:hit_pitch_type).

Uses the event parquet files written by run_same_batter_rematch_analysis.py.
Needs R (lme4) and rpy2.
"""

import logging
from pathlib import Path

import pandas as pd

from src.analysis.avoidance_stats import summarize_avoidance_by_level
from src.analysis.glmer_runner import (
    R_CONVERTER,
    check_convergence,
    extract_fixed_effects_table,
    fit_glmer,
)
from src.analysis.mixed_effects_model import REQUIRED_MODEL_COLUMNS, build_combined_model_dataset
from src.analysis.run_same_batter_rematch_analysis import MIN_BREAKDOWN_N, PLACEBO_REMATCH_PATH, XBH_REMATCH_PATH
from src.preprocessing.pitch_family import map_pitch_family

import rpy2.robjects as ro

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SUMMARY_PATH = PROCESSED_DATA_DIR / "pitch_type_avoidance_summary.csv"
COVARIATES = "balls + strikes + outs_when_up + score_diff + stand + factor(season)"


def usable(events: pd.DataFrame) -> pd.DataFrame:
    return events[events["has_next_ab"] & events["baseline_usage"].notna()]


def fit_type_model(xbh_f: pd.DataFrame, placebo_f: pd.DataFrame, pitch_type: str) -> dict:
    combined = build_combined_model_dataset(
        xbh_f[xbh_f["hit_pitch_type"] == pitch_type], placebo_f[placebo_f["hit_pitch_type"] == pitch_type]
    )
    combined["baseline_c"] = combined["baseline_usage"] - combined["baseline_usage"].mean()
    fit_glmer(combined, f"reused_same_type ~ group * baseline_c + {COVARIATES} + (0 + group | pitcher)")
    fe = extract_fixed_effects_table().set_index("term")
    g = fe.loc["group"]
    return {
        "model_rows": len(combined), "or_group": g["odds_ratio"], "or_ci_low": g["or_ci_low"],
        "or_ci_high": g["or_ci_high"], "p_group": g["p_value"], "singular": "isSingular=True" in check_convergence(),
    }


def heterogeneity_test(xbh_f: pd.DataFrame, placebo_f: pd.DataFrame, pitch_types: list[str]) -> dict:
    combined = build_combined_model_dataset(
        xbh_f[xbh_f["hit_pitch_type"].isin(pitch_types)], placebo_f[placebo_f["hit_pitch_type"].isin(pitch_types)]
    )
    base = f"reused_same_type ~ group * baseline_usage + hit_pitch_type + {COVARIATES}"
    fit_glmer(combined, f"{base} + (0 + group | pitcher)")
    ro.r("m0 <- model")
    fit_glmer(combined, f"{base} + group:hit_pitch_type + (0 + group | pitcher)")
    ro.r("m1 <- model")
    table = ro.r("as.data.frame(anova(m0, m1))")
    with R_CONVERTER.context():
        table = ro.conversion.get_conversion().rpy2py(table)
    last = table.iloc[-1]
    return {"n_rows": len(combined), "chisq": last["Chisq"], "df": last["Df"], "p": last["Pr(>Chisq)"]}


def run() -> None:
    if not (XBH_REMATCH_PATH.exists() and PLACEBO_REMATCH_PATH.exists()):
        raise FileNotFoundError("재대결 이벤트 파일이 없습니다. 먼저 src/analysis/run_same_batter_rematch_analysis.py를 실행하세요.")
    xbh, placebo = pd.read_parquet(XBH_REMATCH_PATH), pd.read_parquet(PLACEBO_REMATCH_PATH)

    table = summarize_avoidance_by_level(xbh, placebo, "hit_pitch_type", "season_usage_rate", MIN_BREAKDOWN_N)
    table.insert(1, "pitch_family", table["hit_pitch_type"].map(map_pitch_family))
    kept = table["hit_pitch_type"].tolist()
    left_out = xbh[~xbh["hit_pitch_type"].isin(kept)]["hit_pitch_type"].value_counts(dropna=False)
    logger.info(
        "분석 구종 %d개 (양쪽 표본 >=%d): %s | 제외된 장타 재대결 %d건: %s",
        len(kept), MIN_BREAKDOWN_N, kept, int(left_out.sum()), left_out.to_dict(),
    )

    xbh_f, placebo_f = usable(xbh), usable(placebo)
    models = pd.DataFrame([{"hit_pitch_type": t, **fit_type_model(xbh_f, placebo_f, t)} for t in kept])
    table = table.merge(models, on="hit_pitch_type")
    table = table.sort_values("or_group").reset_index(drop=True)
    table.to_csv(SUMMARY_PATH, index=False)

    view = table.assign(
        재사용률_장타=table["reuse_xbh"], 재사용률_대조군=table["reuse_placebo"],
    )[
        ["hit_pitch_type", "pitch_family", "n_xbh", "n_placebo", "xbh_baseline", "xbh_post", "reuse_xbh",
         "reuse_placebo", "net_drop", "net_ci_low", "net_ci_high", "reuse_gap_net", "or_group", "or_ci_low",
         "or_ci_high", "p_group", "singular"]
    ]
    pd.set_option("display.width", 250)
    logger.info("구종별 회피 (회피 강한 순 = group 오즈비 낮은 순)\n%s", view.round(4).to_string(index=False))

    het = heterogeneity_test(xbh_f, placebo_f, kept)
    logger.info(
        "구종에 따라 회피 강도가 다른가? 우도비 검정 group:hit_pitch_type (%d행): chi2=%.1f, df=%d, p=%.3e",
        het["n_rows"], het["chisq"], het["df"], het["p"],
    )
    logger.info("저장: %s", SUMMARY_PATH)


if __name__ == "__main__":
    run()
