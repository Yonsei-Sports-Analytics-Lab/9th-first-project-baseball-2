"""How much does the re-throw rate of the hit pitch type fall, per pitch type
(main analysis: same-batter rematch)?

Restricted to the seven most-used types (FF, SI, SL, CH, FC, CU, ST). For each:
- baseline usage rate -> re-throw usage rate (share of the rematch plate
  appearance's pitches that are the hit type) for the XBH and the field_out
  placebo groups, and the placebo-netted drop in %p;
- the relative reduction: the share of the expected re-throw rate that is
  lost, where the expectation applies the placebo's own change to the XBH
  baseline (see avoidance_stats.relative_reduction), with a bootstrap CI;
- the binary reuse rate (re-thrown at least once) for both groups.
A pooled "ALL" row over the seven types is appended for reference.

Uses the event parquet files written by run_same_batter_rematch_analysis.py;
no R needed. Figures: python -m src.visualization.pitch_type_reuse_plots.
"""

import logging
from pathlib import Path

import pandas as pd

from src.analysis.avoidance_stats import summarize_reuse_rate_reduction
from src.analysis.run_same_batter_rematch_analysis import PLACEBO_REMATCH_PATH, PLACEBO_SEED, XBH_REMATCH_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SUMMARY_PATH = PROCESSED_DATA_DIR / "pitch_type_reuse_rate_reduction.csv"
PITCH_TYPES = ["FF", "SI", "SL", "CH", "FC", "CU", "ST"]
BASELINE_COL = "season_usage_rate"
N_BOOT = 2000


def build_table(xbh: pd.DataFrame, placebo: pd.DataFrame) -> pd.DataFrame:
    per_type = summarize_reuse_rate_reduction(
        xbh, placebo, "hit_pitch_type", BASELINE_COL, PITCH_TYPES, n_boot=N_BOOT, seed=PLACEBO_SEED
    )
    selected_x = xbh[xbh["hit_pitch_type"].isin(PITCH_TYPES)].assign(hit_pitch_type="ALL")
    selected_p = placebo[placebo["hit_pitch_type"].isin(PITCH_TYPES)].assign(hit_pitch_type="ALL")
    pooled = summarize_reuse_rate_reduction(
        selected_x, selected_p, "hit_pitch_type", BASELINE_COL, ["ALL"], n_boot=N_BOOT, seed=PLACEBO_SEED
    )
    return pd.concat([per_type, pooled], ignore_index=True)


def run() -> None:
    if not (XBH_REMATCH_PATH.exists() and PLACEBO_REMATCH_PATH.exists()):
        raise FileNotFoundError("재대결 이벤트 파일이 없습니다. 먼저 src/analysis/run_same_batter_rematch_analysis.py를 실행하세요.")
    xbh, placebo = pd.read_parquet(XBH_REMATCH_PATH), pd.read_parquet(PLACEBO_REMATCH_PATH)
    table = build_table(xbh, placebo)
    table.to_csv(SUMMARY_PATH, index=False)

    view = table[
        ["hit_pitch_type", "n_xbh", "n_placebo", "xbh_baseline", "xbh_post", "xbh_drop", "placebo_baseline",
         "placebo_post", "placebo_drop", "net_drop", "net_ci_low", "net_ci_high", "relative_reduction",
         "rel_ci_low", "rel_ci_high", "reuse_xbh", "reuse_placebo", "reuse_diff", "reuse_diff_ci_low",
         "reuse_diff_ci_high"]
    ]
    pd.set_option("display.width", 300, "display.max_columns", 40)
    logger.info("구종별 재구사율 감소 (%s 기준, 부트스트랩 %d회)\n%s", BASELINE_COL, N_BOOT, view.round(4).to_string(index=False))
    logger.info("저장: %s", SUMMARY_PATH)


if __name__ == "__main__":
    run()
