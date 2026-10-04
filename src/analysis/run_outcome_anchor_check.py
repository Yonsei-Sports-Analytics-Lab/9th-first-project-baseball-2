"""Why does usage of the anchor pitch type RISE after a field_out?

In the same-batter rematch, the field_out placebo re-throws the anchor type
more than the pitcher's season baseline (+3.3%p) while the XBH group throws it
less (-11.0%p). Two explanations:
1. Reinforcement: the pitch worked (an out), so the pitcher goes back to it.
2. Selection/persistence: conditioning on a type having been thrown in this
   game already picks games where that type is used more than its season
   average -- whatever the outcome.

This runs the identical rematch design, with the same season x pitch_family
matching to the XBH group, on anchor pitches with different outcomes
(strikeout, single, walk, plus field_out and XBH) and on an outcome-free
anchor (the first pitch of a plate appearance). If reinforcement drove the
rise, strikeout/field_out would sit clearly above the outcome-free anchor.

Uses the event parquet files written by run_same_batter_rematch_analysis.py.
"""

import logging

import numpy as np
import pandas as pd

from src.analysis.avoidance_stats import build_stratified_placebo_candidates, compute_season_usage_rate
from src.analysis.run_same_batter_rematch_analysis import (
    PLACEBO_REMATCH_PATH,
    PLACEBO_SEED,
    XBH_REMATCH_PATH,
    attach_season_usage,
    load_pitches,
)
from src.preprocessing.build_next_ab_dataset import build_event_dataset_for_events, filter_to_same_batter_rematch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTCOME_ANCHORS = ["strikeout", "single", "walk"]
NEUTRAL_LABEL = "첫 투구(결과 무관)"


def build_anchor_group(
    pitches: pd.DataFrame,
    candidate_view: pd.DataFrame,
    xbh: pd.DataFrame,
    season_usage: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Rematch events anchored on the rows of `candidate_view` whose `events`
    equals `label`, matched to the XBH group's season x pitch_family counts.
    """
    candidates = build_stratified_placebo_candidates(
        candidate_view, xbh, {label}, seed=PLACEBO_SEED, candidate_filter=filter_to_same_batter_rematch
    )
    events = build_event_dataset_for_events(pitches, candidates, next_pa_mode="same_batter")
    return attach_season_usage(events, season_usage)


def summarize_anchor(name: str, events: pd.DataFrame) -> dict:
    d = events[events["has_next_ab"] & events["season_usage_rate"].notna()]
    baseline, post = d["season_usage_rate"].mean(), d["same_type_share"].mean()
    return {
        "anchor": name, "n": len(d), "baseline": baseline, "rematch_usage": post,
        "change_pp": (post - baseline) * 100, "reuse_rate": d["reused_same_type"].mean(),
    }


def run() -> None:
    pitches = load_pitches()
    season_usage = compute_season_usage_rate(pitches)
    xbh = pd.read_parquet(XBH_REMATCH_PATH)
    groups = {"장타(2B/3B/HR)": xbh, "field_out": pd.read_parquet(PLACEBO_REMATCH_PATH)}

    for label in OUTCOME_ANCHORS:
        groups[label] = build_anchor_group(pitches, pitches, xbh, season_usage, label)
        logger.info("%s 앵커 %d건 생성", label, len(groups[label]))

    first_pitch_view = pitches.assign(events=np.where(pitches["pitch_number"] == 1, "first_pitch", None))
    groups[NEUTRAL_LABEL] = build_anchor_group(pitches, first_pitch_view, xbh, season_usage, "first_pitch")

    table = pd.DataFrame([summarize_anchor(name, ev) for name, ev in groups.items()])
    neutral = table.loc[table["anchor"] == NEUTRAL_LABEL].iloc[0]
    table["change_vs_neutral_pp"] = table["change_pp"] - neutral["change_pp"]
    table["reuse_vs_neutral_pp"] = (table["reuse_rate"] - neutral["reuse_rate"]) * 100
    logger.info("앵커 투구 결과별 재대결 구사율 변화 (시즌 baseline 기준)\n%s", table.round(4).to_string(index=False))


if __name__ == "__main__":
    run()
