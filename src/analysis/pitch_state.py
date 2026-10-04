"""How good was pitch type T *that day*, before the event pitch?

Alternative explanation for the avoidance: XBHs come off pitch types that are working badly
that day (slower, fewer whiffs, hit harder), and the pitcher would throw them less anyway. To
control for it, every pitch gets the state of its own type over the pitcher's EARLIER pitches
of that type in the same game, each relative to the pitcher's season level of that type:

- velo_gap:  mean release_speed (mph) - season mean
- whiff_gap: whiffs / swings - season whiffs / swings (bunts excluded)
- xwoba_gap: mean xwOBA of batted balls - season mean

A gap is NaN when there is nothing earlier to measure (no earlier pitch / swing / batted ball).
Season levels include the game itself (small look-ahead, same as season_usage_rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WHIFFS = {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
SWINGS = WHIFFS | {"foul", "hit_into_play"}
XWOBA = "estimated_woba_using_speedangle"
STATE_COLUMNS = ["prior_t_n", "velo_gap", "whiff_gap", "xwoba_gap"]


def add_prior_type_state(pitches: pd.DataFrame) -> pd.DataFrame:
    out = pitches.sort_values(["game_pk", "pitcher", "at_bat_number", "pitch_number"]).copy()
    out["_velo_n"] = out["release_speed"].notna().astype(float)
    out["_velo"] = out["release_speed"].fillna(0.0)
    out["_swing"] = out["description"].isin(SWINGS).astype(float)
    out["_whiff"] = out["description"].isin(WHIFFS).astype(float)
    out["_bip_n"] = out[XWOBA].notna().astype(float)
    out["_bip"] = out[XWOBA].fillna(0.0)
    parts = ["_velo_n", "_velo", "_swing", "_whiff", "_bip_n", "_bip"]

    # running totals over EARLIER pitches of the same type in the game (current pitch excluded)
    g = out.groupby(["game_pk", "pitcher", "pitch_type"], sort=False)[parts]
    prior = g.cumsum() - out[parts]
    season = out.groupby(["pitcher", "season", "pitch_type"])[parts].transform("sum")

    def ratio(num, den):
        return (num / den).where(den > 0)

    out["prior_t_n"] = out.groupby(["game_pk", "pitcher", "pitch_type"], sort=False).cumcount()
    out["velo_gap"] = ratio(prior["_velo"], prior["_velo_n"]) - ratio(season["_velo"], season["_velo_n"])
    out["whiff_gap"] = ratio(prior["_whiff"], prior["_swing"]) - ratio(season["_whiff"], season["_swing"])
    out["xwoba_gap"] = ratio(prior["_bip"], prior["_bip_n"]) - ratio(season["_bip"], season["_bip_n"])
    return out.drop(columns=parts)


def state_buckets(events: pd.DataFrame) -> pd.DataFrame:
    """Coarse matching cells for the three gaps (fixed, interpretable thresholds)."""
    out = events.copy()

    def bucket(x: pd.Series, cut: float, low: str, high: str) -> np.ndarray:
        return np.select([x.isna(), x <= -cut, x >= cut], ["측정 불가", low, high], default="보통")

    out["velo_state"] = bucket(out["velo_gap"], 0.7, "느림", "빠름")
    out["whiff_state"] = bucket(out["whiff_gap"], 0.10, "헛스윙 적음", "헛스윙 많음")
    out["xwoba_state"] = bucket(out["xwoba_gap"], 0.15, "약하게 맞음", "강하게 맞음")
    return out
