"""Build the event-level "did the pitcher reuse the same pitch type in the
batter's next plate appearance after allowing an extra-base hit" dataset.
"""

from __future__ import annotations

import pandas as pd

from src.preprocessing.pitch_family import is_cutter, map_pitch_family

EXTRA_BASE_HIT_EVENTS = {"double", "triple", "home_run"}


def identify_extra_base_hit_events(pitches: pd.DataFrame) -> pd.DataFrame:
    return pitches[pitches["events"].isin(EXTRA_BASE_HIT_EVENTS)].copy()


def find_next_at_bat_pitches(game_pitches: pd.DataFrame, next_at_bat_number: int) -> pd.DataFrame:
    """`game_pitches` must already be filtered to one (game_pk, pitcher)."""
    return game_pitches[game_pitches["at_bat_number"] == next_at_bat_number]


def compute_score_diff(hit_row: pd.Series) -> int:
    """Score differential from the pitching team's perspective at the moment of the hit."""
    if hit_row["inning_topbot"] == "Top":
        return int(hit_row["home_score"] - hit_row["away_score"])
    return int(hit_row["away_score"] - hit_row["home_score"])


def compute_baseline_usage(
    game_pitches: pd.DataFrame,
    before_at_bat_number: int,
    before_pitch_number: int,
    hit_pitch_type: str,
) -> float:
    """Share of `hit_pitch_type` among this pitcher's pitches in the same game,
    strictly before the extra-base-hit pitch (earlier at-bats, plus earlier
    pitches within the same at-bat). NaN if there were no prior pitches.
    """
    prior = game_pitches[
        (game_pitches["at_bat_number"] < before_at_bat_number)
        | (
            (game_pitches["at_bat_number"] == before_at_bat_number)
            & (game_pitches["pitch_number"] < before_pitch_number)
        )
    ]
    if len(prior) == 0:
        return float("nan")
    return float((prior["pitch_type"] == hit_pitch_type).mean())


def build_event_dataset(pitches: pd.DataFrame) -> pd.DataFrame:
    events = identify_extra_base_hit_events(pitches)
    grouped = {key: group for key, group in pitches.groupby(["game_pk", "pitcher"])}

    records = []
    for _, hit in events.iterrows():
        game_pitches = grouped[(hit["game_pk"], hit["pitcher"])]

        next_ab = find_next_at_bat_pitches(game_pitches, hit["at_bat_number"] + 1)
        has_next_ab = len(next_ab) > 0
        next_ab_pitch_count = len(next_ab)
        if has_next_ab:
            reused_same_type = int((next_ab["pitch_type"] == hit["pitch_type"]).any())
            same_type_share = float((next_ab["pitch_type"] == hit["pitch_type"]).mean())
        else:
            reused_same_type = float("nan")
            same_type_share = float("nan")

        baseline_usage = compute_baseline_usage(
            game_pitches, hit["at_bat_number"], hit["pitch_number"], hit["pitch_type"]
        )

        records.append(
            {
                "game_pk": hit["game_pk"],
                "game_date": hit["game_date"],
                "season": pd.to_datetime(hit["game_date"]).year,
                "pitcher": hit["pitcher"],
                "pitcher_name": hit["player_name"],
                "batter": hit["batter"],
                "stand": hit["stand"],
                "p_throws": hit["p_throws"],
                "at_bat_number": hit["at_bat_number"],
                "events": hit["events"],
                "hit_pitch_type": hit["pitch_type"],
                "pitch_family": map_pitch_family(hit["pitch_type"]),
                "is_cutter": is_cutter(hit["pitch_type"]),
                "balls": hit["balls"],
                "strikes": hit["strikes"],
                "outs_when_up": hit["outs_when_up"],
                "inning": hit["inning"],
                "inning_topbot": hit["inning_topbot"],
                "score_diff": compute_score_diff(hit),
                "baseline_usage": baseline_usage,
                "has_next_ab": has_next_ab,
                "next_ab_pitch_count": next_ab_pitch_count,
                "reused_same_type": reused_same_type,
                "same_type_share": same_type_share,
            }
        )

    return pd.DataFrame.from_records(records)
