"""Intensive margin: when a pitcher DOES re-throw the pitch type that was just
hit, does he execute it differently (location, movement, velocity)?

Each pitch's execution is measured as its deviation from that pitcher's own
usual execution of that pitch type (a per-pitcher, per-season baseline), not
as a raw pre/post difference:
- location_dev: Euclidean distance (feet) of (plate_x, plate_z) from the
  pitcher's mean location for that type, against same-handed batters
  (baseline keyed by batter stand, since target sides flip with it).
- movement_dev: Euclidean distance of (pfx_x, pfx_z) from the mean movement.
- velocity_dev: release_speed minus the mean speed (mph, signed).

Observations per event (only events where the type WAS reused):
- time=0 (pre): the pitcher's earlier same-type pitches in that game, before
  the event pitch (the event pitch itself is excluded).
- time=1 (post): same-type pitches in the compared plate appearance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOCATION_KEYS = ["pitcher", "season", "pitch_type", "stand"]
PITCH_KEYS = ["pitcher", "season", "pitch_type"]
PITCH_QUALITY_COLUMNS = ["plate_x", "plate_z", "pfx_x", "pfx_z", "release_speed"]
OUTCOMES = ("location_dev", "movement_dev", "velocity_dev")


@dataclass
class ExecutionBaselines:
    location: pd.DataFrame
    movement: pd.DataFrame
    velocity: pd.DataFrame


def _baseline(pitches: pd.DataFrame, keys: list[str], value_cols: list[str], min_pitches: int) -> pd.DataFrame:
    valid = pitches.dropna(subset=value_cols + ["pitch_type"])
    grouped = valid.groupby(keys)
    out = grouped[value_cols].mean().rename(columns=lambda c: f"base_{c}")
    out["n_baseline"] = grouped.size()
    return out[out["n_baseline"] >= min_pitches].reset_index()


def compute_execution_baselines(pitches: pd.DataFrame, min_pitches: int = 20) -> ExecutionBaselines:
    """Per pitcher x season x pitch type mean location (also by batter stand),
    movement and velocity. Cells with fewer than `min_pitches` valid pitches
    are dropped (their pitches get no deviation).
    """
    return ExecutionBaselines(
        location=_baseline(pitches, LOCATION_KEYS, ["plate_x", "plate_z"], min_pitches),
        movement=_baseline(pitches, PITCH_KEYS, ["pfx_x", "pfx_z"], min_pitches),
        velocity=_baseline(pitches, PITCH_KEYS, ["release_speed"], min_pitches),
    )


def add_execution_deviations(observations: pd.DataFrame, baselines: ExecutionBaselines) -> pd.DataFrame:
    """Adds location_dev / movement_dev / velocity_dev; NaN when the pitch has
    no baseline cell or a missing measurement.
    """
    out = observations.merge(
        baselines.location[LOCATION_KEYS + ["base_plate_x", "base_plate_z"]], on=LOCATION_KEYS, how="left"
    )
    out = out.merge(baselines.movement[PITCH_KEYS + ["base_pfx_x", "base_pfx_z"]], on=PITCH_KEYS, how="left")
    out = out.merge(baselines.velocity[PITCH_KEYS + ["base_release_speed"]], on=PITCH_KEYS, how="left")
    out["location_dev"] = np.hypot(out["plate_x"] - out["base_plate_x"], out["plate_z"] - out["base_plate_z"])
    out["movement_dev"] = np.hypot(out["pfx_x"] - out["base_pfx_x"], out["pfx_z"] - out["base_pfx_z"])
    out["velocity_dev"] = out["release_speed"] - out["base_release_speed"]
    return out


def collect_pitch_observations(pitches: pd.DataFrame, events: pd.DataFrame, group_label: int) -> pd.DataFrame:
    """Long-format pitch observations (pre and post) for events where the hit
    pitch type was re-thrown in the compared plate appearance.

    `events` needs game_pk, pitcher, at_bat_number, event_pitch_number,
    next_at_bat_number, hit_pitch_type, reused_same_type, has_next_ab,
    baseline_usage, platoon_match. Each row carries its own count
    (balls/strikes) and pitching-team score differential at that pitch.
    """
    reused = events[events["has_next_ab"] & (events["reused_same_type"] == 1)].copy()
    reused["event_id"] = (
        f"{group_label}_" + reused["game_pk"].astype(str) + "_" + reused["at_bat_number"].astype(int).astype(str)
    )
    event_cols = reused[
        [
            "event_id", "game_pk", "pitcher", "at_bat_number", "event_pitch_number", "next_at_bat_number",
            "hit_pitch_type", "baseline_usage", "platoon_match",
        ]
    ].rename(columns={"at_bat_number": "event_at_bat"})

    pitch_cols = [
        "game_pk", "pitcher", "season", "pitch_type", "stand", "at_bat_number", "pitch_number", "balls", "strikes",
        "inning_topbot", "home_score", "away_score", *PITCH_QUALITY_COLUMNS,
    ]
    joined = pitches[pitch_cols].merge(
        event_cols, left_on=["game_pk", "pitcher", "pitch_type"], right_on=["game_pk", "pitcher", "hit_pitch_type"]
    )
    is_pre = (joined["at_bat_number"] < joined["event_at_bat"]) | (
        (joined["at_bat_number"] == joined["event_at_bat"]) & (joined["pitch_number"] < joined["event_pitch_number"])
    )
    is_post = joined["at_bat_number"] == joined["next_at_bat_number"]
    obs = pd.concat([joined[is_pre].assign(time=0), joined[is_post].assign(time=1)], ignore_index=True)

    obs["score_diff"] = np.where(
        obs["inning_topbot"] == "Top", obs["home_score"] - obs["away_score"], obs["away_score"] - obs["home_score"]
    )
    obs["group"] = group_label
    return obs.drop(columns=["inning_topbot", "home_score", "away_score", "event_at_bat", "event_pitch_number", "next_at_bat_number"])
