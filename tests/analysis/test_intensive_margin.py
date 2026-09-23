import math

import pandas as pd
import pytest

from src.analysis.intensive_margin import (
    add_execution_deviations,
    collect_pitch_observations,
    compute_execution_baselines,
)


def _pitch(**overrides):
    row = {
        "game_pk": 1, "pitcher": 100, "season": 2021, "pitch_type": "FF", "stand": "R",
        "at_bat_number": 1, "pitch_number": 1, "balls": 0, "strikes": 0,
        "inning_topbot": "Top", "home_score": 3, "away_score": 1,
        "plate_x": 0.0, "plate_z": 2.0, "pfx_x": 1.0, "pfx_z": 1.0, "release_speed": 95.0,
    }
    row.update(overrides)
    return row


def _event(**overrides):
    row = {
        "game_pk": 1, "pitcher": 100, "at_bat_number": 2, "event_pitch_number": 2, "next_at_bat_number": 10,
        "hit_pitch_type": "FF", "reused_same_type": 1, "has_next_ab": True, "baseline_usage": 0.5,
        "platoon_match": 1,
    }
    row.update(overrides)
    return row


def _game():
    return pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, plate_x=0.5),                      # pre (FF)
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="SL"),                  # other type: ignored
        _pitch(at_bat_number=2, pitch_number=1, plate_x=-0.5),                     # pre (FF, earlier in the XBH PA)
        _pitch(at_bat_number=2, pitch_number=2, plate_x=9.0),                      # the XBH pitch itself: excluded
        _pitch(at_bat_number=5, pitch_number=1, plate_x=7.0),                      # after the XBH but not the compared PA
        _pitch(at_bat_number=10, pitch_number=1, plate_x=1.0, balls=1, strikes=2), # post (FF)
        _pitch(at_bat_number=10, pitch_number=2, pitch_type="SL"),                 # post, other type: ignored
        _pitch(at_bat_number=10, pitch_number=3, plate_x=1.5, home_score=5, away_score=1),  # post (FF)
    ])


def test_collect_pitch_observations_splits_pre_and_post_same_type_pitches():
    obs = collect_pitch_observations(_game(), pd.DataFrame([_event()]), group_label=1)

    pre = obs[obs["time"] == 0]
    post = obs[obs["time"] == 1]
    assert sorted(pre["plate_x"]) == [-0.5, 0.5]  # XBH pitch (9.0) and later non-compared pitches excluded
    assert sorted(post["plate_x"]) == [1.0, 1.5]
    assert (obs["group"] == 1).all()
    assert obs["event_id"].nunique() == 1


def test_collect_pitch_observations_uses_each_pitchs_own_count_and_score_diff():
    obs = collect_pitch_observations(_game(), pd.DataFrame([_event()]), group_label=1)
    post = obs[obs["time"] == 1].sort_values("pitch_number")
    assert post["balls"].tolist() == [1, 0]
    assert post["strikes"].tolist() == [2, 0]
    # top of the inning => home team pitching: home_score - away_score at that pitch
    assert post["score_diff"].tolist() == [2, 4]


def test_collect_pitch_observations_skips_events_without_reuse_or_next_ab():
    events = pd.DataFrame([
        _event(reused_same_type=0),
        _event(at_bat_number=3, has_next_ab=False, reused_same_type=float("nan")),
    ])
    obs = collect_pitch_observations(_game(), events, group_label=0)
    assert obs.empty


def test_execution_baselines_and_deviations():
    pitches = pd.DataFrame([
        _pitch(plate_x=0.0, plate_z=2.0, pfx_x=1.0, pfx_z=1.0, release_speed=90.0),
        _pitch(plate_x=2.0, plate_z=2.0, pfx_x=3.0, pfx_z=1.0, release_speed=94.0),
    ])
    baselines = compute_execution_baselines(pitches, min_pitches=2)
    probe = pd.DataFrame([_pitch(plate_x=4.0, plate_z=6.0, pfx_x=2.0, pfx_z=4.0, release_speed=91.0)])
    dev = add_execution_deviations(probe, baselines).iloc[0]
    # baseline location (1, 2): distance from (4, 6) = sqrt(9 + 16) = 5
    assert dev["location_dev"] == pytest.approx(5.0)
    # baseline movement (2, 1): distance from (2, 4) = 3
    assert dev["movement_dev"] == pytest.approx(3.0)
    # baseline velocity 92: signed difference
    assert dev["velocity_dev"] == pytest.approx(-1.0)


def test_baselines_need_enough_pitches_and_ignore_missing_values():
    pitches = pd.DataFrame([
        _pitch(release_speed=90.0),
        _pitch(release_speed=float("nan")),
    ])
    baselines = compute_execution_baselines(pitches, min_pitches=2)
    probe = pd.DataFrame([_pitch()])
    dev = add_execution_deviations(probe, baselines).iloc[0]
    assert math.isnan(dev["velocity_dev"])  # only 1 valid speed < min_pitches
    assert not math.isnan(dev["location_dev"])  # location has 2 valid pitches
