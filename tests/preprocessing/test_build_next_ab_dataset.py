import math

import pandas as pd

from src.preprocessing.build_next_ab_dataset import (
    build_event_dataset,
    compute_baseline_usage,
    compute_score_diff,
    find_next_at_bat_pitches,
    identify_extra_base_hit_events,
)


def _pitch(**overrides):
    row = {
        "game_pk": 1, "game_date": "2024-06-01", "pitcher": 100,
        "player_name": "Test Pitcher", "batter": 200, "pitch_type": "FF",
        "events": None, "at_bat_number": 1, "pitch_number": 1,
        "balls": 0, "strikes": 0, "outs_when_up": 0, "inning": 1,
        "inning_topbot": "Top", "stand": "R", "p_throws": "R",
        "home_score": 0, "away_score": 0,
    }
    row.update(overrides)
    return row


def test_identify_extra_base_hit_events_filters_by_events_column():
    pitches = pd.DataFrame([
        _pitch(events="single"),
        _pitch(events="double"),
        _pitch(events="triple"),
        _pitch(events="home_run"),
        _pitch(events=None),
    ])
    result = identify_extra_base_hit_events(pitches)
    assert sorted(result["events"].tolist()) == ["double", "home_run", "triple"]


def test_find_next_at_bat_pitches_matches_only_target_at_bat_number():
    game_pitches = pd.DataFrame([
        _pitch(at_bat_number=3, pitch_number=1),
        _pitch(at_bat_number=4, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=4, pitch_number=2, pitch_type="FF"),
        _pitch(at_bat_number=5, pitch_number=1),
    ])
    next_ab = find_next_at_bat_pitches(game_pitches, next_at_bat_number=4)
    assert len(next_ab) == 2
    assert set(next_ab["pitch_type"]) == {"SL", "FF"}


def test_compute_score_diff_uses_pitching_team_perspective():
    top_row = pd.Series(_pitch(inning_topbot="Top", home_score=5, away_score=2))
    bot_row = pd.Series(_pitch(inning_topbot="Bot", home_score=5, away_score=2))
    assert compute_score_diff(top_row) == 3  # home team pitching: 5 - 2
    assert compute_score_diff(bot_row) == -3  # away team pitching: 2 - 5


def test_compute_baseline_usage_is_nan_when_no_prior_pitches():
    game_pitches = pd.DataFrame([_pitch(at_bat_number=1, pitch_number=1, pitch_type="FF")])
    usage = compute_baseline_usage(
        game_pitches, before_at_bat_number=1, before_pitch_number=1, hit_pitch_type="FF"
    )
    assert math.isnan(usage)


def test_compute_baseline_usage_counts_earlier_at_bats_and_earlier_pitches_in_same_at_bat():
    game_pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="FF"),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="SL"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="FF"),  # earlier pitch, same AB as the hit
        _pitch(at_bat_number=2, pitch_number=2, pitch_type="FF"),  # the hit pitch itself: excluded
    ])
    usage = compute_baseline_usage(
        game_pitches, before_at_bat_number=2, before_pitch_number=2, hit_pitch_type="FF"
    )
    assert usage == 2 / 3  # FF at (1,1) and (2,1) out of 3 prior pitches


def test_build_event_dataset_marks_reuse_when_next_ab_repeats_pitch_type():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=1, pitch_number=2, pitch_type="FF", events="home_run"),
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL"),
        _pitch(at_bat_number=2, pitch_number=2, pitch_type="FF"),
        _pitch(at_bat_number=2, pitch_number=3, pitch_type="FF"),
    ])
    result = build_event_dataset(pitches)
    assert len(result) == 1
    event = result.iloc[0]
    assert event["has_next_ab"] == True
    assert event["next_ab_pitch_count"] == 3
    assert event["reused_same_type"] == 1
    assert event["same_type_share"] == 2 / 3
    assert event["hit_pitch_type"] == "FF"
    assert event["pitch_family"] == "fastball"
    assert event["baseline_usage"] == 0.0  # only prior pitch was SL


def test_build_event_dataset_flags_missing_next_ab_without_dropping_row():
    pitches = pd.DataFrame([
        _pitch(at_bat_number=1, pitch_number=1, pitch_type="FF", events="double"),
        # pitcher is pulled: at_bat_number 2 exists but is thrown by a different pitcher
        _pitch(at_bat_number=2, pitch_number=1, pitch_type="SL", pitcher=999),
    ])
    result = build_event_dataset(pitches)
    assert len(result) == 1
    event = result.iloc[0]
    assert event["has_next_ab"] == False
    assert event["next_ab_pitch_count"] == 0
    assert math.isnan(event["reused_same_type"])
    assert math.isnan(event["same_type_share"])
