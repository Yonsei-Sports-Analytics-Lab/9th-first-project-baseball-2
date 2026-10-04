import numpy as np
import pandas as pd

from src.analysis.pitch_state import add_prior_type_state, state_buckets


def _pitches():
    # one pitcher, one game (game 1) + another game (game 2) in the same season, all SL unless noted
    rows = [
        # game, ab, pitch, type, speed, description, xwoba
        (1, 1, 1, "SL", 85.0, "swinging_strike", np.nan),
        (1, 1, 2, "FF", 95.0, "ball", np.nan),
        (1, 1, 3, "SL", 83.0, "hit_into_play", 0.9),
        (1, 2, 1, "SL", 84.0, "foul", np.nan),
        (2, 1, 1, "SL", 88.0, "called_strike", np.nan),
    ]
    df = pd.DataFrame(rows, columns=["game_pk", "at_bat_number", "pitch_number", "pitch_type", "release_speed",
                                     "description", "estimated_woba_using_speedangle"])
    df["pitcher"], df["season"] = 10, 2025
    return df


def test_state_uses_only_earlier_same_type_pitches_in_the_game():
    r = add_prior_type_state(_pitches()).set_index(["game_pk", "at_bat_number", "pitch_number"])
    season_velo = (85 + 83 + 84 + 88) / 4  # SL season mean = 85
    season_whiff = 1 / 3                     # SL swings: whiff, in play, foul
    season_xwoba = 0.9

    first = r.loc[(1, 1, 1)]
    assert first["prior_t_n"] == 0 and np.isnan(first["velo_gap"]) and np.isnan(first["whiff_gap"])

    third_sl = r.loc[(1, 2, 1)]  # earlier SL in game 1: 85 (whiff), 83 (in play, xwOBA 0.9)
    assert third_sl["prior_t_n"] == 2
    assert np.isclose(third_sl["velo_gap"], 84 - season_velo)
    assert np.isclose(third_sl["whiff_gap"], 1 / 2 - season_whiff)
    assert np.isclose(third_sl["xwoba_gap"], 0.9 - season_xwoba)

    game2 = r.loc[(2, 1, 1)]  # a new game starts with no prior state
    assert game2["prior_t_n"] == 0 and np.isnan(game2["velo_gap"])


def test_state_buckets():
    ev = pd.DataFrame({"velo_gap": [np.nan, -1.0, 0.2, 0.8], "whiff_gap": [0.0, -0.2, 0.15, np.nan],
                       "xwoba_gap": [0.2, 0.0, -0.3, np.nan]})
    b = state_buckets(ev)
    assert b["velo_state"].tolist() == ["측정 불가", "느림", "보통", "빠름"]
    assert b["whiff_state"].tolist() == ["보통", "헛스윙 적음", "헛스윙 많음", "측정 불가"]
    assert b["xwoba_state"].tolist() == ["강하게 맞음", "보통", "약하게 맞음", "측정 불가"]
