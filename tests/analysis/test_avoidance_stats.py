import pandas as pd
import pytest

from src.analysis.avoidance_stats import (
    build_stratified_placebo_candidates,
    compute_expected_reuse_prob,
    compute_season_usage_rate,
    summarize_diff_in_diff,
    summarize_paired_diff,
)


def test_compute_season_usage_rate_computes_share_per_pitcher_season_pitch_type():
    pitches = pd.DataFrame({
        "pitcher": [1, 1, 1, 1, 2, 2],
        "season": [2021, 2021, 2021, 2021, 2021, 2021],
        "pitch_type": ["FF", "FF", "SL", "SL", "FF", "SL"],
    })
    usage = compute_season_usage_rate(pitches)
    row = usage[(usage["pitcher"] == 1) & (usage["pitch_type"] == "FF")].iloc[0]
    assert row["n"] == 2
    assert row["total"] == 4
    assert row["season_usage_rate"] == 0.5


def test_compute_season_usage_rate_ignores_pitches_with_no_pitch_type():
    pitches = pd.DataFrame({
        "pitcher": [1, 1, 1, 1],
        "season": [2021, 2021, 2021, 2021],
        "pitch_type": ["FF", "SL", None, None],
    })
    usage = compute_season_usage_rate(pitches)
    assert usage["pitch_type"].notna().all()  # no bucket for "missing" that a missing event type could match
    ff = usage[usage["pitch_type"] == "FF"].iloc[0]
    assert ff["total"] == 2  # denominator counts only pitches with a known type
    assert ff["season_usage_rate"] == 0.5


def test_build_stratified_placebo_candidates_matches_quota_and_is_reproducible():
    treatment = pd.DataFrame({
        "season": [2021, 2021, 2022],
        "pitch_family": ["fastball", "breaking", "fastball"],
    })  # quota: (2021,fastball)=1, (2021,breaking)=1, (2022,fastball)=1
    pitches = pd.DataFrame({
        "events": ["field_out"] * 10,
        "pitch_type": ["FF", "FF", "FF", "SL", "SL", "SL", "FF", "FF", "FF", "FF"],
        "season": [2021] * 6 + [2022] * 4,
    })

    sample1 = build_stratified_placebo_candidates(pitches, treatment, {"field_out"}, seed=1)
    counts = sample1.groupby(["season", "pitch_family"]).size()
    assert counts[(2021, "fastball")] == 1
    assert counts[(2021, "breaking")] == 1
    assert counts[(2022, "fastball")] == 1

    sample2 = build_stratified_placebo_candidates(pitches, treatment, {"field_out"}, seed=1)
    pd.testing.assert_frame_equal(sample1.sort_index(), sample2.sort_index())


def test_build_stratified_placebo_candidates_takes_all_when_pool_smaller_than_quota():
    treatment = pd.DataFrame({
        "season": [2021, 2021, 2021],
        "pitch_family": ["fastball", "fastball", "fastball"],
    })  # quota: (2021, fastball) = 3
    pitches = pd.DataFrame({
        "events": ["field_out", "field_out"],
        "pitch_type": ["FF", "FF"],
        "season": [2021, 2021],
    })  # only 2 candidates available
    sample = build_stratified_placebo_candidates(pitches, treatment, {"field_out"}, seed=1)
    assert len(sample) == 2


def test_build_stratified_placebo_candidates_applies_candidate_filter_before_sampling():
    treatment = pd.DataFrame({
        "season": [2021, 2021],
        "pitch_family": ["fastball", "fastball"],
    })  # quota: (2021, fastball) = 2
    pitches = pd.DataFrame({
        "events": ["field_out"] * 8,
        "pitch_type": ["FF"] * 8,
        "season": [2021] * 8,
        "eligible": [True, False, False, False, False, False, False, True],
    })

    def only_eligible(_pitches, candidates):
        return candidates[candidates["eligible"]]

    sample = build_stratified_placebo_candidates(
        pitches, treatment, {"field_out"}, seed=1, candidate_filter=only_eligible
    )
    assert len(sample) == 2
    assert sample["eligible"].all()


def test_compute_expected_reuse_prob_formula():
    baseline = pd.Series([0.0, 0.5, 0.2])
    counts = pd.Series([3, 2, 5])
    result = compute_expected_reuse_prob(baseline, counts)
    assert result.iloc[0] == 0.0
    assert result.iloc[1] == pytest.approx(1 - 0.5**2)
    assert result.iloc[2] == pytest.approx(1 - 0.8**5)


def test_summarize_paired_diff_computes_mean_ci_and_pvalues():
    diff = pd.Series([-0.1, -0.12, -0.09, -0.11, -0.1] * 20)
    result = summarize_paired_diff(diff)
    assert result["mean"] == pytest.approx(-0.104, abs=0.001)
    assert result["ci_high"] < 0
    assert result["t_p"] < 0.001
    assert result["w_p"] < 0.001


def test_summarize_diff_in_diff_computes_net_effect_and_significant_p_value():
    d_treatment = pd.Series([0.5, 0.6, 0.55, 0.52, 0.58] * 20)  # mean 0.55
    d_control = pd.Series([0.1, 0.15, 0.12, 0.11, 0.13] * 20)  # mean 0.122
    result = summarize_diff_in_diff(d_treatment, d_control, alternative="greater")
    assert result["net_effect"] == pytest.approx(0.55 - 0.122, abs=0.001)
    assert result["ci_low"] > 0
    assert result["t_p"] < 0.001
    assert result["u_p"] < 0.001
