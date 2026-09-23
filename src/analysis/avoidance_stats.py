"""Statistical helpers for testing whether pitchers avoid re-throwing the
pitch type that was just hit for extra bases.

Three tiers of rigor, from weakest to strongest control for noise/regression
to the mean:
1. Paired comparison against the within-game baseline_usage column.
2. Paired comparison against each pitcher's season-level usage rate of the
   pitch type (much less noisy: hundreds/thousands of pitches instead of a
   handful).
3. A placebo/diff-in-diff comparison against a control outcome (e.g.
   field_out) built through the exact same next-at-bat logic as the real
   extra-base-hit events, stratified to match their (season, pitch_family)
   distribution.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

from src.preprocessing.build_next_ab_dataset import identify_events_by_outcome
from src.preprocessing.pitch_family import map_pitch_family


def compute_season_usage_rate(pitches: pd.DataFrame) -> pd.DataFrame:
    """Per (pitcher, season, pitch_type): share of that pitcher's season's
    pitches that were this pitch type -- a low-noise baseline in place of
    the within-game `baseline_usage` column. Pitches with no pitch_type are
    left out of both counts and totals (they belong to no type, and would
    otherwise give a missing-type event a bogus "usage rate" to match).
    """
    known = pitches.dropna(subset=["pitch_type"])
    counts = known.groupby(["pitcher", "season", "pitch_type"]).size().rename("n").reset_index()
    totals = known.groupby(["pitcher", "season"]).size().rename("total").reset_index()
    usage = counts.merge(totals, on=["pitcher", "season"])
    usage["season_usage_rate"] = usage["n"] / usage["total"]
    return usage


def build_stratified_placebo_candidates(
    pitches: pd.DataFrame,
    treatment_events: pd.DataFrame,
    outcome_events: set[str],
    seed: int,
    candidate_filter: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Sample pitch-level rows whose `events` is in `outcome_events` (e.g.
    {"field_out"}), stratified to match `treatment_events`' (season,
    pitch_family) distribution exactly -- a control group that differs from
    the treatment group only in whether the pitch was hit for extra bases,
    not in when it was thrown or what type it was.

    `treatment_events` must have `season` and `pitch_family` columns (rows
    with a null `pitch_family` are dropped from the quota). If a stratum's
    candidate pool is smaller than its quota, every available candidate in
    that stratum is used instead of raising.

    `candidate_filter(pitches, candidates)` optionally narrows the candidate
    pool BEFORE sampling (e.g. to same-batter-rematch-eligible rows), so the
    control group is matched on the treatment group's eligible strata rather
    than losing rows to eligibility afterwards.
    """
    quota = treatment_events.dropna(subset=["pitch_family"]).groupby(["season", "pitch_family"]).size()

    candidates = identify_events_by_outcome(pitches, outcome_events).copy()
    if candidate_filter is not None:
        candidates = candidate_filter(pitches, candidates).copy()
    candidates["pitch_family"] = candidates["pitch_type"].map(map_pitch_family)
    candidates = candidates.dropna(subset=["pitch_family"])

    rng = np.random.default_rng(seed)
    sampled_parts = []
    for (season, family), n_needed in quota.items():
        pool = candidates[(candidates["season"] == season) & (candidates["pitch_family"] == family)]
        if len(pool) <= n_needed:
            sampled_parts.append(pool)
        else:
            idx = rng.choice(pool.index.to_numpy(), size=n_needed, replace=False)
            sampled_parts.append(pool.loc[idx])

    return pd.concat(sampled_parts).copy()


def compute_expected_reuse_prob(baseline_usage: pd.Series, next_ab_pitch_count: pd.Series) -> pd.Series:
    """P(>=1 reuse) under a null of 'pitcher keeps throwing at baseline_usage
    rate, independently, on each of next_ab_pitch_count pitches'.
    """
    return 1 - (1 - baseline_usage) ** next_ab_pitch_count


def summarize_paired_diff(diff: pd.Series, alpha: float = 0.05) -> dict:
    """One-sample summary of a paired difference against 0: mean, normal-
    approximation CI, one-sample t-test, and Wilcoxon signed-rank test.
    """
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = mean - z * se, mean + z * se
    t_stat, t_p = stats.ttest_1samp(diff, 0)
    w_stat, w_p = stats.wilcoxon(diff)
    return {
        "n": n,
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "t_p": t_p,
        "w_stat": w_stat,
        "w_p": w_p,
    }


def summarize_diff_in_diff(
    d_treatment: pd.Series,
    d_control: pd.Series,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> dict:
    """Compare two independent samples of a per-event difference metric
    (e.g. baseline_usage - same_type_share), returning each group's mean,
    the net (treatment - control) effect, its normal-approximation CI, and
    a Welch t-test / Mann-Whitney U test in the given one-sided direction.
    """
    n1, n2 = len(d_treatment), len(d_control)
    m1, m2 = d_treatment.mean(), d_control.mean()
    net = m1 - m2
    se = np.sqrt(d_treatment.var(ddof=1) / n1 + d_control.var(ddof=1) / n2)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low, ci_high = net - z * se, net + z * se
    t_stat, t_p = stats.ttest_ind(d_treatment, d_control, equal_var=False, alternative=alternative)
    u_stat, u_p = stats.mannwhitneyu(d_treatment, d_control, alternative=alternative)
    return {
        "n_treatment": n1,
        "n_control": n2,
        "treatment_mean": m1,
        "control_mean": m2,
        "net_effect": net,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t_stat": t_stat,
        "t_p": t_p,
        "u_stat": u_stat,
        "u_p": u_p,
    }
