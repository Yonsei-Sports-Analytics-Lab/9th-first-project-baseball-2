"""Mixed-effects logistic regression for the pitch-avoidance question,
fit with R's lme4::glmer via rpy2 (statsmodels' MixedLM does not support a
binomial/logistic response with random effects).

Pure data-prep / post-processing helpers are ordinary, unit-tested Python
functions. Anything that touches R (fitting, extracting model objects) is
integration-level and is exercised by actually running
`run_mixed_effects_model.py` against real data, not by unit tests -- not
every contributor on this project has R installed.
"""

from __future__ import annotations

import math

import pandas as pd

REQUIRED_MODEL_COLUMNS = [
    "reused_same_type",
    "baseline_usage",
    "balls",
    "strikes",
    "outs_when_up",
    "score_diff",
    "stand",
    "pitch_family",
    "season",
    "catcher_changed",
    "pitcher",
]

MODEL_FORMULA = (
    "reused_same_type ~ group * baseline_usage + balls + strikes + outs_when_up "
    "+ score_diff + stand + pitch_family + factor(season) + catcher_changed "
    "+ (1 + group | pitcher)"
)


def build_combined_model_dataset(xbh_events: pd.DataFrame, placebo_events: pd.DataFrame) -> pd.DataFrame:
    """Stack the XBH (group=1) and placebo (group=0) event datasets into one
    modeling-ready frame, dropping any row missing a required covariate
    (glmer's default na.omit would do this silently; we do it explicitly so
    the row count is reportable) and coercing `pitcher` to a string grouping
    factor.
    """
    xbh = xbh_events.assign(group=1)
    placebo = placebo_events.assign(group=0)
    combined = pd.concat([xbh, placebo], ignore_index=True)
    combined = combined.dropna(subset=REQUIRED_MODEL_COLUMNS).copy()
    combined["pitcher"] = combined["pitcher"].astype(str)
    return combined


def compute_icc(pitcher_intercept_variance: float) -> float:
    """Intraclass correlation for the random-intercept variance component of
    a logistic mixed model, using the standard latent-variable residual
    variance pi^2/3 for the binomial/logit link.
    """
    return pitcher_intercept_variance / (pitcher_intercept_variance + (math.pi**2) / 3)


def rank_pitchers_by_group_slope(
    random_effects: pd.DataFrame, pitcher_meta: pd.DataFrame, n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`random_effects`: columns [pitcher, re_intercept, re_group] (one row
    per pitcher's estimated random effects). `pitcher_meta`: columns
    [pitcher, pitcher_name, n_xbh_events]. Returns (top_n, bottom_n) by
    re_group, each with name and sample size attached.
    """
    merged = random_effects.merge(pitcher_meta, on="pitcher", how="left").sort_values("re_group")
    bottom = merged.head(n).reset_index(drop=True)
    top = merged.tail(n).sort_values("re_group", ascending=False).reset_index(drop=True)
    return top, bottom


def compute_calibration_table(observed: pd.Series, predicted: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Bin events into `n_bins` deciles of predicted probability and compare
    each bin's mean predicted probability to its mean observed outcome.
    """
    bins = pd.qcut(predicted, n_bins, duplicates="drop")
    table = pd.DataFrame({"observed": observed.to_numpy(), "predicted": predicted.to_numpy(), "bin": bins})
    return (
        table.groupby("bin", observed=True)
        .agg(n=("observed", "size"), mean_predicted=("predicted", "mean"), mean_observed=("observed", "mean"))
        .reset_index()
    )
