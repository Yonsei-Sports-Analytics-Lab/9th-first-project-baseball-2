"""Shared rpy2/lme4::glmer plumbing, factored out so both a single model
run and a multi-model comparison can reuse the exact same fit/extract
logic without duplication.

Not unit tested (R/lme4/rpy2 integration-level, not every contributor has
R installed) -- exercised by actually running the `run_*.py` scripts in
this package against real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

R_CONVERTER = ro.default_converter + numpy2ri.converter + pandas2ri.converter


def fit_glmer(combined: pd.DataFrame, formula: str):
    """Runs the fit as a plain R code string (rather than calling glmer()
    with Python kwargs) because passing an R glmerControl() object through
    rpy2's argument conversion strips its S3 class and breaks glmer's
    internal dispatch -- building the call in R avoids that entirely.
    """
    ro.r("library(lme4)")
    with R_CONVERTER.context():
        r_df = ro.conversion.get_conversion().py2rpy(combined)
        ro.globalenv["model_data"] = r_df
    ro.r(f"""
    model <- glmer(
        {formula},
        data = model_data,
        family = binomial,
        nAGQ = 0,
        control = glmerControl(optimizer = "bobyqa")
    )
    """)
    return ro.globalenv["model"]


def extract_fixed_effects_table() -> pd.DataFrame:
    base = importr("base")
    stats_r = importr("stats")
    coefs = stats_r.coef(base.summary(ro.globalenv["model"]))
    with R_CONVERTER.context():
        df = ro.conversion.get_conversion().rpy2py(base.as_data_frame(coefs))
    df = df.reset_index().rename(
        columns={
            "index": "term",
            "Estimate": "estimate",
            "Std. Error": "std_error",
            "z value": "z_value",
            "Pr(>|z|)": "p_value",
        }
    )
    df["odds_ratio"] = np.exp(df["estimate"])
    return df


def check_convergence() -> str:
    msg = ro.r("if (is.null(model@optinfo$conv$lme4$messages)) 'OK: no convergence warnings' "
               "else paste(model@optinfo$conv$lme4$messages, collapse='; ')")
    singular = bool(ro.r("isSingular(model)")[0])
    return f"{msg[0]}; isSingular={singular}"


def extract_pitcher_variance_components() -> tuple[float | None, float, float]:
    """Returns (intercept_variance, group_slope_variance, correlation).
    intercept_variance is None when the model has no random intercept term
    (e.g. a `(0 + group | pitcher)` random-effects structure).
    """
    pitcher_vc = ro.r("as.data.frame(VarCorr(model))")
    with R_CONVERTER.context():
        vc_df = ro.conversion.get_conversion().rpy2py(pitcher_vc)
    intercept_rows = vc_df.loc[
        (vc_df["grp"] == "pitcher") & (vc_df["var1"] == "(Intercept)") & vc_df["var2"].isna(), "vcov"
    ]
    intercept_var = float(intercept_rows.iloc[0]) if len(intercept_rows) else None
    group_var = float(
        vc_df.loc[(vc_df["grp"] == "pitcher") & (vc_df["var1"] == "group") & vc_df["var2"].isna(), "vcov"].iloc[0]
    )
    corr_row = vc_df.loc[(vc_df["grp"] == "pitcher") & (vc_df["var2"] == "group")]
    corr = float(corr_row["sdcor"].iloc[0]) if len(corr_row) else float("nan")
    return intercept_var, group_var, corr


def extract_random_effects_table() -> pd.DataFrame:
    pitcher_ranef = ro.r("as.data.frame(ranef(model)$pitcher)")
    with R_CONVERTER.context():
        df = ro.conversion.get_conversion().rpy2py(pitcher_ranef)
    df = df.reset_index().rename(columns={"index": "pitcher", "(Intercept)": "re_intercept", "group": "re_group"})
    return df


def extract_predictions() -> np.ndarray:
    preds = ro.r('predict(model, type="response")')
    with R_CONVERTER.context():
        arr = ro.conversion.get_conversion().rpy2py(preds)
    return np.asarray(arr)
