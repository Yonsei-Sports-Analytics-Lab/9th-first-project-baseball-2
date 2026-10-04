"""Intensive margin analysis (main analysis: same-batter rematch).

Among events where the pitcher DID re-throw the hit pitch type, does he
execute it differently -- location, movement, velocity -- measured as the
deviation from that pitcher's own usual execution of the type (see
intensive_margin.py)? One independent linear mixed model per outcome:

  literal      XBH only:  y ~ time + balls + strikes + score_diff + platoon_match
                          + baseline_usage + (1 + time | pitcher)
  xbh_event_re XBH only:  the same plus (1 | event_id) -- pre and post pitches
                          of one event are not independent
  did          XBH + field_out placebo: y ~ group * time + ... + (1 | event_id);
                          group:time is the placebo-controlled change. The
                          placebo is an addition to the requested spec: the
                          event pitch and the events themselves are selected,
                          so a plain pre/post change can be regression to the
                          mean rather than a reaction to being hit.

If (1 + time | pitcher) is singular it is refit as (0 + time | pitcher), and
the two fits' per-pitcher time slopes are correlated as a robustness check.
Per-pitcher random slopes of the literal model are saved for the case studies.

Needs R (lme4) and rpy2. Uses the event parquet files written by
run_same_batter_rematch_analysis.py when present (so the events are exactly
the main analysis' events), otherwise rebuilds them.
"""

import logging
from pathlib import Path

import pandas as pd

from src.analysis.avoidance_stats import compute_season_usage_rate
from src.analysis.glmer_runner import (
    check_convergence,
    extract_lmer_fixed_effects,
    extract_ranef_table,
    fit_lmer,
)
from src.analysis.intensive_margin import (
    OUTCOMES,
    PITCH_QUALITY_COLUMNS,
    add_execution_deviations,
    collect_pitch_observations,
    compute_execution_baselines,
)
from src.analysis.mixed_effects_model import compare_random_effects
from src.analysis.run_same_batter_rematch_analysis import (
    PLACEBO_REMATCH_PATH,
    XBH_REMATCH_PATH,
    build_placebo_rematch,
    build_xbh_rematch,
    load_pitches,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
FIXED_EFFECTS_PATH = PROCESSED_DATA_DIR / "intensive_margin_fixed_effects.csv"
OBSERVATIONS_PATH = PROCESSED_DATA_DIR / "intensive_margin_pitch_observations.parquet"
MIN_BASELINE_PITCHES = 20
MIN_EVENTS_FOR_RANKING = 10
COVARIATES = "balls + strikes + score_diff + platoon_match + baseline_usage"
MODEL_COLUMNS = ["balls", "strikes", "score_diff", "platoon_match", "baseline_usage", "time", "group", "pitcher", "event_id"]
EVENT_COLUMNS = ["game_pk", "at_bat_number", "event_pitch_number", "next_at_bat_number", "platoon_match"]


def load_or_build_events(pitches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if XBH_REMATCH_PATH.exists() and PLACEBO_REMATCH_PATH.exists():
        xbh, placebo = pd.read_parquet(XBH_REMATCH_PATH), pd.read_parquet(PLACEBO_REMATCH_PATH)
        if all(c in xbh.columns for c in EVENT_COLUMNS) and all(c in placebo.columns for c in EVENT_COLUMNS):
            logger.info("저장된 재대결 이벤트 사용: 장타 %d건, 대조군 %d건", len(xbh), len(placebo))
            return xbh, placebo
    logger.info("저장된 이벤트가 없거나 구버전 -- 재대결 이벤트를 새로 생성")
    season_usage = compute_season_usage_rate(pitches)
    xbh = build_xbh_rematch(pitches, season_usage)
    return xbh, build_placebo_rematch(pitches, xbh, season_usage)


def build_observations(pitches: pd.DataFrame, xbh: pd.DataFrame, placebo: pd.DataFrame) -> pd.DataFrame:
    baselines = compute_execution_baselines(pitches, min_pitches=MIN_BASELINE_PITCHES)
    obs = pd.concat(
        [collect_pitch_observations(pitches, xbh, 1), collect_pitch_observations(pitches, placebo, 0)],
        ignore_index=True,
    )
    obs = add_execution_deviations(obs, baselines)
    obs["pitcher"] = obs["pitcher"].astype(str)
    return obs


def fit_with_fallback(data: pd.DataFrame, outcome: str, template: str) -> dict:
    """`template` holds a `{pitcher_re}` placeholder. Fits (1 + time | pitcher);
    if that is singular, also fits (0 + time | pitcher), correlates the
    per-pitcher time slopes of the two fits, and reports the simplified one.
    """
    formula = template.format(pitcher_re="(1 + time | pitcher)")
    fit_lmer(data, formula)
    singular_note = check_convergence()
    is_singular = "isSingular=True" in singular_note
    original = {"fe": extract_lmer_fixed_effects(), "ranef": extract_ranef_table("pitcher")}
    result = {"structure": "(1 + time | pitcher)", "formula": formula, "convergence": singular_note, **original,
              "singular_original": is_singular, "ranef_corr": None}
    if is_singular:
        simple_formula = template.format(pitcher_re="(0 + time | pitcher)")
        fit_lmer(data, simple_formula)
        simple = {"fe": extract_lmer_fixed_effects(), "ranef": extract_ranef_table("pitcher")}
        result.update(
            structure="(0 + time | pitcher)", formula=simple_formula, convergence=check_convergence(), **simple,
            ranef_corr=compare_random_effects(original["ranef"], simple["ranef"], column="re_time"),
        )
    return result


def run() -> None:
    pitches = load_pitches(tuple(PITCH_QUALITY_COLUMNS))
    xbh, placebo = load_or_build_events(pitches)

    obs = build_observations(pitches, xbh, placebo)
    obs.to_parquet(OBSERVATIONS_PATH, index=False)
    logger.info(
        "관측 %d행 (이벤트 %d건): 장타 pre %d / post %d, 대조군 pre %d / post %d",
        len(obs), obs["event_id"].nunique(),
        int(((obs.group == 1) & (obs.time == 0)).sum()), int(((obs.group == 1) & (obs.time == 1)).sum()),
        int(((obs.group == 0) & (obs.time == 0)).sum()), int(((obs.group == 0) & (obs.time == 1)).sum()),
    )
    logger.info(
        "그룹×시점별 평균 편차 (raw, 대조군 없는 단순 비교)\n%s",
        obs.groupby(["group", "time"])[list(OUTCOMES)].mean().round(4).to_string(),
    )

    rows = []
    for outcome in OUTCOMES:
        data = obs.dropna(subset=[outcome] + MODEL_COLUMNS).copy()
        xbh_only = data[data["group"] == 1]
        specs = {
            "literal": (xbh_only, f"{outcome} ~ time + {COVARIATES} + {{pitcher_re}}"),
            "xbh_event_re": (xbh_only, f"{outcome} ~ time + {COVARIATES} + {{pitcher_re}} + (1 | event_id)"),
            "did": (data, f"{outcome} ~ group * time + {COVARIATES} + {{pitcher_re}} + (1 | event_id)"),
        }
        for spec, (frame, template) in specs.items():
            res = fit_with_fallback(frame, outcome, template)
            terms = ["time"] if spec != "did" else ["time", "group", "group:time"]
            view = res["fe"][res["fe"]["term"].isin(terms)][["term", "estimate", "ci_low", "ci_high", "p_value"]]
            logger.info(
                "[%s / %s] n=%d행, 이벤트 %d건, 투수 %d명 | 랜덤효과 %s (원 구조 singular=%s) | %s\n%s",
                outcome, spec, len(frame), frame["event_id"].nunique(), frame["pitcher"].nunique(),
                res["structure"], res["singular_original"], res["convergence"], view.round(4).to_string(index=False),
            )
            if res["ranef_corr"]:
                c = res["ranef_corr"]
                logger.info(
                    "    투수별 time 기울기 상관 (1+time vs 0+time, 공통 %d명): Pearson=%.4f Spearman=%.4f",
                    c["n_common"], c["pearson"], c["spearman"],
                )
            rows.append(
                res["fe"].assign(
                    outcome=outcome, spec=spec, structure=res["structure"], n_rows=len(frame),
                    n_events=frame["event_id"].nunique(), n_pitchers=frame["pitcher"].nunique(),
                )
            )
            if spec == "literal":
                save_random_effects(outcome, res["ranef"], xbh, xbh_only)
    pd.concat(rows, ignore_index=True).to_csv(FIXED_EFFECTS_PATH, index=False)
    logger.info("고정효과 전체 저장: %s", FIXED_EFFECTS_PATH)


def save_random_effects(outcome: str, ranef: pd.DataFrame, xbh_events: pd.DataFrame, xbh_obs: pd.DataFrame) -> None:
    names = xbh_events[["pitcher", "pitcher_name"]].drop_duplicates("pitcher").assign(pitcher=lambda d: d["pitcher"].astype(str))
    counts = xbh_obs.groupby("pitcher")["event_id"].nunique().rename("n_events").reset_index()
    table = ranef.merge(names, on="pitcher", how="left").merge(counts, on="pitcher", how="left")
    table = table.sort_values("re_time")
    full_path = PROCESSED_DATA_DIR / f"intensive_margin_random_effects_{outcome}.csv"
    table.to_csv(full_path, index=False)
    eligible = table[table["n_events"] >= MIN_EVENTS_FOR_RANKING]
    extremes = pd.concat(
        [eligible.head(10).assign(rank_group="bottom10_re_time"), eligible.tail(10).assign(rank_group="top10_re_time")]
    )
    extremes_path = PROCESSED_DATA_DIR / f"intensive_margin_random_effects_{outcome}_extremes.csv"
    extremes.to_csv(extremes_path, index=False)
    logger.info("    [%s] 투수별 랜덤효과 저장: %s (%d명), 상/하위 10명(이벤트 >=%d): %s",
                outcome, full_path.name, len(table), MIN_EVENTS_FOR_RANKING, extremes_path.name)


if __name__ == "__main__":
    run()
