"""Figures for docs/보고서.md, written to docs/figures/ (committed, unlike data/processed/figures/).

    python -m src.visualization.report_figures

Draws the two explanatory figures (study design, basic avoidance numbers) and copies the
notebook figures the report uses. Run notebooks 03-07 first.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src.analysis.avoidance_stats import compute_season_usage_rate
from src.analysis.hit_quality import add_prior_game_usage, build_batted_ball_events
from src.analysis.robust_inference import weighted_did
from src.analysis.situation_matching import LADDER_STEPS, add_situation_columns
from src.preprocessing.research_sample import list_research_csvs

ROOT = Path(__file__).resolve().parents[2]
SRC_FIG = ROOT / "data" / "processed" / "figures"
OUT = ROOT / "docs" / "figures"
COPIED = {
    "ladder_pure_reduction.png": "04_matching_ladder.png",
    "ladder_balance_smd.png": "05_balance_smd.png",
    "ladder_group_odds_ratio.png": "06_odds_ratio.png",
    "batter_specific_by_slot.png": "07_by_slot.png",
    "batter_specific_prior.png": "08_before_after.png",
    "hit_quality_xwoba.png": "09_hit_quality_xwoba.png",
    "hit_quality_barrel.png": "10_hit_quality_barrel.png",
    "pitch_state_control.png": "11_pitch_state.png",
    "robust_pure_reduction.png": "12_robust_ci.png",
    "robust_seed_spread.png": "13_seed_spread.png",
}

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def design_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0.6, 4.6)
    ax.axis("off")
    rows = [(3.0, "장타군", "#F4C7A1", "타자 A에게\n구종 T로 장타"), (0.9, "대조군", "#C6DBEF", "타자 A'를\n구종 T로 아웃")]
    for y, label, color, first in rows:
        ax.text(0.1, y + 0.45, label, fontsize=11, fontweight="bold", va="center")
        _box(ax, 1.2, y, 1.7, 0.9, first, color)
        for i in range(3):
            _box(ax, 3.4 + i * 1.05, y + 0.15, 0.9, 0.6, ["중간 타자\n1", "…", "중간 타자\n8"][i], "#EEEEEE")
        _box(ax, 6.8, y, 1.7, 0.9, "같은 타자와\n재대결 타석", color)
        ax.annotate("", xy=(3.35, y + 0.45), xytext=(2.95, y + 0.45), arrowprops=dict(arrowstyle="->"))
        ax.annotate("", xy=(6.75, y + 0.45), xytext=(6.45, y + 0.45), arrowprops=dict(arrowstyle="->"))
        ax.text(8.7, y + 0.45, "T를 평소보다\n얼마나 덜 던졌나?", fontsize=9, va="center")
    ax.text(5.5, 2.35, "대조군은 장타군과 같은 칸(시즌, 구종계열, 주자·아웃, 이닝, 카운트, 점수차)에서 뽑은 아웃 타석",
            ha="center", fontsize=9, color="#555555")
    ax.text(5.5, 4.35, "순수 감소 = (장타군의 감소) - (대조군의 감소)", ha="center", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def basic_numbers() -> dict:
    cols = ["game_pk", "game_date", "at_bat_number", "pitch_number", "pitcher", "batter", "stand", "pitch_type",
            "events", "balls", "strikes", "outs_when_up", "inning", "inning_topbot", "on_1b", "on_2b", "on_3b",
            "home_score", "away_score", "launch_speed", "launch_angle", "estimated_woba_using_speedangle"]
    pitches = pd.concat([pd.read_csv(p, usecols=cols, encoding="utf-8-sig", low_memory=False) for p in list_research_csvs()],
                        ignore_index=True)
    pitches["season"] = pd.to_datetime(pitches["game_date"]).dt.year
    pitches = add_prior_game_usage(add_situation_columns(pitches))
    ev = build_batted_ball_events(pitches, compute_season_usage_rate(pitches), "same_batter")
    ev = ev[(ev["next_ab_pitch_count"] > 0) & ev["season_usage_rate"].notna()]
    x, c = ev[ev["outcome"].isin(["2·3루타", "홈런"])], ev[ev["outcome"] == "아웃"]
    strata = list(LADDER_STEPS["M0"])
    out = {}
    for col in ["season_usage_rate", "same_type_share"]:
        r = weighted_did(x, c, strata, value=col)
        out[col] = (r["xbh_mean"], r["ctrl_mean"])
    return out


def basic_figure(path: Path, nums: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    groups = ["장타군", "대조군 (아웃)"]
    before = [nums["season_usage_rate"][0] * 100, nums["season_usage_rate"][1] * 100]
    after = [nums["same_type_share"][0] * 100, nums["same_type_share"][1] * 100]
    xs = range(2)
    ax.bar([x - 0.18 for x in xs], before, width=0.36, color="#BBBBBB", label="평소 (그 투수의 시즌 구사율)")
    ax.bar([x + 0.18 for x in xs], after, width=0.36, color=["#D55E00", "#0072B2"], label="같은 타자와 재대결 타석")
    for x, b, a in zip(xs, before, after):
        ax.annotate(f"{b:.1f}%", (x - 0.18, b), textcoords="offset points", xytext=(0, 3), ha="center")
        ax.annotate(f"{a:.1f}%", (x + 0.18, a), textcoords="offset points", xytext=(0, 3), ha="center")
        ax.annotate(f"{a - b:+.1f}%p", (x, max(a, b) + 4), ha="center", fontsize=12)
    ax.set_xticks(list(xs), groups)
    ax.set_ylabel("구종 T의 비율 (%)")
    ax.set_ylim(0, 50)
    net = (before[0] - after[0]) - (before[1] - after[1])
    ax.text(0.5, 44, f"순수 감소 = {before[0] - after[0]:.1f} - ({before[1] - after[1]:.1f}) = {net:.1f}%p", ha="center", fontsize=11)
    ax.set_title("장타를 맞은 구종 T를 재대결에서 얼마나 던졌나 (재대결, 시즌×구종계열 매칭)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    design_figure(OUT / "02_design.png")
    nums = basic_numbers()
    basic_figure(OUT / "03_basic_avoidance.png", nums)
    for src, dst in COPIED.items():
        shutil.copyfile(SRC_FIG / src, OUT / dst)
    print({k: tuple(round(v * 100, 2) for v in vals) for k, vals in nums.items()})


if __name__ == "__main__":
    main()
