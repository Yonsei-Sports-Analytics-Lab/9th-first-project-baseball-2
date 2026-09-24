"""Charts for the per-pitch-type re-throw analysis (same-batter rematch).

Takes the summary table written by
src/analysis/run_pitch_type_reuse_rate_analysis.py (a DataFrame, or that CSV
via `main()`) and only draws; it has no dependency on the analysis code.
Figures go to data/processed/figures/ (git-ignored -- do not commit images).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
SUMMARY_PATH = PROCESSED_DATA_DIR / "pitch_type_reuse_rate_reduction.csv"
FIGURE_DIR = PROCESSED_DATA_DIR / "figures"

XBH_COLOR = "#D55E00"
PLACEBO_COLOR = "#0072B2"
POOLED_COLOR = "#4D4D4D"
POOLED_KEY = "ALL"
LABELS = {
    "FF": "포심 (FF)", "SI": "싱커 (SI)", "SL": "슬라이더 (SL)", "CH": "체인지업 (CH)",
    "FC": "커터 (FC)", "CU": "커브 (CU)", "ST": "스위퍼 (ST)", POOLED_KEY: "전체 (7개 구종)",
}
KOREAN_FONTS = ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic")


def _use_korean_font() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def ordered_for_plots(table: pd.DataFrame) -> pd.DataFrame:
    """Pitch types by relative reduction (largest first), pooled row last."""
    types = table[table["hit_pitch_type"] != POOLED_KEY].sort_values("relative_reduction", ascending=False)
    pooled = table[table["hit_pitch_type"] == POOLED_KEY]
    return pd.concat([types, pooled], ignore_index=True)


def _setup_rows(ax, data: pd.DataFrame) -> np.ndarray:
    y = np.arange(len(data), dtype=float)
    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(t, t) for t in data["hit_pitch_type"]])
    ax.invert_yaxis()
    if POOLED_KEY in set(data["hit_pitch_type"]):
        ax.axhline(len(data) - 1.5, color="#BBBBBB", lw=0.8, ls="--")
    ax.grid(axis="x", color="#E5E5E5", lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return y


def plot_usage_shift(table: pd.DataFrame) -> Figure:
    """Baseline usage rate (hollow) -> re-throw usage rate (filled), XBH vs placebo."""
    _use_korean_font()
    data = ordered_for_plots(table)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    y = _setup_rows(ax, data)
    for yi, row in zip(y, data.itertuples()):
        for offset, color, base, post in [
            (-0.17, XBH_COLOR, row.xbh_baseline, row.xbh_post),
            (0.17, PLACEBO_COLOR, row.placebo_baseline, row.placebo_post),
        ]:
            ax.annotate("", xy=(post, yi + offset), xytext=(base, yi + offset),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=4, shrinkB=4, mutation_scale=11))
            ax.scatter([base], [yi + offset], s=42, facecolors="white", edgecolors=color, linewidths=1.6, zorder=3)
            ax.scatter([post], [yi + offset], s=42, color=color, zorder=3)
            ax.text(max(base, post) + 0.012, yi + offset, f"{(post - base) * 100:+.1f}%p", va="center", fontsize=8.5, color=color)
    ax.set_xlim(0, 0.52)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("그 구종의 구사율 (재대결 타석에서 같은 투수가 던진 투구 중 비중)")
    ax.set_title("장타 허용 후, 같은 타자와의 재대결에서 맞은 구종의 재구사율", fontsize=13, pad=12)
    ax.legend(
        handles=[
            Line2D([], [], color=XBH_COLOR, lw=2, label="장타 허용"),
            Line2D([], [], color=PLACEBO_COLOR, lw=2, label="대조군 (field_out)"),
            Line2D([], [], marker="o", ls="", mfc="white", mec="#555555", label="사전 구사율 (시즌 평균)"),
            Line2D([], [], marker="o", ls="", color="#555555", label="재대결 구사율"),
        ],
        loc="upper right", frameon=False, fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_net_reduction(table: pd.DataFrame) -> Figure:
    """Placebo-netted drop (%p) and relative reduction (%), with 95% CIs."""
    _use_korean_font()
    data = ordered_for_plots(table)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.6))
    colors = [POOLED_COLOR if t == POOLED_KEY else XBH_COLOR for t in data["hit_pitch_type"]]
    panels = [
        (ax_left, data["net_drop"] * 100, data["net_ci_low"] * 100, data["net_ci_high"] * 100,
         "순수 감소 (%p, 대조군 보정)", "{:.1f}%p"),
        (ax_right, data["relative_reduction"] * 100, data["rel_ci_low"] * 100, data["rel_ci_high"] * 100,
         "상대 감소율 (%, 기대 재구사율 대비)", "{:.0f}%"),
    ]
    for ax, value, low, high, xlabel, fmt in panels:
        y = _setup_rows(ax, data)
        ax.barh(y, value, color=colors, height=0.55, alpha=0.9)
        ax.errorbar(value, y, xerr=[value - low, high - value], fmt="none", ecolor="#222222", elinewidth=1.2, capsize=3)
        for yi, v, hi in zip(y, value, high):
            ax.text(hi + 0.012 * max(high), yi, fmt.format(v), va="center", fontsize=9)
        ax.set_xlim(0, max(high) * 1.18)
        ax.set_xlabel(xlabel)
    ax_right.tick_params(labelleft=False)
    fig.suptitle("구종별 재구사율 감소: 대조군 보정 후 (95% 신뢰구간)", fontsize=13)
    fig.tight_layout()
    return fig


def plot_reuse_rate(table: pd.DataFrame) -> Figure:
    """Share of rematches in which the hit type was re-thrown at least once."""
    _use_korean_font()
    data = ordered_for_plots(table)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = _setup_rows(ax, data)
    for yi, row in zip(y, data.itertuples()):
        ax.plot([row.reuse_xbh, row.reuse_placebo], [yi, yi], color="#AAAAAA", lw=2, zorder=1)
        ax.scatter([row.reuse_placebo], [yi], s=70, color=PLACEBO_COLOR, zorder=3)
        ax.scatter([row.reuse_xbh], [yi], s=70, color=XBH_COLOR, zorder=3)
        ax.text(max(row.reuse_xbh, row.reuse_placebo) + 0.015, yi,
                f"{row.reuse_diff * 100:+.1f}%p [{row.reuse_diff_ci_low * 100:+.1f}, {row.reuse_diff_ci_high * 100:+.1f}]",
                va="center", fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("재대결 타석에서 같은 구종을 한 번이라도 다시 던진 비율")
    ax.set_title("구종별 재사용률: 장타 허용 vs 대조군", fontsize=13, pad=12)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color=XBH_COLOR, label="장타 허용"),
            Line2D([], [], marker="o", ls="", color=PLACEBO_COLOR, label="대조군 (field_out)"),
        ],
        loc="upper right", frameon=False,
    )
    fig.tight_layout()
    return fig


def save_all_figures(table: pd.DataFrame, out_dir: Path = FIGURE_DIR) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, plotter in [
        ("pitch_type_usage_shift", plot_usage_shift),
        ("pitch_type_net_reduction", plot_net_reduction),
        ("pitch_type_reuse_rate", plot_reuse_rate),
    ]:
        fig = plotter(table)
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    for path in save_all_figures(pd.read_csv(SUMMARY_PATH)):
        print(f"저장: {path}")


if __name__ == "__main__":
    main()
