import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.visualization.pitch_type_reuse_plots import (
    ordered_for_plots,
    plot_net_reduction,
    plot_reuse_rate,
    plot_usage_shift,
    save_all_figures,
)


def _table():
    rows = []
    for name, rel in [("FF", 0.30), ("CU", 0.62), ("ALL", 0.43)]:
        rows.append({
            "hit_pitch_type": name, "n_xbh": 100, "n_placebo": 100,
            "xbh_baseline": 0.40, "xbh_post": 0.25, "xbh_drop": 0.15,
            "placebo_baseline": 0.40, "placebo_post": 0.38, "placebo_drop": 0.02,
            "net_drop": 0.13, "net_ci_low": 0.11, "net_ci_high": 0.15,
            "relative_reduction": rel, "rel_ci_low": rel - 0.02, "rel_ci_high": rel + 0.02,
            "reuse_xbh": 0.60, "reuse_placebo": 0.75, "reuse_diff": -0.15,
            "reuse_diff_ci_low": -0.18, "reuse_diff_ci_high": -0.12,
        })
    return pd.DataFrame(rows)


def test_ordered_for_plots_sorts_types_by_relative_reduction_and_puts_the_pooled_row_last():
    ordered = ordered_for_plots(_table())
    assert ordered["hit_pitch_type"].tolist() == ["CU", "FF", "ALL"]


@pytest.mark.parametrize(
    "plotter, n_axes",
    [(plot_usage_shift, 1), (plot_net_reduction, 2), (plot_reuse_rate, 1)],
)
def test_each_plot_returns_a_figure_with_the_expected_axes(plotter, n_axes):
    fig = plotter(_table())
    assert len(fig.axes) == n_axes
    plt.close(fig)


def test_save_all_figures_writes_three_pngs(tmp_path):
    paths = save_all_figures(_table(), tmp_path)
    assert len(paths) == 3
    assert all(p.exists() and p.suffix == ".png" and p.stat().st_size > 0 for p in paths)
