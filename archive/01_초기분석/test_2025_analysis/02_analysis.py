"""측정 타석 T 비율: share of the 기준 타석's final pitch type T in ONE later PA of the same game.

  python 02_analysis.py rematch   측정 타석 = first later PA vs. the SAME batter (analysis 1, default)
  python 02_analysis.py next      측정 타석 = the pitcher's very next PA, i.e. next batter (analysis 2)

Raw comparison only: 측정 타석 T 비율 by 기준 타석 outcome (장타군 vs 대조군), no baselines or adjustments.
CIs: normal approximation (mean +- 1.96 SE).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else 'rematch'
assert MODE in ('rematch', 'next')

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'

GROUP = {
    'home_run': 'HR', 'double': '2B/3B', 'triple': '2B/3B',
    'single': 'single',
    'strikeout': 'strikeout', 'strikeout_double_play': 'strikeout',
    'field_out': 'out', 'force_out': 'out', 'grounded_into_double_play': 'out', 'double_play': 'out',
    'fielders_choice_out': 'out', 'sac_fly': 'out', 'sac_bunt': 'out', 'sac_fly_double_play': 'out',
    'triple_play': 'out',
    'walk': 'walk/HBP', 'hit_by_pitch': 'walk/HBP',
}
XBH = ['HR', '2B/3B']
CONTROL = ['out', 'strikeout', 'single']

# fixed dataset: data/raw/statcast_*_research.csv (see data/raw/README.md)
cols = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter', 'pitch_type', 'events']
df = pd.concat(
    [pd.read_csv(f, encoding='utf-8-sig', usecols=cols) for f in sorted(RAW.glob('statcast_*_research*.csv'))],
    ignore_index=True,
)
df['season'] = df['game_date'].str[:4].astype(int)
df = df.dropna(subset=['pitch_type', 'pitcher', 'batter'])
df = df.sort_values(['game_pk', 'at_bat_number', 'pitch_number']).reset_index(drop=True)

# ---- 기준 타석 table: one row per PA, T = final pitch type ----
pa = (df.dropna(subset=['events'])
        .drop_duplicates(subset=['game_pk', 'at_bat_number'], keep='last')
        [['season', 'game_pk', 'at_bat_number', 'pitcher', 'batter', 'events', 'pitch_type']])

# ---- 측정 타석: rematch = next PA of the same pitcher-batter pair; next = pitcher's next PA ----
grp = ['game_pk', 'pitcher', 'batter'] if MODE == 'rematch' else ['game_pk', 'pitcher']
res = pa.sort_values(grp + ['at_bat_number'])
nxt = res.groupby(grp)[['at_bat_number', 'batter']].shift(-1)
res = res.assign(next_ab=nxt['at_bat_number'], next_batter=nxt['batter'])
res = res.dropna(subset=['next_ab']).sort_values(['game_pk', 'at_bat_number']).copy()
res[['next_ab', 'next_batter']] = res[['next_ab', 'next_batter']].astype(int)

# 측정 타석 T 비율 = T pitches / all pitches in the 측정 타석
pa_n = df.groupby(['game_pk', 'at_bat_number']).size().rename('n_pitches')
pa_t = df.groupby(['game_pk', 'at_bat_number', 'pitch_type']).size().rename('n_t')
res = res.join(pa_n, on=['game_pk', 'next_ab']).join(pa_t, on=['game_pk', 'next_ab', 'pitch_type'])
res['n_t'] = res['n_t'].fillna(0).astype(int)
res['t_rate'] = res['n_t'] / res['n_pitches']
res['group'] = res['events'].map(GROUP).fillna('other')

# 시즌 구사율 / 경기 구사율 of T for the pitcher (all his pitches that season / that game)
def usage(keys, name):
    return (df.groupby(keys + ['pitch_type']).size() / df.groupby(keys).size()).rename(name)

res = res.join(usage(['season', 'pitcher'], 'season_rate'), on=['season', 'pitcher', 'pitch_type'])
res = res.join(usage(['game_pk', 'pitcher'], 'game_rate'), on=['game_pk', 'pitcher', 'pitch_type'])
res.to_csv(PROC / f'{MODE}_results.csv', index=False)
print(f"[{MODE}] 기준 타석 with a 측정 타석: {len(res)}  (장타군: {res['group'].isin(XBH).sum()})\n")


def mean_ci(x):
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return round(m * 100, 2), (round((m - 1.96 * se) * 100, 2), round((m + 1.96 * se) * 100, 2))


def diff_ci(a, b):
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return round(d * 100, 2), (round((d - 1.96 * se) * 100, 2), round((d + 1.96 * se) * 100, 2))


r = res.set_index('group')['t_rate']
ff = res.set_index('group')['pitch_type'].eq('FF')
xbh, ctrl = r.loc[XBH], r.loc[CONTROL]

# ---- 1. 측정 타석 T 비율 by 기준 타석 outcome ----
rows = []
for g, lab in [(XBH, '장타군'), (['HR'], 'HR'), (['2B/3B'], '2B/3B'), (['single'], 'single'),
               (['out'], 'out'), (['strikeout'], 'strikeout'), (CONTROL, '대조군'), (['walk/HBP'], 'walk/HBP')]:
    m, ci = mean_ci(r.loc[g])
    rows.append({'기준 타석 결과': lab, 'n': len(r.loc[g]), 'T비율_%': m, 'ci95': ci,
                 'T비율_%_포심제외': round(r.loc[g][~ff.loc[g]].mean() * 100, 2)})
by_outcome = pd.DataFrame(rows)

# ---- 2. differences (percentage points), incl. four-seam exclusion ----
comps = [('장타군 - 대조군', xbh, ctrl), ('HR - 대조군', r.loc[['HR']], ctrl),
         ('2B/3B - 대조군', r.loc[['2B/3B']], ctrl), ('장타군 - 단타', xbh, r.loc[['single']]),
         ('장타군 - 대조군 (T=포심 제외)', xbh[~ff.loc[XBH]], ctrl[~ff.loc[CONTROL]]),
         ('장타군 - 대조군 (T=포심만)', xbh[ff.loc[XBH]], ctrl[ff.loc[CONTROL]])]
compare = pd.DataFrame([{'비교': k, '차이_%p': diff_ci(a, b)[0], 'ci95': diff_ci(a, b)[1], 'n_앞쪽': len(a)}
                        for k, a, b in comps])

# ---- 3. by T and by season ----
arm = res[res['group'].isin(XBH + CONTROL)].assign(arm=lambda d: np.where(d['group'].isin(XBH), '장타군', '대조군'))
by_type = arm.groupby(['pitch_type', 'arm'])['t_rate'].agg(['size', 'mean']).unstack('arm')
by_type.columns = [f'{b}_{a}' for a, b in by_type.columns]
by_type = by_type[by_type['장타군_size'] >= 300].sort_values('장타군_size', ascending=False)
by_type[['장타군_mean', '대조군_mean']] *= 100
by_type['차이_%p'] = by_type['장타군_mean'] - by_type['대조군_mean']
by_type = by_type.round(2)

by_season = arm.groupby(['season', 'arm'])['t_rate'].mean().unstack('arm') * 100
by_season['차이_%p'] = by_season['장타군'] - by_season['대조군']
by_season = by_season.round(2)

# ---- 4. three rates: 시즌 구사율, 경기 구사율, 측정 타석 T 비율 ----
def rates_row(d, label):
    out = {'기준 타석 결과': label, 'n': len(d)}
    for c, lab in [('season_rate', '시즌'), ('game_rate', '경기'), ('t_rate', '측정')]:
        out[f'{lab}_%'] = round(d[c].mean() * 100, 2)
    for a, b, lab in [('t_rate', 'game_rate', '측정-경기'), ('t_rate', 'season_rate', '측정-시즌'),
                      ('game_rate', 'season_rate', '경기-시즌')]:
        out[f'{lab}_%p'], out[f'{lab}_ci95'] = mean_ci(d[a] - d[b])
    return out


rd = res.set_index('group')
rates = pd.DataFrame([rates_row(rd.loc[g], lab) for g, lab in
                      [(XBH, '장타군'), (['HR'], 'HR'), (['2B/3B'], '2B/3B'), (CONTROL, '대조군'),
                       (['single'], 'single'), (['out'], 'out'), (['strikeout'], 'strikeout')]] +
                     [rates_row(rd.loc[g][rd.loc[g, 'pitch_type'] != 'FF'], f'{lab} (T=포심 제외)')
                      for g, lab in [(XBH, '장타군'), (CONTROL, '대조군')]] +
                     [rates_row(rd.loc[g][rd.loc[g, 'pitch_type'] == 'FF'], f'{lab} (T=포심만)')
                      for g, lab in [(XBH, '장타군'), (CONTROL, '대조군')]])
rates.to_csv(PROC / f'{MODE}_rates.csv', index=False)

by_outcome.to_csv(PROC / f'{MODE}_by_outcome.csv', index=False)
compare.to_csv(PROC / f'{MODE}_compare.csv', index=False)
by_type.to_csv(PROC / f'{MODE}_by_pitch_type.csv')
by_season.to_csv(PROC / f'{MODE}_by_season.csv')

pd.set_option('display.width', 200)
print("-- 측정 타석 T 비율 by 기준 타석 결과 (%) --")
print(by_outcome.to_string(index=False))
print("\n-- 비교 (%p) --")
print(compare.to_string(index=False))
print("\n-- by T (장타군 n >= 300) --")
print(by_type.to_string())
print("\n-- by season --")
print(by_season.to_string())
print("\n-- 시즌 / 경기 / 측정 타석 T 비율 --")
print(rates.to_string(index=False))
