"""Analyses 5 and 6: three usage rates of T, per 기준 타석 (raw, no adjustments).

  python 04_rates.py        analysis 5: 시즌 / 경기 구사율 over ALL batters
  python 04_rates.py hand   analysis 6: 시즌 / 경기 구사율 over batters of the SAME hand (stand L/R)
                            as the 기준 타석 batter

  시즌 구사율        = share of T among the pitcher's pitches that season (all batters, or same-hand batters)
  경기 구사율        = share of T among the pitcher's pitches in that game (all batters, or same-hand batters)
  이후 타자 구사율   = share of T among ALL pitches the pitcher threw to the SAME batter
                       AFTER the 기준 타석, in the same game (every later PA vs that batter, pooled)
Cases: 기준 타석 with >= 1 later PA vs the same batter in the game (same cases as analysis 1).
Same three rates for 장타군 and 대조군 (out/strikeout/single).
CIs: normal approximation (mean +- 1.96 SE).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'
HAND = len(sys.argv) > 1 and sys.argv[1] == 'hand'
OUT = 'rates_hand' if HAND else 'rates'
HK = ['stand'] if HAND else []  # extra grouping key for 시즌 / 경기 구사율

GROUP = {
    'home_run': 'HR', 'double': '2B/3B', 'triple': '2B/3B',
    'single': 'single',
    'strikeout': 'strikeout', 'strikeout_double_play': 'strikeout',
    'field_out': 'out', 'force_out': 'out', 'grounded_into_double_play': 'out', 'double_play': 'out',
    'fielders_choice_out': 'out', 'sac_fly': 'out', 'sac_bunt': 'out', 'sac_fly_double_play': 'out',
    'triple_play': 'out',
}
XBH = ['HR', '2B/3B']
CONTROL = ['out', 'strikeout', 'single']

cols = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'batter', 'stand', 'pitch_type', 'events']
df = pd.concat(
    [pd.read_csv(f, encoding='utf-8-sig', usecols=cols) for f in sorted(RAW.glob('statcast_*_research.csv'))],
    ignore_index=True,
)
df['season'] = df['game_date'].str[:4].astype(int)
df = df.dropna(subset=['pitch_type', 'pitcher', 'batter', 'stand'])

# ---- 기준 타석 (T = final pitch type) with a later PA vs the same batter ----
pa = (df.dropna(subset=['events'])
        .sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        .drop_duplicates(subset=['game_pk', 'at_bat_number'], keep='last')
        [['season', 'game_pk', 'at_bat_number', 'pitcher', 'batter', 'stand', 'events', 'pitch_type']])
pa['group'] = pa['events'].map(GROUP)
pa = pa[pa['group'].notna()]
last_ab = df.groupby(['game_pk', 'pitcher', 'batter'])['at_bat_number'].max().rename('last_ab')
ev = pa.join(last_ab, on=['game_pk', 'pitcher', 'batter'])
ev = ev[ev['at_bat_number'] < ev['last_ab']].drop(columns='last_ab').reset_index(drop=True)

# ---- 시즌 / 경기 구사율 ----
def rate(keys, name):
    return (df.groupby(keys + ['pitch_type']).size() / df.groupby(keys).size()).rename(name)

ev = ev.join(rate(['season', 'pitcher'] + HK, 'season_rate'), on=['season', 'pitcher'] + HK + ['pitch_type'])
ev = ev.join(rate(['game_pk', 'pitcher'] + HK, 'game_rate'), on=['game_pk', 'pitcher'] + HK + ['pitch_type'])

# ---- 이후 타자 구사율: all later pitches to the same batter in the game ----
ev['eid'] = ev.index
m = ev[['eid', 'game_pk', 'pitcher', 'batter', 'at_bat_number', 'pitch_type']].rename(
    columns={'at_bat_number': 'base_ab', 'pitch_type': 'T'}).merge(
    df[['game_pk', 'pitcher', 'batter', 'at_bat_number', 'pitch_type']], on=['game_pk', 'pitcher', 'batter'])
m = m[m['at_bat_number'] > m['base_ab']]
after = m.assign(is_t=m['pitch_type'] == m['T']).groupby('eid').agg(after_rate=('is_t', 'mean'),
                                                                    after_pitches=('is_t', 'size'))
ev = ev.join(after, on='eid')
ev['after_minus_game'] = ev['after_rate'] - ev['game_rate']
ev['after_minus_season'] = ev['after_rate'] - ev['season_rate']
ev['game_minus_season'] = ev['game_rate'] - ev['season_rate']


def ci(x):
    m_, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return round(m_ * 100, 2), (round((m_ - 1.96 * se) * 100, 2), round((m_ + 1.96 * se) * 100, 2))


def summarize(d, label):
    out = {'group': label, 'n': len(d)}
    for c in ['season_rate', 'game_rate', 'after_rate']:
        out[c + '_%'] = round(d[c].mean() * 100, 2)
    for c in ['after_minus_game', 'after_minus_season', 'game_minus_season']:
        out[c + '_%p'], out[c + '_ci95'] = ci(d[c])
    return out


x = ev[ev['group'].isin(XBH)]
c = ev[ev['group'].isin(CONTROL)]
summary = pd.DataFrame([
    summarize(x, '장타군'), summarize(ev[ev['group'] == 'HR'], 'HR'), summarize(ev[ev['group'] == '2B/3B'], '2B/3B'),
    summarize(x[x['pitch_type'] != 'FF'], '장타군 (T=포심 제외)'), summarize(x[x['pitch_type'] == 'FF'], '장타군 (T=포심만)'),
    summarize(c, '대조군'), summarize(ev[ev['group'] == 'single'], 'single'),
    summarize(ev[ev['group'] == 'out'], 'out'), summarize(ev[ev['group'] == 'strikeout'], 'strikeout'),
    summarize(c[c['pitch_type'] != 'FF'], '대조군 (T=포심 제외)'), summarize(c[c['pitch_type'] == 'FF'], '대조군 (T=포심만)'),
    summarize(x[x['stand'] == 'L'], '장타군 (좌타자)'), summarize(x[x['stand'] == 'R'], '장타군 (우타자)'),
    summarize(c[c['stand'] == 'L'], '대조군 (좌타자)'), summarize(c[c['stand'] == 'R'], '대조군 (우타자)'),
])


def by(d, key, min_n=0):
    t = d.groupby(key)[['season_rate', 'game_rate', 'after_rate']].mean() * 100
    t.insert(0, 'n', d.groupby(key).size())
    t = t[t['n'] >= min_n].sort_values('n', ascending=False) if min_n else t
    t['after_minus_game'] = t['after_rate'] - t['game_rate']
    t['after_minus_season'] = t['after_rate'] - t['season_rate']
    return t.round(2)


by_type = pd.concat({'장타군': by(x, 'pitch_type', 300), '대조군': by(c, 'pitch_type', 300)}, names=['arm'])
by_season = pd.concat({'장타군': by(x, 'season'), '대조군': by(c, 'season')}, names=['arm'])

ev.drop(columns='eid').to_csv(PROC / f'{OUT}_results.csv', index=False)
summary.to_csv(PROC / f'{OUT}_summary.csv', index=False)
by_type.to_csv(PROC / f'{OUT}_by_pitch_type.csv')
by_season.to_csv(PROC / f'{OUT}_by_season.csv')

pd.set_option('display.width', 250)
print(f"[{OUT}] cases: {len(ev)}  장타군: {len(x)}  median later pitches to that batter (장타군): {x.after_pitches.median():.0f}\n")
print(summary.to_string(index=False))
print("\n-- by T (n >= 300) --")
print(by_type.to_string())
print("\n-- by season --")
print(by_season.to_string())
