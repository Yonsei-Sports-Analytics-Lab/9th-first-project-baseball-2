"""Analysis 3: 전 비율 vs 후 비율 around the FIRST PA of a given outcome in the game (raw, no adjustments).

For each outcome (장타 = first XBH of any kind, split into HR / 2B/3B by what it was; 대조군: first single,
first strikeout, first out, each separately):
  첫 X 타석 = the pitcher's first PA in the game ending in outcome X, at his k-th PA (k >= 1; needs >= 1 PA after)
  T = final pitch type of that PA
  전 비율 = share of T among the pitcher's pitches in this game UP TO AND INCLUDING the 첫 X 타석
  후 비율 = share of T among the pitcher's pitches in this game AFTER the 첫 X 타석
  전후 변화 = 후 비율 - 전 비율
Also 경기 구사율 / 시즌 구사율 of T. The same pitcher-game can appear once per outcome group.
CIs: normal approximation (mean +- 1.96 SE).
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'

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

cols = ['game_date', 'game_pk', 'at_bat_number', 'pitch_number', 'pitcher', 'pitch_type', 'events']
df = pd.concat(
    [pd.read_csv(f, encoding='utf-8-sig', usecols=cols) for f in sorted(RAW.glob('statcast_*_research.csv'))],
    ignore_index=True,
)
df['season'] = df['game_date'].str[:4].astype(int)
df = df.dropna(subset=['pitch_type', 'pitcher'])

# ---- PAs in pitcher-game order ----
pa = (df.dropna(subset=['events'])
        .sort_values(['game_pk', 'at_bat_number', 'pitch_number'])
        .drop_duplicates(subset=['game_pk', 'at_bat_number'], keep='last')
        [['season', 'game_pk', 'at_bat_number', 'pitcher', 'events', 'pitch_type']]
        .sort_values(['game_pk', 'pitcher', 'at_bat_number'])
        .reset_index(drop=True))
pa['pa_no'] = pa.groupby(['game_pk', 'pitcher']).cumcount() + 1  # pitcher's PA number in the game
pa['n_pa'] = pa.groupby(['game_pk', 'pitcher'])['pa_no'].transform('size')
pa['group'] = pa['events'].map(GROUP)

# ---- first PA of each outcome per pitcher-game (XBH: first of any XBH, as in 첫 장타 타석) ----
first = (pa.dropna(subset=['group'])
           .assign(first_of=lambda d: np.where(d['group'].isin(XBH), 'XBH', d['group']))
           .drop_duplicates(['game_pk', 'pitcher', 'first_of'], keep='first'))
ev = first[first['pa_no'] < first['n_pa']].reset_index(drop=True)
ev['eid'] = ev.index
print("pitcher-games per 첫 X 타석 (usable):", ev['group'].value_counts().to_dict())

# ---- 전 / 후 비율 ----
p = df[['game_pk', 'pitcher', 'at_bat_number', 'pitch_type']].merge(
    ev[['eid', 'game_pk', 'pitcher', 'at_bat_number', 'pitch_type']]
    .rename(columns={'at_bat_number': 'base_ab', 'pitch_type': 'T'}), on=['game_pk', 'pitcher'])
p['side'] = np.where(p['at_bat_number'] <= p['base_ab'], 'pre', 'post')  # 첫 X 타석 belongs to 전
share = (p['pitch_type'] == p['T']).groupby([p['eid'], p['side']]).mean().unstack('side')
ev = ev.join(share, on='eid')
ev['change'] = ev['post'] - ev['pre']


# ---- 시즌 / 경기 구사율 ----
def usage(keys, name):
    return (df.groupby(keys + ['pitch_type']).size() / df.groupby(keys).size()).rename(name)


ev = ev.join(usage(['season', 'pitcher'], 'season_rate'), on=['season', 'pitcher', 'pitch_type'])
ev = ev.join(usage(['game_pk', 'pitcher'], 'game_rate'), on=['game_pk', 'pitcher', 'pitch_type'])
ev['arm'] = np.where(ev['group'].isin(XBH), '장타군', '대조군')


def ci(x):
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return round(m * 100, 2), (round((m - 1.96 * se) * 100, 2), round((m + 1.96 * se) * 100, 2))


def row(d, label):
    out = {'group': label, 'n': len(d), '시즌_%': round(d['season_rate'].mean() * 100, 2),
           '경기_%': round(d['game_rate'].mean() * 100, 2), '전_%': round(d['pre'].mean() * 100, 2),
           '후_%': round(d['post'].mean() * 100, 2), '첫타석_비중_%': round((d['pa_no'] == 1).mean() * 100, 1)}
    out['전후변화_%p'], out['ci95'] = ci(d['change'])
    out['후-경기_%p'], _ = ci(d['post'] - d['game_rate'])
    return out


groups = [(XBH, '장타군'), (['HR'], 'HR'), (['2B/3B'], '2B/3B'), (CONTROL, '대조군'),
          (['single'], 'single'), (['strikeout'], 'strikeout'), (['out'], 'out')]
summary = pd.DataFrame([row(ev[ev['group'].isin(g)], lab) for g, lab in groups] +
                       [row(ev[ev['group'].isin(g) & (ev['pitch_type'] != 'FF')], f'{lab} (T=포심 제외)')
                        for g, lab in [(XBH, '장타군'), (CONTROL, '대조군')]] +
                       [row(ev[ev['group'].isin(g) & (ev['pitch_type'] == 'FF')], f'{lab} (T=포심만)')
                        for g, lab in [(XBH, '장타군'), (CONTROL, '대조군')]])

# timing: compare groups at the same PA number of the 첫 X 타석
ev['timing'] = pd.cut(ev['pa_no'], [1, 2, 5, 10, 19, 999], labels=['1번째', '2~4번째', '5~9번째', '10~18번째', '19번째 이후'],
                      right=False)
by_timing = (ev.groupby(['timing', 'group'], observed=True)['change'].agg(['size', 'mean'])
               .assign(mean=lambda t: (t['mean'] * 100).round(2)).rename(columns={'size': 'n', 'mean': '전후변화_%p'})
               .unstack('group'))
by_timing_arm = (ev.groupby(['timing', 'arm'], observed=True)['change'].mean().unstack('arm') * 100)
by_timing_arm['장타군-대조군'] = by_timing_arm['장타군'] - by_timing_arm['대조군']
by_timing_arm = by_timing_arm.round(2)

by_type = (ev.groupby(['pitch_type', 'arm'])['change'].agg(['size', 'mean']).unstack('arm'))
by_type.columns = [f'{b}_{a}' for a, b in by_type.columns]
by_type = by_type[by_type['장타군_size'] >= 300].sort_values('장타군_size', ascending=False)
by_type[['장타군_mean', '대조군_mean']] *= 100
by_type['장타군-대조군'] = by_type['장타군_mean'] - by_type['대조군_mean']
by_type = by_type.round(2)

by_season = ev.groupby(['season', 'arm'])['change'].mean().unstack('arm') * 100
by_season['장타군-대조군'] = by_season['장타군'] - by_season['대조군']
by_season = by_season.round(2)

ev.drop(columns='eid').to_csv(PROC / 'prepost_results.csv', index=False)
summary.to_csv(PROC / 'prepost_summary.csv', index=False)
by_timing.to_csv(PROC / 'prepost_by_timing.csv')
by_timing_arm.to_csv(PROC / 'prepost_by_timing_arm.csv')
by_type.to_csv(PROC / 'prepost_by_pitch_type.csv')
by_season.to_csv(PROC / 'prepost_by_season.csv')

pd.set_option('display.width', 250)
print("\n-- 시즌 / 경기 / 전 / 후 비율 --")
print(summary.to_string(index=False))
print("\n-- 전후 변화 by 첫 X 타석 순번 (group) --")
print(by_timing.to_string())
print("\n-- 전후 변화 by 첫 X 타석 순번 (장타군 vs 대조군) --")
print(by_timing_arm.to_string())
print("\n-- by T (장타군 n >= 300) --")
print(by_type.to_string())
print("\n-- by season --")
print(by_season.to_string())
