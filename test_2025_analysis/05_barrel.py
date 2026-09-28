"""Analysis 7: changes after a BARREL (raw, no adjustments).

Barrel (baseballr approximation of the Statcast definition), on the 기준 타석's final pitch:
  launch_speed >= 98, launch_angle <= 50, 1.5*speed - angle >= 117, speed + angle >= 124

Reuses per-case outputs of earlier scripts (run them first) and splits 기준 타석 by barrel x outcome:
  ① rematch_results.csv    측정 타석 T 비율 vs the same batter (first rematch PA)
  ② next_results.csv       측정 타석 T 비율 vs the next batter, and 측정 - 경기 구사율
  ⑤ rates_results.csv      이후 타자 구사율 - 경기 / 시즌 구사율 (all batters)
  ⑥ rates_hand_results.csv 이후 타자 구사율 - 좌우 경기 / 좌우 시즌 구사율 (same-hand batters)
CIs: normal approximation (mean +- 1.96 SE).
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROC = ROOT / 'data' / 'processed'

# ---- barrel flag of each PA's final pitch ----
cols = ['game_pk', 'at_bat_number', 'pitch_number', 'events', 'launch_speed', 'launch_angle']
df = pd.concat([pd.read_csv(f, encoding='utf-8-sig', usecols=cols) for f in sorted(RAW.glob('statcast_*_research.csv'))],
               ignore_index=True)
last = (df.dropna(subset=['events']).sort_values('pitch_number')
          .drop_duplicates(['game_pk', 'at_bat_number'], keep='last'))
v, a = last['launch_speed'], last['launch_angle']
last['barrel'] = (v >= 98) & (a <= 50) & (1.5 * v - a >= 117) & (v + a >= 124)
last['batted'] = v.notna() & a.notna()
flag = last.set_index(['game_pk', 'at_bat_number'])[['barrel', 'batted', 'launch_speed', 'launch_angle']]

OUTCOME = {'HR': '홈런', '2B/3B': '2·3루타', 'single': '단타', 'out': '아웃'}


def label(d):
    kind = np.where(d['barrel'], '배럴', np.where(d['batted'], '비배럴', '타구 없음'))
    return pd.Series(kind, index=d.index) + ' ' + d['group'].map(OUTCOME).fillna(d['group'])


def load(name):
    d = pd.read_csv(PROC / f'{name}.csv').join(flag, on=['game_pk', 'at_bat_number'])
    d = d[d['group'].isin(OUTCOME) | (d['group'] == 'strikeout')].copy()
    d['cat'] = label(d)
    d.loc[d['group'] == 'strikeout', 'cat'] = '삼진'
    d['barrel_any'] = np.where(d['barrel'] & d['batted'], '배럴 전체', np.where(d['batted'], '비배럴 타구 전체', '삼진'))
    return d


def ci(x):
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return round(m * 100, 2), (round((m - 1.96 * se) * 100, 2), round((m + 1.96 * se) * 100, 2))


measures = {
    '①': ('rematch_results', 't_rate', None),
    '②': ('next_results', 't_rate', 'game_rate'),
    '⑤': ('rates_results', 'after_rate', 'game_rate'),
    '⑥': ('rates_hand_results', 'after_rate', 'game_rate'),
}
ORDER = ['배럴 전체', '배럴 홈런', '배럴 2·3루타', '배럴 단타', '배럴 아웃',
         '비배럴 타구 전체', '비배럴 홈런', '비배럴 2·3루타', '비배럴 단타', '비배럴 아웃', '삼진']

rows = {c: {'구분': c} for c in ORDER}
for key, (name, val, base) in measures.items():
    d = load(name)
    for keycol in ['cat', 'barrel_any']:
        for c, g in d.groupby(keycol):
            if c not in rows:
                continue
            x = g[val] - g[base] if base else g[val]
            m, cint = ci(x)
            rows[c][f'{key}_n'] = len(g)
            rows[c][f'{key}_%' if not base else f'{key}_차이_%p'] = m
            rows[c][f'{key}_ci95'] = cint
    if key == '①':  # barrel share per outcome, for context
        share = d[d['batted']].groupby('group')['barrel'].mean().mul(100).round(1).to_dict()

summary = pd.DataFrame([rows[c] for c in ORDER])


# key comparison: same contact quality, different outcome (배럴 장타 vs 배럴 아웃)
def diff(name, val, base, a_cat, b_cat):
    d = load(name)
    xa = (d.loc[d['cat'].isin(a_cat), val] - (d.loc[d['cat'].isin(a_cat), base] if base else 0))
    xb = (d.loc[d['cat'].isin(b_cat), val] - (d.loc[d['cat'].isin(b_cat), base] if base else 0))
    dd = xa.mean() - xb.mean()
    se = np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
    return round(dd * 100, 2), (round((dd - 1.96 * se) * 100, 2), round((dd + 1.96 * se) * 100, 2))


pairs = [('배럴 장타 − 배럴 아웃', ['배럴 홈런', '배럴 2·3루타'], ['배럴 아웃']),
         ('배럴 아웃 − 비배럴 아웃', ['배럴 아웃'], ['비배럴 아웃']),
         ('배럴 장타 − 비배럴 장타', ['배럴 홈런', '배럴 2·3루타'], ['비배럴 홈런', '비배럴 2·3루타'])]
compare = pd.DataFrame([{'비교': lab, **{k: diff(n, v, b, pa, pb)[0] for k, (n, v, b) in measures.items()},
                         **{f'{k}_ci95': diff(n, v, b, pa, pb)[1] for k, (n, v, b) in measures.items()}}
                        for lab, pa, pb in pairs])

# ---- 2x2: 타구 질 (비배럴 / 배럴) x 결과 (아웃 / 단타 / 장타 / 전체) ----
grid = {}
for key, (name, val, base) in measures.items():
    d = load(name)
    d = d[d['batted'] == True]
    d['quality'] = np.where(d['barrel'], '배럴', '비배럴')
    d['result'] = d['group'].map({'out': '아웃', 'single': '단타', 'HR': '장타', '2B/3B': '장타'})
    d['v'] = (d[val] - d[base] if base else d[val]) * 100
    t = d.pivot_table(index='quality', columns='result', values='v', aggfunc='mean')
    t['전체'] = d.groupby('quality')['v'].mean()
    n = d.pivot_table(index='quality', columns='result', values='v', aggfunc='size')
    n['전체'] = d.groupby('quality').size()
    grid[key] = t.loc[['비배럴', '배럴'], ['아웃', '단타', '장타', '전체']].round(2)
    grid[key + '_n'] = n.loc[['비배럴', '배럴'], ['아웃', '단타', '장타', '전체']]
pd.concat(grid, names=['measure']).to_csv(PROC / 'barrel_grid.csv')

summary.to_csv(PROC / 'barrel_summary.csv', index=False)
compare.to_csv(PROC / 'barrel_compare.csv', index=False)

pd.set_option('display.width', 250)
print("barrel share of batted balls by outcome (%):", share)
print("\n-- by barrel x outcome --")
print(summary.to_string(index=False))
for k in measures:
    print(f"\n-- 2x2 {k} --"); print(grid[k].to_string()); print(grid[k + '_n'].to_string())
print("\n-- comparisons (%p) --")
print(compare.to_string(index=False))
