"""Number every table in 분석결과_총정리.md and render them as a readable HTML page (분석결과_표.html).

  python3 test_2025_analysis/make_tables.py

Idempotent: old "**[표 N]** ..." captions are removed and re-inserted, so re-run after every doc update.
"""
import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / '분석결과_총정리.md'
OUT = ROOT / '분석결과_표.html'
CAPTION = re.compile(r'^\s*\*\*\[표 \d+\]\*\*')


def is_row(line):
    return line.strip().startswith('|')


def is_sep(line):
    return bool(re.fullmatch(r'\s*\|(\s*:?-+:?\s*\|)+\s*', line))


def cells(line):
    return [c.strip() for c in re.split(r'(?<!\\)\|', line.strip().strip('|'))]


def clean(text):  # markdown inline -> plain text for titles
    return re.sub(r'[*`]', '', text).strip()


# ---- 1. strip old captions ----
lines = MD.read_text(encoding='utf-8').split('\n')
out, i = [], 0
while i < len(lines):
    if CAPTION.match(lines[i]):
        i += 1
        if i < len(lines) and not lines[i].strip():
            i += 1  # blank line after caption
        continue
    out.append(lines[i])
    i += 1
lines = out

# ---- 2. find tables, number them, collect context ----
tables, out, h2, h3, i = [], [], '', '', 0
while i < len(lines):
    line = lines[i]
    if line.startswith('## '):
        h2, h3 = clean(line[3:]), ''
    elif line.startswith('### '):
        h3 = clean(line[4:])
    if is_row(line) and i + 1 < len(lines) and is_sep(lines[i + 1]):
        j = i
        while j < len(lines) and is_row(lines[j]):
            j += 1
        block = lines[i:j]
        # title = analysis tag · subsection · label (bold text right above the table, else the column headers)
        k = len(out) - 1
        while k >= 0 and not out[k].strip():
            k -= 1
        prev = out[k].strip() if k >= 0 else ''
        m = re.match(r'^\*\*(.+?)\*\*', prev)
        head = cells(block[0])
        label = clean(m.group(1)) if m else f"{clean(head[0])}별 " + ', '.join(clean(h) for h in head[1:4]) + (' 등' if len(head) > 4 else '')
        tag = re.search(r'분석 [①-⑳]', h2)
        prefix = tag.group(0) if tag else re.sub(r'^\d+\.\s*', '', h2)
        mid = re.sub(r'^(결과|결론)(:\s*)?', '', h3).strip()
        n = len(tables) + 1
        title = ' · '.join(t for t in [prefix, mid, label] if t)
        indent = re.match(r'\s*', line).group(0)
        if out and out[-1].strip():
            out.append('')
        out += [f'{indent}**[표 {n}]** {title}', '']
        out += block
        aligns = ['right' if c.strip().endswith(':') else 'left' for c in cells(lines[i + 1])]
        tables.append({'n': n, 'h2': h2, 'title': title, 'head': cells(block[0]),
                       'rows': [cells(r) for r in block[2:]], 'aligns': aligns})
        i = j
        continue
    out.append(line)
    i += 1
MD.write_text('\n'.join(out), encoding='utf-8')


# ---- 3. HTML ----
def inline(text):
    t = html.escape(text).replace('&lt;br&gt;', '<br>')
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = re.sub(r'`(.+?)`', r'<code>\1</code>', t)
    t = re.sub(r'\[(−?[\d.]+, −?[\d.]+)\]', r'<span class="ci">[\1]</span>', t)  # confidence intervals
    return t


def sign_class(text):
    s = re.sub(r'[*\s]', '', text)
    if re.match(r'^−\d', s):
        return ' class="neg"'
    if re.match(r'^\+\d', s):
        return ' class="pos"'
    return ''


toc, body, cur = [], [], None
for t in tables:
    if t['h2'] != cur:
        cur = t['h2']
        sid = f"s{t['n']}"
        toc.append(f'<li class="sec"><a href="#{sid}">{html.escape(cur)}</a></li>')
        body.append(f'<h2 id="{sid}">{html.escape(cur)}</h2>')
    toc.append(f'<li><a href="#t{t["n"]}">표 {t["n"]}</a> {html.escape(t["title"])}</li>')
    ths = ''.join(f'<th style="text-align:{a}">{inline(h)}</th>' for h, a in zip(t['head'], t['aligns']))
    trs = ''.join('<tr>' + ''.join(f'<td style="text-align:{a}"{sign_class(c)}>{inline(c)}</td>'
                                   for c, a in zip(r, t['aligns'])) + '</tr>' for r in t['rows'])
    body.append(f'<section id="t{t["n"]}"><h3><span class="num">표 {t["n"]}</span>{html.escape(t["title"])}</h3>'
                f'<div class="wrap"><table><thead><tr>{ths}</tr></thead><tbody>{trs}</tbody></table></div></section>')

page = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>분석 결과 표 모음</title>
<style>
:root {{ --bg:#fbfbfa; --fg:#1f2328; --muted:#6b7280; --line:#e5e7eb; --head:#f3f4f6; --zebra:#f9fafb;
        --hover:#eef2ff; --accent:#4f46e5; --neg:#b42318; --pos:#067647; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16181d; --fg:#e6e6e6; --muted:#9ca3af; --line:#2d3139;
        --head:#20242c; --zebra:#1b1e24; --hover:#262b40; --accent:#a5b4fc; --neg:#f97066; --pos:#47cd89; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 -apple-system, "Apple SD Gothic Neo",
        "Noto Sans KR", sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} .sub {{ color:var(--muted); margin:0 0 20px; }}
nav {{ border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin-bottom:28px; }}
nav ul {{ list-style:none; margin:0; padding:0; columns:2 320px; }} nav li {{ font-size:13px; padding:1px 0; }}
nav li.sec {{ font-weight:600; margin-top:6px; }} a {{ color:var(--accent); text-decoration:none; }}
h2 {{ font-size:18px; margin:36px 0 8px; padding-top:8px; border-top:2px solid var(--line); }}
h3 {{ font-size:15px; margin:20px 0 8px; font-weight:600; }}
.num {{ display:inline-block; background:var(--accent); color:var(--bg); border-radius:6px; padding:1px 8px;
        margin-right:8px; font-size:12px; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums; }}
th, td {{ padding:7px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ background:var(--head); font-weight:600; position:sticky; top:0; white-space:nowrap; }}
tbody tr:nth-child(even) {{ background:var(--zebra); }} tbody tr:hover {{ background:var(--hover); }}
tbody tr:last-child td {{ border-bottom:none; }} td:first-child {{ white-space:nowrap; }}
.neg {{ color:var(--neg); }} .pos {{ color:var(--pos); }}
.ci {{ color:var(--muted); font-size:12px; white-space:nowrap; }}
code {{ font-size:12px; background:var(--head); padding:1px 4px; border-radius:4px; }}
</style></head><body><main>
<h1>분석 결과 표 모음</h1>
<p class="sub">분석결과_총정리.md의 표 {len(tables)}개. 번호는 문서의 [표 N]과 같습니다.
빨간색은 음수(감소), 초록색은 양수(증가), 회색 괄호는 95% 신뢰구간입니다.</p>
<nav><ul>{''.join(toc)}</ul></nav>
{''.join(body)}
</main></body></html>'''
OUT.write_text(page, encoding='utf-8')
print(f"numbered {len(tables)} tables -> {MD.name}, {OUT.name}")
