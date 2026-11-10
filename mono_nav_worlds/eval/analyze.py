#!/usr/bin/env python3
"""Summarize results.csv: success rate (Wilson 95% CI), outcome breakdown, SPL, time,
per-world / per-difficulty / per-hard-case tables, and paired McNemar tests between methods.

usage:
  python3 analyze.py results/results.csv                 # all methods
  python3 analyze.py results/results.csv --ref gt        # McNemar of every method vs gt
"""
import argparse
import csv
import math
from collections import defaultdict

OUTCOMES = ['success', 'collision', 'timeout', 'stuck', 'nav_failed']


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(rows):
    n = len(rows)
    k = sum(int(r['success']) for r in rows)
    lo, hi = wilson(k, n)
    out = {'n': n, 'SR': k / n if n else 0, 'CI': f'[{lo:.2f},{hi:.2f}]'}
    for o in OUTCOMES[1:]:
        out[o] = sum(r['outcome'] == o for r in rows) / n if n else 0
    out['SPL'] = sum(float(r['spl']) for r in rows) / n if n else 0
    succ = [float(r['time_s']) for r in rows if r['success'] == '1']
    out['time_succ'] = sum(succ) / len(succ) if succ else float('nan')
    mc = [float(r['min_clearance']) for r in rows if r['min_clearance'] not in ('', None)]
    out['min_clear_med'] = sorted(mc)[len(mc) // 2] if mc else float('nan')
    return out


def table(title, groups):
    print(f'\n### {title}\n')
    cols = ['n', 'SR', 'CI', 'collision', 'timeout', 'stuck', 'nav_failed', 'SPL', 'time_succ', 'min_clear_med']
    print('| group | ' + ' | '.join(cols) + ' |')
    print('|' + '---|' * (len(cols) + 1))
    for name, rows in groups:
        s = summarize(rows)
        cells = [f'{s[c]:.2f}' if isinstance(s[c], float) else str(s[c]) for c in cols]
        print(f'| {name} | ' + ' | '.join(cells) + ' |')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--ref', default='gt', help='reference method for McNemar')
    ap.add_argument('--out', default=None, help='also write summary csv here')
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    methods = sorted({r['method'] for r in rows})

    g = defaultdict(list)
    for r in rows:
        g[r['method']].append(r)
    table('Overall', [(m, g[m]) for m in methods])

    for key, label in (('world', 'Per world'), ('difficulty', 'Per difficulty')):
        groups = defaultdict(list)
        for r in rows:
            groups[(r['method'], r[key])].append(r)
        table(label, [(f'{m} / {v}', groups[(m, v)]) for (m, v) in sorted(groups)])

    groups = defaultdict(list)
    for r in rows:
        tags = [t for t in r['tags'].split('|') if t] or ['none']
        for t in tags:
            groups[(r['method'], t)].append(r)
    table('Per hard case (an episode can count in several)', [(f'{m} / {t}', groups[(m, t)]) for (m, t) in sorted(groups)])

    # paired McNemar vs reference on episodes both methods ran
    if a.ref in methods:
        ref = {r['episode']: int(r['success']) for r in g[a.ref]}
        print(f'\n### Paired McNemar vs {a.ref}\n')
        print('| method | paired n | SR method | SR ref | b (method ok, ref fail) | c (ref ok, method fail) | p |')
        print('|---|---|---|---|---|---|---|')
        for m in methods:
            if m == a.ref:
                continue
            mine = {r['episode']: int(r['success']) for r in g[m]}
            common = sorted(set(mine) & set(ref))
            b = sum(mine[e] and not ref[e] for e in common)
            c = sum(ref[e] and not mine[e] for e in common)
            n = len(common)
            if n:
                print(f'| {m} | {n} | {sum(mine[e] for e in common) / n:.2f} | '
                      f'{sum(ref[e] for e in common) / n:.2f} | {b} | {c} | {mcnemar_exact(b, c):.4f} |')

    if a.out:
        with open(a.out, 'w', newline='') as f:
            w = None
            for m in methods:
                s = summarize(g[m])
                s = {'method': m, **s}
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(s))
                    w.writeheader()
                w.writerow(s)


if __name__ == '__main__':
    main()
