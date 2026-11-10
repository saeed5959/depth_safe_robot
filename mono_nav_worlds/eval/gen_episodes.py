#!/usr/bin/env python3
"""Generate a fixed, seeded list of start/goal episodes for one world.

Every method (GT depth, mono raw, mono + margin, mono d-2sigma) runs the SAME file.

Test worlds (default):
  * 4 canonical episodes: spawn -> each goal of <world>_goals.yaml
  * sampled pairs, stratified easy / medium / hard
  * each pair repeated with small start noise (±0.1 m, ±10°)
Train worlds (--train): no canonical, more pairs, and a share of goals placed
close to obstacles so the camera records close-range views (where sigma matters).

Difficulty:
  easy   = straight line of sight start->goal, no hard case on the way
  hard   = path passes a hard case (glass / blank wall / dark side) or a narrow gap
  medium = everything else

usage:
  python3 gen_episodes.py --worlds_dir ~/depth_safe_robot/mono_nav_worlds/worlds \
      --world test_02 --out episodes/test_02.yaml
  python3 gen_episodes.py ... --world train_01 --train --out episodes/train_01.yaml
"""
import argparse
import copy
import math
import os

import numpy as np
import yaml

from world_map import WorldMap

PLAN_R = 0.30          # same robot radius the world generator used for reachability
START_CLEAR = 0.45     # start/goal at least this far from any obstacle
NARROW = 0.42          # if no route keeps this clearance -> "narrow" -> hard
V_MAX = 0.30           # Nav2 max_vel_x


def classify(wm, path, start, goal):
    tags = wm.tags(path)
    min_clear = float(min(wm.clearance(x, y) for x, y in path))
    los = wm.line_of_sight(start, goal, PLAN_R)
    # narrow = no comfortable route exists (A* that keeps NARROW clearance fails or detours a lot)
    wide, wide_len = wm.astar(start, goal, NARROW)
    plen = float(np.sum(np.hypot(*np.diff(path, axis=0).T)))
    narrow = wide is None or wide_len > 1.3 * plen
    if narrow:
        tags = tags + ['narrow']
    if tags:
        diff = 'hard'
    elif los:
        diff = 'easy'
    else:
        diff = 'medium'
    return diff, tags, min_clear, los


def start_yaw(path, rng, noise=0.35):
    """Face roughly along the first ~0.8 m of the path (camera only looks forward)."""
    p0 = path[0]
    k = min(len(path) - 1, 16)
    return math.atan2(path[k][1] - p0[1], path[k][0] - p0[0]) + rng.uniform(-noise, noise)


def make_ep(wm, eid, start, yaw, goal, path, length, diff, tags, min_clear, los, kind):
    return {
        'id': eid, 'kind': kind,
        'start': {'x': round(float(start[0]), 3), 'y': round(float(start[1]), 3),
                  'yaw': round(float(yaw), 3)},
        'goal': {'x': round(float(goal[0]), 3), 'y': round(float(goal[1]), 3)},
        'astar_len': round(float(length), 2),
        'difficulty': diff, 'tags': tags, 'los': bool(los),
        'min_path_clearance': round(min_clear, 3),
        't_max': round(2.5 * length / V_MAX + 20.0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--worlds_dir', required=True)
    ap.add_argument('--world', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=None, help='default: hash of world name')
    ap.add_argument('--pairs', type=int, default=None, help='sampled pairs (default 11 test / 40 train)')
    ap.add_argument('--repeats', type=int, default=None, help='runs per pair (default 2 test / 1 train)')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--near_frac', type=float, default=None,
                    help='share of goals near obstacles (default 0 test / 0.3 train)')
    ap.add_argument('--min_len', type=float, default=2.5)
    ap.add_argument('--max_len', type=float, default=14.0)
    ap.add_argument('--preview', default=None, help='png path (default next to --out)')
    a = ap.parse_args()

    pairs = a.pairs if a.pairs is not None else (40 if a.train else 11)
    repeats = a.repeats if a.repeats is not None else (1 if a.train else 2)
    near_frac = a.near_frac if a.near_frac is not None else (0.3 if a.train else 0.0)
    seed = a.seed if a.seed is not None else sum(map(ord, a.world)) * 7 + (1 if a.train else 0)
    rng = np.random.default_rng(seed)

    wd = os.path.expanduser(a.worlds_dir)
    wm = WorldMap(os.path.join(wd, f'{a.world}.sdf'), os.path.join(wd, f'{a.world}_goals.yaml'))
    spawn = wm.meta['start']

    base = []   # unique pairs, before repeats
    # ---- canonical: spawn -> the 4 named goals (test only)
    if not a.train:
        s = (spawn['x'], spawn['y'])
        for name, g in wm.meta['goals_world'].items():
            path, L = wm.astar(s, g, PLAN_R)
            if path is None:
                print(f'  canonical {name}: unreachable, skipped')
                continue
            diff, tags, mc, los = classify(wm, path, s, g)
            base.append(make_ep(wm, f'canon_{name}', s, spawn['yaw'], g, path, L, diff, tags, mc, los, 'canonical'))

    # ---- sampled pairs, stratified
    free = np.argwhere((wm.clear >= START_CLEAR) & wm.inside)
    near = np.argwhere((wm.clear >= PLAN_R + 0.02) & (wm.clear < PLAN_R + 0.20) & wm.inside)
    target = {'easy': pairs // 3, 'medium': pairs // 3}
    target['hard'] = pairs - target['easy'] - target['medium']
    if a.train:                                # train: just diversity, no strict mix
        target = {'easy': pairs, 'medium': pairs, 'hard': pairs}
    counts = {k: 0 for k in target}
    sampled = 0
    for it in range(20000):
        if sampled >= pairs:
            break
        if it == 4000:                         # e.g. test_clean has no hard cases: relax the mix
            target = {k: pairs for k in target}
        s = wm.xy(*free[rng.integers(len(free))])
        pool = near if (rng.random() < near_frac and len(near)) else free
        g = wm.xy(*pool[rng.integers(len(pool))])
        if math.hypot(g[0] - s[0], g[1] - s[1]) < a.min_len:
            continue
        path, L = wm.astar(s, g, PLAN_R)
        if path is None or not (a.min_len <= L <= a.max_len):
            continue
        diff, tags, mc, los = classify(wm, path, s, g)
        if counts[diff] >= target[diff]:
            continue
        counts[diff] += 1
        sampled += 1
        kind = 'near_obstacle' if pool is near else 'sampled'
        base.append(make_ep(wm, f'p{sampled:03d}', s, start_yaw(path, rng), g, path, L,
                            diff, tags, mc, los, kind))
    if sampled < pairs:
        print(f'  WARNING: only {sampled}/{pairs} pairs found, counts={counts}')

    # ---- repeats with start noise
    episodes = []
    for ep in base:
        for r in range(repeats):
            e = copy.deepcopy(ep)
            e['pair'] = ep['id']
            e['repeat'] = r
            e['id'] = f"{a.world}_{ep['id']}_r{r}"
            if r > 0:
                for _ in range(50):
                    x = ep['start']['x'] + rng.uniform(-0.1, 0.1)
                    y = ep['start']['y'] + rng.uniform(-0.1, 0.1)
                    if wm.clearance(x, y) >= START_CLEAR - 0.05:
                        e['start'] = {'x': round(x, 3), 'y': round(y, 3),
                                      'yaw': round(ep['start']['yaw'] + rng.uniform(-0.17, 0.17), 3)}
                        break
            episodes.append(e)

    out = {
        'world': a.world, 'seed': int(seed), 'split': 'train' if a.train else 'test',
        'spawn': spawn,
        'config': {'plan_r': PLAN_R, 'start_clear': START_CLEAR, 'narrow': NARROW,
                   'v_max': V_MAX, 'pairs': pairs, 'repeats': repeats, 'near_frac': near_frac},
        'episodes': episodes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=False)
    mix = {k: sum(e['difficulty'] == k for e in episodes) for k in ('easy', 'medium', 'hard')}
    tagc = {t: sum(t in e['tags'] for e in episodes) for t in ('glass', 'blank', 'dark', 'narrow')}
    print(f'{a.world}: {len(episodes)} episodes  difficulty={mix}  tags={tagc}  -> {a.out}')

    prev = a.preview or os.path.splitext(a.out)[0] + '.png'
    try:
        draw(wm, base, prev)
    except ImportError:
        pass


def draw(wm, eps, path_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from world_map import X0, X1, Y0, Y1
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(wm.occ, origin='lower', extent=[X0, X1, Y0, Y1], cmap='Greys', alpha=0.8)
    for name, m in (('glass', wm.d_glass), ('blank panel', wm.d_panel)):
        if m is not None:
            ax.contour(wm.X, wm.Y, m, levels=[0.06], colors='c' if name == 'glass' else 'm')
    col = {'easy': 'tab:green', 'medium': 'tab:orange', 'hard': 'tab:red'}
    for e in eps:
        s, g = e['start'], e['goal']
        path, _ = wm.astar((s['x'], s['y']), (g['x'], g['y']), PLAN_R)
        if path is not None:
            ax.plot(path[:, 0], path[:, 1], '-', color=col[e['difficulty']], lw=1, alpha=0.7)
        ax.arrow(s['x'], s['y'], 0.4 * math.cos(s['yaw']), 0.4 * math.sin(s['yaw']),
                 head_width=0.12, color='k')
        ax.plot(g['x'], g['y'], '*', color=col[e['difficulty']], ms=10)
        ax.text(g['x'], g['y'] + 0.12, e['id'], fontsize=6)
    ax.set_title(f'{wm.name}: green=easy orange=medium red=hard  (cyan=glass, magenta=blank panel)')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(path_png, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main()
