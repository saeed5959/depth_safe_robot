#!/usr/bin/env python3
"""Generate a 'corridor + hall' indoor world for Gazebo Fortress (ROS 2 Humble).

Layout (top view, x forward from the robot start):

          y=+4 +-----------------------------+
               |                             |
    y=+0.9 +---+          HALL               |
     start |  CORRIDOR (boxes, doors)        |
    y=-0.9 +---+     furniture, boxes,       |
               |     glass / blank panels    |
          y=-4 +-----------------------------+
           x=0  x=8                      x=15

Every world is reproducible from its seed. The script also writes:
  <name>_goals.yaml   start pose + goals (world frame and robot odom frame)
  <name>.png          top-view preview (obstacles, hard cases, free space, goals)

A layout is only accepted when every goal is reachable for a robot of
radius ROBOT_R (checked on a 5 cm occupancy grid).

usage:
  python3 generate_world.py --name train_01 --seed 1 --hard glass,blank,dark --out ../worlds
"""
import argparse
import math
import os
from collections import deque

import numpy as np
import yaml

# ---------------------------------------------------------------- geometry
CORR_LEN, CORR_HALF = 8.0, 0.9          # corridor: x in [0, 8], |y| < 0.9
HALL_X0, HALL_X1, HALL_HALF = 8.0, 15.0, 4.0
WALL_T, WALL_H, CEIL_H = 0.10, 2.5, 2.6
PANEL_W = 1.0
ROBOT_R = 0.30                          # robot radius used for the reachability check
RES = 0.05
START = (0.7, 0.0, 0.0)                 # x, y, yaw

TEX = 'model://mono_nav_assets/materials/textures/'
WALL_TEX = ['wall_brick_white', 'wall_plaster_beige', 'wall_plaster_green',
            'wall_plaster_blue', 'wall_wallpaper']
FLOOR_TEX = ['floor_wood', 'floor_tile_grey', 'floor_tile_terracotta']

# AWS small-house furniture: footprint (x, y) in meters, measured from the
# collision meshes with tools/mesh_bbox.py. against_wall: placed near a wall.
FURNITURE = {
    'Sofa_01':             ((2.45, 2.47), True),
    'TVCabinet_01':        ((0.62, 2.82), True),
    'Wardrobe_01':         ((1.00, 0.64), True),
    'Refrigerator_01':     ((1.04, 0.84), True),
    'ShoeRack_01':         ((0.90, 0.31), True),
    'ReadingDesk_01':      ((0.53, 1.16), True),
    'KitchenTable_01':     ((0.72, 1.82), False),
    'CoffeeTable_01':      ((1.33, 0.67), False),
    'FitnessEquipment_01': ((1.12, 1.12), False),
    'ChairA_01':           ((0.46, 0.48), False),
    'Trash_01':            ((0.28, 0.28), False),
}


class Grid:
    """Occupancy grid over the whole floor, used for placement and reachability."""

    def __init__(self):
        self.x0, self.y0 = -0.5, -HALL_HALF - 0.5
        self.nx = int((HALL_X1 + 1.0 - self.x0) / RES)
        self.ny = int((2 * HALL_HALF + 1.0) / RES)
        self.occ = np.ones((self.ny, self.nx), bool)       # start fully blocked
        self.free_rect(0, CORR_LEN + 0.2, -CORR_HALF, CORR_HALF, value=False)
        self.free_rect(HALL_X0, HALL_X1, -HALL_HALF, HALL_HALF, value=False)

    def idx(self, x, y):
        return int((y - self.y0) / RES), int((x - self.x0) / RES)

    def free_rect(self, x0, x1, y0, y1, value=True):
        """Set a rectangle to occupied (value=True) or free (value=False)."""
        r0, c0 = self.idx(x0, y0)
        r1, c1 = self.idx(x1, y1)
        self.occ[max(r0, 0):r1, max(c0, 0):c1] = value

    def rect_free(self, x0, x1, y0, y1):
        r0, c0 = self.idx(x0, y0)
        r1, c1 = self.idx(x1, y1)
        if r0 < 0 or c0 < 0 or r1 > self.ny or c1 > self.nx:
            return False
        return not self.occ[r0:r1, c0:c1].any()

    def inflated(self, r):
        k = int(math.ceil(r / RES))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        disk = (xx ** 2 + yy ** 2) <= k * k
        out = np.zeros_like(self.occ)
        for dy, dx in zip(*np.nonzero(disk)):
            dy -= k
            dx -= k
            out |= np.roll(np.roll(self.occ, dy, 0), dx, 1)
        return out

    def reachable(self, blocked, start):
        seen = np.zeros_like(blocked)
        q = deque([self.idx(*start)])
        if blocked[q[0]]:
            return seen
        seen[q[0]] = True
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.ny and 0 <= cc < self.nx and not blocked[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        return seen


# ---------------------------------------------------------------- SDF helpers
def material(tex=None, color=(0.9, 0.9, 0.9), transparency=None, rough=0.8, spec=0.1):
    if tex:
        color = (1.0, 1.0, 1.0)          # the texture provides the color
    c = ' '.join(f'{v:.3f}' for v in color)
    s = f'<material><ambient>{c} 1</ambient><diffuse>{c} 1</diffuse><specular>{spec} {spec} {spec} 1</specular>'
    if tex:
        s += (f'<pbr><metal><albedo_map>{TEX}{tex}.png</albedo_map>'
              f'<roughness>{rough}</roughness><metalness>0.0</metalness></metal></pbr>')
    s += '</material>'
    if transparency is not None:
        s += f'<transparency>{transparency}</transparency>'
    return s


def box_elems(name, pose, size, mat, collide=True):
    p = ' '.join(f'{v:.3f}' for v in pose)
    sz = ' '.join(f'{v:.3f}' for v in size)
    s = (f'<visual name="{name}_v"><pose>{p}</pose><geometry><box><size>{sz}</size></box></geometry>'
         f'{mat}</visual>')
    if collide:
        s += (f'<collision name="{name}_c"><pose>{p}</pose><geometry><box><size>{sz}</size></box>'
              f'</geometry></collision>')
    return s


def static_model(name, body, pose=(0, 0, 0, 0, 0, 0)):
    p = ' '.join(f'{v:.3f}' for v in pose)
    return (f'<model name="{name}"><static>true</static><pose>{p}</pose>'
            f'<link name="link">{body}</link></model>\n')


def include(uri, name, x, y, yaw, z=0.0):
    return (f'<include><uri>model://aws_robomaker_residential_{uri}</uri><name>{name}</name>'
            f'<static>true</static><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose></include>\n')


def point_light(name, x, y, z, intensity, rng_m=7.0):
    d = intensity
    return (f'<light type="point" name="{name}"><pose>{x:.2f} {y:.2f} {z:.2f} 0 0 0</pose>'
            f'<diffuse>{d:.2f} {d:.2f} {d * 0.95:.2f} 1</diffuse><specular>0.1 0.1 0.1 1</specular>'
            f'<attenuation><range>{rng_m}</range><constant>0.4</constant><linear>0.08</linear>'
            f'<quadratic>0.01</quadratic></attenuation><cast_shadows>false</cast_shadows></light>\n')


# ---------------------------------------------------------------- world builder
class World:
    def __init__(self, seed, hard):
        self.rng = np.random.default_rng(seed)
        self.hard = hard
        self.grid = Grid()
        self.sdf = []
        self.items = []            # for the preview: (kind, x0, x1, y0, y1)
        self.n = 0

    def uid(self, p):
        self.n += 1
        return f'{p}_{self.n}'

    # -- structure: walls, floor, ceiling, baseboards, doors, pictures
    def wall_run(self, x0, y0, x1, y1, inward, tex, body, blank=False):
        """Wall segment from (x0,y0) to (x1,y1), built from 1 m textured panels.
        inward: unit normal (nx, ny) pointing into the room."""
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        yaw = math.atan2(uy, ux)
        nx, ny = inward
        k, s = 0, 0.0
        while s < L - 1e-6:
            w = min(PANEL_W, L - s)
            cx = x0 + ux * (s + w / 2) - nx * WALL_T / 2
            cy = y0 + uy * (s + w / 2) - ny * WALL_T / 2
            mat = material(None, (0.93, 0.94, 0.93)) if blank else material(tex)
            body.append(box_elems(self.uid('wall'), (cx, cy, WALL_H / 2, 0, 0, yaw),
                                  (w, WALL_T, WALL_H), mat))
            s += w
            k += 1
        if not blank:   # dark baseboard: gives a strong floor/wall contact cue
            cx = (x0 + x1) / 2 + nx * 0.01
            cy = (y0 + y1) / 2 + ny * 0.01
            body.append(box_elems(self.uid('base'), (cx, cy, 0.05, 0, 0, yaw), (L, 0.02, 0.10),
                                  material(None, (0.30, 0.22, 0.16)), collide=False))
        self.items.append(('blank' if blank else 'wall', x0, x1, y0, y1))

    def structure(self):
        r = self.rng
        body = []
        tex_c, tex_h = r.choice(WALL_TEX, 2, replace=False)
        # corridor walls; with the 'blank' hard case, a 2-3 m stretch is plain white
        blank_span = None
        if 'blank' in self.hard:
            b0 = float(r.uniform(2.5, 4.5))
            blank_span = (b0, b0 + float(r.uniform(2.0, 3.0)))
        for side in (+1, -1):
            pts = [0.0, CORR_LEN]
            if blank_span:
                pts = [0.0, blank_span[0], blank_span[1], CORR_LEN]
            for a, b in zip(pts[:-1], pts[1:]):
                blank = blank_span is not None and a == blank_span[0]
                self.wall_run(a, side * CORR_HALF, b, side * CORR_HALF, (0, -side), tex_c, body, blank)
        self.wall_run(0, -CORR_HALF, 0, CORR_HALF, (1, 0), tex_c, body)               # corridor end
        # hall walls
        self.wall_run(HALL_X0, CORR_HALF, HALL_X0, HALL_HALF, (1, 0), tex_h, body)
        self.wall_run(HALL_X0, -HALL_HALF, HALL_X0, -CORR_HALF, (1, 0), tex_h, body)
        self.wall_run(HALL_X0, HALL_HALF, HALL_X1, HALL_HALF, (0, -1), tex_h, body)
        self.wall_run(HALL_X0, -HALL_HALF, HALL_X1, -HALL_HALF, (0, 1), tex_h, body)
        self.wall_run(HALL_X1, -HALL_HALF, HALL_X1, HALL_HALF, (-1, 0), tex_h, body)

        # floor tiles (1 m) and ceiling
        ftex_c, ftex_h = r.choice(FLOOR_TEX), r.choice(FLOOR_TEX)
        for x in np.arange(0, CORR_LEN, 1.0):
            for y in (-0.45, 0.45):
                body.append(box_elems(self.uid('floor'), (x + 0.5, y, -0.01, 0, 0, 0), (1.0, 0.9, 0.02),
                                      material(ftex_c), collide=False))
        for x in np.arange(HALL_X0, HALL_X1, 1.0):
            for y in np.arange(-HALL_HALF, HALL_HALF, 1.0):
                body.append(box_elems(self.uid('floor'), (x + 0.5, y + 0.5, -0.01, 0, 0, 0),
                                      (1.0, 1.0, 0.02), material(ftex_h), collide=False))
        body.append(box_elems('ceil_c', (CORR_LEN / 2, 0, CEIL_H, 0, 0, 0), (CORR_LEN, 2 * CORR_HALF, 0.05),
                              material(None, (0.95, 0.95, 0.93)), collide=False))
        body.append(box_elems('ceil_h', ((HALL_X0 + HALL_X1) / 2, 0, CEIL_H, 0, 0, 0),
                              (HALL_X1 - HALL_X0, 2 * HALL_HALF, 0.05),
                              material(None, (0.95, 0.95, 0.93)), collide=False))
        self.sdf.append(static_model('structure', ''.join(body)))
        self.blank_span = blank_span

        # closed doors along the corridor (outside the blank stretch)
        for i, x in enumerate(r.choice([1.5, 3.0, 4.5, 6.0], 2, replace=False)):
            side = 1 if i == 0 else -1
            if blank_span and blank_span[0] - 0.6 < x < blank_span[1] + 0.6:
                continue
            y = side * (CORR_HALF - 0.08)
            self.sdf.append(include('Door_01', f'door_{i}', float(x), y, 0.0))
            self.grid.free_rect(x - 0.5, x + 0.5, y - 0.08, y + 0.08)
            self.items.append(('door', x - 0.5, x + 0.5, y - 0.08, y + 0.08))

        # framed pictures on the walls (no collision)
        pics = []
        for x in r.uniform(1.0, 7.0, 2):
            if blank_span and blank_span[0] - 0.5 < x < blank_span[1] + 0.5:
                continue
            pics.append((x, CORR_HALF - 0.01, math.pi / 2 * 0))
        for x in r.uniform(9.0, 14.0, 3):
            pics.append((x, (HALL_HALF - 0.01) * (1 if r.random() < .5 else -1), 0))
        body = []
        for k, (x, y, _) in enumerate(pics):
            sgn = 1 if y > 0 else -1
            body.append(box_elems(f'pic{k}', (x, y - sgn * 0.01, 1.35, 0, 0, 0), (0.62, 0.015, 0.47),
                                  material(f'picture_{int(r.integers(0, 6))}'), collide=False))
        if body:
            self.sdf.append(static_model('pictures', ''.join(body)))

    # -- obstacles
    def try_place(self, w, h, xr, yr, tries=200, clearance=0.05):
        for _ in range(tries):
            x = float(self.rng.uniform(min(xr), max(xr)))
            y = float(self.rng.uniform(min(yr), max(yr)))
            if self.grid.rect_free(x - w / 2 - clearance, x + w / 2 + clearance,
                                   y - h / 2 - clearance, y + h / 2 + clearance):
                return x, y
        return None

    def mark(self, kind, x, y, w, h):
        self.grid.free_rect(x - w / 2, x + w / 2, y - h / 2, y + h / 2)
        self.items.append((kind, x - w / 2, x + w / 2, y - h / 2, y + h / 2))

    def cardboard_stack(self, xr, yr_fn, clearance):
        """Pick a stack size, place it (xr: x range, yr_fn(h) -> y range), build it."""
        r = self.rng
        sx, sy = float(r.uniform(0.35, 0.6)), float(r.uniform(0.3, 0.5))
        w, h = sx + 0.1, sy + 0.1                 # footprint incl. small rotation margin
        p = self.try_place(w, h, xr, yr_fn(h), clearance=clearance)
        if not p:
            return
        x, y = p
        levels = int(r.integers(1, 3))
        body, z = [], 0.0
        for k in range(levels):
            sz = float(r.uniform(0.25, 0.45))
            shrink = 0.85 ** k
            body.append(box_elems(f'b{k}', (0, 0, z + sz / 2, 0, 0, float(r.uniform(-.15, .15)) * k),
                                  (sx * shrink, sy * shrink, sz),
                                  material(f'cardboard_{int(r.integers(0, 3))}', rough=0.95)))
            z += sz
        yaw = float(r.uniform(-0.1, 0.1))
        self.sdf.append(static_model(self.uid('boxes'), ''.join(body), (x, y, 0, 0, 0, yaw)))
        self.mark('box', x, y, w, h)

    def corridor_obstacles(self):
        r = self.rng
        for _ in range(int(r.integers(2, 4))):   # box stacks against alternating walls
            side = 1 if r.random() < 0.5 else -1
            self.cardboard_stack((1.8, 7.4), lambda h, s=side: (s * (CORR_HALF - h / 2 - 0.02),) * 2,
                                 clearance=0.02)
        if r.random() < 0.7:                     # shoe rack against a wall
            side = 1 if r.random() < 0.5 else -1
            y = side * (CORR_HALF - 0.31 / 2 - 0.02)
            p = self.try_place(0.9, 0.31, (2.0, 7.0), (y, y), clearance=0.02)
            if p:
                self.sdf.append(include('ShoeRack_01', self.uid('shoerack'), p[0], p[1],
                                        0.0 if side < 0 else math.pi))
                self.mark('furn', p[0], p[1], 0.9, 0.31)

    def hall_furniture(self):
        r = self.rng
        x_rng, y_rng = (HALL_X0 + 0.9, HALL_X1 - 0.3), (-HALL_HALF + 0.3, HALL_HALF - 0.3)
        wall_items = [k for k, (_, wall) in FURNITURE.items() if wall and k != 'ShoeRack_01']
        free_items = ['KitchenTable_01', 'CoffeeTable_01', 'FitnessEquipment_01']
        picks = list(r.choice(wall_items, int(r.integers(3, 5)), replace=False))
        picks += list(r.choice(free_items, int(r.integers(1, 3)), replace=False))
        for name in picks:
            (fx, fy), wall = FURNITURE[name]
            for _ in range(60):
                rot = int(r.integers(0, 4))
                w, h = (fx, fy) if rot % 2 == 0 else (fy, fx)
                if wall:      # snap to one of the three outer hall walls
                    side = int(r.integers(0, 3))
                    if side == 0:
                        xr, yr = x_rng, (HALL_HALF - h / 2 - 0.17,) * 2
                    elif side == 1:
                        xr, yr = x_rng, (-HALL_HALF + h / 2 + 0.17,) * 2
                    else:
                        xr, yr = (HALL_X1 - w / 2 - 0.17,) * 2, y_rng
                else:
                    xr, yr = (HALL_X0 + 1.8, HALL_X1 - 1.8), (-2.5, 2.5)
                p = self.try_place(w, h, xr, yr, tries=8, clearance=0.15 if wall else 0.6)
                if p:
                    self.sdf.append(include(name, self.uid(name.lower()), p[0], p[1], rot * math.pi / 2))
                    self.mark('furn', p[0], p[1], w, h)
                    if name == 'KitchenTable_01':        # chairs around the table
                        for dx in (-1, 1):
                            cx = p[0] + dx * (w / 2 + 0.3)
                            if self.grid.rect_free(cx - .25, cx + .25, p[1] - .25, p[1] + .25):
                                self.sdf.append(include('ChairA_01', self.uid('chair'), cx, p[1],
                                                        (math.pi if dx > 0 else 0) + math.pi / 2))
                                self.mark('furn', cx, p[1], 0.48, 0.48)
                    break
        for _ in range(int(r.integers(2, 5))):        # loose cardboard boxes
            self.cardboard_stack((HALL_X0 + 1.2, HALL_X1 - 0.5), lambda h: (-3.6, 3.6), clearance=0.4)
        p = self.try_place(0.28, 0.28, (HALL_X0 + 0.5, HALL_X1 - 0.5), (-3.7, 3.7), clearance=0.3)
        if p:
            self.sdf.append(include('Trash_01', self.uid('trash'), p[0], p[1], 0.0))
            self.mark('furn', p[0], p[1], 0.28, 0.28)

    def hard_cases(self):
        r = self.rng
        if 'glass' in self.hard:       # free-standing glass partition with thin posts
            L = float(r.uniform(1.5, 2.4))
            vertical = r.random() < 0.5
            w, h = (0.04, L) if vertical else (L, 0.04)
            p = self.try_place(w, h, (HALL_X0 + 1.5, HALL_X1 - 1.5), (-2.8, 2.8), clearance=0.7)
            if p:
                x, y = p
                yaw = math.pi / 2 if vertical else 0.0
                # Realistic, *subtly* visible glass (what real partitions look like):
                #  - tinted, partly reflective pane (was transparency 0.85 = practically invisible)
                #  - frosted safety band ("manifestation", required on glass walls by building
                #    codes) at 0.35-0.60 m so it is inside the low robot camera's view
                #  - aluminium bottom and top rails
                # Band and rails are visual only and lie inside the pane's footprint, so the
                # collision geometry, GT depth and all layouts/goals/episodes stay identical.
                body = box_elems('glass', (0, 0, 1.0, 0, 0, 0), (L, 0.02, 2.0),
                                 material(None, (0.62, 0.76, 0.80), transparency=0.6, rough=0.05,
                                          spec=0.6))
                body += box_elems('glass_band', (0, 0, 0.475, 0, 0, 0), (L, 0.026, 0.25),
                                  material(None, (0.92, 0.94, 0.95), transparency=0.25, rough=0.6),
                                  collide=False)
                body += box_elems('glass_rail_bottom', (0, 0, 0.04, 0, 0, 0), (L, 0.036, 0.08),
                                  material(None, (0.55, 0.56, 0.58), rough=0.4, spec=0.4), collide=False)
                body += box_elems('glass_rail_top', (0, 0, 1.98, 0, 0, 0), (L, 0.036, 0.04),
                                  material(None, (0.55, 0.56, 0.58), rough=0.4, spec=0.4), collide=False)
                for s in (-1, 1):
                    body += box_elems(f'post{s}', (s * L / 2, 0, 1.0, 0, 0, 0), (0.03, 0.03, 2.0),
                                      material(None, (0.6, 0.6, 0.62)))
                self.sdf.append(static_model('glass_partition', body, (x, y, 0, 0, 0, yaw)))
                self.mark('glass', x, y, w, h)
        if 'blank' in self.hard:       # plain white free-standing panel in the hall
            L = float(r.uniform(1.2, 2.0))
            vertical = r.random() < 0.5
            w, h = (0.08, L) if vertical else (L, 0.08)
            p = self.try_place(w, h, (HALL_X0 + 1.5, HALL_X1 - 1.5), (-2.8, 2.8), clearance=0.7)
            if p:
                x, y = p
                yaw = math.pi / 2 if vertical else 0.0
                body = box_elems('panel', (0, 0, 1.0, 0, 0, 0), (L, 0.08, 2.0),
                                 material(None, (0.94, 0.94, 0.94)))
                self.sdf.append(static_model('blank_panel', body, (x, y, 0, 0, 0, yaw)))
                self.mark('blank', x, y, w, h)

    def lights(self):
        out = []
        dark = 'dark' in self.hard
        for k, x in enumerate(np.arange(1.5, CORR_LEN, 2.5)):
            out.append(point_light(f'corr_light_{k}', x, 0, 2.4, 0.8))
        k = 0
        dark_half = int(self.rng.integers(0, 2)) if dark else None
        for x in (HALL_X0 + 1.8, (HALL_X0 + HALL_X1) / 2, HALL_X1 - 1.8):
            for y in (-2.0, 2.0):
                inten = 0.8
                if dark and (y > 0) == bool(dark_half):
                    inten = 0.08                     # one side of the hall is dim
                out.append(point_light(f'hall_light_{k}', x, y, 2.4, inten))
                k += 1
        self.dark_side = None if dark_half is None else ('+y' if dark_half else '-y')
        return ''.join(out)

    # -- goals with guaranteed reachability
    def goals(self):
        blocked = self.grid.inflated(ROBOT_R)
        reach = self.grid.reachable(blocked, START[:2])
        clear = ~self.grid.inflated(ROBOT_R + 0.25) & reach
        cand = {
            'corridor_end': [(7.2, 0.0)],
            'hall_far': [(x, y) for x in np.arange(13.0, 14.5, 0.25) for y in np.arange(-3.0, 3.01, 0.25)],
            'hall_left': [(x, y) for x in np.arange(9.5, 14.0, 0.25) for y in np.arange(2.6, 3.6, 0.25)],
            'hall_right': [(x, y) for x in np.arange(9.5, 14.0, 0.25) for y in np.arange(-3.6, -2.6, 0.25)],
        }
        goals = {}
        for name, pts in cand.items():
            ok = [p for p in pts if clear[self.grid.idx(*p)]]
            if not ok:
                return None, reach
            p = ok[int(self.rng.integers(0, len(ok)))]
            goals[name] = [round(float(p[0]), 2), round(float(p[1]), 2)]
        return goals, reach


def world_sdf(name, content, lights):
    return f'''<?xml version="1.0"?>
<!-- generated by generate_world.py ({name}) - Gazebo Fortress / SDF 1.9 -->
<sdf version="1.9">
  <world name="{name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-contact-system" name="ignition::gazebo::systems::Contact"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <scene>
      <ambient>0.45 0.45 0.45 1</ambient>
      <background>0.7 0.7 0.7 1</background>
      <shadows>true</shadows>
      <grid>false</grid>
    </scene>
    <!-- soft fill light (no shadows, so it passes through the ceiling) -->
    <light type="directional" name="fill">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.35 0.35 0.35 1</diffuse>
      <specular>0.05 0.05 0.05 1</specular>
      <direction>-0.3 0.2 -0.9</direction>
      <cast_shadows>false</cast_shadows>
    </light>
{lights}
    <model name="ground_collision">
      <static>true</static>
      <link name="link">
        <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
      </link>
    </model>
{content}
  </world>
</sdf>
'''


def preview(path, w, goals, reach, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    g = w.grid
    fig, ax = plt.subplots(figsize=(10, 6))
    ext = [g.x0, g.x0 + g.nx * RES, g.y0, g.y0 + g.ny * RES]
    ax.imshow(np.where(reach, 0.92, 1.0), origin='lower', extent=ext, cmap='Greys_r', vmin=0, vmax=1)
    colors = {'wall': '#444', 'blank': '#e0e0e0', 'door': '#8b5a2b', 'box': '#c08a4a',
              'furn': '#5b7fa6', 'glass': '#40c4ff'}
    for kind, x0, x1, y0, y1 in w.items:
        if kind in ('wall', 'blank') and (x0 == x1 or y0 == y1):
            ax.plot([x0, x1], [y0, y1], color=colors[kind], lw=4 if kind == 'blank' else 3,
                    solid_capstyle='butt', zorder=3)
            if kind == 'blank':
                ax.plot([x0, x1], [y0, y1], color='#999', lw=1, ls=':', zorder=4)
        else:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=colors[kind],
                                   ec='k' if kind == 'blank' else None, lw=0.8, zorder=3))
    if w.dark_side:
        y0 = 0 if w.dark_side == '+y' else -HALL_HALF
        ax.add_patch(Rectangle((HALL_X0, y0), HALL_X1 - HALL_X0, HALL_HALF, color='k', alpha=0.15, zorder=2))
        ax.text(HALL_X1 - 0.2, y0 + HALL_HALF / 2, 'dark', ha='right', va='center', color='k', alpha=0.6)
    ax.plot(START[0], START[1], 'g>', ms=12, zorder=5)
    for name, (x, y) in goals.items():
        ax.plot(x, y, 'r*', ms=14, zorder=5)
        ax.text(x, y + 0.25, name, ha='center', fontsize=8, zorder=6)
    ax.set_aspect('equal')
    ax.set_xlim(-0.6, HALL_X1 + 0.6)
    ax.set_ylim(-HALL_HALF - 0.6, HALL_HALF + 0.6)
    ax.set_title(title + '   (grey = reachable for r=%.2f m)' % ROBOT_R, fontsize=10)
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc='lower right', fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build(name, seed, hard, out, preview_dir):
    for attempt in range(50):
        w = World(seed * 1000 + attempt, hard)
        w.structure()
        w.corridor_obstacles()
        w.hard_cases()          # place hard cases before furniture so they always fit
        w.hall_furniture()
        lights = w.lights()
        goals, reach = w.goals()
        if goals:
            break
    else:
        raise RuntimeError(f'{name}: could not build a connected layout')

    with open(os.path.join(out, f'{name}.sdf'), 'w') as f:
        f.write(world_sdf(name, '    '.join(w.sdf), lights))
    sx, sy, syaw = START
    c, s = math.cos(-syaw), math.sin(-syaw)
    goals_odom = {k: [round(c * (x - sx) - s * (y - sy), 2), round(s * (x - sx) + c * (y - sy), 2)]
                  for k, (x, y) in goals.items()}
    meta = {
        'world': name, 'seed': seed, 'hard_cases': sorted(hard),
        'blank_corridor_span_x': [round(v, 2) for v in w.blank_span] if w.blank_span else None,
        'dark_side_of_hall': w.dark_side,
        'start': {'x': sx, 'y': sy, 'yaw': syaw},
        'goals_world': goals,
        'goals_odom': goals_odom,
    }
    with open(os.path.join(out, f'{name}_goals.yaml'), 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    if preview_dir:
        preview(os.path.join(preview_dir, f'{name}.png'), w, goals, reach,
                f'{name}  seed={seed}  hard={",".join(sorted(hard)) or "none"}')
    print(f'{name}: {len(w.sdf)} models, goals ok (attempt {attempt + 1})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--hard', default='', help='comma list: glass,blank,dark')
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'worlds'))
    ap.add_argument('--preview', default=os.path.join(os.path.dirname(__file__), '..', 'previews'))
    a = ap.parse_args()
    hard = {h for h in a.hard.split(',') if h}
    assert hard <= {'glass', 'blank', 'dark'}, hard
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.preview, exist_ok=True)
    build(a.name, a.seed, hard, a.out, a.preview)
