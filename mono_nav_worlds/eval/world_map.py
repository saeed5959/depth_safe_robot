"""2D occupancy / clearance map of a generated mono_nav world (corridor + hall).

Shared by gen_episodes.py (sampling + A*) and episode_runner.py (clearance metric,
optional geometric collision check).

Obstacles come from:
  * every <collision><box> in the SDF whose height range overlaps the robot (walls,
    glass, blank panel, cardboard boxes),
  * every <include> of an AWS furniture model, using the footprints measured for
    generate_world.py (bounding boxes, so tables/chairs are treated as solid).
"""
import heapq
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from scipy.ndimage import distance_transform_edt

RES = 0.05
X0, X1, Y0, Y1 = -0.5, 15.5, -4.5, 4.5
CORRIDOR = (0.0, 8.0, -0.9, 0.9)
HALL = (8.0, 15.0, -4.0, 4.0)
ROBOT_Z = (0.02, 0.40)          # height band the robot body/camera occupies

# footprint (x, y) in meters, same table as generate_world.py
FURNITURE = {
    'Sofa_01': (2.45, 2.47),
    'TVCabinet_01': (0.62, 2.82),
    'Wardrobe_01': (1.00, 0.64),
    'Refrigerator_01': (1.04, 0.84),
    'ShoeRack_01': (0.90, 0.31),
    'ReadingDesk_01': (0.53, 1.16),
    'KitchenTable_01': (0.72, 1.82),
    'CoffeeTable_01': (1.33, 0.67),
    'FitnessEquipment_01': (1.12, 1.12),
    'ChairA_01': (0.46, 0.48),
    'Trash_01': (0.28, 0.28),
    'Door_01': (0.95, 0.12),     # closed door set into the corridor wall (approx.)
}


def _floats(text, n=6):
    v = [float(t) for t in (text or '').split()]
    return v + [0.0] * (n - len(v))


def _compose(a, b):
    """2D pose composition a*b, poses are (x, y, z, yaw)."""
    c, s = math.cos(a[3]), math.sin(a[3])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2], a[3] + b[3])


def _pose(el):
    p = _floats(el.text if el is not None else '')
    return (p[0], p[1], p[2], p[5])


def parse_world(sdf_path):
    """Return list of obstacle boxes: dict(x, y, yaw, sx, sy, model)."""
    world = ET.parse(sdf_path).getroot().find('world')
    boxes, unknown = [], set()
    for model in world.findall('model'):
        mp = _pose(model.find('pose'))
        for link in model.findall('link'):
            lp = _compose(mp, _pose(link.find('pose')))
            for col in link.findall('collision'):
                b = col.find('geometry/box')
                if b is None:
                    continue
                sx, sy, sz = _floats(b.find('size').text, 3)
                p = _compose(lp, _pose(col.find('pose')))
                if p[2] + sz / 2 < ROBOT_Z[0] or p[2] - sz / 2 > ROBOT_Z[1]:
                    continue
                boxes.append(dict(x=p[0], y=p[1], yaw=p[3], sx=sx, sy=sy, model=model.get('name')))
    for inc in world.findall('include'):
        key = inc.find('uri').text.split('residential_')[-1]
        p = _pose(inc.find('pose'))
        name = inc.find('name').text if inc.find('name') is not None else key
        if key in FURNITURE:
            sx, sy = FURNITURE[key]
            boxes.append(dict(x=p[0], y=p[1], yaw=p[3], sx=sx, sy=sy, model=name))
        else:
            unknown.add(key)
    if unknown:
        print(f'[world_map] WARNING: no footprint for {sorted(unknown)} (ignored)')
    return boxes


class WorldMap:
    def __init__(self, sdf_path, goals_yaml=None):
        self.name = os.path.splitext(os.path.basename(sdf_path))[0]
        self.meta = yaml.safe_load(open(goals_yaml)) if goals_yaml else {}
        self.boxes = parse_world(sdf_path)
        self.nx = int(round((X1 - X0) / RES))
        self.ny = int(round((Y1 - Y0) / RES))
        xs = X0 + (np.arange(self.nx) + 0.5) * RES
        ys = Y0 + (np.arange(self.ny) + 0.5) * RES
        self.X, self.Y = np.meshgrid(xs, ys)          # [iy, ix]

        inside = self._rect(*CORRIDOR) | self._rect(*HALL)
        occ = ~inside
        glass = np.zeros_like(occ)
        panel = np.zeros_like(occ)
        for b in self.boxes:
            m = self._box(b)
            occ |= m
            if b['model'] == 'glass_partition':
                glass |= m
            if b['model'] == 'blank_panel':
                panel |= m
        self.occ = occ
        self.inside = inside
        self.clear = distance_transform_edt(~occ) * RES          # m to nearest obstacle cell
        self.d_glass = distance_transform_edt(~glass) * RES if glass.any() else None
        self.d_panel = distance_transform_edt(~panel) * RES if panel.any() else None

    # ---------------------------------------------------------------- raster helpers
    def _rect(self, x0, x1, y0, y1):
        return (self.X > x0) & (self.X < x1) & (self.Y > y0) & (self.Y < y1)

    def _box(self, b):
        c, s = math.cos(b['yaw']), math.sin(b['yaw'])
        dx, dy = self.X - b['x'], self.Y - b['y']
        u = c * dx + s * dy
        v = -s * dx + c * dy
        return (np.abs(u) <= b['sx'] / 2) & (np.abs(v) <= b['sy'] / 2)

    def cell(self, x, y):
        ix = int((x - X0) / RES)
        iy = int((y - Y0) / RES)
        return min(max(iy, 0), self.ny - 1), min(max(ix, 0), self.nx - 1)

    def xy(self, iy, ix):
        return X0 + (ix + 0.5) * RES, Y0 + (iy + 0.5) * RES

    def clearance(self, x, y):
        return float(self.clear[self.cell(x, y)])

    # ---------------------------------------------------------------- planning
    def astar(self, start, goal, radius):
        """8-connected A* on cells with clearance >= radius. Returns (path_xy Nx2, length) or (None, inf)."""
        free = self.clear >= radius
        s, g = self.cell(*start), self.cell(*goal)
        if not free[s] or not free[g]:
            return None, math.inf
        nb = [(-1, -1, 1.4142), (-1, 0, 1.0), (-1, 1, 1.4142), (0, -1, 1.0),
              (0, 1, 1.0), (1, -1, 1.4142), (1, 0, 1.0), (1, 1, 1.4142)]
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        cost = {s: 0.0}
        parent = {s: None}
        pq = [(h(s), s)]
        closed = set()
        while pq:
            _, c = heapq.heappop(pq)
            if c in closed:
                continue
            if c == g:
                break
            closed.add(c)
            for dy, dx, w in nb:
                n = (c[0] + dy, c[1] + dx)
                if not (0 <= n[0] < self.ny and 0 <= n[1] < self.nx) or not free[n]:
                    continue
                nc = cost[c] + w
                if nc < cost.get(n, math.inf):
                    cost[n] = nc
                    parent[n] = c
                    heapq.heappush(pq, (nc + h(n), n))
        if g not in parent:
            return None, math.inf
        cells = []
        c = g
        while c is not None:
            cells.append(c)
            c = parent[c]
        cells.reverse()
        path = np.array([self.xy(*c) for c in cells])
        return path, cost[g] * RES

    def line_of_sight(self, a, b, radius):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (RES / 2)))
        for t in np.linspace(0, 1, n):
            if self.clearance(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) < radius:
                return False
        return True

    # ---------------------------------------------------------------- hard-case tags
    def tags(self, path, near=1.0):
        """Which hard cases a path is exposed to: glass, blank, dark."""
        out = []
        cells = [self.cell(x, y) for x, y in path]
        if self.d_glass is not None and min(self.d_glass[c] for c in cells) < near:
            out.append('glass')
        span = self.meta.get('blank_corridor_span_x')
        blank = False
        if span:
            blank = any(span[0] <= x <= span[1] and abs(y) < 0.9 for x, y in path)
        if self.d_panel is not None and min(self.d_panel[c] for c in cells) < near:
            blank = True
        if blank:
            out.append('blank')
        side = self.meta.get('dark_side_of_hall')
        if side:
            sgn = 1 if side.startswith('+') else -1
            frac = np.mean([(x > HALL[0]) and (sgn * y > 0.3) for x, y in path])
            if frac > 0.25:
                out.append('dark')
        return out
