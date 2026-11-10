#!/usr/bin/env python3
"""Create the textures used by generate_world.py.

Some textures are tiled from the AWS small-house models (MIT-0 license);
the rest are procedural (plaster, tiles, wallpaper, cardboard, pictures).

usage: python3 make_textures.py <aws_models_dir> <out_textures_dir>
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

rng = np.random.default_rng(7)
PX_PER_M = 256                      # texture resolution
PANEL_W, PANEL_H = 1.0, 2.5         # wall panel size in meters (must match generator)


def tile(src, out_w, out_h, src_width_m):
    """Tile an image so one copy covers src_width_m meters."""
    im = Image.open(src).convert('RGB')
    w = int(src_width_m * PX_PER_M)
    h = int(im.height * w / im.width)
    im = im.resize((w, h), Image.LANCZOS)
    out = Image.new('RGB', (out_w, out_h))
    for y in range(0, out_h, h):
        for x in range(0, out_w, w):
            out.paste(im, (x, y))
    return out


def noise(w, h, scale, amp):
    small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale)))
    n = Image.fromarray(((small - small.min()) / (np.ptp(small) + 1e-9) * 255).astype(np.uint8))
    n = np.asarray(n.resize((w, h), Image.BICUBIC), float) / 255 - 0.5
    return n * amp


def plaster(color, w, h):
    base = np.ones((h, w, 3)) * np.array(color, float)
    n = noise(w, h, 64, 18) + noise(w, h, 8, 10) + rng.normal(0, 3, (h, w))
    img = np.clip(base + n[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(img)


def wallpaper(w, h):
    img = plaster((214, 206, 188), w, h)
    d = ImageDraw.Draw(img)
    step = 64
    for x in range(0, w, step):
        d.rectangle([x, 0, x + 8, h], fill=(188, 170, 140))
    for y in range(0, h, step):
        for x in range(step // 2, w, step):
            d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(150, 120, 90), width=3)
    return img.filter(ImageFilter.GaussianBlur(0.6))


def floor_tiles(w, h, tile_m, color, grout):
    img = plaster(color, w, h)
    d = ImageDraw.Draw(img)
    t = int(tile_m * PX_PER_M)
    for i in range(0, w + 1, t):
        d.line([i, 0, i, h], fill=grout, width=4)
    for j in range(0, h + 1, t):
        d.line([0, j, w, j], fill=grout, width=4)
    return img


def cardboard(seed):
    r = np.random.default_rng(seed)
    w = h = 512
    base = np.array([176, 136, 92]) + r.integers(-15, 15, 3)
    img = np.ones((h, w, 3)) * base
    img += (noise(w, h, 16, 14) + noise(w, h, 4, 8))[..., None]
    for x in range(0, w, 6):                      # corrugation lines
        img[:, x] -= 6
    img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    tape = (206, 180, 130) if seed % 2 else (190, 160, 110)
    d.rectangle([0, h // 2 - 30, w, h // 2 + 30], fill=tape)
    d.rectangle([0, 0, w - 1, h - 1], outline=(120, 90, 60), width=10)
    if seed % 3 == 0:                             # printed label + arrows
        d.rectangle([60, 60, 220, 160], fill=(240, 240, 235), outline=(40, 40, 40), width=3)
        for k in range(4):
            d.line([80, 80 + 20 * k, 200, 80 + 20 * k], fill=(60, 60, 60), width=4)
    if seed % 3 == 1:
        for k in range(2):
            x = 330 + 70 * k
            d.polygon([(x, 120), (x - 25, 160), (x + 25, 160)], fill=(30, 30, 30))
            d.rectangle([x - 8, 160, x + 8, 200], fill=(30, 30, 30))
    return img


def picture(seed):
    r = np.random.default_rng(100 + seed)
    w, h = 480, 360
    img = Image.new('RGB', (w, h), tuple(int(c) for c in r.integers(150, 255, 3)))
    d = ImageDraw.Draw(img)
    if seed % 2 == 0:                             # landscape-like
        horizon = int(r.integers(140, 230))
        d.rectangle([0, 0, w, horizon], fill=tuple(int(c) for c in r.integers(90, 220, 3)))
        for _ in range(4):
            x = int(r.integers(0, w)); hh = int(r.integers(40, 120))
            d.polygon([(x - 120, horizon), (x, horizon - hh), (x + 120, horizon)],
                      fill=tuple(int(c) for c in r.integers(30, 140, 3)))
    else:                                         # abstract shapes
        for _ in range(12):
            x0, y0 = int(r.integers(0, w)), int(r.integers(0, h))
            s = int(r.integers(20, 120))
            col = tuple(int(c) for c in r.integers(0, 255, 3))
            (d.ellipse if r.random() < .5 else d.rectangle)([x0, y0, x0 + s, y0 + s], fill=col)
    framed = Image.new('RGB', (w + 40, h + 40), (60, 40, 25))
    framed.paste(img, (20, 20))
    return framed


def main():
    aws, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    pw, ph = int(PANEL_W * PX_PER_M), int(PANEL_H * PX_PER_M)
    m = lambda n: os.path.join(aws, f'aws_robomaker_residential_{n}', 'materials', 'textures', f'aws_{n}.png')

    tile(m('RoomWall_01'), pw, ph, 1.6).save(f'{out}/wall_brick_white.png')
    plaster((222, 210, 186), pw, ph).save(f'{out}/wall_plaster_beige.png')
    plaster((196, 214, 200), pw, ph).save(f'{out}/wall_plaster_green.png')
    plaster((200, 205, 222), pw, ph).save(f'{out}/wall_plaster_blue.png')
    wallpaper(pw, ph).save(f'{out}/wall_wallpaper.png')

    tile(m('FloorB_01'), PX_PER_M, PX_PER_M, 0.6).save(f'{out}/floor_wood.png')
    floor_tiles(PX_PER_M, PX_PER_M, 0.5, (196, 192, 186), (120, 116, 110)).save(f'{out}/floor_tile_grey.png')
    floor_tiles(PX_PER_M, PX_PER_M, 0.33, (176, 140, 110), (110, 85, 65)).save(f'{out}/floor_tile_terracotta.png')

    for i in range(3):
        cardboard(i).save(f'{out}/cardboard_{i}.png')
    for i in range(6):
        picture(i).save(f'{out}/picture_{i}.png')
    print('textures written to', out)


if __name__ == '__main__':
    main()
