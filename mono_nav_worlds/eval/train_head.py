#!/usr/bin/env python3
"""Train the uncertainty head (mu, sigma) on top of frozen Depth Anything V2.

Loss: beta-NLL (Seitzer et al. 2022) of the log-depth error, weighted toward near pixels
    r    = log d_GT - (log d_DA + mu)
    nll  = 0.5 * r^2 * exp(-2s) + s                       (s = log sigma)
    loss = sum(w * nll * stopgrad(sigma^(2*beta))) / sum(w),   w = 1 / max(d_GT, 0.5)

Split is BY WORLD (default: val = train_03). Test worlds are never used.
After training, on the val world:
    k  : smallest k so that 97.7 % of near pixels (< 3 m) satisfy d_GT >= d_DA*exp(mu - k*sigma)
    c  : constant log margin with the SAME coverage for the baseline  d_DA*exp(-c)
Both are stored in the checkpoint and in calib.json.

usage:
  python3 train_head.py --data dataset --da_repo $DA \
      --checkpoint $DA/checkpoints/depth_anything_v2_metric_hypersim_vits.pth --out runs/head_v1
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from mono_nav.uncertainty_head import (DA2Features, UncertaintyHead, net_size,
                                           preprocess, upsample)
except ImportError:      # workspace not sourced: import from the source tree
    sys.path.insert(0, os.path.expanduser('~/depth_safe_robot/ros2_ws/src/mono_nav'))
    from mono_nav.uncertainty_head import (DA2Features, UncertaintyHead, net_size,
                                           preprocess, upsample)

MAX_D, MIN_D, NEAR = 10.0, 0.1, 3.0


# ---------------------------------------------------------------- data
class FrameSet(Dataset):
    def __init__(self, root, worlds, augment, stride=1):
        self.items, self.augment = [], augment
        for w in worlds:
            for d in sorted(glob.glob(os.path.join(root, w, '*'))):
                idx = os.path.join(d, 'index.jsonl')
                if not os.path.exists(idx):
                    continue
                for line in open(idx):
                    r = json.loads(line)
                    if r.get('valid_frac', 1.0) < 0.2:
                        continue
                    self.items.append((os.path.join(d, 'rgb', r['file']),
                                       os.path.join(d, 'depth', r['file']), w))
        self.items = self.items[::stride]
        if not self.items:
            raise RuntimeError(f'no frames for worlds {worlds} in {root}')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rp, dp, _ = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(rp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dep = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        if self.augment:
            if np.random.rand() < 0.5:
                rgb, dep = rgb[:, ::-1], dep[:, ::-1]
            if np.random.rand() < 0.8:          # brightness/contrast, incl. dark scenes
                a = np.random.uniform(0.5, 1.3)
                b = np.random.uniform(-20, 20)
                rgb = np.clip(rgb.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        rgb = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)     # 3,H,W uint8
        dep = torch.from_numpy(np.ascontiguousarray(dep))                        # H,W meters
        return rgb, dep


# ---------------------------------------------------------------- model step
def forward(da2, head, rgb, size):
    """rgb uint8 (B,3,H,W) on device -> d_da, mu, s at full image resolution."""
    H, W = rgb.shape[-2:]
    x = preprocess(rgb, size)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=rgb.is_cuda):
        d_da, feat = da2(x)
    inp = UncertaintyHead.make_input(feat, d_da, x)
    mu, s = head(inp)
    return upsample(d_da, (H, W)), upsample(mu, (H, W)), upsample(s, (H, W))


def loss_fn(d_da, mu, s, gt, beta, near_w=True):
    m = (gt > MIN_D) & (gt < MAX_D) & (d_da > MIN_D)
    r = torch.log(gt.clamp(min=MIN_D)) - (torch.log(d_da.clamp(min=MIN_D)) + mu)
    nll = 0.5 * r ** 2 * torch.exp(-2 * s) + s
    if beta > 0:
        nll = nll * torch.exp(2 * beta * s).detach()
    w = (1.0 / gt.clamp(min=0.5)) if near_w else torch.ones_like(gt)
    w = w * m
    return (nll * w).sum() / w.sum().clamp(min=1.0)


# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(da2, head, loader, size, dev, n_sample=4000):
    """Depth metrics + samples for calibration and sparsification."""
    acc = {k: 0.0 for k in ['n', 'n_near', 'absrel_raw', 'absrel_cor', 'd1_raw', 'd1_cor',
                            'over_raw', 'over_cor', 'sigma', 'nll']}
    z_near, gap_near, err_s = [], [], []        # (log dhat - log gt)/sigma, log dDA - log gt
    t0 = time.time()
    for bi, (rgb, gt) in enumerate(loader):
        if bi % 50 == 0:
            print(f'    eval batch {bi}/{len(loader)}  {time.time() - t0:.0f} s', flush=True)
        rgb, gt = rgb.to(dev), gt.to(dev)
        d_da, mu, s = forward(da2, head, rgb, size)
        d_cor = d_da * torch.exp(mu)
        m = (gt > MIN_D) & (gt < MAX_D) & (d_da > MIN_D)
        g, dr, dc, ss, mm = gt[m], d_da[m], d_cor[m], s[m], mu[m]
        acc['n'] += g.numel()
        acc['absrel_raw'] += ((dr - g).abs() / g).sum().item()
        acc['absrel_cor'] += ((dc - g).abs() / g).sum().item()
        acc['d1_raw'] += (torch.maximum(dr / g, g / dr) < 1.25).sum().item()
        acc['d1_cor'] += (torch.maximum(dc / g, g / dc) < 1.25).sum().item()
        acc['sigma'] += torch.exp(ss).sum().item()
        r = torch.log(g) - torch.log(dc)
        acc['nll'] += (0.5 * r ** 2 * torch.exp(-2 * ss) + ss).sum().item()
        near = g < NEAR
        acc['n_near'] += near.sum().item()
        acc['over_raw'] += ((dr > g + 0.2) & near).sum().item()
        acc['over_cor'] += ((dc > g + 0.2) & near).sum().item()
        # random pixel samples for calibration / sparsification
        idx = torch.randperm(g.numel(), device=dev)[:n_sample]
        gi, dri, dci, si = g[idx], dr[idx], dc[idx], ss[idx]
        ni = gi < NEAR
        z_near.append(((torch.log(dci) - torch.log(gi)) / torch.exp(si))[ni].cpu())
        gap_near.append((torch.log(dri) - torch.log(gi))[ni].cpu())
        err_s.append(torch.stack([(torch.log(dci) - torch.log(gi)).abs(), torch.exp(si)], 1).cpu())
    n, nn_ = max(acc['n'], 1), max(acc['n_near'], 1)
    out = {
        'absrel_raw': acc['absrel_raw'] / n, 'absrel_cor': acc['absrel_cor'] / n,
        'delta1_raw': acc['d1_raw'] / n, 'delta1_cor': acc['d1_cor'] / n,
        'overest_near_raw': acc['over_raw'] / nn_, 'overest_near_cor': acc['over_cor'] / nn_,
        'mean_sigma': acc['sigma'] / n, 'nll': acc['nll'] / n,
    }
    z = torch.cat(z_near).numpy()
    gap = torch.cat(gap_near).numpy()
    es = torch.cat(err_s).numpy()
    return out, z, gap, es


def ause(err, sig, steps=20):
    """Area between the sigma-sparsification curve and the oracle curve (lower = better)."""
    n = len(err)
    o_err = np.sort(err)[::-1]
    o_sig = err[np.argsort(-sig)]
    fr = np.linspace(0, 0.95, steps)
    cur = lambda e: np.array([e[int(f * n):].mean() for f in fr])
    c_or, c_sg = cur(o_err), cur(o_sig)
    trap = getattr(np, 'trapezoid', None) or np.trapz
    return float(trap(c_sg - c_or, fr)), fr.tolist(), c_sg.tolist(), c_or.tolist()


def calibrate(z, gap, target=0.977):
    k = float(np.quantile(z, target))          # d_hat*exp(-k sigma) <= gt for `target` of pixels
    c = float(max(0.0, np.quantile(gap, target)))
    return k, c


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'dataset'))
    ap.add_argument('--da_repo', default=os.environ.get('DA', ''))
    ap.add_argument('--checkpoint', default='')
    ap.add_argument('--encoder', default='vits')
    ap.add_argument('--input_size', type=int, default=518,
                    help='MUST match the input_size used in navigation (eval_stack default 518)')
    ap.add_argument('--train_worlds', default='train_01,train_02,train_04,train_05')
    ap.add_argument('--val_worlds', default='train_03')
    ap.add_argument('--out', default=os.path.join(HERE, 'runs', 'head_v1'))
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--bs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--beta', type=float, default=0.5, help='beta-NLL; 0 = plain NLL')
    ap.add_argument('--no_near_weight', action='store_true')
    ap.add_argument('--coverage', type=float, default=0.977, help='target for k and c')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--val_stride', type=int, default=5,
                    help='use every n-th val frame (neighbouring frames are near-duplicates)')
    ap.add_argument('--eval_every', type=int, default=500, help='validate every N training steps')
    ap.add_argument('--quick_val_stride', type=int, default=5,
                    help='periodic validation uses every n-th val frame (full val at the end)')
    ap.add_argument('--skip_initial_eval', action='store_true',
                    help='skip the raw-DA2 baseline evaluation before training')
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True
    ckpt = a.checkpoint or os.path.join(a.da_repo, 'checkpoints',
                                        f'depth_anything_v2_metric_hypersim_{a.encoder}.pth')
    os.makedirs(a.out, exist_ok=True)
    tr = FrameSet(a.data, a.train_worlds.split(','), augment=True)
    va = FrameSet(a.data, a.val_worlds.split(','), augment=False, stride=a.val_stride)
    print(f'train {len(tr)} frames {a.train_worlds} | val {len(va)} frames {a.val_worlds} | device: {dev}')
    if dev == 'cpu':
        print('WARNING: running on CPU - this will be extremely slow. Check torch.cuda.is_available().')
    tl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    vl = DataLoader(va, a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)

    H, W = tr[0][0].shape[-2:]
    size = net_size(H, W, a.input_size)
    da2 = DA2Features(a.da_repo, ckpt, a.encoder).to(dev)
    head = UncertaintyHead().to(dev)
    print(f'image {H}x{W} -> DA2 {size}; head params {sum(p.numel() for p in head.parameters()) / 1e3:.0f}k')
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * len(tl), pct_start=0.1)

    meta = {'input_size': a.input_size, 'encoder': a.encoder, 'net_size': list(size),
            'train_worlds': a.train_worlds, 'val_worlds': a.val_worlds, 'beta': a.beta,
            'head_args': {}}
    if not a.skip_initial_eval:
        print('baseline evaluation of raw DA2 on val (head = identity) ...')
        m0, z, gap, es = evaluate(da2, head, vl, size, dev)
        print('epoch 0 (identity head):', {k: round(v, 4) for k, v in m0.items()})
    # quick periodic validation on a subset; the full val set is used for calibration at the end
    vq = DataLoader(torch.utils.data.Subset(va, list(range(0, len(va), a.quick_val_stride))),
                    a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
    print(f'validation every {a.eval_every} steps on {len(vq.dataset)} frames')
    best, log, step = math.inf, [], 0
    t_start = time.time()

    def validate(ep, run_loss, n_loss):
        nonlocal best
        head.eval()
        t0 = time.time()
        m, _, _, _ = evaluate(da2, head, vq, size, dev)
        m.update({'step': step, 'epoch': ep, 'train_loss': run_loss / max(n_loss, 1),
                  'eval_s': round(time.time() - t0), 'elapsed_min': round((time.time() - t_start) / 60)})
        log.append(m)
        tag = ''
        if m['nll'] < best:                    # val NLL = fit of both mu and sigma
            best = m['nll']
            torch.save({'head': head.state_dict(), **meta, 'epoch': ep, 'step': step, 'val': m},
                       os.path.join(a.out, 'head_best.pt'))
            tag = '  <- best, saved'
        print(f"VAL step {step} (ep {ep}): nll {m['nll']:.4f} | absrel raw {m['absrel_raw']:.3f} "
              f"-> cor {m['absrel_cor']:.3f} | overest_near raw {m['overest_near_raw']:.3f} "
              f"-> cor {m['overest_near_cor']:.3f} | sigma {m['mean_sigma']:.3f} | "
              f"train {m['train_loss']:.4f}{tag}", flush=True)
        json.dump(log, open(os.path.join(a.out, 'train_log.json'), 'w'), indent=1)
        head.train()

    for ep in range(1, a.epochs + 1):
        head.train()
        tot, nb = 0.0, 0
        for rgb, gt in tl:
            rgb, gt = rgb.to(dev, non_blocking=True), gt.to(dev, non_blocking=True)
            d_da, mu, s = forward(da2, head, rgb, size)
            loss = loss_fn(d_da, mu, s, gt, a.beta, near_w=not a.no_near_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
            step += 1
            if nb % 100 == 0:
                print(f'  ep {ep} it {nb}/{len(tl)} loss {tot / nb:.4f}', flush=True)
            if step % a.eval_every == 0:
                validate(ep, tot, nb)
        if step % a.eval_every != 0:           # always validate at the end of an epoch
            validate(ep, tot, nb)

    # ---- calibration on val with the best head
    ck = torch.load(os.path.join(a.out, 'head_best.pt'), map_location='cpu')
    head.load_state_dict(ck['head'])
    head.eval()
    m, z, gap, es = evaluate(da2, head, vl, size, dev, n_sample=20000)
    k, c = calibrate(z, gap, a.coverage)
    A, fr, cs, co = ause(es[:, 0], es[:, 1])
    cov_k2 = float((z <= 2.0).mean())
    calib = {'k': k, 'margin_c': c, 'coverage_target': a.coverage,
             'coverage_at_k2': cov_k2, 'ause': A, 'val_metrics': m,
             'raw_overest_near_at_c0': float((gap > 0).mean()),
             'sparsification': {'frac_removed': fr, 'by_sigma': cs, 'oracle': co}}
    ck.update({'k': k, 'margin_c': c})
    torch.save(ck, os.path.join(a.out, 'head_best.pt'))
    json.dump(calib, open(os.path.join(a.out, 'calib.json'), 'w'), indent=1)
    print('\n=== calibration on', a.val_worlds, '===')
    print(f'k = {k:.2f}  (coverage of d_safe with k=2 would be {cov_k2:.3f})')
    print(f'constant-margin baseline c = {c:.3f}  ->  d_DA * {math.exp(-c):.2f}')
    print(f'AUSE = {A:.4f} (lower = sigma ranks errors better)')
    print({kk: round(v, 4) for kk, v in m.items()})
    print(f'saved {a.out}/head_best.pt and calib.json')


if __name__ == '__main__':
    main()
