"""Uncertainty head on top of a frozen Depth Anything V2 (metric) model.

Shared by train_head.py (training) and mono_depth_safe.py (ROS node), so
preprocessing, feature extraction and the head are guaranteed identical.

    RGB -> DA2 (frozen) -> d_DA                      (metric depth, net resolution)
                 |
       32-ch decoder features (input of depth_head.scratch.output_conv2)
                 |
    [features 32, log d_DA 1, RGB 3] = 36 ch @ 1/2 net res -> head -> (mu, s)

    log d_hat = log d_DA + mu          corrected depth
    sigma     = exp(s)                 std of the log-depth error
    d_safe    = d_DA * exp(mu - k*sigma)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
FEAT_CH = 32                 # channels entering output_conv2 (features // 2 for vits)
S_MIN, S_MAX = -5.0, 1.5     # clamp of s = log sigma (sigma in [0.007, 4.5])
S_INIT = -1.6                # initial sigma ~0.2 (20 % depth error)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- DA2 wrapper
def net_size(h, w, input_size):
    """Same rule as mono_depth.py: short side -> input_size, multiples of 14."""
    s = input_size / min(h, w)
    return (max(14, round(h * s / 14) * 14), max(14, round(w * s / 14) * 14))


class DA2Features(nn.Module):
    """Frozen DA2. forward(x_norm) -> (depth (B,H',W'), features (B,32,H',W'))."""

    def __init__(self, repo_path, checkpoint, encoder='vits', max_depth=20.0):
        super().__init__()
        sys.path.insert(0, os.path.join(os.path.expanduser(repo_path), 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2
        m = DepthAnythingV2(**MODEL_CONFIGS[encoder], max_depth=max_depth)
        m.load_state_dict(torch.load(os.path.expanduser(checkpoint), map_location='cpu'))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        self.model = m
        self._feat = None
        m.depth_head.scratch.output_conv2.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self._feat = inputs[0]

    def forward(self, x):
        d = self.model(x)                       # (B, H', W'), meters
        f = self._feat
        self._feat = None
        return d, f

    def train(self, mode=True):                 # always stay in eval mode
        return super().train(False)


def preprocess(rgb_u8, size):
    """rgb_u8: (B,3,H,W) uint8 RGB tensor on device -> normalized float at net size."""
    x = rgb_u8.float().div(255.0)
    x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    mean = x.new_tensor(MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------- head
def conv_gn(cin, cout, stride=1, dil=1):
    g = 8 if cout % 8 == 0 else 1
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, padding=dil, dilation=dil, bias=False),
                         nn.GroupNorm(g, cout), nn.ReLU(inplace=True))


class DilBlock(nn.Module):
    """Residual stack of dilated convs: wide context at low cost (glass, blank walls)."""

    def __init__(self, c, dils):
        super().__init__()
        self.convs = nn.ModuleList([conv_gn(c, c, dil=d) for d in dils])

    def forward(self, x):
        for c in self.convs:
            x = x + c(x)
        return x


class UncertaintyHead(nn.Module):
    """Mini U-Net, 36 -> 2 channels (mu, s), runs at 1/2 of the DA2 network resolution."""

    def __init__(self, in_ch=FEAT_CH + 1 + 3, c1=48, c2=64, c3=96):
        super().__init__()
        self.stem = nn.Sequential(conv_gn(in_ch, c1), conv_gn(c1, c1))           # 1/2
        self.down1 = nn.Sequential(conv_gn(c1, c2, stride=2), DilBlock(c2, (1, 2)))   # 1/4
        self.down2 = nn.Sequential(conv_gn(c2, c3, stride=2), DilBlock(c3, (2, 4, 8)))  # 1/8
        self.up2 = conv_gn(c3 + c2, c2)
        self.up1 = conv_gn(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 2, 1)
        nn.init.zeros_(self.out.weight)            # start as identity: mu = 0, sigma = 0.2
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([0.0, S_INIT]))

    @staticmethod
    def make_input(feat, d_da, x_norm):
        """Build the 36-ch input at half the network resolution."""
        h, w = feat.shape[-2] // 2, feat.shape[-1] // 2
        f = F.interpolate(feat.float(), size=(h, w), mode='bilinear', align_corners=False)
        ld = torch.log(d_da.float().clamp(min=0.1))[:, None]
        ld = F.interpolate(ld, size=(h, w), mode='bilinear', align_corners=False) - 1.0
        rgb = F.interpolate(x_norm.float(), size=(h, w), mode='bilinear', align_corners=False)
        return torch.cat([f, ld, rgb], 1)

    def forward(self, inp):
        e1 = self.stem(inp)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        u2 = self.up2(torch.cat([F.interpolate(e3, size=e2.shape[-2:], mode='bilinear',
                                               align_corners=False), e2], 1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=e1.shape[-2:], mode='bilinear',
                                               align_corners=False), e1], 1))
        o = self.out(u1)
        mu = o[:, 0]
        s = o[:, 1].clamp(S_MIN, S_MAX)
        return mu, s


def upsample(t, size):
    """(B,h,w) -> (B,H,W) bilinear."""
    return F.interpolate(t[:, None].float(), size=size, mode='bilinear', align_corners=False)[:, 0]


def load_head(path, device):
    ck = torch.load(os.path.expanduser(path), map_location='cpu')
    head = UncertaintyHead(**ck.get('head_args', {}))
    head.load_state_dict(ck['head'])
    return head.to(device).eval(), ck
