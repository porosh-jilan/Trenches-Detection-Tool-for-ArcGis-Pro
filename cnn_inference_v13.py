# -*- coding: utf-8 -*-
"""CNN INFERENCE - pure NumPy forward pass of the trained U-Net.

ArcMap 10.8 runs Python 2.7 and has NO PyTorch. This module reproduces
the U-Net forward pass with NumPy only, reading the weights exported by
cnn_train.py (cnn_unet_weights.npz). Python 2.7 compatible.
"""
from __future__ import print_function
import numpy as np

# Must match cnn_dataprep.py
TILE = 256


# ----------------------------------------------------- feature stack
def _gaussian_blur(a, sigma):
    if sigma <= 0.3:
        return a.copy()
    rad = int(max(1, round(3 * sigma)))
    x = np.arange(-rad, rad + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    pad = np.pad(a, ((rad, rad), (0, 0)), mode='edge')
    out = np.zeros_like(a)
    for i, kv in enumerate(k):
        out += kv * pad[i:i + a.shape[0], :]
    pad = np.pad(out, ((0, 0), (rad, rad)), mode='edge')
    out2 = np.zeros_like(a)
    for i, kv in enumerate(k):
        out2 += kv * pad[:, i:i + a.shape[1]]
    return out2


def _box_blur_5(a):
    """5x5 uniform box mean (edge-padded). numpy 1.9 safe."""
    out = np.zeros_like(a, dtype=np.float32)
    pad = np.pad(a, ((2, 2), (2, 2)), mode='edge').astype(np.float32)
    for i in range(5):
        for j in range(5):
            out += pad[i:i + a.shape[0], j:j + a.shape[1]]
    return (out / 25.0).astype(np.float32)


def compute_features(dem):
    """v13: 7-channel stack identical to cnn_dataprep.compute_features.
    Channels: detrend_4, detrend_12, detrend_24, slope, lap, locallow_3,
    locallow_5."""
    a = dem.astype(np.float32)
    detrend_4  = a - _gaussian_blur(a, 4.0)
    detrend_12 = a - _gaussian_blur(a, 12.0)
    detrend_24 = a - _gaussian_blur(a, 24.0)
    e = np.empty_like(a); e[:, :-1] = a[:, 1:]; e[:, -1] = a[:, -1]
    w = np.empty_like(a); w[:, 1:] = a[:, :-1]; w[:, 0] = a[:, 0]
    n = np.empty_like(a); n[:-1, :] = a[1:, :]; n[-1, :] = a[-1, :]
    s = np.empty_like(a); s[1:, :] = a[:-1, :]; s[0, :] = a[0, :]
    gx = (e - w) * 0.5
    gy = (s - n) * 0.5
    slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    lap = (e + w + n + s - 4.0 * a).astype(np.float32)
    ne = np.empty_like(a); ne[:-1,1:]=a[1:,:-1]; ne[-1,:]=a[-1,:]; ne[:,0]=a[:,0]
    nw = np.empty_like(a); nw[:-1,:-1]=a[1:,1:]; nw[-1,:]=a[-1,:]; nw[:,-1]=a[:,-1]
    se = np.empty_like(a); se[1:,1:]=a[:-1,:-1]; se[0,:]=a[0,:]; se[:,-1]=a[:,-1]
    sw = np.empty_like(a); sw[1:,:-1]=a[:-1,1:]; sw[0,:]=a[0,:]; sw[:,0]=a[:,0]
    neigh = (n + s + e + w + ne + nw + se + sw) / 8.0
    locallow_3 = (neigh - a).astype(np.float32)
    locallow_5 = (_box_blur_5(a) - a).astype(np.float32)
    # explicit assignment (numpy 1.9 in ArcMap has no np.stack)
    out = np.empty((7,) + a.shape, dtype=np.float32)
    out[0] = detrend_4
    out[1] = detrend_12
    out[2] = detrend_24
    out[3] = slope
    out[4] = lap
    out[5] = locallow_3
    out[6] = locallow_5
    return out


# ------------------------------------------------------ NN ops (numpy)
def _conv2d(x, w, b, pad):
    """x:(Cin,H,W) w:(Cout,Cin,kh,kw) b:(Cout,) -> (Cout,H',W').
    im2col + matmul. 'same' conv when pad == kh//2."""
    cin, h, wid = x.shape
    cout, _, kh, kw = w.shape
    if pad > 0:
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode='constant')
    else:
        xp = x
    hp, wp = xp.shape[1], xp.shape[2]
    oh = hp - kh + 1
    ow = wp - kw + 1
    cols = np.empty((cin * kh * kw, oh * ow), dtype=np.float32)
    idx = 0
    for ci in range(cin):
        for i in range(kh):
            for j in range(kw):
                cols[idx] = xp[ci, i:i + oh, j:j + ow].reshape(-1)
                idx += 1
    wm = w.reshape(cout, -1).astype(np.float32)
    out = np.dot(wm, cols) + b.reshape(-1, 1)
    return out.reshape(cout, oh, ow)


def _relu(x):
    return np.maximum(x, 0.0, out=x)


def _maxpool2(x):
    c, h, w = x.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    xc = x[:, :h2, :w2]
    xr = xc.reshape(c, h2 // 2, 2, w2 // 2, 2)
    return xr.max(axis=4).max(axis=2)


def _upsample2(x):
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


# ------------------------------------------------------ U-Net forward
def _conv_block(x, W, prefix):
    """Sequential: Conv(0)+ReLU+Conv(2)+ReLU. Keys prefix__0__weight etc."""
    x = _conv2d(x, W[prefix + '__0__weight'], W[prefix + '__0__bias'], 1)
    x = _relu(x)
    x = _conv2d(x, W[prefix + '__2__weight'], W[prefix + '__2__bias'], 1)
    x = _relu(x)
    return x


def _unet_decode(feat, W):
    """Shared encoder/decoder. Returns d1 (final 24-ch decoder feature map).
    4-level U-Net matching cnn_train.UNet."""
    s1 = _conv_block(feat, W, 'e1')
    s2 = _conv_block(_maxpool2(s1), W, 'e2')
    s3 = _conv_block(_maxpool2(s2), W, 'e3')
    s4 = _conv_block(_maxpool2(s3), W, 'e4')
    b = _conv_block(_maxpool2(s4), W, 'bott')
    u4 = _conv2d(_upsample2(b), W['up4__weight'], W['up4__bias'], 0)
    d4 = _conv_block(np.concatenate([u4, s4], axis=0), W, 'd4')
    u3 = _conv2d(_upsample2(d4), W['up3__weight'], W['up3__bias'], 0)
    d3 = _conv_block(np.concatenate([u3, s3], axis=0), W, 'd3')
    u2 = _conv2d(_upsample2(d3), W['up2__weight'], W['up2__bias'], 0)
    d2 = _conv_block(np.concatenate([u2, s2], axis=0), W, 'd2')
    u1 = _conv2d(_upsample2(d2), W['up1__weight'], W['up1__bias'], 0)
    d1 = _conv_block(np.concatenate([u1, s1], axis=0), W, 'd1')
    return d1


def unet_forward(feat, W):
    """feat: (4, H, W) already normalised. Returns segmentation prob (H, W)."""
    d1 = _unet_decode(feat, W)
    logit = _conv2d(d1, W['out__weight'], W['out__bias'], 0)
    return _sigmoid(logit[0])


def has_junction_head(W):
    return 'out_junc__weight' in W


def unet_forward_with_junc(feat, W):
    """v12: returns (seg_prob, junc_prob), both (H, W). Requires the
    junction-head weights (out_junc__*)."""
    d1 = _unet_decode(feat, W)
    seg = _sigmoid(_conv2d(d1, W['out__weight'], W['out__bias'], 0)[0])
    junc = _sigmoid(_conv2d(d1, W['out_junc__weight'],
                            W['out_junc__bias'], 0)[0])
    return seg, junc


# ------------------------------------------------ test-time augmentation
def _tta_list(n):
    """Dihedral-group transforms as (k_rot90, flip_rows) pairs.
    n>=8 -> all 8; n==4 -> 4 rotations; else 2 (identity + 180)."""
    rots = [(0, False), (1, False), (2, False), (3, False)]
    flips = [(0, True), (1, True), (2, True), (3, True)]
    if n >= 8:
        return rots + flips
    if n >= 4:
        return rots
    if n >= 2:
        return [(0, False), (2, False)]
    return [(0, False)]


def unet_forward_tta(feat, W, n=4):
    """Average the U-Net output over n dihedral views of the input. All 4
    feature channels are scalar fields (detrend, slope magnitude, laplacian,
    local depression) so a spatial flip/rotation of the stack is exact - no
    directional sign flips needed. numpy-1.9 safe (np.rot90 only; flips via
    slicing, not np.flip)."""
    transforms = _tta_list(int(n))
    if len(transforms) <= 1:
        return unet_forward(feat, W)
    acc = None
    for k, flip in transforms:
        x = feat
        if flip:
            x = x[:, ::-1, :]
        if k:
            x = np.rot90(x, k, axes=(1, 2))
        x = np.ascontiguousarray(x.astype(np.float32))
        p = unet_forward(x, W)            # (H, W) in the transformed frame
        if k:
            p = np.rot90(p, (4 - k) % 4)  # undo rotation
        if flip:
            p = p[::-1, :]                # undo flip
        p = np.ascontiguousarray(p.astype(np.float32))
        acc = p if acc is None else acc + p
    return (acc / float(len(transforms))).astype(np.float32)


def load_weights(npz_path):
    d = np.load(npz_path)
    return {k: d[k].astype(np.float32) for k in d.files}


def predict_probability_map(dem, valid, weights, norm_mean, norm_std,
                            overlap=32, msg=None, tta=0, want_junc=False):
    """Tile a whole DEM through the U-Net, return a stitched trench
    probability map the same size as `dem`.

    dem, valid : (H, W). weights : dict from load_weights.
    norm_mean, norm_std : (4,) channel normalisation from norm_stats.npz.
    tta : 0/1 = single pass (fast); 4 = average 4 rotations; 8 = full
          dihedral group (smoothest, ~tta x slower).
    want_junc : if True AND the weights have a junction head, also stitch
          and return the junction probability map -> returns (seg, junc).
          (junction map is single-pass; TTA applies to segmentation only.)
    """
    H, W = dem.shape
    nodata = -9999999.0
    dem_f = np.where(valid, dem, 0.0).astype(np.float32)
    if np.any(valid):
        med = float(np.median(dem[valid]))
        dem_f = np.where(valid, dem, med).astype(np.float32)

    do_junc = bool(want_junc) and has_junction_head(weights)
    prob = np.zeros((H, W), dtype=np.float32)
    jrob = np.zeros((H, W), dtype=np.float32) if do_junc else None
    wsum = np.zeros((H, W), dtype=np.float32)
    nm = norm_mean.reshape(-1, 1, 1)
    ns = norm_std.reshape(-1, 1, 1)
    step = TILE - overlap

    # cosine-ish weight window so tile seams blend
    win1d = np.ones(TILE, dtype=np.float32)
    for i in range(overlap):
        win1d[i] = (i + 1.0) / (overlap + 1.0)
        win1d[TILE - 1 - i] = (i + 1.0) / (overlap + 1.0)
    win2d = np.outer(win1d, win1d)

    n_tiles = 0
    for r0 in range(0, max(1, H - 1), step):
        for c0 in range(0, max(1, W - 1), step):
            r1 = min(H, r0 + TILE)
            c1 = min(W, c0 + TILE)
            th = r1 - r0
            tw = c1 - c0
            sub = dem_f[r0:r1, c0:c1]
            if th < TILE or tw < TILE:
                # pad to TILE with edge values
                sub = np.pad(sub, ((0, TILE - th), (0, TILE - tw)),
                             mode='edge')
            feat = compute_features(sub)
            feat = (feat - nm) / ns
            feat = feat.astype(np.float32)
            if do_junc:
                pmap, jm = unet_forward_with_junc(feat, weights)
            elif tta and int(tta) > 1:
                pmap = unet_forward_tta(feat, weights, int(tta))
            else:
                pmap = unet_forward(feat, weights)
            wcore = win2d[:th, :tw]
            prob[r0:r1, c0:c1] += pmap[:th, :tw] * wcore
            if do_junc:
                jrob[r0:r1, c0:c1] += jm[:th, :tw] * wcore
            wsum[r0:r1, c0:c1] += wcore
            n_tiles += 1
            if msg is not None and n_tiles % 25 == 0:
                msg("  CNN tiles processed: %d" % n_tiles)
    ok = wsum > 1e-6
    prob[ok] /= wsum[ok]
    prob[~valid] = 0.0
    if do_junc:
        jrob[ok] /= wsum[ok]
        jrob[~valid] = 0.0
        return prob, jrob
    return prob


# -------------------------------------------------- self-test
if __name__ == "__main__":
    import os
    cnn_dir = os.path.dirname(os.path.abspath(__file__))
    W = load_weights(os.path.join(cnn_dir, "cnn_unet_weights.npz"))
    print("Loaded %d weight arrays" % len(W))
    for k in sorted(W.keys()):
        print("  %-22s %s" % (k, W[k].shape))
    # random sanity forward
    nrm = np.load(os.path.join(cnn_dir, "norm_stats.npz"))
    feat = np.random.randn(4, TILE, TILE).astype(np.float32)
    p = unet_forward(feat, W)
    print("Forward OK, prob map shape", p.shape,
          "range [%.3f, %.3f]" % (p.min(), p.max()))
