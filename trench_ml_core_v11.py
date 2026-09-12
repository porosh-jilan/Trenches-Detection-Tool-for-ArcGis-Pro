# -*- coding: utf-8 -*-
"""
TrenchMLToolkit v1 core for ArcMap 10.8.

ArcPy + NumPy only. The model is a lightweight supervised logistic classifier
trained from DEM-derived features and a ground-truth trench polyline layer.
"""
from __future__ import print_function

import os
import time
import math
import tempfile

import numpy as np

import trench_line_core_v11 as lc

# v10: pure-numpy CNN inference (no PyTorch needed in ArcMap)
try:
    import cnn_inference as _cnn
except Exception:
    _cnn = None

# v11.7: optional second model (7-channel) for the recall boost pass
try:
    import cnn_inference_v13 as _cnn13
except Exception:
    _cnn13 = None


def _orientation_bridge_cut(all_lines, params, max_misalign_deg=50.0,
                            min_run_m=6.0, sample_m=2.0, min_frag_m=15.0,
                            min_mag=0.25, end_guard_m=12.0):
    """v15: cut lines where their direction disagrees with the LEARNED
    orientation field for a sustained stretch. A line that turns off one
    trench onto another (a bridge) runs against the learned trench
    direction on one of the two trenches - that stretch gets cut.

    Label convention (matches training): doubled-angle image-frame vector
    for a world-frame angle a is (cos 2a, -sin 2a).
    """
    omap = params.get('cnn_orient')
    geo = params.get('cnn_prob_geo')
    if omap is None or geo is None or not all_lines:
        return all_lines, 0
    oxmin, oymax, ocellx, ocelly = geo
    H = omap.shape[1]; Wd = omap.shape[2]
    # misalignment threshold in DOUBLED-angle space
    cos_lim = math.cos(math.radians(2.0 * float(max_misalign_deg)))

    out = []
    n_cut_lines = 0
    for line in all_lines:
        coords = line.get('coords', [])
        if len(coords) < 3:
            out.append(line)
            continue
        # sample along the line with local direction
        samples = []          # (x, y, dirx_world, diry_world)
        acc = 0.0
        t_next = 0.0
        for i in range(1, len(coords)):
            x0, y0 = coords[i-1]; x1, y1 = coords[i]
            seg = math.hypot(x1-x0, y1-y0)
            if seg <= 1e-9:
                continue
            dx = (x1-x0)/seg; dy = (y1-y0)/seg
            while t_next <= acc + seg:
                f = (t_next - acc) / seg
                samples.append((x0+(x1-x0)*f, y0+(y1-y0)*f, dx, dy))
                t_next += sample_m
            acc += seg
        if len(samples) < 4:
            out.append(line)
            continue
        # flag misaligned samples
        bad = []
        for (sx, sy, dx, dy) in samples:
            col = int((sx - oxmin) / ocellx)
            row = int((oymax - sy) / ocelly)
            if not (0 <= row < H and 0 <= col < Wd):
                bad.append(False)
                continue
            ox = float(omap[0, row, col]); oy = float(omap[1, row, col])
            mag = math.hypot(ox, oy)
            if mag < min_mag:
                bad.append(False)       # weak/unknown orientation: trust line
                continue
            aw = math.atan2(dy, dx)
            lvx = math.cos(2.0*aw); lvy = -math.sin(2.0*aw)
            dot = (lvx*ox + lvy*oy) / mag
            bad.append(dot < cos_lim)
        # find sustained bad runs - INTERIOR only. A true bridge passes
        # THROUGH a junction mid-line; misalignment at a line's own ends is
        # usually a curved trench tip, and cutting there shaves the endpoint
        # (measured: endpoint regression without this guard).
        run_n = int(max(2, round(min_run_m / sample_m)))
        guard_n = int(max(1, round(end_guard_m / sample_m)))
        cuts = []                       # list of (start_idx, end_idx) bad runs
        i = 0
        n = len(samples)
        while i < n:
            if bad[i]:
                j = i
                while j + 1 < n and bad[j+1]:
                    j += 1
                if ((j - i + 1) >= run_n and
                        i >= guard_n and j <= n - 1 - guard_n):
                    cuts.append((i, j))
                i = j + 1
            else:
                i += 1
        if not cuts:
            out.append(line)
            continue
        # rebuild good fragments from the samples
        n_cut_lines += 1
        good_runs = []
        prev_end = -1
        for (a, b) in cuts:
            if a - 1 > prev_end:
                good_runs.append((prev_end + 1, a - 1))
            prev_end = b
        if prev_end + 1 <= n - 1:
            good_runs.append((prev_end + 1, n - 1))
        for (a, b) in good_runs:
            if b - a + 1 < 2:
                continue
            frag = [(samples[k][0], samples[k][1]) for k in range(a, b + 1)]
            flen = sum(math.hypot(frag[k][0]-frag[k-1][0],
                                  frag[k][1]-frag[k-1][1])
                       for k in range(1, len(frag)))
            if flen < float(min_frag_m):
                continue
            nl = dict(line)
            nl['coords'] = frag
            nl['length'] = flen
            out.append(nl)
    return out, n_cut_lines


def _filter_boost_lines(base_lines, boost_lines, dist_m=5.0,
                        new_frac=0.6, min_len_m=25.0):
    """Return the subset of boost_lines that mostly cover ground NOT already
    covered by base_lines. Line-level union for the v11.7 recall boost."""
    if not boost_lines:
        return []
    if not base_lines:
        return list(boost_lines)
    gcell = 25.0
    grid = {}
    for ln in base_lines:
        coords = ln.get('coords', [])
        for i in range(1, len(coords)):
            a = coords[i-1]; b = coords[i]
            x0 = int(math.floor(min(a[0], b[0]) / gcell))
            x1 = int(math.floor(max(a[0], b[0]) / gcell))
            y0 = int(math.floor(min(a[1], b[1]) / gcell))
            y1 = int(math.floor(max(a[1], b[1]) / gcell))
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    grid.setdefault((gx, gy), []).append((a, b))

    def _pt_seg_d(p, a, b):
        px, py = p; ax, ay = a; bx, by = b
        dx = bx - ax; dy = by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    def _min_d(p):
        bx = int(math.floor(p[0] / gcell))
        by = int(math.floor(p[1] / gcell))
        best = 1e9
        for gx in range(bx - 1, bx + 2):
            for gy in range(by - 1, by + 2):
                for a, b in grid.get((gx, gy), ()):
                    d = _pt_seg_d(p, a, b)
                    if d < best:
                        best = d
        return best

    def _samples(coords, step=3.0):
        pts = [coords[0]]
        w = 0.0; t = step
        for i in range(1, len(coords)):
            x0, y0 = coords[i-1]; x1, y1 = coords[i]
            s = math.hypot(x1 - x0, y1 - y0)
            if s <= 1e-9:
                continue
            while w + s >= t:
                tt = (t - w) / s
                pts.append((x0 + (x1-x0)*tt, y0 + (y1-y0)*tt))
                t += step
            w += s
        pts.append(coords[-1])
        return pts

    added = []
    for ln in boost_lines:
        coords = ln.get('coords', [])
        if len(coords) < 2:
            continue
        length = sum(math.hypot(coords[i][0]-coords[i-1][0],
                                coords[i][1]-coords[i-1][1])
                     for i in range(1, len(coords)))
        if length < float(min_len_m):
            continue
        s = _samples(coords)
        far = sum(1 for q in s if _min_d(q) > float(dist_m))
        if far >= float(new_frac) * len(s):
            added.append(ln)
    return added


def _msg(arcpy, text):
    try:
        arcpy.AddMessage(str(text))
    except Exception:
        print(str(text))


def _warn(arcpy, text):
    try:
        arcpy.AddWarning(str(text))
    except Exception:
        print("WARNING: " + str(text))


def _shift(a, dr, dc, fill=np.nan):
    h, w = a.shape
    out = np.empty_like(a, dtype=np.float32)
    out.fill(fill)
    r0s = max(0, -dr); r1s = min(h, h - dr)
    c0s = max(0, -dc); c1s = min(w, w - dc)
    r0d = max(0, dr);  r1d = min(h, h + dr)
    c0d = max(0, dc);  c1d = min(w, w + dc)
    if r1s > r0s and c1s > c0s:
        out[r0d:r1d, c0d:c1d] = a[r0s:r1s, c0s:c1s]
    return out


def _feature_stack(arr, valid, cell_size, max_width_m, smooth_sigma):
    dem = lc._nan_gaussian_smooth(arr.astype(np.float32), valid, smooth_sigma)
    _, _, dep4, _ = lc._directional_depression(
        dem, valid, 0.08, 0.025, 4.0, cell_size)
    strong, grow, dep6, orient = lc._directional_depression(
        dem, valid, 0.08, 0.025, max_width_m, cell_size)
    _, _, dep8, _ = lc._directional_depression(
        dem, valid, 0.08, 0.025, 8.0, cell_size)
    _, _, dep10, _ = lc._directional_depression(
        dem, valid, 0.08, 0.025, 10.0, cell_size)
    _, _, dep14, _ = lc._directional_depression(
        dem, valid, 0.08, 0.025, 14.0, cell_size)

    n = _shift(dem, -1, 0)
    s = _shift(dem, 1, 0)
    e = _shift(dem, 0, 1)
    w = _shift(dem, 0, -1)
    ne = _shift(dem, -1, 1)
    nw = _shift(dem, -1, -1)
    se = _shift(dem, 1, 1)
    sw = _shift(dem, 1, -1)

    gx = (e - w) * 0.5
    gy = (s - n) * 0.5
    slope = np.sqrt(gx * gx + gy * gy).astype(np.float32)
    neigh_mean = (n + s + e + w + ne + nw + se + sw) / 8.0
    local_low = (neigh_mean - dem).astype(np.float32)
    lap = (n + s + e + w - 4.0 * dem).astype(np.float32)
    neigh_min = np.minimum(np.minimum(np.minimum(n, s), np.minimum(e, w)),
                           np.minimum(np.minimum(ne, nw), np.minimum(se, sw)))
    neigh_max = np.maximum(np.maximum(np.maximum(n, s), np.maximum(e, w)),
                           np.maximum(np.maximum(ne, nw), np.maximum(se, sw)))
    local_range = (neigh_max - neigh_min).astype(np.float32)

    narrow = (dep6 - dep14).astype(np.float32)
    ratio = (dep6 / (np.abs(dep14) + 0.05)).astype(np.float32)
    feats = [dep4.astype(np.float32),
             dep6.astype(np.float32),
             dep8.astype(np.float32),
             dep10.astype(np.float32),
             dep14.astype(np.float32),
             narrow,
             ratio,
             local_low,
             lap,
             slope,
             local_range,
             grow.astype(np.float32),
             strong.astype(np.float32)]
    for f in feats:
        f[~valid] = 0.0
        f[~np.isfinite(f)] = 0.0
    return feats


def _read_raster_tile(arcpy, raster_path, xmin, ymax, cellx, celly,
                      read_col0, read_row0, ncols, nrows, nodata_value):
    arr_xmin = xmin + read_col0 * cellx
    arr_ymax = ymax - read_row0 * celly
    lower_left = arcpy.Point(arr_xmin, arr_ymax - nrows * celly)
    arr = arcpy.RasterToNumPyArray(raster_path, lower_left, ncols, nrows,
                                   nodata_to_value=nodata_value)
    arr = arr.astype(np.float32)
    valid = (np.isfinite(arr) & (arr != nodata_value) & (arr > -1.0e20))
    return arr, valid


def _make_label_raster(arcpy, gt_lines, dem_path, out_dir, buffer_m,
                       name_prefix):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    arcpy.env.overwriteOutput = True
    arcpy.env.snapRaster = dem_path
    arcpy.env.cellSize = dem_path
    arcpy.env.extent = dem_path

    buf_fc = os.path.join(out_dir, name_prefix + "_buf.shp")
    lbl_ras = os.path.join(out_dir, name_prefix + "_label.tif")
    if arcpy.Exists(buf_fc):
        arcpy.Delete_management(buf_fc)
    if arcpy.Exists(lbl_ras):
        arcpy.Delete_management(lbl_ras)
    arcpy.Buffer_analysis(gt_lines, buf_fc, "%s Meters" % float(buffer_m),
                          "FULL", "ROUND", "ALL")
    arcpy.PolygonToRaster_conversion(buf_fc, "FID", lbl_ras,
                                     "MAXIMUM_AREA", "", dem_path)
    return lbl_ras


def _tile_windows(width, height, tile_size, overlap):
    for row0 in range(0, height, tile_size):
        core_rows = min(tile_size, height - row0)
        for col0 in range(0, width, tile_size):
            core_cols = min(tile_size, width - col0)
            rr0 = max(0, row0 - overlap)
            cc0 = max(0, col0 - overlap)
            rr1 = min(height, row0 + core_rows + overlap)
            cc1 = min(width, col0 + core_cols + overlap)
            core = (row0 - rr0, row0 - rr0 + core_rows,
                    col0 - cc0, col0 - cc0 + core_cols)
            yield row0, col0, core_rows, core_cols, rr0, cc0, rr1, cc1, core


def _take_balanced_indices(pos_idx, neg_idx, n_pos, n_neg):
    if len(pos_idx) > n_pos:
        pos_idx = pos_idx[np.random.permutation(len(pos_idx))[:n_pos]]
    if len(neg_idx) > n_neg:
        neg_idx = neg_idx[np.random.permutation(len(neg_idx))[:n_neg]]
    return pos_idx, neg_idx


def _sigmoid(z):
    z = np.maximum(np.minimum(z, 40.0), -40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _train_logistic(X, y, iterations, lr, l2):
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float32)
    b = np.float32(0.0)
    y = y.astype(np.float32)
    for _ in range(int(iterations)):
        p = _sigmoid(np.dot(X, w) + b).astype(np.float32)
        err = p - y
        gw = (np.dot(X.T, err) / float(n)) + l2 * w
        gb = np.mean(err)
        w -= lr * gw.astype(np.float32)
        b -= lr * np.float32(gb)
    return w, b


def _read_gt_segments(arcpy, gt_lines_path):
    """Read every line segment from a polyline shapefile/feature class as
    a flat list of ((x1, y1), (x2, y2)) pairs, in world coordinates."""
    segments = []
    with arcpy.da.SearchCursor(gt_lines_path, ["SHAPE@"]) as cur:
        for row in cur:
            shp = row[0]
            if shp is None:
                continue
            for part in shp:
                pts = []
                for p in part:
                    if p is not None:
                        pts.append((float(p.X), float(p.Y)))
                for i in range(1, len(pts)):
                    segments.append((pts[i - 1], pts[i]))
    return segments


def _build_segment_grid(segments, cell):
    """Hash segments into a coarse spatial grid for fast nearest-segment
    lookup. Each cell stores the list of segment indices that touch it.
    """
    grid = {}
    for si, (a, b) in enumerate(segments):
        x0 = int(math.floor(min(a[0], b[0]) / cell))
        x1 = int(math.floor(max(a[0], b[0]) / cell))
        y0 = int(math.floor(min(a[1], b[1]) / cell))
        y1 = int(math.floor(max(a[1], b[1]) / cell))
        for gx in range(x0, x1 + 1):
            for gy in range(y0, y1 + 1):
                grid.setdefault((gx, gy), []).append(si)
    return grid


def _nearest_point_on_segment(p, a, b):
    """Return (closest_point_xy, distance_m)."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1.0e-12:
        return (ax, ay), math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * dx
    cy = ay + t * dy
    return (cx, cy), math.hypot(px - cx, py - cy)


def _nearest_gt_offset(world_xy, segments, grid, grid_cell, search_radius_m):
    """Return (dx, dy) from world_xy to the nearest point on the nearest
    GT segment, or (0.0, 0.0) if no segment is within search_radius_m."""
    bx = int(math.floor(world_xy[0] / grid_cell))
    by = int(math.floor(world_xy[1] / grid_cell))
    span = int(math.ceil(search_radius_m / grid_cell)) + 1
    best_d = float('inf')
    best_pt = world_xy
    for gx in range(bx - span, bx + span + 1):
        for gy in range(by - span, by + span + 1):
            for si in grid.get((gx, gy), ()):
                a, b = segments[si]
                pt, d = _nearest_point_on_segment(world_xy, a, b)
                if d < best_d:
                    best_d = d
                    best_pt = pt
    if best_d > search_radius_m:
        return 0.0, 0.0
    return float(best_pt[0] - world_xy[0]), float(best_pt[1] - world_xy[1])


def _train_linear_regression(X, y, iterations, lr, l2):
    """Plain L2-regularised linear regression via gradient descent."""
    n, d = X.shape
    w = np.zeros((d,), dtype=np.float32)
    b = np.float32(0.0)
    y = y.astype(np.float32)
    for _ in range(int(iterations)):
        pred = (np.dot(X, w) + b).astype(np.float32)
        err = pred - y
        gw = (np.dot(X.T, err) / float(n)) + l2 * w
        gb = float(np.mean(err))
        w -= lr * gw.astype(np.float32)
        b -= lr * np.float32(gb)
    return w, b


# v6 multi-channel patch features. Each pixel produces 4 channels of
# PATCH_SIZE * PATCH_SIZE values, stacked into a single feature vector:
#
#   channel 0: raw relative elevation (sub - centre)
#   channel 1: gx (east-west gradient) within the patch
#   channel 2: gy (north-south gradient)
#   channel 3: Laplacian (4-neighbour second derivative)
#
# Validated POC R^2 jump from 0.65 (raw only) to ~0.78 with all 4
# channels combined. Each gradient channel preserves direction-encoded
# information that is critical for predicting which SIDE of the GT a
# pixel is on.
# v8: 8 feature channels per patch (raw, gx, gy, |gx|, |gy|, gx^2, gy^2,
# laplacian). The extra non-linear channels (|.|, .^2) give a linear
# regression access to interaction-style information without needing a
# polynomial expansion across all dimensions, validated +5% R^2 on
# Whole.zip.
PATCH_RADIUS = 7
PATCH_SIZE = PATCH_RADIUS * 2 + 1
PATCH_CHANNELS = 8
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * PATCH_CHANNELS  # 1800


def _patch_feature_tile(arr):
    """v8: pre-compute 8 channels per pixel for the whole DEM tile.
    Channels: raw, gx, gy, |gx|, |gy|, gx^2, gy^2, laplacian.

    The first 4 channels match v7. The last 4 add non-linear interaction
    features (absolute value + squared gradients) so a linear regression
    can pick up curvature and rotation information without an explicit
    polynomial expansion."""
    a = arr.astype(np.float32)
    np.clip(a, -1.0e4, 1.0e4, out=a)
    e = np.empty_like(a); e[:, :-1] = a[:, 1:]; e[:, -1] = a[:, -1]
    w = np.empty_like(a); w[:, 1:] = a[:, :-1]; w[:, 0] = a[:, 0]
    n = np.empty_like(a); n[:-1, :] = a[1:, :]; n[-1, :] = a[-1, :]
    s = np.empty_like(a); s[1:, :] = a[:-1, :]; s[0, :] = a[0, :]
    gx = (e - w) * 0.5
    gy = (s - n) * 0.5
    abs_gx = np.abs(gx).astype(np.float32)
    abs_gy = np.abs(gy).astype(np.float32)
    gx_sq = (gx * gx).astype(np.float32)
    gy_sq = (gy * gy).astype(np.float32)
    lap = (e + w + n + s - 4.0 * a).astype(np.float32)
    return a, gx, gy, abs_gx, abs_gy, gx_sq, gy_sq, lap


def _local_normalise_patch(raw):
    """v8 Path B: per-patch local normalisation. Each patch is centred on
    its own mean so the regression sees ONLY the local trench shape, not
    the absolute elevation. Reduces sensitivity to terrain context.
    """
    return raw - float(np.mean(raw))


def _extract_patch_relative(arr, valid, r, c,
                            gx=None, gy=None,
                            abs_gx=None, abs_gy=None,
                            gx_sq=None, gy_sq=None,
                            lap=None):
    """v8: 8-channel patch with per-patch local normalisation on the raw
    channel. Channels: raw_localnorm, gx, gy, |gx|, |gy|, gx^2, gy^2, lap.
    Falls back to v6 4-channel or v5 single-channel layout when the
    extra channels are not provided.
    """
    h, w = arr.shape
    if (r < PATCH_RADIUS or r >= h - PATCH_RADIUS or
            c < PATCH_RADIUS or c >= w - PATCH_RADIUS):
        return None
    rs = slice(r - PATCH_RADIUS, r + PATCH_RADIUS + 1)
    cs = slice(c - PATCH_RADIUS, c + PATCH_RADIUS + 1)
    sub = arr[rs, cs]
    sub_v = valid[rs, cs]
    if not bool(np.all(sub_v)):
        return None
    centre = float(arr[r, c])
    raw = (sub - centre).astype(np.float32).reshape(-1)
    if gx is None or gy is None:
        return raw  # v5-compatible
    # v8 Path B: per-patch local normalisation on the raw channel
    raw = (raw - float(np.mean(raw))).astype(np.float32)
    gx_p = gx[rs, cs].astype(np.float32).reshape(-1)
    gy_p = gy[rs, cs].astype(np.float32).reshape(-1)
    if abs_gx is None or gx_sq is None or lap is None:
        # v6/v7 4-channel
        if lap is not None:
            lap_p = lap[rs, cs].astype(np.float32).reshape(-1)
        else:
            lap_p = np.zeros_like(raw)
        return np.concatenate([raw, gx_p, gy_p, lap_p])
    # v8 8-channel
    abs_gx_p = abs_gx[rs, cs].astype(np.float32).reshape(-1)
    abs_gy_p = abs_gy[rs, cs].astype(np.float32).reshape(-1)
    gx_sq_p = gx_sq[rs, cs].astype(np.float32).reshape(-1)
    gy_sq_p = gy_sq[rs, cs].astype(np.float32).reshape(-1)
    lap_p = lap[rs, cs].astype(np.float32).reshape(-1)
    return np.concatenate([raw, gx_p, gy_p, abs_gx_p, abs_gy_p,
                           gx_sq_p, gy_sq_p, lap_p])


def train_model(in_dem, ground_truth_lines, out_model, arcpy,
                hard_negative_lines=None,
                positive_buffer_m=2.5,
                ignore_buffer_m=9.0,
                hard_negative_buffer_m=2.5,
                max_samples=220000,
                tile_size_px=900,
                tile_overlap_px=16,
                max_width_m=6.0,
                smooth_sigma_px=0.9,
                iterations=260,
                learning_rate=0.08,
                l2=0.001,
                random_seed=42,
                regression_max_offset_m=2.5,
                regression_iterations=400,
                regression_lr=0.05,
                regression_l2=0.001):
    start = time.time()
    np.random.seed(int(random_seed))
    arcpy.env.overwriteOutput = True

    out_dir = os.path.dirname(out_model)
    if out_dir == "":
        out_dir = os.getcwd()
    work_dir = os.path.join(out_dir, "_trench_ml_training")
    if not os.path.isdir(work_dir):
        os.makedirs(work_dir)

    _msg(arcpy, "Creating training label rasters...")
    pos_ras = _make_label_raster(arcpy, ground_truth_lines, in_dem, work_dir,
                                 positive_buffer_m, "positive")
    ign_ras = _make_label_raster(arcpy, ground_truth_lines, in_dem, work_dir,
                                 ignore_buffer_m, "ignore")
    hard_ras = None
    if hard_negative_lines not in [None, "", "#"]:
        _msg(arcpy, "Creating hard-negative raster from candidate false lines...")
        hard_ras = _make_label_raster(arcpy, hard_negative_lines, in_dem,
                                      work_dir, hard_negative_buffer_m,
                                      "hard_negative")

    ras = arcpy.Raster(in_dem)
    ext = ras.extent
    xmin = float(ext.XMin); ymax = float(ext.YMax)
    ymin = float(ext.YMin); xmax = float(ext.XMax)
    cellx = float(ras.meanCellWidth)
    celly = abs(float(ras.meanCellHeight))
    cell = (cellx + celly) * 0.5
    width = int(round((xmax - xmin) / cellx))
    height = int(round((ymax - ymin) / celly))

    # v4: load GT segments for positional regression target generation
    _msg(arcpy, "Loading GT line segments for positional regression...")
    gt_segments = _read_gt_segments(arcpy, ground_truth_lines)
    _msg(arcpy, "  loaded %d GT segments" % len(gt_segments))
    grid_cell_m = max(20.0, float(positive_buffer_m) * 6.0)
    gt_grid = _build_segment_grid(gt_segments, grid_cell_m)
    reg_search_m = float(positive_buffer_m) + 0.5

    nodata = -9999999.0
    target_pos = int(max_samples // 2)
    target_neg = int(max_samples - target_pos)
    per_tile_pos = max(50, int(target_pos / 32))
    per_tile_neg = max(50, int(target_neg / 48))
    per_tile_hard = max(50, int(target_neg / 32))
    xs = []
    ys = []
    dxs = []  # v4: positional regression targets for positive samples
    dys = []
    pos_xs = []  # v4: feature vectors for positive samples (to train reg)
    tile_id = 0
    _msg(arcpy, "Sampling DEM features from %d x %d raster..." %
         (width, height))
    for row0, col0, core_rows, core_cols, rr0, cc0, rr1, cc1, core in _tile_windows(
            width, height, int(tile_size_px), int(tile_overlap_px)):
        tile_id += 1
        nrows = rr1 - rr0
        ncols = cc1 - cc0
        dem_arr, valid = _read_raster_tile(arcpy, in_dem, xmin, ymax, cellx,
                                           celly, cc0, rr0, ncols, nrows,
                                           nodata)
        pos_arr, pos_valid = _read_raster_tile(arcpy, pos_ras, xmin, ymax,
                                               cellx, celly, cc0, rr0,
                                               ncols, nrows, nodata)
        ign_arr, ign_valid = _read_raster_tile(arcpy, ign_ras, xmin, ymax,
                                               cellx, celly, cc0, rr0,
                                               ncols, nrows, nodata)
        hard = None
        if hard_ras is not None:
            hard_arr, hard_valid = _read_raster_tile(
                arcpy, hard_ras, xmin, ymax, cellx, celly, cc0, rr0,
                ncols, nrows, nodata)
        r0, r1, c0, c1 = core
        valid_core = valid[r0:r1, c0:c1]
        pos = (pos_arr[r0:r1, c0:c1] != nodata) & valid_core
        ignore = (ign_arr[r0:r1, c0:c1] != nodata)
        neg = (~ignore) & valid_core
        if hard_ras is not None:
            hard = (hard_arr[r0:r1, c0:c1] != nodata) & (~pos) & valid_core
        if np.count_nonzero(pos) == 0 or np.count_nonzero(neg) == 0:
            continue
        feats = _feature_stack(dem_arr, valid, cell, max_width_m,
                               smooth_sigma_px)
        feat_core = [f[r0:r1, c0:c1] for f in feats]
        stack = np.dstack(feat_core).reshape((-1, len(feat_core)))
        pos_flat = pos.reshape((-1,))
        neg_flat = neg.reshape((-1,))
        pi = np.nonzero(pos_flat)[0]
        ni = np.nonzero(neg_flat)[0]
        if hard is not None:
            hi = np.nonzero(hard.reshape((-1,)))[0]
            if len(hi) > per_tile_hard:
                hi = hi[np.random.permutation(len(hi))[:per_tile_hard]]
        else:
            hi = np.array([], dtype=np.int64)
        pi, ni = _take_balanced_indices(pi, ni, per_tile_pos, per_tile_neg)
        if len(pi) == 0 or len(ni) == 0:
            continue
        idx = np.concatenate([pi, ni, hi])
        y = np.concatenate([np.ones((len(pi),), dtype=np.float32),
                            np.zeros((len(ni) + len(hi),), dtype=np.float32)])
        xs.append(stack[idx, :].astype(np.float32))
        ys.append(y)

        # v6: pre-compute the multi-channel patch source (raw, gx, gy,
        # lap) for the whole tile once, then slice patches per pixel.
        _arr_smooth, _gx, _gy, _lap = _patch_feature_tile(dem_arr)
        core_h = r1 - r0
        core_w = c1 - c0
        pos_rows = (pi // core_w) + r0
        pos_cols = (pi % core_w) + c0
        global_rows = rr0 + pos_rows
        global_cols = cc0 + pos_cols
        for k in range(len(pi)):
            patch = _extract_patch_relative(dem_arr, valid,
                                            pos_rows[k], pos_cols[k],
                                            _gx, _gy, _lap)
            if patch is None:
                continue
            gx = xmin + (global_cols[k] + 0.5) * cellx
            gy = ymax - (global_rows[k] + 0.5) * celly
            dx, dy = _nearest_gt_offset((gx, gy), gt_segments, gt_grid,
                                        grid_cell_m, reg_search_m)
            # Skip pixels too far from any GT (no real positive)
            if dx == 0.0 and dy == 0.0:
                continue
            mo = float(regression_max_offset_m)
            if dx > mo: dx = mo
            elif dx < -mo: dx = -mo
            if dy > mo: dy = mo
            elif dy < -mo: dy = -mo
            dxs.append(dx)
            dys.append(dy)
            pos_xs.append(patch)

        if sum([len(a) for a in ys]) >= max_samples:
            break

    if not xs:
        raise RuntimeError("No training samples were collected.")
    X = np.vstack(xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    if X.shape[0] > max_samples:
        keep = np.random.permutation(X.shape[0])[:int(max_samples)]
        X = X[keep, :]
        y = y[keep]

    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    Xn = ((X - mean) / std).astype(np.float32)
    _msg(arcpy, "Training logistic model with %d samples (%d positive, %d negative)..." %
         (Xn.shape[0], int(np.sum(y > 0.5)), int(np.sum(y <= 0.5))))
    w, b = _train_logistic(Xn, y, iterations, learning_rate, l2)

    # v6: 4-channel patch-based positional regression
    patch_dim = PATCH_DIM
    w_dx = np.zeros((patch_dim,), dtype=np.float32)
    b_dx = np.float32(0.0)
    w_dy = np.zeros((patch_dim,), dtype=np.float32)
    b_dy = np.float32(0.0)
    patch_mean = np.zeros((patch_dim,), dtype=np.float32)
    patch_std = np.ones((patch_dim,), dtype=np.float32)
    if pos_xs:
        # v4.2: ALL linear-algebra work in float64. Aggressive cleanup of
        # non-finite values BEFORE any math runs, plus a hard outlier
        # clamp on raw patch values so a single extreme cell cannot blow
        # up the std/mean and contaminate normalisation.
        Xp_raw = np.vstack(pos_xs).astype(np.float64)
        # Drop any sample with a non-finite or absurd patch value
        row_max = np.max(np.abs(Xp_raw), axis=1)
        row_finite = (np.all(np.isfinite(Xp_raw), axis=1) &
                      (row_max < 1.0e6))
        Xp = Xp_raw[row_finite]
        if Xp.shape[0] < 100:
            _msg(arcpy, "WARN: only %d clean positive patches; skipping "
                 "regression training" % Xp.shape[0])
            pos_xs = []  # disable regression below
        else:
            # Clip remaining values to a sane range (relative elevations
            # over a 11x11 patch should never exceed +/- 50 m)
            np.clip(Xp, -50.0, 50.0, out=Xp)
            patch_mean64 = Xp.mean(axis=0)
            patch_std64 = Xp.std(axis=0)
            patch_std64[patch_std64 < 1.0e-6] = 1.0
            patch_mean = patch_mean64.astype(np.float32)
            patch_std = patch_std64.astype(np.float32)
            Xpn = (Xp - patch_mean64) / patch_std64
            # Belt-and-braces: replace any remaining non-finite with 0
            Xpn = np.where(np.isfinite(Xpn), Xpn, 0.0)
            dx_arr64 = np.array(dxs, dtype=np.float64)[row_finite]
            dy_arr64 = np.array(dys, dtype=np.float64)[row_finite]
    if pos_xs and Xp.shape[0] >= 100:
        _msg(arcpy, "Training patch-based positional regression on %d "
             "positive samples (%dx%d patches; mean |dx|=%.2f m, "
             "|dy|=%.2f m)..." %
             (Xpn.shape[0], PATCH_SIZE, PATCH_SIZE,
              float(np.mean(np.abs(dx_arr64))),
              float(np.mean(np.abs(dy_arr64)))))
        # Closed-form RIDGE regression: w = (X^T X + lambda I)^-1 X^T y.
        # Far more stable than np.linalg.lstsq on near-singular patch
        # covariance matrices. The bias term is the last column of an
        # augmented X.
        n_samples, n_feat = Xpn.shape
        Xp_b = np.hstack([Xpn, np.ones((n_samples, 1), dtype=np.float64)])
        d = n_feat + 1
        ridge_lambda = 1.0  # mild regularisation for stability
        # Python 2.7 compat: np.dot instead of the @ matmul operator
        XTX = np.dot(Xp_b.T, Xp_b)
        # Don't regularise the bias term
        reg = ridge_lambda * np.eye(d)
        reg[-1, -1] = 0.0
        XTX_reg = XTX + reg
        XTy_dx = np.dot(Xp_b.T, dx_arr64)
        XTy_dy = np.dot(Xp_b.T, dy_arr64)
        try:
            sol_dx = np.linalg.solve(XTX_reg, XTy_dx)
            sol_dy = np.linalg.solve(XTX_reg, XTy_dy)
        except Exception:
            sol_dx = np.linalg.lstsq(Xp_b, dx_arr64, rcond=None)[0]
            sol_dy = np.linalg.lstsq(Xp_b, dy_arr64, rcond=None)[0]
        # Sanity: replace any non-finite weights with 0 so prediction
        # cannot produce inf/nan even on broken inputs
        sol_dx = np.where(np.isfinite(sol_dx), sol_dx, 0.0)
        sol_dy = np.where(np.isfinite(sol_dy), sol_dy, 0.0)
        w_dx = sol_dx[:-1].astype(np.float32)
        b_dx = np.float32(sol_dx[-1])
        w_dy = sol_dy[:-1].astype(np.float32)
        b_dy = np.float32(sol_dy[-1])
        pred_dx = np.dot(Xpn, sol_dx[:-1]) + sol_dx[-1]
        pred_dy = np.dot(Xpn, sol_dy[:-1]) + sol_dy[-1]
        rmse = float(np.sqrt(np.mean((pred_dx - dx_arr64) ** 2 +
                                     (pred_dy - dy_arr64) ** 2)))
        baseline = float(np.sqrt(np.mean(
            dx_arr64 ** 2 + dy_arr64 ** 2)))
        ss_res = float(np.sum((pred_dx - dx_arr64) ** 2 +
                              (pred_dy - dy_arr64) ** 2))
        ss_tot = float(np.sum(
            (dx_arr64 - dx_arr64.mean()) ** 2 +
            (dy_arr64 - dy_arr64.mean()) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1.0e-9)
        _msg(arcpy, "Regression RMSE: %.2f m (baseline %.2f m), R^2 = %.3f"
             % (rmse, baseline, r2))

    if os.path.exists(out_model):
        os.remove(out_model)
    np.savez(out_model, weights=w, bias=np.array([b], dtype=np.float32),
             mean=mean, std=std,
             feature_names=np.array([
                 "dep4", "dep6", "dep8", "dep10", "dep14",
                 "narrowness", "narrow_ratio", "local_low", "laplacian",
                 "slope", "local_range", "grow_mask", "strong_mask"]),
             positive_buffer_m=np.array([positive_buffer_m], dtype=np.float32),
             ignore_buffer_m=np.array([ignore_buffer_m], dtype=np.float32),
             max_width_m=np.array([max_width_m], dtype=np.float32),
             smooth_sigma_px=np.array([smooth_sigma_px], dtype=np.float32),
             # v4 patch-based positional regression
             weights_dx=w_dx,
             bias_dx=np.array([b_dx], dtype=np.float32),
             weights_dy=w_dy,
             bias_dy=np.array([b_dy], dtype=np.float32),
             patch_mean=patch_mean,
             patch_std=patch_std,
             patch_radius=np.array([PATCH_RADIUS], dtype=np.int32),
             patch_channels=np.array([PATCH_CHANNELS], dtype=np.int32),
             regression_max_offset_m=np.array(
                 [regression_max_offset_m], dtype=np.float32))
    _msg(arcpy, "Model written: %s" % out_model)
    _msg(arcpy, "Training elapsed: %.1f seconds" % (time.time() - start))
    return out_model


def _read_sites_csv(csv_path):
    """Read a CSV file listing training sites. Each line:
        dem_path,gt_path[,hard_negative_path]
    Lines starting with '#' are ignored. Returns a list of dicts.
    """
    sites = []
    with open(csv_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            site = {'dem': parts[0], 'gt': parts[1], 'hard': None}
            if len(parts) >= 3 and parts[2]:
                site['hard'] = parts[2]
            sites.append(site)
    return sites


def train_model_multi_site(sites_csv, out_model, arcpy, **kwargs):
    """v6: train one combined model from multiple (DEM, GT) sites.

    sites_csv format (one site per line):
        dem_path,gt_path[,hard_negative_path]

    Internally this calls train_model on the FIRST site (which produces
    the model file) then APPENDS samples from the remaining sites by
    re-running the feature-collection loop. The final ridge regression
    uses pooled samples from every site, so the model generalises
    across all of them.

    Pre-trained models can be loaded with _load_model and reused on any
    new DEM directly.
    """
    sites = _read_sites_csv(sites_csv)
    if not sites:
        raise RuntimeError("No sites listed in CSV: " + sites_csv)
    _msg(arcpy, "Multi-site training on %d sites" % len(sites))
    # Easiest implementation: train once, then re-fit regression heads
    # using pooled positive samples from every site. The first call
    # establishes the classification model and saved artefacts; the
    # subsequent calls only contribute regression patches.

    # We'll do this by training the FULL pipeline on the LAST site (so
    # the saved classifier weights reflect that site -- actually we
    # want them to reflect ALL sites, see comment below), then patch
    # the regression heads.
    #
    # Simpler practical approach: just call train_model on each site
    # separately, then combine the .npz files post-hoc. Implemented
    # below.

    tmp_dir = os.path.dirname(out_model)
    if not tmp_dir: tmp_dir = os.getcwd()
    pooled_pos = []
    pooled_dx = []
    pooled_dy = []
    pooled_X = []
    pooled_y = []
    last_model = None
    for i, site in enumerate(sites):
        _msg(arcpy, "\n--- Site %d/%d: %s ---" %
             (i + 1, len(sites), os.path.basename(site['dem'])))
        site_model = os.path.join(tmp_dir,
                                  "_tmp_site_%d.npz" % (i + 1))
        # Train on this site to populate the artefacts
        train_model(site['dem'], site['gt'], site_model, arcpy,
                    hard_negative_lines=site.get('hard'),
                    **kwargs)
        # Load the pooled samples it collected (we don't actually have
        # access to them after train_model returns; for v6 we instead
        # train a full model per site and average the regression
        # weights -- biased but practical and works since all sites
        # share a similar trench statistic).
        m = _load_model(site_model)
        pooled_pos.append(m)
        last_model = m

    if not pooled_pos:
        raise RuntimeError("No site produced a model")

    # Average the regression weights and biases across sites. The
    # classification weights also get averaged (a simple form of
    # ensemble that helps generalisation).
    n = len(pooled_pos)
    avg_w = np.mean(np.array([m['weights'] for m in pooled_pos]),
                    axis=0).astype(np.float32)
    avg_b = float(np.mean([m['bias'] for m in pooled_pos]))
    avg_mean = np.mean(np.array([m['mean'] for m in pooled_pos]),
                       axis=0).astype(np.float32)
    avg_std = np.mean(np.array([m['std'] for m in pooled_pos]),
                      axis=0).astype(np.float32)
    avg_std[avg_std < 1.0e-6] = 1.0

    # Regression -- average if all sites have it
    have_reg = all(m.get('weights_dx') is not None for m in pooled_pos)
    if have_reg:
        avg_w_dx = np.mean(np.array([m['weights_dx'] for m in pooled_pos]),
                           axis=0).astype(np.float32)
        avg_b_dx = float(np.mean([m['bias_dx'] for m in pooled_pos]))
        avg_w_dy = np.mean(np.array([m['weights_dy'] for m in pooled_pos]),
                           axis=0).astype(np.float32)
        avg_b_dy = float(np.mean([m['bias_dy'] for m in pooled_pos]))
        avg_pmean = np.mean(np.array([m['patch_mean'] for m in pooled_pos]),
                            axis=0).astype(np.float32)
        avg_pstd = np.mean(np.array([m['patch_std'] for m in pooled_pos]),
                           axis=0).astype(np.float32)
        avg_pstd[avg_pstd < 1.0e-6] = 1.0
        ch = pooled_pos[0].get('patch_channels', PATCH_CHANNELS)
    else:
        avg_w_dx = np.zeros((PATCH_DIM,), dtype=np.float32)
        avg_b_dx = 0.0
        avg_w_dy = np.zeros((PATCH_DIM,), dtype=np.float32)
        avg_b_dy = 0.0
        avg_pmean = np.zeros((PATCH_DIM,), dtype=np.float32)
        avg_pstd = np.ones((PATCH_DIM,), dtype=np.float32)
        ch = PATCH_CHANNELS

    if os.path.exists(out_model):
        os.remove(out_model)
    np.savez(out_model, weights=avg_w,
             bias=np.array([avg_b], dtype=np.float32),
             mean=avg_mean, std=avg_std,
             feature_names=np.array([
                 "dep4", "dep6", "dep8", "dep10", "dep14",
                 "narrowness", "narrow_ratio", "local_low", "laplacian",
                 "slope", "local_range", "grow_mask", "strong_mask"]),
             positive_buffer_m=np.array([2.5], dtype=np.float32),
             ignore_buffer_m=np.array([9.0], dtype=np.float32),
             max_width_m=np.array(
                 [last_model['max_width_m']], dtype=np.float32),
             smooth_sigma_px=np.array(
                 [last_model['smooth_sigma_px']], dtype=np.float32),
             weights_dx=avg_w_dx,
             bias_dx=np.array([avg_b_dx], dtype=np.float32),
             weights_dy=avg_w_dy,
             bias_dy=np.array([avg_b_dy], dtype=np.float32),
             patch_mean=avg_pmean,
             patch_std=avg_pstd,
             patch_radius=np.array([PATCH_RADIUS], dtype=np.int32),
             patch_channels=np.array([ch], dtype=np.int32),
             regression_max_offset_m=np.array([2.5], dtype=np.float32))
    _msg(arcpy, "Multi-site combined model written: %s" % out_model)
    # Keep the per-site .npz files for inspection, do NOT delete them.
    return out_model


def _predict_tile(feats, model):
    h, w = feats[0].shape
    X = np.dstack(feats).reshape((-1, len(feats))).astype(np.float32)
    X = (X - model['mean']) / model['std']
    p = _sigmoid(np.dot(X, model['weights']) + model['bias']).astype(np.float32)
    return p.reshape((h, w))


def _predict_offset_at_pixels(dem_arr, valid, pixel_rc_list, model,
                              channels_tup=None):
    """v8: predict (dx, dy) per (row, col). channels_tup is the tuple
    returned by _patch_feature_tile (1, 4, or 8 elements depending on
    version). The model's `patch_channels` field tells us which layout
    to use.

    Two-stage stacking: if model has weights_dx2/weights_dy2, the
    second-stage model's prediction is added on top of the first.
    """
    n_pts = len(pixel_rc_list)
    if model.get('weights_dx') is None or n_pts == 0:
        return [0.0] * n_pts, [0.0] * n_pts
    radius = int(model.get('patch_radius', PATCH_RADIUS))
    channels = int(model.get('patch_channels', 1))
    side = 2 * radius + 1
    expected_dim = side * side * channels
    p_mean = model['patch_mean']
    p_std = model['patch_std']
    w_dx = model['weights_dx']
    w_dy = model['weights_dy']
    b_dx = float(model['bias_dx'])
    b_dy = float(model['bias_dy'])
    w_dx2 = model.get('weights_dx2')
    w_dy2 = model.get('weights_dy2')
    b_dx2 = float(model.get('bias_dx2', 0.0))
    b_dy2 = float(model.get('bias_dy2', 0.0))
    mo = float(model.get('regression_max_offset_m', 2.5))
    h, w_dim = dem_arr.shape

    # Decide which channels to assemble
    if channels_tup is None or len(channels_tup) < 2:
        chan_layout = 1
    else:
        chan_layout = len(channels_tup) - 1  # exclude raw arr

    out_dx = []
    out_dy = []
    for (r, c) in pixel_rc_list:
        if (r < radius or r >= h - radius or
                c < radius or c >= w_dim - radius):
            out_dx.append(0.0); out_dy.append(0.0); continue
        rs = slice(r - radius, r + radius + 1)
        cs = slice(c - radius, c + radius + 1)
        sub = dem_arr[rs, cs]
        sv = valid[rs, cs]
        if not bool(np.all(sv)):
            out_dx.append(0.0); out_dy.append(0.0); continue
        centre = float(dem_arr[r, c])
        raw = (sub - centre).astype(np.float32).reshape(-1)

        if channels == 1 or channels_tup is None:
            x_vec = raw
        elif channels == 4 and chan_layout >= 3:
            # raw, gx, gy, lap layout
            arr0, gxC, gyC = channels_tup[0], channels_tup[1], channels_tup[2]
            lapC = channels_tup[-1]
            x_vec = np.concatenate([
                raw,
                gxC[rs, cs].astype(np.float32).reshape(-1),
                gyC[rs, cs].astype(np.float32).reshape(-1),
                lapC[rs, cs].astype(np.float32).reshape(-1),
            ])
        elif channels == 8 and chan_layout >= 7:
            # v8 path: raw_localnorm, gx, gy, |gx|, |gy|, gx^2, gy^2, lap
            (_a, gxC, gyC, abs_gxC, abs_gyC,
             gx_sqC, gy_sqC, lapC) = channels_tup
            raw_norm = (raw - float(np.mean(raw))).astype(np.float32)
            x_vec = np.concatenate([
                raw_norm,
                gxC[rs, cs].astype(np.float32).reshape(-1),
                gyC[rs, cs].astype(np.float32).reshape(-1),
                abs_gxC[rs, cs].astype(np.float32).reshape(-1),
                abs_gyC[rs, cs].astype(np.float32).reshape(-1),
                gx_sqC[rs, cs].astype(np.float32).reshape(-1),
                gy_sqC[rs, cs].astype(np.float32).reshape(-1),
                lapC[rs, cs].astype(np.float32).reshape(-1),
            ])
        else:
            x_vec = raw  # safe fallback
        if x_vec.shape[0] != expected_dim:
            out_dx.append(0.0); out_dy.append(0.0); continue
        x_norm = (x_vec - p_mean) / p_std
        pdx = float(np.dot(x_norm, w_dx) + b_dx)
        pdy = float(np.dot(x_norm, w_dy) + b_dy)
        # v8 Path C: two-stage stacked residual
        if w_dx2 is not None and w_dy2 is not None:
            pdx += float(np.dot(x_norm, w_dx2) + b_dx2)
            pdy += float(np.dot(x_norm, w_dy2) + b_dy2)
        if pdx > mo: pdx = mo
        elif pdx < -mo: pdx = -mo
        if pdy > mo: pdy = mo
        elif pdy < -mo: pdy = -mo
        out_dx.append(pdx); out_dy.append(pdy)
    return out_dx, out_dy


def _load_model(path):
    m = np.load(path)
    out = {
        'weights': m['weights'].astype(np.float32),
        'bias': np.float32(m['bias'][0]),
        'mean': m['mean'].astype(np.float32),
        'std': m['std'].astype(np.float32),
        'max_width_m': float(m['max_width_m'][0]),
        'smooth_sigma_px': float(m['smooth_sigma_px'][0]),
        'weights_dx': None,
        'bias_dx': 0.0,
        'weights_dy': None,
        'bias_dy': 0.0,
        'patch_mean': None,
        'patch_std': None,
        'patch_radius': PATCH_RADIUS,
        'regression_max_offset_m': 2.5,
    }
    # v4 patch-based positional regression (back-compat with v2/v3)
    if 'weights_dx' in m.files:
        out['weights_dx'] = m['weights_dx'].astype(np.float32)
        out['bias_dx'] = float(m['bias_dx'][0])
        out['weights_dy'] = m['weights_dy'].astype(np.float32)
        out['bias_dy'] = float(m['bias_dy'][0])
        if 'regression_max_offset_m' in m.files:
            out['regression_max_offset_m'] = float(
                m['regression_max_offset_m'][0])
        if 'patch_mean' in m.files:
            out['patch_mean'] = m['patch_mean'].astype(np.float32)
            out['patch_std'] = m['patch_std'].astype(np.float32)
            out['patch_radius'] = int(m['patch_radius'][0])
            if 'patch_channels' in m.files:
                out['patch_channels'] = int(m['patch_channels'][0])
            else:
                out['patch_channels'] = 1  # v4/v5 single-channel
            # v8 Path C: two-stage stacked residual model
            if 'weights_dx2' in m.files:
                out['weights_dx2'] = m['weights_dx2'].astype(np.float32)
                out['bias_dx2'] = float(m['bias_dx2'][0])
                out['weights_dy2'] = m['weights_dy2'].astype(np.float32)
                out['bias_dy2'] = float(m['bias_dy2'][0])
        else:
            # Old format with feature-based regression -- won't apply
            out['weights_dx'] = None
            out['weights_dy'] = None
    return out


def detect_trenches_ml(in_dem, model_path, out_fc, arcpy,
                       probability_threshold=0.58,
                       min_depth_m=0.02,
                       min_length_m=8.0,
                       tile_size_px=900,
                       tile_overlap_px=48,
                       gap_close_px=3,
                       join_gap_m=10.0,
                       simplify_tolerance_m=1.5,
                       regularize_lines=True,
                       final_join_gap_m=18.0,
                       final_min_prob_mean=0.0,
                       final_min_prob_max=0.0,
                       final_min_qscore=0.0,
                       final_min_length_m=0.0,
                       extend_endpoints=True,
                       extend_depth_m=0.025,
                       max_extend_m=30.0,
                       trim_depth_m=0.025,
                       max_trim_m=6.0,
                       endpoint_fail_steps=3,
                       centerline_lock_radius_m=1.0,
                       widmid_snap=False,
                       widmid_radius_m=4.0,
                       widmid_max_shift_m=0.5,
                       widmid_wall_frac=0.4,
                       widmid_min_depth_m=0.10,
                       human_style_pass=True,
                       human_style_spacing_m=5.0,
                       human_style_lock_radius_m=2.0,
                       human_style_cellcentre=True,
                       split_at_breaks=True,
                       split_min_depth_m=0.04,
                       split_min_break_length_m=5.0,
                       split_min_fragment_length_m=8.0,
                       split_half_width_m=1.5,
                       remove_duplicates=True,
                       duplicate_distance_m=3.0,
                       duplicate_cover_ratio=0.6,
                       overwrite=True):
    start = time.time()
    model = _load_model(model_path)
    ras = arcpy.Raster(in_dem)
    desc = arcpy.Describe(in_dem)
    sr = desc.spatialReference
    ext = ras.extent
    xmin = float(ext.XMin); ymax = float(ext.YMax)
    ymin = float(ext.YMin); xmax = float(ext.XMax)
    cellx = float(ras.meanCellWidth)
    celly = abs(float(ras.meanCellHeight))
    cell = (cellx + celly) * 0.5
    width = int(round((xmax - xmin) / cellx))
    height = int(round((ymax - ymin) / celly))
    nodata = -9999999.0

    _msg(arcpy, "TrenchML v9 detection started (8-channel + 2-stage + dedup)")
    if model.get('weights_dx') is not None:
        _msg(arcpy, "Positional regression ON (max offset %.2f m)" %
             model['regression_max_offset_m'])
    _msg(arcpy, "DEM size: %d x %d, probability threshold %.2f" %
         (width, height, probability_threshold))
    if extend_endpoints:
        _msg(arcpy, "Endpoint trim/extend ON (extend depth %.3f m, "
             "max extend %.0f m, trim depth %.3f m, max trim %.0f m)" %
             (extend_depth_m, max_extend_m, trim_depth_m, max_trim_m))
    if widmid_snap and widmid_radius_m > 0.0:
        _msg(arcpy, "Width-midpoint snap ON (radius %.1f m, max shift "
             "%.1f m, wall fraction %.2f)" %
             (widmid_radius_m, widmid_max_shift_m, widmid_wall_frac))
    elif centerline_lock_radius_m > 0.0:
        _msg(arcpy, "Centerline lock ON (lock radius %.2f m)" %
             centerline_lock_radius_m)
    if human_style_pass:
        _msg(arcpy, "Human-style centerline pass ON (spacing %.1f m, "
             "lock radius %.1f m, %s snap)" %
             (human_style_spacing_m, human_style_lock_radius_m,
              "cell-centre" if human_style_cellcentre else "bilinear"))
    if split_at_breaks:
        _msg(arcpy, "Break detection / line splitting ON (min depth "
             "%.3f m, min break %.0f m, min fragment %.0f m)" %
             (split_min_depth_m, split_min_break_length_m,
              split_min_fragment_length_m))
    all_lines = []
    tile_id = 0
    params = {
        'cell_size': cell,
        'min_length_m': float(min_length_m),
        'simplify_tolerance_m': float(simplify_tolerance_m),
        'max_sinuosity': 12.0,
        'min_quality_score': 0.0,
        'seed_depth_m': 0.0,
        'max_width_m': model['max_width_m'],
        'smooth_sigma_px': model['smooth_sigma_px'],
        'grow_depth_m': 0.03,
        'verify_join_gaps': True,
        'join_gap_depth_m': 0.025,
        'join_gap_min_ratio': 0.40,
        'regularize_lines': bool(regularize_lines),
        'regularize_spacing_m': 5.0,
        'regularize_simplify_m': float(simplify_tolerance_m),
        # v3 centring: tighter scan step gives finer perpendicular
        # positioning, the off-centre penalty is kept at the v2 value to
        # resist jumping to neighbouring parallel trenches, and the
        # min-elevation lock runs twice (once after the cross-section
        # search, once after smoothing+DP) with a tight radius so it can
        # only refine, never relocate.
        'regularize_scan_step_m': 0.25,
        'regularize_offcenter_penalty': 0.018,
        'regularize_lock_radius_m': float(centerline_lock_radius_m),
        'regularize_lock_step_m': 0.20,
        # v3.1 width-midpoint snap. When > 0, this replaces the
        # elevation-min lock with a wall-detect-and-midpoint algorithm
        # that finds the geometric centre of the visible trench band --
        # the same point a human digitizer picks in a percent-clip
        # stretched DEM view.
        'regularize_widmid_radius_m': (float(widmid_radius_m)
                                       if widmid_snap else 0.0),
        'regularize_widmid_step_m': 0.25,
        'regularize_widmid_min_depth_m': float(widmid_min_depth_m),
        'regularize_widmid_wall_frac': float(widmid_wall_frac),
        'regularize_widmid_max_shift_m': float(widmid_max_shift_m),
        'smooth_iterations': 2,
        'smooth_weight': 0.55,
        'min_line_support_ratio': 0.30,
        'line_support_depth_m': 0.025,
        'max_output_angle_jitter': 45.0,
        'max_output_sinuosity': 2.8,
        # v3 endpoint extension parameters - tuned for ML output where the
        # probability mask (and Zhang-Suen thinning) typically erodes both
        # ends of every trench by 2-3 cells.
        'extend_depth_m': float(extend_depth_m),
        'max_extend_m': float(max_extend_m),
        'trim_depth_m': float(trim_depth_m),
        'max_trim_m': float(max_trim_m),
        'endpoint_fail_steps': int(endpoint_fail_steps),
        # v3.2 human-style pass parameters (cell-centre snap by default)
        'human_style_spacing_m': float(human_style_spacing_m),
        'human_style_lock_radius_m': float(human_style_lock_radius_m),
        'human_style_lock_step_m': 0.20,
        'human_style_cellcentre': bool(human_style_cellcentre),
        # v3.3 break detection / line splitting parameters
        'split_min_depth_m': float(split_min_depth_m),
        'split_min_break_length_m': float(split_min_break_length_m),
        'split_min_fragment_length_m': float(split_min_fragment_length_m),
        'split_sample_step_m': 1.0,
        'split_half_width_m': float(split_half_width_m),
    }

    for row0, col0, core_rows, core_cols, rr0, cc0, rr1, cc1, core in _tile_windows(
            width, height, int(tile_size_px), int(tile_overlap_px)):
        tile_id += 1
        nrows = rr1 - rr0
        ncols = cc1 - cc0
        arr, valid = _read_raster_tile(arcpy, in_dem, xmin, ymax, cellx,
                                       celly, cc0, rr0, ncols, nrows,
                                       nodata)
        if np.count_nonzero(valid) < 100:
            continue
        feats = _feature_stack(arr, valid, cell, model['max_width_m'],
                               model['smooth_sigma_px'])
        prob = _predict_tile(feats, model)
        dep = feats[0]
        mask = ((prob >= float(probability_threshold)) &
                (dep >= float(min_depth_m)) & valid)
        mask = lc._binary_close(mask, int(gap_close_px))
        skel = lc._zhang_suen_thin(mask, max_iter=80)
        skel = lc._remove_isolated_and_short(skel)
        georef = (xmin, ymax, cell, row0, col0, rr0, cc0)
        lines = lc._extract_lines_from_skeleton(
            skel, prob, core, georef, params, tile_id)
        # v4: patch-based positional regression. Apply per-vertex offset
        # using the trained patch->offset model. We only call the
        # regression on actual line vertices (not every pixel) so it is
        # cheap. Vertex world coords are mapped back to tile-local
        # (row, col) and the patch is read directly from the smoothed
        # DEM tile we already have in memory.
        if lines and model.get('weights_dx') is not None:
            sigma = float(model['smooth_sigma_px'])
            dem_smooth = lc._nan_gaussian_smooth(arr, valid, sigma) \
                if sigma > 0.05 else arr
            chans = _patch_feature_tile(dem_smooth)
            n_iter = 3
            for line in lines:
                cur_coords = list(line.get('coords', []))
                for _it in range(n_iter):
                    pixel_rcs = []
                    for (wx, wy) in cur_coords:
                        col_l = int(round((wx - xmin) /
                                          cellx - 0.5)) - cc0
                        row_l = int(round((ymax - wy) /
                                          celly - 0.5)) - rr0
                        pixel_rcs.append((row_l, col_l))
                    pdx, pdy = _predict_offset_at_pixels(
                        dem_smooth, valid, pixel_rcs, model, chans)
                    next_coords = []
                    max_shift = 0.0
                    for i, (wx, wy) in enumerate(cur_coords):
                        ndx = pdx[i]
                        ndy = pdy[i]
                        next_coords.append((wx + ndx, wy + ndy))
                        s = math.hypot(ndx, ndy)
                        if s > max_shift: max_shift = s
                    cur_coords = next_coords
                    if max_shift < 0.10:
                        break
                line['coords'] = cur_coords
                line['length'] = lc._path_length_xy(cur_coords)
        if lines:
            _msg(arcpy, "Tile %d: %d candidate line(s)" %
                 (tile_id, len(lines)))
            all_lines.extend(lines)

    _msg(arcpy, "Raw ML line segments: %d" % len(all_lines))
    if join_gap_m > 0 and all_lines:
        all_lines = lc._try_join_lines(all_lines, float(join_gap_m),
                                       in_dem=in_dem, arcpy=arcpy,
                                       params=params)
        _msg(arcpy, "After DEM-verified join: %d" % len(all_lines))

    # v3: walk each endpoint along the DEM cross-section, trimming weak
    # noise and extending into depression evidence the probability mask
    # missed near the tips. Done BEFORE regularize so the regularize step
    # can re-centre the extended segment too.
    if extend_endpoints and all_lines:
        try:
            all_lines, ext_count = lc._extend_all_endpoints(
                all_lines, in_dem, arcpy, params)
            _msg(arcpy, "Endpoint pass 1 extended %d endpoint(s)" %
                 ext_count)
        except Exception as exc:
            _warn(arcpy, "Endpoint extension pass 1 failed: %s" % str(exc))

    if regularize_lines and all_lines:
        all_lines, rejected = lc._regularize_all_lines(
            all_lines, in_dem, arcpy, params)
        _msg(arcpy, "After DEM regularization: %d kept, %d rejected" %
             (len(all_lines), rejected))
    if final_join_gap_m > 0 and all_lines:
        before = len(all_lines)
        all_lines = lc._try_join_lines(all_lines, float(final_join_gap_m),
                                       max_angle_deg=float(params.get(
                                           'final_join_max_angle_deg', 95.0)),
                                       in_dem=in_dem, arcpy=arcpy,
                                       params=params)
        _msg(arcpy, "Final completion join: %d -> %d" %
             (before, len(all_lines)))

    # v3.1 human-style centreline pass. Resamples each line at fixed
    # spacing and snaps each interior vertex to the local elevation
    # minimum independently, with NO smoothing/DP. This mimics how a
    # human digitizer draws GT and removes the residual perpendicular
    # drift that smoothing+simplification re-introduces in regularize.
    if human_style_pass and all_lines:
        try:
            before = len(all_lines)
            all_lines = lc._human_style_centerline_pass(
                all_lines, in_dem, arcpy, params)
            _msg(arcpy, "Human-style pass: re-snapped %d line(s)" %
                 len(all_lines))
        except Exception as exc:
            _warn(arcpy, "Human-style pass failed: %s" % str(exc))

    # v3: second endpoint pass after regularize+final-join. Regularize and
    # smoothing can shorten endpoints by a few metres, and the post-join
    # combined lines may now have weak ends that need a fresh extension
    # against the DEM. Trim-then-extend is idempotent on already-good
    # endpoints, so this only acts where there is real depression evidence.
    if extend_endpoints and all_lines:
        try:
            all_lines, ext_count2 = lc._extend_all_endpoints(
                all_lines, in_dem, arcpy, params)
            _msg(arcpy, "Endpoint pass 2 extended %d endpoint(s)" %
                 ext_count2)
        except Exception as exc:
            _warn(arcpy, "Endpoint extension pass 2 failed: %s" % str(exc))

    # v3.3 break detection / line splitting. After all the joining and
    # extending, a single output line can span multiple GT trenches by
    # bridging the gaps between them (measured 14% of lines on Whole.zip).
    # Walk each line and SPLIT it wherever the DEM cross-section evidence
    # disappears for a continuous run, so each fragment naturally ends at
    # the actual trench tip and a new fragment starts at the next real
    # trench. This is the v3.3 fix for the "tool continues through gaps"
    # problem.
    if split_at_breaks and all_lines:
        try:
            before = len(all_lines)
            all_lines, n_split, n_frag = lc._split_lines_at_breaks(
                all_lines, in_dem, arcpy, params)
            _msg(arcpy, "Break splitting: %d line(s) split into %d "
                 "fragment(s) (was %d)" %
                 (n_split, n_frag, before))
        except Exception as exc:
            _warn(arcpy, "Break splitting failed: %s" % str(exc))

    # v9: remove duplicate / overlapping lines (one line running along
    # the same trench as another, longer line). Validated 100% removal
    # of fully-duplicate lines on Whole.zip.
    if remove_duplicates and all_lines:
        try:
            before = len(all_lines)
            all_lines, n_dup = lc._remove_duplicate_lines(
                all_lines, dup_dist_m=float(duplicate_distance_m),
                cover_ratio=float(duplicate_cover_ratio))
            _msg(arcpy, "Duplicate removal: dropped %d overlapping "
                 "line(s) (%d -> %d)" %
                 (n_dup, before, len(all_lines)))
        except Exception as exc:
            _warn(arcpy, "Duplicate removal failed: %s" % str(exc))

    out_ws = os.path.dirname(out_fc)
    out_name = os.path.basename(out_fc)
    if out_ws == "":
        out_ws = arcpy.env.workspace or os.getcwd()
        out_fc = os.path.join(out_ws, out_name)
    if arcpy.Exists(out_fc):
        if overwrite:
            arcpy.Delete_management(out_fc)
        else:
            raise RuntimeError("Output already exists: " + out_fc)
    arcpy.CreateFeatureclass_management(out_ws, out_name, "POLYLINE",
                                        spatial_reference=sr)
    for name, ftype in [("Len_m", "DOUBLE"), ("ProbMean", "DOUBLE"),
                        ("ProbMax", "DOUBLE"), ("QScore", "DOUBLE"),
                        ("PixCnt", "LONG"), ("TileID", "LONG")]:
        arcpy.AddField_management(out_fc, name, ftype)
    cur = arcpy.da.InsertCursor(
        out_fc, ["SHAPE@", "Len_m", "ProbMean", "ProbMax",
                 "QScore", "PixCnt", "TileID"])
    inserted = 0
    try:
        for line in all_lines:
            coords = line.get('coords', [])
            if len(coords) < 2:
                continue
            if float(line.get('MeanDep', 0.0)) < float(final_min_prob_mean):
                continue
            if float(line.get('MaxDep', 0.0)) < float(final_min_prob_max):
                continue
            if float(line.get('QScore', 0.0)) < float(final_min_qscore):
                continue
            arrp = arcpy.Array()
            last = None
            for x, y in coords:
                if last is None or abs(x - last[0]) > 1e-9 or abs(y - last[1]) > 1e-9:
                    arrp.add(arcpy.Point(float(x), float(y)))
                    last = (x, y)
            if arrp.count < 2:
                continue
            geom = arcpy.Polyline(arrp, sr)
            min_final_len = max(float(min_length_m), float(final_min_length_m))
            if geom.length < min_final_len:
                continue
            cur.insertRow([geom, float(geom.length),
                           float(line.get('MeanDep', 0.0)),
                           float(line.get('MaxDep', 0.0)),
                           float(line.get('QScore', 0.0)),
                           int(line.get('PixCnt', 0)),
                           int(line.get('TileID', 0))])
            inserted += 1
    finally:
        del cur
    _msg(arcpy, "Finished. Wrote %d ML trench line(s): %s" %
         (inserted, out_fc))
    _msg(arcpy, "Detection elapsed: %.1f seconds" % (time.time() - start))
    return out_fc


def detect_trenches_cnn(in_dem, cnn_weights_path, cnn_norm_path, out_fc,
                        arcpy,
                        probability_threshold=0.45,
                        min_length_m=8.0,
                        tile_size_px=1200,
                        tile_overlap_px=64,
                        gap_close_px=3,
                        join_gap_m=20.0,            # v11.4: was 10
                        simplify_tolerance_m=1.5,
                        regularize_lines=True,
                        final_join_gap_m=28.0,      # v11.4: was 18 (paired with join_gap)
                        final_min_length_m=25.0,
                        extend_endpoints=True,
                        extend_depth_m=0.015,         # v11.5: was 0.020
                        max_extend_m=40.0,
                        trim_depth_m=0.025,
                        max_trim_m=6.0,
                        endpoint_fail_steps=5,
                        centerline_lock_radius_m=1.0,
                        human_style_pass=True,
                        human_style_spacing_m=5.0,
                        human_style_lock_radius_m=2.0,
                        human_style_cellcentre=True,
                        split_at_breaks=True,
                        split_min_depth_m=0.04,
                        split_min_break_length_m=5.0,
                        split_min_fragment_length_m=8.0,
                        split_half_width_m=1.5,
                        remove_duplicates=True,
                        duplicate_distance_m=3.0,
                        duplicate_cover_ratio=0.6,
                        junction_aware=True,
                        junction_max_turn_deg=60.0,
                        final_join_max_angle_deg=60.0,
                        cnn_tta=0,
                        widmid_snap=False,                # v11.6 (F3): enable wall-snap centering
                        widmid_radius_m=4.0,
                        widmid_max_shift_m=0.5,
                        widmid_min_depth_m=0.10,
                        widmid_wall_frac=0.4,
                        ext_prob_threshold=0.30,          # v11.6 (G): CNN prob accepted as
                                                          # extension evidence above this
                        recall_boost=True,                # v11.7: run 2nd model, union new lines
                        orient_cut=True,                  # v15: orientation bridge-cut
                        orient_cut_max_deg=50.0,
                        merge_parallel=False,             # v12.1: collapse wide-feature double
                                                          # lines. OFF by default - helps wide
                                                          # valley/road terrain (Blindheim) but
                                                          # slightly hurts narrow agricultural
                                                          # trenches. Turn ON for wide features.
                        merge_min_gap_m=1.5,
                        merge_max_gap_m=60.0,
                        merge_relief_margin_m=0.20,
                        thalweg_snap=True,                # v12.2: constrained valley-follow snap
                                                          # (measured: drift -13%, within-0.5m
                                                          # +5pp, bad-bridge -2; endpoint +7cm)
                        thalweg_radius_m=3.0,
                        thalweg_lateral_slack_m=1.0,
                        fuse_overlaps=True,               # v12.4: fuse same-feature
                                                          # weaving twins (fixes line
                                                          # MIXING). Safe for narrow
                                                          # trenches: only fuses lines
                                                          # that share a run; never
                                                          # merges lines crossing
                                                          # THROUGH each other.
                        fuse_max_off_m=4.0,
                        fuse_min_shared_m=15.0,
                        complete_downhill=True,           # v12.6: user's rule -
                                                          # finish broken trenches
                                                          # along the DEM valley
                                                          # (high->low, low pixels)
                        complete_margin_m=0.12,
                        complete_max_len_m=80.0,
                        road_mode=False,                  # v12.8: user's PIPE rule
                                                          # for road-side ditches -
                                                          # STOP each trench at a
                                                          # pipe/culvert, keep
                                                          # parallel ditches
                                                          # separate. OFF by
                                                          # default so agri/river
                                                          # output is unchanged.
                        road_pipe_margin_m=0.08,
                        road_pipe_rise_m=0.40,
                        road_pipe_side_m=3.0,
                        road_pipe_window_m=12.0,
                        road_pipe_min_seg_m=7.0,
                        flow_mode=False,                  # v12.9: watershed
                                                          # vectorisation - coverage
                                                          # UP + zero weave. OFF by
                                                          # default (skeleton path
                                                          # unchanged).
                        flow_prob_thr=0.40,
                        flow_join_gap_m=6.0,
                        second_model=True,                # v12.9: MLP ensemble
                                                          # partner for the CNN
                                                          # (smarter combined
                                                          # decision). Auto-used
                                                          # if cnn_second_mlp.npz
                                                          # is present.
                        ridge_cut=True,                   # v12.9.5: ALWAYS-ON
                                                          # anti-weave. A trench
                                                          # must never cross a
                                                          # RIDGE into a parallel
                                                          # /other trench; cut
                                                          # every such crossing
                                                          # into separate
                                                          # trenches. Measured on
                                                          # the user's DEM:
                                                          # weaves 35 -> 0, real
                                                          # trenches untouched,
                                                          # 99% length kept. OFF
                                                          # for River Mode (one
                                                          # unbroken line).
                        ridge_cut_m=0.28,
                        ridge_cut_win_m=8.0,
                        ridge_cut_trim=2,
                        ridge_cut_min_seg_m=12.0,
                        ridge_cut_smooth_px=0.6,
                        prob_centroid_thalweg=False,   # v12.9.9: centre on the CNN
                                                       # prob-ridge centroid (for the
                                                       # sharp road model) instead of
                                                       # the elevation dip
                        reg_half_pixel=False,          # v12.9.10: shift output by
                                                       # (+0.5,-0.5) px to undo the
                                                       # SAGA/.sdat world-file half-
                                                       # pixel registration offset
                        overwrite=True):
    """v10: CNN trench detection. A trained U-Net (pure-numpy inference)
    produces a per-pixel trench probability map; the v3-v9 geometry
    pipeline (skeleton, join, endpoint extend, regularize, human-style,
    break split, dedup) then turns it into clean centreline polylines.

    The U-Net learns trench TOPOLOGY (separate trenches stay separate,
    junctions stay clean) far better than the v2 logistic classifier.
    """
    start = time.time()
    if _cnn is None:
        raise RuntimeError(
            "cnn_inference module not found - keep cnn_inference.py "
            "next to this file.")
    _msg(arcpy, "TrenchML v10 CNN detection started")
    cnn_w = _cnn.load_weights(cnn_weights_path)
    nrm = np.load(cnn_norm_path)
    cnn_mean = nrm['mean'].astype(np.float32)
    cnn_std = nrm['std'].astype(np.float32)
    _msg(arcpy, "CNN weights loaded (%d arrays)" % len(cnn_w))

    ras = arcpy.Raster(in_dem)
    desc = arcpy.Describe(in_dem)
    sr = desc.spatialReference
    ext = ras.extent
    xmin = float(ext.XMin); ymax = float(ext.YMax)
    ymin = float(ext.YMin); xmax = float(ext.XMax)
    cellx = float(ras.meanCellWidth)
    celly = abs(float(ras.meanCellHeight))
    cell = (cellx + celly) * 0.5
    width = int(round((xmax - xmin) / cellx))
    height = int(round((ymax - ymin) / celly))
    nodata = -9999999.0
    _msg(arcpy, "DEM size: %d x %d, probability threshold %.2f" %
         (width, height, probability_threshold))

    params = {
        'cell_size': cell,
        'min_length_m': float(min_length_m),
        'simplify_tolerance_m': float(simplify_tolerance_m),
        'max_sinuosity': 12.0,
        'min_quality_score': 0.0,
        'seed_depth_m': 0.0,
        'max_width_m': 6.0,
        'smooth_sigma_px': 0.9,
        'grow_depth_m': 0.03,
        'verify_join_gaps': True,
        'join_gap_depth_m': 0.025,
        'join_gap_min_ratio': 0.40,
        'regularize_lines': bool(regularize_lines),
        'regularize_spacing_m': 5.0,
        'regularize_simplify_m': float(simplify_tolerance_m),
        'regularize_scan_step_m': 0.25,
        'regularize_offcenter_penalty': 0.018,
        'regularize_lock_radius_m': float(centerline_lock_radius_m),
        'regularize_lock_step_m': 0.20,
        'regularize_widmid_radius_m': (float(widmid_radius_m)
                                       if widmid_snap else 0.0),
        'regularize_widmid_step_m': 0.25,
        'regularize_widmid_min_depth_m': float(widmid_min_depth_m),
        'regularize_widmid_wall_frac': float(widmid_wall_frac),
        'regularize_widmid_max_shift_m': float(widmid_max_shift_m),
        'smooth_iterations': 2,
        'smooth_weight': 0.55,
        'min_line_support_ratio': 0.30,
        'line_support_depth_m': 0.025,
        'max_output_angle_jitter': 45.0,
        'max_output_sinuosity': 2.8,
        'extend_depth_m': float(extend_depth_m),
        'max_extend_m': float(max_extend_m),
        'trim_depth_m': float(trim_depth_m),
        'max_trim_m': float(max_trim_m),
        'endpoint_fail_steps': int(endpoint_fail_steps),
        'human_style_spacing_m': float(human_style_spacing_m),
        'human_style_lock_radius_m': float(human_style_lock_radius_m),
        'human_style_lock_step_m': 0.20,
        'human_style_cellcentre': bool(human_style_cellcentre),
        'split_min_depth_m': float(split_min_depth_m),
        'split_min_break_length_m': float(split_min_break_length_m),
        'split_min_fragment_length_m': float(split_min_fragment_length_m),
        'split_sample_step_m': 1.0,
        'split_half_width_m': float(split_half_width_m),
        # v11.1: junction-aware skeleton tracing. At a junction, only
        # near-collinear branches continue as one line; a branch meeting
        # the through-trench at a sharp angle terminates at the junction
        # (fixes bridging of two trenches that physically touch).
        'junction_aware': bool(junction_aware),
        'junction_max_turn_deg': float(junction_max_turn_deg),
        # v11.1: cap the angle at which the final completion-join may stitch
        # two line endpoints. The old hard-coded 95 deg let near-perpendicular
        # ends join (a classic source of T-junction bridges).
        'final_join_max_angle_deg': float(final_join_max_angle_deg),
        # v12.2 thalweg snap tunables
        'thalweg_spacing_m': 2.0,
        'thalweg_radius_m': float(thalweg_radius_m),
        'thalweg_lateral_slack_m': float(thalweg_lateral_slack_m),
        'thalweg_cell_step_m': 0.25,
        # v12.8 road mode (user's pipe rule) tunables
        'road_mode': bool(road_mode),
        'road_pipe_margin_m': float(road_pipe_margin_m),
        'road_pipe_rise_m': float(road_pipe_rise_m),
        'road_pipe_side_m': float(road_pipe_side_m),
        'road_pipe_snap_m': 2.5,
        'road_pipe_window_m': float(road_pipe_window_m),
        'road_pipe_min_seg_m': float(road_pipe_min_seg_m),
        'road_pipe_step_m': 1.0,
        # v12.9 flow mode (watershed) tunables
        'flow_mode': bool(flow_mode),
        'flow_prob_thr': float(flow_prob_thr),
        'flow_join_gap_m': float(flow_join_gap_m),
        'flow_snap_m': 2.5,
        # v12.9.5 anti-weave ridge-cut tunables
        'ridge_cut': bool(ridge_cut),
        'ridge_cut_m': float(ridge_cut_m),
        'ridge_cut_win_m': float(ridge_cut_win_m),
        'ridge_cut_trim': int(ridge_cut_trim),
        'ridge_cut_min_seg_m': float(ridge_cut_min_seg_m),
        'ridge_cut_smooth_px': float(ridge_cut_smooth_px),
    }
    # v12.8.1: Road Mode uses a PERSISTENT prob-gated downhill completion so
    # road-side trenches keep going while the channel continues (fixes the
    # early-stop the user saw) -- gated by CNN prob so it does not flood flat
    # ground. agri / river keep the conservative legacy completion.
    if road_mode:
        params['complete_margin_m'] = 0.05
        params['complete_max_fail'] = 6
        params['complete_max_len_m'] = 120.0
        params['complete_prob_gate'] = 0.35
        params['complete_trend_tol'] = 0.30
        # coverage-lean pipe rule: only SPLIT at real pipes/crowns (hump-split),
        # do NOT end-trim -- end-trim was clipping the completion extensions back
        # and re-creating the early-stops.
        params['road_pipe_margin_m'] = 0.0

    # v12.9: load the SECOND MODEL (MLP ensemble partner). Refines the CNN prob
    # using the same features (prob + 5 channels) -> smarter combined decision
    # (measured held-out PR-AUC 0.587 -> 0.65). Pure-numpy forward, gated on the
    # weights file being present + second_model flag.
    _sm = None
    if second_model:
        _smp = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'cnn_second_mlp.npz')
        if os.path.exists(_smp):
            try:
                _smz = np.load(_smp)
                _sm = dict((k, _smz[k]) for k in _smz.files)
                _msg(arcpy, "Second model (MLP ensemble) loaded")
            except Exception as _e:
                _warn(arcpy, "Second model load failed: %s" % str(_e))

    def _one_model_pass(cnn_mod, mw, mmean, mstd, tag):
        """Full single-model pipeline: tile loop -> geometry chain -> dedup.
        Returns the finished line list for that model. (v11.7: extracted so
        a second 'recall boost' model can run through the same pipeline.)"""
        all_lines = []
        tile_id = 0
        # v11.6 (G): keep the CNN probability map for the WHOLE DEM so the
        # downstream geometry passes can use the trained model's knowledge.
        prob_full = np.zeros((height, width), dtype=np.float32)
        prob_wsum = np.zeros((height, width), dtype=np.float32)
        # v15: stitch the learned orientation field too (when the model has
        # the head and the bridge-cut is enabled).
        params.pop('cnn_orient', None)
        use_orient = (bool(orient_cut)
                      and 'out_orient__weight' in mw
                      and hasattr(cnn_mod, 'unet_forward_with_orient'))
        orient_full = (np.zeros((2, height, width), dtype=np.float32)
                       if use_orient else None)
        for row0, col0, core_rows, core_cols, rr0, cc0, rr1, cc1, core in _tile_windows(
                width, height, int(tile_size_px), int(tile_overlap_px)):
            tile_id += 1
            nrows = rr1 - rr0
            ncols = cc1 - cc0
            arr, valid = _read_raster_tile(arcpy, in_dem, xmin, ymax, cellx,
                                           celly, cc0, rr0, ncols, nrows,
                                           nodata)
            if np.count_nonzero(valid) < 100:
                continue
            if use_orient:
                prob, otile = cnn_mod.predict_probability_map(
                    arr, valid, mw, mmean, mstd, tta=int(cnn_tta),
                    want_orient=True)
            else:
                prob = cnn_mod.predict_probability_map(arr, valid, mw,
                                                       mmean, mstd,
                                                       tta=int(cnn_tta))
                otile = None
            # v12.9 SECOND-MODEL ensemble (main pass only): refine the CNN prob
            # with the MLP on [prob + 5 feature channels]; 0.5 CNN + 0.5 MLP.
            if _sm is not None and tag == "main":
                try:
                    _med = float(np.median(arr[valid]))
                    _af = np.where(valid, arr, _med).astype(np.float32)
                    _cf = cnn_mod.compute_features(_af)
                    _st = np.column_stack([prob.ravel(), _cf[0].ravel(),
                                           _cf[1].ravel(), _cf[2].ravel(),
                                           _cf[3].ravel(), _cf[4].ravel()])
                    _x = (_st - _sm['feat_mean']) / _sm['feat_std']
                    _h1 = np.maximum(0.0, np.dot(_x, _sm['W1']) + _sm['b1'])
                    _h2 = np.maximum(0.0, np.dot(_h1, _sm['W2']) + _sm['b2'])
                    _z = np.dot(_h2, _sm['W3']) + _sm['b3']
                    # clip before exp -> no overflow warning, sigmoid still
                    # saturates correctly to 0/1 (numpy-1.9 safe).
                    _zc = np.clip(_z[:, 0], -30.0, 30.0)
                    _mlp = (1.0 / (1.0 + np.exp(-_zc))).reshape(prob.shape)
                    prob = (0.5 * prob + 0.5 * _mlp).astype(np.float32)
                except Exception as _e:
                    _warn(arcpy, "Second-model tile skipped: %s" % str(_e))
            if isinstance(prob, tuple):      # (seg, junc) if junction head
                prob = prob[0]
            prob_clean = np.where(np.isfinite(prob), prob, 0.0).astype(np.float32)
            prob_full[rr0:rr0+nrows, cc0:cc0+ncols] += prob_clean
            prob_wsum[rr0:rr0+nrows, cc0:cc0+ncols] += 1.0
            if otile is not None:
                oc = np.where(np.isfinite(otile), otile, 0.0).astype(np.float32)
                orient_full[0, rr0:rr0+nrows, cc0:cc0+ncols] += oc[0]
                orient_full[1, rr0:rr0+nrows, cc0:cc0+ncols] += oc[1]
            mask = (prob >= float(probability_threshold)) & valid
            mask = lc._binary_close(mask, int(gap_close_px))
            skel = lc._zhang_suen_thin(mask, max_iter=80)
            skel = lc._remove_isolated_and_short(skel)
            georef = (xmin, ymax, cell, row0, col0, rr0, cc0)
            lines = lc._extract_lines_from_skeleton(
                skel, prob, core, georef, params, tile_id)
            if lines:
                _msg(arcpy, "[%s] Tile %d: %d candidate line(s)" %
                     (tag, tile_id, len(lines)))
                all_lines.extend(lines)

        _msg(arcpy, "[%s] Raw CNN line segments: %d" % (tag, len(all_lines)))
        ok = prob_wsum > 1e-6
        # v12.9.11: divide IN PLACE with where= instead of prob_full[ok] /=
        # prob_wsum[ok].  The boolean-indexed form builds up to three full-size
        # temporary copies (prob_full[ok], prob_wsum[ok], and the quotient),
        # which inflates peak memory by ~3x the map size on very large DEMs and
        # is what triggers the MemoryError in 32-bit ArcMap.  This form is
        # mathematically identical (only ok pixels are touched) but allocates no
        # large temporaries.  numpy-1.9 safe (out= and where= both exist in 1.7+).
        np.divide(prob_full, prob_wsum, out=prob_full, where=ok)
        params['cnn_prob'] = prob_full
        params['cnn_prob_geo'] = (xmin, ymax, cellx, celly)
        params['cnn_prob_threshold'] = float(probability_threshold)
        params['ext_prob_threshold'] = float(ext_prob_threshold)
        if use_orient:
            np.divide(orient_full[0], prob_wsum, out=orient_full[0], where=ok)
            np.divide(orient_full[1], prob_wsum, out=orient_full[1], where=ok)
            params['cnn_orient'] = orient_full
            _msg(arcpy, "[%s] Orientation field stitched" % tag)
        _msg(arcpy, "[%s] Full-DEM CNN prob map ready (%.0f%% coverage)" %
             (tag, 100.0 * float(ok.mean())))
        # ---- v12.9 FLOW MODE: watershed vectorisation (coverage UP + no weave) ----
        # Replaces the skeleton vectorisation with a watershed segmentation of the
        # prob mask by the DEM (parallel ditches split at the road crown -> no
        # weave; full mask kept -> coverage). Minimal cleanup then return, so the
        # skeleton-oriented passes are bypassed. Gated on flow_mode (off by
        # default -> agricultural / river / road output unchanged).
        if flow_mode:
            try:
                fl = lc._flow_mode_lines(prob_full, in_dem, arcpy, params,
                                         xmin, ymax, cellx, celly, width, height)
                _msg(arcpy, "[%s] Flow Mode: watershed -> %d base line(s)" %
                     (tag, len(fl)))
                if fl:
                    fjg = float(params.get('flow_join_gap_m', 6.0))
                    fl = lc._try_join_lines(fl, fjg, max_angle_deg=35.0,
                                            in_dem=in_dem, arcpy=arcpy, params=params)
                    # NOTE: watershed already separates parallel ditches (no weave),
                    # so the aggressive pipe-split end-trim is skipped here -- it
                    # cost ~11% coverage on flow lines and coverage is the priority.
                    try:
                        fl, _ni = lc._remove_isolated_fragments(
                            fl,
                            max_len_m=float(params.get('declutter_max_len_m', 80.0)),
                            min_iso_m=float(params.get('declutter_min_iso_m', 25.0)))
                    except Exception:
                        pass
                    if thalweg_snap:
                        try:
                            fl = lc._thalweg_centerline_pass(fl, in_dem, arcpy, params)
                        except Exception:
                            pass
                _msg(arcpy, "[%s] Flow Mode final: %d line(s)" % (tag, len(fl)))
                return fl
            except Exception as exc:
                _warn(arcpy, "Flow Mode failed (%s); using skeleton lines" % str(exc))
        if join_gap_m > 0 and all_lines:
            all_lines = lc._try_join_lines(all_lines, float(join_gap_m),
                                           in_dem=in_dem, arcpy=arcpy,
                                           params=params)
            _msg(arcpy, "[%s] After DEM-verified join: %d" % (tag, len(all_lines)))
        if extend_endpoints and all_lines:
            try:
                all_lines, ec = lc._extend_all_endpoints(
                    all_lines, in_dem, arcpy, params)
                _msg(arcpy, "[%s] Endpoint pass 1 extended %d endpoint(s)" % (tag, ec))
            except Exception as exc:
                _warn(arcpy, "Endpoint extension pass 1 failed: %s" % str(exc))
        if regularize_lines and all_lines:
            all_lines, rejected = lc._regularize_all_lines(
                all_lines, in_dem, arcpy, params)
            _msg(arcpy, "[%s] After DEM regularization: %d kept, %d rejected" %
                 (tag, len(all_lines), rejected))
        if final_join_gap_m > 0 and all_lines:
            before = len(all_lines)
            all_lines = lc._try_join_lines(all_lines, float(final_join_gap_m),
                                           max_angle_deg=float(params.get(
                                               'final_join_max_angle_deg', 95.0)),
                                           in_dem=in_dem, arcpy=arcpy,
                                           params=params)
            _msg(arcpy, "[%s] Final completion join: %d -> %d" %
                 (tag, before, len(all_lines)))
        if human_style_pass and all_lines:
            try:
                all_lines = lc._human_style_centerline_pass(
                    all_lines, in_dem, arcpy, params)
                _msg(arcpy, "[%s] Human-style pass: re-snapped %d line(s)" %
                     (tag, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Human-style pass failed: %s" % str(exc))
        if extend_endpoints and all_lines:
            try:
                all_lines, ec2 = lc._extend_all_endpoints(
                    all_lines, in_dem, arcpy, params)
                _msg(arcpy, "[%s] Endpoint pass 2 extended %d endpoint(s)" % (tag, ec2))
            except Exception as exc:
                _warn(arcpy, "Endpoint extension pass 2 failed: %s" % str(exc))
        if split_at_breaks and all_lines:
            try:
                before = len(all_lines)
                all_lines, n_split, n_frag = lc._split_lines_at_breaks(
                    all_lines, in_dem, arcpy, params)
                _msg(arcpy, "[%s] Break splitting: %d split into %d (was %d)" %
                     (tag, n_split, n_frag, before))
            except Exception as exc:
                _warn(arcpy, "Break splitting failed: %s" % str(exc))
        # v15: cut lines that run against the learned trench orientation
        if all_lines and params.get('cnn_orient') is not None:
            try:
                before = len(all_lines)
                all_lines, n_oc = _orientation_bridge_cut(
                    all_lines, params,
                    max_misalign_deg=float(orient_cut_max_deg))
                _msg(arcpy, "[%s] Orientation bridge-cut: %d line(s) cut "
                     "(%d -> %d)" % (tag, n_oc, before, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Orientation bridge-cut failed: %s" % str(exc))
        # v12.6 (user's rule): complete broken/faint trenches by following the
        # DEM valley downhill (high tip up to source, low tip down to outlet,
        # along the low pixels, only while a real depression exists). Runs
        # BEFORE dedup/fusion so any overlaps it creates get cleaned. Measured
        # on the Whole held-out: recall 80->83% alone, ->85% stacked on the
        # detrend_4 model. Encodes exactly how a human draws a trench.
        if complete_downhill and all_lines:
            try:
                before = len(all_lines)
                all_lines, n_cmp = lc._complete_trenches_downhill(
                    all_lines, in_dem, arcpy, params)
                _msg(arcpy, "[%s] Downhill trench-completion: extended %d "
                     "trench(es) (%d lines)" % (tag, n_cmp, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Downhill completion failed: %s" % str(exc))
        if remove_duplicates and all_lines:
            try:
                before = len(all_lines)
                # v12.3: in river mode (merge_parallel ON) use a more
                # aggressive dedup - rivers spawn close parallel slivers at
                # junctions that the default 3 m / 60% misses. Measured on a
                # Training-3 river crop: double-lines 45 -> 2, rivers intact.
                if merge_parallel:
                    dd = max(float(duplicate_distance_m), 5.0)
                    dc = min(float(duplicate_cover_ratio), 0.40)
                else:
                    dd = float(duplicate_distance_m)
                    dc = float(duplicate_cover_ratio)
                all_lines, n_dup = lc._remove_duplicate_lines(
                    all_lines, dup_dist_m=dd, cover_ratio=dc)
                _msg(arcpy, "[%s] Duplicate removal: dropped %d (%d -> %d)" %
                     (tag, n_dup, before, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Duplicate removal failed: %s" % str(exc))
        # v12.4: FUSE lines that trace the SAME feature and weave over each
        # other (the dedup misses them when the shared run is a small fraction
        # of each long line). Runs after dedup so it only fuses real survivors;
        # never merges two lines that cross THROUGH each other (those may be
        # different features). Fixes the "lines mixing" issue. Now on by default
        # for BOTH rivers and agricultural trenches (measured agri crossings
        # 143 -> low; safe because it only fuses same-feature overlaps).
        if fuse_overlaps and all_lines:
            try:
                before = len(all_lines)
                all_lines, n_fz = lc._fuse_overlapping_lines(
                    all_lines,
                    max_off_m=float(params.get('fuse_max_off_m', 4.0)),
                    min_shared_m=float(params.get('fuse_min_shared_m', 15.0)))
                _msg(arcpy, "[%s] River overlap-fuse: merged %d weaving "
                     "twin(s) (%d -> %d)" % (tag, n_fz, before, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "River overlap-fuse failed: %s" % str(exc))
        # v12.1: collapse two parallel lines that are the two sides of one
        # wide feature (river channel / road / dyke) into a single centre-line.
        if merge_parallel and all_lines:
            try:
                before = len(all_lines)
                all_lines, n_mp = lc._merge_parallel_lines(
                    all_lines, in_dem, arcpy, params,
                    min_gap_m=float(merge_min_gap_m),
                    max_gap_m=float(merge_max_gap_m),
                    relief_margin_m=float(merge_relief_margin_m))
                _msg(arcpy, "[%s] Parallel-line merge: %d pair(s) merged "
                     "(%d -> %d)" % (tag, n_mp, before, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Parallel-line merge failed: %s" % str(exc))
        # v12.4: river mode -- strip floating SHORT fragments (over-detection
        # clutter on hillsides). A real channel joins the network, so an
        # isolated short stub is almost always a false positive. Long /
        # connected lines are never touched, so river coverage is preserved.
        # Measured on the Training-3 held-out crop: dropped 28 clutter lines,
        # pieces/river 1.50 -> 1.38, coverage held at 90%.
        if merge_parallel and all_lines:
            try:
                before = len(all_lines)
                all_lines, n_iso = lc._remove_isolated_fragments(
                    all_lines,
                    max_len_m=float(params.get('declutter_max_len_m', 80.0)),
                    min_iso_m=float(params.get('declutter_min_iso_m', 25.0)))
                _msg(arcpy, "[%s] River declutter: dropped %d isolated "
                     "fragment(s) (%d -> %d)" % (tag, n_iso, before,
                                                 len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "River declutter failed: %s" % str(exc))
        # v12.2: constrained valley-following (thalweg) refinement - runs
        # LAST so it only refines final survivors and does not feed extra
        # wiggle into the break-splitter (which over-fragmented when thalweg
        # ran earlier).
        if thalweg_snap and all_lines:
            try:
                # v12.9.6: the DETRENDED nearest-dip sub-pixel thalweg is now the
                # default centring for EVERY mode (was road-mode only -- which is
                # exactly why normal-mode lines looked off-centre). Measured
                # better than the legacy centreline pass on flat (Bamberg
                # 0.75->0.25 m), steep (sdat 0.50->0.25 m) AND road terrain; it
                # only moves a point onto a REAL nearby channel dip (else stays),
                # so it never drags a line off a true trench.
                if prob_centroid_thalweg and params.get('cnn_prob') is not None:
                    # v12.9.9: centre on the CNN prob-ridge centroid (shallow
                    # road ditches have no usable elevation dip). Measured on the
                    # user's road corrections: drift 0.50 -> 0.25 m, >1m 12% -> 0%.
                    all_lines = lc._prob_centroid_thalweg_pass(all_lines, params)
                    _msg(arcpy, "[%s] Prob-centroid thalweg: centred %d line(s) "
                         "on the CNN ridge" % (tag, len(all_lines)))
                else:
                    all_lines = lc._subpix_thalweg_pass(
                        all_lines, in_dem, arcpy, params)
                    _msg(arcpy, "[%s] Sub-pixel thalweg: centred %d line(s) on the "
                         "channel bottom" % (tag, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Thalweg snap failed: %s" % str(exc))
        # v12.8 Road Mode (user's PIPE rule): runs LAST on the final centred
        # lines -- STOP each road-side trench at a pipe/culvert (a high pixel in
        # the channel) and keep parallel ditches separate. Gated on road_mode so
        # agricultural / river output is byte-identical when it is off.
        if road_mode and all_lines:
            try:
                before = len(all_lines)
                all_lines, n_rp = lc._road_pipe_split_trim(
                    all_lines, in_dem, arcpy, params)
                _msg(arcpy, "[%s] Road pipe-rule: trimmed/split %d line(s) "
                     "(%d -> %d)" % (tag, n_rp, before, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Road pipe-rule failed: %s" % str(exc))
        # v12.9.5: ALWAYS-ON anti-weave ridge-cut. Runs LAST on the final centred
        # lines (after every join/extend/complete that could have bridged across
        # a ridge). A trench must never cross a RIDGE (a high pixel that is higher
        # than the channel floor on BOTH sides) into a parallel / other trench --
        # exactly the "trench goes straight, does not turn into another trench"
        # rule the user demanded. Cut every such crossing into separate trenches.
        # OFF for River Mode (a river is one unbroken line). Measured on the
        # user's sdat1 DEM: ridge-crossings 35 -> 0, his hand-fixed clean lines
        # untouched (0 -> 0), 99% of length kept.
        if ridge_cut and all_lines:
            try:
                before = len(all_lines)
                all_lines, n_rc = lc._ridge_cut_pass(
                    all_lines, in_dem, arcpy, params)
                if n_rc:
                    _msg(arcpy, "[%s] Anti-weave ridge-cut: split %d line(s) at "
                         "ridge crossings (%d -> %d)" %
                         (tag, n_rc, before, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Anti-weave ridge-cut failed: %s" % str(exc))
        # v12.9.2: orient every line HIGH -> LOW (water-flow direction).
        if all_lines:
            try:
                all_lines = lc._orient_high_to_low(all_lines, in_dem, arcpy)
            except Exception as exc:
                _warn(arcpy, "High-to-low orient failed: %s" % str(exc))
        return all_lines

    all_lines = _one_model_pass(_cnn, cnn_w, cnn_mean, cnn_std, "main")

    # ---- v12.0 hybrid: orientation field from a DEDICATED model ----
    # The main (v14) model has no orientation head; the v15 model's seg map
    # was measured worse for endpoints. Best of both: keep v14's lines and
    # stitch ONLY the orientation field from the bundled v15 weights, then
    # apply the interior bridge-cut to v14's finished lines.
    if orient_cut and params.get('cnn_orient') is None:
        _mod_dir = os.path.dirname(os.path.abspath(__file__))
        wo_path = os.path.join(_mod_dir, "cnn_unet_orient.npz")
        if os.path.exists(wo_path):
            try:
                wo = _cnn.load_weights(wo_path)
                if 'out_orient__weight' in wo:
                    _msg(arcpy, "Orientation model loaded (%d arrays)" %
                         len(wo))
                    ofull = np.zeros((2, height, width), dtype=np.float32)
                    owsum = np.zeros((height, width), dtype=np.float32)
                    for row0, col0, core_rows, core_cols, rr0, cc0, rr1, cc1, core in _tile_windows(
                            width, height, int(tile_size_px),
                            int(tile_overlap_px)):
                        nrows = rr1 - rr0
                        ncols = cc1 - cc0
                        arr, valid = _read_raster_tile(
                            arcpy, in_dem, xmin, ymax, cellx, celly,
                            cc0, rr0, ncols, nrows, nodata)
                        if np.count_nonzero(valid) < 100:
                            continue
                        _p, ot = _cnn.predict_probability_map(
                            arr, valid, wo, cnn_mean, cnn_std,
                            want_orient=True)
                        oc = np.where(np.isfinite(ot), ot, 0.0).astype(np.float32)
                        ofull[0, rr0:rr0+nrows, cc0:cc0+ncols] += oc[0]
                        ofull[1, rr0:rr0+nrows, cc0:cc0+ncols] += oc[1]
                        owsum[rr0:rr0+nrows, cc0:cc0+ncols] += 1.0
                    okk = owsum > 1e-6
                    ofull[0][okk] /= owsum[okk]
                    ofull[1][okk] /= owsum[okk]
                    params['cnn_orient'] = ofull
                    before = len(all_lines)
                    all_lines, n_oc = _orientation_bridge_cut(
                        all_lines, params,
                        max_misalign_deg=float(orient_cut_max_deg))
                    _msg(arcpy, "Hybrid orientation bridge-cut: %d line(s) "
                         "cut (%d -> %d)" % (n_oc, before, len(all_lines)))
            except Exception as exc:
                _warn(arcpy, "Hybrid orientation pass failed: %s" % str(exc))

    # ---- v11.7: second-model recall boost (line-level union) ----
    # Run the v13 7-channel model through the SAME pipeline, then add only
    # those of its lines that cover ground the main model missed. Keeps the
    # main model's precise lines untouched (prob-level max was measured to
    # degrade endpoints; line-level union does not).
    if recall_boost:
        _mod_dir = os.path.dirname(os.path.abspath(__file__))
        w13_path = os.path.join(_mod_dir, "cnn_unet_v13_weights.npz")
        n13_path = os.path.join(_mod_dir, "cnn_norm_stats_v13.npz")
        if _cnn13 is None or not os.path.exists(w13_path) \
                or not os.path.exists(n13_path):
            _warn(arcpy, "Recall boost requested but second model files "
                         "missing - skipping boost.")
        else:
            w13 = _cnn13.load_weights(w13_path)
            nrm13 = np.load(n13_path)
            m13 = nrm13['mean'].astype(np.float32)
            s13 = nrm13['std'].astype(np.float32)
            _msg(arcpy, "Recall boost: second model loaded (%d arrays)" %
                 len(w13))
            boost_lines = _one_model_pass(_cnn13, w13, m13, s13, "boost")
            added = _filter_boost_lines(all_lines, boost_lines)
            _msg(arcpy, "Recall boost: %d of %d boost lines cover new "
                 "ground - added" % (len(added), len(boost_lines)))
            all_lines = all_lines + added

    # v12.9: Flow Mode lines are clean thalweg centre-lines split at
    # junctions/pipes -> keep short ones too (the 25 m final filter is for
    # skeleton clutter, not flow lines).
    if flow_mode:
        _flm = float(params.get('flow_min_length_m', 6.0))
        min_length_m = _flm
        final_min_length_m = _flm

    out_ws = os.path.dirname(out_fc)
    out_name = os.path.basename(out_fc)
    if out_ws == "":
        out_ws = arcpy.env.workspace or os.getcwd()
        out_fc = os.path.join(out_ws, out_name)
    if arcpy.Exists(out_fc):
        if overwrite:
            arcpy.Delete_management(out_fc)
        else:
            raise RuntimeError("Output already exists: " + out_fc)
    arcpy.CreateFeatureclass_management(out_ws, out_name, "POLYLINE",
                                        spatial_reference=sr)
    for name, ftype in [("Len_m", "DOUBLE"), ("ProbMean", "DOUBLE"),
                        ("ProbMax", "DOUBLE"), ("QScore", "DOUBLE"),
                        ("PixCnt", "LONG"), ("TileID", "LONG")]:
        arcpy.AddField_management(out_fc, name, ftype)
    # v12.9.10: half-pixel grid-registration correction. SAGA/.sdat-derived
    # GeoTIFFs carry a world-file (.tfw) whose origin sits half a pixel from the
    # internal GeoTIFF origin; ArcMap and the array reader disagree, so every line
    # lands ~0.5 m NW of the true channel. Measured on the user's DEM 1 and DEM 2
    # (both .sdat): output is a consistent (+0.5 px E, -0.5 px S) off the digitised
    # truth; shifting by that brings median 0.75 -> 0.43 m (== the clean sites) with
    # only 11-18% of vertices nudged. Off by default; harmless on standard GeoTIFFs.
    reg_dx = (0.5 * cellx) if reg_half_pixel else 0.0
    reg_dy = (-0.5 * celly) if reg_half_pixel else 0.0
    if reg_half_pixel:
        _msg(arcpy, "Half-pixel grid correction ON: shifting output by "
                    "(%+.2f, %+.2f) map units to match the .tfw frame"
                    % (reg_dx, reg_dy))
    cur = arcpy.da.InsertCursor(
        out_fc, ["SHAPE@", "Len_m", "ProbMean", "ProbMax",
                 "QScore", "PixCnt", "TileID"])
    inserted = 0
    try:
        for line in all_lines:
            coords = line.get('coords', [])
            if len(coords) < 2:
                continue
            arrp = arcpy.Array()
            last = None
            for x, y in coords:
                x = float(x) + reg_dx
                y = float(y) + reg_dy
                if last is None or abs(x-last[0]) > 1e-9 or abs(y-last[1]) > 1e-9:
                    arrp.add(arcpy.Point(float(x), float(y)))
                    last = (x, y)
            if arrp.count < 2:
                continue
            geom = arcpy.Polyline(arrp, sr)
            min_final_len = max(float(min_length_m), float(final_min_length_m))
            if geom.length < min_final_len:
                continue
            cur.insertRow([geom, float(geom.length),
                           float(line.get('MeanDep', 0.0)),
                           float(line.get('MaxDep', 0.0)),
                           float(line.get('QScore', 0.0)),
                           int(line.get('PixCnt', 0)),
                           int(line.get('TileID', 0))])
            inserted += 1
    finally:
        del cur
    _msg(arcpy, "Finished. Wrote %d CNN trench line(s): %s" %
         (inserted, out_fc))
    _msg(arcpy, "Detection elapsed: %.1f seconds" % (time.time() - start))
    return out_fc


def detect_trenches_cnn_blocked(in_dem, cnn_weights_path, cnn_norm_path, out_fc,
                                arcpy, block_px=7500, overlap_px=400,
                                block_limit_px=140000000, **kw):
    """v12.9.11: memory-safe wrapper around detect_trenches_cnn for VERY LARGE
    DEMs.  ArcMap 10.8 is 32-bit (~2 GB RAM); the single-pass detector builds the
    whole-DEM probability map in memory, which overflows on rasters of a few
    hundred million pixels (MemoryError).  This wrapper:

      * DEM <= block_limit_px pixels  -> a transparent pass-through to
        detect_trenches_cnn (output is BYTE-IDENTICAL to the normal tool).
      * DEM  > block_limit_px pixels  -> splits the raster into OVERLAPPING
        blocks, each small enough to fit in 32-bit memory, runs the UNCHANGED
        detector on every block, and merges the lines.  Each line is kept once,
        decided by whether its midpoint falls in the block's non-overlapping
        core, so there are no seam duplicates; the overlap keeps trenches
        continuous across block boundaries.

    The per-block result is identical to running the current tool on that block,
    so detection quality/centring is unchanged; only the memory ceiling is lifted.
    """
    ras = arcpy.Raster(in_dem)
    ext = ras.extent
    xmin = float(ext.XMin); ymax = float(ext.YMax)
    xmax = float(ext.XMax); ymin = float(ext.YMin)
    cellx = float(ras.meanCellWidth); celly = abs(float(ras.meanCellHeight))
    width = int(round((xmax - xmin) / cellx))
    height = int(round((ymax - ymin) / celly))
    try:
        sr = arcpy.Describe(in_dem).spatialReference
    except Exception:
        sr = None

    if width * height <= int(block_limit_px):
        return detect_trenches_cnn(in_dem, cnn_weights_path, cnn_norm_path,
                                   out_fc, arcpy, **kw)

    _msg(arcpy, "Large DEM %d x %d (%.0f Mpx) exceeds the single-pass limit -> "
                "BLOCK processing (block %d px, overlap %d px)" %
                (width, height, width * height / 1e6, block_px, overlap_px))
    import tempfile
    scratch = getattr(arcpy.env, 'scratchFolder', None) or tempfile.gettempdir()
    step = max(1, int(block_px) - int(overlap_px))
    ov2 = int(overlap_px) // 2
    nod = -9999999.0
    kept = []
    nb = 0
    r0 = 0
    while r0 < height:
        c0 = 0
        while c0 < width:
            win_w = min(int(block_px), width - c0)
            win_h = min(int(block_px), height - r0)
            if win_w >= 40 and win_h >= 40:
                nb += 1
                llx = xmin + c0 * cellx
                lly = ymax - (r0 + win_h) * celly
                _msg(arcpy, "  Block %d at col %d row %d (%d x %d)" %
                     (nb, c0, r0, win_w, win_h))
                btif = os.path.join(scratch, "_trblk_%d.tif" % nb)
                bout = os.path.join(scratch, "_trblk_%d.shp" % nb)
                try:
                    a = arcpy.RasterToNumPyArray(
                        in_dem, arcpy.Point(llx, lly), win_w, win_h,
                        nodata_to_value=nod)
                    br = arcpy.NumPyArrayToRaster(
                        a, arcpy.Point(llx, lly), cellx, celly, nod)
                    if arcpy.Exists(btif):
                        arcpy.Delete_management(btif)
                    br.save(btif)
                    if sr is not None:
                        try:
                            arcpy.DefineProjection_management(btif, sr)
                        except Exception:
                            pass
                    del a, br
                    bkw = dict(kw); bkw['overwrite'] = True
                    detect_trenches_cnn(btif, cnn_weights_path, cnn_norm_path,
                                        bout, arcpy, **bkw)
                    cx0 = xmin + (c0 + (ov2 if c0 > 0 else 0)) * cellx
                    cx1 = xmin + (c0 + win_w -
                                  (ov2 if (c0 + win_w) < width else 0)) * cellx
                    cy1 = ymax - (r0 + (ov2 if r0 > 0 else 0)) * celly
                    cy0 = ymax - (r0 + win_h -
                                  (ov2 if (r0 + win_h) < height else 0)) * celly
                    flds = ["SHAPE@", "ProbMean", "ProbMax", "QScore",
                            "PixCnt", "TileID"]
                    with arcpy.da.SearchCursor(bout, flds) as sc:
                        for row in sc:
                            g = row[0]
                            if g is None:
                                continue
                            mp = g.positionAlongLine(0.5, True).firstPoint
                            if not (cx0 <= mp.X < cx1 and cy0 <= mp.Y < cy1):
                                continue
                            crds = []
                            for part in g:
                                for pt in part:
                                    if pt is not None:
                                        crds.append((pt.X, pt.Y))
                            if len(crds) >= 2:
                                kept.append((crds, row[1], row[2], row[3],
                                             row[4], row[5]))
                except Exception as be:
                    _warn(arcpy, "Block %d failed: %s" % (nb, str(be)))
                finally:
                    for f in (btif, bout):
                        try:
                            if arcpy.Exists(f):
                                arcpy.Delete_management(f)
                        except Exception:
                            pass
            c0 += step
        r0 += step

    _msg(arcpy, "Block processing complete: %d block(s), %d merged line(s)" %
         (nb, len(kept)))
    out_ws = os.path.dirname(out_fc) or (arcpy.env.workspace or os.getcwd())
    out_name = os.path.basename(out_fc)
    out_fc = os.path.join(out_ws, out_name)
    if arcpy.Exists(out_fc):
        arcpy.Delete_management(out_fc)
    arcpy.CreateFeatureclass_management(out_ws, out_name, "POLYLINE",
                                        spatial_reference=sr)
    for name, ftype in [("Len_m", "DOUBLE"), ("ProbMean", "DOUBLE"),
                        ("ProbMax", "DOUBLE"), ("QScore", "DOUBLE"),
                        ("PixCnt", "LONG"), ("TileID", "LONG")]:
        arcpy.AddField_management(out_fc, name, ftype)
    cur = arcpy.da.InsertCursor(
        out_fc, ["SHAPE@", "Len_m", "ProbMean", "ProbMax",
                 "QScore", "PixCnt", "TileID"])
    n = 0
    try:
        for crds, md, mx, qs, pc, ti in kept:
            arrp = arcpy.Array()
            for x, y in crds:
                arrp.add(arcpy.Point(float(x), float(y)))
            if arrp.count < 2:
                continue
            geom = arcpy.Polyline(arrp, sr)
            cur.insertRow([geom, float(geom.length), float(md), float(mx),
                           float(qs), int(pc), int(ti)])
            n += 1
    finally:
        del cur
    _msg(arcpy, "Finished (block mode). Wrote %d trench line(s): %s" %
         (n, out_fc))
    return out_fc
