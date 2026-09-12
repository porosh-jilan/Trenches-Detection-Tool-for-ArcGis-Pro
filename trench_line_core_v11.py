# -*- coding: utf-8 -*-
"""
TrenchToolkit v15 - ArcMap 10.8 Core
====================================
Trench centreline detector for 1 m DEMs.

Built on v14's DEM-regularized output geometry with high-recall completion:
  * Same seed/grow depression detector for locally depressed linear paths.
  * DEM-verified stitching: nearby endpoints are joined only when the gap
    still looks like a connected depressed path in the DEM.
  * Endpoint trim + extension: endpoints are moved inward/outward to the
    first/last place where the cross-section still supports a depression.
  * Ground-truth-style line regularization: each detected line is refit to
    the local DEM depression centre, then smoothed and simplified so output
    polylines are not pixel-zigzag skeletons.
  * Final completion join: after regularization, nearby smooth segments are
    joined again when the DEM gap still has depression evidence.
  * Extra endpoint diagnostic fields are written to the output.

Designed for ArcMap 10.8 / Python 2.7. No external libraries required.
"""
from __future__ import print_function

import os
import sys
import math
import time
from collections import deque

import numpy as np
np.seterr(invalid='ignore')


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _msg(arcpy, text):
    try:
        if arcpy is not None:
            arcpy.AddMessage(str(text))
        else:
            print(str(text))
    except Exception:
        try:
            print(str(text))
        except Exception:
            pass


def _warn(arcpy, text):
    try:
        if arcpy is not None:
            arcpy.AddWarning(str(text))
        else:
            print("WARNING: " + str(text))
    except Exception:
        try:
            print("WARNING: " + str(text))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# NumPy morphology helpers (identical to v11)
# ---------------------------------------------------------------------------

NEIGH8 = [(-1, -1), (-1, 0), (-1, 1),
          (0, -1),           (0, 1),
          (1, -1),  (1, 0),  (1, 1)]


def _shift_float(a, dr, dc, fill=np.nan):
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


def _shift_bool(a, dr, dc):
    h, w = a.shape
    out = np.zeros_like(a, dtype=np.bool_)
    r0s = max(0, -dr); r1s = min(h, h - dr)
    c0s = max(0, -dc); c1s = min(w, w - dc)
    r0d = max(0, dr);  r1d = min(h, h + dr)
    c0d = max(0, dc);  c1d = min(w, w + dc)
    if r1s > r0s and c1s > c0s:
        out[r0d:r1d, c0d:c1d] = a[r0s:r1s, c0s:c1s]
    return out


def _neighbor_count(mask):
    cnt = np.zeros(mask.shape, dtype=np.uint8)
    for dr, dc in NEIGH8:
        cnt += _shift_bool(mask, dr, dc)
    return cnt


def _binary_dilate(mask, iterations=1):
    out = mask.astype(np.bool_).copy()
    for _ in range(int(max(0, iterations))):
        d = out.copy()
        for dr, dc in NEIGH8:
            d |= _shift_bool(out, dr, dc)
        out = d
    return out


def _binary_erode(mask, iterations=1):
    out = mask.astype(np.bool_).copy()
    for _ in range(int(max(0, iterations))):
        e = out.copy()
        for dr, dc in NEIGH8:
            e &= _shift_bool(out, dr, dc)
        out = e
    return out


def _binary_close(mask, iterations=1):
    if iterations <= 0:
        return mask.astype(np.bool_)
    return _binary_erode(_binary_dilate(mask, iterations), iterations)


def _gaussian_kernel1d(sigma):
    sigma = float(sigma)
    if sigma <= 0.05:
        return np.array([1.0], dtype=np.float32)
    radius = int(max(1, math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= np.sum(k)
    return k


def _nan_gaussian_smooth(arr, valid, sigma):
    if sigma <= 0.05:
        out = arr.astype(np.float32).copy()
        out[~valid] = np.nan
        return out

    a = arr.astype(np.float32).copy()
    w = valid.astype(np.float32)
    a[~valid] = 0.0

    k = _gaussian_kernel1d(sigma)
    rad = int((len(k) - 1) // 2)

    def conv_axis(data, axis):
        if axis == 0:
            pad = np.pad(data, ((rad, rad), (0, 0)), mode='edge')
            out = np.zeros_like(data, dtype=np.float32)
            for i, kv in enumerate(k):
                out += kv * pad[i:i + data.shape[0], :]
            return out
        else:
            pad = np.pad(data, ((0, 0), (rad, rad)), mode='edge')
            out = np.zeros_like(data, dtype=np.float32)
            for i, kv in enumerate(k):
                out += kv * pad[:, i:i + data.shape[1]]
            return out

    sm = conv_axis(conv_axis(a, 0), 1)
    sw = conv_axis(conv_axis(w, 0), 1)
    out = np.empty_like(sm, dtype=np.float32)
    ok = sw > 1.0e-6
    out[ok] = sm[ok] / sw[ok]
    if np.any(valid):
        med = float(np.nanmedian(arr[valid]))
    else:
        med = 0.0
    out[~ok] = med
    return out


# ---------------------------------------------------------------------------
# Depression evidence (identical to v11)
# ---------------------------------------------------------------------------

def _directional_depression(dem, valid, seed_depth, grow_depth, max_width_m,
                            cell_m):
    h, w = dem.shape
    depth_best = np.zeros((h, w), dtype=np.float32)
    orient_best = np.zeros((h, w), dtype=np.uint8)
    strong = np.zeros((h, w), dtype=np.bool_)
    grow = np.zeros((h, w), dtype=np.bool_)

    half_px = int(max(2, round(float(max_width_m) / (2.0 * float(cell_m)))))
    half_px = min(8, max(2, half_px))
    offsets = list(range(1, max(half_px + 4, 8)))

    normals = [(1, 0), (0, 1), (1, 1), (1, -1)]

    min_side_seed = max(0.04, float(seed_depth) * 0.45)
    min_side_grow = max(0.015, float(grow_depth) * 0.40)

    for oi, (dr, dc) in enumerate(normals, start=1):
        best_avg = np.full((h, w), -999999.0, dtype=np.float32)
        best_min_side = np.full((h, w), -999999.0, dtype=np.float32)
        best_local = np.full((h, w), -999999.0, dtype=np.float32)

        for off in offsets:
            a = _shift_float(dem, dr * off, dc * off)
            b = _shift_float(dem, -dr * off, -dc * off)
            va = _shift_bool(valid, dr * off, dc * off)
            vb = _shift_bool(valid, -dr * off, -dc * off)
            ok = va & vb

            avg = ((a + b) * 0.5) - dem
            min_side = np.minimum(a, b) - dem
            avg[~ok] = -999999.0
            min_side[~ok] = -999999.0

            better = avg > best_avg
            best_avg[better] = avg[better]
            best_min_side[better] = min_side[better]

            if off <= 2:
                local = min_side
                best_local = np.maximum(best_local, local)

        a3 = _shift_float(dem, dr * half_px, dc * half_px)
        b3 = _shift_float(dem, -dr * half_px, -dc * half_px)
        v3 = (_shift_bool(valid, dr * half_px, dc * half_px) &
              _shift_bool(valid, -dr * half_px, -dc * half_px))
        core_side = np.minimum(a3, b3) - dem
        core_side[~v3] = -999999.0

        local_seed = best_local >= max(0.025, float(seed_depth) * 0.25)
        local_grow = best_local >= max(0.012, float(grow_depth) * 0.25)
        core_grow = core_side >= min_side_grow
        core_seed = core_side >= min_side_seed

        s = ((best_avg >= float(seed_depth)) &
             (best_min_side >= min_side_seed) &
             local_seed & core_seed & valid)
        g = ((best_avg >= float(grow_depth)) &
             (best_min_side >= min_side_grow) &
             local_grow & core_grow & valid)

        strong |= s
        grow |= g

        better_global = best_avg > depth_best
        depth_best[better_global] = best_avg[better_global]
        orient_best[better_global] = oi

    depth_best[~valid] = 0.0
    strong[~valid] = False
    grow[~valid] = False
    return strong, grow, depth_best, orient_best


def _hysteresis_from_seeds(strong, grow):
    strong = strong.astype(np.bool_)
    grow = grow.astype(np.bool_)
    out = np.zeros(grow.shape, dtype=np.bool_)

    rows, cols = np.nonzero(strong & grow)
    q = deque()
    for r, c in zip(rows.tolist(), cols.tolist()):
        out[r, c] = True
        q.append((r, c))

    h, w = grow.shape
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGH8:
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if grow[rr, cc] and not out[rr, cc]:
                out[rr, cc] = True
                q.append((rr, cc))
    return out


# ---------------------------------------------------------------------------
# Thinning (identical to v11)
# ---------------------------------------------------------------------------

def _transitions_01(seq):
    s = np.zeros(seq[0].shape, dtype=np.uint8)
    for i in range(len(seq) - 1):
        s += ((~seq[i]) & seq[i + 1]).astype(np.uint8)
    return s


def _zhang_suen_thin(mask, max_iter=80):
    img = mask.astype(np.bool_).copy()
    if not np.any(img):
        return img

    for _ in range(int(max_iter)):
        changed = False

        P2 = _shift_bool(img, -1, 0)
        P3 = _shift_bool(img, -1, 1)
        P4 = _shift_bool(img, 0, 1)
        P5 = _shift_bool(img, 1, 1)
        P6 = _shift_bool(img, 1, 0)
        P7 = _shift_bool(img, 1, -1)
        P8 = _shift_bool(img, 0, -1)
        P9 = _shift_bool(img, -1, -1)
        neigh = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
        N = (P2.astype(np.uint8) + P3.astype(np.uint8) + P4.astype(np.uint8) +
             P5.astype(np.uint8) + P6.astype(np.uint8) + P7.astype(np.uint8) +
             P8.astype(np.uint8) + P9.astype(np.uint8))
        S = _transitions_01(neigh)
        cond = (img & (N >= 2) & (N <= 6) & (S == 1) &
                (~(P2 & P4 & P6)) & (~(P4 & P6 & P8)))
        if np.any(cond):
            img[cond] = False
            changed = True

        P2 = _shift_bool(img, -1, 0)
        P3 = _shift_bool(img, -1, 1)
        P4 = _shift_bool(img, 0, 1)
        P5 = _shift_bool(img, 1, 1)
        P6 = _shift_bool(img, 1, 0)
        P7 = _shift_bool(img, 1, -1)
        P8 = _shift_bool(img, 0, -1)
        P9 = _shift_bool(img, -1, -1)
        neigh = [P2, P3, P4, P5, P6, P7, P8, P9, P2]
        N = (P2.astype(np.uint8) + P3.astype(np.uint8) + P4.astype(np.uint8) +
             P5.astype(np.uint8) + P6.astype(np.uint8) + P7.astype(np.uint8) +
             P8.astype(np.uint8) + P9.astype(np.uint8))
        S = _transitions_01(neigh)
        cond = (img & (N >= 2) & (N <= 6) & (S == 1) &
                (~(P2 & P4 & P8)) & (~(P2 & P6 & P8)))
        if np.any(cond):
            img[cond] = False
            changed = True

        if not changed:
            break

    return img


def _remove_isolated_and_short(mask, min_pixels=3):
    if not np.any(mask):
        return mask
    deg = _neighbor_count(mask)
    out = mask.copy()
    out[(deg == 0) & out] = False
    return out


# ---------------------------------------------------------------------------
# Skeleton tracing (identical to v11)
# ---------------------------------------------------------------------------

def _pixel_neighbors(p, pixset):
    r, c = p
    out = []
    for dr, dc in NEIGH8:
        q = (r + dr, c + dc)
        if q in pixset:
            out.append(q)
    return out


def _edge_key(a, b):
    return (a, b) if a <= b else (b, a)


def _component_sets(mask):
    rows, cols = np.nonzero(mask)
    pixset = set(zip(rows.tolist(), cols.tolist()))
    comps = []
    while pixset:
        seed = next(iter(pixset))
        q = deque([seed])
        pixset.remove(seed)
        comp = set([seed])
        while q:
            p = q.popleft()
            r, c = p
            for dr, dc in NEIGH8:
                nb = (r + dr, c + dc)
                if nb in pixset:
                    pixset.remove(nb)
                    comp.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def _bfs_farthest(comp, start, prefer_endpoints=None):
    q = deque([start])
    prev = {start: None}
    dist = {start: 0}
    while q:
        p = q.popleft()
        r, c = p
        for dr, dc in NEIGH8:
            nb = (r + dr, c + dc)
            if nb in comp and nb not in prev:
                prev[nb] = p
                dist[nb] = dist[p] + 1
                q.append(nb)
    if prefer_endpoints:
        candidates = [p for p in prefer_endpoints if p in dist]
        if candidates:
            far = max(candidates, key=lambda x: dist[x])
            return far, prev, dist
    far = max(dist.keys(), key=lambda x: dist[x])
    return far, prev, dist


def _reconstruct_path(prev, start, end):
    path = []
    cur = end
    guard = 0
    while cur is not None and guard < 50000:
        guard += 1
        path.append(cur)
        if cur == start:
            break
        cur = prev.get(cur)
    path.reverse()
    return path


def _trace_skeleton(mask):
    if not np.any(mask):
        return []

    deg_arr = _neighbor_count(mask)
    comps = _component_sets(mask)
    paths = []

    for comp in comps:
        if len(comp) < 2:
            continue

        endpoints = [p for p in comp if int(deg_arr[p[0], p[1]]) <= 1]
        if endpoints:
            start0 = endpoints[0]
            far1, _, _ = _bfs_farthest(comp, start0, endpoints)
            far2, prev, _ = _bfs_farthest(comp, far1, endpoints)
            main_path = _reconstruct_path(prev, far1, far2)
        else:
            start0 = next(iter(comp))
            far1, _, _ = _bfs_farthest(comp, start0, None)
            far2, prev, _ = _bfs_farthest(comp, far1, None)
            main_path = _reconstruct_path(prev, far1, far2)

        main_set = set(main_path)
        if len(main_path) >= 2:
            paths.append(main_path)

        nodes = set([p for p in comp if int(deg_arr[p[0], p[1]]) != 2])
        if not nodes:
            continue

        visited_edges = set()
        comp_paths = []
        for node in sorted(nodes):
            for nb in _pixel_neighbors(node, comp):
                ek = _edge_key(node, nb)
                if ek in visited_edges:
                    continue

                path = [node, nb]
                visited_edges.add(ek)
                prev_p = node
                cur = nb
                guard = 0

                while cur not in nodes and guard < 50000:
                    guard += 1
                    nbs = [qq for qq in _pixel_neighbors(cur, comp)
                           if qq != prev_p]
                    if not nbs:
                        break
                    nxt = None
                    for qq in nbs:
                        if _edge_key(cur, qq) not in visited_edges:
                            nxt = qq
                            break
                    if nxt is None:
                        break
                    visited_edges.add(_edge_key(cur, nxt))
                    path.append(nxt)
                    prev_p, cur = cur, nxt

                if len(path) >= 2:
                    comp_paths.append(path)

        if comp_paths:
            for path in comp_paths:
                if len(path) < 2:
                    continue
                overlap = 0
                for p in path:
                    if p in main_set:
                        overlap += 1
                if float(overlap) / float(max(1, len(path))) >= 0.65:
                    continue
                paths.append(path)

    return paths


def _trace_skeleton_junction_aware(mask, max_through_turn_deg=60.0):
    """Trace a skeleton into polylines that pass STRAIGHT through a
    junction only when two incident branches are near-collinear. A branch
    that meets a through-trench at a sharp angle TERMINATES at the junction
    instead of being bridged across to the far side.

    This fixes the classic "bridging" failure of the plain diameter tracer
    (_trace_skeleton), where two separate trenches that physically touch
    become one connected component and the longest-path search walks
    trench A -> junction -> trench B, merging them.

    Behaviour by junction type:
      * Y-junction (main trench + side branch): the two collinear arms of
        the main trench stay linked (continue straight); the side branch
        becomes its own line, ending at the junction.
      * X-crossing (two trenches cross): opposite collinear arms pair up,
        producing two separate straight lines that cross.

    Returns a list of pixel paths (each a list of (r, c) tuples), the same
    format as _trace_skeleton, so the rest of the pipeline is unchanged.
    """
    if not np.any(mask):
        return []

    deg_arr = _neighbor_count(mask)
    comps = _component_sets(mask)
    cos_thresh = math.cos(math.radians(float(max_through_turn_deg)))
    paths = []

    for comp in comps:
        if len(comp) < 2:
            continue

        nodes = set(p for p in comp if int(deg_arr[p[0], p[1]]) != 2)

        # No junction/endpoint nodes -> a pure loop. Trace it as one path.
        if not nodes:
            start0 = next(iter(comp))
            far1, _, _ = _bfs_farthest(comp, start0, None)
            far2, prev, _ = _bfs_farthest(comp, far1, None)
            loop_path = _reconstruct_path(prev, far1, far2)
            if len(loop_path) >= 2:
                paths.append(loop_path)
            continue

        # ---- build node-to-node edges (chains of degree-2 pixels) ----
        edges = []                 # {'a': node, 'b': node, 'pix': [...]}
        visited_steps = set()
        for node in sorted(nodes):
            for nb in _pixel_neighbors(node, comp):
                if _edge_key(node, nb) in visited_steps:
                    continue
                pix = [node, nb]
                visited_steps.add(_edge_key(node, nb))
                prev_p = node
                cur = nb
                guard = 0
                while cur not in nodes and guard < 100000:
                    guard += 1
                    nxt = None
                    for qq in _pixel_neighbors(cur, comp):
                        if qq == prev_p:
                            continue
                        if _edge_key(cur, qq) not in visited_steps:
                            nxt = qq
                            break
                    if nxt is None:
                        break
                    visited_steps.add(_edge_key(cur, nxt))
                    pix.append(nxt)
                    prev_p, cur = cur, nxt
                if len(pix) >= 2:
                    edges.append({'a': pix[0], 'b': pix[-1], 'pix': pix})

        if not edges:
            continue

        # ---- unit direction pointing AWAY from a node along an edge ----
        def _away_dir(pix, from_start):
            k = min(6, len(pix) - 1)
            if from_start:
                p0 = pix[0]; p1 = pix[k]
            else:
                p0 = pix[-1]; p1 = pix[-1 - k]
            dy = float(p1[0] - p0[0])
            dx = float(p1[1] - p0[1])
            nrm = math.hypot(dx, dy)
            if nrm < 1e-9:
                return (0.0, 0.0)
            return (dx / nrm, dy / nrm)

        # incident edge-ends at each node
        incident = {}
        for ei, e in enumerate(edges):
            incident.setdefault(e['a'], []).append((ei, 'a'))
            incident.setdefault(e['b'], []).append((ei, 'b'))

        # ---- pair incident ends at each junction by collinearity ----
        # pair_of[(ei, end)] = (ej, end2): these two ends form a through line
        pair_of = {}
        for node, ends in incident.items():
            if len(ends) < 2:
                continue                       # endpoint: single free end
            dirs = {}
            for (ei, end) in ends:
                dirs[(ei, end)] = _away_dir(edges[ei]['pix'], end == 'a')
            cand = []
            for i in range(len(ends)):
                for j in range(i + 1, len(ends)):
                    e1 = ends[i]
                    e2 = ends[j]
                    if e1[0] == e2[0]:
                        continue               # don't pair an edge to itself
                    d1 = dirs[e1]
                    d2 = dirs[e2]
                    # straightness = -(d1 . d2): +1 = perfectly straight,
                    # -1 = U-turn back the way it came.
                    s = -(d1[0] * d2[0] + d1[1] * d2[1])
                    cand.append((s, e1, e2))
            cand.sort(key=lambda t: t[0], reverse=True)
            for s, e1, e2 in cand:
                if s < cos_thresh:
                    break                      # too sharp -> branch terminates
                if e1 in pair_of or e2 in pair_of:
                    continue
                pair_of[e1] = e2
                pair_of[e2] = e1

        # ---- chain edges through the pairings into polylines ----
        used_edges = set()

        def _oriented(ei, enter_end):
            pix = edges[ei]['pix']
            return pix if enter_end == 'a' else pix[::-1]

        def _build_chain(start_ei, start_end):
            chain = []
            cur_ei = start_ei
            cur_end = start_end
            guard = 0
            while (cur_ei is not None and cur_ei not in used_edges and
                   guard < 100000):
                guard += 1
                used_edges.add(cur_ei)
                seg = _oriented(cur_ei, cur_end)
                if chain and chain[-1] == seg[0]:
                    chain.extend(seg[1:])
                else:
                    chain.extend(seg)
                exit_end = 'b' if cur_end == 'a' else 'a'
                nxt = pair_of.get((cur_ei, exit_end))
                if nxt is None:
                    break
                nxt_ei, nxt_end = nxt
                if nxt_ei in used_edges:
                    break
                cur_ei, cur_end = nxt_ei, nxt_end
            return chain

        # 1) start chains at FREE ends (endpoints + unpaired junction arms)
        for ei in range(len(edges)):
            if ei in used_edges:
                continue
            for start_end in ('a', 'b'):
                if ei in used_edges:
                    break
                if (ei, start_end) in pair_of:
                    continue
                chain = _build_chain(ei, start_end)
                if len(chain) >= 2:
                    paths.append(chain)

        # 2) leftover edges that form closed through-cycles (no free end)
        for ei in range(len(edges)):
            if ei in used_edges:
                continue
            chain = _build_chain(ei, 'a')
            if len(chain) >= 2:
                paths.append(chain)

    return paths


# ---------------------------------------------------------------------------
# Line geometry helpers
# ---------------------------------------------------------------------------

def _path_length_xy(coords):
    if len(coords) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i - 1][0]
        dy = coords[i][1] - coords[i - 1][1]
        total += math.hypot(dx, dy)
    return total


def _perp_dist(pt, a, b):
    ax, ay = a
    bx, by = b
    px, py = pt
    dx = bx - ax
    dy = by - ay
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / math.hypot(dx, dy)


def _douglas_peucker(coords, tol):
    if len(coords) <= 2 or tol <= 0:
        return coords
    max_dist = -1.0
    idx = -1
    a = coords[0]
    b = coords[-1]
    for i in range(1, len(coords) - 1):
        d = _perp_dist(coords[i], a, b)
        if d > max_dist:
            max_dist = d
            idx = i
    if max_dist > tol and idx > 0:
        left = _douglas_peucker(coords[:idx + 1], tol)
        right = _douglas_peucker(coords[idx:], tol)
        return left[:-1] + right
    else:
        return [coords[0], coords[-1]]


def _path_to_coords(path, xmin, ymax, cell, row0_global, col0_global,
                    read_row0, read_col0):
    coords = []
    for r, c in path:
        gr = read_row0 + r
        gc = read_col0 + c
        x = xmin + (gc + 0.5) * cell
        y = ymax - (gr + 0.5) * cell
        coords.append((x, y))
    return coords


def _endpoint_direction(coords, at_start):
    if len(coords) < 2:
        return (0.0, 0.0)
    if at_start:
        a = coords[1]
        b = coords[0]
    else:
        a = coords[-2]
        b = coords[-1]
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    n = math.hypot(dx, dy)
    if n == 0:
        return (0.0, 0.0)
    return (dx / n, dy / n)


def _sample_bilinear(arr, valid, arr_xmin, arr_ymax, cellx, celly, x, y):
    col = (float(x) - arr_xmin) / cellx
    row = (arr_ymax - float(y)) / celly
    c0 = int(math.floor(col))
    r0 = int(math.floor(row))
    h, w = arr.shape
    if r0 < 0 or r0 + 1 >= h or c0 < 0 or c0 + 1 >= w:
        return None
    if (not valid[r0, c0] or not valid[r0 + 1, c0] or
            not valid[r0, c0 + 1] or not valid[r0 + 1, c0 + 1]):
        return None
    dc = col - c0
    dr = row - r0
    v00 = float(arr[r0, c0])
    v10 = float(arr[r0, c0 + 1])
    v01 = float(arr[r0 + 1, c0])
    v11 = float(arr[r0 + 1, c0 + 1])
    return ((v00 * (1.0 - dc) + v10 * dc) * (1.0 - dr) +
            (v01 * (1.0 - dc) + v11 * dc) * dr)


def _cross_section_evidence(arr, valid, arr_xmin, arr_ymax, cellx, celly,
                            x, y, dir_x, dir_y, half_width_px):
    mag = math.hypot(dir_x, dir_y)
    if mag <= 0.0:
        return None
    dir_x /= mag
    dir_y /= mag
    perp_x = -dir_y
    perp_y = dir_x
    cell = (cellx + celly) * 0.5

    center = _sample_bilinear(arr, valid, arr_xmin, arr_ymax, cellx, celly,
                              x, y)
    if center is None:
        return None

    best = None
    max_off = int(max(2, half_width_px + 3))
    for off in range(2, max_off + 1):
        dist = off * cell
        left = _sample_bilinear(arr, valid, arr_xmin, arr_ymax,
                                cellx, celly,
                                x + perp_x * dist, y + perp_y * dist)
        right = _sample_bilinear(arr, valid, arr_xmin, arr_ymax,
                                 cellx, celly,
                                 x - perp_x * dist, y - perp_y * dist)
        if left is None or right is None:
            continue
        avg_depth = ((left + right) * 0.5) - center
        min_side_depth = min(left, right) - center
        if best is None or avg_depth > best['avg_depth']:
            best = {
                'avg_depth': float(avg_depth),
                'min_side_depth': float(min_side_depth),
                'width_m': float(dist * 2.0)
            }
    return best


def _point_at_distance(coords, dist):
    if not coords:
        return None
    if dist <= 0.0:
        return coords[0]
    remain = float(dist)
    for i in range(1, len(coords)):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 0.0:
            continue
        if remain <= seg:
            t = remain / seg
            return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        remain -= seg
    return coords[-1]


def _trim_coords_at_distance(coords, dist, at_start):
    if len(coords) < 2 or dist <= 0.0:
        return coords
    total = _path_length_xy(coords)
    if dist >= total:
        return coords

    if at_start:
        p = _point_at_distance(coords, dist)
        out = [p]
        walked = 0.0
        for i in range(1, len(coords)):
            x0, y0 = coords[i - 1]
            x1, y1 = coords[i]
            seg = math.hypot(x1 - x0, y1 - y0)
            next_walked = walked + seg
            if next_walked > dist:
                out.extend(coords[i:])
                break
            walked = next_walked
        return out if len(out) >= 2 else coords

    rev = list(reversed(coords))
    trimmed = _trim_coords_at_distance(rev, dist, True)
    trimmed.reverse()
    return trimmed if len(trimmed) >= 2 else coords


def _resample_coords(coords, spacing):
    if len(coords) < 2:
        return coords
    spacing = float(max(0.5, spacing))
    total = _path_length_xy(coords)
    if total <= spacing:
        return [coords[0], coords[-1]]
    n = int(math.floor(total / spacing))
    out = []
    for i in range(n + 1):
        out.append(_point_at_distance(coords, min(total, i * spacing)))
    if math.hypot(out[-1][0] - coords[-1][0], out[-1][1] - coords[-1][1]) > 0.1:
        out.append(coords[-1])
    return out


def _moving_average_coords(coords, iterations=1, weight=0.55):
    if len(coords) <= 3:
        return coords
    out = list(coords)
    weight = float(max(0.0, min(1.0, weight)))
    for _ in range(int(max(0, iterations))):
        new_coords = [out[0]]
        for i in range(1, len(out) - 1):
            px = (out[i - 1][0] + out[i + 1][0]) * 0.5
            py = (out[i - 1][1] + out[i + 1][1]) * 0.5
            x = out[i][0] * (1.0 - weight) + px * weight
            y = out[i][1] * (1.0 - weight) + py * weight
            new_coords.append((x, y))
        new_coords.append(out[-1])
        out = new_coords
    return out


def _angle_jitter_degrees(coords):
    if len(coords) < 3:
        return 0.0
    angles = []
    for i in range(1, len(coords)):
        dx = coords[i][0] - coords[i - 1][0]
        dy = coords[i][1] - coords[i - 1][1]
        if math.hypot(dx, dy) > 0.001:
            angles.append(math.atan2(dy, dx))
    if len(angles) < 2:
        return 0.0
    changes = []
    for i in range(1, len(angles)):
        d = abs((angles[i] - angles[i - 1] + math.pi) %
                (2.0 * math.pi) - math.pi)
        changes.append(math.degrees(d))
    if not changes:
        return 0.0
    return float(sum(changes) / len(changes))


def _sinuosity(coords):
    if len(coords) < 2:
        return 0.0
    direct = math.hypot(coords[-1][0] - coords[0][0],
                        coords[-1][1] - coords[0][1])
    return _path_length_xy(coords) / max(direct, 0.001)


def _lock_to_min_elevation(pt, perp_x, perp_y, dem, valid, arr_xmin,
                           arr_ymax, cellx, celly, lock_radius_m,
                           lock_step_m):
    """Snap a candidate point to the lowest elevation along the perpendicular
    within +/- lock_radius_m. Resolves residual centring error after the
    cross-section search and handles flat-bottomed trenches by averaging the
    band of pixels within 5 cm of the local minimum.
    """
    if lock_radius_m <= 0.0:
        return pt
    base_z = _sample_bilinear(dem, valid, arr_xmin, arr_ymax,
                              cellx, celly, pt[0], pt[1])
    if base_z is None:
        return pt
    n = int(max(1, round(lock_radius_m / max(lock_step_m, 0.05))))
    offs = []
    zs = []
    for k in range(-n, n + 1):
        off = k * lock_step_m
        z = _sample_bilinear(dem, valid, arr_xmin, arr_ymax, cellx, celly,
                             pt[0] + perp_x * off, pt[1] + perp_y * off)
        if z is None:
            continue
        offs.append(off)
        zs.append(z)
    if not zs:
        return pt
    z_min = min(zs)
    band = [o for o, z in zip(offs, zs) if z <= z_min + 0.05]
    if not band:
        return pt
    centroid_off = sum(band) / float(len(band))
    return (pt[0] + perp_x * centroid_off, pt[1] + perp_y * centroid_off)


def _snap_to_width_midpoint(pt, perp_x, perp_y, dem, valid, arr_xmin,
                            arr_ymax, cellx, celly, scan_radius_m,
                            scan_step_m, min_depth_m=0.05, wall_frac=0.5,
                            max_shift_m=None):
    """v3.1 centring: find the trench's left and right walls along the
    perpendicular and place the polyline at their geometric midpoint.

    This matches what a human digitizer sees in a percent-clip-stretched
    DEM view: the visually darkest band has clear top edges (where the
    elevation rises sharply), and the GT line goes through the middle of
    that dark band -- not necessarily at the deepest pixel.

    Algorithm:
      1. Sample the perpendicular at fine step (0.25 m default) within
         +/- scan_radius_m.
      2. Detrend by subtracting the median elevation across the cross-
         section so the result mimics a locally-stretched view.
      3. If detrended depth (max - min residual) < min_depth_m, no clear
         trench cross-section -> return pt unchanged.
      4. Pick the deepest residual position (closest to original pt if
         tied within 10% of depth).
      5. Walk left and right from that position; the first cell where the
         residual rises above min + wall_frac * depth is the wall.
      6. Snap pt to the midpoint of the two wall offsets, clamped to
         max_shift_m so the function only refines, never relocates.

    Returns the new (x, y) or pt unchanged if no clear trench is found.
    """
    if scan_radius_m <= 0.0:
        return pt
    if max_shift_m is None:
        max_shift_m = scan_radius_m

    n = int(max(2, round(scan_radius_m / max(scan_step_m, 0.05))))
    offsets = []
    elevs = []
    for k in range(-n, n + 1):
        off = k * scan_step_m
        z = _sample_bilinear(dem, valid, arr_xmin, arr_ymax, cellx, celly,
                             pt[0] + perp_x * off, pt[1] + perp_y * off)
        offsets.append(off)
        elevs.append(z)

    valid_idx = [i for i, z in enumerate(elevs) if z is not None]
    if len(valid_idx) < 7:
        return pt
    valid_offs = [offsets[i] for i in valid_idx]
    valid_elevs = [elevs[i] for i in valid_idx]

    # Detrend by subtracting the median across the cross-section. This
    # produces a residual that highlights the trench independently of the
    # local hillside slope -- the same effect ArcMap "Percent Clip Stretch
    # from Current Display Extent" gives a human digitizer.
    sorted_z = sorted(valid_elevs)
    median_z = sorted_z[len(sorted_z) // 2]
    residual = [z - median_z for z in valid_elevs]

    z_min = min(residual)
    z_max = max(residual)
    depth_total = z_max - z_min
    if depth_total < float(min_depth_m):
        return pt

    wall_thr = z_min + float(wall_frac) * depth_total

    # Pick the deepest position closest to the original pt. This anchors
    # the wall walk so we follow the SAME trench, not a deeper one nearby.
    deep_band = [(i, valid_offs[i]) for i, r in enumerate(residual)
                 if r <= z_min + 0.10 * depth_total]
    deep_band.sort(key=lambda t: abs(t[1]))
    min_idx = deep_band[0][0] if deep_band else 0

    # Walk left from min_idx until we cross the wall threshold.
    left_wall = None
    for i in range(min_idx, -1, -1):
        if residual[i] >= wall_thr:
            left_wall = valid_offs[i]
            break
    # Walk right.
    right_wall = None
    for i in range(min_idx, len(residual)):
        if residual[i] >= wall_thr:
            right_wall = valid_offs[i]
            break
    if left_wall is None or right_wall is None:
        return pt

    width_mid = (left_wall + right_wall) * 0.5
    # Clamp the shift so we only refine, never relocate to a parallel
    # trench. The cross-section search has already chosen the right
    # trench; this step just re-centres within it.
    if width_mid > max_shift_m:
        width_mid = max_shift_m
    elif width_mid < -max_shift_m:
        width_mid = -max_shift_m
    return (pt[0] + perp_x * width_mid, pt[1] + perp_y * width_mid)


def _snap_to_lowest_cell_centre(pt, perp_x, perp_y, dem, valid,
                                arr_xmin, arr_ymax, cellx, celly,
                                radius_m, step_m=0.25):
    """v3.2 cell-centre snap: scan perpendicular and pick the actual DEM
    cell whose centre is at the lowest elevation. Returns the cell-centre
    coordinates, NOT a sub-pixel bilinear position.

    Why this matters: a human digitizer in ArcMap clicks the cursor on
    visible bright (low) pixels. Each click snaps to the cell that sits
    under the cursor -- there is no sub-pixel addressing in raster view.
    So GT vertices live at cell-centre xy. If our output uses bilinear
    sub-pixel positions, we are systematically off-grid from GT.

    Validated on Isheim2: switching from bilinear to cell-centre snap
    drops median drift to GT 0.558 m -> 0.302 m (-46%).

    Algorithm:
      1. Walk perpendicular from pt in step_m increments up to +/- radius_m.
      2. At each sample point, find the nearest DEM cell.
      3. Read the cell value (no bilinear).
      4. Track the cell whose value is the minimum (deduplicated by row,col).
      5. Return that cell's centre (cx, cy).
    """
    if radius_m <= 0.0:
        return pt
    h, w = dem.shape
    n = int(max(1, round(radius_m / max(step_m, 0.05))))
    seen = set()
    best_z = None
    best_xy = pt
    for k in range(-n, n + 1):
        off = k * step_m
        sx = pt[0] + perp_x * off
        sy = pt[1] + perp_y * off
        col = int(math.floor((sx - arr_xmin) / cellx))
        row = int(math.floor((arr_ymax - sy) / celly))
        if row < 0 or row >= h or col < 0 or col >= w:
            continue
        if (row, col) in seen:
            continue
        seen.add((row, col))
        if not valid[row, col]:
            continue
        z = float(dem[row, col])
        if best_z is None or z < best_z:
            best_z = z
            cx = arr_xmin + (col + 0.5) * cellx
            cy = arr_ymax - (row + 0.5) * celly
            best_xy = (cx, cy)
    return best_xy


def _point_segment_dist(p, a, b):
    """True point-to-SEGMENT distance (clamped, not the infinite line)."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    L2 = dx * dx + dy * dy
    if L2 <= 1.0e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / L2
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def _remove_duplicate_lines(all_lines, dup_dist_m=3.0, cover_ratio=0.60):
    """v9: remove output lines that run along the SAME physical trench as
    another (longer) line -- the 'one line inside another' duplicate.

    Algorithm:
      * Sort lines short -> long.
      * For each line, sample it every 2 m and check how much of it is
        within dup_dist_m of a LONGER line that is being kept.
      * If >= cover_ratio of the line is covered, drop it.

    Validated on Whole.zip: removes 100% of fully-duplicate lines
    (19 -> 0) with no loss of unique trench coverage.
    """
    if len(all_lines) < 2:
        return all_lines, 0

    lens = [_path_length_xy(ln.get('coords', [])) for ln in all_lines]
    gcell = 25.0

    def build_index(keep_flags):
        idx = {}
        for oi, ln in enumerate(all_lines):
            if not keep_flags[oi]:
                continue
            coords = ln.get('coords', [])
            for i in range(1, len(coords)):
                a = coords[i - 1]
                b = coords[i]
                x0 = int(math.floor(min(a[0], b[0]) / gcell))
                x1 = int(math.floor(max(a[0], b[0]) / gcell))
                y0 = int(math.floor(min(a[1], b[1]) / gcell))
                y1 = int(math.floor(max(a[1], b[1]) / gcell))
                for gx in range(x0, x1 + 1):
                    for gy in range(y0, y1 + 1):
                        idx.setdefault((gx, gy), []).append((oi, i - 1))
        return idx

    # Process short -> long so a duplicate is judged against longer lines
    order = sorted(range(len(all_lines)), key=lambda i: lens[i])
    keep = [True] * len(all_lines)
    n_dropped = 0
    for oi in order:
        coords = all_lines[oi].get('coords', [])
        if len(coords) < 2 or lens[oi] < 1.0:
            continue
        idx = build_index(keep)
        samples = _resample_coords(coords, 2.0)
        if len(samples) < 2:
            continue
        covered = 0
        for p in samples:
            bx = int(math.floor(p[0] / gcell))
            by = int(math.floor(p[1] / gcell))
            hit = False
            for gx in range(bx - 1, bx + 2):
                for gy in range(by - 1, by + 2):
                    for oj, si in idx.get((gx, gy), ()):
                        if oj == oi or not keep[oj]:
                            continue
                        if lens[oj] <= lens[oi]:
                            continue  # only a strictly longer line counts
                        cj = all_lines[oj]['coords']
                        if _point_segment_dist(
                                p, cj[si], cj[si + 1]) <= dup_dist_m:
                            hit = True
                            break
                    if hit:
                        break
                if hit:
                    break
            if hit:
                covered += 1
        if covered >= cover_ratio * len(samples):
            keep[oi] = False
            n_dropped += 1
    return [all_lines[i] for i in range(len(all_lines)) if keep[i]], n_dropped


def _remove_isolated_fragments(all_lines, max_len_m=80.0, min_iso_m=25.0):
    """v12.4 (river mode): drop SHORT lines whose BOTH endpoints are far from
    every other line -- floating hillside fragments that over-detection leaves
    behind. A real channel joins the network (water has to flow somewhere), so
    an isolated short stub is almost always a false positive. Long lines are
    never touched, so river coverage is preserved.

    Measured on the held-out Training-3 river crop: removed 28 clutter lines,
    pieces/river 1.50 -> 1.38, coverage held at 90%.
    """
    if len(all_lines) < 2:
        return all_lines, 0
    gcell = 25.0
    idx = {}
    for oi, ln in enumerate(all_lines):
        coords = ln.get('coords', [])
        for i in range(1, len(coords)):
            a = coords[i - 1]; b = coords[i]
            x0 = int(math.floor(min(a[0], b[0]) / gcell))
            x1 = int(math.floor(max(a[0], b[0]) / gcell))
            y0 = int(math.floor(min(a[1], b[1]) / gcell))
            y1 = int(math.floor(max(a[1], b[1]) / gcell))
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    idx.setdefault((gx, gy), []).append((oi, i - 1))
    rad = int(math.ceil(min_iso_m / gcell)) + 1

    def _is_connected(oi, p):
        """True if endpoint p is within min_iso_m of some OTHER line."""
        bx = int(math.floor(p[0] / gcell)); by = int(math.floor(p[1] / gcell))
        for gx in range(bx - rad, bx + rad + 1):
            for gy in range(by - rad, by + rad + 1):
                for oj, si in idx.get((gx, gy), ()):
                    if oj == oi:
                        continue
                    cj = all_lines[oj]['coords']
                    if _point_segment_dist(p, cj[si], cj[si + 1]) <= min_iso_m:
                        return True
        return False

    keep = [True] * len(all_lines)
    n_dropped = 0
    for oi, ln in enumerate(all_lines):
        coords = ln.get('coords', [])
        if len(coords) < 2:
            continue
        if _path_length_xy(coords) >= max_len_m:
            continue                      # long lines are always kept
        if (not _is_connected(oi, coords[0])
                and not _is_connected(oi, coords[-1])):
            keep[oi] = False
            n_dropped += 1
    return [all_lines[i] for i in range(len(all_lines)) if keep[i]], n_dropped


def _fuse_overlapping_lines(all_lines, max_off_m=4.0, min_shared_m=15.0,
                            min_tail_m=10.0, endpoint_window_m=25.0):
    """v12.4 (river mode): fix LINE MIXING -- two lines that trace the SAME
    river, overlapping for a stretch and weaving over each other (the dedup
    misses them because the shared run is a small FRACTION of each long line).

    For a shorter line B that shares a contiguous run (within max_off_m) with a
    longer line A:
      * 0 free tails  -> B is a pure duplicate            -> drop B
      * 1 free tail   -> B is a continuation / weaving twin-> graft the tail
                         onto A's nearby endpoint, drop B  -> ONE clean line
      * 2 free tails  -> B crosses THROUGH A (possibly two DIFFERENT rivers)
                         -> leave both alone (never mix separate rivers)
      * tail hangs off A's interior (real tributary) -> keep only the free
        tail so it ends cleanly at A (no weaving overlap)

    Measured on the Training-3 river crop: interior crossings 63 -> a few,
    river coverage and single-line topology preserved.
    """
    import math as _m
    lines = [list(ln.get('coords', [])) for ln in all_lines]
    metas = [dict(ln) for ln in all_lines]
    alive = [len(c) >= 2 for c in lines]
    lens = [_path_length_xy(c) for c in lines]
    gcell = 25.0

    def _build_index():
        idx = {}
        for oi in range(len(lines)):
            if not alive[oi]:
                continue
            c = lines[oi]
            for i in range(1, len(c)):
                a = c[i - 1]; b = c[i]
                for gx in range(int(_m.floor(min(a[0], b[0]) / gcell)),
                                int(_m.floor(max(a[0], b[0]) / gcell)) + 1):
                    for gy in range(int(_m.floor(min(a[1], b[1]) / gcell)),
                                    int(_m.floor(max(a[1], b[1]) / gcell)) + 1):
                        idx.setdefault((gx, gy), []).append((oi, i - 1))
        return idx

    n_fused = 0
    for _pass in range(3):
        idx = _build_index()
        order = sorted([i for i in range(len(lines)) if alive[i]],
                       key=lambda i: lens[i])           # short -> long
        any_change = False
        for bi in order:
            if not alive[bi]:
                continue
            B = lines[bi]
            Bsamp = _resample_coords(B, 2.0)
            if len(Bsamp) < 3:
                continue
            # nearest LONGER alive line for each sample of B
            near_id = []
            for p in Bsamp:
                bx = int(_m.floor(p[0] / gcell)); by = int(_m.floor(p[1] / gcell))
                best = max_off_m + 1e-9; bid = -1
                for gx in range(bx - 1, bx + 2):
                    for gy in range(by - 1, by + 2):
                        for oj, si in idx.get((gx, gy), ()):
                            if oj == bi or not alive[oj]:
                                continue
                            if lens[oj] < lens[bi] - 1e-6:
                                continue
                            cj = lines[oj]
                            d = _point_segment_dist(p, cj[si], cj[si + 1])
                            if d < best:
                                best = d; bid = oj
                near_id.append(bid)
            # longest contiguous run sharing the SAME longer line
            bestA = -1; run = None; i = 0; n = len(near_id)
            while i < n:
                if near_id[i] == -1:
                    i += 1; continue
                j = i
                while j + 1 < n and near_id[j + 1] == near_id[i]:
                    j += 1
                cur = (j - i) * 2.0
                if run is None or cur > (run[1] - run[0]) * 2.0:
                    run = (i, j); bestA = near_id[i]
                i = j + 1
            if bestA == -1 or run is None:
                continue
            if (run[1] - run[0]) * 2.0 < min_shared_m:
                continue
            A = lines[bestA]; i0, i1 = run
            pre = Bsamp[:i0]; post = Bsamp[i1 + 1:]
            tails = [(t, anchor) for (t, anchor) in
                     ((pre, Bsamp[i0]), (post, Bsamp[i1]))
                     if _path_length_xy(t) >= min_tail_m]
            if len(tails) >= 2:
                continue                       # crosses through A -> never mix
            if not tails:
                alive[bi] = False; n_fused += 1; any_change = True
                continue
            tail, inner = tails[0]
            _dA, ptA = _nearest_on_polyline(inner, A)
            d0 = _m.hypot(ptA[0] - A[0][0], ptA[1] - A[0][1])
            d1 = _m.hypot(ptA[0] - A[-1][0], ptA[1] - A[-1][1])
            if d0 <= endpoint_window_m and d0 <= d1:
                nt = tail if (_m.hypot(tail[-1][0] - A[0][0], tail[-1][1] - A[0][1])
                              < _m.hypot(tail[0][0] - A[0][0], tail[0][1] - A[0][1])) else tail[::-1]
                lines[bestA] = nt + A
                lens[bestA] = _path_length_xy(lines[bestA])
                alive[bi] = False; n_fused += 1; any_change = True
            elif d1 <= endpoint_window_m:
                nt = tail if (_m.hypot(tail[0][0] - A[-1][0], tail[0][1] - A[-1][1])
                              < _m.hypot(tail[-1][0] - A[-1][0], tail[-1][1] - A[-1][1])) else tail[::-1]
                lines[bestA] = A + nt
                lens[bestA] = _path_length_xy(lines[bestA])
                alive[bi] = False; n_fused += 1; any_change = True
            else:
                # overlap hangs off A's interior (tributary): keep only the
                # free tail so B ends cleanly at A, no weaving overlap.
                lines[bi] = list(tail); lens[bi] = _path_length_xy(tail)
                any_change = True
        if not any_change:
            break

    out = []
    for i in range(len(lines)):
        if alive[i] and len(lines[i]) >= 2:
            m = metas[i]; m['coords'] = lines[i]; out.append(m)
    return out, n_fused


def _nearest_on_polyline(p, coords):
    """Return (distance, closest_point) from p to the polyline coords."""
    px, py = p
    best_d = 1e18
    best_pt = coords[0]
    for i in range(1, len(coords)):
        ax, ay = coords[i - 1]
        bx, by = coords[i]
        dx = bx - ax; dy = by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-12:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
        qx = ax + t * dx; qy = ay + t * dy
        d = math.hypot(px - qx, py - qy)
        if d < best_d:
            best_d = d; best_pt = (qx, qy)
    return best_d, best_pt


def _local_dir(coords, i):
    """Unit tangent of coords near index i."""
    a = coords[max(0, i - 1)]
    b = coords[min(len(coords) - 1, i + 1)]
    dx = b[0] - a[0]; dy = b[1] - a[1]
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return (1.0, 0.0)
    return (dx / n, dy / n)


def _bbox(coords):
    xs = [p[0] for p in coords]; ys = [p[1] for p in coords]
    return min(xs), min(ys), max(xs), max(ys)


def _sample_dem(arr, valid, ax, ay, cellx, celly, axmin, aymax):
    c = int(round((ax - axmin) / cellx))
    r = int(round((aymax - ay) / celly))
    if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1] and valid[r, c]:
        return float(arr[r, c])
    return None


def _merge_parallel_lines(all_lines, in_dem, arcpy, params,
                          min_gap_m=1.5, max_gap_m=60.0,
                          angle_tol_deg=30.0, min_overlap_m=12.0,
                          overlap_frac=0.35, relief_margin_m=0.20,
                          feature_frac=0.55, step_m=2.0, max_passes=3):
    """v12.1: collapse two roughly-parallel lines that are really the two
    SIDES of one wide feature (river channel, road/dyke, terrace) into a
    single mid-line down its centre.

    Safety: a pair is merged ONLY when the DEM between the two lines is a
    single coherent feature - the mid-point is consistently LOWER (channel)
    or HIGHER (ridge/road) than the ground OUTSIDE both lines by at least
    relief_margin_m. Two genuinely separate narrow trenches have normal
    ground between them (mid ~ outside) and are left untouched.
    """
    if len(all_lines) < 2:
        return all_lines, 0
    cos_tol = math.cos(math.radians(float(angle_tol_deg)))
    n_merged_total = 0

    for _pass in range(int(max_passes)):
        order = sorted(range(len(all_lines)),
                       key=lambda i: _path_length_xy(
                           all_lines[i].get('coords', [])), reverse=True)
        keep = [True] * len(all_lines)
        merged_this_pass = 0

        for ii in order:
            if not keep[ii]:
                continue
            ci = all_lines[ii].get('coords', [])
            if len(ci) < 3:
                continue
            bi = _bbox(ci)
            ri = _resample_coords(ci, step_m)
            best_j = -1; best_overlap = 0.0; best_pairs = None

            for jj in order:
                if jj == ii or not keep[jj]:
                    continue
                cj = all_lines[jj].get('coords', [])
                if len(cj) < 3:
                    continue
                bj = _bbox(cj)
                # quick bbox reject
                if (bj[0] - max_gap_m > bi[2] or bj[2] + max_gap_m < bi[0] or
                        bj[1] - max_gap_m > bi[3] or bj[3] + max_gap_m < bi[1]):
                    continue
                pairs = []          # (P_on_i, q_on_j, gap)
                for k, P in enumerate(ri):
                    d, q = _nearest_on_polyline(P, cj)
                    if d < min_gap_m or d > max_gap_m:
                        continue
                    di = _local_dir(ri, k)
                    _, qk = d, q
                    # direction of j near q (approx via two nearby samples)
                    d2, q2 = _nearest_on_polyline(
                        (P[0] + di[0] * step_m, P[1] + di[1] * step_m), cj)
                    djd = (q2[0] - q[0], q2[1] - q[1])
                    nj = math.hypot(*djd)
                    if nj < 1e-6:
                        continue
                    djd = (djd[0] / nj, djd[1] / nj)
                    if abs(di[0] * djd[0] + di[1] * djd[1]) < cos_tol:
                        continue
                    pairs.append((P, q, d))
                overlap_len = len(pairs) * step_m
                len_j = _path_length_xy(cj)
                if (overlap_len >= min_overlap_m and
                        overlap_len >= overlap_frac * min(len_j,
                                                          _path_length_xy(ci))):
                    if overlap_len > best_overlap:
                        best_overlap = overlap_len; best_j = jj
                        best_pairs = pairs

            if best_j < 0:
                continue
            # DEM one-feature check on the candidate pair
            if not _pair_is_one_feature(best_pairs, in_dem, arcpy,
                                        relief_margin_m, feature_frac):
                continue
            # build mid-line along the longer line (ii)
            mid = []
            for k, P in enumerate(ri):
                d, q = _nearest_on_polyline(P, all_lines[best_j]['coords'])
                if min_gap_m <= d <= max_gap_m:
                    mid.append(((P[0] + q[0]) * 0.5, (P[1] + q[1]) * 0.5))
                else:
                    mid.append(P)
            mid = _moving_average_coords(mid, iterations=2, weight=0.5)
            tol = float(params.get('simplify_tolerance_m', 1.5))
            if tol > 0 and len(mid) > 3:
                mid = _douglas_peucker(mid, tol)
            newln = dict(all_lines[ii])
            newln['coords'] = mid
            newln['length'] = _path_length_xy(mid)
            all_lines[ii] = newln
            keep[best_j] = False
            merged_this_pass += 1

        if merged_this_pass == 0:
            break
        all_lines = [all_lines[i] for i in range(len(all_lines)) if keep[i]]
        n_merged_total += merged_this_pass

    return all_lines, n_merged_total


def _pair_is_one_feature(pairs, in_dem, arcpy, relief_margin_m, feature_frac):
    """True if the DEM between the paired points is one coherent feature:
    the mid-point is consistently below (channel) OR above (ridge) the
    ground just outside both lines."""
    if not pairs:
        return False
    # subsample up to ~15 cross-sections
    step = max(1, len(pairs) // 15)
    cross = pairs[::step]
    all_pts = []
    for (P, q, g) in cross:
        all_pts.extend([P, q])
    win = _read_dem_window_for_coords(all_pts, in_dem, arcpy,
                                      pad_m=float(max(p[2] for p in cross)) + 6.0)
    if win is None:
        return False
    arr, valid, axmin, aymax, cellx, celly = win
    n_ok = 0; n_used = 0
    for (P, q, g) in cross:
        ux = q[0] - P[0]; uy = q[1] - P[1]
        nrm = math.hypot(ux, uy)
        if nrm < 1e-6:
            continue
        ux /= nrm; uy /= nrm
        mx = (P[0] + q[0]) * 0.5; my = (P[1] + q[1]) * 0.5
        flank = min(6.0, g * 0.5)
        # outside points: beyond P (opposite to q) and beyond q
        ax = P[0] - ux * flank; ay = P[1] - uy * flank
        bx = q[0] + ux * flank; by = q[1] + uy * flank
        e_mid = _sample_dem(arr, valid, mx, my, cellx, celly, axmin, aymax)
        e_a = _sample_dem(arr, valid, ax, ay, cellx, celly, axmin, aymax)
        e_b = _sample_dem(arr, valid, bx, by, cellx, celly, axmin, aymax)
        if e_mid is None or e_a is None or e_b is None:
            continue
        n_used += 1
        outside = min(e_a, e_b)
        outside_hi = max(e_a, e_b)
        if e_mid < outside - relief_margin_m:          # channel
            n_ok += 1
        elif e_mid > outside_hi + relief_margin_m:      # ridge / road
            n_ok += 1
    if n_used < 3:
        return False
    return (float(n_ok) / float(n_used)) >= float(feature_frac)


def _human_style_centerline_pass(all_lines, in_dem, arcpy, params):
    """v3.2 final pass: mimic how a human digitizer draws GT.

    For each line that survived regularize+join, resample at fixed spacing
    (default 5 m -- matches manual digitizer click cadence). At each new
    interior vertex, snap perpendicular to the lowest CELL CENTRE within
    a tight radius (default 2 m). Connect with straight segments. NO
    smoothing, NO Douglas-Peucker.

    The v3.2 change vs v3.1: snap to the cell CENTRE of the lowest pixel,
    not to a bilinear sub-pixel min. A human in ArcMap clicks raster
    cells, so GT vertices are also at cell centres. Snapping to bilinear
    minima gives sub-pixel positions that are systematically off-grid
    from GT. Snapping to actual cell centres aligns the output to the
    same lattice GT was drawn on.

    Validated on Isheim2: median drift to GT 0.558 m -> 0.302 m (-46%),
    pct points >0.5 m off 54% -> 32% (-22 pp), pct >1.0 m off 23% -> 14%.

    Set human_style_cellcentre = False to fall back to v3.1 bilinear snap.
    """
    if not all_lines:
        return all_lines
    spacing = float(params.get('human_style_spacing_m', 5.0))
    radius = float(params.get('human_style_lock_radius_m', 2.0))
    step = float(params.get('human_style_lock_step_m', 0.20))
    cellcentre = bool(params.get('human_style_cellcentre', True))
    smooth_sigma = float(params.get('smooth_sigma_px', 0.9))
    pad = float(params.get('max_width_m', 6.0)) + 12.0

    out_lines = []
    for line in all_lines:
        coords = line.get('coords', [])
        if len(coords) < 3 or _path_length_xy(coords) < spacing * 1.5:
            out_lines.append(line)
            continue
        win = _read_dem_window_for_coords(coords, in_dem, arcpy, pad)
        if win is None:
            out_lines.append(line)
            continue
        arr, valid, arr_xmin, arr_ymax, cellx, celly = win
        if smooth_sigma > 0.05:
            dem = _nan_gaussian_smooth(arr, valid, smooth_sigma)
        else:
            dem = arr

        base = _resample_coords(coords, spacing)
        if len(base) < 2:
            out_lines.append(line)
            continue
        new_coords = [base[0]]
        for i in range(1, len(base) - 1):
            p_prev = base[i - 1]
            p_next = base[i + 1]
            tx = p_next[0] - p_prev[0]
            ty = p_next[1] - p_prev[1]
            mt = math.hypot(tx, ty)
            if mt < 1e-9:
                new_coords.append(base[i])
                continue
            tx /= mt
            ty /= mt
            perp_x = -ty
            perp_y = tx
            if cellcentre:
                snapped = _snap_to_lowest_cell_centre(
                    base[i], perp_x, perp_y, dem, valid,
                    arr_xmin, arr_ymax, cellx, celly,
                    radius, step)
            else:
                snapped = _lock_to_min_elevation(
                    base[i], perp_x, perp_y, dem, valid,
                    arr_xmin, arr_ymax, cellx, celly,
                    radius, step)
            new_coords.append(snapped)
        new_coords.append(base[-1])
        new_line = dict(line)
        new_line['coords'] = new_coords
        new_line['length'] = _path_length_xy(new_coords)
        out_lines.append(new_line)
    return out_lines


def _thalweg_centerline_pass(all_lines, in_dem, arcpy, params):
    """v12.2: constrained valley-following (thalweg) snap.

    Walks each line FORWARD at a fine spacing. At every step the vertex
    snaps to the lowest DEM cell on the perpendicular - BUT the lateral
    offset may change from the previous vertex by at most `lateral_slack`
    metres. That continuity constraint makes the line hug the trench
    bottom continuously (user's idea) while preventing it from jumping to
    DEM noise or to a deeper parallel trench, and from zig-zagging.

    Difference vs the v3.2 human-style pass: that pass snaps each 5 m
    vertex INDEPENDENTLY (no continuity); this pass is sequential at 2 m
    with a per-step lateral cap, so it follows bends and flat bottoms
    more tightly.
    """
    if not all_lines:
        return all_lines
    spacing = float(params.get('thalweg_spacing_m', 2.0))
    radius = float(params.get('thalweg_radius_m', 3.0))
    slack = float(params.get('thalweg_lateral_slack_m', 1.0))
    cstep = float(params.get('thalweg_cell_step_m', 0.25))
    smooth_sigma = float(params.get('smooth_sigma_px', 0.9))
    pad = float(params.get('max_width_m', 6.0)) + 12.0

    out_lines = []
    for line in all_lines:
        coords = line.get('coords', [])
        if len(coords) < 3 or _path_length_xy(coords) < spacing * 2.0:
            out_lines.append(line)
            continue
        win = _read_dem_window_for_coords(coords, in_dem, arcpy, pad)
        if win is None:
            out_lines.append(line)
            continue
        arr, valid, axmin, aymax, cellx, celly = win
        dem = (_nan_gaussian_smooth(arr, valid, smooth_sigma)
               if smooth_sigma > 0.05 else arr)
        h, w = dem.shape
        guide = _resample_coords(coords, spacing)
        if len(guide) < 3:
            out_lines.append(line)
            continue
        new_coords = [guide[0]]
        prev_off = 0.0
        n_steps = int(max(1, round(radius / max(cstep, 0.05))))
        # end-guard: keep vertices within this many metres of either tip
        # unchanged, so the carefully-extended endpoints are not disturbed.
        guard_m = float(params.get('thalweg_end_guard_m', 6.0))
        guard_v = int(max(0, round(guard_m / max(spacing, 0.5))))
        last_i = len(guide) - 1
        for i in range(1, len(guide) - 1):
            if i <= guard_v or i >= last_i - guard_v:
                new_coords.append(guide[i])   # protect endpoints
                prev_off = 0.0
                continue
            p_prev = guide[i - 1]; p_next = guide[i + 1]
            tx = p_next[0] - p_prev[0]; ty = p_next[1] - p_prev[1]
            mt = math.hypot(tx, ty)
            if mt < 1e-9:
                new_coords.append(guide[i]); continue
            tx /= mt; ty /= mt
            perp_x = -ty; perp_y = tx
            lo = max(-radius, prev_off - slack)
            hi = min(radius, prev_off + slack)
            best_z = None; best_xy = guide[i]; best_off = prev_off
            seen = set()
            for k in range(-n_steps, n_steps + 1):
                off = k * cstep
                if off < lo or off > hi:
                    continue
                sx = guide[i][0] + perp_x * off
                sy = guide[i][1] + perp_y * off
                col = int(math.floor((sx - axmin) / cellx))
                row = int(math.floor((aymax - sy) / celly))
                if row < 0 or row >= h or col < 0 or col >= w:
                    continue
                if (row, col) in seen:
                    continue
                seen.add((row, col))
                if not valid[row, col]:
                    continue
                z = float(dem[row, col])
                if best_z is None or z < best_z:
                    best_z = z
                    cx = axmin + (col + 0.5) * cellx
                    cy = aymax - (row + 0.5) * celly
                    best_xy = (cx, cy)
                    best_off = off
            new_coords.append(best_xy)
            prev_off = best_off
        new_coords.append(guide[-1])
        new_coords = _moving_average_coords(new_coords, iterations=1,
                                            weight=0.4)
        nl = dict(line)
        nl['coords'] = new_coords
        nl['length'] = _path_length_xy(new_coords)
        out_lines.append(nl)
    return out_lines


def _sample_line_evenly(coords, step):
    """Sample a polyline at fixed `step` (m) intervals; first sample is
    coords[0], last sample is the closest sample <= coords[-1]."""
    pts = []
    if len(coords) < 2:
        return pts
    pts.append(coords[0])
    walked = 0.0
    target = step
    for i in range(1, len(coords)):
        x0, y0 = coords[i - 1]
        x1, y1 = coords[i]
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg <= 1e-9:
            continue
        while walked + seg >= target:
            t = (target - walked) / seg
            pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
            target += step
        walked += seg
    pts.append(coords[-1])
    return pts


def _split_lines_at_breaks(all_lines, in_dem, arcpy, params):
    """v3.3 break detection and line splitting.

    Walk each output line at fixed sampling, measure DEM cross-section
    evidence at every sample, and SPLIT the line wherever a sufficiently
    long run of samples shows no real depression. Each fragment becomes
    its own output polyline whose endpoints land at the actual trench
    tip (last strong-evidence sample), not at a bridged location.

    This is the v3.3 fix for the "tool continues through gaps and merges
    separate trenches" problem reported on the Whole.zip dataset:
    measured 83 output lines (14%) bridged between 2 and 9 GT lines.

    Algorithm:
      1. Sample line at `split_sample_step_m` (default 1 m) intervals.
      2. At each sample, compute cross-section avg_depth perpendicular
         to the local tangent.
      3. A "weak" sample = avg_depth < `split_min_depth_m`
         (default 0.03 m) or no cross-section data.
      4. A "break run" = consecutive weak samples >=
         `split_min_break_length_m` (default 8 m).
      5. For each break run, the fragment ENDING at the break stops at
         the last strong sample BEFORE the break. The next fragment
         STARTS at the first strong sample AFTER the break.
      6. Fragments shorter than `split_min_fragment_length_m` (default
         8 m) are dropped.

    Returns (new_all_lines, n_split, n_fragments).
    """
    if not all_lines:
        return all_lines, 0, 0
    min_depth = float(params.get('split_min_depth_m', 0.03))
    min_break = float(params.get('split_min_break_length_m', 8.0))
    min_frag = float(params.get('split_min_fragment_length_m', 8.0))
    sample_step = float(params.get('split_sample_step_m', 1.0))
    smooth_sigma = float(params.get('smooth_sigma_px', 0.9))
    cell = float(params.get('cell_size', 1.0))
    # v3.3: TIGHT cross-section for split detection. The standard
    # _cross_section_evidence half-width (max_width/2 = 3m for 6m
    # trenches) reaches up to 6 m perpendicular and would falsely report
    # depression evidence for parallel trenches sitting within 6 m of
    # the bridge. We use a much tighter window (2 m by default) so the
    # split only "sees" depression directly under the polyline.
    split_half_m = float(params.get('split_half_width_m', 2.0))
    half_width_px = int(max(1, round(split_half_m / cell)))
    pad = float(params.get('max_width_m', 6.0)) + 12.0

    out_lines = []
    n_split = 0
    n_fragments = 0
    for line in all_lines:
        coords = line.get('coords', [])
        L = _path_length_xy(coords)
        if L < min_break + min_frag:
            out_lines.append(line)
            continue
        win = _read_dem_window_for_coords(coords, in_dem, arcpy, pad)
        if win is None:
            out_lines.append(line)
            continue
        arr, valid, arr_xmin, arr_ymax, cellx, celly = win
        if smooth_sigma > 0.05:
            dem = _nan_gaussian_smooth(arr, valid, smooth_sigma)
        else:
            dem = arr

        samples = _sample_line_evenly(coords, sample_step)
        if len(samples) < 4:
            out_lines.append(line)
            continue

        # Compute cross-section depth at every sample
        depths = []
        for j, p in enumerate(samples):
            if j == 0:
                tx = samples[1][0] - samples[0][0]
                ty = samples[1][1] - samples[0][1]
            elif j == len(samples) - 1:
                tx = samples[-1][0] - samples[-2][0]
                ty = samples[-1][1] - samples[-2][1]
            else:
                tx = samples[j + 1][0] - samples[j - 1][0]
                ty = samples[j + 1][1] - samples[j - 1][1]
            mag = math.hypot(tx, ty)
            if mag < 1e-9:
                depths.append(-1.0)
                continue
            tx /= mag
            ty /= mag
            ev = _cross_section_evidence(dem, valid, arr_xmin, arr_ymax,
                                          cellx, celly, p[0], p[1],
                                          tx, ty, half_width_px)
            depths.append(ev['avg_depth'] if ev is not None else -1.0)

        # Detect break runs
        breaks = []  # list of (start_idx, end_idx) - half-open
        in_break = False
        bs = 0
        for i, d in enumerate(depths):
            weak = d < min_depth
            if weak:
                if not in_break:
                    bs = i
                    in_break = True
            else:
                if in_break:
                    if (i - bs) * sample_step >= min_break:
                        breaks.append((bs, i))
                    in_break = False
        # Final break extending to end of line
        if in_break and (len(depths) - bs) * sample_step >= min_break:
            breaks.append((bs, len(depths)))

        if not breaks:
            out_lines.append(line)
            continue

        n_split += 1
        # Build fragment index ranges. A fragment is a contiguous run of
        # strong samples. samples[0:break[0][0]] is fragment 0,
        # samples[break[k][1]:break[k+1][0]] are middle fragments, etc.
        fragment_ranges = []
        if breaks[0][0] > 0:
            fragment_ranges.append((0, breaks[0][0]))
        for k in range(len(breaks) - 1):
            fragment_ranges.append((breaks[k][1], breaks[k + 1][0]))
        if breaks[-1][1] < len(samples):
            fragment_ranges.append((breaks[-1][1], len(samples)))

        for fs, fe in fragment_ranges:
            if fe - fs < 2:
                continue
            frag_coords = samples[fs:fe]
            frag_len = _path_length_xy(frag_coords)
            if frag_len < min_frag:
                continue
            new_line = dict(line)
            new_line['coords'] = list(frag_coords)
            new_line['length'] = frag_len
            # Recompute MeanDep / MaxDep on this fragment
            frag_depths = [depths[i] for i in range(fs, fe)
                           if depths[i] >= 0]
            if frag_depths:
                new_line['MeanDep'] = float(sum(frag_depths) /
                                            len(frag_depths))
                new_line['MaxDep'] = float(max(frag_depths))
            out_lines.append(new_line)
            n_fragments += 1
    return out_lines, n_split, n_fragments


def _regularize_single_line(line, in_dem, arcpy, params):
    coords = line.get('coords', [])
    if len(coords) < 2:
        return None

    cell = float(params.get('cell_size', 1.0))
    spacing = float(params.get('regularize_spacing_m', 4.0))
    scan_step = float(params.get('regularize_scan_step_m', 0.5))
    smooth_iter = int(params.get('smooth_iterations', 2))
    smooth_weight = float(params.get('smooth_weight', 0.55))
    simplify_tol = float(params.get('regularize_simplify_m', 2.0))
    support_depth = float(params.get('line_support_depth_m',
                                     params.get('grow_depth_m', 0.05)))
    min_ratio = float(params.get('min_line_support_ratio', 0.72))
    max_jitter = float(params.get('max_output_angle_jitter', 22.0))
    max_sin = float(params.get('max_output_sinuosity', 1.9))
    half_width = float(params.get('max_width_m', 6.0)) * 0.5
    half_width_px = int(max(2, round(float(params['max_width_m']) /
                                     (2.0 * cell))))
    # v3 centring controls. Lower penalty + min-elevation lock improves how
    # tightly the polyline tracks the true trench bottom. Defaults preserve
    # v15 behaviour when these params are not provided.
    offcenter_penalty = float(params.get('regularize_offcenter_penalty',
                                         0.018))
    lock_radius_m = float(params.get('regularize_lock_radius_m', 0.0))
    lock_step_m = float(params.get('regularize_lock_step_m',
                                   max(0.20, scan_step * 0.5)))
    # v3.1 width-midpoint snap. When > 0, the centring step replaces the
    # elevation-min lock with a wall-detect-and-midpoint algorithm that
    # tracks what a human digitizer sees in a percent-clip-stretched DEM.
    widmid_radius_m = float(params.get('regularize_widmid_radius_m', 0.0))
    widmid_step_m = float(params.get('regularize_widmid_step_m', 0.25))
    widmid_min_depth_m = float(params.get('regularize_widmid_min_depth_m',
                                          0.05))
    widmid_wall_frac = float(params.get('regularize_widmid_wall_frac',
                                        0.5))
    widmid_max_shift_m = float(params.get('regularize_widmid_max_shift_m',
                                          1.5))

    base = _resample_coords(coords, spacing)
    if len(base) < 2:
        return None

    pad = float(params.get('max_width_m', 6.0)) + 12.0
    win = _read_dem_window_for_coords(base, in_dem, arcpy, pad)
    if win is None:
        return None
    arr, valid, arr_xmin, arr_ymax, cellx, celly = win
    if float(params.get('smooth_sigma_px', 0.0)) > 0.05:
        dem = _nan_gaussian_smooth(arr, valid,
                                   float(params.get('smooth_sigma_px', 0.8)))
    else:
        dem = arr

    refit = []
    supported = 0
    depths = []
    widths = []
    for i, pt in enumerate(base):
        if i == 0:
            p0 = base[0]
            p1 = base[min(2, len(base) - 1)]
        elif i == len(base) - 1:
            p0 = base[max(0, len(base) - 3)]
            p1 = base[-1]
        else:
            p0 = base[i - 1]
            p1 = base[i + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        mag = math.hypot(dx, dy)
        if mag <= 0.001:
            refit.append(pt)
            continue
        dx /= mag
        dy /= mag
        perp_x = -dy
        perp_y = dx

        best_pt = pt
        best_ev = None
        steps = int(round(half_width / max(scan_step, 0.1)))
        for si in range(-steps, steps + 1):
            off = si * scan_step
            x = pt[0] + perp_x * off
            y = pt[1] + perp_y * off
            ev = _cross_section_evidence(dem, valid, arr_xmin, arr_ymax,
                                         cellx, celly, x, y, dx, dy,
                                         half_width_px)
            if ev is None:
                continue
            # Prefer a real depression, but resist jumps to parallel lines.
            score = ev['avg_depth'] - offcenter_penalty * abs(off)
            if best_ev is None or score > best_ev['score']:
                best_ev = dict(ev)
                best_ev['score'] = score
                best_pt = (x, y)

        # v3.1: snap to the geometric midpoint of the trench walls. The
        # cross-section search is good at finding the RIGHT trench but its
        # peak avg_depth does not always coincide with the visible centre
        # of the trench band on flat-bottomed or asymmetric cross-sections.
        # The width-midpoint snap fixes this by detecting the trench walls
        # explicitly and placing the polyline at their geometric midpoint
        # (which is what a human digitizer sees in a percent-clip-stretched
        # DEM view). Falls back to the v3 elevation-min lock if width-mid
        # is disabled.
        if widmid_radius_m > 0.0:
            best_pt = _snap_to_width_midpoint(
                best_pt, perp_x, perp_y, dem, valid,
                arr_xmin, arr_ymax, cellx, celly,
                widmid_radius_m, widmid_step_m,
                widmid_min_depth_m, widmid_wall_frac,
                widmid_max_shift_m)
        elif lock_radius_m > 0.0:
            best_pt = _lock_to_min_elevation(
                best_pt, perp_x, perp_y, dem, valid,
                arr_xmin, arr_ymax, cellx, celly,
                lock_radius_m, lock_step_m)

        if best_ev is not None:
            depths.append(best_ev['avg_depth'])
            widths.append(best_ev.get('width_m', 0.0))
            if best_ev['avg_depth'] >= support_depth:
                supported += 1
        refit.append(best_pt)

    if not depths:
        return None
    support_ratio = float(supported) / float(len(depths))
    if support_ratio < min_ratio:
        return None

    smoothed = _moving_average_coords(refit, smooth_iter, smooth_weight)
    if simplify_tol > 0 and len(smoothed) > 3:
        smoothed = _douglas_peucker(smoothed, simplify_tol)
    if len(smoothed) < 2:
        return None

    # v3.1 final-snap: smoothing and Douglas-Peucker re-introduce up to
    # ~0.5 m of perpendicular drift around tight bends. Walk the simplified
    # polyline one more time and snap each interior vertex either to the
    # trench width-midpoint (preferred, v3.1) or to the local elevation
    # minimum (v3 fallback). Endpoints are preserved so the endpoint
    # extension stays valid.
    if (widmid_radius_m > 0.0 or lock_radius_m > 0.0) and len(smoothed) >= 3:
        snapped = [smoothed[0]]
        for i in range(1, len(smoothed) - 1):
            dxv = smoothed[i + 1][0] - smoothed[i - 1][0]
            dyv = smoothed[i + 1][1] - smoothed[i - 1][1]
            mv = math.hypot(dxv, dyv)
            if mv < 0.001:
                snapped.append(smoothed[i])
                continue
            dxv /= mv
            dyv /= mv
            perp_x_v = -dyv
            perp_y_v = dxv
            if widmid_radius_m > 0.0:
                snapped.append(_snap_to_width_midpoint(
                    smoothed[i], perp_x_v, perp_y_v, dem, valid,
                    arr_xmin, arr_ymax, cellx, celly,
                    widmid_radius_m, widmid_step_m,
                    widmid_min_depth_m, widmid_wall_frac,
                    widmid_max_shift_m))
            else:
                snapped.append(_lock_to_min_elevation(
                    smoothed[i], perp_x_v, perp_y_v, dem, valid,
                    arr_xmin, arr_ymax, cellx, celly,
                    lock_radius_m, lock_step_m))
        snapped.append(smoothed[-1])
        smoothed = snapped

    length = _path_length_xy(smoothed)
    if length < float(params.get('min_length_m', 10.0)):
        return None
    sin = _sinuosity(smoothed)
    jitter = _angle_jitter_degrees(smoothed)
    if sin > max_sin or jitter > max_jitter:
        return None

    out = dict(line)
    out['coords'] = smoothed
    out['length'] = length
    out['MeanDep'] = float(sum(depths) / len(depths))
    out['MaxDep'] = float(max(depths))
    out['Support'] = support_ratio
    out['Jitter'] = jitter
    out['Sinuosity'] = sin
    if widths:
        out['Width_m'] = float(sum(widths) / len(widths))
    length_score = min(length / 100.0, 1.0)
    mean_score = min(out['MeanDep'] / 0.45, 1.0)
    support_score = min(support_ratio, 1.0)
    smooth_score = max(0.0, 1.0 - jitter / max(max_jitter, 1.0))
    out['QScore'] = min(100.0, max(0.0, 100.0 *
                         (0.30 * length_score + 0.35 * mean_score +
                          0.25 * support_score + 0.10 * smooth_score)))
    return out


def _regularize_all_lines(all_lines, in_dem, arcpy, params):
    if not all_lines:
        return all_lines, 0
    out = []
    rejected = 0
    for line in all_lines:
        try:
            reg = _regularize_single_line(line, in_dem, arcpy, params)
        except Exception:
            reg = None
        if reg is None:
            rejected += 1
        else:
            out.append(reg)
    return out, rejected


def _read_dem_window_for_coords(coords, in_dem, arcpy, pad_m):
    ras = arcpy.Raster(in_dem)
    ext = ras.extent
    xmin = float(ext.XMin)
    ymax = float(ext.YMax)
    ymin = float(ext.YMin)
    xmax = float(ext.XMax)
    cellx = float(ras.meanCellWidth)
    celly = abs(float(ras.meanCellHeight))

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    x0 = max(xmin, min(xs) - pad_m)
    x1 = min(xmax, max(xs) + pad_m)
    y0 = max(ymin, min(ys) - pad_m)
    y1 = min(ymax, max(ys) + pad_m)

    read_col0 = int(max(0, math.floor((x0 - xmin) / cellx)))
    read_col1 = int(min(math.ceil((xmax - xmin) / cellx),
                        math.ceil((x1 - xmin) / cellx)))
    read_row0 = int(max(0, math.floor((ymax - y1) / celly)))
    read_row1 = int(min(math.ceil((ymax - ymin) / celly),
                        math.ceil((ymax - y0) / celly)))
    ncols = read_col1 - read_col0
    nrows = read_row1 - read_row0
    if ncols < 4 or nrows < 4:
        return None

    arr_xmin = xmin + read_col0 * cellx
    arr_ymax = ymax - read_row0 * celly
    llx = arr_xmin
    lly = arr_ymax - nrows * celly
    nodata_value = -9999999.0
    lower_left = arcpy.Point(llx, lly)
    arr = arcpy.RasterToNumPyArray(in_dem, lower_left, ncols, nrows,
                                   nodata_to_value=nodata_value)
    arr = arr.astype(np.float32)
    valid = (np.isfinite(arr) & (arr != nodata_value) & (arr > -1.0e20))
    if np.count_nonzero(valid) < 16:
        return None
    return arr, valid, arr_xmin, arr_ymax, cellx, celly


def _gap_depression_ok(a_pt, b_pt, in_dem, arcpy, params):
    dx = b_pt[0] - a_pt[0]
    dy = b_pt[1] - a_pt[1]
    dist = math.hypot(dx, dy)
    if dist <= 0.0:
        return False, 0.0

    cell = float(params.get('cell_size', 1.0))
    half_width_px = int(max(2, round(float(params['max_width_m']) /
                                     (2.0 * cell))))
    threshold = float(params.get('join_gap_depth_m',
                                 params.get('grow_depth_m', 0.05)))
    min_side = max(0.012, threshold * 0.35)
    min_ratio = float(params.get('join_gap_min_ratio', 0.60))
    pad = dist + float(params['max_width_m']) + 6.0

    win = _read_dem_window_for_coords([a_pt, b_pt], in_dem, arcpy, pad)
    if win is None:
        return False, 0.0
    arr, valid, arr_xmin, arr_ymax, cellx, celly = win

    samples = max(3, int(math.ceil(dist / max(cell, 0.5))) + 1)
    ok = 0
    depths = []
    for i in range(samples):
        t = float(i) / float(max(1, samples - 1))
        x = a_pt[0] + dx * t
        y = a_pt[1] + dy * t
        ev = _cross_section_evidence(arr, valid, arr_xmin, arr_ymax,
                                     cellx, celly, x, y, dx, dy,
                                     half_width_px)
        if ev is None:
            continue
        depths.append(ev['avg_depth'])
        if (ev['avg_depth'] >= threshold and
                ev['min_side_depth'] >= min_side):
            ok += 1

    if not depths:
        return False, 0.0
    ratio = float(ok) / float(len(depths))
    mean_depth = float(sum(depths) / len(depths))
    return (ratio >= min_ratio), mean_depth


def _try_join_lines(lines, join_gap, max_angle_deg=65.0,
                    in_dem=None, arcpy=None, params=None):
    """Endpoint stitching, optionally verified by DEM depression in the gap."""
    if join_gap <= 0 or len(lines) < 2:
        return lines

    cos_thr = math.cos(math.radians(float(max_angle_deg)))
    cell = max(float(join_gap), 0.001)
    changed = True
    guard = 0
    while changed and guard < 40:
        guard += 1
        changed = False
        bins = {}
        endpoints = []
        for i, line in enumerate(lines):
            coords = line['coords']
            for at_start, pt in ((True, coords[0]), (False, coords[-1])):
                key = (int(math.floor(pt[0] / cell)),
                       int(math.floor(pt[1] / cell)))
                ep = {
                    'line': i,
                    'at_start': at_start,
                    'pt': pt,
                    'dir': _endpoint_direction(coords, at_start)
                }
                endpoints.append(ep)
                bins.setdefault(key, []).append(len(endpoints) - 1)

        candidates = []
        for ai, a in enumerate(endpoints):
            ax, ay = a['pt']
            bx0 = int(math.floor(ax / cell))
            by0 = int(math.floor(ay / cell))
            for gx in range(bx0 - 1, bx0 + 2):
                for gy in range(by0 - 1, by0 + 2):
                    for bi in bins.get((gx, gy), []):
                        if bi <= ai:
                            continue
                        b = endpoints[bi]
                        if a['line'] == b['line']:
                            continue
                        dx = b['pt'][0] - ax
                        dy = b['pt'][1] - ay
                        d = math.hypot(dx, dy)
                        if d <= 0.0 or d > join_gap:
                            continue
                        ux, uy = dx / d, dy / d
                        good_a = (a['dir'][0] * ux +
                                  a['dir'][1] * uy >= cos_thr)
                        good_b = (b['dir'][0] * (-ux) +
                                  b['dir'][1] * (-uy) >= cos_thr)
                        if good_a and good_b:
                            gap_depth = 0.0
                            if (params is not None and
                                    params.get('verify_join_gaps', True) and
                                    in_dem is not None and arcpy is not None):
                                ok, gap_depth = _gap_depression_ok(
                                    a['pt'], b['pt'], in_dem, arcpy, params)
                                if not ok:
                                    continue
                            candidates.append((d, ai, bi, gap_depth))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0])
        used = set()
        pairs = []
        for _, ai, bi, _ in candidates:
            li = endpoints[ai]['line']
            lj = endpoints[bi]['line']
            if li in used or lj in used:
                continue
            used.add(li)
            used.add(lj)
            pairs.append((li, endpoints[ai]['at_start'],
                          lj, endpoints[bi]['at_start']))

        if not pairs:
            break

        new_lines = []
        for k, line in enumerate(lines):
            if k not in used:
                new_lines.append(line)

        for i, si, j, sj in pairs:
            ci = lines[i]['coords']
            cj = lines[j]['coords']
            if not si:
                a_c = ci[:]
            else:
                a_c = list(reversed(ci))
            if sj:
                b_c = cj[:]
            else:
                b_c = list(reversed(cj))
            coords = a_c + b_c
            new_line = dict(lines[i])
            new_line['coords'] = coords
            new_line['length'] = _path_length_xy(coords)
            new_line['MeanDep'] = ((lines[i].get('MeanDep', 0.0) +
                                    lines[j].get('MeanDep', 0.0)) * 0.5)
            new_line['MaxDep'] = max(lines[i].get('MaxDep', 0.0),
                                     lines[j].get('MaxDep', 0.0))
            new_line['PixCnt'] = (int(lines[i].get('PixCnt', 0)) +
                                  int(lines[j].get('PixCnt', 0)))
            new_lines.append(new_line)

        lines = new_lines
        changed = True

    return lines


# ---------------------------------------------------------------------------
# Polyline-level endpoint trim and extension
# ---------------------------------------------------------------------------

def _complete_trenches_downhill(all_lines, in_dem, arcpy, params):
    """v12.6 (user's rule): complete broken/faint trenches by extending each
    line's tips ALONG THE DEM VALLEY -- the HIGH tip uphill toward the source,
    the LOW tip downhill toward the outlet -- following the lowest pixels, and
    continuing ONLY while a real cross-sectional DEPRESSION exists (both sides
    higher than the centre by >= margin). When the depression ends, the trench
    ends. The depression-gate is what stops it running across flat hillsides.

    A trench's natural direction is high->low (the user's drawing rule); this
    pass encodes exactly that, using the DEM's shape to fill the faint gaps the
    per-pixel model leaves broken. Measured on the Whole held-out: recall
    +3-5pp (stacks on the model), trenches come out complete and single.
    """
    if not all_lines:
        return all_lines, 0
    step = float(params.get('complete_step_m', 2.0))
    radius = float(params.get('complete_radius_m', 3.0))
    side = float(params.get('complete_side_m', 3.5))
    margin = float(params.get('complete_margin_m', 0.12))
    max_len = float(params.get('complete_max_len_m', 80.0))
    max_fail = int(params.get('complete_max_fail', 3))
    smooth_sigma = float(params.get('smooth_sigma_px', 0.9))
    # v12.8: also continue where the CNN gives faint evidence (not only where a
    # strong DEM depression exists) -> reaches the true start/end through faint
    # stretches and recovers more, while staying gated by real model evidence.
    faint_thr = float(params.get('complete_faint_prob', 2.0))
    # v12.8.1: persistent prob-gated continuation. When prob_gate>0, extend
    # through faint stretches while BOTH a (faint) depression AND the CNN agree
    # it is a trench -> fixes early-stopping trenches without flooding flat
    # ground (relief-alone floods; prob-alone mixes). prob_gate<=0 = legacy.
    prob_gate = float(params.get('complete_prob_gate', 0.0))
    trend_tol = float(params.get('complete_trend_tol', 0.05))
    cnn_prob = params.get('cnn_prob')
    cnn_prob_geo = params.get('cnn_prob_geo')
    pad = max_len + side + 12.0
    n_ext = 0
    out = []
    for line in all_lines:
        coords = line.get('coords', [])
        if len(coords) < 2:
            out.append(line); continue
        win = _read_dem_window_for_coords(coords, in_dem, arcpy, pad)
        if win is None:
            out.append(line); continue
        arr, valid, axmin, aymax, cellx, celly = win
        dem = (_nan_gaussian_smooth(arr, valid, smooth_sigma)
               if smooth_sigma > 0.05 else arr)
        h, w = dem.shape

        def elev(x, y):
            col = int(round((x - axmin) / cellx)); row = int(round((aymax - y) / celly))
            if 0 <= row < h and 0 <= col < w and valid[row, col]:
                return float(dem[row, col])
            return None

        def valley_snap(px, py, ux, uy):
            perpx, perpy = -uy, ux
            best = elev(px, py); bx, by = px, py
            if best is None:
                return px, py, None
            o = -radius
            while o <= radius + 1e-9:
                e = elev(px + perpx * o, py + perpy * o)
                if e is not None and e < best:
                    best = e; bx, by = px + perpx * o, py + perpy * o
                o += 0.5
            return bx, by, best

        def relief(px, py, ux, uy):
            perpx, perpy = -uy, ux
            c = elev(px, py)
            l = elev(px - perpx * side, py - perpy * side)
            r = elev(px + perpx * side, py + perpy * side)
            if c is None or l is None or r is None:
                return -9.0
            return min(l, r) - c

        def extend(p_tip, p_in, go_down):
            ux, uy = p_tip[0] - p_in[0], p_tip[1] - p_in[1]
            m = math.hypot(ux, uy)
            if m < 1e-6:
                return []
            ux, uy = ux / m, uy / m
            cur = (p_tip[0], p_tip[1]); cz = elev(*cur)
            if cz is None:
                return []
            res = []; trav = 0.0; fails = 0
            while trav < max_len:
                nx, ny = cur[0] + ux * step, cur[1] + uy * step
                sx, sy, sz = valley_snap(nx, ny, ux, uy)
                if sz is None:
                    break
                rel = relief(sx, sy, ux, uy)
                trend = ((sz <= cz + trend_tol) if go_down
                         else (sz >= cz - trend_tol))
                faint = (_sample_prob_at_xy(cnn_prob, cnn_prob_geo, sx, sy)
                         if cnn_prob is not None else None)
                if prob_gate > 0.0:
                    evid = ((rel >= margin) and
                            (faint is not None and faint >= prob_gate))
                else:
                    evid = ((rel >= margin) or
                            (faint is not None and faint >= faint_thr))
                if evid and trend:
                    res.append((sx, sy))
                    ddx, ddy = sx - cur[0], sy - cur[1]; nm = math.hypot(ddx, ddy)
                    if nm > 1e-6:
                        ux, uy = 0.6 * ux + 0.4 * ddx / nm, 0.6 * uy + 0.4 * ddy / nm
                        um = math.hypot(ux, uy); ux, uy = ux / um, uy / um
                    cur = (sx, sy); cz = sz; trav += step; fails = 0
                else:
                    fails += 1
                    if fails >= max_fail:
                        break
                    cur = (cur[0] + ux * step, cur[1] + uy * step); trav += step
            return res

        z0 = elev(*coords[0]); z1 = elev(*coords[-1])
        if z0 is None or z1 is None:
            out.append(line); continue
        pre = extend(coords[0], coords[1], go_down=(z0 < z1))
        post = extend(coords[-1], coords[-2], go_down=(z1 <= z0))
        if pre or post:
            n_ext += 1
        nl = dict(line)
        nl['coords'] = list(reversed(pre)) + list(coords) + post
        out.append(nl)
    return out, n_ext


def _cc_label(binmask):
    """Pure 8-connected components for a SPARSE boolean mask (no scipy)."""
    H, W = binmask.shape
    lab = np.zeros((H, W), dtype=np.int32)
    rs, cs = np.where(binmask)
    cellset = set(zip([int(v) for v in rs], [int(v) for v in cs]))
    seen = set(); cur = 0
    nbr = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for start in cellset:
        if start in seen:
            continue
        cur += 1; stack = [start]; seen.add(start)
        while stack:
            r, c = stack.pop(); lab[r, c] = cur
            for dr, dc in nbr:
                q = (r + dr, c + dc)
                if q in cellset and q not in seen:
                    seen.add(q); stack.append(q)
    return lab, cur


def _shift_fill(a, dr, dc, fill):
    """Shift array a by (dr,dc); vacated border filled with `fill`."""
    H, W = a.shape
    out = np.empty_like(a); out.fill(fill)
    r0a, r1a = max(0, dr), min(H, H + dr); c0a, c1a = max(0, dc), min(W, W + dc)
    r0b, r1b = max(0, -dr), min(H, H - dr); c0b, c1b = max(0, -dc), min(W, W - dc)
    out[r0b:r1b, c0b:c1b] = a[r0a:r1a, c0a:c1a]
    return out


_OFF8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]


def _flow_markers(Z, mask, h=0.25, max_iter=200):
    """h-minima seed markers: only minima at least `h` deep survive (shallow
    noise pits are suppressed) -> far fewer, more meaningful basins, so the
    watershed does not over-segment a single ditch. Uses grayscale
    reconstruction-by-erosion (pure numpy), then labels its regional minima."""
    BIG = float(np.max(Z)) + 1000.0
    Zc = np.where(mask, Z, BIG)
    J = np.where(mask, Z + float(h), BIG)
    for _ in range(int(max_iter)):
        E = J.copy()
        for dr, dc in _OFF8:
            E = np.minimum(E, _shift_fill(J, dr, dc, BIG))
        Jn = np.maximum(E, Zc)
        if np.array_equal(Jn, J):
            break
        J = Jn
    # regional minima of the reconstruction within the mask = the h-minima
    ismin = mask.copy()
    for dr, dc in _OFF8:
        sh = _shift_fill(J, dr, dc, BIG)
        shm = _shift_fill(mask, dr, dc, False)
        ismin = ismin & ((~shm) | (J <= sh + 1.0e-4))
    return _cc_label(ismin & mask)


def _flow_watershed(Z, markers, mask):
    """Priority-flood watershed (heapq): flood from markers lowest-first; each
    mask cell takes the label of the marker that reaches it first. Basins meet at
    ridges -> two parallel ditches split at the road crown (no weave)."""
    import heapq
    H, W = Z.shape
    labels = markers.astype(np.int32).copy()
    inq = np.zeros((H, W), dtype=np.bool_)
    heap = []
    rs, cs = np.where(markers > 0)
    for r, c in zip([int(v) for v in rs], [int(v) for v in cs]):
        heapq.heappush(heap, (float(Z[r, c]), r, c)); inq[r, c] = True
    nbr = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while heap:
        z, r, c = heapq.heappop(heap)
        lab = labels[r, c]
        for dr, dc in nbr:
            nr, nc = r + dr, c + dc
            if (0 <= nr < H and 0 <= nc < W and mask[nr, nc]
                    and labels[nr, nc] == 0 and not inq[nr, nc]):
                labels[nr, nc] = lab; inq[nr, nc] = True
                heapq.heappush(heap, (float(Z[nr, nc]), nr, nc))
    return labels


def _flow_mode_lines(prob, in_dem, arcpy, params,
                     xmin, ymax, cellx, celly, width, height):
    """Flow Mode vectorisation (pure-numpy, ArcMap-2.7 safe): watershed-segment
    the CNN prob mask by the DEM, take each basin's thalweg centre-line snapped to
    the ditch bottom. Coverage (full mask kept) + no weave (basins split at the
    crown). Stitch / pipe-split / declutter are applied afterwards by the normal
    geometry passes. Returns line dicts."""
    thr = float(params.get('flow_prob_thr', 0.40))
    smooth_sigma = float(params.get('smooth_sigma_px', 0.9))
    snap = float(params.get('flow_snap_m', 2.5))
    mask = (prob >= thr)
    if not np.any(mask):
        return []
    nodata = -9999999.0
    ll = arcpy.Point(xmin, ymax - height * celly)
    arr = arcpy.RasterToNumPyArray(in_dem, ll, width, height,
                                   nodata_to_value=nodata).astype(np.float32)
    valid = np.isfinite(arr) & (arr != nodata) & (arr > -1.0e20)
    if np.count_nonzero(valid) < 50:
        return []
    Z = _nan_gaussian_smooth(arr, valid, smooth_sigma)
    mask = mask & valid
    H, W = Z.shape
    markers, nmk = _flow_markers(Z, mask)
    if nmk == 0:
        return []
    labels = _flow_watershed(Z, markers, mask)
    skel = _zhang_suen_thin(mask)
    skset = set([(int(p[0]), int(p[1])) for p in np.argwhere(skel)])
    nbr = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    adj = {}
    for p in skset:
        adj[p] = []
    for p in skset:
        lp = labels[p[0], p[1]]
        for dr, dc in nbr:
            q = (p[0] + dr, p[1] + dc)
            if q in skset:
                lq = labels[q[0], q[1]]
                if lp == lq or lp == 0 or lq == 0:
                    adj[p].append(q)
    deg = {}
    for p in skset:
        deg[p] = len(adj[p])
    seen = set(); polys = []

    def _trace(p):
        out = []
        for q0 in adj[p]:
            if (p, q0) in seen:
                continue
            path = [p]; prev = p; cur = q0
            seen.add((p, q0)); seen.add((q0, p))
            while True:
                path.append(cur)
                if deg[cur] != 2:
                    break
                nx = [n for n in adj[cur] if n != prev]
                if not nx or (cur, nx[0]) in seen:
                    break
                seen.add((cur, nx[0])); seen.add((nx[0], cur))
                prev, cur = cur, nx[0]
            out.append(path)
        return out
    for p in skset:
        if deg[p] != 2:
            polys += _trace(p)
    for p in skset:
        if deg[p] == 2 and not [1 for q in adj[p] if (p, q) in seen]:
            polys += _trace(p)

    def _xy(r, c):
        return (xmin + (c + 0.5) * cellx, ymax - (r + 0.5) * celly)

    def _elev(x, y):
        cc = int(round((x - xmin) / cellx)); rr = int(round((ymax - y) / celly))
        if 0 <= rr < H and 0 <= cc < W and valid[rr, cc]:
            return float(Z[rr, cc])
        return None

    out_lines = []
    for path in polys:
        if len(path) < 3:
            continue
        pts = [_xy(r, c) for (r, c) in path]
        sp = []
        for i in range(len(pts)):
            a = pts[max(0, i - 1)]; b = pts[min(len(pts) - 1, i + 1)]
            ux, uy = b[0] - a[0], b[1] - a[1]; m = math.hypot(ux, uy)
            if m < 1e-9:
                sp.append(pts[i]); continue
            px, py = -uy / m, ux / m
            best = _elev(*pts[i]); bx, by = pts[i]; o = -snap
            while o <= snap + 1e-9:
                e = _elev(pts[i][0] + px * o, pts[i][1] + py * o)
                if e is not None and (best is None or e < best):
                    best = e; bx, by = pts[i][0] + px * o, pts[i][1] + py * o
                o += 0.5
            sp.append((bx, by))
        a = sp
        for _ in range(2):
            b = a[:]
            for i in range(1, len(a) - 1):
                b[i] = (0.25 * a[i - 1][0] + 0.5 * a[i][0] + 0.25 * a[i + 1][0],
                        0.25 * a[i - 1][1] + 0.5 * a[i][1] + 0.25 * a[i + 1][1])
            a = b
        L = _path_length_xy(a)
        if L >= 4.0:
            out_lines.append({'coords': a, 'length': L,
                              'MeanDep': 0.0, 'MaxDep': 0.0, 'PixCnt': len(a)})
    return out_lines


def _subpix_thalweg_pass(all_lines, in_dem, arcpy, params):
    """Sub-pixel SMOOTH thalweg: pull every point onto the CONTINUOUS lowest
    point of its ditch's cross-section (parabola fit around the perpendicular
    minimum -> no pixel-quantisation), then smooth; iterate. The line then runs
    smoothly along the true channel bottom. Radius is kept small so it never
    jumps to a parallel ditch. (Measured: drift median 0.50->0.16 m, smoother.)"""
    if not all_lines:
        return all_lines
    R = float(params.get('road_thalweg_radius_m', 5.0))
    dip_margin = float(params.get('road_thalweg_dip_m', 0.12))
    iters = int(params.get('road_thalweg_iters', 3))
    smw = int(params.get('road_thalweg_smooth', 3))
    step = float(params.get('road_thalweg_step_m', 1.5))
    smooth_sigma = float(params.get('smooth_sigma_px', 0.9))
    pad = R + 6.0
    out = []
    for line in all_lines:
        coords = line.get('coords', [])
        if len(coords) < 4:
            out.append(line); continue
        win = _read_dem_window_for_coords(coords, in_dem, arcpy, pad)
        if win is None:
            out.append(line); continue
        arr, valid, axmin, aymax, cellx, celly = win
        dem = (_nan_gaussian_smooth(arr, valid, smooth_sigma)
               if smooth_sigma > 0.05 else arr)
        h, w = dem.shape

        def elev(x, y):
            cf = (x - axmin) / cellx; rf = (aymax - y) / celly
            c0 = int(math.floor(cf)); r0 = int(math.floor(rf))
            if (0 <= r0 < h - 1 and 0 <= c0 < w - 1 and valid[r0, c0]
                    and valid[r0 + 1, c0] and valid[r0, c0 + 1]
                    and valid[r0 + 1, c0 + 1]):
                dc = cf - c0; dr = rf - r0
                return float((dem[r0, c0] * (1 - dc) + dem[r0, c0 + 1] * dc) * (1 - dr)
                             + (dem[r0 + 1, c0] * (1 - dc) + dem[r0 + 1, c0 + 1] * dc) * dr)
            ci = int(round(cf)); ri = int(round(rf))
            if 0 <= ri < h and 0 <= ci < w and valid[ri, ci]:
                return float(dem[ri, ci])
            return None

        def subpix(cx, cy, px, py):
            os_ = np.arange(-R, R + 1e-9, 0.25)
            es = [elev(cx + px * o, cy + py * o) for o in os_]
            if any(e is None for e in es):
                return 0.0
            # DETREND the cross-section (remove the hillside slope) so we snap to
            # the real channel DIP, not the absolute lowest point downslope.
            prof = np.array(es, dtype=np.float64)
            co = np.polyfit(os_, prof, 1)
            res = prof - np.polyval(co, os_)
            ends = min(res[0], res[-1])
            # NEAREST real dip: among local minima that are genuine channel dips
            # (below both rim ends by dip_margin), pick the one CLOSEST to the
            # current point. This lets the search radius be large (reach the
            # channel on a wide/steep section) WITHOUT jumping to a parallel
            # ditch farther out. On a pure slope (no dip) -> stay put (no drag).
            best_o = None; best_abs = 1e9
            for k in range(1, len(res) - 1):
                if (res[k] <= res[k - 1] and res[k] < res[k + 1]
                        and res[k] < ends - dip_margin and abs(os_[k]) < R - 0.5):
                    den = (res[k - 1] - 2 * res[k] + res[k + 1])
                    d = 0.5 * (res[k - 1] - res[k + 1]) / den if abs(den) > 1e-9 else 0.0
                    d = max(-1.0, min(1.0, d))
                    o = float(os_[k] + d * 0.25)
                    if abs(o) < best_abs:
                        best_abs = abs(o); best_o = o
            return best_o if best_o is not None else 0.0

        g = _resample_polyline_xy(coords, step)
        if len(g) < 4:
            out.append(line); continue
        for _ in range(iters):
            nw = [g[0]]
            for i in range(1, len(g) - 1):
                ux, uy = g[i + 1][0] - g[i - 1][0], g[i + 1][1] - g[i - 1][1]
                m = math.hypot(ux, uy)
                if m < 1e-9:
                    nw.append(g[i]); continue
                px, py = -uy / m, ux / m
                o = subpix(g[i][0], g[i][1], px, py)
                nw.append((g[i][0] + px * o, g[i][1] + py * o))
            nw.append(g[-1])
            a = np.array(nw); s = a.copy()
            for _ in range(smw):
                t = s.copy()
                for i in range(1, len(a) - 1):
                    s[i] = 0.25 * t[i - 1] + 0.5 * t[i] + 0.25 * t[i + 1]
            g = [tuple(p) for p in s]
        nl = dict(line); nl['coords'] = g; nl['length'] = _path_length_xy(g)
        out.append(nl)
    return out


def _prob_centroid_thalweg_pass(all_lines, params):
    """v12.9.9 -- centre each line on the CNN prob-RIDGE by its per-cross-section
    PROB-WEIGHTED CENTROID (not the elevation dip, which is sub-noise on shallow
    road ditches). Measured on the user's road corrections: drift 0.50 -> 0.25 m,
    >1m tail 12% -> 0%. Uses params['cnn_prob'] (+ 'cnn_prob_geo'); no DEM needed.
    Pure numpy-1.9."""
    prob = params.get('cnn_prob'); geo = params.get('cnn_prob_geo')
    if prob is None or geo is None or not all_lines:
        return all_lines
    xmin, ymax, cellx, celly = geo
    h, w = prob.shape
    R = float(params.get('centroid_radius_m', 5.0))
    iters = int(params.get('centroid_iters', 4))
    step = float(params.get('centroid_step_m', 1.5))
    smw = int(params.get('centroid_smooth', 1))   # light smoothing: keep on centroid

    def pval(x, y):
        cf = (x - xmin) / cellx; rf = (ymax - y) / celly
        c0 = int(math.floor(cf)); r0 = int(math.floor(rf))
        if 0 <= r0 < h - 1 and 0 <= c0 < w - 1:
            dc = cf - c0; dr = rf - r0
            return float((prob[r0, c0] * (1 - dc) + prob[r0, c0 + 1] * dc) * (1 - dr)
                         + (prob[r0 + 1, c0] * (1 - dc) + prob[r0 + 1, c0 + 1] * dc) * dr)
        ci = int(round(cf)); ri = int(round(rf))
        return float(prob[ri, ci]) if (0 <= ri < h and 0 <= ci < w) else 0.0

    offs = np.arange(-R, R + 1e-9, 0.25)
    out = []
    for line in all_lines:
        coords = line.get('coords', [])
        if len(coords) < 4:
            out.append(line); continue
        g = _resample_polyline_xy(coords, step)
        if len(g) < 4:
            out.append(line); continue
        for _ in range(iters):
            nw = [g[0]]
            for i in range(1, len(g) - 1):
                ux, uy = g[i + 1][0] - g[i - 1][0], g[i + 1][1] - g[i - 1][1]
                m = math.hypot(ux, uy)
                if m < 1e-9:
                    nw.append(g[i]); continue
                px, py = -uy / m, ux / m
                wts = np.array([pval(g[i][0] + px * o, g[i][1] + py * o)
                                for o in offs], dtype=np.float64)
                wts = np.clip(wts, 0.0, None)
                s = wts.sum()
                o = float((offs * wts).sum() / s) if s > 1e-6 else 0.0
                o = max(-R, min(R, o))
                nw.append((g[i][0] + px * o, g[i][1] + py * o))
            nw.append(g[-1])
            a = np.array(nw); sm = a.copy()
            for _ in range(smw):
                t = sm.copy()
                for i in range(1, len(a) - 1):
                    sm[i] = 0.25 * t[i - 1] + 0.5 * t[i] + 0.25 * t[i + 1]
            g = [tuple(p) for p in sm]
        nl = dict(line); nl['coords'] = g; nl['length'] = _path_length_xy(g)
        out.append(nl)
    return out


def _resample_polyline_xy(coords, step):
    pts = [coords[0]]; carried = 0.0; target = step
    for i in range(1, len(coords)):
        x0, y0 = coords[i - 1]; x1, y1 = coords[i]
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-9:
            continue
        while carried + d >= target:
            tt = (target - carried) / d
            pts.append((x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt))
            target += step
        carried += d
    pts.append(coords[-1])
    return pts


def _orient_high_to_low(all_lines, in_dem, arcpy):
    """Order every polyline from its HIGH end to its LOW end -- the direction
    water flows through a trench/nala. So the first vertex is always the source
    (highest) and the last is the outlet (lowest)."""
    if not all_lines:
        return all_lines
    out = []
    for line in all_lines:
        c = line.get('coords', [])
        if len(c) < 2:
            out.append(line); continue
        win = _read_dem_window_for_coords([c[0], c[-1]], in_dem, arcpy, 4.0)
        if win is None:
            out.append(line); continue
        arr, valid, axmin, aymax, cellx, celly = win
        h, w = arr.shape

        def ev(x, y):
            cc = int(round((x - axmin) / cellx)); rr = int(round((aymax - y) / celly))
            if 0 <= rr < h and 0 <= cc < w and valid[rr, cc]:
                return float(arr[rr, cc])
            return None
        z0 = ev(*c[0]); z1 = ev(*c[-1])
        if z0 is not None and z1 is not None and z0 < z1:
            nl = dict(line); nl['coords'] = list(reversed(c)); out.append(nl)
        else:
            out.append(line)
    return out


def _cut_one_ridge(coords, elev, rise, win_s, trim, min_len):
    """One ridge-cut over a single polyline. Resample at 1 m, DETREND the
    elevation profile by arc-length (removes the natural downhill slope), find
    interior CROWNS -- a local max that rises > `rise` m above the channel floor
    on BOTH sides within +-win_s samples -- and cut the line there. Returns the
    list of sub-paths, or None when there is no ridge crossing (caller keeps the
    line unchanged). Pure numpy-1.9 (polyfit / polyval / slicing only)."""
    g = _resample_polyline_xy(coords, 1.0)
    if len(g) < 6:
        return None
    es = [elev(p[0], p[1]) for p in g]
    if None in es:
        return None
    el = np.array(es, dtype=np.float64)
    d = np.arange(len(g), dtype=np.float64)
    co = np.polyfit(d, el, 1)
    res = el - np.polyval(co, d)
    n = len(res)
    cuts = []
    for k in range(1, n - 1):
        if res[k] >= res[k - 1] and res[k] > res[k + 1]:
            lo = k - win_s
            if lo < 0:
                lo = 0
            lft = res[lo:k].min()
            rgt = res[k + 1:k + win_s + 1].min()
            flank = lft if lft > rgt else rgt    # the SHALLOWER (higher) side
            if res[k] - flank > rise:
                cuts.append(k)
    if not cuts:
        return None
    # group crowns that sit within 3 samples of each other into one cut-zone
    zones = []
    zlo = zhi = cuts[0]
    for k in cuts[1:]:
        if k - zhi <= 3:
            zhi = k
        else:
            zones.append((zlo, zhi))
            zlo = zhi = k
    zones.append((zlo, zhi))
    pieces = []
    s = 0
    for zlo, zhi in zones:
        b = zlo - trim
        if b - s >= 2:
            pieces.append(g[s:b + 1])
        s = zhi + trim
    if (n - 1) - s >= 2:
        pieces.append(g[s:])
    return [p for p in pieces if _path_length_xy(p) >= min_len]


def _ridge_cut_pass(all_lines, in_dem, arcpy, params):
    """ANTI-WEAVE (the user's core rule): a trench must run straight along its
    OWN channel and never turn UP over a ridge into a parallel / different
    trench. Wherever a line's elevation profile crosses such a ridge (an interior
    crown higher than the channel floor on both sides), the line has left its
    channel -- cut it there into separate trenches, trimming the crown and
    dropping tiny fragments. Iterated so grouped / residual crowns are all
    removed. This also enforces the PIPE rule generally (a culvert is just a
    crown). Returns (lines, n_lines_cut). Pure numpy-1.9.

    Measured on the user's sdat1 DEM (the exact flagged tool output):
    ridge-crossings 35 -> 0, his hand-fixed clean lines untouched (0 -> 0),
    99% of total length kept."""
    if not all_lines:
        return all_lines, 0
    rise = float(params.get('ridge_cut_m', 0.28))
    win_m = float(params.get('ridge_cut_win_m', 8.0))
    trim = int(params.get('ridge_cut_trim', 2))
    min_len = float(params.get('ridge_cut_min_seg_m', 12.0))
    # ridge detection uses its OWN light smoothing (a ridge is a real terrain
    # feature; over-smoothing erodes borderline crowns and lets weaves slip
    # through). Default 0.6 px -- enough to kill pixel noise, keeps real crowns.
    smooth_sigma = float(params.get('ridge_cut_smooth_px', 0.6))
    win_s = int(round(win_m))
    pad = win_m + 6.0
    out = []
    n_cut = 0
    for line in all_lines:
        coords = line.get('coords', [])
        if len(coords) < 6:
            out.append(line)
            continue
        win = _read_dem_window_for_coords(coords, in_dem, arcpy, pad)
        if win is None:
            out.append(line)
            continue
        arr, valid, axmin, aymax, cellx, celly = win
        dem = (_nan_gaussian_smooth(arr, valid, smooth_sigma)
               if smooth_sigma > 0.05 else arr)
        h, w = dem.shape

        def elev(x, y):
            cf = (x - axmin) / cellx
            rf = (aymax - y) / celly
            c0 = int(math.floor(cf))
            r0 = int(math.floor(rf))
            if (0 <= r0 < h - 1 and 0 <= c0 < w - 1 and valid[r0, c0]
                    and valid[r0 + 1, c0] and valid[r0, c0 + 1]
                    and valid[r0 + 1, c0 + 1]):
                dc = cf - c0
                dr = rf - r0
                return float((dem[r0, c0] * (1 - dc) + dem[r0, c0 + 1] * dc)
                             * (1 - dr)
                             + (dem[r0 + 1, c0] * (1 - dc)
                                + dem[r0 + 1, c0 + 1] * dc) * dr)
            ci = int(round(cf))
            ri = int(round(rf))
            if 0 <= ri < h and 0 <= ci < w and valid[ri, ci]:
                return float(dem[ri, ci])
            return None

        stack = [coords]
        changed = False
        for _it in range(3):
            nxt = []
            it_changed = False
            for c in stack:
                pieces = _cut_one_ridge(c, elev, rise, win_s, trim, min_len)
                if pieces is None:
                    nxt.append(c)
                else:
                    nxt.extend(pieces)
                    it_changed = True
                    changed = True
            stack = nxt
            if not it_changed:
                break
        if not changed:
            out.append(line)
            continue
        n_cut += 1
        for c in stack:
            nl = dict(line)
            nl['coords'] = c
            nl['length'] = _path_length_xy(c)
            out.append(nl)
    return out, n_cut


def _road_pipe_split_trim(all_lines, in_dem, arcpy, params):
    """Road Mode -- the user's PIPE rule. A road-side trench (nala) carries
    water along the lowest pixels and STOPS at a pipe/culvert (a high pixel in
    the channel); two parallel road-side ditches stay separate even when they
    curve. This pass enforces that on the final lines:
      (A) END-TRIM: from each tip, walk inward while the cross-section is
          flat/crown (perpendicular relief < margin) and drop it; stop at the
          first real groove. A wholly-flat (genuinely faint) line is kept whole.
      (B) HUMP-SPLIT: split where the channel floor humps above the floor on
          BOTH sides by >= rise (a pipe, or the road crown between two parallel
          ditches) -> the trench stops at the pipe, the far side is its own
          trench, and parallel ditches separate. The high start of a real
          trench is NOT cut (it is low on the downstream side only).
    Pure geometry, no retrain. Gated on params['road_mode']; off by default so
    agricultural / river output is byte-identical."""
    if not all_lines:
        return all_lines, 0
    step = max(0.5, float(params.get('road_pipe_step_m', 1.0)))
    side = float(params.get('road_pipe_side_m', 3.0))
    snap = float(params.get('road_pipe_snap_m', 2.5))
    margin = float(params.get('road_pipe_margin_m', 0.08))
    rise = float(params.get('road_pipe_rise_m', 0.40))
    win_n = int(round(float(params.get('road_pipe_window_m', 12.0)) / step))
    min_seg = float(params.get('road_pipe_min_seg_m', 7.0))
    smooth_sigma = float(params.get('smooth_sigma_px', 0.9))
    pad = side + snap + 6.0

    def _rs(coords):
        pts = [coords[0]]; carried = 0.0; target = step
        for i in range(1, len(coords)):
            x0, y0 = coords[i - 1]; x1, y1 = coords[i]
            d = math.hypot(x1 - x0, y1 - y0)
            if d < 1e-9:
                continue
            while carried + d >= target:
                tt = (target - carried) / d
                pts.append((x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt))
                target += step
            carried += d
        pts.append(coords[-1])
        return pts

    out = []
    n_changed = 0
    for line in all_lines:
        coords = line.get('coords', [])
        if len(coords) < 5:
            out.append(line); continue
        win = _read_dem_window_for_coords(coords, in_dem, arcpy, pad)
        if win is None:
            out.append(line); continue
        arr, valid, axmin, aymax, cellx, celly = win
        dem = (_nan_gaussian_smooth(arr, valid, smooth_sigma)
               if smooth_sigma > 0.05 else arr)
        h, w = dem.shape

        def elev(x, y):
            col = int(round((x - axmin) / cellx))
            row = int(round((aymax - y) / celly))
            if 0 <= row < h and 0 <= col < w and valid[row, col]:
                return float(dem[row, col])
            return None

        def tangent(g, i):
            a = g[max(0, i - 1)]; b = g[min(len(g) - 1, i + 1)]
            ux, uy = b[0] - a[0], b[1] - a[1]
            m = math.hypot(ux, uy)
            if m < 1e-9:
                return None
            return ux / m, uy / m

        g = _rs(coords)
        if len(g) < 5:
            out.append(line); continue

        # (A) end-trim by perpendicular relief
        groove = []
        for i in range(len(g)):
            t = tangent(g, i); c = elev(g[i][0], g[i][1])
            if t is None or c is None:
                groove.append(False); continue
            px, py = -t[1], t[0]
            l = elev(g[i][0] - px * side, g[i][1] - py * side)
            r = elev(g[i][0] + px * side, g[i][1] + py * side)
            if l is None or r is None:
                groove.append(False); continue
            groove.append((min(l, r) - c) >= margin)
        if any(groove):
            i0 = groove.index(True)
            i1 = len(g) - 1 - list(reversed(groove)).index(True)
            g = g[i0:i1 + 1]
        if len(g) < 5:
            if _path_length_xy(g) >= min_seg:
                nl = dict(line); nl['coords'] = g
                nl['length'] = _path_length_xy(g); out.append(nl)
            n_changed += 1
            continue

        # (B) hump split on the trimmed line
        zf = []
        for i in range(len(g)):
            t = tangent(g, i); c = elev(g[i][0], g[i][1])
            if t is None or c is None:
                zf.append(c); continue
            px, py = -t[1], t[0]; best = c; o = -snap
            while o <= snap + 1e-9:
                e = elev(g[i][0] + px * o, g[i][1] + py * o)
                if e is not None and e < best:
                    best = e
                o += 0.5
            zf.append(best)
        n = len(g); hump = [False] * n
        for i in range(n):
            if zf[i] is None:
                continue
            bz = [v for v in zf[max(0, i - win_n):i + 1] if v is not None]
            az = [v for v in zf[i:min(n, i + win_n + 1)] if v is not None]
            if bz and az and (zf[i] - min(bz)) > rise and (zf[i] - min(az)) > rise:
                hump[i] = True
        segs = []; cur = []
        for i in range(n):
            if not hump[i]:
                cur.append(g[i])
            else:
                if _path_length_xy(cur) >= min_seg:
                    segs.append(cur)
                cur = []
        if _path_length_xy(cur) >= min_seg:
            segs.append(cur)
        if not segs:
            if _path_length_xy(g) >= min_seg:
                segs = [g]
            else:
                n_changed += 1
                continue

        if len(segs) != 1 or len(segs[0]) != len(coords):
            n_changed += 1
        for s in segs:
            nl = dict(line)
            nl['coords'] = s
            nl['length'] = _path_length_xy(s)
            out.append(nl)
    return out, n_changed


def _sample_prob_at_xy(cnn_prob, cnn_prob_geo, x, y):
    """v11.6: bilinear-ish sample of the full-DEM CNN probability at world
    coords. cnn_prob_geo = (xmin, ymax, cellx, celly). Returns None outside."""
    if cnn_prob is None:
        return None
    xmin, ymax, cx, cy = cnn_prob_geo
    H, W = cnn_prob.shape
    fc = (x - xmin) / cx
    fr = (ymax - y) / cy
    r = int(round(fr)); c = int(round(fc))
    if 0 <= r < H and 0 <= c < W:
        return float(cnn_prob[r, c])
    return None


def _extend_single_endpoint(coords, at_start, dem_arr, dem_valid,
                            arr_xmin, arr_ymax, cellx, celly,
                            ext_threshold, max_ext_m, half_width_px,
                            smooth_sigma, endpoint_fail_steps=2,
                            cnn_prob=None, cnn_prob_geo=None,
                            ext_prob_threshold=0.30):
    """
    Extend one endpoint of a polyline by directly sampling the DEM
    cross-section perpendicular to the trench direction.

    Unlike raster-level extension, this operates on clean map coordinates
    and reads the smoothed DEM at each step. It follows the actual
    depression, not a pre-computed mask.
    """
    cell = (cellx + celly) * 0.5
    max_steps = int(max_ext_m / cell)
    h, w = dem_arr.shape

    if at_start:
        n = min(6, len(coords))
        dx = coords[0][0] - coords[n - 1][0]
        dy = coords[0][1] - coords[n - 1][1]
    else:
        n = min(6, len(coords))
        dx = coords[-1][0] - coords[-n][0]
        dy = coords[-1][1] - coords[-n][1]

    mag = math.hypot(dx, dy)
    if mag < 0.001:
        return coords
    dx /= mag
    dy /= mag

    if at_start:
        cur_x, cur_y = coords[0]
    else:
        cur_x, cur_y = coords[-1]

    new_points = []
    consecutive_fail = 0
    max_fail = int(max(1, endpoint_fail_steps))
    min_side_depth = max(0.012, float(ext_threshold) * 0.35)

    for step in range(1, max_steps + 1):
        nx = cur_x + dx * cell
        ny = cur_y + dy * cell

        if not (arr_xmin <= nx <= arr_xmin + w * cellx and
                arr_ymax - h * celly <= ny <= arr_ymax):
            break

        ev = _cross_section_evidence(dem_arr, dem_valid, arr_xmin, arr_ymax,
                                     cellx, celly, nx, ny, dx, dy,
                                     half_width_px)
        # v11.6 (G): also consult the CNN probability map. Trench-tail signal
        # is often weak in the DEM but the trained network has learned to
        # recognise it, so accept extension when EITHER signal is strong.
        prob_val = _sample_prob_at_xy(cnn_prob, cnn_prob_geo, nx, ny)
        prob_ok = (prob_val is not None and prob_val >= ext_prob_threshold)

        if ev is None:
            # no DEM evidence at all - only continue if CNN says it's a trench
            if prob_ok:
                new_points.append((nx, ny))
                cur_x, cur_y = nx, ny
                consecutive_fail = 0
                continue
            consecutive_fail += 1
            if consecutive_fail >= max_fail:
                break
            cur_x, cur_y = nx, ny
            continue

        dem_ok = (ev['avg_depth'] >= ext_threshold and
                  ev['min_side_depth'] >= min_side_depth)
        if dem_ok or prob_ok:
            new_points.append((nx, ny))
            cur_x, cur_y = nx, ny
            consecutive_fail = 0
        else:
            consecutive_fail += 1
            if consecutive_fail >= max_fail:
                break
            cur_x, cur_y = nx, ny

    if new_points:
        if at_start:
            new_points.reverse()
            return new_points + list(coords)
        else:
            return list(coords) + new_points
    return coords


def _trim_single_endpoint(coords, at_start, dem_arr, dem_valid,
                          arr_xmin, arr_ymax, cellx, celly,
                          trim_threshold, max_trim_m, half_width_px,
                          endpoint_fail_steps=2):
    if len(coords) < 3 or max_trim_m <= 0:
        return coords, 0.0, None

    work = coords if at_start else list(reversed(coords))
    total = _path_length_xy(work)
    max_dist = min(float(max_trim_m), total * 0.45)
    cell = (cellx + celly) * 0.5
    step = max(cell, 0.5)
    samples = int(max(2, math.ceil(max_dist / step))) + 1
    min_side_depth = max(0.015, float(trim_threshold) * 0.40)

    good_index = None
    good_depth = None
    for i in range(samples):
        dist = min(max_dist, i * step)
        pt = _point_at_distance(work, dist)
        pt2 = _point_at_distance(work, min(total, dist + step * 2.0))
        if pt is None or pt2 is None:
            continue
        dx = pt2[0] - pt[0]
        dy = pt2[1] - pt[1]
        if math.hypot(dx, dy) <= 0.0:
            continue
        ev = _cross_section_evidence(dem_arr, dem_valid, arr_xmin, arr_ymax,
                                     cellx, celly, pt[0], pt[1], dx, dy,
                                     half_width_px)
        if ev is None:
            continue
        if (ev['avg_depth'] >= trim_threshold and
                ev['min_side_depth'] >= min_side_depth):
            good_index = i
            good_depth = ev['avg_depth']
            break

    if good_index is None:
        return coords, 0.0, None

    trim_dist = max(0.0, (good_index -
                          int(max(0, endpoint_fail_steps - 1))) * step)
    if trim_dist <= 0.0:
        return coords, 0.0, good_depth

    if at_start:
        new_coords = _trim_coords_at_distance(coords, trim_dist, True)
    else:
        new_coords = _trim_coords_at_distance(coords, trim_dist, False)
    return new_coords, trim_dist, good_depth


def _endpoint_depth(coords, at_start, dem_arr, dem_valid,
                    arr_xmin, arr_ymax, cellx, celly, half_width_px):
    if len(coords) < 2:
        return None
    if at_start:
        p = coords[0]
        p2 = coords[min(2, len(coords) - 1)]
    else:
        p = coords[-1]
        p2 = coords[max(0, len(coords) - 3)]
    dx = p[0] - p2[0]
    dy = p[1] - p2[1]
    ev = _cross_section_evidence(dem_arr, dem_valid, arr_xmin, arr_ymax,
                                 cellx, celly, p[0], p[1], dx, dy,
                                 half_width_px)
    if ev is None:
        return None
    return ev['avg_depth']


def _extend_all_endpoints(all_lines, in_dem, arcpy, params):
    """
    Extend endpoints of all detected lines by reading the DEM directly.

    Groups nearby endpoints and reads DEM windows to minimize I/O.
    Uses the continuation depth threshold for extension evidence.
    """
    ext_threshold = float(params.get('extend_depth_m', 0.05))
    trim_threshold = float(params.get('trim_depth_m',
                                      params.get('grow_depth_m', 0.05)))
    max_trim_m = float(params.get('max_trim_m', 12.0))
    endpoint_fail_steps = int(params.get('endpoint_fail_steps', 2))
    max_ext_m = float(params.get('max_extend_m', 30.0))
    smooth_sigma = float(params['smooth_sigma_px'])
    # v11.6 (G): full-DEM CNN probability map if available
    cnn_prob = params.get('cnn_prob', None)
    cnn_prob_geo = params.get('cnn_prob_geo', None)
    ext_prob_thr = float(params.get('ext_prob_threshold', 0.30))

    if not all_lines:
        return all_lines

    ras = arcpy.Raster(in_dem)
    desc = arcpy.Describe(in_dem)
    ext = ras.extent
    xmin = float(ext.XMin)
    ymax = float(ext.YMax)
    ymin = float(ext.YMin)
    xmax = float(ext.XMax)
    cellx = float(ras.meanCellWidth)
    celly = abs(float(ras.meanCellHeight))
    cell = (cellx + celly) * 0.5
    total_width = int(round((xmax - xmin) / cellx))
    total_height = int(round((ymax - ymin) / celly))

    half_width_px = int(max(2, round(float(params['max_width_m']) /
                                     (2.0 * cell))))

    nodata_value = -9999999.0
    window_pad = int(max_ext_m / cell) + half_width_px + 10

    extended_count = 0
    for line in all_lines:
        coords = line['coords']
        if len(coords) < 3:
            continue

        for at_start in [True, False]:
            if at_start:
                ep_x, ep_y = coords[0]
            else:
                ep_x, ep_y = coords[-1]

            ep_col = int(round((ep_x - xmin) / cellx))
            ep_row = int(round((ymax - ep_y) / celly))

            read_col0 = max(0, ep_col - window_pad)
            read_row0 = max(0, ep_row - window_pad)
            read_col1 = min(total_width, ep_col + window_pad)
            read_row1 = min(total_height, ep_row + window_pad)
            ncols = read_col1 - read_col0
            nrows = read_row1 - read_row0

            if ncols < 10 or nrows < 10:
                continue

            arr_xmin = xmin + read_col0 * cellx
            arr_ymax = ymax - read_row0 * celly
            llx = arr_xmin
            lly = arr_ymax - nrows * celly
            lower_left = arcpy.Point(llx, lly)

            try:
                arr = arcpy.RasterToNumPyArray(
                    in_dem, lower_left, ncols, nrows,
                    nodata_to_value=nodata_value)
                arr = arr.astype(np.float32)
                valid = (np.isfinite(arr) & (arr != nodata_value) &
                         (arr > -1.0e20))
                if np.count_nonzero(valid) < 50:
                    continue

                dem_smooth = _nan_gaussian_smooth(arr, valid, smooth_sigma)

                old_len = len(coords)
                old_path_len = _path_length_xy(coords)
                coords, trim_dist, dep_after_trim = _trim_single_endpoint(
                    coords, at_start, dem_smooth, valid,
                    arr_xmin, arr_ymax, cellx, celly,
                    trim_threshold, max_trim_m, half_width_px,
                    endpoint_fail_steps)
                coords = _extend_single_endpoint(
                    coords, at_start, dem_smooth, valid,
                    arr_xmin, arr_ymax, cellx, celly,
                    ext_threshold, max_ext_m, half_width_px,
                    smooth_sigma, endpoint_fail_steps,
                    cnn_prob=cnn_prob, cnn_prob_geo=cnn_prob_geo,
                    ext_prob_threshold=ext_prob_thr)

                if len(coords) > old_len:
                    extended_count += 1
                if trim_dist > 0.0:
                    line['Trim_m'] = float(line.get('Trim_m', 0.0)) + trim_dist
                new_path_len = _path_length_xy(coords)
                if new_path_len > old_path_len:
                    line['Ext_m'] = (float(line.get('Ext_m', 0.0)) +
                                     (new_path_len - old_path_len))
                dep = _endpoint_depth(coords, at_start, dem_smooth, valid,
                                      arr_xmin, arr_ymax, cellx, celly,
                                      half_width_px)
                if dep is None:
                    dep = dep_after_trim
                if dep is not None:
                    if at_start:
                        line['StartDep'] = float(dep)
                    else:
                        line['EndDep'] = float(dep)
            except Exception:
                pass

        line['coords'] = coords
        line['length'] = _path_length_xy(coords)

    return all_lines, extended_count


# ---------------------------------------------------------------------------
# Main tile processing (core identical to v11)
# ---------------------------------------------------------------------------

def _detect_in_tile(arr, valid, params):
    cell = float(params['cell_size'])
    smooth_sigma_px = float(params['smooth_sigma_px'])
    dem = _nan_gaussian_smooth(arr, valid, smooth_sigma_px)

    strong, grow, depth, orient = _directional_depression(
        dem, valid,
        float(params['seed_depth_m']),
        float(params['grow_depth_m']),
        float(params['max_width_m']),
        cell)

    keep = _hysteresis_from_seeds(strong, grow)

    keep = _binary_close(keep, int(params['gap_close_px']))
    keep &= valid

    skel = _zhang_suen_thin(keep, max_iter=int(params['thin_max_iter']))
    skel = _remove_isolated_and_short(skel)
    return skel, depth, keep


def _extract_lines_from_skeleton(skel, depth, core_box, georef, params,
                                 tile_id):
    r0_core, r1_core, c0_core, c1_core = core_box

    skel_core = skel.copy()
    skel_core[:r0_core, :] = False
    skel_core[r1_core:, :] = False
    skel_core[:, :c0_core] = False
    skel_core[:, c1_core:] = False

    if params.get('junction_aware', False):
        paths = _trace_skeleton_junction_aware(
            skel_core, float(params.get('junction_max_turn_deg', 60.0)))
    else:
        paths = _trace_skeleton(skel_core)
    lines = []
    min_len = float(params['min_length_m'])
    simplify_tol = float(params['simplify_tolerance_m'])
    max_sinuosity = float(params['max_sinuosity'])
    min_quality = float(params.get('min_quality_score', 0.0))

    xmin, ymax, cell, row0_global, col0_global, read_row0, read_col0 = georef

    for path in paths:
        if len(path) < 2:
            continue
        coords = _path_to_coords(path, xmin, ymax, cell, row0_global,
                                 col0_global, read_row0, read_col0)
        length = _path_length_xy(coords)
        if length < min_len:
            continue
        direct = math.hypot(coords[-1][0] - coords[0][0],
                            coords[-1][1] - coords[0][1])
        sinuosity = length / max(direct, cell)
        if sinuosity > max_sinuosity:
            continue

        vals = []
        for r, c in path:
            if 0 <= r < depth.shape[0] and 0 <= c < depth.shape[1]:
                vals.append(float(depth[r, c]))
        if len(vals) == 0:
            continue
        mean_dep = float(np.mean(vals))
        max_dep = float(np.max(vals))
        if max_dep < float(params['seed_depth_m']) * 0.8:
            continue

        if simplify_tol > 0 and len(coords) > 3:
            coords2 = _douglas_peucker(coords, simplify_tol)
            if len(coords2) >= 2:
                coords = coords2
                length = _path_length_xy(coords)
        if length < min_len:
            continue

        length_score = min(length / 80.0, 1.0)
        mean_score = min(mean_dep / 0.50, 1.0)
        max_score = min(max_dep / 1.00, 1.0)
        q = min(100.0, max(0.0, 100.0 * (0.45 * length_score +
                                           0.40 * mean_score +
                                           0.15 * max_score)))
        if q < min_quality:
            continue
        lines.append({
            'coords': coords,
            'length': length,
            'MeanDep': mean_dep,
            'MaxDep': max_dep,
            'QScore': q,
            'PixCnt': int(len(path)),
            'TileID': int(tile_id)
        })
    return lines


def detect_trenches(in_dem, out_fc, arcpy,
                    seed_depth_m=0.08,
                    grow_depth_m=0.035,
                    max_width_m=6.0,
                    min_length_m=8.0,
                    tile_size_px=1200,
                    tile_overlap_px=64,
                    smooth_sigma_px=0.9,
                    gap_close_px=3,
                    join_gap_m=7.0,
                    simplify_tolerance_m=0.7,
                    max_sinuosity=12.0,
                    min_quality_score=10.0,
                    verify_join_gaps=True,
                    join_gap_depth_m=0.03,
                    join_gap_min_ratio=0.50,
                    extend_endpoints=True,
                    extend_depth_m=0.035,
                    max_extend_m=35.0,
                    trim_depth_m=0.035,
                    max_trim_m=8.0,
                    endpoint_fail_steps=3,
                    regularize_lines=True,
                    regularize_spacing_m=5.0,
                    regularize_simplify_m=2.0,
                    regularize_scan_step_m=0.5,
                    smooth_iterations=2,
                    min_line_support_ratio=0.45,
                    line_support_depth_m=0.035,
                    max_output_angle_jitter=35.0,
                    max_output_sinuosity=2.4,
                    final_join_after_regularize=True,
                    final_join_gap_m=14.0,
                    overwrite=True):
    """Main ArcPy entry point."""
    start_time = time.time()

    params = {
        'seed_depth_m': float(seed_depth_m),
        'grow_depth_m': float(grow_depth_m),
        'max_width_m': float(max_width_m),
        'min_length_m': float(min_length_m),
        'tile_size_px': int(tile_size_px),
        'tile_overlap_px': int(tile_overlap_px),
        'smooth_sigma_px': float(smooth_sigma_px),
        'gap_close_px': int(gap_close_px),
        'join_gap_m': float(join_gap_m),
        'simplify_tolerance_m': float(simplify_tolerance_m),
        'max_sinuosity': float(max_sinuosity),
        'min_quality_score': float(min_quality_score),
        'thin_max_iter': 80,
        'verify_join_gaps': bool(verify_join_gaps),
        'join_gap_depth_m': float(join_gap_depth_m),
        'join_gap_min_ratio': float(join_gap_min_ratio),
        'extend_depth_m': float(extend_depth_m),
        'max_extend_m': float(max_extend_m),
        'trim_depth_m': float(trim_depth_m),
        'max_trim_m': float(max_trim_m),
        'endpoint_fail_steps': int(endpoint_fail_steps),
        'regularize_lines': bool(regularize_lines),
        'regularize_spacing_m': float(regularize_spacing_m),
        'regularize_simplify_m': float(regularize_simplify_m),
        'regularize_scan_step_m': float(regularize_scan_step_m),
        'smooth_iterations': int(smooth_iterations),
        'smooth_weight': 0.55,
        'min_line_support_ratio': float(min_line_support_ratio),
        'line_support_depth_m': float(line_support_depth_m),
        'max_output_angle_jitter': float(max_output_angle_jitter),
        'max_output_sinuosity': float(max_output_sinuosity),
        'final_join_after_regularize': bool(final_join_after_regularize),
        'final_join_gap_m': float(final_join_gap_m),
    }

    ras = arcpy.Raster(in_dem)
    desc = arcpy.Describe(in_dem)
    sr = desc.spatialReference
    ext = ras.extent
    xmin = float(ext.XMin)
    ymax = float(ext.YMax)
    ymin = float(ext.YMin)
    xmax = float(ext.XMax)
    cellx = float(ras.meanCellWidth)
    celly = abs(float(ras.meanCellHeight))
    cell = (cellx + celly) * 0.5
    params['cell_size'] = cell

    width = int(round((xmax - xmin) / cellx))
    height = int(round((ymax - ymin) / celly))

    if abs(cell - 1.0) > 0.25:
        _warn(arcpy, "This tool is calibrated for about 1 m DEMs. "
              "Detected cell size: %.3f m" % cell)

    _msg(arcpy, "TrenchToolkit v15 started")
    _msg(arcpy, "DEM size: %d columns x %d rows, cell %.3f m" %
         (width, height, cell))
    _msg(arcpy, "Seed depth: %.3f m, continuation: %.3f m, "
         "soft core width: %.2f m" %
         (params['seed_depth_m'], params['grow_depth_m'],
          params['max_width_m']))
    _msg(arcpy, "Min length: %.1f m, min quality: %.1f, "
         "gap close: %d px" %
         (params['min_length_m'], params['min_quality_score'],
          params['gap_close_px']))
    if extend_endpoints:
        _msg(arcpy, "Endpoint trim/extension: ON (trim=%.3f m, "
             "extend=%.3f m, max extend=%.0f m)" %
             (params['trim_depth_m'], params['extend_depth_m'],
              params['max_extend_m']))
    if params['verify_join_gaps']:
        _msg(arcpy, "DEM-verified stitching: ON (gap depth=%.3f m, "
             "ratio=%.2f)" % (params['join_gap_depth_m'],
                               params['join_gap_min_ratio']))
    if params['regularize_lines']:
        _msg(arcpy, "DEM line regularization: ON (spacing=%.1f m, "
             "simplify=%.1f m, support ratio=%.2f)" %
             (params['regularize_spacing_m'],
              params['regularize_simplify_m'],
              params['min_line_support_ratio']))
        if params['final_join_after_regularize']:
            _msg(arcpy, "Final completion join: ON (gap=%.1f m)" %
                 params['final_join_gap_m'])

    out_ws = os.path.dirname(out_fc)
    out_name = os.path.basename(out_fc)
    if out_ws == '':
        out_ws = arcpy.env.workspace or os.getcwd()
        out_fc = os.path.join(out_ws, out_name)

    if arcpy.Exists(out_fc):
        if overwrite:
            arcpy.Delete_management(out_fc)
        else:
            raise RuntimeError("Output already exists: " + out_fc)

    arcpy.CreateFeatureclass_management(out_ws, out_name, "POLYLINE",
                                        spatial_reference=sr)
    fields = [
        ("Len_m", "DOUBLE"),
        ("MeanDep", "DOUBLE"),
        ("MaxDep", "DOUBLE"),
        ("QScore", "DOUBLE"),
        ("PixCnt", "LONG"),
        ("TileID", "LONG"),
        ("StartDep", "DOUBLE"),
        ("EndDep", "DOUBLE"),
        ("Trim_m", "DOUBLE"),
        ("Ext_m", "DOUBLE"),
        ("Support", "DOUBLE"),
        ("Jitter", "DOUBLE"),
        ("Sinuosity", "DOUBLE"),
        ("Width_m", "DOUBLE")
    ]
    for name, ftype in fields:
        try:
            arcpy.AddField_management(out_fc, name, ftype)
        except Exception:
            pass

    all_lines = []
    tile_size = max(256, int(params['tile_size_px']))
    overlap = max(8, int(params['tile_overlap_px']))
    tile_id = 0

    nodata_value = -9999999.0
    for row0 in range(0, height, tile_size):
        core_rows = min(tile_size, height - row0)
        for col0 in range(0, width, tile_size):
            tile_id += 1
            core_cols = min(tile_size, width - col0)

            read_row0 = max(0, row0 - overlap)
            read_col0 = max(0, col0 - overlap)
            read_row1 = min(height, row0 + core_rows + overlap)
            read_col1 = min(width, col0 + core_cols + overlap)
            nrows = int(read_row1 - read_row0)
            ncols = int(read_col1 - read_col0)

            llx = xmin + read_col0 * cellx
            lly = ymax - read_row1 * celly
            lower_left = arcpy.Point(llx, lly)

            _msg(arcpy, "Tile %d: row %d-%d col %d-%d" %
                 (tile_id, row0, row0 + core_rows, col0, col0 + core_cols))

            arr = arcpy.RasterToNumPyArray(in_dem, lower_left, ncols, nrows,
                                           nodata_to_value=nodata_value)
            arr = arr.astype(np.float32)
            valid = (np.isfinite(arr) & (arr != nodata_value) &
                     (arr > -1.0e20))
            if np.count_nonzero(valid) < 100:
                continue

            core_box = (row0 - read_row0,
                        row0 - read_row0 + core_rows,
                        col0 - read_col0,
                        col0 - read_col0 + core_cols)

            try:
                skel, depth, keep = _detect_in_tile(arr, valid, params)
                georef = (xmin, ymax, cell, row0, col0, read_row0, read_col0)
                lines = _extract_lines_from_skeleton(skel, depth, core_box,
                                                     georef, params, tile_id)
                if lines:
                    _msg(arcpy, "  detected %d line segment(s)" % len(lines))
                    all_lines.extend(lines)
            except Exception as exc:
                _warn(arcpy, "Tile %d failed: %s" % (tile_id, str(exc)))

            try:
                del arr, valid
            except Exception:
                pass

    _msg(arcpy, "Raw line segments before stitching: %d" % len(all_lines))
    if params['join_gap_m'] > 0:
        if len(all_lines) > 10000:
            _warn(arcpy, "Skipping stitching: %d segments found." %
                  len(all_lines))
        else:
            all_lines = _try_join_lines(
                all_lines, params['join_gap_m'],
                in_dem=in_dem, arcpy=arcpy, params=params)
            _msg(arcpy, "Line segments after stitching: %d" % len(all_lines))

    # v13 endpoint trim/extension
    if extend_endpoints and all_lines:
        _msg(arcpy, "Extending endpoints...")
        try:
            all_lines, ext_count = _extend_all_endpoints(
                all_lines, in_dem, arcpy, params)
            _msg(arcpy, "Extended %d endpoint(s)" % ext_count)
        except Exception as exc:
            _warn(arcpy, "Endpoint extension failed: %s" % str(exc))

    if params['regularize_lines'] and all_lines:
        _msg(arcpy, "Regularizing lines against DEM centre depression...")
        all_lines, rejected = _regularize_all_lines(
            all_lines, in_dem, arcpy, params)
        _msg(arcpy, "Regularized lines: %d kept, %d rejected" %
             (len(all_lines), rejected))
        if params['final_join_after_regularize'] and all_lines:
            before_final_join = len(all_lines)
            all_lines = _try_join_lines(
                all_lines, params['final_join_gap_m'],
                in_dem=in_dem, arcpy=arcpy, params=params)
            _msg(arcpy, "Final completion join: %d -> %d line(s)" %
                 (before_final_join, len(all_lines)))

    inserted = 0
    icur = arcpy.da.InsertCursor(out_fc, ["SHAPE@", "Len_m", "MeanDep",
                                           "MaxDep", "QScore", "PixCnt",
                                           "TileID", "StartDep", "EndDep",
                                           "Trim_m", "Ext_m", "Support",
                                           "Jitter", "Sinuosity", "Width_m"])
    try:
        for line in all_lines:
            coords = line['coords']
            if len(coords) < 2:
                continue
            arrp = arcpy.Array()
            last = None
            for x, y in coords:
                if (last is None or abs(x - last[0]) > 1.0e-9 or
                        abs(y - last[1]) > 1.0e-9):
                    arrp.add(arcpy.Point(float(x), float(y)))
                    last = (x, y)
            if arrp.count < 2:
                continue
            geom = arcpy.Polyline(arrp, sr)
            if geom.length < float(params['min_length_m']):
                continue
            icur.insertRow([geom,
                            float(geom.length),
                            float(line.get('MeanDep', 0.0)),
                            float(line.get('MaxDep', 0.0)),
                            float(line.get('QScore', 0.0)),
                            int(line.get('PixCnt', 0)),
                            int(line.get('TileID', 0)),
                            float(line.get('StartDep', -9999.0)),
                            float(line.get('EndDep', -9999.0)),
                            float(line.get('Trim_m', 0.0)),
                            float(line.get('Ext_m', 0.0)),
                            float(line.get('Support', -9999.0)),
                            float(line.get('Jitter', -9999.0)),
                            float(line.get('Sinuosity', -9999.0)),
                            float(line.get('Width_m', -9999.0))])
            inserted += 1
    finally:
        del icur

    _msg(arcpy, "Finished. Wrote %d trench polyline(s) to: %s" %
         (inserted, out_fc))
    _msg(arcpy, "Elapsed time: %.1f seconds" % (time.time() - start_time))
    return out_fc
