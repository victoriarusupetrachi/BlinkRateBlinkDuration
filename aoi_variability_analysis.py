# -*- coding: utf-8 -*-
"""
aoi_variability_analysis.py

Variability and scanpath analyses applied to the per-fixation files produced by
aoi_fixation_extract.py.

Produces
--------
bootstrap_all.csv   AOI fixation shares with 95% confidence intervals from
                    (a) a case-resampling bootstrap over individual fixation
                        events, and
                    (b) a moving-block bootstrap over the time-ordered fixation
                        sequence, with block lengths of 5, 10 and 20 fixations.
                    The block bootstrap preserves the temporal correlation
                    between consecutive fixations, which the case-resampling
                    bootstrap ignores.
entropy_all.csv     Shannon entropy of the AOI-to-AOI transition-probability
                    distribution, per flight and phase, computed both with and
                    without transitions that remain within the same AOI, with
                    the number of AOIs actually visited reported alongside.
window30s_all.csv   Mean and standard deviation of AOI fixation shares across
                    consecutive non-overlapping 30 s windows within each phase.

Only AOIs with at least MIN_FIX fixations in a phase enter the bootstrap, to
avoid reporting confidence intervals estimated from a handful of events.
Fixations falling outside all AOIs are excluded throughout.
"""

from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuration
SEED = 42
N_BOOT = 2000
BLOCK_LENGTHS = (5, 10, 20)
MIN_FIX = 5
WINDOW_S = 30

PHASES = [
    ('F1', 'Preflight'), ('F1', 'Takeoff'), ('F1', 'Cruise'), ('F1', 'Landing'),
    ('F2', 'Preflight'), ('F2', 'Takeoff'), ('F2', 'Cruise'), ('F2', 'Turbulence'), ('F2', 'Landing'),
    ('F3', 'Preflight'), ('F3', 'Takeoff'), ('F3', 'Cruise'), ('F3', 'Turbulence'), ('F3', 'Landing'),
    ('Real', 'Preflight'), ('Real', 'Taxi'), ('Real', 'Takeoff'),
    ('Real', 'Cruise'), ('Real', 'Turbulence'), ('Real', 'Landing'),
]

# phase boundaries in recording-timestamp units (microseconds); needed only to
# place the 30 s windows relative to the start of each phase
BOUNDS = {
    ('F1', 'Preflight'): (0, 199967000), ('F1', 'Takeoff'): (199967000, 373513000),
    ('F1', 'Cruise'): (373513000, 897112000), ('F1', 'Landing'): (897112000, 1011005000),
    ('F2', 'Preflight'): (0, 863011000), ('F2', 'Takeoff'): (863011000, 984870000),
    ('F2', 'Cruise'): (984870000, 1314326000), ('F2', 'Turbulence'): (1314326000, 1438823000),
    ('F2', 'Landing'): (1438823000, 1621995000),
    ('F3', 'Preflight'): (145276000, 218942000), ('F3', 'Takeoff'): (218942000, 324109000),
    ('F3', 'Cruise'): (324109000, 551381000), ('F3', 'Turbulence'): (551381000, 788984000),
    ('F3', 'Landing'): (788984000, 925080000),
    ('Real', 'Preflight'): (939484000, 1347500000), ('Real', 'Taxi'): (1347500000, 1713507000),
    ('Real', 'Takeoff'): (1713507000, 1927080000), ('Real', 'Cruise'): (1927080000, 3125848000),
    ('Real', 'Turbulence'): (3125848000, 3291405000), ('Real', 'Landing'): (3291405000, 3774557000),
}

rng = np.random.default_rng(SEED)


def ci_case(dur, aoi, target):
    """95% CI from resampling individual fixation events with replacement."""
    n = len(dur)
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = rng.integers(0, n, n)
        d, a = dur[s], aoi[s]
        t = d.sum()
        out[b] = d[a == target].sum() / t * 100 if t > 0 else np.nan
    return np.nanpercentile(out, [2.5, 97.5])


def ci_block(dur, aoi, target, length):
    """95% CI from a circular moving-block bootstrap over the fixation sequence."""
    n = len(dur)
    n_blocks = int(np.ceil(n / length))
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        starts = rng.integers(0, n, n_blocks)
        idx = np.concatenate([(np.arange(s, s + length) % n) for s in starts])[:n]
        d, a = dur[idx], aoi[idx]
        t = d.sum()
        out[b] = d[a == target].sum() / t * 100 if t > 0 else np.nan
    return np.nanpercentile(out, [2.5, 97.5])


def shannon(pairs):
    counts = Counter(pairs)
    total = sum(counts.values())
    p = np.array([c / total for c in counts.values()])
    return -np.sum(p * np.log2(p)), total, len(counts)


boot_rows, ent_rows, win_rows = [], [], []

for flight, phase in PHASES:
    ev = pd.read_csv(f'fx_{flight}_{phase}.csv')
    ev = ev.dropna(subset=['aoi'])
    ev = ev[ev['aoi'].astype(str) != 'None'].reset_index(drop=True)

    dur = ev['dur'].values.astype(float)
    aoi = ev['aoi'].values
    total = dur.sum()

    # --- bootstrap ----------------------------------------------------------
    for target in sorted(set(aoi)):
        k = int((aoi == target).sum())
        if k < MIN_FIX:
            continue
        point = dur[aoi == target].sum() / total * 100
        lo, hi = ci_case(dur, aoi, target)
        row = dict(flight=flight, phase=phase, aoi=target, n_fix=k,
                   pct=round(point, 2), ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                   width_case=round(hi - lo, 2))
        for L in BLOCK_LENGTHS:
            blo, bhi = ci_block(dur, aoi, target, L)
            row[f'block{L}_lo'] = round(blo, 2)
            row[f'block{L}_hi'] = round(bhi, 2)
            row[f'width_block{L}'] = round(bhi - blo, 2)
        boot_rows.append(row)

    # --- transition entropy -------------------------------------------------
    seq = list(aoi)
    n_visited = len(set(seq))
    incl = list(zip(seq[:-1], seq[1:]))
    excl = [(a, b) for a, b in incl if a != b]
    h_incl, n_incl, u_incl = shannon(incl)
    h_excl, n_excl, u_excl = shannon(excl)
    ent_rows.append(dict(
        flight=flight, phase=phase, n_aoi_visited=n_visited,
        n_transitions=n_incl, entropy_bits=round(h_incl, 3), n_unique=u_incl,
        n_transitions_excl_same_aoi=n_excl, entropy_bits_excl_same_aoi=round(h_excl, 3),
        n_unique_excl_same_aoi=u_excl))

    # --- 30 s windows -------------------------------------------------------
    t0, t1 = BOUNDS[(flight, phase)]
    ev['t_rel_s'] = (ev['tmin'] - t0) / 1e6
    n_windows = int(((t1 - t0) / 1e6) // WINDOW_S)   # trailing partial window excluded
    for w in range(n_windows):
        sub = ev[(ev['t_rel_s'] >= w * WINDOW_S) & (ev['t_rel_s'] < (w + 1) * WINDOW_S)]
        tot = sub['dur'].sum()
        if tot == 0:
            continue
        for target in sorted(set(aoi)):
            win_rows.append(dict(flight=flight, phase=phase, window=w, aoi=target,
                                 pct=sub[sub['aoi'] == target]['dur'].sum() / tot * 100))

pd.DataFrame(boot_rows).to_csv('bootstrap_all.csv', index=False)
pd.DataFrame(ent_rows).to_csv('entropy_all.csv', index=False)

W = pd.DataFrame(win_rows)
(W.groupby(['flight', 'phase', 'aoi'])['pct']
  .agg(['mean', 'std', 'count']).round(2).reset_index()
  .to_csv('window30s_all.csv', index=False))

print('written: bootstrap_all.csv, entropy_all.csv, window30s_all.csv')
