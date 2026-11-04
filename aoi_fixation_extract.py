# -*- coding: utf-8 -*-
"""
aoi_fixation_extract.py

Extracts per-fixation AOI assignments and durations from a Tobii Pro Lab
sample-level data export, for each flight phase.

Input  : Tobii Pro Lab sample-level data exports (TSV or CSV): either one
         export per recording, or one per flight phase, containing
         at minimum the columns
            Recording timestamp
            Mapped eye movement type [<reference image>]
            Mapped eye movement type index [<reference image>]
            AOI hit [<reference image> - <AOI name>]   (one per AOI)
Output : fx_<FLIGHT>_<PHASE>.csv with columns tmin, dur, aoi
            tmin - fixation start, in recording-timestamp units (microseconds)
            dur  - fixation duration, in microseconds
            aoi  - AOI name, or 'None' if the fixation fell outside all AOIs

Notes
-----
* Fixations are reconstructed by grouping samples on the MAPPED eye-movement
  index, not the raw one, because the AOI hit columns are computed against the
  mapped reference image.
* Each fixation's duration is estimated as (last sample - first sample) plus one
  median inter-sample period, since the export gives sample timestamps rather
  than fixation durations. Validated against the Tobii Pro Lab metrics export:
  fixation counts match exactly and fixation shares agree to within 0.3
  percentage points.
* A fixation is assigned to the AOI hit by the majority of its samples.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- configuration
# Folders searched for the Tobii exports, in order. '.' means the folder the
# script is run from.
SRC = ('.',)

FLIGHTS = {
    'F1': dict(file='Test Simulator Bogdan 7.08 Data Export zbor 1.tsv',
               sep='\t', ref='cockpit crop',
               phases={'Preflight': (0, 199967000),
                       'Takeoff':   (199967000, 373513000),
                       'Cruise':    (373513000, 897112000),
                       'Landing':   (897112000, 1011005000)}),
    'F2': dict(file='Test Simulator Bogdan 7.08 Data Export zbor 2.tsv',
               sep='\t', ref='cockpit crop',
               phases={'Preflight':  (0, 863011000),
                       'Takeoff':    (863011000, 984870000),
                       'Cruise':     (984870000, 1314326000),
                       'Turbulence': (1314326000, 1438823000),
                       'Landing':    (1438823000, 1621995000)}),
    'F3': dict(file='Test Simulator Bogdan 7.08 Data Export zbor 3.tsv',
               sep='\t', ref='cockpit crop',
               phases={'Preflight':  (145276000, 218942000),
                       'Takeoff':    (218942000, 324109000),
                       'Cruise':     (324109000, 551381000),
                       'Turbulence': (551381000, 788984000),
                       'Landing':    (788984000, 925080000)}),
    # The operational-flight export was split per phase in Tobii Pro Lab, so
    # each phase has its own file. 'file' may therefore be a single export
    # covering the whole recording, or a dict mapping phase -> file.
    'Real': dict(file={'Preflight':  'BC-phase1-preflight.csv.gz',
                       'Taxi':       'BC-phase2-taxi.csv.gz',
                       'Takeoff':    'BC-phase3-takeoff.csv',
                       'Cruise':     'BC-phase4-cruise.csv.gz',
                       'Turbulence': 'BC-phase4-turbulence.csv',
                       'Landing':    'BC-phase6-landing.csv.gz'},
                 sep=',', ref='Cockpit Beechcraft',
                 phases={'Preflight':  (939484000, 1347500000),
                         'Taxi':       (1347500000, 1713507000),
                         'Takeoff':    (1713507000, 1927080000),
                         'Cruise':     (1927080000, 3125848000),
                         'Turbulence': (3125848000, 3291405000),
                         'Landing':    (3291405000, 3774557000)}),
}


def _locate(name):
    """Return the first existing path for `name` among the SRC folders."""
    for folder in SRC:
        candidate = os.path.join(folder, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f'{name} not found in {SRC}')


def _read_fixations(path, sep, ref):
    """Read a Tobii export and return its fixation samples with an AOI label."""
    mtype = f'Mapped eye movement type [{ref}]'
    midx = f'Mapped eye movement type index [{ref}]'

    header = pd.read_csv(path, sep=sep, nrows=0)
    aoi_cols = [c for c in header.columns if c.startswith(f'AOI hit [{ref} - ')]
    if not aoi_cols:
        raise RuntimeError(f'no AOI hit columns for reference image "{ref}" in {path}')
    aoi_names = [c.split(' - ')[-1].rstrip(']') for c in aoi_cols]

    df = pd.read_csv(path, sep=sep, low_memory=False,
                     usecols=['Recording timestamp', mtype, midx] + aoi_cols)
    fix = df[df[mtype] == 'Fixation'].copy()

    hits = fix[aoi_cols].fillna(0).astype(int).values

    def which(row):
        for i, v in enumerate(row):
            if v == 1:
                return aoi_names[i]
        return 'None'

    fix['aoi'] = [which(r) for r in hits]
    return fix, midx


def extract(flight, cfg):
    ref, sep = cfg['ref'], cfg['sep']
    per_phase = isinstance(cfg['file'], dict)

    if not per_phase:
        fix, midx = _read_fixations(_locate(cfg['file']), sep, ref)

    for phase, (t0, t1) in cfg['phases'].items():
        if per_phase:
            fix, midx = _read_fixations(_locate(cfg['file'][phase]), sep, ref)

        sub = fix[(fix['Recording timestamp'] >= t0) &
                  (fix['Recording timestamp'] < t1)]
        if sub.empty:
            print(f'{flight} {phase}: no fixations in interval')
            continue

        g = sub.groupby(midx).agg(
            tmin=('Recording timestamp', 'min'),
            tmax=('Recording timestamp', 'max'),
            n=('Recording timestamp', 'count'),
            aoi=('aoi', lambda s: s.mode()[0] if len(s.mode()) else 'None'))

        g['span'] = g['tmax'] - g['tmin']
        multi = g[g['n'] > 1]
        period = (multi['span'] / (multi['n'] - 1)).median() if len(multi) else 4000.0
        g['dur'] = g['span'] + period

        g = g.sort_values('tmin').reset_index(drop=True)
        g[['tmin', 'dur', 'aoi']].to_csv(f'fx_{flight}_{phase}.csv', index=False)

        in_aoi = (g['aoi'] != 'None').sum()
        print(f'{flight} {phase:11s} mapped fixations={len(g):5d} '
              f'in AOI={in_aoi:5d} sample period={period/1000:.2f} ms')


if __name__ == '__main__':
    for flight, cfg in FLIGHTS.items():
        extract(flight, cfg)
