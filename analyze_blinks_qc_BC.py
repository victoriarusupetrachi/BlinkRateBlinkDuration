# -*- coding: utf-8 -*-
"""
Adapted from analyze_blinks_qc.py (original, by Marcus) — same detection,
duration-classification, and gaze-shift reclassification logic (150 ms
window, 8 deg threshold), run for BOTH eyes (original hardcoded EYE="right"
only) on the newly generated BC-phaseN-*.csv files (current, correct phase
boundaries from "Intervale zbor Baneasa Craiova.docx").
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

import blink

TRACK_GOOD = 70.0
TRACK_MODERATE = 40.0
T_MIN_BLINK = 50              # ms
T_MAX_BLINK = 500             # ms  -> clipire fiziologica
T_MAX_CLOSURE = 10000         # ms  -> peste = artefact de tracking

GAZE_SHIFT_DEG = 8.0
GAZE_WIN_MS = 150

settings = blink.Settings()
settings.Fs = 50
bd = blink.BlinkDetector(settings)


def classify_duration(dur_ms):
    if dur_ms < T_MIN_BLINK:
        return "too_short"
    if dur_ms <= T_MAX_BLINK:
        return "blink"
    if dur_ms <= T_MAX_CLOSURE:
        return "long_closure"
    return "artifact"


def tracking_ratio(df, eye):
    v = df[f"Validity {eye}"]
    n_samples = v.notna().sum()
    n_valid = (v == "Valid").sum()
    return (100.0 * n_valid / n_samples) if n_samples else 0.0


def _mean_gaze_dir(t, gaze, t0, t1):
    i0 = np.searchsorted(t, t0)
    i1 = np.searchsorted(t, t1)
    seg = gaze[i0:i1]
    seg = seg[~np.isnan(seg).any(axis=1)]
    if len(seg) == 0:
        return None
    v = np.nanmean(seg, axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def gaze_shift_deg(t, gaze, onset_ms, offset_ms):
    before = _mean_gaze_dir(t, gaze, onset_ms - GAZE_WIN_MS, onset_ms)
    after = _mean_gaze_dir(t, gaze, offset_ms, offset_ms + GAZE_WIN_MS)
    if before is None or after is None:
        return np.nan
    dot = float(np.clip(np.dot(before, after), -1.0, 1.0))
    return np.degrees(np.arccos(dot))


def process_file(path, eye):
    df = pd.read_csv(path, low_memory=False)
    pid = os.path.basename(path).split("-")[0]
    phase = os.path.basename(path)[:-4].split("-", 1)[1]

    pupil = np.array(df[f"Pupil diameter {eye}"], dtype=float)
    t = np.array(df["Recording timestamp"], dtype=float)
    t = (t - t[0]) / 1000.0
    rec_s = (t[-1] - t[0]) / 1000.0

    track = tracking_ratio(df, eye)

    pupil_valid = pupil[df[f"Validity {eye}"].values == "Valid"]
    pupil_mean = float(np.nanmean(pupil_valid)) if pupil_valid.size else np.nan
    pupil_std = float(np.nanstd(pupil_valid)) if pupil_valid.size else np.nan

    df_blinks = bd.blink_detector_pupil(
        t, pupil, settings.Fs,
        gap_dur=settings.gap_dur,
        min_dur=settings.min_blink_dur,
        remove_outliers=False,
        min_separation=settings.min_separation,
    )

    df_blinks["event_type"] = df_blinks["duration"].apply(classify_duration)

    gaze = np.c_[df[f"Gaze direction {eye} X"],
                 df[f"Gaze direction {eye} Y"],
                 df[f"Gaze direction {eye} Z"]].astype(float)
    df_blinks["gaze_shift_deg"] = [
        round(gaze_shift_deg(t, gaze, r.onset, r.offset), 1)
        for r in df_blinks.itertuples()
    ]
    shift = (df_blinks["gaze_shift_deg"] > GAZE_SHIFT_DEG) & \
            df_blinks["event_type"].isin(["blink", "long_closure"])
    df_blinks.loc[shift, "event_type"] = "gaze_shift"

    n_total = len(df_blinks)
    counts = df_blinks["event_type"].value_counts().to_dict()
    n_blink = counts.get("blink", 0)
    blinks = df_blinks[df_blinks.event_type == "blink"]

    if track >= TRACK_GOOD:
        quality = "good"
    elif track >= TRACK_MODERATE:
        quality = "moderate"
    else:
        quality = "low"

    row = {
        "pid": pid,
        "phase": phase,
        "eye": eye,
        "rec_s": round(rec_s, 1),
        "tracking_valid_pct": round(track, 1),
        "n_events_raw": n_total,
        "n_blink": n_blink,
        "n_gaze_shift": counts.get("gaze_shift", 0),
        "n_long_closure": counts.get("long_closure", 0),
        "n_artifact": counts.get("artifact", 0),
        "n_too_short": counts.get("too_short", 0),
        "blink_rate_min_RAW": round(60 * n_total / rec_s, 2) if rec_s else None,
        "blink_rate_min_CLEAN": round(60 * n_blink / rec_s, 2) if rec_s else None,
        "blink_dur_mean_ms": round(blinks["duration"].mean(), 1) if n_blink else None,
        "blink_dur_median_ms": round(blinks["duration"].median(), 1) if n_blink else None,
        "pupil_diam_mean_mm": round(pupil_mean, 3) if not np.isnan(pupil_mean) else None,
        "pupil_diam_std_mm": round(pupil_std, 3) if not np.isnan(pupil_std) else None,
        "tracking_quality": quality,
    }
    return row, df_blinks


def main():
    outdir = Path("final")
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob("data/experiment-BC/*.csv"))

    summary = []
    for f in files:
        for eye in ['left', 'right']:
            try:
                row, df_blinks = process_file(f, eye)
            except Exception as e:
                print(f"[skip] {f} ({eye}): {e}")
                continue
            summary.append(row)
            stem = os.path.basename(f)[:-4]
            df_blinks.to_csv(outdir / f"{stem}_{eye}_events.csv", index=False)

    summ = pd.DataFrame(summary)
    summ.to_csv(outdir / "summary_final_BC.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== TABEL FINAL BC (ambii ochi, granite de faza actualizate) ===\n")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
