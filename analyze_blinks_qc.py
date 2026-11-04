# -*- coding: utf-8 -*-
"""
Pipeline complet blink + QC pentru date Tobii Fusion (fara eye-openness).

Reface detectia din datele BRUTE (data/experiment-f1, -f2, -f3) folosind DOAR
ochiul drept (cel stang are tracking sub ~35% la toate fazele), apoi:
  1. calculeaza tracking ratio (% esantioane valide) per faza
  2. ruleaza detectorul pe pupila (blink_detector_pupil) pe pupila dreapta
  3. clasifica evenimentele pe durata: blink / long_closure / artifact / too_short
  4. marcheaza fazele cu tracking sub prag ca EXCLUSE (nesigure pentru blink rate)
  5. produce un singur tabel final, gata de pus in lucrare

Rezultat: results new/final/summary_final.csv
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

import blink

# ---- Praguri (ajustabile) -------------------------------------------------
EYE = "right"                 # ochiul stang e inutilizabil (lumina pe dreapta)
# Nu excludem nicio faza de zbor (sunt necesare pt. corelarea cu HR/HRV/EEG).
# tracking_valid_pct ramane doar ca indicator informativ de calitate.
TRACK_GOOD = 70.0             # >= good ; 40-70 moderate ; < 40 low  (doar flag)
TRACK_MODERATE = 40.0
T_MIN_BLINK = 50              # ms
T_MAX_BLINK = 500             # ms  -> clipire fiziologica
T_MAX_CLOSURE = 10000         # ms  -> peste = artefact de tracking

# Filtrare pe privire: daca directia privirii inainte vs dupa eveniment difera
# cu mai mult de GAZE_SHIFT_DEG, evenimentul e o privire deviata, nu o clipire.
GAZE_SHIFT_DEG = 8.0          # grade
GAZE_WIN_MS = 150             # fereastra pt. media directiei privirii in jurul evenimentului
# ---------------------------------------------------------------------------

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
    """% esantioane valide dintre esantioanele reale ale camerei ochiului."""
    v = df[f"Validity {eye}"]
    n_samples = v.notna().sum()          # randuri cu esantion de ochi
    n_valid = (v == "Valid").sum()
    return (100.0 * n_valid / n_samples) if n_samples else 0.0


def _mean_gaze_dir(t, gaze, t0, t1):
    """Media (normalizata) a vectorilor de directie a privirii in [t0, t1] ms."""
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
    """Unghiul (grade) intre directia privirii inainte de onset si dupa offset.
    Returneaza NaN daca nu exista date de privire valide de o parte/alta."""
    before = _mean_gaze_dir(t, gaze, onset_ms - GAZE_WIN_MS, onset_ms)
    after = _mean_gaze_dir(t, gaze, offset_ms, offset_ms + GAZE_WIN_MS)
    if before is None or after is None:
        return np.nan
    dot = float(np.clip(np.dot(before, after), -1.0, 1.0))
    return np.degrees(np.arccos(dot))


def process_file(path):
    df = pd.read_csv(path, low_memory=False)
    pid = os.path.basename(path).split("-")[0]
    phase = os.path.basename(path)[:-4].split("-", 1)[1]

    pupil = np.array(df[f"Pupil diameter {EYE}"], dtype=float)
    t = np.array(df["Recording timestamp"], dtype=float)
    t = (t - t[0]) / 1000.0                          # us -> ms
    rec_s = (t[-1] - t[0]) / 1000.0                  # secunde

    track = tracking_ratio(df, EYE)

    # Diametrul pupilei (mm) - doar ochiul drept, pe esantioanele valide.
    # Pupilele sunt consensuale, deci ochiul drept reprezinta si stangul.
    # nanmean exclude automat clipirile (unde pupila e NaN).
    pupil_valid = pupil[df[f"Validity {EYE}"].values == "Valid"]
    pupil_mean = float(np.nanmean(pupil_valid)) if pupil_valid.size else np.nan
    pupil_std = float(np.nanstd(pupil_valid)) if pupil_valid.size else np.nan

    # Detectie pe pupila (aceeasi metoda ca in run_classification.py)
    df_blinks = bd.blink_detector_pupil(
        t, pupil, settings.Fs,
        gap_dur=settings.gap_dur,
        min_dur=settings.min_blink_dur,
        remove_outliers=False,
        min_separation=settings.min_separation,
    )

    df_blinks["event_type"] = df_blinks["duration"].apply(classify_duration)

    # ---- Filtrare pe privire ------------------------------------------------
    gaze = np.c_[df[f"Gaze direction {EYE} X"],
                 df[f"Gaze direction {EYE} Y"],
                 df[f"Gaze direction {EYE} Z"]].astype(float)
    df_blinks["gaze_shift_deg"] = [
        round(gaze_shift_deg(t, gaze, r.onset, r.offset), 1)
        for r in df_blinks.itertuples()
    ]
    # Un eveniment clasat 'blink'/'long_closure' dar cu deviatie mare de privire
    # este de fapt o mutare a privirii (privire in afara camerei), nu o clipire.
    shift = (df_blinks["gaze_shift_deg"] > GAZE_SHIFT_DEG) & \
            df_blinks["event_type"].isin(["blink", "long_closure"])
    df_blinks.loc[shift, "event_type"] = "gaze_shift"
    # ------------------------------------------------------------------------

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
        "blink_dur_median_ms": round(blinks["duration"].median(), 1) if n_blink else None,
        "blink_dur_mean_ms": round(blinks["duration"].mean(), 1) if n_blink else None,
        "pupil_diam_mean_mm": round(pupil_mean, 3) if not np.isnan(pupil_mean) else None,
        "pupil_diam_std_mm": round(pupil_std, 3) if not np.isnan(pupil_std) else None,
        "tracking_quality": quality,   # informativ; NICIO faza nu e exclusa
    }
    return row, df_blinks


def main():
    outdir = Path("results new/final")
    outdir.mkdir(parents=True, exist_ok=True)

    files = []
    for exp in sorted(glob.glob("data/experiment-f*")):
        files += sorted(glob.glob(exp + "/*.csv"))

    summary = []
    for f in files:
        try:
            row, df_blinks = process_file(f)
        except Exception as e:
            print(f"[skip] {f}: {e}")
            continue
        summary.append(row)
        stem = os.path.basename(f)[:-4]
        df_blinks.to_csv(outdir / f"{stem}_{EYE}_events.csv", index=False)

    summ = pd.DataFrame(summary)
    summ.to_csv(outdir / "summary_final.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(f"\n=== TABEL FINAL (ochi={EYE}, TOATE fazele pastrate) ===\n")
    print(summ.to_string(index=False))

    print("\n--- Sinteza pe toate fazele de zbor ---")
    print(f"blink rate curat: {summ.blink_rate_min_CLEAN.min():.1f} - "
          f"{summ.blink_rate_min_CLEAN.max():.1f}/min (median {summ.blink_rate_min_CLEAN.median():.1f})")
    print(f"durata mediana blink: {summ.blink_dur_median_ms.min():.0f} - "
          f"{summ.blink_dur_median_ms.max():.0f} ms")
    print(f"durata medie blink:   {summ.blink_dur_mean_ms.min():.0f} - "
          f"{summ.blink_dur_mean_ms.max():.0f} ms")
    print("calitate tracking:", summ.tracking_quality.value_counts().to_dict())
    print(f"\nFaze: {len(summ)} (toate pastrate)   -> {outdir/'summary_final.csv'}")


if __name__ == "__main__":
    main()
