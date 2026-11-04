# BlinkRateBlinkDuration — blink and pupil analysis for pilot eye-tracking recordings

This repository contains the analysis code for the study

> Rusu, V.; Calefariu, G.; Lom, I. *Pilot Attention and Vigilance in Full-Motion Simulation and
> Operational Flight: An Eye-Tracking Case Comparison Within the Human Performance Envelope
> Framework.* Applied Sciences (under review).

It is a fork of the `BlinkDetector` package released by Nyström and colleagues, which analyses the
performance of blink detection based on two sample data sets. One of the dataset recorded with the Tobii 
Pro Spectum at 600 Hz and the second one Tobii Pro Fusion at 120 Hz data frequency.

The algorithm was adapted to perform the blink detection on the data recorded with **Tobii Pro Glasses 2 at 50 Hz**,
with sample data collected during three full-motion simulator sessions and one operational flight in a Beechcraft 
King Air C90GTx.

The algorithm computes the blink-related and pupil-related values reported in **Table 7** of the
manuscript. Additionally, was extended the logic to also get the fixation-share, bootstrap, scanpath-entropy and 30-second-window analyses reported in **Tables 4, 5 and 6**.

---

## 1. Relation to the original BlinkDetector

The upstream package detects blinks in the **eye-openness signal** produced by Tobii Pro
Spectrum and Tobii Pro Fusion eye trackers:

> Nyström, M.; Andersson, R.; Niehorster, D. C.; Hessels, R. S.; Hooge, I. T. C. What is a blink?
> Classifying and characterizing blinks in eye openness signals. *Behavior Research Methods* 2024.
> https://github.com/marcus-nystrom/BlinkDetector

**Tobii Pro Glasses 2 does not provide an eye-openness signal.** Only the
`blink_detector_pupil()` branch of the original algorithm is therefore used here, which
identifies blinks as interruptions of the pupil-diameter signal. The eye-openness branch
(`blink_detector_eo()`) is retained in `blink.py` unchanged but is not called.

Because pupil-signal loss is not specific to blinks — it also occurs when the pilot looks
outside the camera's field of view, when the glasses shift, or when the tracker loses the eye —
a **two-stage post-detection classification** was added in this fork. That classification, the
quality-control reporting, and the AOI analyses are the contributions of this repository; the
detector itself is unchanged from the upstream release.

The adaptation of the detector to the sampling rate and data format of the Tobii Pro Glasses 2
was carried out by Diana Patraș (Software Engineer), who also independently reviewed and verified
the bootstrap, entropy and window-based analysis scripts.

---

## 2. Requirements

```
Python 3.10 (blink analysis) / Python 3.11 (AOI analysis)
numpy
scipy
pandas
```

```bash
pip install -r requirements.txt
```

---

## 3. Input data

The scripts read **sample-level data exports from Tobii Pro Lab**, one CSV per flight phase,
comma-separated, at 50 Hz. The raw recordings are not included in this repository: they contain
cockpit video and participant-related visual data and cannot be released for privacy reasons.
The column names below are those produced by Tobii Pro Lab and are the only requirement for
running the code on another dataset.

**Blink and pupil analysis** (`analyze_blinks_qc.py`):

| Column | Use |
|---|---|
| `Recording timestamp` | time base, microseconds |
| `Pupil diameter left` / `Pupil diameter right` | blink detection and pupil statistics |
| `Validity left` / `Validity right` | data-coverage computation |
| `Gaze direction left X/Y/Z`, `Gaze direction right X/Y/Z` | gaze-shift filter |

**AOI analysis** (`aoi_fixation_extract.py`):

| Column | Use |
|---|---|
| `Recording timestamp` | time base |
| `Mapped eye movement type [<reference image>]` | selects fixation samples |
| `Mapped eye movement type index [<reference image>]` | groups samples into fixations |
| `AOI hit [<reference image> - <AOI name>]` | one column per AOI |

Phase boundaries are given in recording-timestamp units in the configuration block at the top
of each script.

---

## 4. Pipeline

```
Tobii Pro Lab export (50 Hz, one CSV per flight phase)
        │
        ├── analyze_blinks_qc.py
        │        │
        │        ├── blink.py  →  blink_detector_pupil()      detection
        │        ├── stage 1: duration-based classification
        │        ├── stage 2: gaze-shift reclassification
        │        ├── data coverage and tracking-quality label
        │        └── pupil-diameter mean and SD on valid samples
        │                 │
        │                 ├── <phase>_<eye>_events.csv        one row per event
        │                 └── summary_final.csv
        │
        └── aoi_fixation_extract.py  →  fx_<FLIGHT>_<PHASE>.csv
                 │
                 └── aoi_variability_analysis.py
                         ├── bootstrap_all.csv
                         ├── entropy_all.csv
                         └── window30s_all.csv
```

### 4.1 Detection (`blink.py`, unchanged from upstream)

`blink_detector_pupil()` interpolates gaps in the pupil signal shorter than `gap_dur`, treats
the remaining missing-data segments as candidate events, discards candidates whose onset or
offset falls inside a sample-level recording gap, merges candidates separated by less than
`min_separation`, and rejects those shorter than `min_dur`.

Settings used (`blink.Settings`, with `Fs` set to 50 in the analysis scripts):

| Parameter | Value | Meaning |
|---|---|---|
| `Fs` | 50 Hz | sampling rate of Tobii Pro Glasses 2 |
| `gap_dur` | 40 ms | gaps shorter than this are interpolated, not treated as events |
| `min_blink_dur` | 30 ms | candidates shorter than this are rejected |
| `min_separation` | 100 ms | candidates closer than this are merged into one event |
| `remove_outliers` | `False` | no pupil-size outlier removal applied |

### 4.2 Stage 1 — duration-based classification

Each detected event is assigned to one of four classes by its duration:

| Class | Duration | Treatment |
|---|---|---|
| `too_short` | < 50 ms | noise, excluded |
| `blink` | 50–500 ms | physiological spontaneous blink, **retained** |
| `long_closure` | 500–10 000 ms | prolonged closure or microsleep, excluded |
| `artifact` | > 10 000 ms | tracking loss, excluded |

### 4.3 Stage 2 — gaze-shift reclassification

The detector cannot distinguish a blink from a large gaze movement that takes the pupil out of
the camera's view. For every event classified as `blink` or `long_closure`, the mean normalised
gaze-direction vector over the **150 ms preceding the onset** is compared with the mean vector
over the **150 ms following the offset**. If the angle between them exceeds **8°**, the event is
reclassified as `gaze_shift` and excluded from the blink count.

### 4.4 Data coverage and tracking quality

Data coverage is the percentage of eye-camera samples labelled `Valid` by Tobii Pro Lab within a
phase, computed separately for each eye. Each phase and eye receives a categorical label:

| Label | Coverage |
|---|---|
| `good` | ≥ 70% |
| `moderate` | 40–69% |
| `low` | < 40% |

No phase is excluded on the basis of coverage; the label is reported alongside every value so
that readers can weight it accordingly. In this dataset, left-eye coverage in the simulator was
consistently low because the pilot sat on the left while all illuminated displays were on the
right, so left-eye values are reported for completeness only.

### 4.5 Blink rate

Blink rate is computed from the retained events only, as Equation (3) of the manuscript:

```
blink_rate_min_CLEAN = 60 * n_blink / rec_s
```

where `n_blink` is the number of events retained after both classification stages and `rec_s`
is the duration of the phase in seconds. `blink_rate_min_RAW`, computed from all detected
events before classification, is reported alongside it for transparency and is **not** the value
used in the manuscript.

---

## 5. Running the code

```bash
# simulator flights: reads data/experiment-f1, -f2, -f3 and -BC
python analyze_blinks_qc.py

# AOI fixation extraction, then variability and scanpath analyses
python aoi_fixation_extract.py
python aoi_variability_analysis.py
```

Each script is run from the repository root and writes its outputs to the folders shown in the
pipeline diagram above. The AOI scripts use a fixed random seed (42), so the bootstrap
confidence intervals are reproduced exactly on repeated runs.

---

## 6. Output files

### `summary_final.csv` — one row per phase and eye

| Column | Meaning |
|---|---|
| `pid` | recording identifier (F1, F2, F3, BC) |
| `phase` | flight phase |
| `eye` | `left` or `right` |
| `rec_s` | phase duration, seconds |
| `tracking_valid_pct` | data coverage, % of valid samples |
| `n_events_raw` | all events returned by the detector |
| `n_blink` | events retained as physiological blinks |
| `n_gaze_shift` | events reclassified by the gaze-shift filter |
| `n_long_closure` | events of 500–10 000 ms |
| `n_artifact` | events longer than 10 000 ms |
| `n_too_short` | events shorter than 50 ms |
| `blink_rate_min_RAW` | 60 × `n_events_raw` / `rec_s` (diagnostic only) |
| `blink_rate_min_CLEAN` | 60 × `n_blink` / `rec_s` — **the value reported in Table 7** |
| `blink_dur_mean_ms` | mean duration of retained blinks |
| `blink_dur_median_ms` | median duration of retained blinks |
| `pupil_diam_mean_mm` | mean pupil diameter over valid samples |
| `pupil_diam_std_mm` | standard deviation of pupil diameter over valid samples |
| `tracking_quality` | `good`, `moderate` or `low` |

### `<phase>_<eye>_events.csv` — one row per detected event

| Column | Meaning |
|---|---|
| `onset`, `offset` | event boundaries, ms from the start of the phase |
| `duration` | event duration, ms |
| `event_type` | `blink`, `gaze_shift`, `long_closure`, `artifact` or `too_short` |
| `gaze_shift_deg` | angle between the mean gaze direction before and after the event |

The standard deviation of blink duration reported in Table 7 is computed over the rows of these
files with `event_type == "blink"`.

### `bootstrap_all.csv`, `entropy_all.csv`, `window30s_all.csv`

AOI fixation shares with circular moving-block bootstrap confidence intervals (2 000 resamples;
block lengths of 5, 10 and 20 fixations), Shannon entropy of the AOI-to-AOI transition
distribution, and mean and standard deviation of fixation share across consecutive
non-overlapping 30-second windows. Fixations falling outside all AOIs are excluded throughout;
AOIs with fewer than five fixations in a phase are not reported; transitions remaining within
the same AOI are included in the entropy calculation.

---

## 7. Reproducing Table 7 of the manuscript

| Table 7 column | Source |
|---|---|
| Phase duration (s) | `rec_s` |
| Blinks (n) | `n_blink` |
| Blink rate (blinks/min) | `blink_rate_min_CLEAN` |
| Blink duration (ms, mean ± SD) | `blink_dur_mean_ms`, SD from the per-phase events file |
| Pupil diameter (mm, mean ± SD) | `pupil_diam_mean_mm`, `pupil_diam_std_mm` |
| Data coverage (%) | `tracking_valid_pct` |
| Tracking quality | `tracking_quality` |

---

## 8. Known limitations of the procedure

- Detection relies on pupil-signal loss rather than on a direct measure of eye openness, so
  partial blinks that do not interrupt the pupil signal are not detected.
- The detected events were **not** compared against manually annotated blinks. The two
  classification stages remove the event types that are known to contaminate pupil-based
  detection, but no event-by-event validation was performed and the retained events are not
  described as artefact-free.
- Identical thresholds were applied to both eyes. Where data coverage is low, the number of
  retained events depends on data quality as much as on oculomotor behaviour.
- At 50 Hz the sampling interval is 20 ms, so differences in mean blink duration smaller than
  this should not be interpreted as physiological changes.

---

## 9. Citation

If you use this code, please cite both the original algorithm and the present study:

```bibtex
@article{nystrom2024blink,
  title   = {What is a blink? Classifying and characterizing blinks in eye openness signals},
  author  = {Nystr{\"o}m, Marcus and Andersson, Richard and Niehorster, Diederick C.
             and Hessels, Roy S. and Hooge, Ignace T. C.},
  journal = {Behavior Research Methods},
  year    = {2024},
  doi     = {10.3758/s13428-023-02333-9}
}
```

The reference for the present study will be added once the article is published.

---

## 10. License

The original BlinkDetector code is redistributed under its own licence; see `LICENSE`. The
scripts added in this fork are released under the same terms.

```
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
