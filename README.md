# Eye Conjunctiva Image Capture Tool (Work Package B)

A desktop application that guides users to capture eye conjunctiva photos, records a short video during the hold period, and automatically finds the frame where the **lower-eyelid palpebral conjunctiva is clearest**. A live **oval follows the eye** while you aim. For every capture the tool keeps **three candidate frames** (best sharpness, best exposure, lowest glare), picks the **single best frame** from them, prints the measured values on the images, **crops that single eye**, and organizes all outputs with standardized metadata in a structured dataset folder and CSV index.

Built with **Python**, **OpenCV** (colour analysis, scoring, Haar eye detection), **MediaPipe** (optional, precise eyelid landmarks) and **Playwright** (browser-based live camera UI).

---

## Features

- **Guided capture with a live, eye-following oval:** The live camera preview shows an oval that locks onto the eye whose lower-lid conjunctiva is reddest, rotates with head tilt, and changes colour to show whether the conjunctiva is clear. A configurable hold countdown (default 5 seconds) runs during recording.
- **Video recording:** The full hold period is recorded as a video (`.webm`).
- **Eye-based frame analysis:** Frames are judged on the **eye region** (the lower-eyelid conjunctiva strip), not on the whole oval. In each analysed frame both eyes are detected and the conjunctiva strip of each is measured for redness, sharpness, brightness and glare.
- **Red-visibility gate:** Only frames where the red palpebral conjunctiva is most visible (close to the best in the whole video) are eligible, so a sharp or dark frame without visible conjunctiva can never win.
- **Three candidate frames per capture:** The best-**sharpness** frame, the best-**exposure** frame and the lowest-**glare** frame are all stored.
- **Single best frame:** The best of the three candidates (highest overall conjunctiva score) becomes the final frame.
- **Values printed on the images:** Sharpness, brightness, glare (and red conjunctiva coverage) are printed on the final frame and on each candidate, together with the eye oval.
- **Single-eye crop:** The selected eye is cropped from the full-resolution frame (with padding) and saved clean (no text) alongside the full frame.
- **Automatic eye-detector fallback:** MediaPipe FaceMesh (if available), then the OpenCV Haar eye detector. An optional last-resort guess from the oval halves is available with `--allow-guess`.
- **Honest failure reporting:** If no eye is found in any frame, the review screen says so and no crop is saved. The three candidates are then chosen from the whole static oval and labelled accordingly.
- **Review screen:** Shows the final annotated frame, the three candidates as thumbnails (selected one highlighted, with their values), the cropped eye (small overlay) and a metrics line, so the user can choose **Keep** or **Discard**.
- **Fits the screen:** The browser opens maximized and the layout scales to the window, so no zooming out is needed.
- **Structured saving:** Video, final frame, three candidate frames and eye crop are saved under `dataset/<device>/<distance>/<lighting>/` with standardized filenames.
- **CSV index:** Each kept capture appends a row to `dataset.csv` with file paths, the measured values and eye-crop metadata.

---

## Project Structure

```
Task_B/
├── capture_app.py        # Main application script
├── requirements.txt      # Python dependencies (pinned)
├── README.md             # Project documentation
└── dataset/              # Created automatically on first run
    ├── dataset.csv
    └── <device>/<distance>/<lighting>/
        ├── videos/       # Recorded .webm videos
        ├── images/       # Final best frame, annotated (.png)
        ├── candidates/   # The 3 candidate frames, annotated (.png)
        └── crops/        # Single-eye conjunctiva crops, clean (.png)
```

---

## Requirements

- Python 3.9 or newer
- A webcam (or a phone connected as a webcam via apps like DroidCam or Iriun)
- A desktop environment with a display (the browser runs non-headless)
- Windows, macOS, or Linux
- Internet access for first-time installation only

### Python Dependencies

| Package | Version (pinned) | Purpose |
|---|---|---|
| `opencv-python` | `4.11.0.86` | Video decoding, colour analysis, scoring, image saving, Haar eye detector |
| `numpy` | `2.1.2` | Array operations for mask scoring |
| `playwright` | `1.64.0` | Launches the browser UI and bridges Python with HTML/JS |
| `mediapipe` | `1.1.1` | *Optional.* Precise eyelid landmarks (FaceMesh) for the eye crop and oval |

`requirements.txt`:

```
opencv-python==4.11.0.86
numpy==2.1.2
playwright==1.64.0
mediapipe==1.1.1
```

> **MediaPipe is optional.** The tool checks at startup whether MediaPipe provides the FaceMesh API it uses. If MediaPipe is missing or incompatible, the console prints `[info] mediapipe (legacy solutions API) not available -> using Haar eye detection.` and the tool continues with the Haar detector. If `mediapipe` fails to install, remove that line from `requirements.txt`.

---

## Setup

Run these commands in a terminal (PowerShell on Windows, Bash on macOS/Linux) from the project directory. Using a **clean virtual environment** is strongly recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
# source venv/bin/activate

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Download the browser binary used by Playwright (one-time setup)
python -m playwright install chromium
```

> **Note for Windows Users:** If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry.
> **Storage Tip:** If your project folder resides in OneDrive and you encounter file-lock or permission errors, move the project to a plain directory such as `C:\Task_B`.
> **OpenCV Note:** MediaPipe installs `opencv-contrib-python` as a dependency, and both it and `opencv-python` provide the `cv2` module. This normally works, but if you see `cv2` import errors, run `python -m pip uninstall -y opencv-python opencv-contrib-python` and then `python -m pip install opencv-contrib-python`.

---

## Usage

```bash
python capture_app.py --device <device_id> --distance <distance> --lighting <condition>
```

### Command Line Arguments

| Argument | Required | Description | Default / Example |
|---|---|---|---|
| `--device` | Yes | Device label used for folder and file names (metadata only) | `pixel7`, `laptop1` |
| `--distance` | Yes | Camera-to-eye distance label | `5cm`, `10cm` |
| `--lighting` | Yes | Lighting condition label | `indoor`, `daylight`, `flash` |
| `--duration` | No | Hold time in seconds | `5` |
| `--out` | No | Output root directory | `dataset` |
| `--stride` | No | Analyse every Nth frame (`1` = every frame, slower but most thorough) | `2` |
| `--allow-guess` | No | If no eye is detected, guess eye regions from the oval halves. A guess is never marked `crop_ok`. Not used by the live oval | Flag (off) |
| `--fake-camera` | No | Uses Chromium's simulated camera for testing without a physical webcam | Flag |

> **Format Rule:** Do **not** use spaces in `--device`, `--distance`, or `--lighting` string values because they form part of filenames and paths.

### Examples

```bash
# Test the full pipeline without a physical webcam
python capture_app.py --device test --distance 10cm --lighting indoor --fake-camera

# Standard capture session with real hardware
python capture_app.py --device pixel7 --distance 10cm --lighting indoor

# Analyse every frame for maximum thoroughness
python capture_app.py --device laptop1 --distance 10cm --lighting indoor --stride 1
```

> With `--fake-camera` the video is a synthetic test pattern, so no real eye is present. Expect the oval to stay dashed white and a "no eye could be found" warning. This is normal and confirms the rest of the pipeline works.

---

## User Guide

1. Execute the launch command. A maximized Chromium browser window opens automatically (your primary system browser remains untouched and camera permissions are auto-granted).
2. Gently pull the lower eyelid down so the palpebral conjunctiva (the red/pink inner surface) is visible, and bring the eye into view at the designated distance. Light the face from the front and avoid a bright wall or window behind you.
3. Watch the oval. It starts as a dashed white guide, then locks onto the eye (see the colour table below). Adjust the lid, angle and lighting until it turns green.
4. Click **Start capture**.
5. Hold steady while the countdown runs for the specified duration. The oval keeps following the eye.
6. Wait for the analysis phase ("Analysing frames...").
7. On the review screen, evaluate the final frame (values printed on it), the three candidate thumbnails (the selected one is highlighted), the cropped eye (small overlay in the bottom-right corner), and the metrics line:
   - **Keep:** Saves the `.webm` video, the final `.png`, the three candidate `.png` files, the eye-crop `.png`, and writes a metadata record to `dataset.csv`.
   - **Discard:** Purges temporary buffers without saving.
8. Repeat the process for additional captures. **Close the browser window when finished to terminate the script.**

### Live Oval Colours

| Oval | Meaning |
|---|---|
| Dashed white (centre of screen) | Searching. No eye detected yet. |
| Amber | Eye found, but the conjunctiva is not yet clear (low red coverage, glare, or poor exposure). |
| Green | Eye found and the conjunctiva passes the clarity check. |

### Review Screen Messages

| Message | Meaning | Recommended action |
|---|---|---|
| "no eye could be found in any frame" | No eye was detected anywhere in the recording. No crop is produced; the best whole-oval frames are shown for reference. | Retake with the eye in view and well lit. |
| "conjunctiva not clearly visible" | An eye was found but even the best conjunctiva strip failed the clarity check (low redness coverage, glare, or poor exposure). | Pull the lower lid down further, reduce glare, adjust lighting, retake. |
| "Best frame selected (from `<criterion>`). Keep it?" | The conjunctiva strip passed all checks. The criterion is the candidate (sharpness, exposure or glare) that won. | Keep. |

---

## Output Data Specifications

### Folder Structure & Naming Conventions

```
dataset/laptop1/10cm/indoor/
├── videos/laptop1_10cm_indoor_20261014T101500.webm
├── images/laptop1_10cm_indoor_20261014T101500.png
├── candidates/laptop1_10cm_indoor_20261014T101500_sharpness.png
├── candidates/laptop1_10cm_indoor_20261014T101500_exposure.png
├── candidates/laptop1_10cm_indoor_20261014T101500_glare.png
└── crops/laptop1_10cm_indoor_20261014T101500_eye.png
```

Standard Filename Pattern:
```
<device>_<distance>_<lighting>_<YYYYMMDDTHHMMSS>
```

The matching `image_id` is applied to the video, the final image, the candidates (with `_sharpness`, `_exposure`, `_glare` suffixes) and the crop (with an `_eye` suffix) for pairing. If two captures are kept within the same second, a `_2`, `_3`, ... suffix is added so nothing is overwritten.

### What is printed on the images

The final frame in `images/` and the three frames in `candidates/` are **annotated**:

- A title line (`FINAL BEST (<criterion>)` or `Best <criterion>`), with `[SELECTED]` on the winner.
- `Sharpness`, `Brightness`, `Glare` and `Red conj.` values measured on the conjunctiva strip.
- The eye oval: green if the conjunctiva passed the clarity check, amber otherwise.
- If no eye was found: the line `no eye found - whole-oval metrics`, and the values describe the whole static oval.

| Value | Definition |
|---|---|
| Sharpness | Laplacian variance inside the conjunctiva strip (higher = sharper) |
| Brightness | Mean gray level (0 to 255) inside the strip |
| Glare | Percentage of strip pixels that are very bright and unsaturated |
| Red conj. | Percentage of strip pixels classified as red/pink conjunctiva |

> **Use the crops for analysis.** The annotation covers part of the full frame, so the images in `images/` and `candidates/` are for review, not for pixel-level analysis. The files in `crops/` are saved clean (no text, no oval).

### `dataset.csv` Format

Each saved capture appends a row to `dataset.csv`:

| Column | Description |
|---|---|
| `image_id` | Unique identifier (filename without extension) |
| `video_path` | Relative path to the saved `.webm` video |
| `image_path` | Relative path to the final annotated `.png` |
| `sharpness_path` | Relative path to the best-sharpness candidate |
| `exposure_path` | Relative path to the best-exposure candidate |
| `glare_path` | Relative path to the lowest-glare candidate |
| `crop_path` | Relative path to the single-eye `.png` crop (empty if no eye was found) |
| `best_criterion` | Which candidate became the final frame: `sharpness`, `exposure` or `glare` |
| `best_frame_idx` | 1-based position of the final frame in the decoded recording |
| `sharpness` | Sharpness of the final frame (Laplacian variance in the strip) |
| `brightness` | Mean gray level (0 to 255) of the final frame's strip |
| `glare` | Glare fraction (0 to 1) of the final frame's strip |
| `eye_side` | `image_left`, `image_right`, or `single` (see note below) |
| `crop_method` | Detector used: `mediapipe`, `haar`, or `oval_halves` |
| `conj_coverage` | Fraction (0 to 1) of the lower-lid ROI that is red/pink |
| `conj_score` | Combined conjunctiva clarity score (0 to 1, higher is better) |
| `crop_ok` | `True` if the conjunctiva passed the clarity check, otherwise `False` |

```csv
image_id,video_path,image_path,sharpness_path,exposure_path,glare_path,crop_path,best_criterion,best_frame_idx,sharpness,brightness,glare,eye_side,crop_method,conj_coverage,conj_score,crop_ok
laptop1_10cm_indoor_20261014T101500,dataset/laptop1/10cm/indoor/videos/laptop1_10cm_indoor_20261014T101500.webm,dataset/laptop1/10cm/indoor/images/laptop1_10cm_indoor_20261014T101500.png,dataset/laptop1/10cm/indoor/candidates/laptop1_10cm_indoor_20261014T101500_sharpness.png,dataset/laptop1/10cm/indoor/candidates/laptop1_10cm_indoor_20261014T101500_exposure.png,dataset/laptop1/10cm/indoor/candidates/laptop1_10cm_indoor_20261014T101500_glare.png,dataset/laptop1/10cm/indoor/crops/laptop1_10cm_indoor_20261014T101500_eye.png,sharpness,64,182.400,131.200,0.00000,image_left,mediapipe,0.612,0.540,True
```

> **Eye side note:** `eye_side` is relative to the image, not the patient. `image_left` is the patient's right eye. The camera is not mirrored in the saved files. When the Haar detector finds only one eye, `eye_side` is `single`.
>
> **Candidates can differ in eye and frame:** each of the three candidates is chosen independently, so they may come from different frames, and even from different eyes. If one frame wins two criteria, it is saved twice under different candidate names.
>
> **Upgrading from an older version:** If an existing `dataset.csv` has an older column layout, the app automatically renames it to `dataset_old_<timestamp>.csv` at startup (nothing is overwritten) and starts a new `dataset.csv` with the current header.

---

## Frame Quality & Selection Methodology

Frames are selected by looking at the **eye**, not the whole oval. This avoids bright walls, faces, or backgrounds inside the oval causing good eye frames to be rejected.

### Stage 1: Frame sampling (pass 1)

The recording is decoded and every `--stride`-th frame (default every 2nd) is analysed. Colour conversions are done once per frame and shared by both eyes. In this pass only the measurements are kept (no images), so memory use stays low.

### Stage 2: Eye detection

For each analysed frame, both eyes are located using the first method that succeeds:

| Order | Method (`crop_method`) | Notes |
|---|---|---|
| 1 | **MediaPipe FaceMesh** (`mediapipe`) | Exact upper and lower eyelid landmarks. The conjunctiva strip and the oval follow head tilt. Only used if MediaPipe's FaceMesh API is available, and needs most of a face in view. |
| 2 | **OpenCV Haar eye cascade** (`haar`) | Detects eye boxes; the region directly below each box is used as the conjunctiva ROI. Used when MediaPipe is unavailable or finds no face. The oval is level (no tilt). |
| 3 | **Oval halves** (`oval_halves`) | Only with `--allow-guess`: left and right halves of the guide oval are treated as the two eyes. Never marked `crop_ok`. |

If no method finds an eye in a frame, that frame is skipped.

### Stage 3: Conjunctiva measurements

For each eye, a **lower-eyelid ROI** (a strip directly below the lower-lid margin, about 0.5x the eye width tall) is measured. A second strip further down (skin) is used as a reference:

| Component | Method | Default Parameter |
|---|---|---|
| **Redness coverage** | Fraction of ROI pixels with LAB `a*` above `max(CONJ_A_MIN, skin_a* + CONJ_A_DELTA)` and not too dark. The skin reference prevents plain reddish skin from counting as conjunctiva. | `CONJ_A_MIN = 145`, `CONJ_A_DELTA = 6`, `CONJ_L_MIN = 40` |
| **Sharpness** | Laplacian variance inside the ROI | `CONJ_SHARP_REF = 100` (used to normalize the score) |
| **Glare** | Bright, low-saturation pixel fraction inside the ROI | Penalty in score; limit `CONJ_GLARE_MAX = 0.02` |
| **Brightness / exposure** | Mean gray level inside the ROI | `BRIGHT_MIN = 70` to `BRIGHT_MAX = 190` (score halved if outside); ideal value `BRIGHT_TARGET = 130` |

The combined score is `coverage x normalized sharpness x glare penalty x exposure factor`.

`crop_ok` is `True` only if coverage is at least `CONJ_MIN_COVERAGE` (0.25), glare is within `CONJ_GLARE_MAX`, and exposure is within range.

### Stage 4: Red gate and the three candidates

1. **Red gate.** The highest red coverage found in the whole recording is taken as the reference. A frame is eligible only if its coverage is at least `RED_REL_GATE` (80%) of that reference. If the recording reaches `CONJ_MIN_COVERAGE` (25%), the 25% minimum also applies. This keeps only the frames where the red palpebral conjunctiva is most visible, even in poorly lit recordings where nothing reaches 25%.
2. **Candidates.** Among the eligible frames (each frame/eye combination counts separately):

| Candidate | Chosen as |
|---|---|
| **Sharpness** | Highest Laplacian variance in the conjunctiva strip |
| **Exposure** | Mean gray level closest to `BRIGHT_TARGET` (130) |
| **Glare** | Lowest glare. Glare is compared in 0.1% steps, and frames that tie (for example all with no glare) are decided by the better exposure, so a dark frame cannot win just because it has no highlights |

   Remaining ties are broken by the combined score.

### Stage 5: Choosing the single best frame (pass 2)

Only the three winning frames are read again from the video. The **final frame** is the candidate with the highest combined score, with candidates that pass all checks (`crop_ok = True`) always preferred over those that do not. The crop is taken from that full-resolution frame with 30% padding (`CROP_PAD`). The metrics shown in the UI and written to the CSV describe the winning conjunctiva strip.

**No-eye fallback:** If no eye is found in any frame, the three candidates are chosen the same way (sharpness / exposure / glare) from the whole static oval (using `GLARE_MAX_FRAC` and `CLIP_MAX_FRAC` for the exposure and clipping checks), shown for reference, and no crop is saved.

### Dynamic oval (live tracking)

While the camera preview is open, the browser sends a small copy of the live frame (640 px wide) to Python about 10 times per second. Python detects the eyes, measures the conjunctiva strip of each, and returns the oval of the eye with the **highest red coverage** (ties broken by the combined score).

| Detector | Oval shape |
|---|---|
| MediaPipe | Fitted to both eyelid outlines plus the top `OVAL_BAND_FRAC` (35%) of the lower-lid strip, scaled by `EYE_OVAL_SCALE` (1.15), and rotated to match head tilt |
| Haar | A level landscape oval centred on the detected eye (slightly low, to include the top of the lower lid) |

The oval is smoothed over time (`SMOOTH`), snaps to the new position if it jumps to the other eye, and disappears after `LOST_MAX` consecutive missed frames, when the dashed white hint returns. The same oval is drawn on the saved annotated images.

The oval **follows the eye shape** (eyelid landmarks), not the red pixels. Redness decides *which* eye gets the oval and its colour.

### Customizing Parameters

Constants can be adjusted at the top of `capture_app.py`:

```python
# Search hint oval (also used by the no-eye fallback), as a fraction of frame size
OVAL_RX, OVAL_RY = 0.30, 0.25
GLARE_MAX_FRAC = 0.01          # oval fallback only
CLIP_MAX_FRAC = 0.15           # oval fallback only

# Exposure range (applied to the conjunctiva ROI)
BRIGHT_MIN, BRIGHT_MAX = 70, 190
BRIGHT_TARGET = (BRIGHT_MIN + BRIGHT_MAX) / 2.0   # ideal brightness for the exposure candidate

# Eye / conjunctiva selection
FRAME_STRIDE = 2
CONJ_BAND_FRAC = 0.50
CONJ_A_MIN = 145
CONJ_A_DELTA = 6
CONJ_L_MIN = 40
CONJ_MIN_COVERAGE = 0.25
CONJ_GLARE_MAX = 0.02
CONJ_SHARP_REF = 100.0
RED_REL_GATE = 0.80            # frames must reach this share of the best red coverage to be eligible
CROP_PAD = 0.30

# Dynamic oval
EYE_OVAL_SCALE = 1.15          # margin around the eye (higher = looser oval)
OVAL_BAND_FRAC = 0.35          # how far the oval reaches into the lower lid (0 = eye only, 1 = whole strip)
```

and in the page script (inside the `HTML` string):

```javascript
const TRACK_W=640;   // width of the live frames sent to Python for tracking
const SMOOTH=0.55;   // 0..1, higher = oval follows faster, lower = steadier
const LOST_MAX=6;    // missed frames in a row before the oval is dropped
```

> **Tuning Notice:** The conjunctiva colour thresholds (`CONJ_A_MIN`, `CONJ_A_DELTA`, `CONJ_L_MIN`, `CONJ_MIN_COVERAGE`, `CONJ_GLARE_MAX`) are starting values. They depend on skin tone, lighting, and camera white balance. Tune them on real captures or replace them with the supervisor's criteria. The `Red conj.` value on the images and `R=` in the console help to see how much to change them.
>
> **Sync Notice:** `OVAL_RX` / `OVAL_RY` are passed into the page automatically, so the dashed search hint and the fallback mask stay in sync without editing the HTML.

---

## Console Output

After each analysis the console prints the selection result, for example:

```
[best] frames=148 selected=sharpness@64 sharp=182.4 bright=131 glare=0.00% eye=image_left/mediapipe conj=61% ok=True
       candidates: sharpness@64 S=182.4 B=131 G=0.00% R=61% | exposure@72 S=140.1 B=129 G=0.20% R=58% | glare@80 S=96.0 B=118 G=0.00% R=55%
```

The format is `<candidate>@<frame number>` followed by sharpness (S), brightness (B), glare (G) and red coverage (R). `eye=.../mediapipe` or `.../haar` tells you which detector was used.

---

## Pilot Dataset Collection

To collect data across multiple setups or devices into a unified dataset:

```powershell
python capture_app.py --device laptop1 --distance 10cm --lighting indoor
python capture_app.py --device phone_webcam --distance 10cm --lighting indoor
```

All runs append to the same central `dataset/dataset.csv`. Before executing production data collection, delete test subdirectories (e.g., `dataset/test/`) and remove associated `test_` entries from `dataset.csv`.

After a session, check the `crop_method`, `crop_ok` and `conj_coverage` columns to see which detector was used and how many crops passed the clarity check.

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `ModuleNotFoundError: No module named 'playwright'` | Playwright is not installed in the active environment. | Run `python -m pip install -r requirements.txt`. On Windows, ensure you are using the correct Python executable (`py -m pip ...`). |
| `playwright: command not found` | Chromium browser binary missing. | Execute `python -m playwright install chromium`. |
| `Camera error: NotFoundError` | No webcam detected or hardware is locked by another process (Zoom, Teams). | Close conflicting applications or run with `--fake-camera` to verify app logic. |
| `Camera error: NotAllowedError` | OS privacy permissions blocking browser access. | Enable camera permissions: **Windows Settings → Privacy → Camera** or **macOS System Settings → Privacy & Security → Camera**. |
| `No frames could be decoded` | Empty recording buffer or corrupted video stream. | Ensure OpenCV is installed (includes FFmpeg binaries). Retry the recording. |
| Permission / File-lock errors | System locking files inside synchronized folders. | Relocate project folder out of OneDrive or Google Drive to a standard local path. |
| `ImportError` or odd errors involving `cv2` | `opencv-python` and `opencv-contrib-python` conflict. | Use a clean virtual environment, or uninstall both and install only `opencv-contrib-python`. |
| Startup message: `mediapipe ... not available` | MediaPipe is not installed, or no longer provides the FaceMesh API. | Expected fallback. The tool uses the Haar detector. No action needed. |
| `[warn] mediapipe failed: ...` in the console | MediaPipe raised an error. | The tool falls back to Haar. Printed once per session; no action needed unless crops are consistently poor. |
| Startup message: `dataset.csv ... renamed to dataset_old_...csv` | The existing CSV used an older column layout. | Expected. The old file is kept untouched and a new `dataset.csv` was started. Merge them manually if needed. |
| Oval stays dashed white | No eye is detected in the live view. | Bring the eye into view, improve lighting, move slightly farther back so the eye and some surrounding face are visible. |
| Oval is amber and never turns green | Eye found, but the conjunctiva is not clear (low red coverage, glare, or exposure out of range). | Pull the lower lid down further, light the face from the front, reduce glare, then tune the `CONJ_*` constants if needed. |
| Oval jitters or lags | Tracking speed vs. smoothing trade-off. | Lower `SMOOTH` for a steadier oval, raise it for faster following. |
| "no eye could be found in any frame" | No eye detected in any analysed frame. | Bring the eye into view, improve lighting, move slightly farther back, or try `--stride 1`. |
| Selected frame looks dark or has no visible red lid | Backlit scene (bright wall or window behind) leaves the eye region very dark (brightness far below 70), and very dark pixels are not counted as red. | Light the face from the front or use the flash. If it still happens, lower `CONJ_L_MIN` or `CONJ_A_MIN`. Check the `Red conj.` value on the image. |
| `crop_method` is always `haar` | MediaPipe unavailable, or at close range it cannot see enough of the face. | Expected for tight close-ups. The crop and oval are rougher; review crops manually or adjust the distance. |
| Frequent "conjunctiva not clearly visible" | Lower lid not pulled down far enough, glare, or thresholds not matching your camera or lighting. | Improve technique and lighting, then tune the `CONJ_*` constants. |
| Analysis takes long | Long recording or `--stride 1` with MediaPipe. | Use the default `--stride 2` or a larger value, or shorten `--duration`. |
| Live preview feels slow | Eye tracking competes with the browser for CPU, especially with MediaPipe. | Lower `TRACK_W` (for example `480`) in the page script, or close other heavy applications. |
| UI does not fit the window | Browser zoom was changed. | Press `Ctrl` + `0` to reset zoom. |

To use an installed system instance of Google Chrome instead of Playwright's Chromium binary, modify the launch configuration in `capture_app.py`:

```python
browser = p.chromium.launch(channel="chrome", headless=False, args=chrome_args)
```

---

## Limitations

- **Platform Target:** Designed for desktop environments; mobile devices are supported exclusively as webcam feeds (via DroidCam, Iriun, etc.).
- **Container Format:** Video files are saved as `.webm` as generated by the browser's native `MediaRecorder` API.
- **Processing Time:** Eye detection runs on every analysed frame, so longer recordings, `--stride 1`, and MediaPipe increase processing time. The recording is read a second time to fetch the three winning frames.
- **Detector precision:** MediaPipe gives exact eyelid landmarks, but it needs most of a face in view. At very close range the Haar fallback is used, where crop boundaries, the conjunctiva strip and the oval are approximate and the oval cannot follow head tilt.
- **Oval follows the eye, not the red area:** The live oval is placed from eyelid landmarks (or the Haar box). Redness only selects which eye is shown and the oval's colour.
- **Heuristic conjunctiva check:** Clarity is estimated from colour, sharpness, glare, and exposure. It does not verify clinically that the palpebral conjunctiva is everted, and thresholds are not validated across skin tones or cameras.
- **Sharpness is not part of `crop_ok`:** A frame can pass the clarity check (coverage, glare, exposure) and still be soft. Check the sharpness value on the image or in the CSV.
- **Final frame is chosen among the three candidates only:** The final frame is the best of the three candidates, which are all drawn from the frames with the most visible red conjunctiva, not from every frame in the video.
- **Backlit or very dark scenes:** Under poor lighting the red coverage is underestimated, so frame ranking becomes less reliable. Good, frontal lighting matters more than any threshold.
- **Single eye per capture:** Only one eye crop is saved per capture; the other eye is visible only in the full-frame images.
- **Annotated full frames:** Images in `images/` and `candidates/` carry printed text and the oval; use the clean files in `crops/` for pixel-level analysis.
- **Camera aspect ratio:** The live oval is mapped correctly for any camera aspect ratio, but the dashed search hint is drawn in preview coordinates while the no-eye fallback mask uses frame coordinates, so the two can differ slightly when the camera is not 16:9 (the preview uses `object-fit: cover`).
