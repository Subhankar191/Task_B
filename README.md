# Eye Conjunctiva Image Capture Tool (Work Package B)

A desktop application that guides users to capture eye conjunctiva photos, records a short video during the hold period, automatically finds the frame where the **lower-eyelid palpebral conjunctiva is clearest**, **crops that single eye** (of the two), and organizes all saved outputs with standardized metadata in a structured dataset folder and CSV index.

Built with **Python**, **OpenCV** (colour analysis, scoring, Haar eye detection), **MediaPipe** (optional, precise eyelid landmarks) and **Playwright** (browser-based live camera UI).

---

## Features

- **Guided capture:** Live camera preview with an on-screen oval alignment guide and a configurable hold countdown (default 5 seconds).
- **Video recording:** The full hold period is recorded as a video (`.webm`).
- **Eye-based best-frame selection:** Frames are judged on the **eye region** (the lower-eyelid conjunctiva strip), not on the whole oval. In each analysed frame both eyes are detected and the conjunctiva strip of each is scored for redness, sharpness, glare and exposure. The best eye in the best frame is selected.
- **Single-eye crop:** The selected eye is cropped from the full-resolution frame (with padding) and saved alongside the full frame.
- **Automatic eye-detector fallback:** MediaPipe FaceMesh (if available), then the OpenCV Haar eye detector. An optional last-resort guess from the oval halves is available with `--allow-guess`.
- **Honest failure reporting:** If no eye is found in any frame, the review screen says so and no crop is saved.
- **Review screen:** Shows the selected frame, the cropped eye (as a small overlay), and metrics (sharpness, brightness, glare of the conjunctiva strip, conjunctiva coverage and score) so the user can choose **Keep** or **Discard**.
- **Fits the screen:** The browser opens maximized and the layout scales to the window, so no zooming out is needed.
- **Structured saving:** Video, full-frame image, and eye crop are saved under `dataset/<device>/<distance>/<lighting>/` with standardized filenames.
- **CSV index:** Each kept capture appends a row to `dataset.csv` with file paths and eye-crop metadata.

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
        ├── images/       # Selected best full frames (.png)
        └── crops/        # Single-eye conjunctiva crops (.png)
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
| `mediapipe` | `1.1.1` | *Optional.* Precise eyelid landmarks (FaceMesh) for the eye crop |

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
| `--allow-guess` | No | If no eye is detected, guess eye regions from the oval halves. A guess is never marked `crop_ok` | Flag (off) |
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

> With `--fake-camera` the video is a synthetic test pattern, so no real eye is present. Expect a "no eye could be found" warning. This is normal and confirms the rest of the pipeline works.

---

## User Guide

1. Execute the launch command. A maximized Chromium browser window opens automatically (your primary system browser remains untouched and camera permissions are auto-granted).
2. Gently pull the lower eyelid down so the palpebral conjunctiva (the red/pink inner surface) is visible, and align the eye inside the green dashed oval guide at the designated distance.
3. Click **Start capture**.
4. Hold steady while the countdown runs for the specified duration.
5. Wait for the analysis phase ("Analysing frames...").
6. On the review screen, evaluate the chosen frame, the cropped eye (small overlay in the bottom-right corner of the image), and the metrics line:
   - **Keep:** Saves the `.webm` video, the full-frame `.png`, the eye-crop `.png`, and writes a metadata record to `dataset.csv`.
   - **Discard:** Purges temporary buffers without saving.
7. Repeat the process for additional captures. **Close the browser window when finished to terminate the script.**

### Review Screen Messages

| Message | Meaning | Recommended action |
|---|---|---|
| "no eye could be found in any frame" | No eye was detected anywhere in the recording. No crop is produced; the best whole-oval frame is shown for reference. | Retake with the eye centred in the oval and well lit. |
| "conjunctiva not clearly visible" | An eye was found but even the best conjunctiva strip failed the clarity check (low redness coverage, glare, or poor exposure). | Pull the lower lid down further, reduce glare, adjust lighting, retake. |
| "Best frame and eye selected" | The conjunctiva strip passed all checks. | Keep. |

---

## Output Data Specifications

### Folder Structure & Naming Conventions

```
dataset/laptop1/10cm/indoor/
├── videos/laptop1_10cm_indoor_20261014T101500.webm
├── images/laptop1_10cm_indoor_20261014T101500.png
└── crops/laptop1_10cm_indoor_20261014T101500_eye.png
```

Standard Filename Pattern:
```
<device>_<distance>_<lighting>_<YYYYMMDDTHHMMSS>
```

The matching `image_id` is applied to the video, the full image, and the crop (with an `_eye` suffix) for pairing. If two captures are kept within the same second, a `_2`, `_3`, ... suffix is added so nothing is overwritten.

### `dataset.csv` Format

Each saved capture appends a row to `dataset.csv`:

| Column | Description |
|---|---|
| `image_id` | Unique identifier (filename without extension) |
| `video_path` | Relative path to the saved `.webm` video |
| `image_path` | Relative path to the selected full-frame `.png` |
| `crop_path` | Relative path to the single-eye `.png` crop (empty if no eye was found) |
| `eye_side` | `image_left`, `image_right`, or `single` (see note below) |
| `crop_method` | Detector used: `mediapipe`, `haar`, or `oval_halves` |
| `conj_coverage` | Fraction (0 to 1) of the lower-lid ROI that is red/pink |
| `conj_score` | Combined conjunctiva clarity score (0 to 1, higher is better) |
| `crop_ok` | `True` if the conjunctiva passed the clarity check, otherwise `False` |

```csv
image_id,video_path,image_path,crop_path,eye_side,crop_method,conj_coverage,conj_score,crop_ok
laptop1_10cm_indoor_20261014T101500,dataset/laptop1/10cm/indoor/videos/laptop1_10cm_indoor_20261014T101500.webm,dataset/laptop1/10cm/indoor/images/laptop1_10cm_indoor_20261014T101500.png,dataset/laptop1/10cm/indoor/crops/laptop1_10cm_indoor_20261014T101500_eye.png,image_left,mediapipe,0.612,0.540,True
```

> **Eye side note:** `eye_side` is relative to the image, not the patient. `image_left` is the patient's right eye. The camera is not mirrored in the saved files. When the Haar detector finds only one eye, `eye_side` is `single`.
>
> **Upgrading from an older version:** If an existing `dataset.csv` has an older header (for example the original 3-column one), the app prints a warning at startup. Rename or move the old file so a new one with the current header is created.

---

## Frame Quality & Selection Methodology

Frames are selected by looking at the **eye**, not the whole oval. This avoids bright walls, faces, or backgrounds inside the oval causing good eye frames to be rejected.

### Stage 1: Frame sampling

The recording is decoded and every `--stride`-th frame (default every 2nd) is analysed. Colour conversions are done once per frame and shared by both eyes.

### Stage 2: Eye detection

For each analysed frame, both eyes are located using the first method that succeeds:

| Order | Method (`crop_method`) | Notes |
|---|---|---|
| 1 | **MediaPipe FaceMesh** (`mediapipe`) | Exact upper and lower eyelid landmarks. The conjunctiva strip follows head tilt. Only used if MediaPipe's FaceMesh API is available, and needs most of a face in view. |
| 2 | **OpenCV Haar eye cascade** (`haar`) | Detects eye boxes; the region directly below each box is used as the conjunctiva ROI. Used when MediaPipe is unavailable or finds no face. |
| 3 | **Oval halves** (`oval_halves`) | Only with `--allow-guess`: left and right halves of the guide oval are treated as the two eyes. Never marked `crop_ok`. |

If no method finds an eye in a frame, that frame is skipped.

### Stage 3: Conjunctiva clarity

For each eye, a **lower-eyelid ROI** (a strip directly below the lower-lid margin, about 0.5x the eye width tall) is scored. A second strip further down (skin) is used as a reference:

| Component | Method | Default Parameter |
|---|---|---|
| **Redness coverage** | Fraction of ROI pixels with LAB `a*` above `max(CONJ_A_MIN, skin_a* + CONJ_A_DELTA)` and not too dark. The skin reference prevents plain reddish skin from counting as conjunctiva. | `CONJ_A_MIN = 145`, `CONJ_A_DELTA = 6`, `CONJ_L_MIN = 40` |
| **Sharpness** | Laplacian variance inside the ROI, normalized | `CONJ_SHARP_REF = 100` |
| **Glare** | Bright, low-saturation pixel fraction inside the ROI | Penalty in score; limit `CONJ_GLARE_MAX = 0.02` |
| **Exposure** | Mean gray level inside the ROI | `BRIGHT_MIN = 70` to `BRIGHT_MAX = 190` (score halved if outside) |

The combined score is `coverage x normalized sharpness x glare penalty x exposure factor`.

`crop_ok` is `True` only if coverage is at least `CONJ_MIN_COVERAGE` (0.25), glare is within `CONJ_GLARE_MAX`, and exposure is within range.

### Stage 4: Choosing the result

Across all analysed frames and both eyes, the best candidate is chosen, with eyes that pass all checks (`crop_ok = True`) always preferred over those that do not, and the higher score winning within each group. The output frame is the one containing the winning eye, and the crop is taken from that full-resolution frame with 30% padding (`CROP_PAD`). The metrics shown in the UI (sharpness, brightness, glare) describe the winning conjunctiva strip.

**No-eye fallback:** If no eye is found in any frame, the tool falls back to the best whole-oval frame (scored on sharpness, exposure, clipping and glare inside the oval, using `GLARE_MAX_FRAC` and `CLIP_MAX_FRAC`), shows it for reference, and saves no crop.

### Customizing Parameters

Constants can be adjusted at the top of `capture_app.py`:

```python
# Oval (also used by the no-eye fallback)
OVAL_RX, OVAL_RY = 0.30, 0.25
GLARE_MAX_FRAC = 0.01          # oval fallback only
CLIP_MAX_FRAC = 0.15           # oval fallback only

# Exposure range (applied to the conjunctiva ROI)
BRIGHT_MIN, BRIGHT_MAX = 70, 190

# Eye / conjunctiva selection
FRAME_STRIDE = 2
CONJ_BAND_FRAC = 0.50
CONJ_A_MIN = 145
CONJ_A_DELTA = 6
CONJ_L_MIN = 40
CONJ_MIN_COVERAGE = 0.25
CONJ_GLARE_MAX = 0.02
CONJ_SHARP_REF = 100.0
CROP_PAD = 0.30
```

> **Tuning Notice:** The conjunctiva colour thresholds (`CONJ_A_MIN`, `CONJ_A_DELTA`, `CONJ_MIN_COVERAGE`, `CONJ_GLARE_MAX`) are starting values. They depend on skin tone, lighting, and camera white balance. Tune them on real captures or replace them with the supervisor's criteria.
>
> **Sync Notice:** If you alter the relative dimensions `OVAL_RX` / `OVAL_RY`, update the corresponding `rx` and `ry` SVG ellipse attributes in the HTML template so the visual UI alignment guide and the OpenCV fallback mask remain in sync.

---

## Pilot Dataset Collection

To collect data across multiple setups or devices into a unified dataset:

```powershell
python capture_app.py --device laptop1 --distance 10cm --lighting indoor
python capture_app.py --device phone_webcam --distance 10cm --lighting indoor
```

All runs append to the same central `dataset/dataset.csv`. Before executing production data collection, delete test subdirectories (e.g., `dataset/test/`) and remove associated `test_` entries from `dataset.csv`.

After a session, check the `crop_method` and `crop_ok` columns to see which detector was used and how many crops passed the clarity check.

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
| "no eye could be found in any frame" | No eye detected in any analysed frame. | Centre the eye in the oval, improve lighting, move slightly farther back so the eye and some surrounding face are visible, or try `--stride 1`. |
| `crop_method` is always `haar` | MediaPipe unavailable, or at close range it cannot see enough of the face. | Expected for tight close-ups. The crop is rougher; review crops manually or adjust the distance. |
| Frequent "conjunctiva not clearly visible" | Lower lid not pulled down far enough, glare, or thresholds not matching your camera or lighting. | Improve technique and lighting, then tune the `CONJ_*` constants. |
| Analysis takes long | Long recording or `--stride 1` with MediaPipe. | Use the default `--stride 2` or a larger value, or shorten `--duration`. |
| UI does not fit the window | Browser zoom was changed. | Press `Ctrl` + `0` to reset zoom. |
| Warning about older `dataset.csv` header | Existing CSV was created by an earlier version. | Rename or move the old `dataset.csv` so the new header is written. |

To use an installed system instance of Google Chrome instead of Playwright's Chromium binary, modify the launch configuration in `capture_app.py`:

```python
browser = p.chromium.launch(channel="chrome", headless=False, args=chrome_args)
```

---

## Limitations

- **Platform Target:** Designed for desktop environments; mobile devices are supported exclusively as webcam feeds (via DroidCam, Iriun, etc.).
- **Container Format:** Video files are saved as `.webm` as generated by the browser's native `MediaRecorder` API.
- **Processing Time:** Eye detection runs on every analysed frame, so longer recordings, `--stride 1`, and MediaPipe increase processing time.
- **Detector precision:** MediaPipe gives exact eyelid landmarks, but it needs most of a face in view. At very close range the Haar fallback is used, where crop boundaries and the conjunctiva strip are approximate.
- **Heuristic conjunctiva check:** Clarity is estimated from colour, sharpness, glare, and exposure. It does not verify clinically that the palpebral conjunctiva is everted, and thresholds are not validated across skin tones or cameras.
- **Single eye per capture:** Only one eye crop is saved per capture; the other eye is visible only in the full-frame image.
- **Oval alignment:** The on-screen oval and the fallback mask can differ slightly if the camera is not 16:9, because the preview uses `object-fit: cover`.
