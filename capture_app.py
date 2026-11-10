#!/usr/bin/env python3
"""
Conjunctiva image capture: guided capture -> video -> eye-based frame analysis
-> 3 candidate frames (best sharpness / best exposure / lowest glare) -> best single frame
-> single-eye crop (clearest lower-eyelid palpebral conjunctiva) -> structured save + CSV.

Frames are judged on the EYE region (lower-lid conjunctiva strip), not on the whole oval.
A live oval follows the eye whose lower-lid conjunctiva is reddest (replaces the static oval).

Usage:
  python capture_app.py --device pixel7 --distance 10cm --lighting indoor
  python capture_app.py --device test --distance 10cm --lighting indoor --fake-camera

Eye detection order: MediaPipe FaceMesh (if available) -> OpenCV Haar -> (optional) oval-halves guess.
"""
import argparse
import base64
import csv
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

try:
    import mediapipe as mp
    _HAS_MP = hasattr(mp, "solutions")
except Exception:  # mediapipe not installed or incompatible
    mp = None
    _HAS_MP = False

# Static search-guide / fallback oval, as a fraction of frame size.
# Only used (a) as the hint shown until an eye is found and (b) when no eye is found in any frame.
OVAL_RX, OVAL_RY = 0.30, 0.25

# Exposure / glare limits (replace with the supervisor's criteria).
# Applied to the conjunctiva ROI of each eye (and to the oval only in the no-eye fallback).
BRIGHT_MIN, BRIGHT_MAX = 70, 190      # acceptable mean gray
BRIGHT_TARGET = (BRIGHT_MIN + BRIGHT_MAX) / 2.0   # "ideal" mean gray -> the best-exposure candidate
GLARE_MAX_FRAC = 0.01                 # oval fallback: max fraction of glare pixels
CLIP_MAX_FRAC = 0.15                  # oval fallback: max fraction of near-black/near-white pixels

# Eye / conjunctiva selection (tune with the supervisor's criteria)
FRAME_STRIDE = 2           # analyse every Nth frame (1 = every frame, slower)
CONJ_BAND_FRAC = 0.50      # height of lower-lid ROI band, as a fraction of eye width
CONJ_A_MIN = 145           # LAB 'a' (OpenCV scale, 128 = neutral): absolute floor for "red/pink"
CONJ_A_DELTA = 6           # ROI pixel must also be this much redder than the skin just below it
CONJ_L_MIN = 40            # ignore very dark pixels (lashes, shadow)
CONJ_MIN_COVERAGE = 0.25   # min fraction of ROI that must be red/pink for the crop to be "clear"
CONJ_GLARE_MAX = 0.02      # max glare fraction inside the conjunctiva ROI
CONJ_SHARP_REF = 100.0     # Laplacian variance treated as "fully sharp" in the ROI
RED_REL_GATE = 0.80        # a frame is only eligible as a candidate if its red-conjunctiva coverage is at least
                          # this fraction of the best coverage seen in the whole video (and >= CONJ_MIN_COVERAGE
                          # when the video reaches that)
CROP_PAD = 0.30            # padding around the eye box, as a fraction of box size
EYE_OVAL_SCALE = 1.15      # dynamic oval = eye region scaled by this (1.41 would touch the corners)
OVAL_BAND_FRAC = 0.35      # share of the lower-lid conjunctiva band the oval reaches (0 = eye only, 1 = whole band)

# The three candidate frames kept for every capture
CRITERIA = ("sharpness", "exposure", "glare")

# MediaPipe FaceMesh landmark indices (image-left eye = subject's right eye)
EYES_MP = {
    "image_left": dict(lower=[33, 7, 163, 144, 145, 153, 154, 155, 133],
                       upper=[246, 161, 160, 159, 158, 157, 173], corners=(33, 133)),
    "image_right": dict(lower=[362, 382, 381, 380, 374, 373, 390, 249, 263],
                        upper=[466, 388, 387, 386, 385, 384, 398], corners=(362, 263)),
}
_face_mesh = None
_haar = None
_mp_warned = False

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_OK = (118, 230, 0)       # BGR green  (#00e676)
COL_WARN = (40, 202, 255)    # BGR amber  (#ffca28)


# ----------------------------------------------------------------------------
# Oval scoring (only used as a fallback when no eye is found in any frame)
# ----------------------------------------------------------------------------
def oval_mask(h, w):
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(OVAL_RX * w), int(OVAL_RY * h)), 0, 0, 360, 255, -1)
    return mask


def score_frame(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    inside = mask > 0
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F)[inside].var())
    g = gray[inside]
    brightness = float(g.mean())
    clipped = float(((g < 10) | (g > 245)).mean())
    glare_px = (hsv[..., 2] > 240) & (hsv[..., 1] < 40) & inside
    glare = float(glare_px.sum() / inside.sum())
    exposure_ok = BRIGHT_MIN <= brightness <= BRIGHT_MAX and clipped <= CLIP_MAX_FRAC
    return dict(sharpness=sharp, brightness=brightness, clipped=clipped,
                glare=glare, exposure_ok=exposure_ok, glare_ok=glare <= GLARE_MAX_FRAC)


def oval_candidates(video_path, stride=1):
    """
    Fallback when no eye was found anywhere: the same three candidates (sharpness / exposure /
    glare), but judged on the whole static oval instead of an eye ROI.
    Returns (n_frames, {criterion: candidate}).
    """
    cap = cv2.VideoCapture(str(video_path))
    mask, best, n = None, {}, 0
    while True:
        if not cap.grab():
            break
        n += 1
        if (n - 1) % max(1, stride):
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        if mask is None:
            mask = oval_mask(*frame.shape[:2])
        m = score_frame(frame, mask)
        penalty = (0 if m["exposure_ok"] else 0.5) + min(m["glare"] * 20, 0.5)
        score = m["sharpness"] * max(0.05, 1 - penalty)
        keys = {"sharpness": (m["sharpness"], score),
                "exposure": (-abs(m["brightness"] - BRIGHT_TARGET), score),
                "glare": (-m["glare"], score)}
        fc = None
        for name, key in keys.items():
            if name not in best or key > best[name]["key"]:
                if fc is None:
                    fc = frame.copy()
                best[name] = dict(key=key, frame=fc, idx=n, sharpness=m["sharpness"],
                                  brightness=m["brightness"], glare=m["glare"], score=score,
                                  ok=False, eye=None)
    cap.release()
    return n, best


# ----------------------------------------------------------------------------
# Eye detection (finds eyes + lower-eyelid conjunctiva ROI + skin reference band + oval)
# ----------------------------------------------------------------------------
def _padded_box(x0, y0, x1, y1, w, h, pad=CROP_PAD):
    bw, bh = x1 - x0, y1 - y0
    return (int(max(0, x0 - pad * bw)), int(max(0, y0 - pad * bh)),
            int(min(w, x1 + pad * bw)), int(min(h, y1 + pad * bh)))


def _oval_from_points(pts, angle_deg=0.0, scale=EYE_OVAL_SCALE):
    """
    Oval (cx, cy, rx, ry, angle in degrees) enclosing the points, aligned with the eye axis.
    Same convention as cv2.ellipse / canvas ellipse (image coordinates, y down).
    """
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    a = np.deg2rad(angle_deg)
    u = np.array([np.cos(a), np.sin(a)], np.float32)      # along the eye axis
    v = np.array([-np.sin(a), np.cos(a)], np.float32)     # across it (down the face)
    pu, pv = pts @ u, pts @ v
    u0, u1, v0, v1 = float(pu.min()), float(pu.max()), float(pv.min()), float(pv.max())
    centre = ((u0 + u1) / 2) * u + ((v0 + v1) / 2) * v
    return dict(cx=float(centre[0]), cy=float(centre[1]),
                rx=float(max(4.0, (u1 - u0) / 2 * scale)),
                ry=float(max(4.0, (v1 - v0) / 2 * scale)),
                angle=float(angle_deg))


def _box_corners(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def _get_face_mesh():
    global _face_mesh
    if _face_mesh is None:
        _face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.3)
    return _face_mesh


def eyes_mediapipe(frame):
    """Precise eyelid landmarks. Needs most of a face in view. ROI follows head tilt."""
    h, w = frame.shape[:2]
    res = _get_face_mesh().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return []
    lm = res.multi_face_landmarks[0].landmark

    def pts(idx):
        return np.array([[lm[i].x * w, lm[i].y * h] for i in idx], np.float32)

    eyes = []
    for side, cfg in EYES_MP.items():
        lower, upper = pts(cfg["lower"]), pts(cfg["upper"])
        c = pts(cfg["corners"])
        axis = c[1] - c[0]
        eye_w = float(np.linalg.norm(axis))
        if eye_w < 1:
            continue
        axis /= eye_w
        normal = np.array([-axis[1], axis[0]], np.float32)   # perpendicular to the eye axis
        if normal[1] < 0:                                    # make it point "down" the face
            normal = -normal
        off = normal * (CONJ_BAND_FRAC * eye_w)
        # ROI = strip below the lower-lid margin (where the everted palpebral conjunctiva shows)
        poly = np.vstack([lower, (lower + off)[::-1]])
        # Reference = next strip further down (skin), for a relative redness check
        ref_poly = np.vstack([lower + off, (lower + 2 * off)[::-1]])
        roi = np.zeros((h, w), np.uint8)
        ref = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi, [poly.astype(np.int32)], 255)
        cv2.fillPoly(ref, [ref_poly.astype(np.int32)], 255)
        allp = np.vstack([lower, upper, poly])
        box = _padded_box(allp[:, 0].min(), allp[:, 1].min(), allp[:, 0].max(), allp[:, 1].max(), w, h)
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        angle = float(np.degrees(np.arctan2(axis[1], axis[0])))   # head tilt
        if angle > 90:
            angle -= 180
        elif angle <= -90:
            angle += 180
        # oval hugs the eye: both eyelid outlines + just the top of the conjunctiva band
        oval_pts = np.vstack([lower, upper, lower + off * OVAL_BAND_FRAC])
        eyes.append(dict(side=side, method="mediapipe", roi_mask=roi, ref_mask=ref, box=box,
                         oval=_oval_from_points(oval_pts, angle)))
    return eyes


def eyes_haar(frame):
    """Fallback: OpenCV Haar eye detector; ROI = region just below each detected eye."""
    global _haar
    if _haar is None:
        _haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    h, w = frame.shape[:2]
    gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    found = _haar.detectMultiScale(gray, 1.1, 6, minSize=(w // 14, w // 14))
    if len(found) == 0:
        return []
    found = sorted(found, key=lambda r: r[2] * r[3], reverse=True)[:2]
    found = sorted(found, key=lambda r: r[0])           # left-to-right
    eyes = []
    for i, (x, y, bw, bh) in enumerate(found):
        roi = np.zeros((h, w), np.uint8)
        ref = np.zeros((h, w), np.uint8)
        y0, y1 = int(y + 0.55 * bh), int(min(h, y + 1.5 * bh))
        cv2.rectangle(roi, (x, y0), (x + bw, y1), 255, -1)
        cv2.rectangle(ref, (x, y1), (x + bw, int(min(h, y1 + 0.9 * bh))), 255, -1)
        box = _padded_box(x, y, x + bw, y1, w, h)
        side = ("image_left" if i == 0 else "image_right") if len(found) == 2 else "single"
        # Haar boxes are ~square, so fit a landscape (eye-shaped) oval instead of the tall box:
        # centred on the eye (slightly low, to include the top of the lower lid), wider than tall.
        oval = dict(cx=float(x + bw / 2), cy=float(y + 0.60 * bh),
                    rx=float(0.58 * bw), ry=float(0.42 * bw), angle=0.0)
        eyes.append(dict(side=side, method="haar", roi_mask=roi, ref_mask=ref, box=box, oval=oval))
    return eyes


def eyes_halves(frame):
    """Optional last resort (--allow-guess): left/right halves of the guide oval. Never marked 'ok'."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    rx, ry = int(OVAL_RX * w), int(OVAL_RY * h)
    om = oval_mask(h, w)
    eyes = []
    for side, (xa, xb) in {"image_left": (cx - rx, cx), "image_right": (cx, cx + rx)}.items():
        roi = np.zeros((h, w), np.uint8)
        roi[cy:cy + ry, xa:xb] = 255                    # lower half of that side of the oval
        roi = cv2.bitwise_and(roi, om)
        box = (max(0, xa), max(0, cy - ry), min(w, xb), min(h, cy + ry))
        eyes.append(dict(side=side, method="oval_halves", roi_mask=roi, ref_mask=None, box=box,
                         oval=_oval_from_points(_box_corners(*box), 0.0)))
    return eyes


def detect_eyes(frame, allow_guess=False):
    global _mp_warned
    if _HAS_MP:
        try:
            eyes = eyes_mediapipe(frame)
            if eyes:
                return eyes
        except Exception as e:
            if not _mp_warned:                          # warn once, not once per frame
                print(f"[warn] mediapipe failed: {e}")
                _mp_warned = True
    eyes = eyes_haar(frame)
    if eyes:
        return eyes
    return eyes_halves(frame) if allow_guess else []


# ----------------------------------------------------------------------------
# Conjunctiva clarity
# ----------------------------------------------------------------------------
def prepare(frame):
    """Colour conversions done once per frame and shared by all eyes."""
    return dict(gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                lab=cv2.cvtColor(frame, cv2.COLOR_BGR2LAB),
                hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))


def conjunctiva_score(prep, eye):
    """How clearly the lower-lid conjunctiva is visible in the ROI (0..1, plus diagnostics)."""
    zero = dict(score=0.0, coverage=0.0, sharp=0.0, glare=1.0, brightness=0.0, ok=False, valid=False)
    x0, y0, x1, y1 = eye["box"]
    roi = eye["roi_mask"][y0:y1, x0:x1] > 0
    n = int(roi.sum())
    if n < 500:
        return zero

    gray = prep["gray"][y0:y1, x0:x1]
    lab = prep["lab"][y0:y1, x0:x1]
    hsv = prep["hsv"][y0:y1, x0:x1]

    # Redness threshold: absolute floor, raised if the skin just below is already reddish
    thr = CONJ_A_MIN
    if eye.get("ref_mask") is not None:
        ref = eye["ref_mask"] > 0
        if int(ref.sum()) >= 300:
            skin_a = float(np.median(prep["lab"][..., 1][ref]))
            thr = max(CONJ_A_MIN, skin_a + CONJ_A_DELTA)

    red = (lab[..., 1] > thr) & (lab[..., 0] > CONJ_L_MIN) & roi
    coverage = float(red.sum() / n)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F)[roi].var())
    glare = float(((hsv[..., 2] > 240) & (hsv[..., 1] < 40) & roi).sum() / n)
    bright = float(gray[roi].mean())
    exposure_ok = BRIGHT_MIN <= bright <= BRIGHT_MAX

    score = (coverage * min(sharp / CONJ_SHARP_REF, 1.0)
             * (1 - min(glare * 20, 0.8)) * (1.0 if exposure_ok else 0.5))
    ok = coverage >= CONJ_MIN_COVERAGE and glare <= CONJ_GLARE_MAX and exposure_ok
    return dict(score=score, coverage=coverage, sharp=sharp, glare=glare, brightness=bright,
                ok=bool(ok), valid=True)


# ----------------------------------------------------------------------------
# Live tracking (drives the dynamic oval in the browser)
# ----------------------------------------------------------------------------
def track_eye(frame):
    """
    Eye whose lower-lid conjunctiva is reddest in this frame, as an oval in frame pixels.
    Called ~10x/second on a downscaled live frame; returns plain JSON-able values.
    """
    eyes = detect_eyes(frame, allow_guess=False)
    if not eyes:
        return dict(found=False)
    prep = prepare(frame)
    best = None
    for eye in eyes:
        s = conjunctiva_score(prep, eye)
        key = (s["valid"], s["coverage"], s["score"])        # reddest palpebral region wins
        if best is None or key > best[0]:
            best = (key, eye, s)
    _, eye, s = best
    o = eye["oval"]
    h, w = frame.shape[:2]
    return dict(found=True, w=int(w), h=int(h), cx=o["cx"], cy=o["cy"], rx=o["rx"], ry=o["ry"],
                angle=o["angle"], ok=bool(s["ok"]), coverage=float(s["coverage"]),
                side=eye["side"], method=eye["method"])


# ----------------------------------------------------------------------------
# Frame selection: 3 candidates (sharpness / exposure / glare) -> 1 best
# ----------------------------------------------------------------------------
def _criterion_key(name, s):
    """
    Sort key for one criterion (bigger = better), used only among frames that already show
    the red lower-lid conjunctiva well enough (see the red gate in select_best_frame).
      sharpness: highest Laplacian variance in the conjunctiva ROI
      exposure : mean gray closest to BRIGHT_TARGET
      glare    : lowest glare (0.1% bins, so "no glare" frames tie and the better-exposed one wins -
                 otherwise the darkest frame would win just because it has no highlights)
    Ties are finally broken by the overall conjunctiva score.
    """
    err = abs(s["brightness"] - BRIGHT_TARGET)
    if name == "sharpness":
        return (s["sharp"], s["score"])
    if name == "exposure":
        return (-err, s["score"])
    return (-round(s["glare"], 3), -err, s["score"])


def _read_frames(video_path, wanted):
    """Second pass: fetch the frames with the given 1-based indices (same counting as pass 1)."""
    out, cap, n = {}, cv2.VideoCapture(str(video_path)), 0
    last = max(wanted)
    while n < last:
        if not cap.grab():
            break
        n += 1
        if n in wanted:
            ok, frame = cap.retrieve()
            if ok:
                out[n] = frame.copy()
    cap.release()
    return out


def select_best_frame(video_path, stride=FRAME_STRIDE, allow_guess=False):
    """
    Pass 1: in every analysed frame detect the eyes and measure each eye's lower-lid conjunctiva
            (red coverage, sharpness, brightness, glare). Only numbers are kept.
    Red gate: only frames whose red-conjunctiva coverage is close to the best in the whole video
            (RED_REL_GATE x best, and >= CONJ_MIN_COVERAGE if the video reaches it) are eligible,
            so the chosen frames are the ones where the red palpebral conjunctiva is most visible.
    Candidates: among eligible frames keep the best sharpness, best exposure and lowest glare.
    Pass 2: re-read just those frames. The single best frame is the candidate with the highest
            overall conjunctiva score (coverage x sharpness x glare x exposure, "ok" first).

    Returns dict(n_frames, candidates={criterion: cand}, best=criterion, eye_found=bool).
    cand = dict(frame, idx, sharpness, brightness, glare, coverage, score, ok, eye)
    eye  = dict(crop, side, method, box, oval, conj) or None (whole-oval fallback, no eye found).
    """
    cap = cv2.VideoCapture(str(video_path))
    records = []
    n = 0
    while True:
        if not cap.grab():
            break
        n += 1
        if (n - 1) % max(1, stride):
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue

        eyes = detect_eyes(frame, allow_guess)
        if not eyes:
            continue
        prep = prepare(frame)
        for eye in eyes:
            s = conjunctiva_score(prep, eye)
            guess = eye["method"] == "oval_halves"
            if guess:
                s["ok"] = False                          # a guess is never "clear"
            records.append(dict(idx=n, s=s, usable=s["valid"] and not guess, side=eye["side"],
                                method=eye["method"], box=eye["box"], oval=eye["oval"]))
    cap.release()

    if n == 0:
        raise RuntimeError("No frames could be decoded from the recording.")

    if records:
        pool = [r for r in records if r["usable"]] or records
        cmax = max(r["s"]["coverage"] for r in pool)
        gate = RED_REL_GATE * cmax
        if cmax >= CONJ_MIN_COVERAGE:
            gate = max(gate, CONJ_MIN_COVERAGE)
        eligible = [r for r in pool if r["s"]["coverage"] >= gate]
        chosen = {name: max(eligible, key=lambda r: _criterion_key(name, r["s"])) for name in CRITERIA}
        frames = _read_frames(video_path, {r["idx"] for r in chosen.values()})

        cands = {}
        for name, r in chosen.items():
            frame = frames.get(r["idx"])
            if frame is None:                            # decoder disagreed between passes (very rare)
                continue
            s = r["s"]
            x0, y0, x1, y1 = r["box"]
            cands[name] = dict(
                frame=frame, idx=r["idx"], sharpness=s["sharp"], brightness=s["brightness"],
                glare=s["glare"], coverage=s["coverage"], score=s["score"], ok=s["ok"],
                eye=dict(side=r["side"], method=r["method"], box=r["box"], oval=r["oval"],
                         conj=s, crop=frame[y0:y1, x0:x1].copy()))
        if len(cands) == len(CRITERIA):
            best = max(CRITERIA, key=lambda k: (cands[k]["ok"], cands[k]["score"]))
            return dict(n_frames=n, candidates=cands, best=best, eye_found=True)

    # No eye anywhere: same three candidates from the whole oval, so the operator still sees something
    _, cands = oval_candidates(video_path, stride)
    best = max(CRITERIA, key=lambda k: cands[k]["score"])
    return dict(n_frames=n, candidates=cands, best=best, eye_found=False)


# ----------------------------------------------------------------------------
# Drawing the measured values (and the dynamic oval) onto a saved image
# ----------------------------------------------------------------------------
def annotate(frame, cand, title, selected=False):
    """Copy of the frame with the eye oval and the sharpness / brightness / glare values printed on it."""
    img = frame.copy()
    h, w = img.shape[:2]
    scale = max(0.5, h / 720.0)
    thick = max(1, int(round(scale * 2)))

    eye = cand.get("eye")
    if eye is not None:
        o = eye["oval"]
        cv2.ellipse(img, (int(o["cx"]), int(o["cy"])), (int(o["rx"]), int(o["ry"])), o["angle"],
                    0, 360, COL_OK if cand["ok"] else COL_WARN, max(2, int(round(scale * 3))), cv2.LINE_AA)

    lines = [(title + ("  [SELECTED]" if selected else ""), COL_OK if selected else (255, 255, 255)),
             (f"Sharpness : {cand['sharpness']:.1f}", (255, 255, 255)),
             (f"Brightness: {cand['brightness']:.1f}", (255, 255, 255)),
             (f"Glare     : {cand['glare'] * 100:.2f}%", (255, 255, 255))]
    if eye is not None:
        lines.append((f"Red conj. : {cand['coverage'] * 100:.0f}%", (255, 255, 255)))
    if eye is None:
        lines.append(("no eye found - whole-oval metrics", COL_WARN))

    sizes = [cv2.getTextSize(t, FONT, scale, thick)[0] for t, _ in lines]
    pad = int(10 * scale)
    lh = max(s[1] for s in sizes) + int(10 * scale)
    box_w = min(w, max(s[0] for s in sizes) + 2 * pad)
    box_h = min(h, lh * len(lines) + pad)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    for i, (t, col) in enumerate(lines):
        cv2.putText(img, t, (pad, int(pad / 2 + lh * (i + 1))), FONT, scale, col, thick, cv2.LINE_AA)
    return img


def _jpeg_b64(img, width=None, quality=85):
    if width and img.shape[1] > width:
        img = cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))),
                         interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


# ----------------------------------------------------------------------------
# Browser UI
# ----------------------------------------------------------------------------
HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Conjunctiva Capture</title>
<style>
 html,body{height:100%}
 body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif;text-align:center;
      height:100dvh;overflow:hidden;display:flex;flex-direction:column;align-items:center}
 #msg{font-size:1.05rem;min-height:1.4rem;margin:6px 10px 4px;flex:none}
 /* stage is sized from the viewport height so the whole UI always fits */
 #stage{position:relative;flex:none;aspect-ratio:16/9;background:#000;
        width:min(96vw,calc((100dvh - 150px) * 16 / 9))}
 video,img#rev{width:100%;height:100%;object-fit:cover;position:absolute;inset:0}
 canvas#guide{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
 #info{font-size:.85rem;margin:4px 10px 0;flex:none;max-width:96vw}
 #btns{flex:none;margin-top:4px;min-height:48px}
 button{font-size:1rem;padding:8px 22px;margin:4px;border:0;border-radius:8px;cursor:pointer}
 #go{background:#2e7d32;color:#fff}#keep{background:#2e7d32;color:#fff}#disc{background:#c62828;color:#fff}
 #count{position:absolute;top:10px;right:20px;font-size:2.5rem;font-weight:bold;color:#fff;text-shadow:0 0 6px #000}
 /* cropped-eye preview overlays the bottom-right corner of the stage */
 #cropbox{position:absolute;right:8px;bottom:8px;background:rgba(0,0,0,.65);padding:4px 6px;
          border-radius:8px;font-size:.75rem;z-index:5}
 #crop{display:block;max-height:26dvh;max-width:24vw;border:2px solid #00e676;border-radius:6px;margin-top:2px}
 /* the three candidate frames overlay the bottom-left corner */
 #cands{position:absolute;left:8px;bottom:8px;display:flex;gap:6px;z-index:5}
 .cand{background:rgba(0,0,0,.7);padding:3px 4px;border:2px solid #555;border-radius:6px;font-size:.7rem}
 .cand.best{border-color:#00e676}
 .cand img{display:block;max-height:16dvh;max-width:19vw;border-radius:4px;margin-bottom:2px}
 .hide{display:none!important}
</style></head><body>
<div id="msg">Camera starting...</div>
<div id="stage">
  <video id="cam" autoplay muted playsinline></video>
  <img id="rev" class="hide">
  <canvas id="guide"></canvas>
  <div id="count"></div>
  <div id="cands" class="hide"></div>
  <div id="cropbox" class="hide"><div>Cropped eye</div><img id="crop"></div>
</div>
<div id="info"></div>
<div id="btns">
  <button id="go" class="hide">Start capture</button>
  <button id="keep" class="hide">Keep</button>
  <button id="disc" class="hide">Discard</button>
</div>
<script>
const $=id=>document.getElementById(id);
const DURATION=%DURATION%;
const OVAL_RX=%OVAL_RX%, OVAL_RY=%OVAL_RY%;   // search hint shown until an eye is found
const TRACK_W=640;      // width of the live frames sent to Python for eye tracking
const SMOOTH=0.55;      // 0..1, higher = oval follows faster (lower = steadier)
const LOST_MAX=6;       // misses in a row before the oval is dropped
let stream, tracking=false, cur=null, lost=0;
const grab=document.createElement('canvas'), gctx=grab.getContext('2d');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));

async function init(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720,facingMode:"environment"},audio:false});
    $('cam').srcObject=stream;
    $('msg').textContent="Pull the lower eyelid down and bring the camera close. The oval locks onto the eye with the reddest conjunctiva.";
    $('go').classList.remove('hide');
    tracking=true; trackLoop();
  }catch(e){$('msg').textContent="Camera error: "+e;}
}
function toB64(blob){return new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result.split(',')[1]);f.readAsDataURL(blob);});}

// ---- dynamic oval: green = conjunctiva clear, amber = eye found but not clear, dashed white = searching
function drawGuide(){
  const c=$('guide'), st=$('stage'), W=st.clientWidth, H=st.clientHeight, v=$('cam');
  if(!W||!H) return;
  if(c.width!==W||c.height!==H){c.width=W;c.height=H;}
  const g=c.getContext('2d'); g.clearRect(0,0,W,H);
  const vw=v.videoWidth||16, vh=v.videoHeight||9;
  const s=Math.max(W/vw,H/vh), ox=(W-vw*s)/2, oy=(H-vh*s)/2;   // same mapping as object-fit:cover
  g.lineWidth=3;
  g.beginPath();
  if(cur){
    g.setLineDash([]); g.strokeStyle=cur.ok?'#00e676':'#ffca28';
    g.ellipse(cur.cx*s+ox,cur.cy*s+oy,cur.rx*s,cur.ry*s,cur.ang,0,2*Math.PI);
  }else{
    g.setLineDash([10,6]); g.strokeStyle='rgba(255,255,255,.55)';
    g.ellipse(W/2,H/2,OVAL_RX*W,OVAL_RY*H,0,0,2*Math.PI);
  }
  g.stroke();
}
async function trackLoop(){
  const v=$('cam');
  while(true){
    if(!tracking||!v.videoWidth){ if(v.videoWidth&&$('rev').classList.contains('hide')) drawGuide(); await sleep(100); continue; }
    const tw=TRACK_W, th=Math.round(TRACK_W*v.videoHeight/v.videoWidth);
    grab.width=tw; grab.height=th; gctx.drawImage(v,0,0,tw,th);
    let r=null;
    try{ r=await window.pyTrack(grab.toDataURL('image/jpeg',0.7).split(',')[1]); }catch(e){}
    if(!tracking) continue;                       // analysis started meanwhile
    if(r&&r.found){
      const k=v.videoWidth/r.w;                   // tracked-frame px -> camera px
      const t={cx:r.cx*k,cy:r.cy*k,rx:r.rx*k,ry:r.ry*k,ang:r.angle*Math.PI/180,ok:r.ok};
      if(cur&&Math.hypot(t.cx-cur.cx,t.cy-cur.cy)<cur.rx*1.5){
        for(const q of ['cx','cy','rx','ry','ang']) cur[q]+=(t[q]-cur[q])*SMOOTH;
        cur.ok=t.ok;
      }else cur=t;                                // first lock, or jumped to the other eye
      lost=0;
    }else if(++lost>LOST_MAX) cur=null;
    drawGuide();
    await sleep(30);
  }
}

function showCands(list){
  const box=$('cands'); box.innerHTML='';
  list.forEach(c=>{
    const d=document.createElement('div'); d.className='cand'+(c.best?' best':'');
    const im=document.createElement('img'); im.src='data:image/jpeg;base64,'+c.image;
    const t=document.createElement('div');
    t.textContent=`${c.best?'SELECTED - ':''}${c.name}: S ${c.sharpness.toFixed(0)} | B ${c.brightness.toFixed(0)} | G ${(c.glare*100).toFixed(1)}% | R ${(c.coverage*100).toFixed(0)}%`;
    d.appendChild(im); d.appendChild(t); box.appendChild(d);
  });
  box.classList.remove('hide');
}

$('go').onclick=async()=>{
  $('go').classList.add('hide'); $('info').textContent='';
  const chunks=[]; const rec=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp8',videoBitsPerSecond:8000000});
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};
  const done=new Promise(r=>rec.onstop=r);
  $('msg').textContent="Hold steady...";
  rec.start(); let t=DURATION; $('count').textContent=t;
  const iv=setInterval(()=>{t--; $('count').textContent=t>0?t:'';},1000);
  await new Promise(r=>setTimeout(r,DURATION*1000));
  clearInterval(iv); rec.stop(); await done;
  tracking=false;
  $('msg').textContent="Analysing frames...";
  const res=await window.pyProcess(await toB64(new Blob(chunks,{type:'video/webm'})));
  if(res.error){$('msg').textContent="Error: "+res.error; $('go').classList.remove('hide'); tracking=true; return;}
  $('rev').src='data:image/jpeg;base64,'+res.image; $('rev').classList.remove('hide');
  $('cam').classList.add('hide'); $('guide').classList.add('hide');
  if(res.crop){$('crop').src='data:image/jpeg;base64,'+res.crop; $('cropbox').classList.remove('hide');}
  showCands(res.cands);
  let m;
  if(!res.crop) m="Warning: no eye could be found in any frame. Retake?";
  else if(!res.crop_ok) m="Warning: conjunctiva not clearly visible (pull the lower lid down, reduce glare). Keep or retake?";
  else m=`Best frame selected (from ${res.best}). Keep it?`;
  $('msg').textContent=m;
  let txt=`frames=${res.frames} | selected=${res.best} | sharpness=${res.sharpness.toFixed(1)} brightness=${res.brightness.toFixed(0)} glare=${(res.glare*100).toFixed(2)}%`;
  if(res.crop) txt+=` | eye=${res.eye_side} (${res.eye_method}) conj_coverage=${(res.conj_coverage*100).toFixed(0)}% conj_score=${res.conj_score.toFixed(2)}`;
  $('info').textContent=txt;
  $('keep').classList.remove('hide'); $('disc').classList.remove('hide');
};
function reset(){
  $('rev').classList.add('hide'); $('cam').classList.remove('hide'); $('guide').classList.remove('hide');
  $('cropbox').classList.add('hide'); $('cands').classList.add('hide');
  $('keep').classList.add('hide'); $('disc').classList.add('hide'); $('go').classList.remove('hide');
  cur=null; lost=0; tracking=true;
}
$('keep').onclick=async()=>{const r=await window.pyKeep(); reset(); $('msg').textContent="Saved: "+r+". Ready for next capture."; };
$('disc').onclick=async()=>{await window.pyDiscard(); reset(); $('msg').textContent="Discarded. Ready to retake.";};
init();
</script></body></html>"""


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
CSV_HEADER = ["image_id", "video_path", "image_path",
              "sharpness_path", "exposure_path", "glare_path", "crop_path",
              "best_criterion", "best_frame_idx", "sharpness", "brightness", "glare",
              "eye_side", "crop_method", "conj_coverage", "conj_score", "crop_ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True, help="device id, e.g. pixel7")
    ap.add_argument("--distance", required=True, help="e.g. 10cm")
    ap.add_argument("--lighting", required=True, help="e.g. indoor, daylight, flash")
    ap.add_argument("--duration", type=int, default=5, help="hold time in seconds")
    ap.add_argument("--out", default="dataset", help="output root folder")
    ap.add_argument("--stride", type=int, default=FRAME_STRIDE,
                    help="analyse every Nth frame (1 = every frame, slower)")
    ap.add_argument("--allow-guess", action="store_true",
                    help="if no eye is detected, guess eye regions from the oval halves (never marked ok)")
    ap.add_argument("--fake-camera", action="store_true", help="use Chromium's fake camera for testing")
    args = ap.parse_args()

    out = Path(args.out)
    # dataset/<device>/<distance>/<lighting>/{videos,images,candidates,crops}
    base = out / args.device / args.distance / args.lighting
    vid_dir, img_dir, crop_dir, cand_dir = (base / "videos", base / "images",
                                            base / "crops", base / "candidates")
    for d in (vid_dir, img_dir, crop_dir, cand_dir):
        d.mkdir(parents=True, exist_ok=True)
    csv_path = out / "dataset.csv"
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            first = next(csv.reader(f), [])
        if first != CSV_HEADER:
            # columns changed: keep the old file untouched and start a new one
            backup = out / f"dataset_old_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
            csv_path.rename(backup)
            print(f"[info] {csv_path} has an older column layout -> renamed to {backup}; "
                  f"starting a new dataset.csv")
    if not csv_path.exists():
        out.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)

    if not _HAS_MP:
        print("[info] mediapipe (legacy solutions API) not available -> using Haar eye detection.")

    tmp = Path(tempfile.mkdtemp(prefix="conj_"))
    pending = {}        # temp file paths only
    pending_meta = {}   # metrics / eye metadata for the CSV

    def py_track(b64):
        """Live eye tracking for the dynamic oval (downscaled JPEG in, oval out)."""
        try:
            frame = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return dict(found=False)
            return track_eye(frame)
        except Exception as e:
            return dict(found=False, error=str(e))

    def py_process(b64):
        try:
            tmp_vid = tmp / "last.webm"
            tmp_vid.write_bytes(base64.b64decode(b64))
            res = select_best_frame(tmp_vid, args.stride, args.allow_guess)
            cands, best = res["candidates"], res["best"]
            bc = cands[best]
            eye = bc["eye"]

            pending.clear()
            pending_meta.clear()
            pending["video"] = tmp_vid

            # the three candidates, each with its measured values printed on it
            thumbs = []
            for name in CRITERIA:
                c = cands[name]
                img = annotate(c["frame"], c, f"Best {name}", selected=(name == best))
                p = tmp / f"last_{name}.png"
                cv2.imwrite(str(p), img)
                pending[f"cand_{name}"] = p
                thumbs.append(dict(name=name, image=_jpeg_b64(img, width=420, quality=80),
                                   sharpness=c["sharpness"], brightness=c["brightness"],
                                   glare=c["glare"], coverage=c.get("coverage", 0.0), best=(name == best)))

            # the single best frame
            final_img = annotate(bc["frame"], bc, f"FINAL BEST ({best})", selected=True)
            tmp_img = tmp / "last.png"
            cv2.imwrite(str(tmp_img), final_img)
            pending["image"] = tmp_img

            pending_meta.update(best_criterion=best, best_frame_idx=bc["idx"],
                                sharpness=bc["sharpness"], brightness=bc["brightness"], glare=bc["glare"])
            result = dict(image=_jpeg_b64(final_img), frames=res["n_frames"], best=best,
                          sharpness=bc["sharpness"], brightness=bc["brightness"], glare=bc["glare"],
                          passed=bc["ok"], crop=None, crop_ok=False, cands=thumbs)

            if eye is not None:
                tmp_crop = tmp / "last_crop.png"
                cv2.imwrite(str(tmp_crop), eye["crop"])          # crop stays clean (no text)
                pending["crop"] = tmp_crop
                pending_meta.update(eye_side=eye["side"], crop_method=eye["method"],
                                    conj_coverage=eye["conj"]["coverage"],
                                    conj_score=eye["conj"]["score"], crop_ok=eye["conj"]["ok"])
                result.update(crop=_jpeg_b64(eye["crop"], quality=90), crop_ok=eye["conj"]["ok"],
                              eye_side=eye["side"], eye_method=eye["method"],
                              conj_coverage=eye["conj"]["coverage"], conj_score=eye["conj"]["score"])

            print(f"[best] frames={res['n_frames']} selected={best}@{bc['idx']} "
                  f"sharp={bc['sharpness']:.1f} bright={bc['brightness']:.0f} glare={bc['glare']*100:.2f}% "
                  + (f"eye={eye['side']}/{eye['method']} conj={eye['conj']['coverage']*100:.0f}% "
                     f"ok={eye['conj']['ok']}" if eye else "eye=NOT FOUND"))
            print("       candidates: " + " | ".join(
                f"{n}@{cands[n]['idx']} S={cands[n]['sharpness']:.1f} B={cands[n]['brightness']:.0f} "
                f"G={cands[n]['glare']*100:.2f}% R={cands[n].get('coverage', 0)*100:.0f}%" for n in CRITERIA))
            return result
        except Exception as e:
            return dict(error=str(e))

    def py_keep():
        if not pending:
            return "nothing pending"
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        image_id = f"{args.device}_{args.distance}_{args.lighting}_{stamp}"
        k = 2
        while (vid_dir / f"{image_id}.webm").exists():     # two Keeps in the same second
            image_id = f"{args.device}_{args.distance}_{args.lighting}_{stamp}_{k}"
            k += 1
        v_dst = vid_dir / f"{image_id}.webm"
        i_dst = img_dir / f"{image_id}.png"
        shutil.move(str(pending["video"]), v_dst)
        shutil.move(str(pending["image"]), i_dst)
        cand_dst = {}
        for name in CRITERIA:
            src = pending.get(f"cand_{name}")
            if src is not None:
                dst = cand_dir / f"{image_id}_{name}.png"
                shutil.move(str(src), dst)
                cand_dst[name] = dst.as_posix()
        c_dst = ""
        if "crop" in pending:
            c_dst = crop_dir / f"{image_id}_eye.png"
            shutil.move(str(pending["crop"]), c_dst)
            c_dst = c_dst.as_posix()
        meta = dict(pending_meta)
        pending.clear()
        pending_meta.clear()

        def f3(key):
            return f"{meta[key]:.3f}" if key in meta else ""

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                image_id, v_dst.as_posix(), i_dst.as_posix(),
                cand_dst.get("sharpness", ""), cand_dst.get("exposure", ""), cand_dst.get("glare", ""),
                c_dst,
                meta.get("best_criterion", ""), meta.get("best_frame_idx", ""),
                f3("sharpness"), f3("brightness"), f"{meta['glare']:.5f}" if "glare" in meta else "",
                meta.get("eye_side", ""), meta.get("crop_method", ""),
                f3("conj_coverage"), f3("conj_score"), meta.get("crop_ok", ""),
            ])
        print(f"[saved] {image_id}")
        return image_id

    def py_discard():
        for p in pending.values():
            Path(p).unlink(missing_ok=True)
        pending.clear()
        pending_meta.clear()
        print("[discarded]")
        return True

    chrome_args = ["--use-fake-ui-for-media-stream",       # auto-allow camera prompt
                   "--start-maximized"]
    if args.fake_camera:
        chrome_args.append("--use-fake-device-for-media-stream")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=chrome_args)
        # no_viewport=True: use the real window size so the UI fits the screen
        ctx = browser.new_context(permissions=["camera"], no_viewport=True)
        page = ctx.new_page()
        page.expose_function("pyTrack", py_track)
        page.expose_function("pyProcess", py_process)
        page.expose_function("pyKeep", py_keep)
        page.expose_function("pyDiscard", py_discard)
        # Serve the UI from a fake https origin so getUserMedia gets a secure context
        html = (HTML.replace("%DURATION%", str(args.duration))
                    .replace("%OVAL_RX%", str(OVAL_RX)).replace("%OVAL_RY%", str(OVAL_RY)))
        page.route("https://capture.local/", lambda r: r.fulfill(body=html, content_type="text/html"))
        page.goto("https://capture.local/")
        print("Close the browser window to finish.")
        try:
            page.wait_for_event("close", timeout=0)
        except Exception:
            pass
        browser.close()
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()