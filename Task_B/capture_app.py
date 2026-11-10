#!/usr/bin/env python3
"""
Conjunctiva image capture: guided capture -> video -> eye-based best-frame selection
-> single-eye crop (clearest lower-eyelid palpebral conjunctiva) -> structured save + CSV.

Frames are judged on the EYE region (lower-lid conjunctiva strip), not on the whole oval.

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

# Oval guide geometry as a fraction of frame size (must match the SVG in HTML below)
OVAL_RX, OVAL_RY = 0.30, 0.25

# Exposure / glare limits (replace with the supervisor's criteria).
# Applied to the conjunctiva ROI of each eye (and to the oval only in the no-eye fallback).
BRIGHT_MIN, BRIGHT_MAX = 70, 190      # acceptable mean gray
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
CROP_PAD = 0.30            # padding around the eye box, as a fraction of box size

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


def best_oval_frame(video_path):
    """Fallback when no eye was found anywhere: best frame by whole-oval quality."""
    cap = cv2.VideoCapture(str(video_path))
    mask, best, best_key = None, None, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mask is None:
            mask = oval_mask(*frame.shape[:2])
        m = score_frame(frame, mask)
        penalty = (0 if m["exposure_ok"] else 0.5) + min(m["glare"] * 20, 0.5)
        key = m["sharpness"] * max(0.05, 1 - penalty)
        if key > best_key:
            best_key, best = key, (frame.copy(), m)
    cap.release()
    return best


# ----------------------------------------------------------------------------
# Eye detection (finds eyes + lower-eyelid conjunctiva ROI + skin reference band)
# ----------------------------------------------------------------------------
def _padded_box(x0, y0, x1, y1, w, h, pad=CROP_PAD):
    bw, bh = x1 - x0, y1 - y0
    return (int(max(0, x0 - pad * bw)), int(max(0, y0 - pad * bh)),
            int(min(w, x1 + pad * bw)), int(min(h, y1 + pad * bh)))


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
        eyes.append(dict(side=side, method="mediapipe", roi_mask=roi, ref_mask=ref, box=box))
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
        eyes.append(dict(side=side, method="haar", roi_mask=roi, ref_mask=ref, box=box))
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
        eyes.append(dict(side=side, method="oval_halves", roi_mask=roi, ref_mask=None,
                         box=(max(0, xa), max(0, cy - ry), min(w, xb), min(h, cy + ry))))
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
    zero = dict(score=0.0, coverage=0.0, sharp=0.0, glare=1.0, brightness=0.0, ok=False)
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
    return dict(score=score, coverage=coverage, sharp=sharp, glare=glare, brightness=bright, ok=ok)


# ----------------------------------------------------------------------------
# Frame selection (eye-based)
# ----------------------------------------------------------------------------
def select_best_frame(video_path, stride=FRAME_STRIDE, allow_guess=False):
    """
    Analyse frames of the recording. In each, detect the eyes and score the lower-lid
    conjunctiva of each one. Return the frame/eye with the best score (clear-and-ok first).

    Returns (best_frame, metrics, n_frames, eye_result_or_None).
    If no eye is found in any frame, falls back to the best whole-oval frame with eye=None.
    """
    cap = cv2.VideoCapture(str(video_path))
    best = None                          # dict(key, frame, eye)
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
            if eye["method"] == "oval_halves":
                s["ok"] = False                          # a guess is never "clear"
            key = (s["ok"], s["score"])
            if best is None or key > best["key"]:        # copy frame only when it becomes the best
                x0, y0, x1, y1 = eye["box"]
                best = dict(key=key, frame=frame.copy(), conj=s, side=eye["side"],
                            method=eye["method"], box=eye["box"],
                            crop=frame[y0:y1, x0:x1].copy())
    cap.release()

    if n == 0:
        raise RuntimeError("No frames could be decoded from the recording.")

    if best is not None:
        s = best["conj"]
        metrics = dict(sharpness=s["sharp"], brightness=s["brightness"], glare=s["glare"],
                       passed_all_checks=s["ok"])
        eye = dict(crop=best["crop"], side=best["side"], method=best["method"],
                   box=best["box"], conj=s)
        return best["frame"], metrics, n, eye

    # No eye anywhere: return the best oval frame so the operator still sees something
    fallback = best_oval_frame(video_path)
    frame, m = fallback
    m["passed_all_checks"] = False
    return frame, m, n, None


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
 svg{position:absolute;inset:0;width:100%;height:100%}
 #info{font-size:.85rem;margin:4px 10px 0;flex:none;max-width:96vw}
 #btns{flex:none;margin-top:4px;min-height:48px}
 button{font-size:1rem;padding:8px 22px;margin:4px;border:0;border-radius:8px;cursor:pointer}
 #go{background:#2e7d32;color:#fff}#keep{background:#2e7d32;color:#fff}#disc{background:#c62828;color:#fff}
 #count{position:absolute;top:10px;right:20px;font-size:2.5rem;font-weight:bold;color:#fff;text-shadow:0 0 6px #000}
 /* cropped-eye preview overlays the bottom-right corner of the stage */
 #cropbox{position:absolute;right:8px;bottom:8px;background:rgba(0,0,0,.65);padding:4px 6px;
          border-radius:8px;font-size:.75rem;z-index:5}
 #crop{display:block;max-height:26dvh;max-width:24vw;border:2px solid #00e676;border-radius:6px;margin-top:2px}
 .hide{display:none}
</style></head><body>
<div id="msg">Camera starting...</div>
<div id="stage">
  <video id="cam" autoplay muted playsinline></video>
  <img id="rev" class="hide">
  <svg id="guide" viewBox="0 0 100 100" preserveAspectRatio="none">
    <ellipse cx="50" cy="50" rx="30" ry="25" fill="none" stroke="#00e676" stroke-width="0.5" stroke-dasharray="2 1"/>
  </svg>
  <div id="count"></div>
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
let stream;
async function init(){
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720,facingMode:"environment"},audio:false});
    $('cam').srcObject=stream;
    $('msg').textContent="Pull the lower eyelid down, align the eye inside the oval, bring the camera close.";
    $('go').classList.remove('hide');
  }catch(e){$('msg').textContent="Camera error: "+e;}
}
function toB64(blob){return new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result.split(',')[1]);f.readAsDataURL(blob);});}

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
  $('msg').textContent="Analysing frames...";
  const res=await window.pyProcess(await toB64(new Blob(chunks,{type:'video/webm'})));
  if(res.error){$('msg').textContent="Error: "+res.error; $('go').classList.remove('hide'); return;}
  $('rev').src='data:image/png;base64,'+res.image; $('rev').classList.remove('hide');
  $('cam').classList.add('hide'); $('guide').classList.add('hide');
  if(res.crop){$('crop').src='data:image/jpeg;base64,'+res.crop; $('cropbox').classList.remove('hide');}
  let m;
  if(!res.crop) m="Warning: no eye could be found in any frame. Retake?";
  else if(!res.crop_ok) m="Warning: conjunctiva not clearly visible (pull the lower lid down, reduce glare). Keep or retake?";
  else m="Best frame and eye selected. Keep it?";
  $('msg').textContent=m;
  let txt=`frames=${res.frames} sharpness=${res.sharpness.toFixed(1)} brightness=${res.brightness.toFixed(0)} glare=${(res.glare*100).toFixed(2)}%`;
  if(res.crop) txt+=` | eye=${res.eye_side} (${res.eye_method}) conj_coverage=${(res.conj_coverage*100).toFixed(0)}% conj_score=${res.conj_score.toFixed(2)}`;
  $('info').textContent=txt;
  $('keep').classList.remove('hide'); $('disc').classList.remove('hide');
};
function reset(){
  $('rev').classList.add('hide'); $('cam').classList.remove('hide'); $('guide').classList.remove('hide');
  $('cropbox').classList.add('hide');
  $('keep').classList.add('hide'); $('disc').classList.add('hide'); $('go').classList.remove('hide');
}
$('keep').onclick=async()=>{const r=await window.pyKeep(); reset(); $('msg').textContent="Saved: "+r+". Ready for next capture."; };
$('disc').onclick=async()=>{await window.pyDiscard(); reset(); $('msg').textContent="Discarded. Ready to retake.";};
init();
</script></body></html>"""


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
CSV_HEADER = ["image_id", "video_path", "image_path", "crop_path",
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
    # dataset/<device>/<distance>/<lighting>/{videos,images,crops}
    base = out / args.device / args.distance / args.lighting
    vid_dir, img_dir, crop_dir = base / "videos", base / "images", base / "crops"
    for d in (vid_dir, img_dir, crop_dir):
        d.mkdir(parents=True, exist_ok=True)
    csv_path = out / "dataset.csv"
    if not csv_path.exists():
        out.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_HEADER)
    else:
        with open(csv_path, newline="") as f:
            first = next(csv.reader(f), [])
        if first != CSV_HEADER:
            print(f"[warn] {csv_path} has an older header {first}; "
                  f"move/rename it so a new one with crop columns is created.")

    if not _HAS_MP:
        print("[info] mediapipe (legacy solutions API) not available -> using Haar eye detection.")

    tmp = Path(tempfile.mkdtemp(prefix="conj_"))
    pending = {}        # temp file paths only
    pending_meta = {}   # eye metadata for the CSV

    def py_process(b64):
        try:
            tmp_vid = tmp / "last.webm"
            tmp_vid.write_bytes(base64.b64decode(b64))
            frame, m, n, eye = select_best_frame(tmp_vid, args.stride, args.allow_guess)
            tmp_img = tmp / "last.png"
            cv2.imwrite(str(tmp_img), frame)
            pending.clear()
            pending_meta.clear()
            pending.update(video=tmp_vid, image=tmp_img)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            result = dict(image=base64.b64encode(buf).decode(), frames=n,
                          sharpness=m["sharpness"], brightness=m["brightness"],
                          glare=m["glare"], passed=m["passed_all_checks"],
                          crop=None, crop_ok=False)

            if eye is not None:
                tmp_crop = tmp / "last_crop.png"
                cv2.imwrite(str(tmp_crop), eye["crop"])
                pending["crop"] = tmp_crop
                pending_meta.update(eye_side=eye["side"], crop_method=eye["method"],
                                    conj_coverage=eye["conj"]["coverage"],
                                    conj_score=eye["conj"]["score"], crop_ok=eye["conj"]["ok"])
                _, cbuf = cv2.imencode(".jpg", eye["crop"], [cv2.IMWRITE_JPEG_QUALITY, 90])
                result.update(crop=base64.b64encode(cbuf).decode(), crop_ok=eye["conj"]["ok"],
                              eye_side=eye["side"], eye_method=eye["method"],
                              conj_coverage=eye["conj"]["coverage"], conj_score=eye["conj"]["score"])

            print(f"[best] frames={n} sharp={m['sharpness']:.1f} bright={m['brightness']:.0f} "
                  f"glare={m['glare']*100:.2f}% passed={m['passed_all_checks']} "
                  + (f"eye={eye['side']}/{eye['method']} conj={eye['conj']['coverage']*100:.0f}% "
                     f"ok={eye['conj']['ok']}" if eye else "eye=NOT FOUND"))
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
        c_dst = ""
        if "crop" in pending:
            c_dst = crop_dir / f"{image_id}_eye.png"
            shutil.move(str(pending["crop"]), c_dst)
            c_dst = c_dst.as_posix()
        meta = dict(pending_meta)
        pending.clear()
        pending_meta.clear()
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                image_id, v_dst.as_posix(), i_dst.as_posix(), c_dst,
                meta.get("eye_side", ""), meta.get("crop_method", ""),
                f"{meta['conj_coverage']:.3f}" if "conj_coverage" in meta else "",
                f"{meta['conj_score']:.3f}" if "conj_score" in meta else "",
                meta.get("crop_ok", ""),
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
        page.expose_function("pyProcess", py_process)
        page.expose_function("pyKeep", py_keep)
        page.expose_function("pyDiscard", py_discard)
        # Serve the UI from a fake https origin so getUserMedia gets a secure context
        html = HTML.replace("%DURATION%", str(args.duration))
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
