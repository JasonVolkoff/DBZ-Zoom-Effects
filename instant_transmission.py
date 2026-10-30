#!/usr/bin/env python3
"""
Instant Transmission for Zoom.

Put two fingers (index + middle, ring + pinky folded) on your forehead and hold
for a moment. Your body tears into vertical strips with white streaks, vanishes for a few seconds,
then flicker back in.

Webcam -> MediaPipe (hands, face, body mask) -> effect -> OBS Virtual Camera -> Zoom

Keys in the preview window:
  t  trigger the effect manually (for testing)
  d  toggle debug overlay (landmarks, forehead target, mask)
  r  reset the learned background
  q  quit
"""

import argparse
import math
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request
import wave

import cv2
import numpy as np

import audio
import disk as disk_fx
import hair_overlay
import kamehameha as kame_fx
import power_up as power_fx

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
GONE_SECONDS = 3.0        # how long you stay invisible
HOLD_SECONDS = 0.25       # how long the pose must be held before it fires
COOLDOWN_SECONDS = 1.5    # after reappearing, ignore the gesture for this long
PROC_WIDTH = 640          # detection runs at this width (faster); output is full res

STREAK_COLOR = (255, 255, 255)    # BGR: the white vertical streak lines
STREAK_GLOW = (255, 240, 225)     # faint bluish-white glow around them
STREAK_COUNT = 45                 # how many streak lines over your body
STRIP_SHIFT = 0.10                # how far body strips slide up/down (fraction of frame height)
SMEAR = 0.10                      # vertical smear length (fraction of frame height)

# Timeline (seconds after the trigger)
T_OUT_TEAR = 0.6          # body tears into vertical strips and vanishes
# then gone for GONE_SECONDS
T_IN_TEAR = 0.55          # strips tear back in and settle

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
SOUND_DIR = os.path.join(HERE, "sounds")

MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    "selfie_segmenter.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}


# --------------------------------------------------------------------------
# Setup helpers
# --------------------------------------------------------------------------
def ensure_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    for name, url in MODELS.items():
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f"Downloading {name} ...")
        with urllib.request.urlopen(url, context=ctx) as r, open(path, "wb") as f:
            f.write(r.read())


def _write_wav(path, samples, sr=44100):
    samples = np.clip(samples, -1, 1)
    data = (samples * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


def _chirp(f0, f1, dur, sr):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # exponential sweep
    k = math.log(f1 / f0) / dur
    phase = 2 * math.pi * f0 * (np.exp(k * t) - 1) / k
    return t, phase


def ensure_sounds():
    """Synthesize original teleport sounds unless you've supplied your own."""
    os.makedirs(SOUND_DIR, exist_ok=True)
    sr = 44100
    rng = np.random.default_rng(7)

    out_path = os.path.join(SOUND_DIR, "teleport_out.wav")
    if not os.path.exists(out_path):
        dur = 0.42
        t, ph = _chirp(2400, 260, dur, sr)
        tone = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.15 * np.sin(3.01 * ph)
        buzz = 0.5 + 0.5 * np.sign(np.sin(2 * math.pi * 55 * t))          # electric flutter
        noise = rng.standard_normal(len(t))
        noise = np.convolve(noise, np.ones(12) / 12, mode="same")           # soft whoosh
        env = np.exp(-t * 7) * np.minimum(1, t / 0.008)
        s = env * (0.55 * tone * (0.7 + 0.3 * buzz) + 0.45 * noise)
        # quick "zip" blip at the start
        bt = t[: int(0.05 * sr)]
        s[: len(bt)] += 0.5 * np.sin(2 * math.pi * 3200 * bt) * np.exp(-bt * 60)
        _write_wav(out_path, 0.8 * s / np.max(np.abs(s)), sr)

    in_path = os.path.join(SOUND_DIR, "teleport_in.wav")
    if not os.path.exists(in_path):
        dur = 0.38
        t, ph = _chirp(260, 2200, dur, sr)
        tone = np.sin(ph) + 0.35 * np.sin(2 * ph)
        noise = np.convolve(rng.standard_normal(len(t)), np.ones(12) / 12, mode="same")
        env = np.minimum(1, t / 0.25) ** 2 * np.exp(-np.maximum(0, t - 0.3) * 40)
        s = env * (0.6 * tone + 0.4 * noise)
        # landing "thump"
        thump_t = np.linspace(0, 0.12, int(0.12 * sr), endpoint=False)
        thump = np.sin(2 * math.pi * (90 - 300 * thump_t) * thump_t) * np.exp(-thump_t * 30)
        s = np.concatenate([s, 0.9 * thump])
        _write_wav(in_path, 0.8 * s / np.max(np.abs(s)), sr)


CUSTOM_SOUND_DIR = os.path.join(SOUND_DIR, "custom")
CUSTOM_SOUND_EXTS = (".mp3", ".wav", ".m4a", ".aiff", ".aif")   # mp3 and wav, plus other formats afplay handles


def sound_path(name):
    """Your override in sounds/custom/ (any supported format, same base name) wins over the built-in sound."""
    base = os.path.splitext(name)[0].lower()
    try:
        for f in sorted(os.listdir(CUSTOM_SOUND_DIR)):
            stem, ext = os.path.splitext(f)
            if stem.lower() == base and ext.lower() in CUSTOM_SOUND_EXTS:
                return os.path.join(CUSTOM_SOUND_DIR, f)
    except OSError:
        pass
    path = os.path.join(SOUND_DIR, name)
    return path if os.path.exists(path) else None


def play_sound(name):
    path = sound_path(name)
    if path is None:
        return None
    player = shutil.which("afplay") or shutil.which("aplay") or shutil.which("paplay")
    if player:
        return subprocess.Popen([player, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return None


def list_custom_sounds():
    try:
        return [f for f in sorted(os.listdir(CUSTOM_SOUND_DIR))
                if os.path.splitext(f)[1].lower() in CUSTOM_SOUND_EXTS]
    except OSError:
        return []


def stop_sound(proc):
    try:
        if proc is not None and proc.poll() is None:
            proc.terminate()
    except Exception:
        pass


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
class Detector:
    def __init__(self):
        import mediapipe as mp
        self.mp = mp
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions
        mode = vision.RunningMode.VIDEO

        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base(model_asset_path=os.path.join(MODEL_DIR, "hand_landmarker.task")),
            running_mode=mode, num_hands=2,
            min_hand_detection_confidence=0.5, min_tracking_confidence=0.5))
        self.face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=base(model_asset_path=os.path.join(MODEL_DIR, "blaze_face_short_range.tflite")),
            running_mode=mode, min_detection_confidence=0.4))
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=base(model_asset_path=os.path.join(MODEL_DIR, "selfie_segmenter.tflite")),
            running_mode=mode, output_confidence_masks=True, output_category_mask=False))
        self._last_ts = 0
        self.hand_meta = []          # per hand: 3D world landmarks + handedness

    def run(self, small_bgr):
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        h, w = small_bgr.shape[:2]

        hands, self.hand_meta = [], []
        hres = self.hands.detect_for_video(img, ts)
        for i, lms in enumerate(hres.hand_landmarks):
            hands.append(np.array([(p.x * w, p.y * h) for p in lms], dtype=np.float32))
            world = hres.hand_world_landmarks[i] if i < len(hres.hand_world_landmarks) else None
            label = hres.handedness[i][0].category_name if i < len(hres.handedness) and hres.handedness[i] else "Left"
            self.hand_meta.append(dict(
                world=np.array([(p.x, p.y, p.z) for p in world], dtype=np.float32) if world else None,
                label=label))

        faces = []                                         # every detection; the head tracker picks
        for d in self.face.detect_for_video(img, ts).detections:
            kps = np.array([(k.x * w, k.y * h) for k in d.keypoints], dtype=np.float32)
            score = d.categories[0].score if d.categories else 1.0
            faces.append((kps, float(score)))

        masks = self.seg.segment_for_video(img, ts).confidence_masks
        m = masks[0].numpy_view() if len(masks) == 1 else 1.0 - masks[0].numpy_view()
        mask = np.squeeze(m).astype(np.float32)
        return hands, faces, mask


def forehead_point(face_kps):
    """BlazeFace keypoints: 0 right eye, 1 left eye, 2 nose, 3 mouth, 4/5 ears."""
    eye_mid = (face_kps[0] + face_kps[1]) / 2
    eye_dist = float(np.linalg.norm(face_kps[0] - face_kps[1]))
    up = eye_mid - face_kps[3]
    up = up / (np.linalg.norm(up) + 1e-6)
    return eye_mid + up * eye_dist * 0.75, eye_dist


def is_two_finger_pose(lm):
    """Index + middle extended and together, ring + pinky curled."""
    d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b]))
    hand_size = d(0, 9) + 1e-6
    index_up = d(0, 8) > d(0, 6) * 1.15
    middle_up = d(0, 12) > d(0, 10) * 1.15
    ring_down = d(0, 16) < d(0, 14) * 1.1
    pinky_down = d(0, 20) < d(0, 18) * 1.1
    together = d(8, 12) < 0.55 * hand_size
    return index_up and middle_up and ring_down and pinky_down and together


class GestureTrigger:
    def __init__(self):
        self.held_since = None
        self.last_face = None
        self.last_face_t = 0.0

    def update(self, hands, face, now):
        if face is not None:
            self.last_face, self.last_face_t = face, now
        elif self.last_face is not None and now - self.last_face_t < 1.0:
            face = self.last_face        # your hand may hide part of your face
        if face is None:
            self.held_since = None
            return False, None

        target, eye_dist = forehead_point(face)
        touching = False
        for lm in hands:
            tip = (lm[8] + lm[12]) / 2
            if is_two_finger_pose(lm) and np.linalg.norm(tip - target) < eye_dist * 0.95:
                touching = True
                break

        if not touching:
            self.held_since = None
            return False, (target, eye_dist, False)
        if self.held_since is None:
            self.held_since = now
        return now - self.held_since >= HOLD_SECONDS, (target, eye_dist, True)


# --------------------------------------------------------------------------
# Background plate (what's behind you)
# --------------------------------------------------------------------------
class BackgroundPlate:
    """Learns the background from every pixel your body isn't covering."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.bg = None
        self.seen = None
        self._cache = None
        self._cache_t = -1.0

    def cached_plate(self, now, max_age=0.5):
        if self._cache is None or now - self._cache_t > max_age:
            self._cache = self.plate()
            self._cache_t = now
        return self._cache

    def update(self, frame, person_mask, learning=True):
        if self.bg is None:
            self.bg = frame.astype(np.float32)
            self.seen = np.zeros(frame.shape[:2], np.float32)
        if not learning:
            return
        # stay well clear of your body's edges
        h, w = frame.shape[:2]
        sm = cv2.resize(person_mask, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
        sm = cv2.dilate((sm > 0.2).astype(np.uint8), np.ones((9, 9), np.uint8))
        clear = (cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST) == 0).astype(np.float32)
        rate = np.where(self.seen > 0, 0.08, 1.0) * clear
        self.bg += (frame.astype(np.float32) - self.bg) * rate[..., None]
        self.seen = np.maximum(self.seen, clear)

    def plate(self):
        bg = np.clip(self.bg, 0, 255).astype(np.uint8)
        unseen = (self.seen == 0)
        if not unseen.any():
            return bg
        # fill never-seen areas (usually right behind you) by inpainting at low res
        h, w = bg.shape[:2]
        sw, sh = 256, max(1, int(256 * h / w))
        small = cv2.resize(bg, (sw, sh), interpolation=cv2.INTER_AREA)
        hole = cv2.resize(unseen.astype(np.uint8) * 255, (sw, sh), interpolation=cv2.INTER_NEAREST)
        filled = cv2.inpaint(small, hole, 5, cv2.INPAINT_TELEA)
        filled = cv2.GaussianBlur(filled, (0, 0), 3)
        filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
        out = bg.copy()
        out[unseen] = filled[unseen]
        return out


# --------------------------------------------------------------------------
# Effect rendering
# --------------------------------------------------------------------------
def soft_mask(mask, grow=9, feather=7):
    """Grown + feathered body mask, computed at 1/4 res for speed."""
    h, w = mask.shape[:2]
    q = 4
    m = (cv2.resize(mask, (w // q, h // q), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8) * 255
    g = max(1, grow // q)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g * 2 + 1, g * 2 + 1)))
    m = cv2.GaussianBlur(m, (0, 0), max(1, feather / q))
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return m[..., None]


def composite(frame, plate, alpha):
    """alpha = 1 where you should be hidden."""
    return (frame.astype(np.float32) * (1 - alpha) + plate.astype(np.float32) * alpha).astype(np.uint8)


def flash_body(frame, alpha, amount):
    bright = cv2.addWeighted(frame, 1 - amount, np.full_like(frame, (255, 245, 220)), amount, 0)
    return composite(frame, bright, alpha)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


class Tear:
    """
    The vertical tear: your body is cut into thin vertical strips that slide
    up/down, smear vertically and drop out one by one, while white streak lines
    flash over you.

    progress k: 0 = fully you, 1 = fully gone.
    """

    def __init__(self, mask, rng):
        h, w = mask.shape[:2]
        self.h, self.w = h, w
        m = (mask > 0.5).astype(np.uint8)
        bx, by, bw, bh = cv2.boundingRect(m)
        if bw < 10 or bh < 10:                        # no body found: use the middle of the frame
            bx, by, bw, bh = w // 3, h // 6, w // 3, h * 5 // 6
        pad = int(0.03 * w)
        self.x0, self.x1 = max(0, bx - pad), min(w, bx + bw + pad)
        self.y0, self.y1 = max(0, by - pad), min(h, by + bh + pad)
        rw, rh = self.x1 - self.x0, self.y1 - self.y0

        # vertical strips of random width (narrower strips look more like the anime)
        edges = [0]
        while edges[-1] < rw:
            edges.append(edges[-1] + int(rng.integers(max(3, w // 200), max(6, w // 60))))
        n = len(edges) - 1
        self.col_strip = np.zeros(rw, np.int32)
        for i in range(n):
            self.col_strip[edges[i]:edges[i + 1]] = i
        self.offset = rng.uniform(-1, 1, n).astype(np.float32)       # slide direction/amount
        self.vanish = rng.uniform(0.35, 0.95, n).astype(np.float32)  # when each strip drops out
        self.bright = rng.uniform(0, 1, n).astype(np.float32)        # some strips glow brighter

        # streak lines, placed where your body actually is
        cover = m[self.y0:self.y1, self.x0:self.x1].sum(axis=0).astype(np.float64) + 0.5
        xs = rng.choice(rw, size=STREAK_COUNT, p=cover / cover.sum()) + self.x0
        self.streaks = []
        for x in xs:
            col = m[self.y0:self.y1, x]
            ys = np.nonzero(col)[0]
            top, bot = (ys[0], ys[-1]) if len(ys) else (0, rh)
            length = rng.uniform(0.25, 0.9) * (bot - top + 1)
            y_top = self.y0 + top + rng.uniform(-0.15, 0.6) * (bot - top + 1 - length) - 0.1 * length
            self.streaks.append(dict(
                x=float(x), y=float(y_top), len=float(length),
                start=float(rng.uniform(0.0, 0.55)),        # when it appears (in k)
                thick=int(rng.integers(1, 4)),
                bright=float(rng.uniform(0.55, 1.0)),
                speed=float(rng.uniform(-0.6, 0.6)),       # slides a little while visible
            ))

        yy, xx = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        self.grid_x, self.grid_y = xx, yy

    def body(self, frame, mask, k):
        """Returns the torn body and its alpha, both cropped to the effect box."""
        shift = self.offset[self.col_strip] * STRIP_SHIFT * self.h * smoothstep(0.05, 0.9, k)
        map_y = (self.grid_y - shift[None, :]).astype(np.float32)   # remap needs float32 maps
        crop_img = cv2.remap(frame, self.grid_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        crop_mask = cv2.remap(mask, self.grid_x, map_y, cv2.INTER_LINEAR, borderValue=0)

        smear = int(SMEAR * self.h * smoothstep(0.0, 0.7, k))     # vertical motion smear
        if smear > 2:
            crop_img = cv2.blur(crop_img, (1, smear))
            crop_mask = cv2.blur(crop_mask, (1, smear))

        # strips brighten toward white just before dropping out
        glow_amt = (smoothstep(self.vanish - 0.3, self.vanish, k) * (0.25 + 0.5 * self.bright))[self.col_strip]
        crop_img = (crop_img.astype(np.float32) * (1 - glow_amt[None, :, None])
                    + 255 * glow_amt[None, :, None])

        keep = (1 - smoothstep(self.vanish - 0.12, self.vanish, k))[self.col_strip]
        alpha = np.clip(crop_mask, 0, 1) * keep[None, :]
        return crop_img, alpha[..., None]

    def draw_streaks(self, img, k, env):
        """White vertical lines with a soft glow. env = overall strength 0..1."""
        if env <= 0.01:
            return img
        h, w = self.h, self.w
        q = 4
        glow = np.zeros((h // q, w // q, 3), np.uint8)
        core_a = np.zeros((h, w), np.uint8)
        for s in self.streaks:
            vis = smoothstep(s["start"], s["start"] + 0.12, k) * env * s["bright"]
            if np.random.rand() < 0.15:                 # each line flickers a little
                vis *= 0.3
            if vis <= 0.02:
                continue
            grow = 0.6 + 0.6 * smoothstep(s["start"], s["start"] + 0.4, k)
            length = s["len"] * grow
            y = s["y"] + s["speed"] * k * s["len"] - (length - s["len"]) / 2
            x = int(s["x"])
            p1, p2 = (x, int(y)), (x, int(y + length))
            c = int(255 * vis)
            cv2.line(core_a, p1, p2, c, s["thick"], cv2.LINE_AA)
            g = tuple(int(v * vis) for v in STREAK_GLOW)
            cv2.line(glow, (x // q, int(y) // q), (x // q, int(y + length) // q), g, 1, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), 1.5)
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
        out = cv2.addWeighted(img, 1.0, glow, 0.45, 0)
        ys, xs = np.nonzero(core_a)
        if len(ys):
            a = core_a[ys, xs, None].astype(np.float32) / 255.0
            col = np.array(STREAK_COLOR, np.float32)
            out[ys, xs] = (out[ys, xs] * (1 - a) + col * a).astype(np.uint8)
        return out

    def render(self, frame, mask, base, k, env):
        """Draw the torn body (from `frame`) over `base` (the scene with you removed)."""
        body, alpha = self.body(frame, mask, k)
        out = base.copy()
        region = out[self.y0:self.y1, self.x0:self.x1].astype(np.float32)
        out[self.y0:self.y1, self.x0:self.x1] = (region * (1 - alpha) + body * alpha).astype(np.uint8)
        return self.draw_streaks(out, k, env)


class InstantTransmission:
    def __init__(self):
        self.start = None
        self.tear_out = None
        self.tear_in = None
        self.sound_in_played = False
        self.cooldown_until = 0.0
        self.rng = np.random.default_rng()
        self.hair = False          # hair overlay currently on?
        self.hair_next = False     # what you'll have when you reappear
        self.carry_level = 0       # power-up form you teleported out of (its hair tears out with you)

    @property
    def active(self):
        return self.start is not None

    def total(self):
        return T_OUT_TEAR + GONE_SECONDS + T_IN_TEAR

    def trigger(self, now, mask):
        if self.active or now < self.cooldown_until:
            return False
        self.start = now
        self.tear_out = Tear(mask, self.rng)
        self.tear_in = None
        self.sound_in_played = False
        self.hair_next = not self.hair             # each teleport toggles the hair
        self.carry_level = 0
        play_sound("teleport_out.wav")
        return True

    def leave_form(self, level):
        """Teleporting out of a power-up form: you come back with no hair overlay at all."""
        self.hair = False
        self.hair_next = False
        self.carry_level = level

    def carried_form(self, now):
        """The form whose hair should still show while you tear out (0 = none)."""
        if self.active and self.carry_level and now - self.start < T_OUT_TEAR + GONE_SECONDS * 0.5:
            return self.carry_level
        return 0

    def hair_visible(self, now):
        """Old hair until you're gone, new hair when you come back."""
        if not self.active:
            return self.hair
        if now - self.start >= T_OUT_TEAR + GONE_SECONDS * 0.5:
            return self.hair_next
        return self.hair

    def render(self, frame, mask, plate_fn, now):
        t = now - self.start
        a_live = soft_mask(mask)
        t1 = T_OUT_TEAR
        t2 = t1 + GONE_SECONDS
        t3 = t2 + T_IN_TEAR

        plate = plate_fn()
        base = composite(frame, plate, a_live)          # the room, with you removed

        if t < t1:                                       # 1. tear out
            k = t / T_OUT_TEAR
            env = smoothstep(0.0, 0.15, k) * (1 - smoothstep(0.8, 1.0, k))
            return self.tear_out.render(frame, mask, base, k, env)

        if t < t2:                                       # 2. gone
            return base

        if not self.sound_in_played:
            play_sound("teleport_in.wav")
            self.sound_in_played = True
            self.tear_in = Tear(mask, self.rng)          # built where you are now

        if t < t3:                                       # 3. tear back in
            kin = (t - t2) / T_IN_TEAR
            k = 1 - kin                                  # same effect, played backwards
            env = smoothstep(0.0, 0.15, kin) * (1 - smoothstep(0.7, 1.0, kin))
            return self.tear_in.render(frame, mask, base, k, env)

        self.start = None                                # done
        self.hair = self.hair_next
        self.carry_level = 0
        self.cooldown_until = now + COOLDOWN_SECONDS
        return frame


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def obs_app_running():
    if sys.platform != "darwin":
        return False
    r = subprocess.run(["pgrep", "-ix", "obs"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def open_virtual_camera(w, h, fps):
    """Returns (camera or None, status message)."""
    if obs_app_running():
        msg = ("The OBS app is open. Quit OBS completely (Cmd+Q), then run this again. "
               "While OBS is open, Zoom shows OBS's own scene, not this effect.")
        print("\n" + msg + "\n")
        return None, "VIRTUAL CAM OFF: quit the OBS app, then restart this"
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=w, height=h, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR)
        print(f"Virtual camera started: {cam.device}  ->  pick it as your camera in Zoom.")
        return cam, "VIRTUAL CAM ON"
    except Exception as e:
        print("\nCould not start the virtual camera:", repr(e))
        print("Fixes: install OBS 30+, open it, click 'Start Virtual Camera' once (approve the prompt),")
        print("stop it, quit OBS, then run this again. Continuing with preview only.\n")
        return None, "VIRTUAL CAM OFF: see the Terminal for the error"


def draw_status(img, text, ok):
    color = (60, 200, 60) if ok else (40, 40, 230)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img, (10, img.shape[0] - th - 30), (30 + tw, img.shape[0] - 10), (0, 0, 0), -1)
    cv2.putText(img, text, (20, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def draw_debug(img, hands, face_info, scale, mask):
    for lm in hands:
        for p in lm:
            cv2.circle(img, tuple((p * scale).astype(int)), 3, (0, 255, 0), -1)
        for i in (8, 12):
            cv2.circle(img, tuple((lm[i] * scale).astype(int)), 7, (0, 255, 255), 2)
    if face_info:
        target, eye_dist, touching = face_info
        cv2.circle(img, tuple((target * scale).astype(int)), int(eye_dist * 0.95 * scale),
                   (0, 255, 0) if touching else (0, 0, 255), 2)
    small = cv2.resize((mask * 255).astype(np.uint8), (160, 90))
    img[10:100, -170:-10] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    return img


def main():
    global GONE_SECONDS
    ap = argparse.ArgumentParser(description="DBZ webcam effects: Instant Transmission + Kamehameha")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gone", type=float, default=GONE_SECONDS, help="seconds to stay invisible")
    ap.add_argument("--no-vcam", action="store_true", help="preview window only")
    args = ap.parse_args()
    GONE_SECONDS = args.gone

    ensure_models()
    ensure_sounds()
    kame_fx.ensure_sounds(SOUND_DIR)
    power_fx.ensure_sounds(SOUND_DIR)
    disk_fx.ensure_sounds(SOUND_DIR)
    os.makedirs(CUSTOM_SOUND_DIR, exist_ok=True)
    custom = list_custom_sounds()
    if custom:
        print("Custom sounds:", ", ".join(custom))

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        sys.exit("Couldn't read from the webcam. On a Mac, allow camera access for your terminal app in "
                 "System Settings > Privacy & Security > Camera, then try again.")
    H, W = frame.shape[:2]
    print(f"Webcam: {W}x{H}")

    detector = Detector()
    gesture = GestureTrigger()
    plate = BackgroundPlate()
    effect = InstantTransmission()
    kame = kame_fx.Kamehameha(W, H, play_sound, stop_sound)
    head = hair_overlay.HeadTracker()
    hair = hair_overlay.load_hair(HERE)
    transform_hair = hair_overlay.TransformHairs(HERE, power_fx.MAX_LEVEL)
    hum = audio.LoopSound(sound_path, play_sound, stop_sound)
    power = power_fx.PowerUp(W, H, play_sound, stop_sound, loop=hum)
    disk = disk_fx.EnergyDisk(W, H, play_sound, stop_sound, loop=audio.LoopSound(sound_path, play_sound, stop_sound))
    if args.no_vcam:
        vcam, vcam_status = None, "PREVIEW ONLY (--no-vcam)"
    else:
        vcam, vcam_status = open_virtual_camera(W, H, args.fps)

    def teleport(now, mask):
        form = power.hair_level if power.busy else 0
        if effect.trigger(now, mask):
            disk.cancel(now)                             # a disk in your hand fizzles out
            if power.busy:
                effect.leave_form(form)                  # come back with no hair overlay
                power.dissipate(now)                     # the aura doesn't come with you

    proc_h = int(PROC_WIDTH * H / W)
    scale = W / PROC_WIDTH
    debug = False
    frame_i = 0
    print("Ready.")
    print("  Instant Transmission: two fingers to your forehead")
    print("  Kamehameha: both hands open, wrists together, palms apart; hold to charge")
    print("  Energy disk: one hand open, palm up, fingers stretched; hold still, then swish to throw")
    print("  Teleporting toggles the hair overlay (on when you reappear, off the next time)")
    print("  Power-up aura: clench both fists (again for level 2, again for level 3); p dissipates it")
    print("  Keys: t=test teleport  k=test kamehameha  p=power up / dissipate aura  u=next aura level")
    print("        x=energy disk (form / throw)  h=toggle hair")
    print("        d=debug  r=reset bg  q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != W or frame.shape[0] != H:
            frame = cv2.resize(frame, (W, H))
        now = time.monotonic()

        small = cv2.resize(frame, (PROC_WIDTH, proc_h), interpolation=cv2.INTER_AREA)
        hands, faces, small_mask = detector.run(small)
        mask = cv2.resize(small_mask, (W, H), interpolation=cv2.INTER_LINEAR)

        hands_full = [lm * scale for lm in hands]
        chosen = head.select([(k * scale, sc) for k, sc in faces], hands_full, now)
        face = chosen / scale if chosen is not None else None   # your real face (not a hand)
        hp = head.get(now)
        if hp is not None:
            form = effect.carried_form(now) or power.hair_level
            th = transform_hair.get(form) if form > 0 else None
            if th is not None:
                frame, mask = th.apply(frame, mask, hp, now)     # aura on: transformation hair
            elif effect.hair_visible(now):
                frame, mask = hair.apply(frame, mask, hp, now)   # hair becomes part of "you"

        frame_i += 1
        if frame_i % 3 == 0 or plate.bg is None:
            plate.update(frame, mask)

        fired, face_info = gesture.update(hands, face, now)
        if fired and not kame.busy:
            teleport(now, mask)

        power.update(power_fx.detect_fists(hands) and not effect.active and not kame.busy, now)

        meta = getattr(detector, "hand_meta", [])
        if len(meta) != len(hands_full):
            meta = [None] * len(hands_full)
        disk.update(hands_full, meta, now, allowed=not effect.active and not kame.busy)

        kpose = kame_fx.detect_pose(hands_full)
        kame.update(None if (effect.active or disk.busy) else kpose, now)

        out = effect.render(frame, mask, lambda: plate.cached_plate(now), now) if effect.active else frame
        out = power.render(out, mask, now)
        out = kame.render(out, now)
        out = disk.render(out, now)

        if vcam is not None:
            vcam.send(out)

        preview = cv2.flip(out, 1)        # mirror the preview like a selfie view
        if debug:
            dbg = draw_debug(out.copy(), hands, face_info, scale, small_mask)
            dbg = kame.debug(dbg, kpose)
            dbg = disk.debug(dbg, hands_full, meta)
            preview = cv2.flip(dbg, 1)
        preview = draw_status(preview.copy(), vcam_status, vcam is not None)   # preview only, not sent to Zoom
        cv2.imshow("DBZ effects (preview)", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("t") and not kame.busy:
            teleport(now, mask)
        if key == ord("p") and not effect.active:
            power.toggle(now)                                # power up / dissipate the aura
        if key == ord("x") and not effect.active:
            disk.test(now)                                   # form a disk / fire it
        if key == ord("u") and not effect.active:
            if not power.level_up(now):                      # next aura level (test without fists)
                power.start(now, force=True)
        if key == ord("k") and not effect.active:
            kame.force(now)
        if key == ord("h") and not effect.active:
            effect.hair = not effect.hair
        if key == ord("d"):
            debug = not debug
        if key == ord("r"):
            plate.reset()

    hum.stop(immediate=True)
    if disk.loop is not None:
        disk.loop.stop(immediate=True)
    cap.release()
    if vcam is not None:
        vcam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
