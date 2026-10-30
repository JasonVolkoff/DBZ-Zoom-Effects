"""
Hair overlay that follows your head.

Default: original glowing "power-up" flame hair, drawn procedurally.
Custom: put a transparent PNG at overlays/hair.png and it's used instead.
"""

import math
import os

import cv2
import numpy as np

# ---- flame hair look (BGR, 0..1) ----
FLAME_OUTER = np.array([0.85, 0.20, 0.55], np.float32)   # deep violet
FLAME_MID = np.array([1.0, 0.55, 0.95], np.float32)      # pink-violet
FLAME_CORE = np.array([1.0, 1.0, 1.0], np.float32)
FLAME_COUNT = 13          # number of flame tendrils
FLAME_HEIGHT = 1.0        # height multiplier

# ---- custom PNG placement (fractions of head width) ----
PNG_WIDTH = 2.5           # PNG width as a multiple of head width
PNG_ANCHOR_Y = 0.8       # point in the PNG (0 top .. 1 bottom) that sits on your hairline
PNG_HAIRLINE = 0.32       # hairline height above the eyes, in head widths

# ---- transformation hair (overlays/hair_transformation[_2|_3].png or .gif), shown with the aura ----
TRANSFORM_PNG_WIDTH = PNG_WIDTH
TRANSFORM_PNG_ANCHOR_Y = PNG_ANCHOR_Y
TRANSFORM_PNG_HAIRLINE = PNG_HAIRLINE

# Per-level placement. Anything left out uses the TRANSFORM_PNG_* values above.
#   width     size, as a multiple of your head width (bigger = larger hair)
#   hairline  height above your eyes, in head widths (bigger = hair sits higher)
#   anchor_y  which point of the image (0 top .. 1 bottom) sits on your hairline
#   offset_x  sideways shift in head widths, as seen in the preview (+ = right, - = left)
#   mirror    flip the image left-right
TRANSFORM_HAIR = {
    1: dict(),
    2: dict(width=3.0, hairline=0.35, anchor_y=0.7, offset_x=-0.06, mirror=True),
    3: dict(width=2.08, hairline=0.305),
}


def _norm(v):
    return v / (np.linalg.norm(v) + 1e-6)


# ---- head tracking (keeps the hair on your head when your hands come up) ----
FACE_MIN_SCORE = 0.5      # ignore weaker face detections
HEAD_GATE = 0.6           # a detection farther than this from your head (in head widths) is ignored
HEAD_REACQUIRE = 4        # frames a new, far-away position must be seen before the hair moves there
HEAD_HOLD = 1.5           # seconds the hair stays put when your face can't be seen (e.g. hands in front)


def _measure(kps):
    re, le, mouth, rear, lear = kps[0], kps[1], kps[3], kps[4], kps[5]
    eye_mid = (re + le) / 2
    ed = float(np.linalg.norm(re - le))
    hw = float(np.clip(np.linalg.norm(rear - lear), 1.9 * ed, 2.8 * ed))
    return dict(eye=eye_mid, hw=hw, up=_norm(eye_mid - mouth))


def _hand_boxes(hands, grow=0.15):
    boxes = []
    for lm in hands:
        x0, y0 = lm.min(axis=0)
        x1, y1 = lm.max(axis=0)
        gx, gy = (x1 - x0) * grow, (y1 - y0) * grow
        boxes.append((x0 - gx, y0 - gy, x1 + gx, y1 + gy))
    return boxes


def _inside(p, box):
    return box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3]


def _on_hand(kps, boxes):
    """True if most of the 'face' (eyes, nose, mouth) sits on a hand: likely a hand mistaken for a face."""
    return any(sum(_inside(kps[i], b) for i in range(4)) >= 3 for b in boxes)


def _near_hand(m, boxes):
    """A hand overlapping the face area (the face keypoints get less reliable)."""
    r = m["hw"] * 0.8
    fx0, fy0, fx1, fy1 = m["eye"][0] - r, m["eye"][1] - r, m["eye"][0] + r, m["eye"][1] + r * 1.4
    return any(not (b[2] < fx0 or b[0] > fx1 or b[3] < fy0 or b[1] > fy1) for b in boxes)


class HeadTracker:
    """
    Follows your head from BlazeFace detections (full-res pixels), ignoring hands
    that the detector mistakes for faces: it sticks to the detection nearest your
    head, ignores sudden jumps, and only moves to a new spot after seeing it
    consistently for a few frames (and not on a hand).
    """

    def __init__(self):
        self.pose = None
        self.t = -10.0
        self.pending = None
        self.pending_n = 0

    def _set(self, m, now, smooth):
        if self.pose is None or smooth is None:
            self.pose = dict(eye=m["eye"].copy(), hw=m["hw"], up=m["up"].copy())
        else:
            a_pos, a_size = smooth
            p = self.pose
            p["eye"] = (1 - a_pos) * p["eye"] + a_pos * m["eye"]
            p["hw"] = (1 - a_size) * p["hw"] + a_size * m["hw"]
            p["up"] = _norm((1 - a_size) * p["up"] + a_size * m["up"])
        self.t = now

    def select(self, faces, hands, now):
        """
        faces: list of (keypoints [6,2], score); hands: list of [21,2] landmarks (both full-res).
        Updates the head and returns the keypoints it accepted as your face (or None).
        """
        boxes = _hand_boxes(hands)
        cands = [(k, _measure(k)) for k, sc in faces if sc >= FACE_MIN_SCORE]
        tracked = self.pose is not None and now - self.t < HEAD_HOLD

        if tracked and cands:
            p = self.pose
            k, m = min(cands, key=lambda c: np.linalg.norm(c[1]["eye"] - p["eye"]))
            close = np.linalg.norm(m["eye"] - p["eye"]) < HEAD_GATE * p["hw"]
            same_size = 0.65 < m["hw"] / p["hw"] < 1.5
            if close and same_size:
                # steadier while a hand is over your face, responsive otherwise
                self._set(m, now, (0.2, 0.1) if _near_hand(m, boxes) else (0.5, 0.3))
                self.pending, self.pending_n = None, 0
                return k

        # (re)acquire: a candidate that isn't on a hand, seen consistently for a few frames
        good = [c for c in cands if not _on_hand(c[0], boxes)]
        if not good:
            self.pending, self.pending_n = None, 0
            return None
        k, m = max(good, key=lambda c: c[1]["hw"])
        if self.pending is not None and np.linalg.norm(m["eye"] - self.pending["eye"]) < 0.5 * self.pending["hw"]:
            self.pending_n += 1
        else:
            self.pending_n = 1
        self.pending = m
        need = HEAD_REACQUIRE if tracked else 2
        if self.pending_n >= need:
            self._set(m, now, None)
            self.pending, self.pending_n = None, 0
            return k
        return None

    def update(self, kps, now):
        """Old interface: a single face."""
        self.select([] if kps is None else [(kps, 1.0)], [], now)

    def get(self, now):
        return self.pose if self.pose is not None and now - self.t < HEAD_HOLD else None


def _composite_roi(frame, mask, x0, y0, color, alpha, mask_add):
    """color: float 0..1 (h,w,3) premultiplied-ish light; alpha: opacity (h,w)."""
    h, w = alpha.shape
    roi = frame[y0:y0 + h, x0:x0 + w].astype(np.float32)
    roi = roi * (1 - alpha[..., None]) + color * 255
    frame[y0:y0 + h, x0:x0 + w] = np.clip(roi, 0, 255).astype(np.uint8)
    mroi = mask[y0:y0 + h, x0:x0 + w]
    mask[y0:y0 + h, x0:x0 + w] = np.maximum(mroi, mask_add)


class FlameHair:
    def __init__(self, seed=5):
        rng = np.random.default_rng(seed)
        n = FLAME_COUNT
        base = np.linspace(-1.65, 1.65, n)
        self.tendrils = [dict(theta=float(b + rng.uniform(-0.08, 0.08)),
                              len=float(rng.uniform(0.85, 1.15)),
                              width=float(rng.uniform(0.85, 1.2)),
                              phase=float(rng.uniform(0, 6.3)),
                              speed=float(rng.uniform(1.3, 2.1)))
                         for b in base]
        # draw the outer ones first so the middle ones sit on top
        self.tendrils.sort(key=lambda d: -abs(d["theta"]))

    def _geometry(self, head, now):
        u = head["up"]
        s = np.array([-u[1], u[0]], np.float32)
        hw = head["hw"]
        C = head["eye"] + u * 0.10 * hw
        a, b = 0.57 * hw, 0.72 * hw

        def skull(th):
            return C + s * a * math.sin(th) + u * b * math.cos(th)

        # cap over the top of the head, closed along a hairline across the forehead
        arc = [skull(th) for th in np.linspace(-1.62, 1.62, 40)]
        edge = math.sin(1.62)
        hair_y_mid, hair_y_edge = (0.30 - 0.10) * hw, math.cos(1.62) * b
        line = []
        for x in np.linspace(1, -1, 20):
            y = hair_y_mid * (1 - x * x) + hair_y_edge * x * x
            line.append(C + s * a * edge * x + u * y)
        cap = np.array(arc + line, np.float32)

        flames = []
        for d in self.tendrils:
            th = d["theta"]
            root = skull(th)
            normal = _norm(s * math.sin(th) / a + u * math.cos(th) / b)
            dirv = _norm(normal * 0.45 + u * 0.8)
            perp = np.array([-dirv[1], dirv[0]], np.float32)
            L = hw * FLAME_HEIGHT * (0.42 + 0.38 * math.cos(th * 0.8)) * d["len"]
            L *= 1 + 0.10 * math.sin(now * 3.1 + d["phase"])
            W0 = hw * 0.19 * d["width"]
            left, right, center = [], [], []
            for k in np.linspace(0, 1, 12):
                curl = s * math.sin(th) * L * 0.35 * k * k              # sweep outward
                wig = perp * hw * 0.06 * k * math.sin(2 * math.pi * (1.2 * k - d["speed"] * now) + d["phase"])
                c = root - dirv * W0 * 0.3 + dirv * L * k + curl + wig
                wk = W0 * (1 - k) ** 0.85 + 0.5
                left.append(c + perp * wk)
                right.append(c - perp * wk)
                center.append((c, wk))
            flames.append((np.array(left + right[::-1], np.float32), center))
        return cap, flames

    def apply(self, frame, mask, head, now):
        cap, flames = self._geometry(head, now)
        hw = head["hw"]
        pts = np.vstack([cap] + [f[0] for f in flames])
        pad = int(hw * 0.3)
        H, W = frame.shape[:2]
        x0 = max(0, int(pts[:, 0].min()) - pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(W, int(pts[:, 0].max()) + pad)
        y1 = min(H, int(pts[:, 1].max()) + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return frame, mask
        off = np.array([x0, y0], np.float32)
        h, w = y1 - y0, x1 - x0
        hw = hw / 2                                         # draw at half res, upscale at the end
        h2, w2 = (h + 1) // 2, (w + 1) // 2

        outer = np.zeros((h2, w2), np.uint8)
        mid = np.zeros((h2, w2), np.uint8)
        core = np.zeros((h2, w2), np.uint8)
        to_i = lambda p: np.round((p - off) * 2).astype(np.int32)    # half res, 2 bits subpixel
        cv2.fillPoly(outer, [to_i(cap)], 255, cv2.LINE_AA, shift=2)
        cv2.fillPoly(mid, [to_i(cap)], 70, cv2.LINE_AA, shift=2)
        for poly, center in flames:
            cv2.fillPoly(outer, [to_i(poly)], 255, cv2.LINE_AA, shift=2)
            # mid and core: thinner copies of the same flame
            for canvas, frac, val in ((mid, 0.6, 200), (core, 0.2, 170)):
                left, right = [], []
                for i, (c, wk) in enumerate(center):
                    k = i / (len(center) - 1)
                    if k > 0.9:
                        break
                    if i + 1 < len(center):
                        dv = _norm(center[i + 1][0] - c)
                    perp = np.array([-dv[1], dv[0]], np.float32)
                    left.append(c + perp * wk * frac)
                    right.append(c - perp * wk * frac)
                if len(left) > 2:
                    cv2.fillPoly(canvas, [to_i(np.array(left + right[::-1], np.float32))], val,
                                 cv2.LINE_AA, shift=2)

        outer = outer.astype(np.float32) / 255
        mid = cv2.GaussianBlur(mid, (0, 0), max(1, hw * 0.012)).astype(np.float32) / 255
        core = cv2.GaussianBlur(core, (0, 0), max(1, hw * 0.015)).astype(np.float32) / 255
        flicker = 0.9 + 0.1 * math.sin(now * 23)
        glow = cv2.GaussianBlur(outer, (0, 0), max(1, hw * 0.07)) * flicker

        color = (outer[..., None] * FLAME_OUTER * 0.8 + mid[..., None] * FLAME_MID * 0.45
                 + core[..., None] * FLAME_CORE * 0.7 + glow[..., None] * FLAME_OUTER * 0.45)
        alpha = np.clip(outer * 0.9 + core * 0.1, 0, 1)
        mask_add = np.clip(outer + glow * 0.8, 0, 1)
        up = lambda x: cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR)
        _composite_roi(frame, mask, x0, y0, up(color), up(alpha), up(mask_add))
        return frame, mask


def _load_frames(path):
    """RGBA frames (float 0..1) and per-frame durations in seconds. PNG or GIF (animated OK)."""
    frames, durations = [], []
    try:
        from PIL import Image, ImageSequence
        im = Image.open(path)
        for fr in ImageSequence.Iterator(im):
            rgba = np.array(fr.convert("RGBA"))
            frames.append(rgba)
            durations.append(max(0.02, fr.info.get("duration", im.info.get("duration", 100)) / 1000))
    except ImportError:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3 and img.shape[2] == 4:
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
            durations.append(0.1)
    if not frames:
        raise ValueError(f"couldn't read {path}")
    out = []
    for rgba in frames:
        if rgba[..., 3].min() == 255:
            raise ValueError(f"{path} needs a transparent background")
        bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32) / 255
        out.append((bgr, rgba[..., 3].astype(np.float32) / 255))
    return out, durations


class ImageHair:
    """A transparent PNG or (animated) GIF placed on your head."""

    def __init__(self, path, width=None, anchor_y=None, hairline=None, offset_x=0.0, mirror=False):
        self.width = PNG_WIDTH if width is None else width
        self.anchor_y = PNG_ANCHOR_Y if anchor_y is None else anchor_y
        self.hairline = PNG_HAIRLINE if hairline is None else hairline
        self.offset_x = offset_x
        self.frames, self.durations = _load_frames(path)
        if mirror:
            self.frames = [(np.ascontiguousarray(b[:, ::-1]), np.ascontiguousarray(a[:, ::-1]))
                           for b, a in self.frames]
        self.total = float(sum(self.durations))
        self.h, self.w = self.frames[0][1].shape[:2]

    def _frame(self, now):
        if len(self.frames) == 1:
            return self.frames[0]
        t = now % self.total
        for fr, d in zip(self.frames, self.durations):
            if t < d:
                return fr
            t -= d
        return self.frames[-1]

    def apply(self, frame, mask, head, now):
        bgr, alpha = self._frame(now)
        u, hw = head["up"], head["hw"]
        sc = self.width * hw / self.w
        sn, c = float(u[0]), float(-u[1])               # rotate the image's "up" onto your head's up
        R = np.array([[c, -sn], [sn, c]], np.float32) * sc
        anchor = np.array([self.w / 2, self.anchor_y * self.h], np.float32)
        target = head["eye"] + u * self.hairline * hw
        if self.offset_x:
            right_in_frame = np.array([c, sn], np.float32)      # the image's right, in the frame
            target = target - right_in_frame * self.offset_x * hw   # preview is mirrored, so flip the sign
        t = target - R @ anchor
        M = np.hstack([R, t[:, None]]).astype(np.float32)

        corners = np.array([[0, 0], [self.w, 0], [0, self.h], [self.w, self.h]], np.float32)
        pc = corners @ R.T + t
        H, W = frame.shape[:2]
        x0, y0 = max(0, int(pc[:, 0].min()) - 2), max(0, int(pc[:, 1].min()) - 2)
        x1, y1 = min(W, int(pc[:, 0].max()) + 2), min(H, int(pc[:, 1].max()) + 2)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return frame, mask
        M[:, 2] -= (x0, y0)
        size = (x1 - x0, y1 - y0)
        col = cv2.warpAffine(bgr, M, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        a = cv2.warpAffine(alpha, M, size, flags=cv2.INTER_LINEAR, borderValue=0)
        _composite_roi(frame, mask, x0, y0, col * a[..., None], a, a)
        return frame, mask


PngHair = ImageHair          # old name


def _find_image(here, base):
    for ext in (".png", ".gif"):
        path = os.path.join(here, "overlays", base + ext)
        if os.path.exists(path):
            return path
    return None


def load_hair(here):
    path = _find_image(here, "hair")
    if path:
        try:
            print(f"Using custom hair overlay: {path}")
            return ImageHair(path)
        except Exception as e:
            print(f"Couldn't use {path}:", e, "- using the built-in flame hair.")
    return FlameHair()


class TransformHair:
    """overlays/<base>.png or .gif, loaded when it appears (or changes) on disk."""

    def __init__(self, here, base="hair_transformation", settings=None):
        self.here, self.base = here, base
        self.settings = settings or {}
        self.key = None
        self.hair = None
        self.warned = False

    def get(self):
        path = _find_image(self.here, self.base)
        if path is None:
            if not self.warned:
                print(f"No overlays/{self.base}.png or .gif yet.")
                self.warned = True
            return None
        key = (path, os.path.getmtime(path))
        if key != self.key:
            self.key = key
            try:
                st = self.settings
                self.hair = ImageHair(path, st.get("width", TRANSFORM_PNG_WIDTH),
                                      st.get("anchor_y", TRANSFORM_PNG_ANCHOR_Y),
                                      st.get("hairline", TRANSFORM_PNG_HAIRLINE),
                                      st.get("offset_x", 0.0), st.get("mirror", False))
                n = len(self.hair.frames)
                print(f"Loaded {os.path.basename(path)}" + (f" ({n} animation frames)" if n > 1 else ""))
            except Exception as e:
                print(f"Couldn't use {path}:", e)
                self.hair = None
        return self.hair


class TransformHairs:
    """One transformation hair per aura level. Falls back to the closest lower level's hair."""

    def __init__(self, here, levels=3):
        self.by_level = {1: TransformHair(here, "hair_transformation", TRANSFORM_HAIR.get(1))}
        for lv in range(2, levels + 1):
            self.by_level[lv] = TransformHair(here, f"hair_transformation_{lv}", TRANSFORM_HAIR.get(lv))

    def get(self, level):
        for lv in range(level, 0, -1):
            h = self.by_level[lv].get()
            if h is not None:
                return h
        return None
