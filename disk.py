"""
Energy disk.

Gesture: one hand open, fingers stretched, palm facing UP (held up like the
reference, or flat like holding a plate). Hold it still for a moment and a
swirling energy disk forms above your palm. Once it's formed, swish that hand
and the disk flies off in the direction you swished.
"""

import math
import os
import wave
from collections import deque

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
HOLD_SECONDS = 0.6          # hold the palm-up pose this long to start forming
FORM_SECONDS = 1.3          # forming time
POSE_GRACE = 0.5            # pose may drop out this long (holding or forming) without resetting
LOST_SECONDS = 1.0          # formed disk fades if your hand is gone this long
SWISH_SPEED = 0.55           # how fast a swish must be, in frame-heights per second (higher = harder)
SWISH_DISTANCE = 0.07       # ...and how far the hand must travel during it (fraction of frame height)
SWISH_MIN_READY = 0.4       # ignore swishes in the first moments after it forms
FLY_SPEED = 0.9            # frame diagonals per second at launch
FLY_ACCEL = 1.4             # speeds up as it flies
COOLDOWN = 0.8

DISK_SIZE = 1.7             # disk radius as a multiple of your hand width
DISK_MIN, DISK_MAX = 0.10, 0.32   # radius limits (fraction of frame height)
DISK_ASPECT = 0.30          # flatness of the disk (height / width)
DISK_LIFT = 0.55            # how high above your palm it floats (in hand widths, plus its own height)
SPIN_SPEED = 9.0            # radians per second while hovering

PALM_UP_MIN = 0.45          # how squarely the palm must face up to start (0..1)
PALM_UP_KEEP = 0.2          # ...and to stay in the pose once recognised
POSE_SMOOTHING = 0.3        # 0..1, lower = steadier but slower to react
PALM_FLIP = -1               # set to -1 if "palm up" is detected when your palm faces DOWN

# BGR 0..1
WHITE = np.array([1.0, 1.0, 1.0], np.float32)
PALE = np.array([0.62, 0.97, 1.0], np.float32)       # pale yellow
GOLD = np.array([0.10, 0.72, 1.0], np.float32)       # golden yellow
ORANGE = np.array([0.04, 0.45, 0.98], np.float32)    # orange edge / glow


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# Gesture
# --------------------------------------------------------------------------
def palm_measure(world):
    """
    world: [21,3] MediaPipe world landmarks (x right, y down, z away from the camera).
    Returns (vertical part of the palm normal before handedness is applied,
             fraction of the four fingers that are stretched).
    """
    vi, vp = world[5] - world[0], world[17] - world[0]
    n = np.cross(vi, vp)
    n /= np.linalg.norm(n) + 1e-9
    d = lambda a, b: float(np.linalg.norm(world[a] - world[b]))
    stretched = sum(d(0, tip) > 1.12 * d(0, pip) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))) / 4
    return float(n[1]), stretched


def palm_state(world, label):
    """(palm_up -1..1, fingers_stretched bool) for one frame. MediaPipe assumes a mirrored
    image and we feed it an unmirrored one, so its "Left" is your right hand."""
    ny, stretched = palm_measure(world)
    up = -ny * (1 if label == "Left" else -1) * PALM_FLIP
    return up, stretched >= 0.75


class HandTrack:
    """One hand followed across frames, with its readings smoothed so single bad frames don't count."""

    def __init__(self, palm, now):
        self.palm = palm
        self.t = now
        self.vote = 0.0          # > 0: MediaPipe says "Left" (your right hand), < 0: "Right"
        self.up = 0.0
        self.stretched = 0.0
        self.posed = False
        self.lm = None

    def feed(self, lm, m, now):
        a = POSE_SMOOTHING
        self.palm, self.lm, self.t = hand_geom(lm)[0], lm, now
        if m is None or m.get("world") is None:
            return
        # handedness by majority vote: MediaPipe's label can flip for a frame
        self.vote = 0.85 * self.vote + 0.15 * (1 if m.get("label", "Left") == "Left" else -1)
        ny, stretched = palm_measure(m["world"])
        up = -ny * (1 if self.vote >= 0 else -1) * PALM_FLIP
        self.up = (1 - a) * self.up + a * up
        self.stretched = (1 - a) * self.stretched + a * stretched
        if self.posed:                                   # hysteresis: easier to stay than to enter
            self.posed = self.up >= PALM_UP_KEEP and self.stretched >= 0.45
        else:
            self.posed = self.up >= PALM_UP_MIN and self.stretched >= 0.65


def hand_geom(lm):
    """Palm centre and hand width (pixels) from 2D landmarks."""
    palm = lm[[0, 5, 9, 13, 17]].mean(axis=0)
    span = float(max(np.ptp(lm[:, 0]), np.ptp(lm[:, 1]), 1.0))
    return palm, span


# --------------------------------------------------------------------------
# Sounds (original, synthesized on first run)
# --------------------------------------------------------------------------
def _write_wav(path, s, sr, peak=0.8):
    s = peak * s / (np.max(np.abs(s)) + 1e-9)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((s * 32767).astype(np.int16).tobytes())


def _lp(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def ensure_sounds(sound_dir):
    os.makedirs(sound_dir, exist_ok=True)
    sr = 44100
    rng = np.random.default_rng(31)

    path = os.path.join(sound_dir, "disk_form.wav")
    if not os.path.exists(path):
        dur = FORM_SECONDS + 0.2
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        k = np.clip(t / FORM_SECONDS, 0, 1)
        f = 180 + 900 * k ** 1.6
        whir = np.sin(2 * np.pi * np.cumsum(f) / sr) * (1 + 0.5 * np.sin(2 * np.pi * (8 + 30 * k) * t))
        air = _lp(rng.standard_normal(len(t)), 5) * (0.2 + 0.8 * k)
        env = np.minimum(1, t / 0.05) * (0.3 + 0.7 * k)
        _write_wav(path, env * (0.6 * whir + 0.3 * air), sr, 0.7)

    path = os.path.join(sound_dir, "disk_hum.wav")
    if not os.path.exists(path):
        dur = 2.0                                   # every tone completes whole cycles: loops cleanly
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        whine = np.sin(2 * np.pi * 1080 * t + 0.6 * np.sin(2 * np.pi * 24 * t))
        buzz = np.sin(2 * np.pi * 135 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 12 * t))
        _write_wav(path, 0.35 * whine + 0.5 * buzz, sr, 0.35)

    path = os.path.join(sound_dir, "disk_fire.wav")
    if not os.path.exists(path):
        dur = 0.9
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        f = 1400 * np.exp(-t * 2.2) + 200                # doppler-ish drop as it flies away
        whine = np.sin(2 * np.pi * np.cumsum(f) / sr)
        whoosh = _lp(rng.standard_normal(len(t)), 9) * 3
        env = np.minimum(1, t / 0.015) * np.exp(-t * 3.2)
        _write_wav(path, env * (0.5 * whine + 0.6 * whoosh), sr, 0.8)


# --------------------------------------------------------------------------
# Effect
# --------------------------------------------------------------------------
class EnergyDisk:
    S = 256                      # disk texture size
    UNIT = 0.27                  # rim radius as a fraction of S (room for wisps and glow)

    def __init__(self, W, H, play_sound, stop_sound, loop=None):
        self.W, self.H = W, H
        self.play, self.stop, self.loop = play_sound, stop_sound, loop
        self.rng = np.random.default_rng()
        self.state = "idle"
        self.t0 = 0.0
        self.pose_since = None
        self.last_pose_t = -10.0
        self.last_hand_t = -10.0
        self.cool_until = 0.0
        self.snd = None
        self.humming = False
        self.forced = False

        self.tracks = []             # every hand in view, followed across frames
        self.hold_from = None        # where the palm was when the hold started
        self.hand = None             # tracked palm centre
        self.span = H * 0.12
        self.hist = deque(maxlen=12)
        self.pos = None              # disk centre
        self.radius = H * 0.18
        self.vel = np.zeros(2, np.float32)
        self.spin = 0.0
        self.tilt = 0.0
        self.trail = deque(maxlen=5)
        self.last_now = None
        self.fade_from = 1.0

        S = self.S
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        u = (xx - S / 2) / (S * self.UNIT)
        v = (yy - S / 2) / (S * self.UNIT)
        self.r = np.sqrt(u * u + v * v)
        self.th = np.arctan2(v, u)
        self.stars = [(self.rng.uniform(0, 2 * np.pi), self.rng.uniform(1.1, 1.6), self.rng.uniform(0, 1),
                       self.rng.uniform(0.8, 2.0)) for _ in range(14)]
        self.sparks = [(self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0, 1), self.rng.uniform(1.0, 1.8))
                       for _ in range(18)]

    # ---------------- state ----------------
    @property
    def busy(self):
        return self.state != "idle"

    def _sound(self, name):
        if self.snd is not None:
            self.stop(self.snd)
        self.snd = self.play(name)

    def _hum(self, on):
        if self.loop is None:
            return
        if on and not self.humming:
            self.loop.start("disk_hum.wav")
        elif not on and self.humming:
            self.loop.stop()
        self.humming = on

    def cancel(self, now):
        if self.state in ("forming", "ready"):
            self._hum(False)
            self.fade_from = 1.0 if self.state == "ready" else min(1.0, (now - self.t0) / FORM_SECONDS)
            self.state, self.t0 = "fading", now

    def test(self, now):
        """x key: form a disk (at your palm if visible); press again to fire it sideways."""
        if self.state == "idle":
            if self.hand is None or now - self.last_hand_t > 0.5:
                self.hand = np.array([self.W * 0.62, self.H * 0.72], np.float32)
                self.span = self.H * 0.13
            self.forced = True
            self._start(now)
        elif self.state in ("forming", "ready"):
            self._fire(now, np.array([1.0, -0.15], np.float32))

    def _start(self, now):
        self.state, self.t0 = "forming", now
        self.pos = self._target()
        self.hist.clear()
        self.trail.clear()
        self._sound("disk_form.wav")

    def _fire(self, now, direction):
        d = direction / (np.linalg.norm(direction) + 1e-6)
        self.vel = d * FLY_SPEED * math.hypot(self.W, self.H)
        self.tilt = float(np.clip(d[1] * 0.35, -0.5, 0.5)) * (1 if d[0] >= 0 else -1)
        self.state, self.t0 = "flying", now
        self.trail.clear()
        self._hum(False)
        self._sound("disk_fire.wav")

    def _target(self):
        R = float(np.clip(DISK_SIZE * self.span, DISK_MIN * self.H, DISK_MAX * self.H))
        self.radius = R if self.state == "idle" else 0.8 * self.radius + 0.2 * R
        lift = DISK_LIFT * self.span + self.radius * DISK_ASPECT
        return self.hand + np.array([0, -lift], np.float32)

    def _swish(self, now):
        """Hand velocity (pixels/second) if the last ~0.18 s was a real swish, else None."""
        pts = [(t, p) for t, p in self.hist if now - t <= 0.18]
        if len(pts) < 3:
            return None
        (ta, pa), (tb, pb) = pts[0], pts[-1]
        if tb - ta < 0.06:
            return None
        move = pb - pa
        v = move / (tb - ta)
        if np.linalg.norm(move) < SWISH_DISTANCE * self.H or np.linalg.norm(v) < SWISH_SPEED * self.H:
            return None
        # mostly one direction (not jitter back and forth)
        path = sum(np.linalg.norm(pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1))
        return v if np.linalg.norm(move) > 0.6 * path else None

    def _update_tracks(self, hands, meta, now):
        new = []
        free = list(self.tracks)
        for lm, m in zip(hands, meta):
            palm, span = hand_geom(lm)
            tr = min(free, key=lambda t: np.linalg.norm(t.palm - palm), default=None)
            if tr is None or np.linalg.norm(tr.palm - palm) > 2.5 * span or now - tr.t > 0.5:
                tr = HandTrack(palm, now)
            else:
                free.remove(tr)
            tr.feed(lm, m, now)
            new.append(tr)
        # keep briefly-lost hands for a moment so a dropped frame doesn't reset them
        new += [t for t in free if now - t.t < 0.3]
        self.tracks = new

    def update(self, hands, meta, now, allowed=True):
        """hands: [21,2] full-res landmarks; meta: dicts with 'world' [21,3] and 'label'."""
        self._update_tracks(hands, meta, now)
        visible = [t for t in self.tracks if t.t == now]
        posed = [(t.lm, t.up) for t in visible if t.posed]
        if posed:
            self.last_pose_t = now

        # follow the tracked hand (nearest to where it was)
        cur = None
        if self.hand is not None and visible:
            tr = min(visible, key=lambda t: np.linalg.norm(t.palm - self.hand))
            palm, span = hand_geom(tr.lm)
            if np.linalg.norm(palm - self.hand) < 3.0 * self.span:
                cur = (palm, span, tr.lm)

        if self.state == "idle":
            if not allowed or now < self.cool_until:
                self.pose_since = None
                return
            if posed:
                lm = max(posed, key=lambda p: p[1])[0]
                palm, span = hand_geom(lm)
                if self.pose_since is None or self.hold_from is None or \
                        np.linalg.norm(palm - self.hold_from) > 1.5 * span:
                    self.pose_since, self.hold_from = now, palm          # (re)start the hold
                self.hand, self.span = palm, span
                self.last_hand_t = now
                if now - self.pose_since >= HOLD_SECONDS:
                    self.forced = False
                    self._start(now)
            elif self.pose_since is not None and now - self.last_pose_t > POSE_GRACE:
                self.pose_since = None                                  # really let go
            return

        if cur is not None and self.state in ("forming", "ready"):
            palm, span, lm = cur
            self.hand = 0.6 * self.hand + 0.4 * palm if self.hand is not None else palm
            self.span = 0.85 * self.span + 0.15 * span
            self.last_hand_t = now
            self.hist.append((now, palm.copy()))

        t = now - self.t0
        if self.state == "forming":
            if not allowed:
                self.cancel(now)
            elif not self.forced and now - self.last_pose_t > POSE_GRACE and t > POSE_GRACE:
                self.cancel(now)                              # let go before it formed
            elif t >= FORM_SECONDS:
                self.state, self.t0 = "ready", now
                self._hum(True)
        elif self.state == "ready":
            v = self._swish(now)
            if not allowed:
                self.cancel(now)
            elif t >= SWISH_MIN_READY and v is not None:
                self._fire(now, v)
            elif not self.forced and now - self.last_hand_t > LOST_SECONDS:
                self.cancel(now)
        elif self.state == "flying":
            if t > 2.0 or self._offscreen():
                self.state, self.cool_until = "idle", now + COOLDOWN
                self.pose_since = None
        elif self.state == "fading":
            if t >= 0.4:
                self.state, self.cool_until = "idle", now + 0.3
                self.pose_since = None

    def _offscreen(self):
        R = self.radius
        x, y = self.pos
        return x < -R * 1.5 or x > self.W + R * 1.5 or y < -R * 1.5 or y > self.H + R * 1.5

    # ---------------- drawing ----------------
    def _texture(self, now, energy, form):
        """Disk in its own flat space: (color [S,S,3] float, alpha [S,S])."""
        r, th, phi = self.r, self.th, self.spin
        swirl = th - phi + r * 1.2
        s2 = np.sin(13 * swirl - 5 * r + 1.3)
        s3 = np.sin(23 * swirl + 2.0)
        spikes = np.maximum(0, np.sin(11 * swirl + 0.7)) ** 4
        # ragged, spiky outer edge
        edge = 1 + 0.06 * (0.6 * s2 + 0.4 * s3) + 0.10 * spikes * energy + 0.03 * math.sin(now * 13)
        flicks = edge + 0.22 * np.maximum(0, np.sin(9 * swirl + 1.0)) ** 6 * energy      # short flame flicks
        body = smoothstep(edge + 0.04, edge - 0.08, r)
        reach = flicks - edge                                          # 0 between flicks
        tendril = np.clip(1 - (r - edge) / np.maximum(reach, 1e-3), 0, 1) * (reach > 0.01) * (r > edge - 0.04) * 0.7
        # streaks flowing around the rim (follow the circle, wobbling as it spins)
        streaks = 0.8 + 0.2 * np.sin(26 * r + 0.8 * np.sin(3 * swirl) + 0.6 * s2)
        core = smoothstep(0.78, 0.0, r) ** 0.6                         # broad bright centre
        to_gold = smoothstep(0.45, 0.8, r)
        to_orange = smoothstep(0.78, 1.02, r)
        band = (PALE[None, None] * (1 - to_gold[..., None]) + GOLD[None, None] * to_gold[..., None])
        band = band * (1 - to_orange[..., None]) + ORANGE[None, None] * to_orange[..., None]

        fill = body * (1 - core)
        color = (core[..., None] * WHITE
                 + fill[..., None] * band * (streaks * (0.55 + 0.45 * to_gold))[..., None]
                 + tendril[..., None] * GOLD * 0.85)
        glow = cv2.GaussianBlur(np.maximum(body, tendril), (0, 0), 5)
        color += glow[..., None] * ORANGE * 0.4
        alpha = np.clip(body * 0.9 + tendril * 0.6 + core * 0.1, 0, 1)
        window = smoothstep(1.85, 1.5, r)                                # fade out well before the texture edge
        color *= window[..., None]
        alpha *= window

        canvas = np.zeros((self.S, self.S), np.float32)               # twinkling sparkles
        c = self.S / 2
        unit = self.S * self.UNIT
        for a, rad, ph, sp in self.stars:
            rad = min(rad, 1.45)
            tw = 0.5 + 0.5 * math.sin(now * sp * 6 + ph * 6.28)
            if tw < 0.55:
                continue
            x, y = c + math.cos(a + phi * 0.2) * rad * unit, c + math.sin(a + phi * 0.2) * rad * unit
            cv2.circle(canvas, (int(x), int(y)), 2, tw, -1, cv2.LINE_AA)
        if form < 1:                                                    # energy spiralling in while forming
            for a, ph, sp in self.sparks:
                f = (now * sp + ph) % 1.0
                rad = (0.9 + 0.6 * (1 - f)) * unit
                ang = a + 2.5 * (1 - f)
                p1 = (int(c + math.cos(ang) * rad), int(c + math.sin(ang) * rad))
                p2 = (int(c + math.cos(ang + 0.35) * (rad + 0.25 * unit)),
                      int(c + math.sin(ang + 0.35) * (rad + 0.25 * unit)))
                cv2.line(canvas, p1, p2, f * (1 - form) + 0.2, 2, cv2.LINE_AA)
        color += canvas[..., None] * PALE
        alpha = np.maximum(alpha, canvas * 0.8)
        return color, alpha

    def _stamp(self, out, color, alpha, center, R, strength):
        """Warp the flat disk onto the frame as a tilted ellipse."""
        S, unit = self.S, self.S * self.UNIT
        sx, sy = R / unit, R * DISK_ASPECT / unit
        c, s = math.cos(self.tilt), math.sin(self.tilt)
        A = np.array([[c * sx, -s * sy], [s * sx, c * sy]], np.float32)
        t = np.asarray(center, np.float32) - A @ np.array([S / 2, S / 2], np.float32)
        ext = 1.9 * R
        x0, y0 = int(max(0, center[0] - ext)), int(max(0, center[1] - ext))
        x1, y1 = int(min(self.W, center[0] + ext)), int(min(self.H, center[1] + ext))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return out
        M = np.hstack([A, (t - (x0, y0))[:, None]]).astype(np.float32)
        size = (x1 - x0, y1 - y0)
        col = cv2.warpAffine(color, M, size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        a = cv2.warpAffine(alpha, M, size, flags=cv2.INTER_LINEAR, borderValue=0) * strength
        roi = out[y0:y1, x0:x1].astype(np.float32)
        roi = roi * (1 - a[..., None]) + col * 255 * strength
        out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        return out

    def render(self, frame, now):
        if self.state == "idle" or self.pos is None:
            self.last_now = now
            return frame
        dt = 0.0 if self.last_now is None else min(0.1, now - self.last_now)
        self.last_now = now
        t = now - self.t0
        out = frame.copy()

        if self.state in ("forming", "ready"):
            target = self._target()
            self.pos = 0.55 * self.pos + 0.45 * target
            bob = np.array([0, math.sin(now * 3.0) * 0.04 * self.radius], np.float32)
            form = min(1.0, t / FORM_SECONDS) if self.state == "forming" else 1.0
            self.spin += dt * SPIN_SPEED * (0.4 + 0.6 * form)
            self.tilt = 0.85 * self.tilt
            R = self.radius * (0.15 + 0.85 * smoothstep(0, 1, form))
            color, alpha = self._texture(now, 0.4 + 0.6 * form, form)
            return self._stamp(out, color, alpha, self.pos + bob, R, 0.5 + 0.5 * form)

        if self.state == "flying":
            self.trail.appendleft(self.pos.copy())
            self.vel = self.vel * (1 + FLY_ACCEL * dt)
            self.pos = self.pos + self.vel * dt
            self.spin += dt * SPIN_SPEED * 2.5
            color, alpha = self._texture(now, 1.0, 1.0)
            for i, p in enumerate(list(self.trail)[1:], 1):            # motion trail
                out = self._stamp(out, color, alpha, p, self.radius * (1 - 0.08 * i), 0.45 / i)
            return self._stamp(out, color, alpha, self.pos, self.radius, 1.0)

        if self.state == "fading":
            k = min(1.0, t / 0.4)
            self.spin += dt * SPIN_SPEED
            color, alpha = self._texture(now, 1.0, 1.0)
            return self._stamp(out, color, alpha, self.pos, self.radius * (1 - 0.6 * k) * max(self.fade_from, 0.2),
                               (1 - k) * max(self.fade_from, 0.3))
        return out

    def debug(self, img, hands, meta):
        """Per hand: arrow length = how much the palm faces up (smoothed; points down if it faces
        down), circle = fingers stretched, green = pose recognised."""
        for tr in self.tracks:
            if tr.lm is None:
                continue
            palm, span = hand_geom(tr.lm)
            col = (0, 220, 0) if tr.posed else (0, 140, 255)
            tip = palm + np.array([0, -tr.up * span * 0.8])
            cv2.arrowedLine(img, tuple(palm.astype(int)), tuple(tip.astype(int)), col, 3, tipLength=0.3)
            if tr.stretched >= 0.45:
                cv2.circle(img, tuple(palm.astype(int)), int(span * 0.15), col, 2)
        return img
