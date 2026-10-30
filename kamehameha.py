"""
Kamehameha: gesture detection, sounds and rendering.

Pose: both hands open, wrists together, palms apart, like a clam shell.
  Hold it and a blue energy ball charges between your palms, then it fires.
  The beam goes the way your fingers point. Point the opening of your hands
  straight at the camera (one hand's fingers up, the other's down) and it blasts
  toward the viewer instead. The beam lasts while you hold the pose.
"""

import math
import os
import wave

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
HOLD_SECONDS = 0.3        # hold the pose this long before charging starts
CHARGE_SECONDS = 1.3      # charging time before it fires
BEAM_MIN = 1.5            # beam lasts at least this long once fired
BEAM_MAX = 4.0            # ...and at most this long
BEAM_END = 0.35           # fade-out time
POSE_GRACE = 0.35         # tracking may drop out briefly without cancelling
COOLDOWN = 1.0
FRONTAL_SPREAD = 0.45     # lower = harder to get the toward-the-camera blast

BEAM_WIDTH = 1.0          # beam thickness multiplier
SHAKE = 1.0               # screen shake multiplier (0 = off)

# BGR colors, 0..1
DEEP_BLUE = np.array([1.0, 0.30, 0.05], np.float32)
CYAN = np.array([1.0, 0.62, 0.30], np.float32)      # light electric blue
WHITE = np.array([1.0, 1.0, 1.0], np.float32)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# Gesture
# --------------------------------------------------------------------------
def _hand_open(lm):
    d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b]))
    extended = sum(d(0, tip) > d(0, pip) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
    return extended >= 3


def _palm(lm):
    return lm[[0, 5, 9, 13, 17]].mean(axis=0)


def detect_pose(hands):
    """hands: list of (21, 2) landmark arrays in output-pixel coordinates."""
    best = None
    for i in range(len(hands)):
        for j in range(i + 1, len(hands)):
            a, b = hands[i], hands[j]
            la, lb = np.linalg.norm(a[9] - a[0]), np.linalg.norm(b[9] - b[0])
            size = (la + lb) / 2 + 1e-6
            wrist_gap = np.linalg.norm(a[0] - b[0])
            if wrist_gap > 0.9 * size:
                continue                                    # wrists not together
            if np.linalg.norm(a[12] - b[12]) < 0.8 * size:
                continue                                    # palms pressed together (praying/clasped)
            if not (_hand_open(a) and _hand_open(b)):
                continue
            va, vb = (a[9] - a[0]) / (la + 1e-6), (b[9] - b[0]) / (lb + 1e-6)
            avg = (va + vb) / 2
            spread = float(np.linalg.norm(avg))             # ~0 when the hands open toward the camera
            pose = dict(center=(_palm(a) + _palm(b)) / 2,
                        dir=avg / (spread + 1e-6), spread=spread, size=float(size),
                        gap=float(wrist_gap / size))
            if best is None or pose["gap"] < best["gap"]:
                best = pose
    return best


# --------------------------------------------------------------------------
# Sounds (original, synthesized on first run)
# --------------------------------------------------------------------------
def _write_wav(path, s, sr):
    s = 0.85 * s / (np.max(np.abs(s)) + 1e-9)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((s * 32767).astype(np.int16).tobytes())


def _lowpass(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def ensure_sounds(sound_dir):
    os.makedirs(sound_dir, exist_ok=True)
    sr = 44100
    rng = np.random.default_rng(11)

    path = os.path.join(sound_dir, "kame_charge.wav")
    if not os.path.exists(path):
        dur = CHARGE_SECONDS + 0.15
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        k = t / dur
        f = 70 + 170 * k ** 1.5
        ph = 2 * np.pi * np.cumsum(f) / sr
        hum = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3.01 * ph)
        hum *= 1 + 0.35 * np.sin(2 * np.pi * (5 + 18 * k) * t)        # wobble speeds up
        air = _lowpass(rng.standard_normal(len(t)), 6) * (0.2 + 0.8 * k)
        crackle = np.zeros(len(t))
        hits = rng.random(len(t)) < (0.0004 + 0.003 * k)
        crackle[hits] = rng.uniform(-1, 1, hits.sum())
        ct = np.arange(int(0.006 * sr)) / sr
        crackle = np.convolve(crackle, np.sin(2 * np.pi * 3200 * ct) * np.exp(-ct * 900), mode="same")
        env = np.minimum(1, t / 0.05) * (0.35 + 0.65 * k)
        _write_wav(path, env * (0.6 * hum + 0.25 * air + 0.8 * crackle), sr)

    path = os.path.join(sound_dir, "kame_fire.wav")
    if not os.path.exists(path):
        dur = BEAM_MAX + 0.4
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # boom
        bt = t[: int(0.5 * sr)]
        boom = np.sin(2 * np.pi * np.cumsum(130 * np.exp(-bt * 5) + 35) / sr) * np.exp(-bt * 6)
        burst = rng.standard_normal(len(bt)) * np.exp(-bt * 18)
        # roar
        roar = _lowpass(rng.standard_normal(len(t)), 25) * 3.0
        roar += 0.5 * _lowpass(rng.standard_normal(len(t)), 4)
        vib = 1 + 0.02 * np.sin(2 * np.pi * 7 * t)
        saw = sum(np.sin(2 * np.pi * np.cumsum(f0 * vib * n) / sr) / n
                  for f0 in (55, 82.5) for n in (1, 2, 3, 4))
        whine = 0.25 * np.sin(2 * np.pi * np.cumsum(900 - 250 * np.minimum(1, t / 1.5)) / sr)
        env = np.minimum(1, t / 0.04) * np.clip((dur - t) / 0.8, 0, 1)
        s = env * (0.5 * roar + 0.35 * saw + whine)
        s[: len(bt)] += 1.4 * boom + 0.6 * burst
        _write_wav(path, s, sr)


# --------------------------------------------------------------------------
# Effect
# --------------------------------------------------------------------------
class Kamehameha:
    def __init__(self, W, H, play_sound, stop_sound):
        self.W, self.H = W, H
        self.q = 4                                         # soft glow fields at 1/4 res
        self.w, self.h = W // self.q, H // self.q
        self.lq = 2                                        # crisp lines (streaks, sparks) at 1/2 res
        self.lw, self.lh = W // self.lq, H // self.lq
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.xx, self.yy = xx, yy
        self.play, self.stop = play_sound, stop_sound
        self.rng = np.random.default_rng()

        self.state = "idle"
        self.t0 = 0.0
        self.pose_since = None
        self.last_pose_t = -10.0
        self.cool_until = 0.0
        self.forced = False
        self.snd = None

        self.center = None
        self.dir = np.array([1.0, 0.0], np.float32)
        self.size = H * 0.1
        self.spread = 1.0
        self.frontal = False

        r = self.rng
        self.ball_waves = [(n, r.uniform(0, 6.3), r.uniform(-9, 9)) for n in (5, 7, 11, 13, 17)]
        self.edge_waves = [(r.uniform(0.02, 0.05), r.uniform(25, 40), r.uniform(0, 6.3)),
                           (r.uniform(0.07, 0.12), r.uniform(45, 70), r.uniform(0, 6.3)),
                           (r.uniform(0.18, 0.3), r.uniform(80, 120), r.uniform(0, 6.3))]
        self.streaks = [(r.uniform(-0.95, 0.95), r.uniform(25, 70), r.uniform(0, 1), r.uniform(4, 9))
                        for _ in range(16)]
        self.rays = [(r.uniform(-np.pi, np.pi), r.uniform(25, 60), r.uniform(0, 1), r.uniform(3, 7))
                     for _ in range(28)]
        self.sparks = [(r.uniform(0, 2 * np.pi), r.uniform(0, 1), r.uniform(1.2, 2.2)) for _ in range(26)]

    # ---------------- state ----------------
    @property
    def busy(self):
        return self.state != "idle"

    def _sound(self, name):
        if self.snd is not None:
            self.stop(self.snd)
        self.snd = self.play(name)

    def _start_charge(self, now):
        self.state, self.t0 = "charging", now
        self._sound("kame_charge.wav")

    def force(self, now):
        """Test without the gesture (k key)."""
        if self.busy:
            return
        if self.center is None or now - self.last_pose_t > 1.0:
            self.center = np.array([self.W * 0.62, self.H * 0.62], np.float32)
            self.dir = np.array([1.0, 0.0], np.float32)
            self.size = self.H * 0.1
            self.spread = 1.0
        self.forced = True
        self._start_charge(now)

    def _track(self, pose, now):
        fresh = now - self.last_pose_t > 0.5 or self.center is None
        if self.state == "firing":
            a_pos, a_dir = 0.3, 0.15
        else:
            a_pos, a_dir = 0.5, 0.35
        if fresh:
            a_pos = a_dir = 1.0
        self.center = (1 - a_pos) * (self.center if not fresh else pose["center"]) + a_pos * pose["center"]
        d = (1 - a_dir) * self.dir + a_dir * pose["dir"]
        self.dir = (d / (np.linalg.norm(d) + 1e-6)).astype(np.float32)
        self.size = (1 - a_pos) * self.size + a_pos * pose["size"] if not fresh else pose["size"]
        if self.state != "firing":                          # frontal/side is locked once fired
            self.spread = (1 - a_dir) * self.spread + a_dir * pose["spread"] if not fresh else pose["spread"]
        self.last_pose_t = now

    def update(self, pose, now):
        if pose is not None:
            self._track(pose, now)
        lost = now - self.last_pose_t > POSE_GRACE

        if self.state == "idle":
            if pose is None or now < self.cool_until:
                self.pose_since = None
                return
            if self.pose_since is None:
                self.pose_since = now
            if now - self.pose_since >= HOLD_SECONDS:
                self.forced = False
                self._start_charge(now)

        elif self.state == "charging":
            if lost and not self.forced:                    # let go: cancel
                if self.snd is not None:
                    self.stop(self.snd)
                self.state, self.pose_since = "idle", None
                self.cool_until = now + 0.3
                return
            if now - self.t0 >= CHARGE_SECONDS:
                self.state, self.t0 = "firing", now
                self.frontal = self.spread < FRONTAL_SPREAD
                self._sound("kame_fire.wav")

        elif self.state == "firing":
            t = now - self.t0
            done = t >= BEAM_MAX or (t >= BEAM_MIN and (lost or (self.forced and t >= 2.5)))
            if done:
                self.state, self.t0 = "ending", now

        elif self.state == "ending":
            if now - self.t0 >= BEAM_END:
                if self.snd is not None:
                    self.stop(self.snd)
                    self.snd = None
                self.state, self.pose_since = "idle", None
                self.cool_until = now + COOLDOWN

    # ---------------- fields (all at 1/q resolution) ----------------
    def origin(self):
        if self.frontal and self.state in ("firing", "ending"):
            return self.center
        if self.spread < FRONTAL_SPREAD:
            return self.center
        return self.center + self.dir * self.size * 0.9

    def _ball(self, o, r, now, energy):
        dx, dy = self.xx - o[0], self.yy - o[1]
        dist = np.sqrt(dx * dx + dy * dy)
        th = np.arctan2(dy, dx)
        spk = sum(np.sin(n * th + ph + sp * now) for n, ph, sp in self.ball_waves) / len(self.ball_waves)
        spk = np.maximum(0, spk) ** 1.5
        flick = 1 + 0.06 * math.sin(now * 47) + 0.04 * math.sin(now * 31)
        r = r * flick
        rr = r * (1 + 0.9 * energy * spk)
        outer = smoothstep(rr * 1.5, rr * 0.75, dist)
        mid = smoothstep(r * 1.0, r * 0.6, dist)
        core = smoothstep(r * 0.7, r * 0.2, dist)
        return outer, mid, core

    def _beam(self, o, d, hw, L, now):
        p = np.array([-d[1], d[0]], np.float32)
        dx, dy = self.xx - o[0], self.yy - o[1]
        u = dx * d[0] + dy * d[1]
        v = np.abs(dx * p[0] + dy * p[1])
        w = hw * (1 + 0.6 * np.clip(u / (self.w * 0.9), 0, 1))     # widens as it travels
        jag = sum(np.sin(f * u - s * now + ph) for f, s, ph in self.edge_waves) / len(self.edge_waves)
        jag2 = np.sin(0.45 * u - 160 * now)                           # fine spiky fringe
        w_out = w * (1.5 + 0.45 * jag + 0.18 * jag2)
        along = smoothstep(-hw * 0.3, hw * 0.4, u) * smoothstep(L, L - hw * 1.5, u)
        outer = smoothstep(w_out, w_out * 0.6, v) * along
        mid = smoothstep(w * 1.0, w * 0.72, v) * along
        core = smoothstep(w * 0.62, w * 0.25, v) * along
        return outer, mid, core

    def _blast(self, o, Rb, now):
        dx, dy = self.xx - o[0], self.yy - o[1]
        dist = np.sqrt(dx * dx + dy * dy)
        th = np.arctan2(dy, dx)
        spk = sum(np.sin(n * th + ph + sp * now) for n, ph, sp in self.ball_waves) / len(self.ball_waves)
        rr = Rb * (1 + 0.28 * np.maximum(0, spk) + 0.08 * np.sin(23 * th - 30 * now))
        outer = smoothstep(rr * 1.3, rr * 0.85, dist)
        mid = smoothstep(Rb * 0.9, Rb * 0.6, dist)
        core = smoothstep(Rb * 0.65, Rb * 0.2, dist)
        return outer, mid, core

    def _beam_streaks(self, o, d, hw, L, now, e):
        """Speed lines racing along the beam. o, hw, L in full-res pixels."""
        f = self.lq
        canvas = np.zeros((self.lh, self.lw), np.float32)
        p = np.array([-d[1], d[0]])
        for c, ln, ph, spd in self.streaks:
            seg = ln * hw / 25                              # dash length scales with beam
            period = seg * 2.5
            off = ((now * spd * 60 + ph * period) % period)
            u = off
            while u < L:
                widen = 1 + 0.6 * min(1.0, u / (self.W * 0.9))
                v = c * hw * 1.3 * widen
                a = o + d * u + p * v
                b = o + d * min(u + seg, L) + p * v
                cv2.line(canvas, (int(a[0] / f), int(a[1] / f)), (int(b[0] / f), int(b[1] / f)),
                         0.9 * e, 1, cv2.LINE_AA)
                u += period
        return canvas

    def _blast_rays(self, o, Rb, now, e):
        f = self.lq
        canvas = np.zeros((self.lh, self.lw), np.float32)
        for a, ln, ph, spd in self.rays:
            seg = ln * Rb / 120
            period = seg * 2.5
            rad = Rb * 0.4 + ((now * spd * 60 + ph * period) % period)
            c, s_ = math.cos(a), math.sin(a)
            while rad < Rb * 1.35:
                p1 = (int((o[0] + c * rad) / f), int((o[1] + s_ * rad) / f))
                p2 = (int((o[0] + c * (rad + seg)) / f), int((o[1] + s_ * (rad + seg)) / f))
                cv2.line(canvas, p1, p2, 0.9 * e, 1, cv2.LINE_AA)
                rad += period
        return canvas

    def _sparks(self, o, r, now, k):
        """o, r in full-res pixels."""
        canvas = np.zeros((self.lh, self.lw), np.float32)
        o = o / self.lq
        r = r / self.lq
        for a, ph, spd in self.sparks:
            f = (now * spd + ph) % 1.0
            rad = r * (1.2 + 3.5 * (1 - f))
            tail = r * (0.25 + 0.9 * (1 - f))
            c, s = math.cos(a + 0.6 * f), math.sin(a + 0.6 * f)
            p1 = (int(o[0] + c * rad), int(o[1] + s * rad))
            p2 = (int(o[0] + c * (rad + tail)), int(o[1] + s * (rad + tail)))
            cv2.line(canvas, p1, p2, float(f * (0.4 + 0.6 * k)), 1, cv2.LINE_AA)
        if self.rng.random() < 0.25 + 0.5 * k:             # little lightning arcs
            for _ in range(int(1 + 2 * k)):
                a0 = self.rng.uniform(0, 2 * np.pi)
                pts = []
                for i in range(7):
                    a = a0 + i * 0.22
                    rad = r * self.rng.uniform(1.05, 1.6)
                    pts.append((o[0] + math.cos(a) * rad, o[1] + math.sin(a) * rad))
                cv2.polylines(canvas, [np.array(pts, np.int32)], False, 1.0, 1, cv2.LINE_AA)
        return canvas

    # ---------------- render ----------------
    def render(self, frame, now):
        if self.state == "idle" or self.center is None:
            return frame
        q = self.q
        t = now - self.t0
        o_full = self.origin()
        o = o_full / q
        R_full = max(self.size, self.H * 0.06)
        R = R_full / q
        lines = None
        shake = 0.0
        tint = 0.0

        if self.state == "charging":
            k = min(1.0, t / CHARGE_SECONDS)
            r = R * (0.2 + 0.7 * k)
            outer, mid, core = self._ball(o, r, now, 0.3 + 0.5 * k)
            lines = self._sparks(o_full, r * q, now, k)
            tint = 0.08 * k
            shake = 1.5 * k
        else:
            if self.state == "firing":
                e = smoothstep(0, 0.12, t)
                tf = t
            else:
                e = 1 - smoothstep(0, BEAM_END, t)
                tf = BEAM_MIN + t
            if self.frontal:
                grow = smoothstep(0, 0.35, tf)
                Rb = (R * 1.2 + (min(self.w, self.h) * 0.42 - R * 1.2) * grow) * (1 + 0.04 * math.sin(now * 37))
                Rb *= max(e, 0.05)
                outer, mid, core = self._blast(o, Rb, now)
                lines = self._blast_rays(o_full, Rb * q, now, e)
            else:
                hw = R * 0.85 * BEAM_WIDTH * (0.3 + 0.7 * e)
                diag = math.hypot(self.w, self.h)
                L = min(diag * 1.3, tf * diag / 0.22)
                outer, mid, core = self._beam(o, self.dir, hw, L, now)
                bo, bm, bc = self._ball(o, R * 1.15 * (0.4 + 0.6 * e), now, 0.7)
                outer, mid, core = np.maximum(outer, bo), np.maximum(mid, bm), np.maximum(core, bc)
                outer, mid, core = outer * e, mid * e, core * e
                lines = self._beam_streaks(o_full, self.dir, hw * q, L * q, now, e)
            tint = 0.22 * e
            shake = 9.0 * e

        glow = (outer[..., None] * DEEP_BLUE * 0.8 + mid[..., None] * CYAN * 0.75
                + core[..., None] * WHITE)
        glow += cv2.GaussianBlur(outer, (0, 0), 5)[..., None] * DEEP_BLUE * 0.6     # bloom
        alpha = np.clip(outer * 0.35 + mid * 0.6 + core, 0, 1)

        size = (self.W, self.H)
        lh, lw = self.lh, self.lw
        glow = cv2.resize(glow, (lw, lh), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (lw, lh), interpolation=cv2.INTER_LINEAR)
        if lines is not None:
            lines = np.clip(lines, 0, 1)
            glow = glow + lines[..., None] * WHITE
            alpha = np.maximum(alpha, lines)
        glow = cv2.resize(np.clip(glow * 255, 0, 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
        inv = cv2.resize(((1 - alpha) * 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)

        base = frame
        if tint > 0:
            tint_img = np.empty_like(frame)
            tint_img[:] = (DEEP_BLUE * 0.6 * 255).astype(np.uint8)
            base = cv2.addWeighted(frame, 1 - tint, tint_img, tint, 0)
        out = cv2.multiply(base, cv2.merge([inv, inv, inv]), scale=1 / 255.0)
        out = cv2.add(out, glow)

        shake *= SHAKE
        if shake > 0.5:
            dx, dy = self.rng.uniform(-shake, shake, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out = cv2.warpAffine(out, M, size, borderMode=cv2.BORDER_REFLECT)
        return out

    def debug(self, img, pose):
        if pose is None:
            return img
        c = tuple(pose["center"].astype(int))
        e = tuple((pose["center"] + pose["dir"] * pose["size"] * 2).astype(int))
        col = (255, 200, 0) if pose["spread"] >= FRONTAL_SPREAD else (0, 200, 255)
        cv2.circle(img, c, int(pose["size"] * 0.5), col, 2)
        cv2.arrowedLine(img, c, e, col, 3, tipLength=0.3)
        return img
