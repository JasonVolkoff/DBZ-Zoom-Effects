"""
Power-up aura, three levels.

Gesture: both hands visible, clenched into fists. Hold briefly.
  Level 1 (yellow): from no aura.  Hair: overlays/hair_transformation.(png|gif)
  Level 2 (wine red): fists again while at level 1.  Hair: hair_transformation_2
  Level 3 (azure blue): fists again while at level 2.  Hair: hair_transformation_3
Each power-up: thin crackles of electricity (in that level's color) flicker
around your body while the aura builds, then a flash. Open your hands between
power-ups so one long fist doesn't skip levels. Press p to dissipate.
"""

import math
import os
import wave

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
HOLD_SECONDS = 0.5          # hold both fists this long to start
POWERUP_SECONDS = 2.0       # crackles + aura building
FLASH_SECONDS = 0.18        # bright flash at the end of the power-up
DISSIPATE_SECONDS = 0.8
COOLDOWN = 1.0

AURA_REACH = 0.14           # how far flames reach from your body (fraction of frame height)
AURA_RISE = 0.22            # extra reach upward (flames lick up)
AURA_OPACITY = 0.6
AURA_FLICKER_SPEED = 1.0
OUTLINE_STRENGTH = 0.35     # the subtle yellow outline (0 = off)
CRACKLES_WHILE_ACTIVE = 0.15  # occasional sparks while the aura is on (0 = none)
SHAKE = 1.0

REARM_SECONDS = 0.3         # hands must be open this long before the next power-up counts
MAX_LEVEL = 3

# Colors per level, BGR 0..1
#   edge:  outer flames      core: inner flames       outline: rim around your body
#   spark_core / spark_glow: the lightning crackles   flash: ignition flash
_c = lambda *v: np.array(v, np.float32)
LEVELS = {
    1: dict(name="yellow",
            edge=_c(0.10, 0.80, 1.00), core=_c(0.90, 0.98, 1.00), outline=_c(0.25, 0.90, 1.00),
            spark_core=_c(1.0, 1.0, 1.0), spark_glow=_c(1.0, 0.80, 0.55), spark_boost=1.5,  # pale electric blue
            flash=(180, 245, 255), opacity=1.0),
    2: dict(name="wine red",
            edge=_c(0.16, 0.02, 0.42), core=_c(0.30, 0.20, 0.66), outline=_c(0.22, 0.06, 0.66),
            spark_core=_c(0.40, 0.36, 1.0), spark_glow=_c(0.10, 0.02, 1.0),   # red lightning
            flash=(150, 140, 240), spark_boost=2.2, opacity=1.45),   # denser so it stays wine, not pink
    3: dict(name="azure",
            edge=_c(1.00, 0.50, 0.00), core=_c(1.00, 0.95, 0.85), outline=_c(1.00, 0.62, 0.15),
            spark_core=_c(1.0, 0.78, 0.40), spark_glow=_c(1.0, 0.45, 0.02),   # blue lightning
            flash=(255, 225, 180), spark_boost=2.2, opacity=1.0),
}


def _mix(a, b, k):
    return {key: (a[key] * (1 - k) + b[key] * k if isinstance(a[key], (np.ndarray, float)) else
                  (b[key] if k >= 0.5 else a[key])) for key in a}


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------
# Gesture
# --------------------------------------------------------------------------
def is_fist(lm):
    d = lambda a, b: float(np.linalg.norm(lm[a] - lm[b]))
    curled = sum(d(0, tip) < d(0, pip) for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)))
    return curled >= 3


def detect_fists(hands):
    return sum(is_fist(lm) for lm in hands) >= 2


# --------------------------------------------------------------------------
# Sounds (original, synthesized on first run)
# --------------------------------------------------------------------------
def _write_wav(path, s, sr, peak=0.85):
    s = peak * s / (np.max(np.abs(s)) + 1e-9)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((s * 32767).astype(np.int16).tobytes())


def _lp(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def _crackle(n, rate, rng, sr):
    c = np.zeros(n)
    hits = rng.random(n) < rate
    c[hits] = rng.uniform(-1, 1, hits.sum())
    ct = np.arange(int(0.004 * sr)) / sr
    return np.convolve(c, np.sin(2 * np.pi * 4200 * ct) * np.exp(-ct * 1200), mode="same")


def _powerup_sound(path, pitch, level, sr):
    if os.path.exists(path):
        return
    rng = np.random.default_rng(23 + level)
    dur = POWERUP_SECONDS + 0.9
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    k = np.clip(t / POWERUP_SECONDS, 0, 1)
    rumble = _lp(rng.standard_normal(len(t)), 60) * 6 * (0.3 + 0.7 * k)
    f = (45 + 120 * k ** 2) * pitch
    tone = np.sin(2 * np.pi * np.cumsum(f) / sr) * (0.2 + 0.8 * k)
    crack = _crackle(len(t), (0.0008 + 0.006 * k) * (1 + 0.4 * (level - 1)), rng, sr) * 1.2
    s = 0.5 * rumble + 0.4 * tone + crack
    # burst when the aura ignites
    i0 = int(POWERUP_SECONDS * sr)
    bt = t[: len(t) - i0]
    burst = (_lp(rng.standard_normal(len(bt)), 8) * 2 + np.sin(2 * np.pi * 70 * pitch * bt)) * np.exp(-bt * 4)
    s[i0:] = s[i0:] * np.exp(-bt * 6) + 1.3 * burst
    s *= np.minimum(1, t / 0.05)
    _write_wav(path, s, sr)



def ensure_sounds(sound_dir):
    os.makedirs(sound_dir, exist_ok=True)
    sr = 44100
    rng = np.random.default_rng(23)

    for level, pitch in ((1, 1.0), (2, 1.25), (3, 1.55)):
        name = "powerup.wav" if level == 1 else f"powerup_{level}.wav"
        _powerup_sound(os.path.join(sound_dir, name), pitch, level, sr)

    path = os.path.join(sound_dir, "aura_loop.wav")
    marker = os.path.join(sound_dir, ".aura_loop_v2")         # v2: seamless (no fades at the ends)
    if not os.path.exists(path) or not os.path.exists(marker):
        dur = 4.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        roar = _lp(rng.standard_normal(len(t)), 40) * 5
        roar *= 1 + 0.25 * np.sin(2 * np.pi * 0.5 * t) + 0.15 * np.sin(2 * np.pi * 3.25 * t)
        hum = np.sin(2 * np.pi * 55 * t) + 0.4 * np.sin(2 * np.pi * 110 * t)
        s = 0.6 * roar + 0.25 * hum + 0.3 * _crackle(len(t), 0.0004, rng, sr)
        # all tones/modulations complete whole cycles in 4 s, so it loops cleanly
        _write_wav(path, s, sr, peak=0.35)
        open(marker, "w").close()

    path = os.path.join(sound_dir, "aura_down.wav")
    if not os.path.exists(path):
        dur = DISSIPATE_SECONDS + 0.3
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        whoosh = _lp(rng.standard_normal(len(t)), 20) * 4 * np.exp(-t * 3)
        tone = np.sin(2 * np.pi * np.cumsum(220 * np.exp(-t * 2.5) + 40) / sr) * np.exp(-t * 3.5)
        _write_wav(path, (0.6 * whoosh + 0.5 * tone) * np.minimum(1, t / 0.02), sr, peak=0.6)


# --------------------------------------------------------------------------
# Effect
# --------------------------------------------------------------------------
class PowerUp:
    def __init__(self, W, H, play_sound, stop_sound, loop=None):
        self.W, self.H = W, H
        self.q = 4
        self.w, self.h = W // self.q, H // self.q
        self.lq = 2
        self.lw, self.lh = W // self.lq, H // self.lq
        self.play, self.stop = play_sound, stop_sound
        self.rng = np.random.default_rng()

        self.state = "idle"
        self.t0 = 0.0
        self._t = 0.0
        self.fist_since = None
        self.cool_until = 0.0
        self.snd = None
        self.loop_snd = None
        self.loop = loop           # gapless looper for the hum (audio.LoopSound)
        self.looping = False
        self.dissipate_from = 1.0
        self.level = 0             # current aura level (0 = none)
        self.target = 0            # level being powered up to
        self.armed = True          # fists must open between power-ups
        self.open_since = None

        # flame noise texture, tiled
        r = np.random.default_rng(9)
        base = r.random((32, 32)).astype(np.float32)
        tex = cv2.resize(np.tile(base, (3, 3)), (768, 768), interpolation=cv2.INTER_CUBIC)[256:512, 256:512]
        self.tex = cv2.normalize(tex, None, 0, 1, cv2.NORM_MINMAX)
        base2 = r.random((64, 64)).astype(np.float32)
        tex2 = cv2.resize(np.tile(base2, (3, 3)), (768, 768), interpolation=cv2.INTER_CUBIC)[256:512, 256:512]
        self.tex2 = cv2.normalize(tex2, None, 0, 1, cv2.NORM_MINMAX)
        yy, xx = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.xx, self.yy = xx, yy

    # ---------------- state ----------------
    @property
    def busy(self):
        return self.state != "idle"

    @property
    def hair_level(self):
        """Which transformation hair to show (0 = your normal hair)."""
        if self.state == "active":
            return self.level
        if self.state == "powering":
            return self.target if self._t >= POWERUP_SECONDS else self.level
        if self.state == "dissipating":
            return self.level if self._t < DISSIPATE_SECONDS * 0.4 else 0
        return 0

    @property
    def transformed(self):
        return self.hair_level > 0

    def _stop_hum(self):
        if self.loop is not None and self.looping:
            self.loop.stop()
        self.looping = False
        if self.loop_snd is not None:
            self.stop(self.loop_snd)
            self.loop_snd = None

    def _sound(self, name):
        if self.snd is not None:
            self.stop(self.snd)
        self.snd = self.play(name)

    def _power_to(self, level, now):
        self.state, self.t0, self._t = "powering", now, 0.0
        self.target = level
        self._stop_hum()
        self._sound("powerup.wav" if level == 1 else f"powerup_{level}.wav")

    def start(self, now, force=False):
        """From no aura to level 1."""
        if self.state != "idle" or (now < self.cool_until and not force):
            return
        self.level = 0
        self._power_to(1, now)

    def level_up(self, now):
        """Next level, only from an active aura below the max."""
        if self.state == "active" and self.level < MAX_LEVEL:
            self._power_to(self.level + 1, now)
            return True
        return False

    def dissipate(self, now):
        if self.state in ("powering", "active"):
            if self.state == "powering":
                k = min(1.0, max(0.0, (now - self.t0) / POWERUP_SECONDS))
                if self.level == 0:
                    self.dissipate_from = k
                else:                                       # cancelled mid level-up: fade the old level
                    self.dissipate_from = 1.0
                    self.target = self.level
            else:
                self.dissipate_from = 1.0
            self.state, self.t0 = "dissipating", now
            self._stop_hum()
            self._sound("aura_down.wav")

    def toggle(self, now):
        if self.state == "idle":
            self.start(now, force=True)
        else:
            self.dissipate(now)

    def update(self, fists, now):
        self._t = now - self.t0
        # fists must be released (for a moment) before they count again
        if not fists:
            if self.open_since is None:
                self.open_since = now
            if now - self.open_since >= REARM_SECONDS:
                self.armed = True
            self.fist_since = None
        else:
            self.open_since = None

        if self.state in ("idle", "active") and fists and self.armed:
            ready = self.state == "active" and self.level < MAX_LEVEL or \
                    (self.state == "idle" and now >= self.cool_until)
            if ready:
                if self.fist_since is None:
                    self.fist_since = now
                if now - self.fist_since >= HOLD_SECONDS:
                    self.armed = False
                    self.fist_since = None
                    if self.state == "idle":
                        self.start(now)
                    else:
                        self.level_up(now)
                    return

        if self.state == "powering":
            if self._t >= POWERUP_SECONDS + FLASH_SECONDS:
                self.level = self.target
                self.state, self.t0 = "active", now
                self._t = 0.0
        elif self.state == "active":
            if self.loop is not None:                     # gapless loop
                if not self.looping:
                    self.loop.start("aura_loop.wav")
                    self.looping = True
                return
            # fallback: relaunch the player as soon as it ends
            if self.loop_snd is not None:
                finished = self.loop_snd.poll() is not None
            else:
                finished = now - getattr(self, "loop_t", -10) > 3.9
            if finished:
                self.loop_snd = self.play("aura_loop.wav")
                self.loop_t = now
        elif self.state == "dissipating":
            if self._t >= DISSIPATE_SECONDS:
                self.state, self.fist_since = "idle", None
                self.level = self.target = 0
                self.cool_until = now + COOLDOWN

    # ---------------- drawing ----------------
    def _aura_fields(self, mask_small, now, reach_scale):
        """Returns (alpha, color) at 1/q res."""
        h = self.h
        mb = (mask_small > 0.5).astype(np.uint8)
        if mb.sum() < 20:
            return None
        # stretch the silhouette upward so flames rise above head and shoulders
        rise = int(AURA_RISE * h * reach_scale)
        up = mb.copy()
        steps = 8
        for i in range(1, steps + 1):
            dy = int(rise * i / steps)
            if dy <= 0:
                continue
            shifted = np.zeros_like(mb)
            shifted[:-dy] = mb[dy:]
            up = np.maximum(up, shifted)
        dist = cv2.distanceTransform(1 - up, cv2.DIST_L2, 3)
        reach = max(2.0, AURA_REACH * h * reach_scale)
        field = np.clip(1 - dist / reach, 0, 1)
        # fade the stretched part so the top tongues taper
        body_dist = cv2.distanceTransform(1 - mb, cv2.DIST_L2, 3)
        field *= np.clip(1.15 - body_dist / (reach + rise + 1e-6), 0, 1)

        # rising flame noise: stretched vertically, scrolling up
        sp = AURA_FLICKER_SPEED
        mx = (self.xx * 2.2) % 256
        my = (self.yy * 0.4 + now * 90 * sp) % 256
        n1 = cv2.remap(self.tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        mx2 = (self.xx * 4.5 + 40) % 256
        my2 = (self.yy * 1.1 + now * 160 * sp) % 256
        n2 = cv2.remap(self.tex2, mx2, my2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n = 0.65 * n1 + 0.35 * n2

        v = field + 0.85 * (n - 0.5)
        flame = smoothstep(0.32, 0.44, v)
        inner = smoothstep(0.6, 0.9, v)
        # keep your body clear: the aura only surrounds you
        keep_out = 1 - cv2.GaussianBlur(mb.astype(np.float32), (0, 0), 1.2)
        flame *= keep_out
        inner *= keep_out
        alpha = flame * (0.55 + 0.45 * inner)               # translucent edges, denser near you
        glow = cv2.GaussianBlur(flame, (0, 0), 3) * keep_out
        return alpha, flame, inner, glow

    @staticmethod
    def _colorize(flame, inner, glow, pal):
        # level color toward the outer edge, fading to light/white in the inner part
        return ((flame * (1 - 0.55 * inner))[..., None] * pal["edge"] * 0.7
                + inner[..., None] * pal["core"] * 0.75 + glow[..., None] * pal["edge"] * 0.3)

    def _outline(self, mask, strength, now):
        """Thin flickering yellow rim around the body, at 1/2 res."""
        m = cv2.resize(mask, (self.lw, self.lh), interpolation=cv2.INTER_AREA)
        mb = (m > 0.5).astype(np.float32)
        b = cv2.GaussianBlur(mb, (0, 0), 1.6)
        rim = np.clip(b * (1 - b) * 4, 0, 1)
        rim = np.clip(rim - mb * 0.6, 0, 1)                 # keep it on the outside edge
        flick = 0.85 + 0.15 * math.sin(now * 29) * math.sin(now * 13)
        return rim * strength * flick

    def _crackles(self, mask_small, intensity):
        """Thin jagged electricity around the body, at 1/2 res."""
        canvas = np.zeros((self.lh, self.lw), np.float32)
        if intensity <= 0.01:
            return canvas
        mb = (mask_small > 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return canvas
        pts = np.vstack([c[:, 0, :] for c in cs]).astype(np.float32) * (self.q / self.lq)
        cx, cy = pts.mean(axis=0)
        n_arcs = int(round(intensity * 10 + self.rng.random() * 2 * intensity))
        size = self.lh
        for _ in range(n_arcs):
            p = pts[self.rng.integers(len(pts))]
            out = p - np.array([cx, cy])
            out /= np.linalg.norm(out) + 1e-6
            tang = np.array([-out[1], out[0]])
            start = p + out * self.rng.uniform(0, 0.04) * size
            direction = tang * self.rng.choice([-1, 1]) + out * self.rng.uniform(-0.6, 0.9)
            direction /= np.linalg.norm(direction) + 1e-6
            length = self.rng.uniform(0.05, 0.14) * size
            seg = 7
            path = [start]
            for i in range(1, seg + 1):
                base = start + direction * length * i / seg
                jitter = np.array([-direction[1], direction[0]]) * self.rng.uniform(-1, 1) * length * 0.18
                path.append(base + jitter)
            bright = self.rng.uniform(0.6, 1.0)
            cv2.polylines(canvas, [np.array(path, np.int32)], False, bright, 1, cv2.LINE_AA)
            if self.rng.random() < 0.5:                       # a branch
                j = self.rng.integers(2, seg)
                bdir = direction + np.array([-direction[1], direction[0]]) * self.rng.uniform(-1.2, 1.2)
                b0 = path[j]
                b1 = b0 + bdir * length * 0.25 + self.rng.uniform(-3, 3, 2)
                b2 = b1 + bdir * length * 0.2 + self.rng.uniform(-3, 3, 2)
                cv2.polylines(canvas, [np.array([b0, b1, b2], np.int32)], False, bright * 0.7, 1, cv2.LINE_AA)
        return canvas

    def render(self, frame, mask, now):
        if self.state == "idle":
            return frame
        t = now - self.t0
        mask_small = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_AREA)

        aura_amt, reach, crackle, outline, flash, shake = 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
        lv = self.level if self.state in ("active", "dissipating") else self.target
        pal = LEVELS.get(max(1, lv), LEVELS[1])
        spark_pal = pal
        if self.state == "powering":
            k = min(1.0, t / POWERUP_SECONDS)
            crackle = 0.3 + 0.7 * k
            if self.level == 0:                              # building from nothing
                aura_amt = smoothstep(0.35, 1.0, k) * 0.7
                reach = 0.35 + 0.5 * k
                outline = OUTLINE_STRENGTH * k
            else:                                            # level-up: old aura shifts to the new color
                old = LEVELS[self.level]
                pal = _mix(old, LEVELS[self.target], smoothstep(0.3, 1.0, k))
                aura_amt = 1.0
                reach = 1.0 + 0.25 * k + 0.05 * math.sin(now * 9)
                outline = OUTLINE_STRENGTH
            spark_pal = LEVELS[self.target]                  # crackles are already the new color
            shake = 2.5 * k * (1 + 0.3 * (self.target - 1))
            if t >= POWERUP_SECONDS:                         # ignition flash
                f = (t - POWERUP_SECONDS) / FLASH_SECONDS
                flash = 0.5 * (1 - f)
                aura_amt, reach, crackle, shake = 1.0, 1.15, 1.0, 6.0
        elif self.state == "active":
            aura_amt = 1.0
            reach = 1.0 + 0.04 * math.sin(now * 5.3)
            outline = OUTLINE_STRENGTH
            crackle = CRACKLES_WHILE_ACTIVE if self.rng.random() < 0.35 else 0.0
        elif self.state == "dissipating":
            k = min(1.0, t / DISSIPATE_SECONDS)
            fade = 1 - smoothstep(0, 1, k)
            aura_amt = fade * max(self.dissipate_from, 0.3)
            reach = 1.0 + 0.6 * k                            # flares out as it fades
            outline = OUTLINE_STRENGTH * fade

        size = (self.W, self.H)
        out = frame
        if aura_amt > 0.01:
            fields = self._aura_fields(mask_small, now, reach)
            if fields is not None:
                alpha, flame, inner, glow_f = fields
                color = self._colorize(flame, inner, glow_f, pal)
                alpha = np.clip(alpha * AURA_OPACITY * pal["opacity"], 0, 0.92) * aura_amt
                color = color * aura_amt
                glow = cv2.resize(np.clip(color * 255, 0, 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
                inv = cv2.resize(((1 - alpha) * 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
                out = cv2.add(cv2.multiply(out, cv2.merge([inv, inv, inv]), scale=1 / 255.0), glow)

        light = np.zeros((self.lh, self.lw, 3), np.float32)
        any_light = False
        if outline > 0.01:
            rim = self._outline(mask, outline, now)
            light += rim[..., None] * pal["outline"]
            light += cv2.GaussianBlur(rim, (0, 0), 3)[..., None] * pal["outline"] * 0.6
            any_light = True
        if crackle > 0.01:
            arcs = self._crackles(mask_small, crackle)
            light += (arcs[..., None] * spark_pal["spark_core"]
                      + cv2.GaussianBlur(arcs, (0, 0), 2.5)[..., None] * spark_pal["spark_glow"]
                      * spark_pal["spark_boost"])
            any_light = True
        if any_light:
            add = cv2.resize(np.clip(light * 255, 0, 255).astype(np.uint8), size, interpolation=cv2.INTER_LINEAR)
            out = cv2.add(out, add)

        if flash > 0.01:
            white = np.empty_like(out)
            white[:] = LEVELS[max(1, self.target)]["flash"]
            out = cv2.addWeighted(out, 1 - flash, white, flash, 0)

        shake *= SHAKE
        if shake > 0.5:
            dx, dy = self.rng.uniform(-shake, shake, 2)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            out = cv2.warpAffine(out, M, size, borderMode=cv2.BORDER_REFLECT)
        return out
