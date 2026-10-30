"""
Gapless looping sound (used for the aura hum).

The sound is decoded once and streamed continuously with `sounddevice`, so it
repeats with no gap. The end is crossfaded into the start so the seam is smooth,
and any digital silence at the edges (common in MP3s) is trimmed.

Falls back to relaunching the system player if `sounddevice` isn't available.
"""

import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np

LOOP_CROSSFADE = 0.025      # seconds of end blended into the start (0 = none)
LOOP_FADE_OUT = 0.08        # fade when the loop is stopped (avoids a click)
LOOP_VOLUME = 1.0

_cache = {}


def _read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
        v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float32) / (1 << 23)
    elif sw == 4:
        x = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648
    elif sw == 1:
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    else:
        raise ValueError(f"unsupported WAV sample width: {sw}")
    return x.reshape(-1, ch), sr


def _decode(path):
    """Any format the Mac can play -> (float32 samples [n, ch], sample rate)."""
    if path.lower().endswith(".wav"):
        try:
            return _read_wav(path)
        except Exception:
            pass                                    # e.g. float WAV: convert below
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    try:
        if shutil.which("afconvert"):               # built into macOS
            cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16", path, tmp.name]
        elif shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-acodec", "pcm_s16le", tmp.name]
        else:
            raise RuntimeError("no decoder (afconvert/ffmpeg) for " + path)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return _read_wav(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def make_loop(x, sr, crossfade=LOOP_CROSSFADE):
    """Trim digital silence at the edges, then crossfade the end into the start."""
    peak = np.abs(x).max(axis=1)
    loud = np.nonzero(peak > 1e-4)[0]
    if len(loud):
        lim = int(0.1 * sr)                          # never trim more than 100 ms per side
        a = min(loud[0], lim)
        b = max(loud[-1] + 1, len(x) - lim)
        x = x[a:b]
    n = int(crossfade * sr)
    if n > 0 and len(x) > 4 * n:
        t = np.linspace(0, 1, n, endpoint=False, dtype=np.float32)[:, None]
        fade_in, fade_out = np.sqrt(t), np.sqrt(1 - t)          # equal-power crossfade
        head = x[:n] * fade_in + x[-n:] * fade_out
        x = np.concatenate([head, x[n:-n]])
    return np.ascontiguousarray(x, dtype=np.float32)


def load_loop(path):
    key = (path, os.path.getmtime(path))
    if key not in _cache:
        x, sr = _decode(path)
        _cache[key] = (make_loop(x, sr), sr)
    return _cache[key]


class LoopSound:
    """start(name) plays `name` on a gapless loop until stop()."""

    def __init__(self, resolve_path, fallback_play=None, fallback_stop=None):
        self.resolve = resolve_path
        self.fallback_play, self.fallback_stop = fallback_play, fallback_stop
        self.stream = None
        self.lock = threading.Lock()
        self.fb_thread = None
        self.fb_running = False
        try:
            import sounddevice
            self.sd = sounddevice
        except Exception as e:
            print("Gapless loop unavailable (sounddevice):", e, "- looping sounds may have small gaps.")
            self.sd = None

    @property
    def playing(self):
        return self.stream is not None or self.fb_running

    def start(self, name):
        self.stop(immediate=True)
        path = self.resolve(name)
        if path is None:
            return
        if self.sd is not None:
            try:
                self._start_stream(path)
                return
            except Exception as e:
                print(f"Couldn't stream {os.path.basename(path)} ({e}); using the system player.")
        self._start_fallback(name)

    def _start_stream(self, path):
        buf, sr = load_loop(path)
        st = dict(pos=0, gain=0.0, target=1.0)
        n_total, ch = buf.shape
        ramp = 1.0 / max(1, int(LOOP_FADE_OUT * sr))
        sd = self.sd

        def callback(outdata, frames, time_info, status):
            out = np.empty((frames, ch), np.float32)
            pos, filled = st["pos"], 0
            while filled < frames:                        # wrap around the loop seamlessly
                k = min(frames - filled, n_total - pos)
                out[filled:filled + k] = buf[pos:pos + k]
                filled += k
                pos = (pos + k) % n_total
            st["pos"] = pos
            g0, tgt = st["gain"], st["target"]
            g = g0 + np.clip(tgt - g0, -ramp * frames, ramp * frames) * np.linspace(0, 1, frames, dtype=np.float32)
            st["gain"] = float(g[-1])
            outdata[:] = out * (g[:, None] * LOOP_VOLUME)
            if tgt == 0.0 and st["gain"] <= 1e-4:
                raise sd.CallbackStop

        stream = sd.OutputStream(samplerate=sr, channels=ch, dtype="float32", callback=callback)
        stream._loop_state = st
        stream.start()
        with self.lock:
            self.stream = stream

    def _start_fallback(self, name):
        if self.fallback_play is None:
            return
        self.fb_running = True

        def run():
            while self.fb_running:
                p = self.fallback_play(name)
                if p is None:
                    time.sleep(1.0)
                    continue
                while self.fb_running and p.poll() is None:
                    time.sleep(0.005)
                if not self.fb_running and self.fallback_stop:
                    self.fallback_stop(p)

        self.fb_thread = threading.Thread(target=run, daemon=True)
        self.fb_thread.start()

    def stop(self, immediate=False):
        with self.lock:
            stream, self.stream = self.stream, None
        if stream is not None:
            if immediate:
                stream.abort()
                stream.close()
            else:
                stream._loop_state["target"] = 0.0          # fade out, then the callback stops itself
                threading.Timer(LOOP_FADE_OUT + 0.3, stream.close).start()
        self.fb_running = False
