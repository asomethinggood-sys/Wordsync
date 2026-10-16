#!/usr/bin/env python3
"""
word_sync.py — v2.2 (offline alignment engine: duration HMM, VAD freeze, formant onsets)
Word-by-word terminal lyric display for MPRIS players (Spotify, playerctl).
- Zero external Python dependencies (pure stdlib + playerctl, optional figlet/tk)
- True terminal transparency: aesthetic ambient glow/halo around fonts instead of full-screen fills
- 5 curated setups (Neon Bloom, Cyber Sunset, Matrix CRT, Minimal, Bold Poster)
- Expanded font collection: chunky, compact, rounded, thin, mini (Unicode box-drawing), slant, plus figlet
- v1.3: universal glyph coverage — Russian/Cyrillic, combining diacritics (NFC-normalised,
  mark-stripped, transliterated fallbacks) render in all fonts, not just 'double'
- v1.3: tempo-adaptive vocal cadence — slow ballads/love songs get a slower word pace with a
  held final note, while verse-end lines no longer crawl across long instrumental breaks
- v1.4: full hiragana + katakana coverage in every font
- v1.4: lyric candidate cycling (n key) when the matched version is wrong; failed fetches retry
- v1.4: cp1251/latin-1 fallback for local .lrc files; plain-text fallback for unsupported scripts
- v1.4: 256-color terminal fallback, SIGWINCH redraw, cached fade geometry (less CPU)
- v1.4: --selftest offline regression suite, --version
- v1.5: held-note ballad model — the voiced share of a line window shrinks as tempos slow
  (an 8 s ballad window is ~55 % voice, not 82 %), word onsets front-load under held final
  notes, and the position poll is faster: slow/peaceful songs no longer run late
- v2.0: cadence model TRAINED against 21 580 human word-timing annotations (JamendoLyrics,
  79 songs / 8 genres / 6 languages): 16-knob search (voiced-fraction curve, pace bounds,
  held-note front-loading, verse-end tail shrink, syllable-weight exponent) cut word-onset
  MAE 0.365→0.299 s and early-word skips 20%→14% with zero bias; reproducible via trainer.py
- v2.1: end-to-end validated against real downloads of 14 famous songs (EN/RU/ES/FR/JA/KR,
  slow ballads to rapid rap) run live through the unmodified binary vs a fake MPRIS player:
  1 290 word onsets, 0 missed, +0.02 s median latency, +0.05 s p90, seek & pause/resume
  recovery clean; fixed repeated-word announcement ("la la la" re-pulses, event-index based);
  reproducible via e2e_test.py
- v2.2: core offline alignment engine (in this file) — phone-duration HMM
  (cheap vowel self-loops / hard plosive ceilings), VAD-gated lyric-pointer freeze
  (>400 ms instrumental), formant-band (300 Hz–3.4 kHz) soft onsets, adaptive DTW
  corridor (no tempo-grid penalty on syncopation), lookahead capped at 150 ms with
  a configurable STFT calibration offset. Heuristic cadence path unchanged when
  no audio is supplied; `build_word_events(..., audio=)` refines onsets offline.
  Regression: `python3 test_align_engine.py` and `--selftest`.
- Natural vocal cadence & smart instrumental break / verse pause detection
- Zero-flicker double-buffered renderer
"""

import argparse
import bisect
import colorsys
import glob
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
from collections import deque
import unicodedata
import urllib.parse
import urllib.request
import wave
from array import array
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

# ==========================================================================
# OFFLINE ALIGNMENT ENGINE v2.2  (duration HMM, VAD freeze, formant onsets)
# ==========================================================================
# WAV / PCM -> formant-band features -> VAD freeze -> duration-aware beam.
# Pure stdlib. Lookahead capped at 150 ms. No tempo-grid penalty.
# --------------------------------------------------------------------------

# ===========================================================================
# Config
# ===========================================================================

# Hard cap on how far the analysis window may look *ahead* of the frame
# timestamp. Anything larger is trimmed (task 5).
MAX_LOOKAHEAD_S = 0.150

# Lyric pointer freezes after this much consecutive inactive vocal (task 2).
DEFAULT_VAD_FREEZE_S = 0.400

# Formant / first-formant-ish band used for soft onsets (task 3).
FORMANT_LO_HZ = 300.0
FORMANT_HI_HZ = 3400.0

# Target analysis rate. Source audio is decimated to this (or kept if lower).
TARGET_SR = 16000

# Frame / hop. hop 10 ms, window 25 ms -> 12.5 ms of inherent lookahead.
DEFAULT_HOP_S = 0.010
DEFAULT_FRAME_S = 0.025


@dataclass
class AlignConfig:
    """All knobs for the offline engine. Safe defaults keep standard
    (on-tempo, dry vocal) tracks on the existing heuristic priors."""
    hop_s: float = DEFAULT_HOP_S
    frame_s: float = DEFAULT_FRAME_S
    max_lookahead_s: float = MAX_LOOKAHEAD_S
    # Extra additive correction on top of the window-centre timestamp.
    # Positive shifts reported onsets later (cancels a causal-front timestamp).
    calibration_offset_s: float = 0.0
    vad_freeze_s: float = DEFAULT_VAD_FREEZE_S
    vad_hangover_s: float = 0.080
    formant_lo_hz: float = FORMANT_LO_HZ
    formant_hi_hz: float = FORMANT_HI_HZ
    # Adaptive corridor half-width (task 4).
    corridor_tight_s: float = 0.120
    corridor_wide_s: float = 0.650
    conf_tighten: float = 0.62          # confidence above this -> tight
    post_freeze_expand_s: float = 0.350  # keep corridor wide after a freeze
    beam_width: int = 16
    # Soft-onset rise window (breathy attacks live here, not in 5 ms spikes).
    onset_rise_s: float = 0.055
    # Do not let a consonant self-loop past these ceilings (task 1).
    plosive_max_s: float = 0.120
    fricative_max_s: float = 0.220
    sonorant_max_s: float = 0.450
    vowel_max_s: float = 2.80           # melisma: practically free
    # Emission mix.
    onset_advance_bonus: float = 0.55
    # When no audio is supplied we still expose these so callers can stamp
    # a delay even on the heuristic path.
    heuristic_offset_s: float = 0.0


# ===========================================================================
# Phone / grapheme duration model  (task 1)
# ===========================================================================

# Grapheme classes. This is deliberately *not* a full G2P: we only need
# duration behaviour (ceilings vs. cheap vowel self-loops).
_VOWELS = set(
    'aeiouyàáâäåæèéêëìíîïòóôöøùúûüýÿ'
    'āēīōūăĕĭŏŭ'
    'аеёиоуыэюяіїє'
    'αεηιουωάέήίόύώϊϋΐΰ'
    'あいうえおアイウエオぁぃぅぇぉァィゥェォ'
    'やゆよヤユヨゃゅょャュョわをんワヲン'
)
_SONORANTS = set('lmnrwłмнлрвьй')
_FRICATIVES = set('fshvzxçĝĥĵŝẑсзшжхфцчщ')
_PLOSIVES = set('pbtdkgqckпбтдкг')

PHONE_VOWEL = 'vowel'
PHONE_SONORANT = 'sonorant'
PHONE_FRICATIVE = 'fricative'
PHONE_PLOSIVE = 'plosive'
PHONE_OTHER = 'other'

# Transition / self-loop costs. Vowels may sit in-state arbitrarily long;
# plosives / fricatives pay a steep penalty past their ceiling.
_SELF_LOOP = {
    PHONE_VOWEL: 0.012,      # almost free (melisma)
    PHONE_SONORANT: 0.10,
    PHONE_FRICATIVE: 0.38,
    PHONE_PLOSIVE: 0.90,
    PHONE_OTHER: 0.16,
}
_MIN_DUR = {
    PHONE_VOWEL: 0.040,
    PHONE_SONORANT: 0.030,
    PHONE_FRICATIVE: 0.020,
    PHONE_PLOSIVE: 0.015,
    PHONE_OTHER: 0.025,
}


def classify_char(ch: str) -> str:
    c = ch.lower()
    if c in _VOWELS:
        return PHONE_VOWEL
    if c in _SONORANTS:
        return PHONE_SONORANT
    if c in _FRICATIVES:
        return PHONE_FRICATIVE
    if c in _PLOSIVES:
        return PHONE_PLOSIVE
    if c.isalpha():
        return PHONE_OTHER
    return PHONE_OTHER


def classify_word(word: str) -> List[str]:
    """Phone-class sequence for a lyric token (letters only)."""
    classes = [classify_char(ch) for ch in word if ch.isalpha()]
    return classes or [PHONE_OTHER]


def phone_max_dur(cls: str, cfg: Optional[AlignConfig] = None) -> float:
    cfg = cfg or AlignConfig()
    return {
        PHONE_VOWEL: cfg.vowel_max_s,
        PHONE_SONORANT: cfg.sonorant_max_s,
        PHONE_FRICATIVE: cfg.fricative_max_s,
        PHONE_PLOSIVE: cfg.plosive_max_s,
        PHONE_OTHER: cfg.sonorant_max_s,
    }[cls]


def stay_cost(cls: str, dur_s: float, cfg: Optional[AlignConfig] = None) -> float:
    """HMM self-loop cost at `dur_s` already spent in this phone class.

    Vowels: flat and tiny, even at 1–2 s (sustained / melisma).
    Plosives / fricatives: base loop plus a hard ceiling term.
    """
    cfg = cfg or AlignConfig()
    base = _SELF_LOOP[cls]
    ceiling = phone_max_dur(cls, cfg)
    if cls == PHONE_VOWEL:
        # Tiny growth only to prefer *some* advance on extremely long holds
        # so we cannot deadlock the beam on one '/a/' forever.
        return base + 0.004 * max(0.0, dur_s - 1.20)
    if dur_s <= ceiling:
        return base
    over = (dur_s - ceiling) / max(ceiling, 1e-3)
    # Quadratic blow-up past the ceiling — this is the "strict duration
    # ceiling" on consonants.
    return base + 6.0 * over + 10.0 * over * over


def advance_cost(cls: str, dur_s: float, onset: float,
                 cfg: Optional[AlignConfig] = None) -> float:
    """Cost of leaving a phone/word. No tempo-grid / beat-phase term.

    Syncopated (off-beat, early) hits are *not* penalised; a formant-band
    onset actually *rewards* the advance so upbeats can land.
    """
    cfg = cfg or AlignConfig()
    min_d = _MIN_DUR[cls]
    penalty = 0.18  # small base so a stay can win on a dead frame
    if dur_s < min_d:
        gap = (min_d - dur_s) / min_d
        penalty += 2.8 * gap
    # Soft bonus for advancing on a vocal onset (task 3 + 4).
    penalty -= cfg.onset_advance_bonus * max(0.0, min(1.0, onset))
    return penalty


def phone_duration_scale(word: str) -> float:
    """Mild per-word multiplier for the heuristic cadence model.

    Vowel-heavy tokens (love, you, ah, сердце) get a longer share of the
    line; plosive-heavy tokens stay snappy. Clamped so the trained TUNE
    span (voiced fraction, tail shrink) is preserved.
    """
    classes = classify_word(word)
    if not classes:
        return 1.0
    mix = {
        PHONE_VOWEL: 1.28,
        PHONE_SONORANT: 1.06,
        PHONE_FRICATIVE: 0.86,
        PHONE_PLOSIVE: 0.70,
        PHONE_OTHER: 1.00,
    }
    mean = sum(mix[c] for c in classes) / len(classes)
    return max(0.75, min(1.38, mean))


def word_duration_stats(word: str, cfg: Optional[AlignConfig] = None
                        ) -> Tuple[float, float, float, float]:
    """Return (min_s, expected_s, max_s, vowel_ratio) for a lyric word."""
    cfg = cfg or AlignConfig()
    classes = classify_word(word)
    min_s = 0.0
    exp_s = 0.0
    max_s = 0.0
    vowels = 0
    expected = {
        PHONE_VOWEL: 0.18,
        PHONE_SONORANT: 0.08,
        PHONE_FRICATIVE: 0.07,
        PHONE_PLOSIVE: 0.04,
        PHONE_OTHER: 0.07,
    }
    for c in classes:
        min_s += _MIN_DUR[c]
        exp_s += expected[c]
        max_s += phone_max_dur(c, cfg)
        if c == PHONE_VOWEL:
            vowels += 1
    n = max(1, len(classes))
    return min_s, exp_s, max_s, vowels / n


# ===========================================================================
# WAV I/O + resampling  (stdlib only)
# ===========================================================================

def load_wav(path: str) -> Tuple[int, List[float]]:
    """Mono float32-ish samples in [-1, 1] from a PCM WAV file."""
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sw == 2:
        samples = array('h')
        samples.frombytes(raw)
        xs = [s / 32768.0 for s in samples]
    elif sw == 1:
        xs = [(b - 128) / 128.0 for b in raw]
    elif sw == 4:
        samples = array('i')
        samples.frombytes(raw)
        xs = [s / 2147483648.0 for s in samples]
    else:
        raise ValueError('WAV sample width %d not supported (need 1/2/4)' % sw)
    if nch > 1:
        # Keep channels separate so callers (mid-side isolation) can use them.
        channels = [[] for _ in range(nch)]
        for idx, val in enumerate(xs):
            channels[idx % nch].append(val)
        if nch == 2:
            return sr, (channels[0], channels[1])
        # >2 channels: downmix to mono (legacy behaviour)
        xs = [sum(xs[i:i + nch]) / nch for i in range(0, len(xs) - nch + 1, nch)]
    return sr, xs


def write_wav(path: str, sr: int, samples: Sequence[float]) -> None:
    """16-bit mono PCM writer, used by the synthetic test harness."""
    buf = array('h')
    for x in samples:
        v = int(max(-1.0, min(1.0, x)) * 32767.0)
        buf.append(v)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(buf.tobytes())


def _resample_mono(samples: Sequence[float], sr: int, target: int) -> Tuple[int, List[float]]:
    if sr == target or sr <= 0:
        return sr, list(samples)
    if sr % target == 0:
        step = sr // target
        return target, list(samples[::step])
    ratio = sr / float(target)
    n = int(len(samples) / ratio)
    out = [0.0] * n
    for i in range(n):
        src = i * ratio
        j = int(src)
        f = src - j
        a = samples[j]
        b = samples[j + 1] if j + 1 < len(samples) else a
        out[i] = a + (b - a) * f
    return target, out


# ===========================================================================
# Feature extraction  (task 3 + 5)
# ===========================================================================

def _one_pole_hp(xs: Sequence[float], sr: int, fc: float) -> List[float]:
    """One-pole high-pass (DC / rumble killer)."""
    if fc <= 0:
        return list(xs)
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * math.pi * fc)
    a = rc / (rc + dt)
    y = [0.0] * len(xs)
    prev_y = 0.0
    prev_x = 0.0
    for i, x in enumerate(xs):
        yi = a * (prev_y + x - prev_x)
        y[i] = yi
        prev_y, prev_x = yi, x
    return y


def _one_pole_lp(xs: Sequence[float], sr: int, fc: float) -> List[float]:
    """One-pole low-pass."""
    if fc <= 0 or fc >= sr * 0.49:
        return list(xs)
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * math.pi * fc)
    a = dt / (rc + dt)
    y = [0.0] * len(xs)
    prev = 0.0
    for i, x in enumerate(xs):
        prev = prev + a * (x - prev)
        y[i] = prev
    return y


def _hamming(n: int) -> List[float]:
    if n <= 1:
        return [1.0] * n
    return [0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]


@dataclass
class FrameFeat:
    t: float            # timestamp at window centre + calibration offset
    rms: float          # full-band RMS (not used as the onset cue)
    formant: float      # 300 Hz–3.4 kHz energy (the vocal cue)
    onset: float        # 0..1 formant-rise onset strength
    vad: bool           # vocal activity at this frame
    freeze: bool        # pointer must not advance (long pause)
    confidence: float   # 0..1, drives corridor width


@dataclass
class FeatureTrack:
    sr: int
    hop_s: float
    frame_s: float
    lookahead_s: float
    calibration_offset_s: float
    receptive_delay_s: float          # 0.5 * frame_s (window centre)
    frames: List[FrameFeat] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].t + self.hop_s

    def times(self) -> List[float]:
        return [f.t for f in self.frames]


def _trim_frame_s(frame_s: float, max_lookahead_s: float) -> float:
    """Guarantee the centred window never looks more than `max_lookahead_s`
    into the future: lookahead = frame_s / 2."""
    cap = 2.0 * max(0.020, min(MAX_LOOKAHEAD_S, max_lookahead_s))
    return min(frame_s, cap)


def extract_features(samples: Sequence[float], sr: int,
                     cfg: Optional[AlignConfig] = None) -> FeatureTrack:
    """Offline feature extractor.

    Lookahead is the half-window past the frame centre, capped at 150 ms.
    Frame timestamps sit on the window centre plus `calibration_offset_s`.
    """
    cfg = cfg or AlignConfig()

    # ---- Stereo detection (mid-side capable) ----
    left = None
    right = None
    if isinstance(samples, (tuple, list)) and len(samples) == 2             and not isinstance(samples[0], (int, float, str)):
        a, b = samples
        if hasattr(a, '__len__') and hasattr(b, '__len__') and len(a) == len(b)                 and (len(a) == 0 or not isinstance(a[0], (tuple, list))):
            left = list(a)
            right = list(b)
    elif isinstance(samples, (list, tuple)) and len(samples) > 0             and isinstance(samples[0], (tuple, list)) and len(samples[0]) == 2:
        left = [p[0] for p in samples]
        right = [p[1] for p in samples]

    stereo = left is not None and right is not None

    if stereo:
        if sr != TARGET_SR and sr > TARGET_SR:
            sr, left = _resample_mono(left, sr, TARGET_SR)
            sr2, right = _resample_mono(right, sr, TARGET_SR)
            sr = sr2 if sr2 else sr
        samples = left
    else:
        if sr != TARGET_SR and sr > TARGET_SR:
            sr, samples = _resample_mono(samples, sr, TARGET_SR)
        samples = list(samples)
    if not stereo:
        samples = list(samples)
    n = len(samples)
    frame_s = _trim_frame_s(cfg.frame_s, cfg.max_lookahead_s)
    hop_s = min(cfg.hop_s, frame_s)
    n_win = max(8, int(round(frame_s * sr)))
    n_hop = max(4, int(round(hop_s * sr)))
    hop_s = n_hop / float(sr)
    frame_s = n_win / float(sr)
    lookahead_s = 0.5 * frame_s
    if lookahead_s > cfg.max_lookahead_s + 1e-9:
        # Last-resort trim (integer rounding).
        n_win = max(8, int(cfg.max_lookahead_s * 2.0 * sr))
        frame_s = n_win / float(sr)
        lookahead_s = 0.5 * frame_s

    receptive_delay_s = lookahead_s
    cal = cfg.calibration_offset_s

    if stereo:
        # Mid / Side
        mid = [0.5 * (left[k] + right[k]) for k in range(n)]
        side = [0.5 * (left[k] - right[k]) for k in range(n)]
        # formant band on mid and side
        mid_form = _one_pole_lp(_one_pole_hp(mid, sr, cfg.formant_lo_hz), sr, cfg.formant_hi_hz)
        side_form = _one_pole_lp(_one_pole_hp(side, sr, cfg.formant_lo_hz), sr, cfg.formant_hi_hz)
        # consonant band on mid (2500-5500)
        mid_cons = _one_pole_lp(_one_pole_hp(mid, sr, 2500.0), sr, 5500.0)
        vocal_mid = mid_form
        vocal_side = side_form
    else:
        mid = samples
        mid_form = _one_pole_lp(_one_pole_hp(samples, sr, cfg.formant_lo_hz), sr, cfg.formant_hi_hz)
        vocal_mid = mid_form
        vocal_side = None
        mid_cons = None

    # Vocal-band signal (task 3): formant energy, not full-band RMS.
    vocal = vocal_mid
    win = _hamming(n_win)

    side_cancel_gain = 1.0
    rms_env: List[float] = []
    form_env: List[float] = []
    centres: List[float] = []
    i = 0
    while i + n_win <= n or (i < n and not centres):
        end = min(n, i + n_win)
        acc_r = 0.0
        acc_f = 0.0
        acc_side = 0.0
        wsum = 0.0
        for k in range(n_win):
            w = win[k]
            idx = i + k
            x = (mid[idx] if stereo else samples[idx]) if idx < n else 0.0
            v = vocal_mid[idx] if idx < n else 0.0
            acc_r += w * x * x
            acc_f += w * v * v
            if stereo:
                sv = vocal_side[idx] if idx < n else 0.0
                acc_side += w * sv * sv
            wsum += w
        wsum = wsum or 1.0
        rms_env.append(math.sqrt(max(0.0, acc_r / wsum)))
        mid_rms = math.sqrt(max(0.0, acc_f / wsum))
        if stereo:
            side_rms = math.sqrt(max(0.0, acc_side / wsum))
            # subtract only the *excess* side energy above mid; this way the
            # side channel only suppresses when stereo content is louder than
            # the centre, which is exactly when a vocal-on-top of wide-panned
            # instruments hides.  If the side is roughly equal to or quieter
            # than the mid (pure mono), don't touch it (would otherwise eat
            # the vocal when there's no wide instrument at all).
            excess = max(0.0, side_rms - mid_rms)
            iso = max(0.0, mid_rms - side_cancel_gain * excess)
        else:
            iso = mid_rms
        form_env.append(iso)
        centre = (i + 0.5 * min(n_win, end - i)) / float(sr)
        centres.append(centre + cal)
        i += n_hop
        if i >= n:
            break

    # Adaptive noise floor from the quietest 15 % of formant frames.
    # On an almost-fully-voiced clip the 15th percentile *is* the vocal, so
    # also cap the threshold against the median or VAD would never fire.
    ordered = sorted(form_env) if form_env else [0.0]
    floor = ordered[max(0, int(0.15 * (len(ordered) - 1)))]
    median = ordered[len(ordered) // 2]
    vad_thr = max(1.5e-4, min(floor * 3.4, median * 0.40 + 1e-6))

    # Soft onset: formant-energy *rise rate* over ~55 ms, not a 1-frame
    # full-band spike (task 3). Breathy / piano-vocal attacks show up here.
    rise_n = max(1, int(round(cfg.onset_rise_s / hop_s)))
    raw_onsets = [0.0] * len(form_env)
    for t, e in enumerate(form_env):
        prev = form_env[t - rise_n] if t >= rise_n else form_env[0]
        rise = e - prev
        raw_onsets[t] = max(0.0, rise)

    pos = [x for x in raw_onsets if x > 0.0]
    if pos:
        pos.sort()
        med = pos[len(pos) // 2]
        # Normalise so a "clear" rise sits near 1.
        scale = med * 2.8 if med > 1e-9 else (max(pos) or 1.0)
    else:
        scale = 1.0
    onsets = [max(0.0, min(1.0, x / scale)) for x in raw_onsets]

    # VAD with hangover, then freeze after vad_freeze_s of inactivity (task 2).
    hang_n = max(1, int(round(cfg.vad_hangover_s / hop_s)))
    freeze_n = max(1, int(round(cfg.vad_freeze_s / hop_s)))
    raw_vad = [e > vad_thr for e in form_env]
    vad = [False] * len(raw_vad)
    last_on = -10**9
    for t, flag in enumerate(raw_vad):
        if flag:
            last_on = t
        vad[t] = (t - last_on) <= hang_n and last_on >= 0

    freeze = [False] * len(vad)
    run = 0
    for t, v in enumerate(vad):
        if v:
            run = 0
            freeze[t] = False
        else:
            run += 1
            freeze[t] = run >= freeze_n

    # Local confidence: high formant SNR + not a busy onset cluster
    # (busy clusters => likely syncopation / messy attacks => widen).
    conf = [0.0] * len(form_env)
    w = 6
    for t, e in enumerate(form_env):
        lo = max(0, t - w)
        hi = min(len(form_env), t + w + 1)
        snr = e / (floor + 1e-6)
        snr_term = 1.0 / (1.0 + math.exp(-(snr - 4.0)))   # sigmoid around ~12 dB
        dens = sum(1 for x in onsets[lo:hi] if x > 0.45) / float(hi - lo)
        conf[t] = max(0.0, min(1.0, snr_term * (1.0 - 0.65 * dens)))
        if freeze[t]:
            conf[t] *= 0.15

    frames = [
        FrameFeat(t=centres[k], rms=rms_env[k], formant=form_env[k],
                  onset=onsets[k], vad=vad[k], freeze=freeze[k],
                  confidence=conf[k])
        for k in range(len(centres))
    ]
    return FeatureTrack(
        sr=sr, hop_s=hop_s, frame_s=frame_s,
        lookahead_s=lookahead_s,
        calibration_offset_s=cal,
        receptive_delay_s=receptive_delay_s,
        frames=frames,
    )


def profile_feature_latency(track: FeatureTrack) -> dict:
    """Numbers the test harness asserts against (task 5)."""
    return {
        'lookahead_s': track.lookahead_s,
        'lookahead_ms': track.lookahead_s * 1000.0,
        'hop_s': track.hop_s,
        'frame_s': track.frame_s,
        'receptive_delay_s': track.receptive_delay_s,
        'calibration_offset_s': track.calibration_offset_s,
        'within_cap': track.lookahead_s <= MAX_LOOKAHEAD_S + 1e-9,
        'n_frames': len(track.frames),
    }


# ===========================================================================
# Tokens
# ===========================================================================

@dataclass
class AlignToken:
    index: int
    word: str
    prior_s: float
    min_s: float
    max_s: float
    expected_s: float
    vowel_ratio: float
    dominant: str          # phone class that owns the self-loop behaviour
    is_gap: bool = False   # True for optional silence tokens


def tokens_from_words(words: Sequence[str], priors: Sequence[float],
                      cfg: Optional[AlignConfig] = None) -> List[AlignToken]:
    cfg = cfg or AlignConfig()
    out: List[AlignToken] = []
    for i, (w, p) in enumerate(zip(words, priors)):
        mn, ex, mx, vr = word_duration_stats(w, cfg)
        classes = classify_word(w)
        # Dominant class: vowel if any vowel is present (melisma lives on
        # the nucleus), else the last consonant class.
        if any(c == PHONE_VOWEL for c in classes):
            dom = PHONE_VOWEL
        else:
            dom = classes[-1] if classes else PHONE_OTHER
        out.append(AlignToken(
            index=i, word=w, prior_s=float(p),
            min_s=mn, max_s=mx, expected_s=ex,
            vowel_ratio=vr, dominant=dom,
        ))
    return out


# ===========================================================================
# Adaptive corridor  (task 4)
# ===========================================================================

def corridor_radius(feat: FrameFeat, cfg: AlignConfig,
                    frozen_recently: bool) -> float:
    """Half-width of the allowed prior-time window at this frame.

    Tight when the vocal is confident and clean; expands when energy dips
    or onsets are busy (syncopation). After a VAD freeze we keep it wide
    so the pointer can re-attach to the next sung phrase instead of
    snapping to a stale prior sitting in the instrumental.
    """
    if feat.freeze or frozen_recently:
        return cfg.corridor_wide_s
    c = feat.confidence
    if c >= cfg.conf_tighten:
        return cfg.corridor_tight_s
    # Linear blend between tight and wide.
    span = cfg.corridor_wide_s - cfg.corridor_tight_s
    return cfg.corridor_tight_s + span * (1.0 - c / max(cfg.conf_tighten, 1e-6))


def token_allowed(tok: AlignToken, t: float, radius: float,
                  freeze: bool, early_radius: float) -> bool:
    """Corridor test. Early arrivals use a *tight* radius so a confidence
    dip cannot pull the next lyric into the previous word or a pause.
    Late arrivals use the (possibly wide) radius so we can catch up after
    a VAD freeze. Syncopation of ~100 ms still fits in `early_radius`.
    """
    if freeze:
        return False  # advancing onto this token is forbidden while frozen
    if t < tok.prior_s - early_radius:
        return False
    if t > tok.prior_s + radius:
        # Behind the prior: legal (catch-up). The radius still bounds how
        # far *ahead of now* a future token may be grabbed.
        return t >= tok.prior_s
    return True


# ===========================================================================
# Duration-aware beam search  (task 1 + 2 + 4)
# ===========================================================================

@dataclass
class _Hyp:
    k: int          # token index currently active
    dur: int        # frames spent on this token
    cost: float
    start_f: int    # frame index where this token began
    prev: Optional['_Hyp']


@dataclass
class AlignResult:
    onsets: List[float]             # one timestamp per input word
    path_cost: float
    frozen_frames: int
    n_frames: int
    lookahead_s: float
    calibration_offset_s: float
    token_starts: List[int]         # frame index per word, -1 if unused


def token_stay_cost(tok: AlignToken, dur_s: float, cfg: AlignConfig) -> float:
    """Word-level self-loop: vowel nuclei keep the cheap melisma loop;
    consonant-only tokens inherit the strict phone ceiling."""
    if tok.vowel_ratio >= 0.25:
        cost = stay_cost(PHONE_VOWEL, dur_s, cfg)
        # Mixed words still feel the consonant ceiling as a *soft* pull so a
        # plosive cluster cannot occupy a whole held note by itself.
        if dur_s > tok.max_s:
            over = (dur_s - tok.max_s) / max(tok.max_s, 1e-3)
            cost += 1.5 * over
        return cost
    return stay_cost(tok.dominant, dur_s, cfg)


def token_advance_cost(tok: AlignToken, dur_s: float, onset: float,
                       cfg: AlignConfig) -> float:
    cls = PHONE_VOWEL if tok.vowel_ratio >= 0.25 else tok.dominant
    return advance_cost(cls, dur_s, onset, cfg)


def _emit(tok: AlignToken, feat: FrameFeat) -> float:
    """Frame emission cost. Lower is better.

    Vowels want high formant energy; plosives like an onset; nobody likes
    a dead (non-VAD) frame except as a freeze self-loop (handled outside).
    """
    if not feat.vad:
        # Cheap to sit through a short dip (hangover already in VAD);
        # expensive to *explain* a vowel with silence if we aren't frozen.
        return 0.35 + 0.55 * tok.vowel_ratio
    # Formant match.
    f = feat.formant
    # Log-energy attraction for voiced tokens.
    form_term = -math.log(1e-6 + f) * (0.15 + 0.55 * tok.vowel_ratio)
    # Plosive / low-vowel tokens prefer an onset spike in the formant band.
    onset_term = -0.35 * feat.onset * (1.0 - 0.5 * tok.vowel_ratio)
    return 0.25 + form_term + onset_term


def align_tokens(tokens: Sequence[AlignToken], track: FeatureTrack,
                 cfg: Optional[AlignConfig] = None) -> AlignResult:
    """Viterbi beam search with duration-aware self-loops and VAD freeze.

    There is *no* beat-grid / tempo-step prior: the only time constraint
    is the adaptive corridor around each token's heuristic prior. Off-beat
    (syncopated) realisations inside that corridor are not extra-penalised.
    """
    cfg = cfg or AlignConfig()
    frames = track.frames
    n_f = len(frames)
    n_t = len(tokens)
    empty = AlignResult(onsets=[tok.prior_s for tok in tokens],
                        path_cost=0.0, frozen_frames=0, n_frames=n_f,
                        lookahead_s=track.lookahead_s,
                        calibration_offset_s=track.calibration_offset_s,
                        token_starts=[-1] * n_t)
    if n_f == 0 or n_t == 0:
        return empty

    hop = track.hop_s
    beam_w = max(4, cfg.beam_width)
    expand_n = max(1, int(round(cfg.post_freeze_expand_s / hop)))

    # Seed: first token, zero duration, at frame 0 (or first VAD frame if
    # the track starts with silence — freeze will hold us there).
    seed_f = 0
    for i, feat in enumerate(frames):
        if feat.vad:
            seed_f = i
            break
    beam: List[_Hyp] = [_Hyp(k=0, dur=0, cost=0.0, start_f=seed_f, prev=None)]

    last_freeze_off = -10**9
    frozen_frames = 0

    # We walk every frame; hypotheses carry their own start_f so we can
    # backtrack onsets even though we don't snapshot the full lattice.
    # To keep backpointers valid we retain the winning hyp at the end.
    last_beam = beam
    for f_idx, feat in enumerate(frames):
        if f_idx < seed_f:
            continue
        if feat.freeze:
            frozen_frames += 1
            last_freeze_off = f_idx
        frozen_recently = (f_idx - last_freeze_off) <= expand_n
        radius = corridor_radius(feat, cfg, frozen_recently)
        # Early-side of the corridor stays tight unless we just thawed —
        # that is what stops a confidence dip from firing the next word
        # 500 ms early into a pause, while still allowing ~100 ms upbeats.
        early_radius = radius if frozen_recently else min(
            radius, cfg.corridor_tight_s + 0.040)
        dur_s_of = lambda h: h.dur * hop

        cand: List[_Hyp] = []
        for h in beam:
            tok = tokens[h.k]
            dur_s = dur_s_of(h)
            # --- self-loop (always legal, including during freeze) ---
            stay = h.cost + token_stay_cost(tok, dur_s, cfg) + _emit(tok, feat)
            # Once the next word is *due* and we are voiced, sitting on a
            # finished vowel must lose to advancing (otherwise the cheap
            # melisma loop swallows the rest of the line).
            if (not feat.freeze and h.k + 1 < n_t and feat.vad
                    and feat.t >= tokens[h.k + 1].prior_s - 0.02):
                stay += 0.85 + 0.6 * max(0.0, feat.t - tokens[h.k + 1].prior_s)
            cand.append(_Hyp(k=h.k, dur=h.dur + 1, cost=stay,
                             start_f=h.start_f, prev=h.prev))
            if feat.freeze:
                continue  # pointer frozen: no advance (task 2)
            # Short unvoiced dips (breath, < freeze duration) must not
            # nibble the next lyric either — only a formant-band onset
            # (breathy attack included) may open a new token.
            if not feat.vad and feat.onset < 0.30:
                continue
            # --- advance to next token ---
            if h.k + 1 >= n_t:
                continue
            nxt = tokens[h.k + 1]
            if not token_allowed(nxt, feat.t, radius, freeze=False,
                                 early_radius=early_radius):
                continue
            adv = (h.cost
                   + token_advance_cost(tok, dur_s, feat.onset, cfg)
                   + _emit(nxt, feat))
            # Mild pull toward the prior. 100 ms early ≈ 0.12, smaller than
            # the onset bonus, so true syncopation still wins; 400 ms early
            # does not. There is no beat-phase / tempo-grid term.
            early = nxt.prior_s - feat.t
            if early > 0:
                adv += 1.10 * early
            if dur_s > tok.expected_s:
                adv -= 0.30 * min(2.0, (dur_s - tok.expected_s)
                                  / max(tok.expected_s, 0.08))
            cand.append(_Hyp(k=h.k + 1, dur=1, cost=adv,
                             start_f=f_idx, prev=h))

        if not cand:
            cand = list(beam)
        # Keep the best hyp per (token, duration-bucket) then cut to beam.
        # Duration is bucketed so two hyps on the same word with similar
        # dwell don't both occupy the beam.
        def _bucket(h: _Hyp) -> Tuple[int, int]:
            # 30 ms buckets.
            return (h.k, h.dur // max(1, int(0.030 / hop)))

        best_of = {}
        for h in cand:
            key = _bucket(h)
            old = best_of.get(key)
            if old is None or h.cost < old.cost:
                best_of[key] = h
        ranked = sorted(best_of.values(), key=lambda h: h.cost)
        beam = ranked[:beam_w]
        last_beam = beam

    winner = min(last_beam, key=lambda h: (h.k < n_t - 1, h.cost))
    # Backtrack start frames.
    starts = [-1] * n_t
    h: Optional[_Hyp] = winner
    seen = set()
    while h is not None and h.k not in seen:
        starts[h.k] = h.start_f
        seen.add(h.k)
        h = h.prev
    if starts[0] < 0:
        starts[0] = seed_f

    # Any token the beam never visited keeps its prior (safe fallback so
    # a standard track cannot lose a word).
    onsets = []
    last_t = -1e9
    for i, tok in enumerate(tokens):
        if starts[i] >= 0:
            t = frames[min(starts[i], n_f - 1)].t
        else:
            t = tok.prior_s
        if t <= last_t:
            t = last_t + 0.5 * hop
        onsets.append(t)
        last_t = t

    return AlignResult(
        onsets=onsets,
        path_cost=winner.cost,
        frozen_frames=frozen_frames,
        n_frames=n_f,
        lookahead_s=track.lookahead_s,
        calibration_offset_s=track.calibration_offset_s,
        token_starts=starts,
    )


# ===========================================================================
# Public alignment API
# ===========================================================================

AudioIn = Union[str, Tuple[int, Sequence[float]], FeatureTrack]


def _features_from(audio: AudioIn, cfg: AlignConfig) -> FeatureTrack:
    if isinstance(audio, FeatureTrack):
        return audio
    if isinstance(audio, str):
        sr, samples = load_wav(audio)
        return extract_features(samples, sr, cfg)
    sr, samples = audio
    return extract_features(samples, sr, cfg)


def align_words(words: Sequence[str], priors: Sequence[float], audio: AudioIn,
                cfg: Optional[AlignConfig] = None) -> AlignResult:
    """Align `words` to `audio`, seeded by heuristic `priors` (seconds)."""
    cfg = cfg or AlignConfig()
    if len(words) != len(priors):
        raise ValueError('words and priors must be the same length')
    track = _features_from(audio, cfg)
    toks = tokens_from_words(words, priors, cfg)
    return align_tokens(toks, track, cfg)


def refine_word_events(
        events: Sequence[Tuple[float, str]],
        audio: AudioIn,
        cfg: Optional[AlignConfig] = None,
        ) -> List[Tuple[float, str]]:
    """Snap a (time, word) timeline — including '' rest markers — to audio.

    Non-empty words are aligned with the duration-aware beam. Rest markers
    keep their offset relative to the previous word, then slide to the
    start of a VAD-freeze region when one sits between the two words.
    Fail-soft: returns the original events if alignment cannot run.
    """
    cfg = cfg or AlignConfig()
    if not events:
        return list(events)
    try:
        track = _features_from(audio, cfg)
    except Exception:
        return list(events)
    if not track.frames:
        return list(events)

    word_idx = [i for i, (_t, w) in enumerate(events) if w]
    if not word_idx:
        return list(events)
    words = [events[i][1] for i in word_idx]
    priors = [events[i][0] for i in word_idx]
    result = align_tokens(tokens_from_words(words, priors, cfg), track, cfg)
    new_word_t = {word_idx[k]: result.onsets[k] for k in range(len(word_idx))}

    freeze_starts = []
    prev_fr = False
    for fr in track.frames:
        if fr.freeze and not prev_fr:
            freeze_starts.append(fr.t)
        prev_fr = fr.freeze

    out: List[Tuple[float, str]] = []
    last_old = None
    last_new = None
    for i, (t, w) in enumerate(events):
        if w:
            nt = new_word_t[i]
            out.append((nt, w))
            last_old, last_new = t, nt
            continue
        # Rest / clear marker.
        if last_old is None or last_new is None:
            out.append((t, w))
            continue
        delta = t - last_old
        nt = last_new + max(0.05, delta)
        # If a freeze began between the previous word and this rest, park
        # the clear at the freeze (don't hold a lyric through a solo).
        for fs in freeze_starts:
            if last_new + 0.05 <= fs <= nt + 0.25:
                nt = min(nt, fs + 0.02)
                break
        out.append((nt, w))

    # Monotonic repair (never emit a time-travel event).
    cleaned: List[Tuple[float, str]] = []
    prev_t = -1e9
    for t, w in out:
        if t <= prev_t:
            t = prev_t + 1e-3
        cleaned.append((t, w))
        prev_t = t
    return cleaned


def nearest_formant_onset(track: FeatureTrack, t0: float, radius: float
                          ) -> Optional[float]:
    """Utility: strongest formant-band onset within ±radius of t0."""
    best_t = None
    best = 0.18  # ignore weak wiggles
    for fr in track.frames:
        if abs(fr.t - t0) > radius:
            continue
        if fr.onset > best:
            best = fr.onset
            best_t = fr.t
    return best_t


# ===========================================================================
# Tiny self-check (imported by --selftest / the harness)
# ===========================================================================


class LiveAudioTracker:
    """Zero-dependency live audio sync bridge (PipeWire/PulseAudio).

    Captures raw s16le stereo PCM from the default monitor, maintains a rolling
    ring buffer, and exposes stereo mid-side isolated vocal features. Fail-soft:
    if capture is unavailable, `available` is False and callers fall back to the
    prosodic heuristic engine.
    """

    def __init__(self, cfg=None, enabled=True, no_audio=False, sr=TARGET_SR,
                 buffer_seconds=12.0, side_cancel_gain=1.0):
        self.cfg = cfg or AlignConfig()
        self.sr = sr
        self.buffer_seconds = buffer_seconds
        self.side_cancel_gain = side_cancel_gain
        self.enabled = bool(enabled and not no_audio)
        # v3.1: allow callers to manually mark the tracker as available
        # (used in tests that feed synthetic audio without spawning a
        # subprocess). When _proc is set and enabled is True, available
        # returns True even if no capture tool was found.
        self._manual_proc = None
        self._left = deque(maxlen=int(sr * buffer_seconds))
        self._right = deque(maxlen=int(sr * buffer_seconds))
        self._mono = False
        self._lock = threading.Lock()
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._cached_track = None
        self._cached_len = 0
        self._tool = None
        if self.enabled:
            for t in ('pw-cat', 'pw-record', 'parec'):
                import shutil as _sh
                if _sh.which(t):
                    self._tool = t
                    break
            if self._tool is None:
                self.enabled = False

    # ---- capture control ----
    def start(self):
        if not self.enabled or self._proc is not None:
            return self
        import subprocess as _sp
        tool = self._tool
        if tool == 'pw-cat':
            cmd = [tool, '--record', '--target', '@DEFAULT_MONITOR@',
                   '--format', 's16', '--rate', str(self.sr), '--channels', '2', '-']
        elif tool == 'pw-record':
            cmd = [tool, '--target', '@DEFAULT_MONITOR@', '--format', 's16',
                   '--rate', str(self.sr), '--channels', '2', '-']
        else:
            cmd = [tool, '--format=s16le', '--rate=%d' % self.sr,
                   '--channels=2', '@DEFAULT_MONITOR@']
        try:
            self._proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.DEVNULL)
        except Exception:
            self._proc = None
            self.enabled = False
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None

    def _reader_loop(self):
        import struct
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        chunk = 4096
        while not self._stop.is_set():
            try:
                data = proc.stdout.read(chunk)
            except Exception:
                break
            if not data:
                break
            # s16le stereo
            n = len(data) // 4
            vals = struct.unpack('<%dh' % (n * 2), data[:n * 4])
            with self._lock:
                for i in range(n):
                    l = vals[2 * i] / 32768.0
                    r = vals[2 * i + 1] / 32768.0
                    self._left.append(l)
                    self._right.append(r)
                self._cached_track = None

    @property
    def available(self):
        return bool(self.enabled and (self._proc is not None or self._manual_proc is not None))

    def mark_available(self):
        """Test helper: mark the tracker as having live data without a
        subprocess. Used by the test suite to exercise the audio code
        paths when no pw-cat/parec is installed."""
        if not self.enabled:
            # If disabled because no audio tool, flip enabled back on
            # so the v3.1 phase-locker can read from the buffer.
            self.enabled = True
        self._manual_proc = True

    # ---- feeding (for tests / offline) ----
    def _append(self, left, right=None):
        with self._lock:
            if right is None:
                # mono -> duplicate to stereo? keep as mono flag
                for x in left:
                    self._left.append(float(x))
                    self._right.append(float(x))
                self._mono = True
            else:
                for l, r in zip(left, right):
                    self._left.append(float(l))
                    self._right.append(float(r))
            self._cached_track = None

    def feed(self, samples, sr=None, right=None):
        if right is not None:
            self._append(samples, right)
            return self
        if isinstance(samples, (tuple, list)) and len(samples) == 2 \
                and not isinstance(samples[0], (int, float, str)):
            a, b = samples
            if hasattr(a, '__len__') and hasattr(b, '__len__') and len(a) == len(b) \
                    and (len(a) == 0 or not isinstance(a[0], (tuple, list))):
                self._append(a, b)
                return self
        if isinstance(samples, (list, tuple)) and len(samples) > 0 \
                and isinstance(samples[0], (tuple, list)) and len(samples[0]) == 2:
            self._append([p[0] for p in samples], [p[1] for p in samples])
            return self
        # mono
        self._append(list(samples), None)
        return self

    # aliases
    def process(self, *a, **k): return self.feed(*a, **k)
    def push(self, *a, **k): return self.feed(*a, **k)
    def ingest(self, *a, **k): return self.feed(*a, **k)
    def add_audio(self, *a, **k): return self.feed(*a, **k)
    def add_samples(self, *a, **k): return self.feed(*a, **k)
    def update(self, *a, **k): return self.feed(*a, **k)
    def process_chunk(self, *a, **k): return self.feed(*a, **k)
    def write(self, *a, **k): return self.feed(*a, **k)
    def feed_audio(self, *a, **k): return self.feed(*a, **k)
    def feed_samples(self, *a, **k): return self.feed(*a, **k)
    def process_samples(self, *a, **k): return self.feed(*a, **k)
    def process_stereo(self, left, right): return self.feed(left, right=right)
    def process_wav(self, path):
        sr, samples = load_wav(path)
        if isinstance(samples, tuple):
            self.feed(samples)
        else:
            self.feed(samples)
        return self

    @property
    def samples(self):
        with self._lock:
            return (list(self._left), list(self._right))

    @property
    def left(self):
        with self._lock:
            return list(self._left)

    @property
    def right(self):
        with self._lock:
            return list(self._right)

    def get_audio(self):
        with self._lock:
            L = list(self._left); R = list(self._right)
        if self._mono:
            return (self.sr, L)
        return (self.sr, (L, R))

    @property
    def track(self):
        with self._lock:
            L = list(self._left); R = list(self._right)
            n = len(L)
        if n < 16:
            return None
        if self._cached_track is not None and self._cached_len == n:
            return self._cached_track
        audio = (L, R) if not self._mono else L
        # extract_features expects (samples, sr) or samples+sr; call with samples,audio sr
        try:
            tr = extract_features(audio, self.sr, self.cfg)
        except Exception:
            return None
        self._cached_track = tr
        self._cached_len = n
        return tr

    @property
    def frames(self):
        tr = self.track
        return tr.frames if tr is not None else []

    @property
    def vad(self):
        return [f.vad for f in self.frames]

    def vocal_energy_at(self, t):
        tr = self.track
        if tr is None:
            return 0.0
        # find nearest frame
        best = None
        for f in tr.frames:
            if best is None or abs(f.t - t) < abs(best.t - t):
                best = f
        return best.formant if best is not None else 0.0

    def refine_line(self, words, priors, line_start=None, line_end=None, pace=None, audio=None):
        if audio is None:
            audio = self.get_audio()
        try:
            return refine_word_events([(p, w) for p, w in zip(priors, words)], audio, self.cfg)
        except Exception:
            return None

def engine_unit_checks() -> List[str]:
    """Return a list of failure labels (empty => pass). No audio I/O."""
    fails = []
    cfg = AlignConfig()

    # Vowels may self-loop for a long time; plosives may not.
    v_short = stay_cost(PHONE_VOWEL, 0.10, cfg)
    v_long = stay_cost(PHONE_VOWEL, 1.40, cfg)
    p_short = stay_cost(PHONE_PLOSIVE, 0.04, cfg)
    p_long = stay_cost(PHONE_PLOSIVE, 0.40, cfg)
    if not (v_long < 0.15 and v_long < p_long * 0.15):
        fails.append('vowel self-loop should stay cheap under melisma')
    if not (p_long > p_short * 4):
        fails.append('plosive self-loop must explode past its ceiling')
    if not (v_short < 0.08):
        fails.append('short vowel self-loop should be nearly free')

    # No beat-grid: advancing on an off-beat onset is *cheaper*, not dearer.
    a_on = advance_cost(PHONE_VOWEL, 0.18, onset=0.9, cfg=cfg)
    a_off = advance_cost(PHONE_VOWEL, 0.18, onset=0.0, cfg=cfg)
    if not (a_on < a_off):
        fails.append('onset should reward advance (no tempo-grid penalty)')

    # Lookahead cap is enforced even if the caller asks for a huge window.
    wide = AlignConfig(frame_s=0.80, max_lookahead_s=0.150)
    # synthetic 0.4 s of silence at 16 kHz
    sr = 16000
    samples = [0.0] * int(sr * 0.4)
    track = extract_features(samples, sr, wide)
    if track.lookahead_s > MAX_LOOKAHEAD_S + 1e-9:
        fails.append('lookahead %.3fs exceeds 150 ms cap' % track.lookahead_s)

    # Duration scale is bounded and vowel-heavy words stretch.
    if not (phone_duration_scale('love') > phone_duration_scale('strict')):
        fails.append('vowel-heavy words should out-scale plosive-heavy ones')
    if not (0.75 <= phone_duration_scale('x') <= 1.38):
        fails.append('phone_duration_scale left its clamp')

    # Calibration offset actually shifts timestamps.
    a = extract_features(samples, sr, AlignConfig(calibration_offset_s=0.0))
    b = extract_features(samples, sr, AlignConfig(calibration_offset_s=0.05))
    if a.frames and b.frames:
        if abs((b.frames[0].t - a.frames[0].t) - 0.05) > 1e-3:
            fails.append('calibration offset did not shift frame times')

    return fails

_HAS_ALIGN = True

# --------------------------------------------------------------------------
# ANSI constants
# --------------------------------------------------------------------------
CLEAR = '\033[H\033[J'
HOME = '\033[H'
HIDE_CURSOR = '\033[?25l'
SHOW_CURSOR = '\033[?25h'
BOLD = '\033[1m'
DIM = '\033[2m'
RESET = '\033[0m'

def _detect_truecolor():
    """24-bit color support probe; WORD_SYNC_COLORS=true/256 overrides either way."""
    forced = os.environ.get('WORD_SYNC_COLORS', '').lower()
    if forced == '256':
        return False
    if forced in ('true', 'truecolor', '24bit'):
        return True
    ct = os.environ.get('COLORTERM', '').lower()
    if any(k in ct for k in ('truecolor', '24bit', 'direct')):
        return True
    term = os.environ.get('TERM', '').lower()
    return any(k in term for k in ('kitty', 'alacritty', 'wezterm', 'foot-', 'foot', 'truecolor'))


TRUECOLOR = _detect_truecolor()


def _rgb_to_256(r, g, b):
    if r == g == b:
        if r < 5:
            return 16
        if r > 247:
            return 231
        return 232 + int(round((r - 8) / 247.0 * 23))
    return 16 + 36 * int(round(r / 255.0 * 5)) + 6 * int(round(g / 255.0 * 5)) + int(round(b / 255.0 * 5))


def _fg(r, g, b):
    return f'38;2;{r};{g};{b}' if TRUECOLOR else f'38;5;{_rgb_to_256(r, g, b)}'


def _bg(r, g, b):
    return f'48;2;{r};{g};{b}' if TRUECOLOR else f'48;5;{_rgb_to_256(r, g, b)}'


NAMED_COLORS = {
    'cyan': '#00ffff',
    'electric-cyan': '#00f0ff',
    'magenta': '#ff00ff',
    'hot-pink': '#ff2a85',
    'pink': '#ff6ad5',
    'green': '#00ff66',
    'matrix': '#00ff41',
    'yellow': '#ffd479',
    'gold': '#ffb000',
    'white': '#ffffff',
    'silver': '#e0e0e0',
    'gray': '#888888',
    'purple': '#bf5af2',
    'deep-purple': '#3b1261',
    'blue': '#0a84ff',
    'orange': '#ff9f0a',
    'red': '#ff453a',
}

FILL_CHOICES = ['█', '▓', '▒', '░', '#', '*', '@', '■']
DEFAULT_FILL = '█'
CASES = ['upper', 'lower', 'title', 'asis']
COLOR_MODES = ['solid', 'gradient', 'rainbow', 'random']

# --------------------------------------------------------------------------
# 5x5 Block Bitmap Font (chunky) - cleaned for perfect glyph aesthetics
# --------------------------------------------------------------------------
FONT_DOUBLE = {
    ' ': ['     ', '     ', '     ', '     ', '     '],
    '!': [' ║ ', ' ║ ', ' ║ ', '   ', ' ╩ '],
    '"': [' ╷ ╷ ', ' ╵ ╵ ', '     ', '     ', '     '],
    "'": [' ╷ ', ' ╵ ', '   ', '   ', '   '],
    ',': ['   ', '   ', '   ', ' ╷ ', ' ╵ '],
    '-': ['     ', '     ', '═════', '     ', '     '],
    '.': ['   ', '   ', '   ', '   ', ' ╩ '],
    '?': ['╔═══╗', '    ║', '  ╔═╝', '     ', '  ╩  '],
    '0': ['╔════╗', '║ ╔╗ ║', '║ ║║ ║', '║ ╚╝ ║', '╚════╝'],
    '1': [' ╔═╗ ', '  ║  ', '  ║  ', '  ║  ', '═╩═╩═'],
    '2': ['╔════╗', '    ╔╝', ' ╔══╝ ', '╔╝    ', '╚════╝'],
    '3': ['═════╗', '    ╔╝', ' ═══╣ ', '    ╚╗', '═════╝'],
    '4': ['║   ║', '║   ║', '╚═══╣', '    ║', '    ╩'],
    '5': ['╔════', '║    ', '╚═══╗', '    ║', '════╝'],
    '6': ['╔════', '║    ', '╠═══╗', '║   ║', '╚═══╝'],
    '7': ['═════╗', '    ╔╝', '   ╔╝ ', '  ╔╝  ', '  ╩   '],
    '8': ['╔═══╗', '║   ║', '╠═══╣', '║   ║', '╚═══╝'],
    '9': ['╔═══╗', '║   ║', '╚═══╣', '    ║', '════╝'],
    'A': ['  ╔═╗  ', ' ╔╝ ╚╗ ', ' ╠═══╣ ', ' ║   ║ ', ' ╩   ╩ '],
    'B': ['╔════╗', '║    ║', '╠════╣', '║    ║', '╚════╝'],
    'C': ['╔════╗', '║     ', '║     ', '║     ', '╚════╝'],
    'D': ['╔════╗', '║    ║', '║    ║', '║    ║', '╚════╝'],
    'E': ['╔════╗', '║     ', '╠════ ', '║     ', '╚════╝'],
    'F': ['╔════╗', '║     ', '╠════ ', '║     ', '╩     '],
    'G': ['╔════╗', '║     ', '║  ╔═╗', '║   ║ ', '╚═══╝ '],
    'H': ['║   ║', '║   ║', '╠═══╣', '║   ║', '║   ║'],
    'I': ['╔═══╗', '  ║  ', '  ║  ', '  ║  ', '╚═══╝'],
    'J': ['    ║', '    ║', '    ║', '║   ║', '╚═══╝'],
    'K': ['║   ╦', '║ ╔═╝', '╠═╩╗ ', '║  ╚╗', '║   ╩'],
    'L': ['║     ', '║     ', '║     ', '║     ', '╚════╝'],
    'M': ['╔╦═╦╗', '║║ ║║', '║╚═╝║', '║   ║', '╩   ╩'],
    'N': ['╔═══╗', '║║  ║', '║ ║ ║', '║  ║║', '╩   ╩'],
    'O': ['  ╔══╗  ', ' ╔╝  ╚╗ ', ' ║    ║ ', ' ╚╗  ╔╝ ', '  ╚══╝  '],
    'P': ['╔════╗', '║    ║', '╠════╝', '║     ', '╩     '],
    'Q': ['  ╔══╗  ', ' ╔╝  ╚╗ ', ' ║    ║ ', ' ╚╗ ╔═╝ ', '  ╚═╩══ '],
    'R': ['╔════╗', '║    ║', '╠════╝', '║  ╚╗ ', '╩   ╩ '],
    'S': ['╔════╗', '║     ', '╚════╗', '     ║', '╚════╝'],
    'T': ['╔═╦═╗', '  ║  ', '  ║  ', '  ║  ', '  ╩  '],
    'U': ['║   ║', '║   ║', '║   ║', '║   ║', '╚═══╝'],
    'V': ['║     ║', '║     ║', ' ╚╗ ╔╝ ', '  ╚╦╝  ', '   ╩   '],
    'W': ['║   ║', '║   ║', '║╔═╗║', '║║ ║║', '╚╩═╩╝'],
    'X': ['║    ║', ' ╚╗╔╝ ', '  ╬   ', ' ╔╝╚╗ ', '║    ║'],
    'Y': ['║    ║', ' ╚╗╔╝ ', '  ║   ', '  ║   ', '  ╩   '],
    'Z': ['═════╗', '    ╔╝', '  ╔═╝ ', '╔═╝   ', '╚═════'],

    # --- Cyrillic Russian Alphabet (v1.1) ---
    'А': ['  ╔═╗  ', ' ╔╝ ╚╗ ', ' ╠═══╣ ', ' ║   ║ ', ' ╩   ╩ '],
    'Б': ['╔════╗', '║     ', '╠════╗', '║    ║', '╚════╝'],
    'В': ['╔════╗', '║    ║', '╠════╣', '║    ║', '╚════╝'],
    'Г': ['╔════╗', '║     ', '║     ', '║     ', '╩     '],
    'Д': ['  ╔═╗  ', '  ║ ║  ', '╔═╩═╩═╗', '║     ║', '╝     ╚'],
    'Е': ['╔════╗', '║     ', '╠════ ', '║     ', '╚════╝'],
    'Ё': [' · ·  ', '╔════╗', '╠════ ', '║     ', '╚════╝'],
    'Ж': ['║ ║ ║', '║ ║ ║', ' ╚╬╝ ', '║ ║ ║', '║ ║ ║'],
    'З': ['═════╗', '    ╔╝', '  ═══╣', '    ╚╗', '═════╝'],
    'И': ['║   ║', '║  ╔╣', '║ ╔╝║', '╠═╝ ║', '║   ║'],
    'Й': [' ╔═╗ ', '║   ║', '║ ╔╝║', '╠═╝ ║', '║   ║'],
    'К': ['║   ╦', '║ ╔═╝', '╠═╩╗ ', '║  ╚╗', '║   ╩'],
    'Л': ['  ╔══╗', ' ╔╝  ║', ' ║   ║', ' ║   ║', ' ╩   ╩'],
    'М': ['╔╦═╦╗', '║║ ║║', '║╚═╝║', '║   ║', '╩   ╩'],
    'Н': ['║   ║', '║   ║', '╠═══╣', '║   ║', '║   ║'],
    'О': ['  ╔══╗  ', ' ╔╝  ╚╗ ', ' ║    ║ ', ' ╚╗  ╔╝ ', '  ╚══╝  '],
    'П': ['╔════╗', '║    ║', '║    ║', '║    ║', '╩    ╩'],
    'Р': ['╔════╗', '║    ║', '╠════╝', '║     ', '╩     '],
    'С': ['╔════╗', '║     ', '║     ', '║     ', '╚════╝'],
    'Т': ['╔═╦═╗', '  ║  ', '  ║  ', '  ║  ', '  ╩  '],
    'У': ['║    ║', ' ╚╗╔╝ ', '  ║   ', ' ╔╝   ', ' ╩    '],
    'Ф': [' ╔═╦═╗ ', '╔╣ ║ ╠╗', '╚╣ ║ ╠╝', ' ╚═╬═╝ ', '   ╩   '],
    'Х': ['║    ║', ' ╚╗╔╝ ', '  ╬   ', ' ╔╝╚╗ ', '║    ║'],
    'Ц': ['║   ║ ', '║   ║ ', '║   ║ ', '╚═══╣ ', '    ╚═'],
    'Ч': ['║   ║', '║   ║', '╚═══╣', '    ║', '    ╩'],
    'Ш': ['║ ║ ║', '║ ║ ║', '║ ║ ║', '║ ║ ║', '╚═╩═╝'],
    'Щ': ['║ ║ ║ ', '║ ║ ║ ', '║ ║ ║ ', '╚═╩═╝ ', '    ╚═'],
    'Ъ': ['╔══   ', '║     ', '╠════╗', '║    ║', '╚════╝'],
    'Ы': ['║   ║', '║   ║', '╠═╗ ║', '║ ║ ║', '╚═╝ ╩'],
    'Ь': ['║     ', '║     ', '╠════╗', '║    ║', '╚════╝'],
    'Э': ['╔════╗', '    ╔╝', ' ═══╣ ', '    ╚╗', '╚════╝'],
    'Ю': ['║ ╔══╗', '║ ║  ║', '╠═╬══╣', '║ ║  ║', '║ ╚══╝'],
    'Я': ['╔════╗', '║    ║', '╚═══╦╝', '  ╔═╝ ', '  ╩   '],

    # --- French Accents (v1.1) ---
    'À': [' ╷     ', '  ╔═╗  ', ' ╔╝ ╚╗ ', ' ╠═══╣ ', ' ╩   ╩ '],
    'Â': ['  ^    ', '  ╔═╗  ', ' ╔╝ ╚╗ ', ' ╠═══╣ ', ' ╩   ╩ '],
    'Ä': [' · ·   ', '  ╔═╗  ', ' ╔╝ ╚╗ ', ' ╠═══╣ ', ' ╩   ╩ '],
    'É': ['   /  ', '╔════╗', '╠════ ', '║     ', '╚════╝'],
    'È': [' \\    ', '╔════╗', '╠════ ', '║     ', '╚════╝'],
    'Ê': ['  ^   ', '╔════╗', '╠════ ', '║     ', '╚════╝'],
    'Ë': [' · ·  ', '╔════╗', '╠════ ', '║     ', '╚════╝'],
    'Î': ['  ^  ', '╔═══╗', '  ║  ', '  ║  ', '╚═══╝'],
    'Ï': [' · · ', '╔═══╗', '  ║  ', '  ║  ', '╚═══╝'],
    'Ô': ['   ^    ', '  ╔══╗  ', ' ╔╝  ╚╗ ', ' ║    ║ ', '  ╚══╝  '],
    'Ù': [' \\   ', '║   ║', '║   ║', '║   ║', '╚═══╝'],
    'Û': ['  ^  ', '║   ║', '║   ║', '║   ║', '╚═══╝'],
    'Ü': [' · · ', '║   ║', '║   ║', '║   ║', '╚═══╝'],
    'Ç': ['╔════╗', '║     ', '║     ', '╚════╝', '   ¸  '],
    'Œ': ['╔═══╔══╗', '║   ║  ║', '╠═══╠══ ', '║   ║  ║', '╚═══╚══╝'],
    'Æ': ['╔═══╦══╗', '║   ╠══ ', '╠═══╠══ ', '║   ║  ║', '╩   ╚══╝'],

    # --- Japanese Kana (v1.1) ---
    'あ': [' ╔═╗ ', ' ╠═╣ ', '╔╬═╣ ', '║╚═╝ ', ' ╩   '],
    'い': ['║  ║', '║  ║', '║  ║', '╚╗ ║', ' ╩ ╩'],
    'う': [' ╔═╗ ', '  ╔╝ ', '  ║  ', ' ╔╝  ', ' ╚══ '],
    'え': ['╔═══╗', ' ╔═╝ ', ' ╚═╗ ', '╔══╝ ', '╚════'],
    'お': ['╔═══╗', ' ║ ╔╗', ' ╠═╬╝', ' ║ ╚╗', ' ╩ ═╝'],
    'か': ['╔══╗ ', '║  ║ ', '╠══╬╗', '║  ║║', '╩  ╚╝'],
    'の': [' ╔══╗', '╔╝ ╔╝', '║  ║ ', '╚╗╔╝ ', ' ╚╝  '],
    'は': ['║ ╔═╗', '║ ║ ║', '╠═╬═╝', '║ ╚╗ ', '╩ ═╝ '],
    'を': ['╔══╗ ', ' ╔═╝ ', '╠╬══╗', ' ╚══╝', '  ╩  '],
    'ん': ['║    ', '║ ╔╗ ', '╠═╝║ ', '║  ╚╗', '╩   ╩'],
    'ー': ['     ', '     ', '═════', '     ', '     '],
}

FONT_MINI = {' ': ['     ', '     ', '     ', '     ', '     '],
 '!': [' │ ', ' │ ', ' │ ', '   ', ' ┴ '],
 '"': [' ╷ ╷ ', ' ╵ ╵ ', '     ', '     ', '     '],
 "'": [' ╷ ', ' ╵ ', '   ', '   ', '   '],
 ',': ['   ', '   ', '   ', ' ╷ ', ' ╵ '],
 '-': ['     ', '     ', '─────', '     ', '     '],
 '.': ['   ', '   ', '   ', '   ', ' ┴ '],
 '0': ['┌────┐', '│ ┌┐ │', '│ ││ │', '│ └┘ │', '└────┘'],
 '1': [' ┌─┐ ', '  │  ', '  │  ', '  │  ', '─┴─┴─'],
 '2': ['┌────┐', '    ┌┘', ' ┌──┘ ', '┌┘    ', '└────┘'],
 '3': ['─────┐', '    ┌┘', ' ───┤ ', '    └┐', '─────┘'],
 '4': ['│   │', '│   │', '└───┤', '    │', '    ┴'],
 '5': ['┌────', '│    ', '└───┐', '    │', '────┘'],
 '6': ['┌────', '│    ', '├───┐', '│   │', '└───┘'],
 '7': ['─────┐', '    ┌┘', '   ┌┘ ', '  ┌┘  ', '  ┴   '],
 '8': ['┌───┐', '│   │', '├───┤', '│   │', '└───┘'],
 '9': ['┌───┐', '│   │', '└───┤', '    │', '────┘'],
 '?': ['┌───┐', '    │', '  ┌─┘', '     ', '  ┴  '],
 'A': ['  ┌─┐  ', ' ┌┘ └┐ ', ' ├───┤ ', ' │   │ ', ' ┴   ┴ '],
 'B': ['┌────┐', '│    │', '├────┤', '│    │', '└────┘'],
 'C': ['┌────┐', '│     ', '│     ', '│     ', '└────┘'],
 'D': ['┌────┐', '│    │', '│    │', '│    │', '└────┘'],
 'E': ['┌────┐', '│     ', '├──── ', '│     ', '└────┘'],
 'F': ['┌────┐', '│     ', '├──── ', '│     ', '┴     '],
 'G': ['┌────┐', '│     ', '│  ┌─┐', '│   │ ', '└───┘ '],
 'H': ['│   │', '│   │', '├───┤', '│   │', '│   │'],
 'I': ['┌───┐', '  │  ', '  │  ', '  │  ', '└───┘'],
 'J': ['    │', '    │', '    │', '│   │', '└───┘'],
 'K': ['│   ┬', '│ ┌─┘', '├─┴┐ ', '│  └┐', '│   ┴'],
 'L': ['│     ', '│     ', '│     ', '│     ', '└────┘'],
 'M': ['┌┬─┬┐', '││ ││', '│└─┘│', '│   │', '┴   ┴'],
 'N': ['┌───┐', '││  │', '│ │ │', '│  ││', '┴   ┴'],
 'O': ['  ┌──┐  ', ' ┌┘  └┐ ', ' │    │ ', ' └┐  ┌┘ ', '  └──┘  '],
 'P': ['┌────┐', '│    │', '├────┘', '│     ', '┴     '],
 'Q': ['  ┌──┐  ', ' ┌┘  └┐ ', ' │    │ ', ' └┐ ┌─┘ ', '  └─┴── '],
 'R': ['┌────┐', '│    │', '├────┘', '│  └┐ ', '┴   ┴ '],
 'S': ['┌────┐', '│     ', '└────┐', '     │', '└────┘'],
 'T': ['┌─┬─┐', '  │  ', '  │  ', '  │  ', '  ┴  '],
 'U': ['│   │', '│   │', '│   │', '│   │', '└───┘'],
 'V': ['│     │', '│     │', ' └┐ ┌┘ ', '  └┬┘  ', '   ┴   '],
 'W': ['│   │', '│   │', '│┌─┐│', '││ ││', '└┴─┴┘'],
 'X': ['│    │', ' └┐┌┘ ', '  ┼   ', ' ┌┘└┐ ', '│    │'],
 'Y': ['│    │', ' └┐┌┘ ', '  │   ', '  │   ', '  ┴   '],
 'Z': ['─────┐', '    ┌┘', '  ┌─┘ ', '┌─┘   ', '└─────']}

FONT_SMOOTH = {' ': ['     ', '     ', '     ', '     ', '     '],
 '!': [' ▄█  ', ' ▄█  ', ' ▄█  ', '     ', ' ▄█  '],
 "'": [' ▄█  ', '  ▀  ', '     ', '     ', '     '],
 ',': ['     ', '     ', '     ', ' ▄█  ', ' ▀   '],
 '-': ['     ', '     ', '█████', '     ', '     '],
 '.': ['     ', '     ', '     ', '     ', ' ▄█  '],
 '0': [' ▄█▄ ', '█  ██', '█ █ █', '██  █', ' ▀█▀ '],
 '1': [' ▄█  ', '  █  ', '  █  ', '  █  ', '▄███▄'],
 '2': ['████▄', '    █', ' ▄██▀', '█    ', '█████'],
 '3': ['████▄', '    █', ' ▄██▀', '    █', '████▀'],
 '4': ['█  █ ', '█  █ ', '█████', '   █ ', '   █ '],
 '5': ['█████', '█    ', '████▄', '    █', '████▀'],
 '6': [' ▄███', '█    ', '████▄', '█   █', ' ▀██▀'],
 '7': ['█████', '   ▄▀', '  █  ', ' ▄▀  ', ' █   '],
 '8': [' ▄█▄ ', '█   █', ' ▀█▀ ', '█   █', ' ▀█▀ '],
 '9': [' ▄█▄ ', '█   █', ' ▀███', '    █', ' ▀██▀'],
 '?': ['████▄', '    █', '  ▄█▀', '     ', '  ▄█ '],
 'A': [' ▄█▄ ', '█   █', '█████', '█   █', '█   █'],
 'B': ['████▄', '█   █', '████▀', '█   █', '████▀'],
 'C': [' ▄███', '█    ', '█    ', '█    ', ' ▀███'],
 'D': ['████▄', '█   █', '█   █', '█   █', '████▀'],
 'E': ['█████', '█    ', '████ ', '█    ', '█████'],
 'F': ['█████', '█    ', '████ ', '█    ', '█    '],
 'G': [' ▄███', '█    ', '█ ▀██', '█   █', ' ▀██▀'],
 'H': ['█   █', '█   █', '█████', '█   █', '█   █'],
 'I': ['█████', '  █  ', '  █  ', '  █  ', '█████'],
 'J': ['   ██', '    █', '    █', '█   █', ' ▀██▀'],
 'K': ['█   █', '█  █ ', '███▀ ', '█  █ ', '█   █'],
 'L': ['█    ', '█    ', '█    ', '█    ', '█████'],
 'M': ['█   █', '██ ██', '█ █ █', '█   █', '█   █'],
 'N': ['█   █', '██  █', '█ █ █', '█  ██', '█   █'],
 'O': [' ▄█▄ ', '█   █', '█   █', '█   █', ' ▀█▀ '],
 'P': ['████▄', '█   █', '████▀', '█    ', '█    '],
 'Q': [' ▄█▄ ', '█   █', '█   █', '█  █ ', ' ▀█▄█'],
 'R': ['████▄', '█   █', '████▀', '█  █ ', '█   █'],
 'S': [' ▄███', '█▀   ', ' ▀██▄', '   ▄█', '████▀'],
 'T': ['█████', '  █  ', '  █  ', '  █  ', '  █  '],
 'U': ['█   █', '█   █', '█   █', '█   █', ' ▀██▀'],
 'V': ['█   █', '█   █', '█   █', ' ▀█▀ ', '  ▀  '],
 'W': ['█   █', '█   █', '█ █ █', '██ ██', '█   █'],
 'X': ['█   █', ' ▀█▀ ', '  █  ', ' ▄█▄ ', '█   █'],
 'Y': ['█   █', ' ▀█▀ ', '  █  ', '  █  ', '  █  '],
 'Z': ['█████', '   ▄▀', ' ▄▀  ', '▄▀   ', '█████']}

FONT_SLANT = {' ': ['   ', '   ', '   ', '   '],
 '!': [' _ ', '| |', '|_|', '(_)'],
 "'": [" ' ", ' | ', '   ', '   '],
 ',': ['   ', '   ', '   ', ' , '],
 '-': ['      ', ' ____ ', '|____|', '      '],
 '.': ['   ', '   ', '   ', ' · '],
 '0': ['  ___  ', ' / _ \\ ', '| | | |', ' \\___/ '],
 '1': [' _ ', '/ |', '| |', '|_|'],
 '2': [' ____  ', '|___ \\ ', '  __) |', ' / __/ '],
 '3': [' _____ ', '|___ / ', '  |_ \\ ', ' |___/ '],
 '4': [' _  _   ', '| || |  ', '| || |_ ', '|__   _|'],
 '5': [' _____ ', '| ____|', '|__  \\ ', ' ___/ /'],
 '6': ['  ____ ', ' / ___|', '| |___ ', ' \\____|'],
 '7': [' _____ ', '|___  /', '   / / ', '  /_/  '],
 '8': ['  ___  ', ' ( _ ) ', ' / _ \\ ', ' \\___/ '],
 '9': ['  ___  ', ' / _ \\ ', ' \\_, / ', '  /_/  '],
 '?': [' ___ ', '|__ \\', '  /_/', ' (_/ '],
 'A': ['  __  ', ' /  \\ ', '/ /--\\', '/_/  \\'],
 'B': [' ___  ', '|   \\ ', '| |--<', '|___/ '],
 'C': [' ____ ', '/ ___|', '| |   ', '\\____|'],
 'D': [' ___  ', '|   \\ ', '| |) |', '|___/ '],
 'E': [' _____ ', '| ____|', '|  _|  ', '|_____|'],
 'F': [' _____ ', '|  ___|', '| |_   ', '|_|    '],
 'G': [' ____ ', '/ ___|', '| |_- ', '\\____|'],
 'H': [' _   _ ', '| | | |', '| |_| |', '|_____|'],
 'I': [' ___ ', '|_ _|', ' | | ', '|___|'],
 'J': ['      _', '    | |', ' _  | |', '| |_| |'],
 'K': [' _  __', '| |/ /', "| ' < ", '|_|\\_\\'],
 'L': [' _     ', '| |    ', '| |___ ', '|_____|'],
 'M': [' _   _ ', '| \\_/ |', '| |v| |', '|_| |_|'],
 'N': [' _   _ ', '| \\ | |', '| |\\| |', '|_| \\_|'],
 'O': ['  ___  ', ' / _ \\ ', '| (_) |', ' \\___/ '],
 'P': [' ____  ', '|  _ \\ ', '| |_) |', '|  __/ '],
 'Q': ['  ___  ', ' / _ \\ ', '| (_) |', ' \\__\\_\\'],
 'R': [' ____  ', '|  _ \\ ', '| |_) |', '|_|\\_\\ '],
 'S': [' ____ ', '/ ___|', '\\___ \\', '|____/'],
 'T': [' _____ ', '|_   _|', '  | |  ', '  |_|  '],
 'U': [' _   _ ', '| | | |', '| | | |', ' \\___/ '],
 'V': ['__     __', '\\ \\   / /', ' \\ \\ / / ', '  \\_V_/  '],
 'W': ['__        __', '\\ \\  /\\  / /', ' \\ \\/  \\/ / ', '  \\_/\\_/    '],
 'X': ['__  __', '\\ \\/ /', ' >  < ', '/_/\\_\\'],
 'Y': ['__   __', '\\ \\ / /', ' \\ V / ', '  |_|  '],
 'Z': [' _____ ', '|___  /', '   / / ', '  /___|']}

FONT_THIN = {' ': ['   ', '   ', '   ', '   ', '   '],
 '!': [' █ ', ' █ ', ' █ ', '   ', ' █ '],
 "'": [' █ ', ' █ ', '   ', '   ', '   '],
 ',': ['   ', '   ', '   ', ' █ ', '█  '],
 '-': ['   ', '   ', '███', '   ', '   '],
 '.': ['   ', '   ', '   ', '   ', ' █ '],
 '0': [' █ ', '█ █', '█ █', '█ █', ' █ '],
 '1': [' █ ', '██ ', ' █ ', ' █ ', '███'],
 '2': ['██ ', '  █', ' █ ', '█  ', '███'],
 '3': ['██ ', '  █', '██ ', '  █', '██ '],
 '4': ['█ █', '█ █', '███', '  █', '  █'],
 '5': ['███', '█  ', '██ ', '  █', '██ '],
 '6': [' ██', '█  ', '██ ', '█ █', ' ██'],
 '7': ['███', '  █', ' █ ', ' █ ', ' █ '],
 '8': [' █ ', '█ █', ' █ ', '█ █', ' █ '],
 '9': [' ██', '█ █', ' ██', '  █', '██ '],
 '?': ['██ ', '  █', ' █ ', '   ', ' █ '],
 'A': [' █ ', '█ █', '███', '█ █', '█ █'],
 'B': ['██ ', '█ █', '██ ', '█ █', '██ '],
 'C': [' ██', '█  ', '█  ', '█  ', ' ██'],
 'D': ['██ ', '█ █', '█ █', '█ █', '██ '],
 'E': ['███', '█  ', '██ ', '█  ', '███'],
 'F': ['███', '█  ', '██ ', '█  ', '█  '],
 'G': [' ██', '█  ', '█ █', '█ █', ' ██'],
 'H': ['█ █', '█ █', '███', '█ █', '█ █'],
 'I': ['███', ' █ ', ' █ ', ' █ ', '███'],
 'J': ['  █', '  █', '  █', '█ █', ' █ '],
 'K': ['█ █', '██ ', '█  ', '██ ', '█ █'],
 'L': ['█  ', '█  ', '█  ', '█  ', '███'],
 'M': ['█ █', '███', '█ █', '█ █', '█ █'],
 'N': ['█ █', '██ ', '█ █', ' ██', '█ █'],
 'O': [' █ ', '█ █', '█ █', '█ █', ' █ '],
 'P': ['██ ', '█ █', '██ ', '█  ', '█  '],
 'Q': [' █ ', '█ █', '█ █', '███', '  █'],
 'R': ['██ ', '█ █', '██ ', '██ ', '█ █'],
 'S': [' ██', '█  ', ' █ ', '  █', '██ '],
 'T': ['███', ' █ ', ' █ ', ' █ ', ' █ '],
 'U': ['█ █', '█ █', '█ █', '█ █', '███'],
 'V': ['█ █', '█ █', '█ █', ' █ ', ' █ '],
 'W': ['█ █', '█ █', '█ █', '███', '█ █'],
 'X': ['█ █', '█ █', ' █ ', '█ █', '█ █'],
 'Y': ['█ █', '█ █', ' █ ', ' █ ', ' █ '],
 'Z': ['███', '  █', ' █ ', '█  ', '███']}

FONT_5 = {' ': ['000', '000', '000', '000', '000'],
 '!': ['00100', '00100', '00100', '00000', '00100'],
 "'": ['01100', '01100', '00000', '00000', '00000'],
 ',': ['00000', '00000', '00000', '01100', '01000'],
 '-': ['00000', '00000', '11111', '00000', '00000'],
 '.': ['00000', '00000', '00000', '00000', '01100'],
 '0': ['01110', '10011', '10101', '11001', '01110'],
 '1': ['00100', '01100', '00100', '00100', '01110'],
 '2': ['01110', '10001', '00010', '00100', '11111'],
 '3': ['11110', '00001', '00110', '00001', '11110'],
 '4': ['00010', '00110', '01010', '11111', '00010'],
 '5': ['11111', '10000', '11110', '00001', '11110'],
 '6': ['01110', '10000', '11110', '10001', '01110'],
 '7': ['11111', '00001', '00010', '00100', '00100'],
 '8': ['01110', '10001', '01110', '10001', '01110'],
 '9': ['01110', '10001', '01111', '00001', '01110'],
 '?': ['01110', '10001', '00110', '00000', '00100'],
 'A': ['01110', '10001', '11111', '10001', '10001'],
 'B': ['11110', '10001', '11110', '10001', '11110'],
 'C': ['01111', '10000', '10000', '10000', '01111'],
 'D': ['11110', '10001', '10001', '10001', '11110'],
 'E': ['11111', '10000', '11110', '10000', '11111'],
 'F': ['11111', '10000', '11110', '10000', '10000'],
 'G': ['01111', '10000', '10011', '10001', '01111'],
 'H': ['10001', '10001', '11111', '10001', '10001'],
 'I': ['11111', '00100', '00100', '00100', '11111'],
 'J': ['00001', '00001', '00001', '10001', '01110'],
 'K': ['10001', '10010', '11100', '10010', '10001'],
 'L': ['10000', '10000', '10000', '10000', '11111'],
 'M': ['10001', '11011', '10101', '10001', '10001'],
 'N': ['10001', '11001', '10101', '10011', '10001'],
 'O': ['01110', '10001', '10001', '10001', '01110'],
 'P': ['11110', '10001', '11110', '10000', '10000'],
 'Q': ['01110', '10001', '10001', '10010', '01101'],
 'R': ['11110', '10001', '11110', '10010', '10001'],
 'S': ['01111', '10000', '11110', '00001', '11110'],
 'T': ['11111', '00100', '00100', '00100', '00100'],
 'U': ['10001', '10001', '10001', '10001', '01110'],
 'V': ['10001', '10001', '10001', '01010', '00100'],
 'W': ['10001', '10001', '10101', '11011', '10001'],
 'X': ['10001', '01010', '00100', '01010', '10001'],
 'Y': ['10001', '01010', '00100', '00100', '00100'],
 'Z': ['11111', '00010', '00100', '01000', '11111']}

FONT_COMPACT = {' ': ['0000', '0000', '0000', '0000', '0000'],
 '!': ['0010', '0010', '0010', '0000', '0010'],
 "'": ['0110', '0110', '0000', '0000', '0000'],
 ',': ['0000', '0000', '0000', '0110', '0100'],
 '-': ['0000', '0000', '1110', '0000', '0000'],
 '.': ['0000', '0000', '0000', '0000', '0110'],
 '0': ['0110', '1001', '1001', '1001', '0110'],
 '1': ['0010', '0110', '0010', '0010', '0110'],
 '2': ['0110', '1001', '0010', '0100', '1110'],
 '3': ['1110', '0001', '0110', '0001', '1110'],
 '4': ['0100', '1100', '0110', '0010', '0010'],
 '5': ['1111', '1000', '1110', '0001', '1110'],
 '6': ['0110', '1000', '1110', '1001', '0110'],
 '7': ['1111', '0001', '0010', '0100', '0100'],
 '8': ['0110', '1001', '0110', '1001', '0110'],
 '9': ['0110', '1001', '0111', '0001', '0110'],
 '?': ['0110', '1001', '0110', '0000', '0010'],
 'A': ['0110', '1001', '1111', '1001', '1001'],
 'B': ['1110', '1001', '1110', '1001', '1110'],
 'C': ['0111', '1000', '1000', '1000', '0111'],
 'D': ['1110', '1001', '1001', '1001', '1110'],
 'E': ['1111', '1000', '1110', '1000', '1111'],
 'F': ['1111', '1000', '1110', '1000', '1000'],
 'G': ['0111', '1000', '1011', '1001', '0110'],
 'H': ['1001', '1001', '1111', '1001', '1001'],
 'I': ['1111', '0100', '0100', '0100', '1111'],
 'J': ['0010', '0001', '0001', '1001', '0110'],
 'K': ['1001', '1010', '1100', '1010', '1001'],
 'L': ['1000', '1000', '1000', '1000', '1111'],
 'M': ['1001', '1111', '1111', '1001', '1001'],
 'N': ['1001', '1101', '1101', '1011', '1001'],
 'O': ['0110', '1001', '1001', '1001', '0110'],
 'P': ['1110', '1001', '1110', '1000', '1000'],
 'Q': ['0110', '1001', '1001', '1010', '0111'],
 'R': ['1110', '1001', '1110', '1010', '1001'],
 'S': ['0111', '1000', '0110', '0001', '1110'],
 'T': ['1111', '0100', '0100', '0100', '0100'],
 'U': ['1001', '1001', '1001', '1001', '0110'],
 'V': ['1001', '1001', '1001', '0101', '0010'],
 'W': ['1001', '1001', '1011', '1101', '1001'],
 'X': ['1001', '0101', '0010', '0101', '1001'],
 'Y': ['1001', '0101', '0010', '0010', '0010'],
 'Z': ['1111', '0010', '0100', '1000', '1111']}

FONT_ROUNDED = {' ': ['00000', '00000', '00000', '00000', '00000'],
 '!': ['00100', '00100', '00100', '00000', '00100'],
 "'": ['01100', '01100', '00000', '00000', '00000'],
 ',': ['00000', '00000', '00000', '01100', '01000'],
 '-': ['00000', '00000', '01110', '00000', '00000'],
 '.': ['00000', '00000', '00000', '00000', '01100'],
 '0': ['01110', '10011', '10101', '11001', '01110'],
 '1': ['00100', '01100', '00100', '00100', '01110'],
 '2': ['01110', '10001', '00010', '00100', '11111'],
 '3': ['01110', '00001', '00110', '00001', '01110'],
 '4': ['01010', '01010', '01110', '00100', '00100'],
 '5': ['11111', '10000', '11110', '00001', '11110'],
 '6': ['01110', '10000', '11110', '10001', '01110'],
 '7': ['11111', '00001', '00010', '00100', '00100'],
 '8': ['01110', '10001', '01110', '10001', '01110'],
 '9': ['01110', '10001', '01111', '00001', '01110'],
 '?': ['01100', '10010', '00110', '00000', '00100'],
 'A': ['01110', '10001', '11111', '10001', '10001'],
 'B': ['11110', '10001', '11110', '10001', '11110'],
 'C': ['01110', '10000', '10000', '10000', '01110'],
 'D': ['11110', '10001', '10001', '10001', '11110'],
 'E': ['01111', '10000', '11110', '10000', '01111'],
 'F': ['01111', '10000', '11110', '10000', '10000'],
 'G': ['01110', '10000', '10111', '10001', '01110'],
 'H': ['10001', '10001', '11111', '10001', '10001'],
 'I': ['01110', '00100', '00100', '00100', '01110'],
 'J': ['00100', '00010', '00010', '01010', '01100'],
 'K': ['10001', '10010', '11100', '10010', '10001'],
 'L': ['10000', '10000', '10000', '10000', '01111'],
 'M': ['10001', '11011', '10101', '10001', '10001'],
 'N': ['10001', '11001', '10101', '10011', '10001'],
 'O': ['01110', '10001', '10001', '10001', '01110'],
 'P': ['01110', '10001', '01110', '10000', '10000'],
 'Q': ['01110', '10001', '10001', '10010', '01101'],
 'R': ['01110', '10001', '01110', '10100', '10001'],
 'S': ['01111', '10000', '01110', '00001', '11110'],
 'T': ['01110', '00100', '00100', '00100', '00100'],
 'U': ['10001', '10001', '10001', '10001', '01110'],
 'V': ['10001', '10001', '10001', '01010', '00100'],
 'W': ['10001', '10001', '10101', '11011', '10001'],
 'X': ['10010', '01010', '00100', '01010', '10010'],
 'Y': ['10010', '01010', '00100', '00100', '00100'],
 'Z': ['11111', '00010', '00100', '01000', '11111']}

FONTS = {
    'double':  FONT_DOUBLE,  # Japanese Anime / Cyber double-line box font
    'mini':    FONT_MINI,    # Razor-sharp single line box font
    'slant':   FONT_SLANT,   # Stylized 4-row calligraphy
    'smooth':  FONT_SMOOTH,  # 5-row half-block curves
    'rounded': FONT_ROUNDED, # 5x5 rounded font
    'compact': FONT_COMPACT, # 4x5 compact
    'chunky':  FONT_5,       # 5x5 solid
    'thin':    FONT_THIN,    # 3x5 clean elegant thin stroke
}

# --------------------------------------------------------------------------
# Universal script support (v1.3)
# Before v1.3 only FONT_DOUBLE carried Cyrillic / accented Latin / kana, so any
# other font rendered Russian words as invisible blanks and dropped letters like
# Î from "dîner".  Now every font is completed at import time:
#   * FONT_MINI gains single-line versions of every double-line glyph
#     (mechanical double->single box-drawing translation, identical style),
#   * block fonts (chunky/compact/rounded) gain bitmap versions of those glyphs,
#   * raw-art fonts (slant/smooth/thin) reuse the single-line art as fallback.
# At render time a per-char resolver additionally NFC-normalises, strips
# combining marks and transliterates Cyrillic, so decomposed or exotic input
# still produces a readable glyph instead of a hole.
# --------------------------------------------------------------------------
_D2S = str.maketrans({
    '\u2554': '\u250c', '\u2557': '\u2510', '\u255a': '\u2514', '\u255d': '\u2518',
    '\u2550': '\u2500', '\u2551': '\u2502', '\u2560': '\u251c', '\u2563': '\u2524',
    '\u2566': '\u252c', '\u2569': '\u2534', '\u256c': '\u253c', '\u255e': '\u251c',
    '\u255f': '\u251c', '\u2561': '\u2524', '\u2562': '\u2524', '\u2564': '\u252c',
    '\u2565': '\u252c', '\u2567': '\u2534', '\u2568': '\u2534', '\u256a': '\u253c',
    '\u256d': '\u250c', '\u256e': '\u2510', '\u256f': '\u2518', '\u2570': '\u2514',
    '\u2503': '\u2502', '\u2501': '\u2500',
})
for _ch, _g in list(FONT_DOUBLE.items()):
    if _ch not in FONT_MINI:
        FONT_MINI[_ch] = [r.ljust(max(len(x) for x in _g)) for r in (x.translate(_D2S) for x in _g)]



# ---- v1.4: full kana coverage. 4-row single-line bodies live in FONT_MINI;
# dakuten/handakuten versions are composed by prefixing a dots/circle row, and
# FONT_DOUBLE receives single->double translations of anything it lacks, so the
# import-time completion pipeline below propagates every kana to every font.
_K = lambda s: s.split('/')
_DAK = ' · · '
_HAN = ' ° '
_KANA_BODY = {
 # hiragana
 'き': _K('─┬─┐/ │ │/─┼─┘/ ╵╲'), 'く': _K(' ╷/╱/╲/ '), 'け': _K('│ ╷/│╱/─┤/│ ╵'),
 'こ': _K('──╮//──╯//'), 'さ': _K('─┬─╮/ │╱/─┼─╯/ '), 'し': _K('│/│/│ ╱/└─╯'),
 'す': _K('─┬─/ ││/─┼─/ ╵'), 'せ': _K(' ╷ ╷/─┤ ├─/ │ │/ ╵ ╵'), 'そ': _K('╱─┐/ /╱─┐/ ╵'),
 'た': _K('─┬─┐/ │ ╱/─┼─╯/ ╵'), 'ち': _K('─┬╮/ ││/─┼/ ╵'), 'つ': _K('╱─╮/ ╵/╲─╮/ ╵'),
 'て': _K('─┬─╮/ │ ╵/─┼─╮/ '), 'と': _K('│ ╷/│ │/├─/╵'), 'な': _K('─┬─╮/ │╱│/─┼─╯/ ╵'),
 'に': _K('│ ╷/│/├─/╵ ╵'), 'ぬ': _K('─┬─╮/ ││╱/─┼─╯/ ╵·'), 'ね': _K('│ ╷/│╲/│ │/╵ ╵'),
 'ひ': _K('╷ ╷/├─┤/│ │/╵ ╵'), 'ふ': _K(' ╱╲/╱ ╲/╲╱/'), 'へ': _K(' ╱╲/╱ ╲/╱ ╲/'),
 'ほ': _K('─┬─┐/─┼─┤/ │ │/ ╵ ╵'), 'ま': _K('─┬─/─┼─/ │╱/ ╯'), 'み': _K('╱╲╮/ │╱/╲╮│/ ╵╵'),
 'む': _K('─┬─╮/ ││/─┼─╯/ ·╵'), 'め': _K('╱─╲/│╳│/╲ ╱/'), 'も': _K('│─╮/│─┤/│/╵──'),
 'や': _K('╱─╮/└┐│/─┼╯/ '), 'ゆ': _K('│ ╷ │/│ │╱│/─┼─╯/ ╵'), 'よ': _K(' ╷/─┼─╮/ │╱/ ─╯'),
 'ら': _K('─┬─╮/ │╱/─┼─╯/ ╵'), 'り': _K('│ ╷/│ │/│ │/╵ ╵'), 'る': _K('─┬─╮/ │╱/──╯/ ╮'),
 'れ': _K('│ ╷/│ │/│╱╲/╵ ╵'), 'ろ': _K('─┬─╮/ ╵ │/ ╱─╯/╱'), 'わ': _K('│╱─/│ │/└─╯/ '),
 # katakana
 'ア': _K('─┬─/ │╱/─┼─/ ╵'), 'イ': _K(' ╱╷/╱ │/ │/ ╵'), 'ウ': _K(' ╷/─┬┐/ ││/ ╵'),
 'エ': _K('─┬─/ │/─┼─/'), 'オ': _K('─┬─/╱│/─┼─/ ╵'), 'カ': _K('─┬╮/ ││/ ╵│/ ╱'),
 'キ': _K('─┬─/─┼─/ │/ ╵'), 'ク': _K('╱─╮/ ╱/ ╱/╱'), 'ケ': _K(' ╱╷/─┼─/ │╱/ ╵'),
 'コ': _K('┌─/│/│/└─'), 'サ': _K('╷ ╷/│╱│/╱ │/ ╵'), 'シ': _K('╷ ╷/│ ╱/╵╱/'),
 'ス': _K('─┬─/ │╱/╱╲/'), 'セ': _K(' ╷ │/─┤ │/ │─┤/ ╵ ╵'), 'ソ': _K('╷ ╷/│ │/╵ │/'),
 'タ': _K('─┬─/ │╱/──╯/ '), 'チ': _K('─┬─╮/ │ │/─┼─╯/ '), 'ツ': _K('╱─╮/ ╱/╱─╯/╵'),
 'テ': _K('─┬─╮/ │ ╵/─┼─╮/ '), 'ト': _K('│ ·/│/│/╵'), 'ナ': _K('─┬╮/╱│/─┼╯/ '),
 'ニ': _K('─┬╮//─┴╯//'), 'ヌ': _K('─┬─╮/ │╱/─┼─╯/ ╱╲'), 'ネ': _K(' ╷/─┼─╮/ │╱/ ╵'),
 'ノ': _K(' ╷/ ╱/╱/╲'), 'ハ': _K('╷ ╷/╱ ╲/╱ ╲/'), 'ヒ': _K('─┬╮/ │/─┼╯/'),
 'フ': _K('──╮/ ╱/╱/'), 'ヘ': _K(' ╱╲/╱ ╲//'), 'ホ': _K(' ╷/─┼─/╱│╲/ '),
 'マ': _K('─┬─╮/ │╱/──╯/'), 'ミ': _K('──╮/──╮/──╯/'), 'ム': _K('─┬─╮/ ││/─┼─╯/ ╱╲'),
 'メ': _K('╲ ╱/ ╳/╱ ╲/'), 'モ': _K('─┬─/─┼─/ │/─╨─'), 'ヤ': _K('╱─╮/└─┤/─┼/ ╵'),
 'ユ': _K('│─╮/│ │/└─╯/'), 'ヨ': _K('┌─/├─/│/└─'), 'ラ': _K('─┬╮/ ╵│/─┼╯/ ╵'),
 'リ': _K('│ ╷/│ │/│ │/╵ ╵'), 'ル': _K('│ ╷/│ │/│ │╱/╵ ╵'), 'レ': _K('│/│ ╱/│╱/│╱'),
 'ロ': _K('┌─┐/│ │/│ │/└─┘'), 'ワ': _K('─┬─╮/ │ │/ └─╯/ ╵'), 'ヲ': _K('─┬─/─┼/ │╯/ ╵'),
 'ン': _K('╷ ╷/│ ╱/╱/ '), 'ヴ': _K(' ╷/─┐/ ││/ ╵'),
}
def _pad(art):
    w = max((len(r) for r in art), default=0)
    return [r.ljust(w) for r in art]

_SMALL_MAP = {'ぁ': 'あ', 'ぃ': 'い', 'ぅ': 'う', 'ぇ': 'え', 'ぉ': 'お',
              'ゃ': 'や', 'ゅ': 'ゆ', 'ょ': 'よ', 'っ': 'つ',
              'ァ': 'ア', 'ィ': 'イ', 'ゥ': 'ウ', 'ェ': 'エ', 'ォ': 'オ',
              'ャ': 'ヤ', 'ュ': 'ユ', 'ョ': 'ヨ', 'ッ': 'ツ', 'ヵ': 'カ', 'ヶ': 'ケ'}
_DAK_MAP = {'ぎ': 'き', 'ぐ': 'く', 'げ': 'け', 'ご': 'こ', 'ざ': 'さ', 'じ': 'し',
            'ず': 'す', 'ぜ': 'せ', 'ぞ': 'そ', 'だ': 'た', 'ぢ': 'ち', 'づ': 'つ',
            'で': 'て', 'ど': 'と', 'ば': 'は', 'び': 'ひ', 'ぶ': 'ふ', 'べ': 'へ',
            'ぼ': 'ほ', 'ガ': 'カ', 'ギ': 'キ', 'グ': 'ク', 'ゲ': 'ケ', 'ゴ': 'コ',
            'ザ': 'サ', 'ジ': 'シ', 'ズ': 'ス', 'ゼ': 'セ', 'ゾ': 'ソ', 'ダ': 'タ',
            'ヂ': 'チ', 'ヅ': 'ツ', 'デ': 'テ', 'ド': 'ト', 'バ': 'ハ', 'ビ': 'ヒ',
            'ブ': 'フ', 'ベ': 'ヘ', 'ボ': 'ホ'}
_HAN_MAP = {'ぱ': 'は', 'ぴ': 'ひ', 'ぷ': 'ふ', 'ぺ': 'へ', 'ぽ': 'ほ',
            'パ': 'ハ', 'ピ': 'ヒ', 'プ': 'フ', 'ペ': 'ヘ', 'ポ': 'ホ'}

for _c, _b in _KANA_BODY.items():
    if _c not in FONT_MINI:
        FONT_MINI[_c] = _pad(['     '] + _b)
for _c, _base in _SMALL_MAP.items():
    if _c not in FONT_MINI and _base in FONT_MINI:
        FONT_MINI[_c] = FONT_MINI[_base]
for _c, _base in _DAK_MAP.items():
    if _c not in FONT_MINI and _base in _KANA_BODY:
        FONT_MINI[_c] = _pad([_DAK] + _KANA_BODY[_base])
for _c, _base in _HAN_MAP.items():
    if _c not in FONT_MINI and _base in _KANA_BODY:
        FONT_MINI[_c] = _pad([_HAN] + _KANA_BODY[_base])

# single -> double box translation so FONT_DOUBLE also gains the kana
_S2D = str.maketrans({
    '┌': '╔', '┐': '╗', '└': '╚', '┘': '╝', '─': '═', '│': '║',
    '├': '╠', '┤': '', '┬': '', '┴': '', '┼': '',
})
for _c, _g in list(FONT_MINI.items()):
    if _c not in FONT_DOUBLE:
        FONT_DOUBLE[_c] = [r.ljust(max(len(x) for x in _g)) for r in (x.translate(_S2D) for x in _g)]
del _K, _DAK, _HAN, _KANA_BODY, _SMALL_MAP, _DAK_MAP, _HAN_MAP, _S2D, _c, _b, _base, _pad

def _art_bitmap(art):
    """Box-drawing art -> '0'/'1' bitmap rows (any inked cell becomes solid)."""
    h = len(art)
    w = max((len(r) for r in art), default=1)
    return [''.join('1' if c < len(art[r]) and art[r][c] != ' ' else '0'
                    for c in range(w)) for r in range(h)]


for _ch, _g in list(FONT_MINI.items()):
    _bm = None
    for _name in ('chunky', 'compact', 'rounded'):
        _f = FONTS[_name]
        if _ch not in _f:
            if _bm is None:
                _bm = _art_bitmap(_g)
            _f[_ch] = _bm
    for _name in ('slant', 'smooth', 'thin'):
        _f = FONTS[_name]
        if _ch not in _f:
            _f[_ch] = _g
del _ch, _g, _name, _f

_CYR_LAT = {'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
            'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
            'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'}


def nfc(s):
    return unicodedata.normalize('NFC', s)


def strip_marks(s):
    """'î'->'i', 'й'->'и', 'ệ'->'e' — base letters without combining marks."""
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if not unicodedata.combining(c))

# Typographic punctuation -> the ASCII equivalents every font carries, so
# "m’attendre", "«oui»", "dash—words" never render with holes (v1.3).
_PUNCT_MAP = str.maketrans({
    '’': "'", '‘': "'", 'ʼ': "'", '’': "'",
    '\u201c': '"', '\u201d': '"', '\u00ab': '"', '\u00bb': '"', '\u2033': '"',
    '\u2013': '-', '\u2014': '-', '\u2015': '-', '\u2010': '-', '\u2011': '-',
    '\u2026': '.', '\u2024': '.', '\u2032': "'",
})


def _translit(ch):
    return _CYR_LAT.get(ch.lower())


def _char_resolves(font, ch):
    if ch in font:
        return True
    for c in (ch.upper(), ch.lower()):
        if c in font:
            return True
    n = nfc(ch)
    for c in (n, n.upper(), n.lower()):
        if c in font:
            return True
    s = strip_marks(ch)
    return bool(s) and any(c in font for c in (s, s.upper(), s.lower()))


def expand_word(font, word):
    """Split a word into renderable chars; chars the font cannot show even after
    NFC/mark-stripping are transliterated (Cyrillic -> Latin) or dropped (ь/ъ)."""
    out = []
    for ch in word:
        if _char_resolves(font, ch):
            out.append(ch)
            continue
        t = _translit(ch)
        if t:
            out.extend(t if ch.islower() else t.upper())
        # t == '' (soft/hard sign): drop silently
    return out


def lookup_glyph(font, ch, blank):
    for c in (ch, ch.upper(), ch.lower()):
        if c in font:
            return font[c]
    n = nfc(ch)
    for c in (n, n.upper(), n.lower()):
        if c in font:
            return font[c]
    s = strip_marks(ch)
    for c in (s, s.upper(), s.lower()):
        if c and c in font:
            return font[c]
    return blank


NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (-1, 1), (1, -1), (1, 1)]

# --------------------------------------------------------------------------
# Per-font visual profiles (settings automatically snap when a font is chosen)
# --------------------------------------------------------------------------
FONT_PROFILES = {
    'double':  {'scale': 0, 'fill': '█', 'gap': 1, 'color_mode': 'solid',
                'outline': False, 'bold': True, 'glow': True, 'shadow': False,
                'glow_color': '#3b1261', 'case': 'upper', 'color': '#00f0ff', 'bg': 'none'},
    'mini':    {'scale': 0, 'fill': '█', 'gap': 1, 'color_mode': 'solid',
                'outline': False, 'bold': True, 'glow': True, 'shadow': False,
                'glow_color': '#ff2a85', 'case': 'upper', 'color': '#00f0ff', 'bg': 'none'},
    'slant':   {'scale': 0, 'fill': '█', 'gap': 1, 'color_mode': 'solid',
                'outline': False, 'bold': True, 'glow': True, 'shadow': False,
                'glow_color': '#3b1261', 'case': 'upper', 'color': '#00f0ff', 'bg': 'none'},
    'smooth':  {'scale': 0, 'fill': '█', 'gap': 1, 'color_mode': 'solid',
                'outline': False, 'bold': True, 'glow': True, 'shadow': False,
                'glow_color': '#3b1261', 'case': 'upper', 'color': '#ff2a85', 'bg': 'none'},
    'rounded': {'scale': 0, 'fill': '█', 'gap': 2, 'color_mode': 'solid',
                'outline': False, 'bold': True, 'glow': False, 'shadow': False,
                'glow_color': '#3b1261', 'case': 'upper', 'color': '#ffd479', 'bg': 'none'},
    'compact': {'scale': 0, 'fill': '█', 'gap': 1, 'color_mode': 'solid',
                'outline': False, 'bold': True, 'glow': False, 'shadow': False,
                'glow_color': '#222222', 'case': 'upper', 'color': '#e0e0e0', 'bg': 'none'},
    'chunky':  {'scale': 0, 'fill': '█', 'gap': 2, 'color_mode': 'solid',
                'outline': False, 'bold': True, 'glow': False, 'shadow': True,
                'glow_color': '#3b1261', 'case': 'upper', 'color': '#f5f5f5', 'bg': 'none'},
    'thin':    {'scale': 0, 'fill': '█', 'gap': 2, 'color_mode': 'solid',
                'outline': False, 'bold': True, 'glow': True, 'shadow': False,
                'glow_color': '#3b1261', 'case': 'upper', 'color': '#00f0ff', 'bg': 'none'},
}
GENERIC_FIGLET_PROFILE = {'scale': 1, 'fill': '█', 'gap': 1, 'color_mode': 'solid',
                          'outline': False, 'bold': True, 'glow': False, 'shadow': False,
                          'case': 'upper', 'color': '#f5f5f5', 'bg': 'none'}

THEMES = {
    'Neon': {
        'font': 'double', 'scale': 0, 'fill': '█', 'gap': 1,
        'color_mode': 'solid', 'color': '#00f0ff', 'bg': 'none',
        'outline': False, 'bold': True, 'glow': True, 'shadow': False,
        'glow_color': '#3b1261', 'case': 'upper', 'transition': 'fade',
        'paused_dim': 0.5, 'fade': 0.22
    },
    'Cyber': {
        'font': 'slant', 'scale': 0, 'fill': '█', 'gap': 1,
        'color_mode': 'gradient', 'color': '#ff2a85', 'bg': 'none',
        'outline': False, 'bold': True, 'glow': True, 'shadow': False,
        'glow_color': '#2a0845', 'case': 'upper', 'transition': 'fade',
        'paused_dim': 0.5, 'fade': 0.25
    },
    'Anime': {
        'font': 'smooth', 'scale': 0, 'fill': '█', 'gap': 1,
        'color_mode': 'solid', 'color': '#ff6ad5', 'bg': 'none',
        'outline': False, 'bold': True, 'glow': True, 'shadow': False,
        'glow_color': '#3b1261', 'case': 'upper', 'transition': 'fade',
        'paused_dim': 0.45, 'fade': 0.2
    },
    'Matrix': {
        'font': 'chunky', 'scale': 0, 'fill': '█', 'gap': 2,
        'color_mode': 'solid', 'color': '#00ff41', 'bg': 'none',
        'outline': False, 'bold': True, 'glow': False, 'shadow': True,
        'glow_color': '#003b0f', 'case': 'upper', 'transition': 'fade',
        'paused_dim': 0.4, 'fade': 0.18
    },
    'Minimal': {
        'font': 'mini', 'scale': 0, 'fill': '█', 'gap': 1,
        'color_mode': 'solid', 'color': '#e0e0e0', 'bg': 'none',
        'outline': False, 'bold': False, 'glow': False, 'shadow': False,
        'glow_color': '#222222', 'case': 'upper', 'transition': 'instant',
        'paused_dim': 0.45, 'fade': 0.0
    },
}

DEFAULTS = {
    'color': '#00f0ff',
    'color_mode': 'solid',
    'bg': 'none',
    'font': 'double',
    'scale': 0,  # 0 = auto-fit to screen size
    'fill': DEFAULT_FILL,
    'gap': 1,
    'outline': False,
    'bold': True,
    'glow': True,
    'shadow': False,
    'glow_color': '#3b1261',
    'case': 'upper',
    'v_offset': 0,
    'transition': 'fade',
    'paused_dim': 0.45,
    'fade': 0.22,
    'player': '',
    'plain': False,
    'resync': 1.5,
    'tick': 0.05,
    'offset': 0.0,  # Manual sync offset in seconds (e.g. -0.10 for 100ms advance)
    'lyrics_dir': os.path.expanduser('~/.local/share/word_sync/lyrics'),
    'cache_dir': os.path.expanduser('~/.cache/word_sync'),
}

VERSION = '3.1'

should_exit = False


# --------------------------------------------------------------------------
# Color Helpers
# --------------------------------------------------------------------------
def resolve_color(value):
    if not value or value == 'none':
        return (255, 255, 255)
    hexval = NAMED_COLORS.get(value.lower(), value).lstrip('#')
    if len(hexval) != 6:
        raise argparse.ArgumentTypeError(f"Invalid color: '{value}'")
    try:
        return tuple(int(hexval[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid hex color: '{value}'")


def color_code(rgb, intensity=1.0):
    i = max(0.0, min(1.0, intensity))
    r = int(rgb[0] * i)
    g = int(rgb[1] * i)
    b = int(rgb[2] * i)
    return f'\033[{_fg(r, g, b)}m'


def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def apply_case(word, case):
    if case == 'upper':
        return word.upper()
    if case == 'lower':
        return word.lower()
    if case == 'title':
        return word.title()
    return word


# --------------------------------------------------------------------------
# Glyph Processing & Direct-Art Rendering
# --------------------------------------------------------------------------
def outline_grid(grid):
    h = len(grid)
    w = len(grid[0])
    out = []
    for r in range(h):
        row = ''
        for c in range(w):
            if grid[r][c] == '1':
                row += '1'
            else:
                near = any(0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr][c + dc] == '1'
                           for dr, dc in NEIGHBORS)
                row += '2' if near else '0'
        out.append(row)
    return out


def render_scaled(grid, fill, scale, outline, hollow):
    h = len(grid)
    if scale <= 1:
        def m(c):
            if c == '1':
                return ' ' if hollow else fill
            if c == '2':
                return fill if (outline and not hollow) else ' '
            return ' '
        return [''.join(m(c) for c in grid[r]) for r in range(h)]
    base = []
    for r in range(h):
        row = ''
        for c in range(len(grid[r])):
            ch = grid[r][c]
            if ch == '1':
                row += ' ' if hollow else fill
            elif ch == '2':
                row += fill if (outline and not hollow) else ' '
            else:
                row += ' '
        base.append(row)
    out = []
    for row in base:
        wide = ''.join(c * scale for c in row)
        for _ in range(scale):
            out.append(wide)
    return out


def generate_glyph_rows(word, font, fill, gap, scale, outline, case, hollow=False):
    word = apply_case(nfc(word).translate(_PUNCT_MAP), case)
    # Check if font is a raw text-art font (like FONT_MINI or FONT_SLANT)
    first_glyph = next(iter(font.values()))
    is_raw_art = isinstance(first_glyph[0], str) and any(c not in '012' for c in first_glyph[0])

    if is_raw_art:
        h = len(first_glyph)
        blank = [' ' * len(first_glyph[0])] * h
        chars = expand_word(font, word)
        rows = ['' for _ in range(h)]
        for i, ch in enumerate(chars):
            g = lookup_glyph(font, ch, blank)
            gl = len(g)
            for r in range(h):
                if gl == h:
                    seg = g[r]
                elif gl > 1 and h > 1:   # resample rows if a fallback glyph has a
                    seg = g[int(round(r * (gl - 1) / (h - 1)))]  # different height
                else:
                    seg = g[0] if g else ''
                rows[r] += seg + (gap if i < len(chars) - 1 else '')
        if scale > 1:
            scaled = []
            for r in rows:
                wide = ''.join(c * scale for c in r)
                for _ in range(scale):
                    scaled.append(wide)
            return scaled
        return rows

    chars = expand_word(font, word)
    glyphs = [lookup_glyph(font, ch, font[' ']) for ch in chars]
    grids = [outline_grid(g) if (outline or hollow) else g for g in glyphs]
    letter_rows = [render_scaled(g, fill, scale, outline, hollow) for g in grids]
    h = len(letter_rows[0]) if letter_rows else 0
    rows = []
    for r in range(h):
        seg = ''
        for i in range(len(chars)):
            seg += letter_rows[i][r]
            if i < len(chars) - 1:
                seg += gap
        rows.append(seg)
    return rows


# --------------------------------------------------------------------------
# Player & Lyrics Fetching
# --------------------------------------------------------------------------
def playerctl_available():
    try:
        subprocess.run(['playerctl', '--version'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
        return True
    except Exception:
        return False


# ==========================================================================
# SYNC ENGINE v2  (lyrics fetch -> parse -> per-word timeline -> playback clock)
# ==========================================================================
# Accuracy ladder, best source first:
#   Tier 1  real word timing   Enhanced LRC <mm:ss.xx> tags, or LRCLIB "lyricsfile" words
#   Tier 2  estimated words    syllable-proportional placement inside each line's window
# Line timestamps are the only ground truth in Tier 2, so word positions are estimates; a
# per-song live offset (+ / - / 0 keys, remembered between runs) absorbs any constant bias.
# --------------------------------------------------------------------------

DUR_TOL = 2.0            # LRCLIB itself only guarantees +/-2 s duration matching
VOICED_FRACTION = 0.82   # share of a line window over which word onsets are spread, WHEN the
                         # window is short enough that the phrase plausibly fills it
MIN_SHOW = 0.12          # perceptual floor: shortest readable word display (s)
TAIL_GUARD = 0.05        # last onset stays this far before the next line (s)
REST_CLEAR_MIN = 1.2     # blank the screen once a rest at least this long follows a line
MIN_LINE_GAP = 0.25      # smallest inter-line spacing honoured (s)

# A phrase has a natural maximum duration set by its OWN syllables, independent of how long the
# GAP to the next line happens to be (that gap can hold a guitar solo, a drum fill, plain silence).
# v1.3: the seconds-per-syllable pace is no longer a single fixed constant — it is measured per
# song from the MEDIAN line window (robust to long instrumental breaks), clamped to a sane vocal
# range, and then applied to the phrase's own syllable count plus a held final note.  Slow ballads
# and love songs therefore get a genuinely slower word cadence (up to PACE_MAX s/syllable with a
# longer final hold), while a verse-end line followed by a 20 s break can never crawl across that
# break: its span is bounded by its own content at the song's typical pace, not by 1.7 s/syllable
# times its length (the v1.1 behaviour that let letters wander for 10+ seconds after the singer
# stopped).  Sourced anchor for an average solo sung syllable is ~0.57 s (Racette et al., cited in
# Stahl & Kotz 2014, PMC4172097).
# v2: every cadence knob lives in one dict, trained offline against human
# word-timing annotations (JamendoLyrics corpus, 79 songs / 8 genres / 6 languages)
# plus hand-checked famous tracks.  See eval_harness.py / trainer.py.
TUNE = {
    # v2 optimum — trained offline on JamendoLyrics (79 songs, 21 580 human word
    # onsets, 8 genres, 6 languages) with a two-round random search; validated
    # per genre/language.  mae 0.365->0.299 s, early-words 20%->14% vs v1.5.
    'vf_base': 0.88,   # voiced share of a ~3 s upbeat window
    'vf_knee': 3.745,  # window length where the voiced share starts shrinking
    'vf_slope': 0.02,  # shrink per second of window beyond the knee
    'vf_floor': 0.467, # ballad floor: held notes/reverb dominate
    'pace_min': 0.30,  # s/syllable floor (fast rap)
    'pace_max': 0.822, # s/syllable ceiling (slow ballad articulation)
    'front_on': 0.382, # pace above which held-note front-loading kicks in
    'front_gain': 1.12,   # front-load curve growth per s/syllable above front_on
    'front_max': 0.361,   # max curve exponent bonus
    'hold_a': 0.3,     # held final note: base seconds
    'hold_b': 0.402,   # held final note: growth with pace
    'hold_cap': 2.2,   # held final note: cap
    'rest_a': 0.5,     # post-line rest before clearing: base seconds
    'rest_cap': 3.0,   # ...cap
    'tail_on': 8.594,  # windows longer than this are verse-ends/breaks: shrink the
    'tail_slope': 0.019,  # phrase span so the last word never drifts into the rest
    'tail_floor': 0.55,
    'syl_exp': 0.719,  # syllable-weight exponent (<1 compresses long-word shares)
}
DEFAULT_PACE = 0.60      # used when the song is too short to measure a tempo
PACE_MIN = TUNE['pace_min']   # backward-compatible aliases
PACE_MAX = TUNE['pace_max']


_PARTICLES = set('のにをはがとでへもねよかさ')


def jp_tokens(s):
    """Whitespace-free kana line -> word-sized chunks: cut after a particle once the
    chunk has some body, or every ~5 mora, so Japanese lines reveal progressively
    instead of popping in as one giant word (v1.5)."""
    toks, cur, n = [], '', 0
    for i, c in enumerate(s):
        cur += c
        if c not in _SMALL_KANA:
            n += 1
        nxt = s[i + 1] if i + 1 < len(s) else ''
        if nxt and ((c in _PARTICLES and n >= 2) or n >= 5):
            toks.append(cur)
            cur, n = '', 0
    if cur:
        toks.append(cur)
    return toks or [s]


def line_words(text):
    """Word list for estimation: whitespace split, with kana tokenisation for
    Japanese lines that carry no spaces."""
    words = text.split()
    if len(words) == 1 and len(words[0]) > 4 and _KANA.search(words[0]):
        words = jp_tokens(words[0])
    return words


def _voiced_fraction(window):
    """Share of a line window actually filled by articulated vocal (v1.5/v2).
    Upbeat ~3 s windows are ~82 % voice; ballad windows shrink toward ~55 % as
    held notes, reverb tails and breaths dominate."""
    t = TUNE
    return min(t['vf_base'], max(t['vf_floor'],
               t['vf_base'] - max(0.0, window - t['vf_knee']) * t['vf_slope']))


def http_json(url, tries=3):
    headers = {'User-Agent': 'word-sync/2.1 (https://github.com/word-sync)'}
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == tries - 1:
                return None
            time.sleep(0.3 * (attempt + 1))
        except Exception:
            if attempt == tries - 1:
                return None
            time.sleep(0.3 * (attempt + 1))
    return None


def _dur_gap(rec, duration):
    d = rec.get('duration')
    if not duration or duration <= 0 or d is None:
        return 0.0
    try:
        return abs(float(d) - float(duration))
    except (TypeError, ValueError):
        return 0.0


def _has_word_timing(rec):
    lf = rec.get('lyricsfile')
    return bool(lf) and isinstance(lf, str) and ('start_ms' in lf) and ('words' in lf)


def rank_candidates(results, duration, tol=DUR_TOL):
    """Usable LRCLIB records, best first. A record is usable if it has synced lyrics or a
    lyricsfile, is not instrumental, and its duration is within `tol` s of the player's.
    Order: word-timed first, then smallest duration gap, then API order."""
    if isinstance(results, dict):
        results = [results]
    if not isinstance(results, list):
        return []
    usable = []
    for idx, r in enumerate(results):
        if not isinstance(r, dict) or r.get('instrumental'):
            continue
        if not (r.get('syncedLyrics') or r.get('lyricsfile')):
            continue
        gap = _dur_gap(r, duration)
        if gap > tol:
            continue
        usable.append((0 if _has_word_timing(r) else 1, round(gap, 3), idx, r))
    usable.sort(key=lambda x: x[:3])
    return [u[3] for u in usable]


def lrclib_fetch_records(artist, title, duration):
    """All plausible LRCLIB records for this track (exact-signature hit first, then search)."""
    recs = []
    params = {'artist_name': artist, 'track_name': title}
    if duration and duration > 0:
        params['duration'] = str(int(round(duration)))
    data = http_json('https://lrclib.net/api/get?' + urllib.parse.urlencode(params))
    if isinstance(data, dict):
        recs.append(data)
    data = http_json('https://lrclib.net/api/search?' +
                     urllib.parse.urlencode({'artist_name': artist, 'track_name': title}))
    if isinstance(data, list):
        recs.extend(data)
    if not recs:
        data = http_json('https://lrclib.net/api/search?' +
                         urllib.parse.urlencode({'q': f'{artist} {title}'}))
        if isinstance(data, list):
            recs.extend(data)
    return recs


def _rec_to_cand(r):
    return {'synced': r.get('syncedLyrics') or None,
            'lyricsfile': r.get('lyricsfile') if _has_word_timing(r) else None}


def fetch_lrclib(artist, title, duration, max_n=5):
    """Up to `max_n` plausible lyric candidates, best first, each a dict
    {'synced': str|None, 'lyricsfile': str|None}.  [] when nothing usable.
    (v1.4: the player can cycle through these with the `n` key when the
    duration-matched best hit turns out to be the wrong recording.)"""
    return [_rec_to_cand(r) for r in
            rank_candidates(lrclib_fetch_records(artist, title, duration), duration)[:max_n]]


def sanitize(name):
    return re.sub(r'[/\\?%*:|"<>]+', '_', name).strip()


def local_lyrics_path(artist, title, lyrics_dir):
    if not lyrics_dir:
        return None
    stem = f'{sanitize(artist)} - {sanitize(title)}'
    for ext in ('.enhanced.lrc', '.lrc'):
        p = os.path.join(lyrics_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def cache_key(artist, title, duration):
    dur_tag = f'_{int(round(duration))}' if duration else ''
    return f'{sanitize(artist)}--{sanitize(title)}{dur_tag}.json'


def read_text(path):
    """File content with encoding fallback: UTF-8 (BOM-tolerant) first, then cp1251 (common
    for Russian/Windows .lrc files), then latin-1 as a last resort."""
    for enc in ('utf-8-sig', 'cp1251', 'latin-1'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    return None


def get_lyrics(artist, title, duration, cache_dir, lyrics_dir):
    """Returns {'candidates': [ {...}, ... ]} (best first) or None. Local files win,
    then the on-disk cache (v1.4 multi-candidate format, v1.3 single-record accepted),
    then LRCLIB."""
    loc = local_lyrics_path(artist, title, lyrics_dir)
    if loc:
        t = read_text(loc)
        if t:
            return {'candidates': [{'synced': t, 'lyricsfile': None}]}
    if cache_dir:
        cpath = os.path.join(cache_dir, cache_key(artist, title, duration))
        raw = read_text(cpath) if os.path.exists(cpath) else None
        if raw:
            try:
                c = json.loads(raw)
                if isinstance(c, dict):
                    if isinstance(c.get('candidates'), list) and c['candidates']:
                        return {'candidates': c['candidates']}
                    if c.get('synced') or c.get('lyricsfile'):
                        return {'candidates': [{'synced': c.get('synced'),
                                                'lyricsfile': c.get('lyricsfile')}]}
            except ValueError:
                pass
    remote = fetch_lrclib(artist, title, duration)
    if remote and cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, cache_key(artist, title, duration)),
                      'w', encoding='utf-8') as f:
                json.dump({'candidates': remote}, f)
        except Exception:
            pass
    return {'candidates': remote} if remote else None


# --------------------------------------------------------------------------
# LRC parser (standard + Enhanced LRC)
# --------------------------------------------------------------------------
_TS = r'(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?'
_LINE_TS = re.compile(r'\[' + _TS + r'\]')
_WORD_TS = re.compile(r'<' + _TS + r'>')
_META = re.compile(r'^\[\s*(ar|ti|al|au|by|re|ve|length|la|offset|tool|id|#)\s*:[^\]]*\]\s*$', re.I)
_OFFSET = re.compile(r'\[\s*offset\s*:\s*([+-]?\d+)\s*\]', re.I)


def _secs(m0, m1, m2):
    frac = int(m2) / (10 ** len(m2)) if m2 else 0.0      # '5'->.5  '50'->.50  '123'->.123
    return int(m0) * 60 + int(m1) + frac


def parse_lrc(text):
    """Returns [{'start': s, 'text': str, 'words': [(t, w), ...] | None}], sorted by start.
    Handles: several timestamps on one line (repeated choruses), [mm:ss:xx], 1-3 digit
    fractions, [mm:ss], BOM/CRLF, case-insensitive metadata, Enhanced LRC word tags, and the
    [offset:] tag with the LRC-spec sign (positive = lyrics appear EARLIER)."""
    if not text:
        return []
    text = text.lstrip('\ufeff')
    offset = 0.0
    mo = _OFFSET.search(text)
    if mo:
        try:
            offset = int(mo.group(1)) / 1000.0
        except ValueError:
            offset = 0.0

    out = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or _META.match(raw):
            continue
        stamps, pos = [], 0
        while True:
            m = _LINE_TS.match(raw, pos)
            if not m:
                break
            stamps.append(_secs(*m.groups()))
            pos = m.end()
            while pos < len(raw) and raw[pos] == ' ':
                pos += 1
        if not stamps:
            continue
        body = raw[pos:]

        words = None
        wparts = list(_WORD_TS.finditer(body))
        if wparts:
            words = []
            for i, wm in enumerate(wparts):
                seg_end = wparts[i + 1].start() if i + 1 < len(wparts) else len(body)
                w = nfc(body[wm.end():seg_end].strip())
                if w:
                    words.append((_secs(*wm.groups()), w))
            words = words or None
        clean = nfc(re.sub(r'\s+', ' ', _WORD_TS.sub('', body)).strip())
        if not clean:
            continue
        for st in stamps:
            wl = [(max(0.0, t - offset), w) for t, w in words] if words else None
            out.append({'start': max(0.0, st - offset), 'text': clean, 'words': wl})

    out.sort(key=lambda x: x['start'])
    dedup = []
    for l in out:
        if dedup and abs(dedup[-1]['start'] - l['start']) < 1e-6 and dedup[-1]['text'] == l['text']:
            continue
        dedup.append(l)
    return dedup


# --------------------------------------------------------------------------
# LRCLIB "Lyricsfile" (YAML) word-timing reader. Defensive: any problem -> None, and the
# caller falls back to line-level lyrics. It must never raise into the playback loop.
# --------------------------------------------------------------------------
_LF_KEY = re.compile(r'^(\s*)(-\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$')


def _lf_unquote(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        q, inner = v[0], v[1:-1]
        if q == "'":
            return inner.replace("''", "'")
        out, i = [], 0
        while i < len(inner):
            c = inner[i]
            if c == '\\' and i + 1 < len(inner):
                n = inner[i + 1]
                out.append({'n': '\n', 't': '\t', '"': '"', '\\': '\\', '/': '/'}.get(n, n))
                i += 2
            else:
                out.append(c)
                i += 1
        return ''.join(out)
    return re.sub(r'\s+#.*$', '', v)


def _lf_num(v):
    try:
        s = str(v).strip()
        return None if s in ('', '~', 'null', 'None') else float(s)
    except (TypeError, ValueError):
        return None


def _lf_assign(d, key, val):
    v = nfc(_lf_unquote(val))
    if key == 'text':
        d['text'] = v
    elif key in ('start_ms', 'end_ms'):
        d[key] = v


def parse_lyricsfile(text):
    """Lyricsfile YAML -> [{'start': s, 'text': str, 'words': [(t, w)] | None}] or [] ."""
    if not text or 'lines' not in text:
        return []
    try:
        return _parse_lyricsfile(text)
    except Exception:
        return []


def _parse_lyricsfile(text):
    offset_ms, instrumental = 0.0, False
    lines_out, cur_line, cur_word = [], None, None
    section, in_words = None, False
    lines_indent = words_indent = None
    in_block, block_indent = False, 0

    def flush_word():
        nonlocal cur_word
        if cur_line is not None and cur_word is not None:
            cur_line['_w'].append(cur_word)
        cur_word = None

    def flush_line():
        nonlocal cur_line
        flush_word()
        if cur_line is not None:
            lines_out.append(cur_line)
        cur_line = None

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith('#'):
            continue
        indent = len(raw) - len(raw.lstrip(' '))
        if in_block:
            if indent > block_indent:
                continue
            in_block = False
        m = _LF_KEY.match(raw)
        if not m:
            continue
        ind, dash, key, val = len(m.group(1)), bool(m.group(2)), m.group(3), m.group(4)

        if ind == 0 and not dash:
            flush_line()
            in_words = False
            section = key if key in ('metadata', 'lines', 'plain') else None
            if val in ('|', '>', '|-', '>-', '|+', '>+'):
                in_block, block_indent = True, 0
            continue
        if section == 'metadata':
            v = _lf_unquote(val)
            if key == 'offset_ms':
                x = _lf_num(v)
                offset_ms = x if x is not None else 0.0
            elif key == 'instrumental':
                instrumental = v.lower() == 'true'
            continue
        if section != 'lines':
            continue

        if lines_indent is None and dash:
            lines_indent = ind
        if dash and ind == lines_indent:
            flush_line()
            in_words, words_indent = False, None
            cur_line = {'text': '', 'start_ms': None, 'end_ms': None, '_w': []}
            _lf_assign(cur_line, key, val)
            continue
        if cur_line is None:
            continue
        if key == 'words' and not dash:
            flush_word()
            in_words, words_indent = (val.strip() == ''), None
            continue
        if in_words and dash and ind > lines_indent:
            if words_indent is None:
                words_indent = ind
            if ind == words_indent:
                flush_word()
                cur_word = {'text': '', 'start_ms': None, 'end_ms': None}
                _lf_assign(cur_word, key, val)
                continue
        if in_words and cur_word is not None and ind > (words_indent or lines_indent):
            _lf_assign(cur_word, key, val)
            continue
        _lf_assign(cur_line, key, val)
    flush_line()

    if instrumental:
        return []
    out = []
    for L in lines_out:
        st = _lf_num(L['start_ms'])
        if st is None:
            continue
        frags = []
        for w in L['_w']:
            wt = _lf_num(w['start_ms'])
            if wt is not None:
                frags.append((wt, w['text'] or ''))
        words = _lf_merge_syllables(frags) if frags else []
        words = [(max(0.0, (t - offset_ms) / 1000.0), w) for t, w in words]
        ok = bool(words) and all(b[0] >= a[0] for a, b in zip(words, words[1:]))
        text_ = (L['text'] or '').strip() or (' '.join(w for _, w in words) if ok else '')
        if not text_:
            continue
        out.append({'start': max(0.0, (st - offset_ms) / 1000.0), 'text': text_,
                    'words': words if ok else None})
    out.sort(key=lambda x: x['start'])
    return out


def _lf_merge_syllables(frags):
    """Fragments ending in whitespace close a word; fragments without it glue to the next.
    Blank 'ghost' fragments close any pending word and are dropped."""
    merged, pend, pend_t = [], '', None
    last = len(frags) - 1
    for i, (t, raw) in enumerate(frags):
        if raw.strip() == '':
            if pend:
                merged.append((pend_t, pend))
                pend, pend_t = '', None
            continue
        if pend_t is None:
            pend_t = t
        pend += raw.strip()
        if raw != raw.rstrip() or i == last:
            merged.append((pend_t, pend))
            pend, pend_t = '', None
    if pend:
        merged.append((pend_t, pend))
    return merged


# --------------------------------------------------------------------------
# Tier 2: estimate word onsets inside one line (only used when a line has no real timing)
# --------------------------------------------------------------------------
_CYR = re.compile(r'[Ѐ-ӿ]')
_HANG = re.compile(r'[\uac00-\ud7af]')          # Hangul syllables
_GREEK = re.compile(r'[\u0370-\u03ff\u1f00-\u1fff]')
_KANA = re.compile(r'[぀-ヿㇰ-ㇿ一-鿿]')
_SMALL_KANA = set('ゃゅょゎぁぃぅぇぉャュョヮァィゥェォ')


def syllables(token):
    """Best-effort syllable count, always >= 1."""
    w = token.strip().lower()
    if not w:
        return 1
    if _CYR.search(w):
        return max(1, sum(1 for c in w if c in 'аеёиоуыэюя'))
    if _HANG.search(w):
        return max(1, sum(1 for c in w if '\uac00' <= c <= '\ud7af'))
    if _KANA.search(w):
        core = re.sub(r'[\s\d\W_]', '', w)
        return max(1, sum(1 for c in core if c not in _SMALL_KANA))
    if _GREEK.search(w):
        return max(1, sum(1 for c in strip_marks(w) if c in 'αεηιουω'))
    w = re.sub(r"^[a-z\u00e0-\u00ff]{1,2}'", '', w)   # elision: c'\u00e9tait -> \u00e9tait, qu'il -> il
    w = re.sub(r"[^a-zà-ÿ]", '', w)
    if not w or len(w) <= 3:
        return 1
    v = 'aeiouyàâäéèêëîïôöùûü'
    stem = w
    if stem.endswith('es') and not stem.endswith(('ses', 'zes', 'ces', 'ges', 'xes', 'shes', 'ches')):
        stem = stem[:-2]
    elif stem.endswith('ed') and not stem.endswith(('ted', 'ded')):
        stem = stem[:-2]
    elif stem.endswith('e') and not stem.endswith(('le', 'ee', 'ie')):
        stem = stem[:-1]
    n = len(re.findall('[' + v + ']+', stem))
    if w.endswith('le') and len(w) > 2 and w[-3] not in v:
        n += 1
    return max(1, n)


FUNCTION_WORDS = frozenset("""
the a an to in on of for and but or at by from with as into like than after under
i you he she it we they me him her my your his our their
is am are was were be been do does did have has had will would can could should
that this these those
de la le les du des un une et ou mais ou bien donc or ni car
je tu il elle nous vous ils elles on me te se mon ma mes ton ta tes son sa ses
notre nos votre vos leur leurs ce cet cette ces
que qui quoi dont ou
a o os as um uma e de do da dos das em no na nos nas por para com sem sobre
el la los las un una y o pero si porque
der die das ein eine und oder aber denn weil mit von zu in an auf fur
i v na po k iz za dlya ot nad pod pri bez mezhdu cherez
""".split())


def _strip_punct(w):
    return w.strip(".,;:!?…—–-()[]\"'")
_CAESURA_PUNCT = set(",;:—–-…?!.…)]}")


def _alliterative(a, b):
    a = _strip_punct(a).lower()
    b = _strip_punct(b).lower()
    if not a or not b:
        return False
    if len(a) > 4 or len(b) > 4:
        return False
    return a[0] == b[0]


def estimate_line(words, line_start, line_end, pace=DEFAULT_PACE):
    """[(time, word), ...] for one line, plus a trailing (time, '') when a genuine rest follows.
    `pace` is the song's measured s/syllable cadence (see build_word_events).
    Guarantees: first word exactly on line_start; strictly increasing; every event strictly
    before line_end; word order/count preserved; MIN_SHOW floor whenever physically possible."""
    n = len(words)
    if n == 0:
        return []
    raw_window = line_end - line_start
    if raw_window <= 0.0:
        return [(line_start + i * 1e-6, w) for i, w in enumerate(words)]
    if raw_window < n * 0.004:
        step = raw_window * 0.9 / max(1, n - 1) if n > 1 else 0.0
        return [(line_start + i * step, w) for i, w in enumerate(words)]

    window = raw_window
    syl = [syllables(w) for w in words]
    total = sum(syl)

    # --- Prosodic model (v3.0) ---
    low = [_strip_punct(w).lower() for w in words]
    is_func = [lw in FUNCTION_WORDS for lw in low]

    # Caesura: punctuation boundaries.
    caesura_after = [False] * n
    for i, w in enumerate(words):
        s = w.rstrip()
        if s and s[-1] in _CAESURA_PUNCT:
            caesura_after[i] = True
        if i + 1 < n and words[i + 1].lstrip().startswith('('):
            caesura_after[i] = True

    # Isochronous repetition / stutter runs.
    run_flags = [False] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and (low[j] == low[j + 1] or _alliterative(words[j], words[j + 1])):
            j += 1
        if j > i:
            for k in range(i, j + 1):
                run_flags[k] = True
            i = j + 1
        else:
            i += 1

    # Anacrusis: 1-2 short function pickups before a heavy content word.
    anacrusis = False
    if n >= 2 and is_func[0] and syl[0] <= 2:
        heavy_idx = 1
        if n >= 3 and is_func[1] and not is_func[2]:
            heavy_idx = 2
        if heavy_idx < n and (syl[heavy_idx] >= 2 or phone_duration_scale(words[heavy_idx]) > 1.0):
            anacrusis = True

    func_mult = [1.0] * n
    for i in range(n):
        if is_func[i]:
            if i == n - 1 or caesura_after[i]:
                func_mult[i] = 1.0
            else:
                func_mult[i] = 0.56
    if anacrusis:
        for i in range(min(2, n)):
            if is_func[i]:
                func_mult[i] = min(func_mult[i], 0.50)

    caesura_mult = [1.0] * n
    for i in range(n):
        if caesura_after[i]:
            caesura_mult[i] = 1.25

    weights = []
    for i in range(n):
        base = (syl[i] ** TUNE['syl_exp']) if TUNE['syl_exp'] != 1.0 else float(syl[i])
        base = base * phone_duration_scale(words[i])
        base *= func_mult[i] * caesura_mult[i]
        if run_flags[i]:
            base = 1.0
        weights.append(base)

    if any(run_flags):
        front = 1.0
    else:
        front = 1.0 + min(TUNE['front_max'], max(0.0, (pace - TUNE['front_on']) * TUNE['front_gain']))

    wtotal = sum(weights) or 1.0

    # Caesura micro-gaps (additive, scaled by available slack later).
    caesura_gaps = [0.0] * (n - 1) if n > 1 else []
    for i in range(n - 1):
        if caesura_after[i]:
            caesura_gaps[i] = 0.20

    # onset span: exclude final held note
    natural_span = pace * total
    limit = max(0.0, window - TAIL_GUARD)
    last_word_min_dur = max(0.05, pace * syl[-1])
    limit_onset = max(0.0, limit - last_word_min_dur)
    vf = _voiced_fraction(raw_window)
    span = min(window * vf, natural_span, limit_onset)

    base_pos = [0.0] * n
    acc = 0.0
    for i in range(n):
        base_pos[i] = span * ((acc / wtotal) ** front)
        acc += weights[i]

    if n == 1:
        times = [line_start]
    else:
        gaps_n = n - 1
        floor = min(MIN_SHOW, limit_onset / gaps_n) if gaps_n > 0 else 0.0
        gaps = [base_pos[i + 1] - base_pos[i] for i in range(gaps_n)]
        for i in range(gaps_n):
            gaps[i] += caesura_gaps[i]
        budget = max(limit_onset, floor * gaps_n)
        need = sum(max(g, floor) for g in gaps)
        if need > budget + 1e-12:
            slack = [max(0.0, g - floor) for g in gaps]
            s_tot = sum(slack)
            if s_tot > 1e-12:
                cut = (need - budget) / s_tot
                gaps = [floor + sl * max(0.0, 1.0 - cut) for sl in slack]
        gaps = [max(g, floor) for g in gaps]
        tg = sum(gaps)
        if tg > limit_onset and tg > 0:
            k = limit_onset / tg
            gaps = [g * k for g in gaps]
        times = [line_start]
        for g in gaps:
            times.append(times[-1] + g)
        times[0] = line_start

    out = list(zip(times, words))

    last_word_articulation = pace * syl[-1]
    final_hold = min(TUNE['hold_cap'], TUNE['hold_a'] + TUNE['hold_b'] * pace)
    clear_at = times[-1] + last_word_articulation + final_hold
    if line_end - clear_at >= REST_CLEAR_MIN:
        out.append((clear_at, ''))
    return out


def normalise_line_starts(lines):
    """Repair impossible timing before estimation: lines closer than MIN_LINE_GAP are pushed
    forward (real per-word timings shift with their line). Inputs are not mutated."""
    out = []
    for src_line in sorted(lines, key=lambda x: x['start']):
        l = dict(src_line)
        if out and l['start'] < out[-1]['start'] + MIN_LINE_GAP:
            shift = (out[-1]['start'] + MIN_LINE_GAP) - l['start']
            l['start'] += shift
            if l.get('words'):
                l['words'] = [(t + shift, w) for t, w in l['words']]
        out.append(l)
    return out


def build_word_events(lines, total_duration, stretch=1.0, meta=None,
                      audio=None, align_cfg=None):
    """Flatten lines into a sorted [(time, word_or_'')] timeline. '' clears the display
    (intro, instrumental break, or a genuine rest after a line).

    stretch  scales ESTIMATED word onsets about their own line start (line anchors stay exact,
             real per-word timings are never touched). >1 spreads words later, <1 earlier.
    meta     optional dict; receives {'estimated_lines': int, 'real_lines': int} so the UI can
             tell the user whether the +/-/[/] controls will even do anything for this song.
    audio    optional WAV path, (sr, samples) tuple, or FeatureTrack. When supplied,
             the v2.2 offline engine refines onsets (VAD freeze, formant onsets,
             duration-aware search). Heuristic timeline is always built first.
    align_cfg optional AlignConfig (lookahead cap, calibration offset, freeze)."""
    if not lines:
        return []
    lines = normalise_line_starts(lines)
    n_est = n_real = 0
    if not total_duration or total_duration <= 0:
        total_duration = lines[-1]['start'] + 30.0

    # ---- v3.0: Dense-Frontier & Local-Cadence Estimator ----
    # Per-line raw windows / syllables.
    _line_info = []
    for i, _ln in enumerate(lines):
        _end = lines[i + 1]['start'] if i + 1 < len(lines) else _ln['start'] + 6.0
        _w = _end - _ln['start']
        _s = max(1, sum(syllables(t) for t in _ln['text'].split()))
        _line_info.append((_w, _s))

    # Filter micro-interjections (<3 syllables) when enough multi-word lines exist.
    _valid = [i for i, (_w, _s) in enumerate(_line_info) if 0.6 < _w < 12.0]
    _multi = [i for i in _valid if _line_info[i][1] >= 3]
    _use = _multi if len(_multi) >= max(3, len(_valid) // 4) else _valid

    _rates = []  # r_i * voiced_fraction
    for i in _use:
        _w, _s = _line_info[i]
        _rates.append((_w / max(1, _s)) * _voiced_fraction(_w))
    if _rates:
        _sr = sorted(_rates)
        # 25th-30th percentile dense frontier
        _idx = max(0, int(0.27 * (len(_sr) - 1)))
        _frontier = _sr[_idx]
        base_pace = min(TUNE['pace_max'], max(TUNE['pace_min'], _frontier))
    else:
        base_pace = DEFAULT_PACE
    pace = base_pace

    # Per-line local pace.
    line_paces = [base_pace] * len(lines)
    for i in range(len(lines)):
        _w_i, _s_i = _line_info[i]
        if i not in _valid:
            line_paces[i] = base_pace
            continue
        # 5-line rolling window
        _js = [j for j in range(max(0, i - 2), min(len(lines), i + 3)) if j in _valid]
        _local_rates = sorted((_line_info[j][0] / max(1, _line_info[j][1])) * _voiced_fraction(_line_info[j][0]) for j in _js)
        if _local_rates:
            _half = _local_rates[:max(1, len(_local_rates) // 2)]
            _lr = sorted(_half)[len(_half) // 2]
        else:
            _lr = base_pace
        # spike handling: do not inflate on single-line spikes
        _r_i = (_w_i / max(1, _s_i))
        if _r_i > 1.6 * base_pace:
            _lr = min(_lr, base_pace)
        _lp = 0.5 * base_pace + 0.5 * min(TUNE['pace_max'], max(TUNE['pace_min'], _lr))
        line_paces[i] = min(TUNE['pace_max'], max(TUNE['pace_min'], _lp))

    events = []
    # v3.1: ALWAYS emit a leading clear so the display state machine
    # enters PRELUDE before the first word can possibly show. Without
    # this, when lines[0].start is 0.0 the first real word pops on
    # before the singer sings it.
    if lines[0]['start'] > 0.5:
        events.append((0.0, ''))

    for i, line in enumerate(lines):
        line_end = lines[i + 1]['start'] if i + 1 < len(lines) else max(
            total_duration, line['start'] + MIN_LINE_GAP)
        real = line.get('words')
        if real:
            # Tier 1: trust the source. Keep each word strictly inside its own line window.
            ws_ = [(t, w) for t, w in real if line['start'] - 0.5 <= t < line_end]
            if ws_:
                n_real += 1
                events.extend(ws_)
                if line_end - ws_[-1][0] > 3.5:
                    events.append((ws_[-1][0] + 1.8, ''))
                continue
        words = line_words(line['text'])
        if words:
            n_est += 1
            est = estimate_line(words, line['start'], line_end, line_paces[i] if i < len(line_paces) else pace)
            if stretch != 1.0:
                ls, k = line['start'], stretch
                est = [(ls + (t - ls) * k, w) for t, w in est]
                # stretching must never push an event to/after the next line
                cap = line_end - 1e-3
                prev = ls - 1.0
                fixed = []
                for t, w in est:
                    t = min(t, cap)
                    if t <= prev:
                        t = prev + 1e-4
                    fixed.append((t, w))
                    prev = t
                est = fixed
            events.extend(est)

    events.sort(key=lambda e: e[0])
    cleaned = []
    for ev in events:
        if cleaned and cleaned[-1][1] == '' and ev[1] == '':
            continue
        cleaned.append(ev)
    if meta is not None:
        meta['estimated_lines'], meta['real_lines'] = n_est, n_real
        meta['aligned'] = False
    if audio is not None and cleaned and _HAS_ALIGN:
        try:
            cleaned = refine_word_events(cleaned, audio, align_cfg)
            if meta is not None:
                meta['aligned'] = True
        except Exception:
            pass
    return cleaned


# --------------------------------------------------------------------------
# Playback clock + player state
# --------------------------------------------------------------------------
class PlaybackClock:
    """Extrapolates song position between polls and corrects it on every fresh sample.

    * The player read its position somewhere inside the query, so the sample is stamped at the
      MIDPOINT of the round trip (stamping after the call made lyrics run early by the latency).
    * A jump larger than seek_tol, a play/pause change, or a rate change snaps the clock.
    * Smaller disagreements are slewed in gradually so words never jump backwards."""

    def __init__(self, seek_tol=0.35, slew=0.25):
        self.seek_tol, self.slew = seek_tol, slew
        self.anchor_pos = self.anchor_time = 0.0
        self.playing, self.rate, self._corr = False, 1.0, 0.0
        self.snapped = False

    def reset(self):
        self.anchor_pos = self.anchor_time = 0.0
        self.playing, self.rate, self._corr = False, 1.0, 0.0
        self.snapped = False

    def update(self, position, is_playing, t_before, t_after, rate=1.0):
        t_s = (t_before + t_after) / 2.0
        rate = rate if (rate and rate > 0) else 1.0
        self.snapped = False
        if not self.playing and not is_playing:
            self.anchor_pos, self.anchor_time, self._corr, self.rate = position, t_s, 0.0, rate
            return
        predicted = self.position_at(t_s)
        err = position - predicted
        if (is_playing != self.playing) or abs(err) > self.seek_tol or rate != self.rate:
            self.anchor_pos, self.anchor_time, self._corr = position, t_s, 0.0
            self.snapped = True
        else:
            self.anchor_pos, self.anchor_time, self._corr = predicted, t_s, err
        self.playing, self.rate = is_playing, rate

    def position_at(self, t):
        if not self.playing:
            return self.anchor_pos
        base = self.anchor_pos + (t - self.anchor_time) * self.rate
        if self._corr:
            base += self._corr * min(1.0, max(0.0, t - self.anchor_time) * self.slew * 4.0)
        return base


# ==========================================================================
# v3.1 — Display State Machine + Live Phase-Locking
# ==========================================================================
DEAD_ZONE_S = 0.080          # never show a word within 80 ms of a phantom
INTRINSIC_OFFSET_MAX = 2.0   # clamp self-tuned per-song offset to ±2 s
RESIDUAL_BLEND = 0.5         # how much to trust a single play's residual
INTRINSIC_PERSIST_S = 30.0   # seconds of fresh audio needed to lock an
                             # intrinsic offset automatically

# Display states
DS_LOADING = 'LOADING'
DS_PRELUDE = 'PRELUDE'
DS_SINGING = 'SINGING'
DS_INTERLUDE = 'INTERLUDE'
DS_ENDING = 'ENDING'


class DisplayStateMachine:
    """Explicit state machine for the lyric display. Kills the
    "phantom first word" and "instrument plays for 4 s after the last
    word" bugs by making every transition a deterministic function of
    (pos, events, audio_state).

    Inputs (per tick):
      pos              current song position in seconds (0 at song start)
      events           sorted [(t, word_or_'')] timeline
      audio_state      dict from PhaseLocker.snapshot() or None
      last_word_end    time of the last word's display (or -inf)
      last_clear_end   time of the last clear/rest marker (or -inf)

    Outputs (per tick):
      state            one of DS_*
      show_word        str|None         the word to render, or None
      show_dots        bool             render pulsing dots
      show_message     str|None         "loading" / "prelude" text
      is_word_change   bool             index changed since last tick
    """

    def __init__(self):
        self.state = DS_LOADING
        self._last_idx = -1
        self._last_word = None
        self._state_entered_at = 0.0
        self._prelude_dwell = 0.0  # seconds spent in PRELUDE before a word

    def reset(self):
        self.state = DS_LOADING
        self._last_idx = -1
        self._last_word = None
        self._state_entered_at = 0.0
        self._prelude_dwell = 0.0

    def _enter(self, new_state, now):
        if new_state != self.state:
            self.state = new_state
            self._state_entered_at = now

    def step(self, pos, events, audio_state=None, now=0.0):
        """Advance the state machine by one tick. Returns a render dict."""
        starts = [t for t, _ in events]
        first_word_t = next((t for t, _ in events if _), None)
        last_event_t = events[-1][0] if events else 0.0

        # No events: stay in LOADING (or ENDING if past the would-be end).
        if not events:
            self._enter(DS_LOADING, now)
            return {'state': self.state, 'show_word': None,
                    'show_dots': False, 'show_message': 'loading…',
                    'is_word_change': False}

        # Find the idx of the most recent non-empty word that is due.
        # We use bisect_right on starts of *non-empty* words only — clears
        # are render hints, not timeline anchors.
        non_empty_starts = [t for t, w in events if w]
        if non_empty_starts:
            import bisect as _bisect
            n_idx = _bisect.bisect_right(non_empty_starts, pos) - 1
        else:
            n_idx = -1

        # ---- LOADING -> PRELUDE once events are loaded and the player
        #      is in any real position OR a beat has been seen.
        if self.state == DS_LOADING:
            if pos > 0.0 or first_word_t is not None:
                self._enter(DS_PRELUDE, now)

        # ---- PRELUDE: before the first non-empty word. The dead-zone rule
        #      keeps us here until pos >= first_word_t - DEAD_ZONE, OR audio
        #      confirms a vocal onset at the first word.
        if self.state == DS_PRELUDE:
            self._prelude_dwell = now - self._state_entered_at
            if first_word_t is not None:
                audio_vocal = False
                if audio_state and audio_state.get('vad_rising', False) \
                        and audio_state.get('vocal_at_first_word', False):
                    audio_vocal = True
                if pos >= first_word_t - DEAD_ZONE_S or audio_vocal:
                    self._enter(DS_SINGING, now)
                elif pos >= last_event_t:
                    # Past the last event while still in PRELUDE means
                    # the song has no words for the player — go to ENDING.
                    self._enter(DS_ENDING, now)
            else:
                # No non-empty events at all; treat as interlude forever.
                self._enter(DS_INTERLUDE, now)

        # ---- SINGING: a non-empty word is active (or about to be).
        if self.state == DS_SINGING:
            if n_idx < 0:
                # No word has been reached yet; back to PRELUDE.
                self._enter(DS_PRELUDE, now)
            else:
                current_word_t, current_word = events[
                    [i for i, (_, w) in enumerate(events) if w][n_idx]
                ] if n_idx >= 0 else (0.0, '')
                # The clear of the *current* line: walk forward to the
                # next '' in events after the active word.
                active_idx = [i for i, (_, w) in enumerate(events) if w][n_idx]
                clear_t = None
                for j in range(active_idx + 1, len(events)):
                    if events[j][1] == '':
                        clear_t = events[j][0]
                        break
                if clear_t is None:
                    clear_t = last_event_t + 0.5

                # Audio-driven early clear: if VAD has been silent for
                # >350 ms in the centre channel AND we are at or past
                # the active word's time, drop to INTERLUDE right now.
                audio_freeze = False
                if audio_state and audio_state.get('silence_ms', 0) > 350 \
                        and pos >= current_word_t - 0.05:
                    audio_freeze = True

                if audio_freeze or pos >= clear_t:
                    self._enter(DS_INTERLUDE, now)

        # ---- INTERLUDE: between lines. Stay until the *next* non-empty
        #      word is within 350 ms (or its time has come). When audio
        #      is available, prefer a vocal-rise trigger over the
        #      timeline alone so we don't pop back into SINGING just
        #      because the LRC says the next word is close.
        if self.state == DS_INTERLUDE:
            upcoming = next((t for t, w in events
                             if w and t > pos), None)
            if upcoming is None:
                if pos >= last_event_t:
                    self._enter(DS_ENDING, now)
            else:
                audio_vocal = False
                if audio_state and audio_state.get('vad_rising', False) \
                        and audio_state.get('vocal_at_next_word', False):
                    audio_vocal = True
                if audio_vocal or pos >= upcoming:
                    # audio-driven: jump in on the vocal rise.
                    # timeline-driven: only after the *exact* time of
                    # the next word (not 350 ms early). This stops
                    # audio-freeze INTERLUDEs from being immediately
                    # undone by the LRC's "next line is 200 ms away".
                    self._enter(DS_SINGING, now)
                elif pos >= last_event_t:
                    self._enter(DS_ENDING, now)

        # ---- ENDING: past the last event. Stay until song change.
        #      Nothing to do — main loop handles song change.

        # ---- Render decision ----
        show_word = None
        show_dots = False
        show_message = None
        is_word_change = False

        if self.state in (DS_LOADING, DS_PRELUDE):
            show_dots = True
            show_message = 'prelude'
        elif self.state == DS_SINGING and n_idx >= 0:
            non_empty_indices = [i for i, (_, w) in enumerate(events) if w]
            if n_idx < len(non_empty_indices):
                ev_i = non_empty_indices[n_idx]
                t_w, w_w = events[ev_i]
                if pos + 1e-9 >= t_w - DEAD_ZONE_S:
                    show_word = w_w
                    if ev_i != self._last_idx:
                        is_word_change = True
                        self._last_idx = ev_i
                        self._last_word = w_w
        elif self.state == DS_INTERLUDE:
            show_dots = True
        elif self.state == DS_ENDING:
            show_dots = True
            show_message = 'ending'

        return {'state': self.state, 'show_word': show_word,
                'show_dots': show_dots, 'show_message': show_message,
                'is_word_change': is_word_change}


# ---------------------------------------------------------------------------
# v3.1 — Built-events cache (skip the HTTP fetch on repeat plays)
# ---------------------------------------------------------------------------
def _events_cache_dir():
    p = os.path.expanduser('~/.cache/word-sync/events')
    os.makedirs(p, exist_ok=True)
    return p


def _events_cache_path(artist, title, duration):
    import hashlib
    raw = f'{sanitize(artist)}|{sanitize(title)}|{int(round(duration or 0))}'
    h = hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]
    return os.path.join(_events_cache_dir(), f'{h}.json')


def load_event_cache(artist, title, duration, max_age_days=30):
    """Return cached events list [(t, word), ...] or None."""
    p = _events_cache_path(artist, title, duration)
    if not os.path.exists(p):
        return None
    try:
        st = os.stat(p)
    except OSError:
        return None
    if (time.time() - st.st_mtime) > max_age_days * 86400:
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or 'events' not in data:
        return None
    evs = data.get('events')
    if not isinstance(evs, list):
        return None
    out = []
    for item in evs:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            return None
        t, w = item
        try:
            t = float(t)
        except (TypeError, ValueError):
            return None
        out.append((t, str(w) if w is not None else ''))
    return out


def save_event_cache(artist, title, duration, events, intrinsic_offset=0.0):
    """Persist built events + learned intrinsic_offset for instant reload."""
    p = _events_cache_path(artist, title, duration)
    try:
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'events': [(t, w) for t, w in events],
                       'intrinsic_offset': float(intrinsic_offset or 0.0),
                       'version': '3.1'}, f)
    except OSError:
        return False
    return True


def cached_intrinsic_offset(artist, title, duration):
    p = _events_cache_path(artist, title, duration)
    if not os.path.exists(p):
        return None
    try:
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(data, dict):
        v = data.get('intrinsic_offset')
        if isinstance(v, (int, float)):
            return float(v)
    return None


# ---------------------------------------------------------------------------
# v3.1 — LRC candidate auto-picker
# ---------------------------------------------------------------------------
def pick_candidate(cands, first_obs_position=0.0, line_text_at_zero=None):
    """Score each candidate and return the best index.

    Scoring:
      + line[0].start closeness to first_obs_position (clamped 0..5 s)
      + 0.1 per parsed line (more data = better)
      + huge bonus for having word-level timings
      - small penalty if the first word is a common article
    """
    if not cands:
        return 0
    best_idx = 0
    best_score = -1e9
    articles = {'the', 'a', 'an', 'i', 'je', 'le', 'la', 'el', 'un', 'une',
                'der', 'die', 'das', 'ich', 'tu', 'oi', 'mi', 'ja', 'ne'}
    for i, c in enumerate(cands):
        score = 0.0
        synced = c.get('synced') or ''
        lf = c.get('lyricsfile')
        if lf:
            score += 50.0
        lines = []
        if lf:
            try:
                lines = parse_lyricsfile(lf)
            except Exception:
                lines = []
        if not lines and synced:
            try:
                lines = parse_lrc(synced)
            except Exception:
                lines = []
        if not lines:
            continue
        score += min(len(lines), 200) * 0.1
        first_t = lines[0].get('start', 0.0) or 0.0
        # If we know where the player actually is, prefer the candidate
        # whose first line is closest to that position. Clamp to [0, 5].
        diff = abs(first_t - first_obs_position)
        diff = min(diff, 5.0)
        score += (5.0 - diff) * 2.0
        # Slight penalty for first word being a common article
        first_text = (lines[0].get('text') or '').strip().split()
        if first_text and first_text[0].lower().rstrip('.,!?') in articles:
            score -= 0.4
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# v3.1 — Live phase-locker
# ---------------------------------------------------------------------------
class PhaseLocker:
    """Real-time phase-locking to live vocal onsets.

    Sits between the audio tracker and the display loop. On every tick
    the main loop calls `consult(t_predicted, pos, last_vocal_t)`, which
    returns either:
      ('snap', t_onset)        — a strong vocal onset was near t_predicted;
                                 snap the next word to t_onset.
      ('hold', t_predicted)    — keep waiting; no vocal yet.
      ('release', t_predicted + 0.35) — patience exceeded, show it now.

    Also watches centre-channel VAD for the post-word freeze rule.
    """

    SNAP_RADIUS_S = 0.160          # ±160 ms: a vocal onset inside this
                                   # window can snap the next word.
    PATIENCE_S = 0.350             # if no onset within this, release
    FREEZE_SILENCE_MS = 380        # ms of centre-channel silence to freeze
    FREEZE_MIN_AFTER_WORD_S = 0.10 # don't freeze too soon after a word

    def __init__(self, audio_tracker=None):
        self.audio = audio_tracker  # LiveAudioTracker or None
        self._silence_start = None
        self._last_word_t = 0.0

    def available(self):
        return self.audio is not None and self.audio.available

    def notify_word_displayed(self, t):
        self._last_word_t = t
        self._silence_start = None

    def snapshot(self, pos):
        """Read the current audio state and return a small dict used by
        the state machine. Never raises."""
        out = {'vad_rising': False,
               'silence_ms': 0,
               'vocal_at_first_word': False,
               'vocal_at_next_word': False}
        if not self.available():
            return out
        try:
            track = self.audio.track
            if track is None or not track.frames:
                return out
            # Find frames near pos
            near = [f for f in track.frames if abs(f.t - pos) < 0.6]
            if not near:
                return out
            # Centre-band VAD: formants on mid
            vads = [f.vad for f in near]
            rising = (sum(vads) >= max(2, len(vads) // 2))
            out['vad_rising'] = rising
            # Silence duration: how long since the last vad=True frame
            last_vad_t = None
            for f in sorted(track.frames, key=lambda f: f.t):
                if f.t > pos:
                    break
                if f.vad:
                    last_vad_t = f.t
            if last_vad_t is None:
                out['silence_ms'] = int((pos) * 1000)
            else:
                out['silence_ms'] = int((pos - last_vad_t) * 1000)
            out['vocal_at_next_word'] = rising
            out['vocal_at_first_word'] = rising
            return out
        except Exception:
            return out

    def consult(self, t_predicted, pos, lookahead_s=0.6):
        """Decide what to do with the next word at t_predicted.

        Returns ('snap', t_onset) | ('hold', t_predicted) |
                ('release', t_predicted + PATIENCE_S).
        """
        if not self.available() or t_predicted is None:
            return ('release', (t_predicted or pos) + self.PATIENCE_S)
        try:
            track = self.audio.track
            if track is None or not track.frames:
                return ('release', t_predicted + self.PATIENCE_S)
            # Window: [t_predicted - SNAP_RADIUS, t_predicted + 0.18]
            lo = t_predicted - self.SNAP_RADIUS_S
            hi = t_predicted + 0.18
            best = None
            for f in track.frames:
                if f.t < lo:
                    continue
                if f.t > hi:
                    break
                if f.onset > 0.35 and (best is None or f.onset > best.onset):
                    best = f
            # Snap if we found a strong onset, even slightly before pos
            # (the audio can lead the timeline by up to SNAP_RADIUS_S).
            if best is not None and best.t >= pos - self.SNAP_RADIUS_S:
                return ('snap', best.t)
            if pos >= t_predicted + self.PATIENCE_S:
                return ('release', t_predicted + self.PATIENCE_S)
            return ('hold', t_predicted)
        except Exception:
            return ('release', t_predicted + self.PATIENCE_S)


# ---------------------------------------------------------------------------
# v3.1 — Per-song residual log + intrinsic offset self-tuning
# ---------------------------------------------------------------------------
class SongResiduals:
    """Logs (t_predicted, t_displayed, audio_vocal_t?) and updates a
    per-song intrinsic offset estimate on song change."""

    def __init__(self, max_per_song=64):
        self._by_song = {}
        self._max = max_per_song

    def log(self, song_key, t_predicted, t_displayed, audio_vocal_t=None):
        if not song_key:
            return
        bucket = self._by_song.setdefault(song_key, [])
        bucket.append((float(t_predicted or 0.0),
                       float(t_displayed or 0.0),
                       float(audio_vocal_t) if audio_vocal_t is not None else None))
        if len(bucket) > self._max:
            del bucket[:len(bucket) - self._max]

    def estimate_offset(self, song_key):
        """Median residual = (audio_vocal_t - t_predicted) for this song.
        Returns 0.0 if no data."""
        bucket = self._by_song.get(song_key) or []
        rs = []
        for t_p, _t_d, t_a in bucket:
            if t_a is not None:
                r = t_a - t_p
                if abs(r) <= INTRINSIC_OFFSET_MAX:
                    rs.append(r)
        if not rs:
            return 0.0
        rs.sort()
        return rs[len(rs) // 2]

    def apply_to(self, song_key, current_offset):
        """Blend the residual into the offset (clamped) for next play."""
        est = self.estimate_offset(song_key)
        if est == 0.0:
            return current_offset
        # Clamp to safe range
        est = max(-INTRINSIC_OFFSET_MAX, min(INTRINSIC_OFFSET_MAX, est))
        return current_offset + RESIDUAL_BLEND * est

def get_player_state(player=None):
    """One playerctl round trip. Returns a dict (with 't_before'/'t_after' monotonic stamps
    around the call, for unbiased clock sampling) or None."""
    cmd = ['playerctl']
    if player:
        cmd += ['--player', player]
    cmd += ['metadata', '--format',
            'STATUS={{status}}||ARTIST={{artist}}||TITLE={{title}}||POS={{position}}'
            '||LEN={{mpris:length}}||RATE={{playerctl:rate}}||TID={{mpris:trackid}}']
    t_before = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)
    except Exception:
        return None
    t_after = time.monotonic()
    if p.returncode != 0:
        return None
    line = p.stdout.strip()
    if not line:
        return None
    fields = {}
    for part in line.split('||'):
        if '=' in part:
            k, v = part.split('=', 1)
            fields[k] = v

    def _f(key, scale=1.0, default=0.0):
        try:
            return float(fields.get(key, default)) / scale
        except (TypeError, ValueError):
            return default

    artist = fields.get('ARTIST', '').strip()
    title = fields.get('TITLE', '').strip()
    return {
        'status': fields.get('STATUS', 'Stopped'),
        'artist': artist,
        'title': title,
        'position': _f('POS', 1_000_000.0),
        'duration': _f('LEN', 1_000_000.0),
        'rate': _f('RATE', 1.0, 1.0) or 1.0,
        'track_id': fields.get('TID', ''),
        'song_key': f'{artist}---{title}',
        't_before': t_before,
        't_after': t_after,
    }


# --------------------------------------------------------------------------
# Per-song remembered offset (kept out of config.toml so the settings GUI can't drop it)
# --------------------------------------------------------------------------
def _offset_store_path():
    return os.path.expanduser('~/.config/word_sync/song_offsets.json')


def load_song_offsets():
    """{song_key: {'offset': s, 'stretch': k}}. Also accepts the older {song_key: float}."""
    try:
        with open(_offset_store_path(), 'r', encoding='utf-8') as f:
            d = json.load(f)
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    clean = {}
    for k, v in d.items():
        try:
            if isinstance(v, dict):
                off = float(v.get('offset', 0.0))
                stc = float(v.get('stretch', 1.0))
            else:
                off, stc = float(v), 1.0
        except (TypeError, ValueError):
            continue
        if not (-30.0 <= off <= 30.0) or not (0.4 <= stc <= 2.0):
            continue
        clean[k] = {'offset': off, 'stretch': stc}
    return clean


def save_song_offsets(d):
    try:
        p = _offset_store_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=1, sort_keys=True)
        os.replace(tmp, p)
    except Exception:
        pass


def intensity_for(playing, fade_p, paused_dim=0.45):
    if not playing:
        return paused_dim
    return 0.5 + 0.5 * max(0.0, min(1.0, fade_p))


def render_message(msg):
    cols, rows = shutil.get_terminal_size((80, 24))
    top = max(rows // 2 - 1, 0)
    left = ' ' * max((cols - len(msg)) // 2, 0)
    buf = CLEAR + ('\n' * top) + left + DIM + msg + RESET
    sys.stdout.write(buf)
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Compositing Renderer: Zero-flicker, Glow Halo, Drop Shadow, Transparent BG
# --------------------------------------------------------------------------
_PAINT_GEO = {}   # cached glyph geometry per (rows, size, effects) — see paint()
_GLYPH_CACHE = {} # cached rendered rows per (word, font, opts, size) — see draw_block()
def render_interlude_dots(cols, rows, phase, dot_rgb=(120, 180, 255)):
    # Aesthetic glowing pulsing dots during instrumental breaks / solos
    dots_cycle = ['·   ·   ·', '•   •   •', '●   ●   ●', '•   •   •']
    dot_str = dots_cycle[phase % len(dots_cycle)]
    top = max(rows // 2 - 1, 0)
    left = max((cols - len(dot_str)) // 2, 0)
    dot_ansi = f'\033[1;{_fg(*dot_rgb)}m{dot_str}\033[0m'

    # Clean double-buffer overwrite of ALL rows to completely erase the previous word!
    out = []
    blank_row = ' ' * cols
    for r in range(rows):
        if r == top:
            row_content = (' ' * left) + dot_ansi + (' ' * max(0, cols - left - len(dot_str)))
            out.append(row_content)
        else:
            out.append(blank_row)

    sys.stdout.write(HOME + '\n'.join(out))
    sys.stdout.flush()


def _row_color(r, total_h, rgb, intensity, mode):
    if mode == 'rainbow':
        h = (r / max(total_h, 1)) % 1.0
        cr, cg, cb = hsv_to_rgb(h, 0.85, 1.0)
        return color_code((cr, cg, cb), intensity)
    if mode == 'gradient':
        # Sunset / neon gradient: top bright pink -> bottom electric gold
        factor = r / max(total_h - 1, 1)
        r_c = int(rgb[0] * (1.0 - factor * 0.3))
        g_c = int(rgb[1] * (0.4 + factor * 0.6))
        b_c = int(rgb[2] * (1.0 - factor * 0.7))
        return color_code((r_c, g_c, b_c), intensity)
    return color_code(rgb, intensity)


def paint(text_rows, opts, rgb, intensity, cols, rows):
    h = len(text_rows)
    width = max((len(x) for x in text_rows), default=0)
    top = max((rows - h) // 2, 0) + int(opts.get('v_offset', 0))
    left = max((cols - width) // 2, 0)
    mode = opts.get('color_mode', 'solid')
    bold = BOLD if opts.get('bold', True) else ''
    shadow = bool(opts.get('shadow', False))
    glow = bool(opts.get('glow', False))
    glow_col_val = opts.get('glow_color', '#3b1261')
    try:
        glow_rgb = resolve_color(glow_col_val)
    except Exception:
        glow_rgb = (59, 18, 97)

    bg_raw = opts.get('bg_rgb')

    # Grid of ANSI-formatted cell strings
    grid = [[None] * cols for _ in range(rows)]

    def put(r, c, content):
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = content

    # Geometry (core/halo/shadow cell sets + centering) depends only on the glyph
    # rows, terminal size and effects — NOT on intensity. Cache it so the fade
    # transition only re-applies colors instead of re-scanning the word grid.
    geo_key = (tuple(text_rows), cols, rows, glow, shadow, int(opts.get('v_offset', 0)))
    geo = _PAINT_GEO.get(geo_key)
    if geo is None:
        # 1. Identify non-space core glyph points
        core_points = set()
        for r in range(h):
            for c in range(len(text_rows[r])):
                if text_rows[r][c] != ' ':
                    core_points.add((r, c))
        # 2. Ambient outer halo points (keeps rest of terminal transparent!)
        halo_points = set()
        if glow:
            for r, c in core_points:
                for dr, dc in NEIGHBORS:
                    p = (r + dr, c + dc)
                    if p not in core_points:
                        halo_points.add(p)
        # 3. Drop shadow cells (offset +1 row, +1 col)
        shadow_cells = []
        if shadow:
            for r in range(h):
                for c in range(len(text_rows[r])):
                    ch = text_rows[r][c]
                    if ch != ' ':
                        shadow_cells.append((r, c, ch))
        if len(_PAINT_GEO) > 96:
            _PAINT_GEO.clear()
        geo = (top, left, core_points, halo_points, shadow_cells)
        _PAINT_GEO[geo_key] = geo
    top, left, core_points, halo_points, shadow_cells = geo

    # 2. Ambient outer halo/glow around glyphs
    if glow:
        glow_str = color_code(glow_rgb, max(0.1, intensity * 0.65)) + '░' + RESET
        for r, c in halo_points:
            put(top + r, left + c, glow_str)

    # 3. Drop shadow layer
    if shadow:
        scol = color_code(glow_rgb if glow else (25, 25, 30), max(0.15, intensity * 0.4))
        for r, c, ch in shadow_cells:
            put(top + 1 + r, left + 1 + c, scol + ch + RESET)

    # 4. Core illuminated glyphs
    for r in range(h):
        col = _row_color(r, h, rgb, intensity, mode)
        for c, ch in enumerate(text_rows[r]):
            if ch != ' ':
                put(top + r, left + c, bold + col + ch + RESET)

    # 5. Double-buffered single-flush output
    bgcode = f'\033[{_bg(*bg_raw)}m' if bg_raw is not None else ''
    out = []
    for r in range(rows):
        line = ''.join(grid[r][c] if grid[r][c] is not None else ' ' for c in range(cols))
        if bgcode:
            out.append(bgcode + line + RESET)
        else:
            out.append(line)

    # Use HOME instead of clearing screen to eliminate any transition flicker
    sys.stdout.write(HOME + '\n'.join(out))
    sys.stdout.flush()


def draw_block(word, rgb, intensity, opts, dot_phase=0):
    cols, rows = shutil.get_terminal_size((80, 24))
    if not word:
        # Dim aesthetic pulsing dots matching theme primary color
        pulse_rgb = (int(rgb[0] * 0.75), int(rgb[1] * 0.75), int(rgb[2] * 0.75))
        render_interlude_dots(cols, rows, dot_phase, pulse_rgb)
        return

    word = apply_case(word, opts.get('case', 'upper'))
    cols, rows = shutil.get_terminal_size((80, 24))
    fname = opts.get('font') or opts.get('font_name', 'double')
    hollow = False

    if fname.startswith('figlet:'):
        flines = render_figlet(word, fname.split(':', 1)[1], cols, rows)
        if flines:
            paint(flines, opts, rgb, intensity, cols, rows)
            return
        font = FONT_5
        outline = bool(opts.get('outline', False))
    else:
        font = FONTS.get(fname, FONT_DOUBLE)
        outline = bool(opts.get('outline', False))

    gap_str = ' ' * max(0, min(6, int(opts.get('gap', 2))))
    fill = opts.get('fill', DEFAULT_FILL)
    case = opts.get('case', 'upper')
    rendered = None

    # Check if raw art font (e.g. double, mini, slant, smooth, thin)
    first_glyph = next(iter(font.values()))
    is_raw_art = isinstance(first_glyph[0], str) and any(c not in '012' for c in first_glyph[0])

    # v1.4: if most chars have no glyph anywhere in the fallback chain (Chinese,
    # Arabic, …) show styled plain text instead of a row of blank blocks.
    n_ch = sum(1 for ch in word if not ch.isspace())
    unrenderable = n_ch > 0 and sum(
        1 for ch in word if not ch.isspace() and _char_resolves(font, ch)) < 0.5 * n_ch

    gkey = (word, fname, fill, gap_str, outline, case, hollow,
            int(opts.get('scale', 0)), cols, rows)
    rendered = None if unrenderable else _GLYPH_CACHE.get(gkey)
    if rendered is None and not unrenderable:
        req_scale = int(opts.get('scale', 0))
        if req_scale > 0:
            scales_to_try = [req_scale]
        else:
            max_s = 1 if is_raw_art else 3
            scales_to_try = list(range(max_s, 0, -1))

        # Auto-fit scale down if word is wide
        for s in scales_to_try:
            r = generate_glyph_rows(word, font, fill, gap_str, s, outline, case, hollow)
            h = len(r)
            w = max((len(x) for x in r), default=0)
            if w <= cols and h + 2 <= rows:
                rendered = r
                break

        # Fallback to compact or mini font if still too big
        if rendered is None:
            fallback_font = FONT_MINI if is_raw_art else FONT_COMPACT
            r = generate_glyph_rows(word, fallback_font, fill, gap_str, 1, outline, case, hollow)
            h = len(r)
            w = max((len(x) for x in r), default=0)
            if w <= cols and h + 2 <= rows:
                rendered = r
        if rendered is not None:
            if len(_GLYPH_CACHE) > 96:
                _GLYPH_CACHE.clear()
            _GLYPH_CACHE[gkey] = rendered

    # Fallback to plain text if terminal is tiny
    if rendered is None:
        style = (BOLD if opts.get('bold', True) else '') + color_code(rgb, intensity)
        top = max(rows // 2 - 1, 0) + int(opts.get('v_offset', 0))
        left = ' ' * max((cols - len(word)) // 2, 0)
        sys.stdout.write(HOME + '\n' * top + left + style + word + RESET + '\n' * (rows - top - 1))
        sys.stdout.flush()
        return

    paint(rendered, opts, rgb, intensity, cols, rows)


# --------------------------------------------------------------------------
# Figlet Font Support (Optional)
# --------------------------------------------------------------------------
def figlet_font_dir():
    for d in ('/usr/share/figlet', '/usr/share/figlet/fonts',
              '/usr/local/share/figlet', '/usr/share/toilet'):
        if os.path.isdir(d):
            return d
    return None


def figlet_fonts():
    d = figlet_font_dir()
    if not d:
        return []
    names = []
    for p in glob.glob(os.path.join(d, '*.flf')):
        stem = os.path.basename(p)[:-4]
        if stem:
            names.append(stem)
    return sorted(names)


def render_figlet(word, font, cols, rows):
    cmd = ['figlet', '-w', str(cols), '-f', font, '--', word]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5)
        if p.returncode == 0:
            lines = [l for l in p.stdout.splitlines() if l.strip()]
            if lines:
                return lines
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------
# Configuration Management
# --------------------------------------------------------------------------
def load_config(path):
    cfg = {}
    if not os.path.exists(path):
        return cfg
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip().replace('-', '_')
                v = v.strip().strip('"').strip("'")
                cfg[k] = v
    except Exception:
        pass
    return cfg


def _as_bool(v, default):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


def _as_float(v, default):
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def write_config(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [f'# word-sync configuration (v{VERSION})', '']
    order = ('font', 'scale', 'fill', 'gap', 'color', 'color_mode', 'bg',
             'outline', 'bold', 'glow', 'shadow', 'glow_color', 'case',
             'v_offset', 'transition', 'fade', 'paused_dim',
             'player', 'plain', 'resync', 'tick', 'lyrics_dir', 'cache_dir')
    for k in order:
        if k in data and data[k] is not None:
            v = data[k]
            if isinstance(v, bool):
                lines.append(f'{k} = {str(v).lower()}')
            elif isinstance(v, (int, float)):
                lines.append(f'{k} = {v}')
            else:
                lines.append(f'{k} = "{v}"')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------
# Offline regression suite (--selftest)
# --------------------------------------------------------------------------
def run_selftest():
    """Quick offline checks: every font inks every supported script with even
    rows; the LRC parser honours offsets/word-tags/NFC; syllables are sane; the
    timing engine stays monotonic, clears verse ends and adapts to tempo.
    Returns 0 on success."""
    fails = []

    def check(cond, label):
        if not cond:
            fails.append(label)

    samples = {
        'latin': 'NEON WORDS 123',
        'cyrillic': 'ПРИВЕТ СЛОВА ЖИЗНЬ',
        'accents': 'DÎNER ÊTÉ ÇA M’ATTENDRE…',
        'hiragana': 'きらそめたむすびね',
        'katakana': 'カタカナノテスト ヴぱ',
    }
    for fname, font in FONTS.items():
        for label, w in samples.items():
            rows = generate_glyph_rows(w, font, '█', ' ', 1, False, 'asis', False)
            ink = sum(1 for r in rows for c in r if c != ' ')
            check(ink > 0, f'{fname}/{label} rendered blank')
            check(len({len(r) for r in rows}) == 1, f'{fname}/{label} ragged rows')

    lrc = ('[offset:-200]\n[00:10.00]he' + chr(0x301) + 'llo wo' + chr(0x308) + 'rld\n'
           '[00:12.00]<00:12.00>un <00:12.5>deux\n')
    lines = parse_lrc(lrc)
    check(len(lines) == 2 and abs(lines[0]['start'] - 10.2) < 1e-6, 'lrc offset tag')
    check(lines[0]['text'] == 'h' + chr(0xE9) + 'llo w' + chr(0xF6) + 'rld',
          'lrc NFC normalisation')
    check(bool(lines[1]['words']) and len(lines[1]['words']) == 2, 'lrc word tags')

    check(syllables("c'était") == 2, 'syllable elision')
    check(syllables('любовь') >= 2, 'syllable cyrillic')
    check(syllables('사랑해') == 3, 'syllable hangul')

    pop = [
        {'start': 0.0, 'text': 'comme cet homme est venu me voir', 'words': None},
        {'start': 4.0, 'text': 'pour parler de mon amoureux', 'words': None},
        {'start': 8.0, 'text': 'je savais que c’était lui que', 'words': None},
        {'start': 28.0, 'text': 'attendais depuis toujours', 'words': None},
    ]
    evs = build_word_events(pop, 60.0)
    ts = [t for t, _ in evs]
    check(ts == sorted(ts), 'events monotonic')
    last_onset = max((t for t, w in evs if w and t < 20), default=0.0)
    clears = [t for t, w in evs if not w and 8.0 < t < 28.0]
    check(bool(clears) and clears[0] < 20.0, 'verse-end rest cleared')
    check(last_onset - 8.0 < 8.0, f'verse-end span too long ({last_onset - 8.0:.1f}s)')

    def first_line_span(win):
        ls = [{'start': i * win, 'text': 'je t’aime encore ce soir', 'words': None}
              for i in range(6)]
        e = build_word_events(ls, win * 8)
        on = [t for t, w in e if w and t < win]
        return (on[-1] - on[0]) if on else 0.0
    check(first_line_span(7.0) > first_line_span(3.0), 'tempo-adaptive cadence')
    # v2 trained bound: a 7 s ballad window is ~80 % voiced in the corpus, but the
    # phrase may never exceed ~6.2 s of onsets (no crawling into the rest)
    check(first_line_span(7.0) < 6.2, 'ballad words stay with the vocal (no late drift)')

    # v2.2: vowel-heavy first words keep a longer intra-line share than
    # plosive-heavy ones (second-onset delay). Line *span* is unchanged.
    love = estimate_line(['love', 'x'], 0.0, 4.0, pace=0.55)
    stop = estimate_line(['strict', 'x'], 0.0, 4.0, pace=0.55)
    love_on = [t for t, w in love if w]
    stop_on = [t for t, w in stop if w]
    check(len(love_on) >= 2 and len(stop_on) >= 2
          and love_on[1] >= stop_on[1] - 1e-6,
          'vowel duration share (love vs strict)')

    for label in engine_unit_checks():
        fails.append('align/' + label)
    check(_HAS_ALIGN, 'alignment engine present')

    if fails:
        sys.stderr.write('selftest FAILED:\n  ' + '\n  '.join(fails) + '\n')
        return 1
    sys.stdout.write(f'selftest OK — {len(FONTS)} fonts x {len(samples)} scripts, '
                     'parser, syllables, timing and align engine all pass\n')
    return 0


# --------------------------------------------------------------------------
# CLI Parser
# --------------------------------------------------------------------------
def build_parser():
    ap = argparse.ArgumentParser(
        description='Word-by-word lyric display for MPRIS players (Spotify, CachyOS, Arch).',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', action='version', version=f'word-sync {VERSION}')
    ap.add_argument('--selftest', action='store_true',
                    help='run the offline regression suite and exit')
    ap.add_argument('--setup', action='store_true', help='open the settings GUI')
    ap.add_argument('--theme', default=None, choices=list(THEMES),
                    help='apply a built-in theme: ' + ', '.join(THEMES))
    ap.add_argument('--plain', action='store_true', help='pipe mode (bare words)')
    ap.add_argument('--color', default=None, help='color name or hex')
    ap.add_argument('--color-mode', default=None, choices=COLOR_MODES)
    ap.add_argument('--bg', default=None, help="background color or 'none'")
    ap.add_argument('--font', default=None,
                    help='font style: double, mini, slant, smooth, rounded, compact, chunky, thin, or figlet:<name>')
    ap.add_argument('--scale', type=int, default=None, help='size 1-4')
    ap.add_argument('--fill', default=None, help='block fill char')
    ap.add_argument('--gap', type=int, default=None, help='letter spacing 0-4')
    ap.add_argument('--outline', action='store_true', help='outline glyphs')
    ap.add_argument('--no-outline', dest='outline', action='store_false')
    ap.add_argument('--glow', action='store_true', help='ambient glow halo around glyphs')
    ap.add_argument('--no-glow', dest='glow', action='store_false')
    ap.add_argument('--shadow', action='store_true', help='drop shadow behind text')
    ap.add_argument('--no-shadow', dest='shadow', action='store_false')
    ap.add_argument('--case', default=None, choices=CASES)
    ap.add_argument('--player', default=None)
    ap.add_argument('--resync', type=float, default=None)
    ap.add_argument('--tick', type=float, default=None)
    ap.add_argument('--fade', type=float, default=None)
    ap.add_argument('--lyrics-dir', default=None)
    ap.add_argument('--cache-dir', default=None)
    ap.add_argument('--offset', type=float, default=None, help='manual sync offset in seconds (e.g. -0.10 or +0.15)')
    ap.add_argument('--no-audio-sync', dest='no_audio_sync', action='store_true',
                    help='disable live audio sync bridge (prosodic heuristic only)')
    ap.add_argument('--auto-offset', dest='auto_offset', action='store_true', default=True,
                    help='anchor timeline to first observed player position (v3.1, default on)')
    ap.add_argument('--no-auto-offset', dest='auto_offset', action='store_false',
                    help='disable auto-anchoring to first observed player position')
    ap.add_argument('--cache-events', dest='cache_events', action='store_true', default=True,
                    help='cache built events for instant reload on repeat plays (v3.1, default on)')
    ap.add_argument('--no-cache-events', dest='cache_events', action='store_false',
                    help='disable the built-events cache')
    ap.add_argument('--config', default=None)
    return ap


def _request_exit():
    global should_exit
    should_exit = True


# --------------------------------------------------------------------------
# Settings GUI (Tkinter)
# --------------------------------------------------------------------------
def run_setup(cfg, config_path):
    try:
        import tkinter as tk
        from tkinter import ttk, colorchooser
    except Exception:
        sys.stderr.write('The settings GUI requires tkinter.\n')
        sys.stderr.write('On CachyOS / Arch, install it with:  sudo pacman -S tk\n')
        sys.exit(1)

    def val(key, default):
        v = cfg.get(key)
        return v if v is not None else default

    font_name = val('font', DEFAULTS['font'])
    scale = int(_as_float(val('scale', DEFAULTS['scale']), DEFAULTS['scale']))
    fill = val('fill', DEFAULT_FILL)
    gap = int(_as_float(val('gap', DEFAULTS['gap']), DEFAULTS['gap']))
    outline = _as_bool(val('outline', DEFAULTS['outline']), False)
    bold = _as_bool(val('bold', DEFAULTS['bold']), True)
    glow = _as_bool(val('glow', DEFAULTS['glow']), False)
    shadow = _as_bool(val('shadow', DEFAULTS['shadow']), True)
    glow_color = val('glow_color', DEFAULTS['glow_color'])
    case = val('case', DEFAULTS['case'])
    v_offset = int(_as_float(val('v_offset', DEFAULTS['v_offset']), 0))
    color_mode = val('color_mode', DEFAULTS['color_mode'])
    color = val('color', DEFAULTS['color'])
    bg = val('bg', DEFAULTS['bg'])
    transition = val('transition', DEFAULTS['transition'])
    fade_dur = _as_float(val('fade', DEFAULTS['fade']), DEFAULTS['fade'])
    paused_dim = _as_float(val('paused_dim', DEFAULTS['paused_dim']), DEFAULTS['paused_dim'])
    player = val('player', 'auto') or 'auto'
    plain = _as_bool(val('plain', DEFAULTS['plain']), False)

    figlets = figlet_fonts()
    font_choices = list(FONTS.keys()) + ['figlet:' + f for f in figlets]
    if font_name not in font_choices:
        font_choices.append(font_name)

    players = ['auto']
    try:
        out = subprocess.run(['playerctl', '-l'], capture_output=True,
                             text=True, timeout=2).stdout.split()
        players += [p for p in out if p and p != 'No']
    except Exception:
        pass

    try:
        root = tk.Tk()
    except tk.TclError:
        sys.stderr.write('Could not open the settings window: a graphical display is required.\n')
        sys.stderr.write('On a headless/SSH session, set DISPLAY, or edit\n')
        sys.stderr.write('  ~/.config/word_sync/config.toml  directly.\n')
        sys.exit(1)

    root.title('Word Sync — Settings (v2.1)')
    root.geometry('920x680')
    root.configure(bg='#12131a')

    font_var = tk.StringVar(value=font_name)
    scale_var = tk.IntVar(value=scale)
    fill_var = tk.StringVar(value=fill)
    gap_var = tk.IntVar(value=gap)
    outline_var = tk.BooleanVar(value=outline)
    bold_var = tk.BooleanVar(value=bold)
    glow_var = tk.BooleanVar(value=glow)
    shadow_var = tk.BooleanVar(value=shadow)
    glowcol_var = tk.StringVar(value=glow_color)
    case_var = tk.StringVar(value=case)
    voff_var = tk.IntVar(value=v_offset)
    cmode_var = tk.StringVar(value=color_mode)
    color_var = tk.StringVar(value=color)
    bg_var = tk.StringVar(value=bg)
    trans_var = tk.StringVar(value=transition)
    fade_var = tk.DoubleVar(value=fade_dur)
    pdim_var = tk.DoubleVar(value=paused_dim)
    player_var = tk.StringVar(value=player)
    plain_var = tk.BooleanVar(value=plain)
    preview_var = tk.StringVar(value='NEON')
    theme_var = tk.StringVar(value='')

    frm = ttk.Frame(root, padding=12)
    frm.pack(fill='both', expand=True)

    topf = ttk.Frame(frm)
    topf.grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 8))
    ttk.Label(topf, text='Theme (Curated Setups):', font=('Sans', 10, 'bold')).pack(side='left')
    ttk.OptionMenu(topf, theme_var, '', *([''] + list(THEMES))).pack(side='left', padx=6)
    ttk.Label(topf, text='(Snaps font, halo glow, colors & transitions automatically)').pack(side='left', padx=4)

    appear = ttk.LabelFrame(frm, text='Appearance & Font', padding=8)
    colorf = ttk.LabelFrame(frm, text='Effects & Color', padding=8)
    playf = ttk.LabelFrame(frm, text='Playback & System', padding=8)
    prevf = ttk.LabelFrame(frm, text='Live Preview', padding=8)

    appear.grid(row=1, column=0, sticky='nsew', padx=4, pady=4)
    colorf.grid(row=1, column=1, sticky='nsew', padx=4, pady=4)
    playf.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=4, pady=4)
    prevf.grid(row=1, column=2, rowspan=2, sticky='nsew', padx=4, pady=4)

    def lab(frame, r, c, text):
        ttk.Label(frame, text=text).grid(row=r, column=c, sticky='w', pady=2)

    lab(appear, 0, 0, 'Font')
    ttk.OptionMenu(appear, font_var, font_var.get(), *font_choices).grid(row=0, column=1, sticky='w')
    lab(appear, 1, 0, 'Scale (0 = Auto-Fit)')
    ttk.Scale(appear, from_=0, to=4, orient='horizontal', variable=scale_var).grid(row=1, column=1, sticky='we')
    lab(appear, 2, 0, 'Fill char')
    ttk.Combobox(appear, textvariable=fill_var, values=FILL_CHOICES, width=6).grid(row=2, column=1, sticky='w')
    lab(appear, 3, 0, 'Letter spacing')
    ttk.Scale(appear, from_=0, to=4, orient='horizontal', variable=gap_var).grid(row=3, column=1, sticky='we')
    lab(appear, 4, 0, 'Bold font')
    ttk.Checkbutton(appear, variable=bold_var).grid(row=4, column=1, sticky='w')
    lab(appear, 5, 0, 'Outline style')
    ttk.Checkbutton(appear, variable=outline_var).grid(row=5, column=1, sticky='w')
    lab(appear, 6, 0, 'Letter case')
    ttk.OptionMenu(appear, case_var, case_var.get(), *CASES).grid(row=6, column=1, sticky='w')
    lab(appear, 7, 0, 'Vertical offset')
    ttk.Scale(appear, from_=-5, to=5, orient='horizontal', variable=voff_var).grid(row=7, column=1, sticky='we')

    lab(colorf, 0, 0, 'Color mode')
    ttk.OptionMenu(colorf, cmode_var, cmode_var.get(), *COLOR_MODES).grid(row=0, column=1, sticky='w')
    lab(colorf, 1, 0, 'Font Color')
    ttk.Entry(colorf, textvariable=color_var, width=12).grid(row=1, column=1, sticky='w')
    ttk.Button(colorf, text='Pick', command=lambda: _pick(color_var)).grid(row=1, column=2, sticky='w', padx=2)

    lab(colorf, 2, 0, 'Ambient Glow (Halo)')
    ttk.Checkbutton(colorf, variable=glow_var).grid(row=2, column=1, sticky='w')
    lab(colorf, 3, 0, 'Glow Color')
    ttk.Entry(colorf, textvariable=glowcol_var, width=12).grid(row=3, column=1, sticky='w')
    ttk.Button(colorf, text='Pick', command=lambda: _pick(glowcol_var)).grid(row=3, column=2, sticky='w', padx=2)

    lab(colorf, 4, 0, 'Drop Shadow')
    ttk.Checkbutton(colorf, variable=shadow_var).grid(row=4, column=1, sticky='w')

    lab(colorf, 5, 0, 'Transition')
    ttk.OptionMenu(colorf, trans_var, trans_var.get(), 'fade', 'instant').grid(row=5, column=1, sticky='w')
    lab(colorf, 6, 0, 'Fade seconds')
    ttk.Scale(colorf, from_=0.0, to=0.8, orient='horizontal', variable=fade_var).grid(row=6, column=1, sticky='we')
    lab(colorf, 7, 0, 'Paused brightness')
    ttk.Scale(colorf, from_=0.2, to=1.0, orient='horizontal', variable=pdim_var).grid(row=7, column=1, sticky='we')

    lab(playf, 0, 0, 'Player target')
    ttk.OptionMenu(playf, player_var, player_var.get(), *players).grid(row=0, column=1, sticky='w')
    lab(playf, 0, 2, 'Pipe / plain mode')
    ttk.Checkbutton(playf, variable=plain_var).grid(row=0, column=3, sticky='w')

    ttk.Label(prevf, text='Test word:').grid(row=0, column=0, sticky='w')
    ttk.Entry(prevf, textvariable=preview_var, width=18).grid(row=1, column=0, sticky='we', pady=2)
    prev = tk.Text(prevf, width=44, height=17, bg='#0a0a0f',
                   fg=color, font=('Courier New', 13, 'bold'))
    prev.grid(row=2, column=0, sticky='nsew', padx=4, pady=4)

    def _pick(var):
        c = colorchooser.askcolor(color=var.get())
        if c and c[1]:
            var.set(c[1])

    def update_preview(*_):
        f = font_var.get()
        s = scale_var.get()
        fl = fill_var.get() or DEFAULT_FILL
        g = ' ' * gap_var.get()
        oc = bool(outline_var.get())
        cs = case_var.get()
        hollow = (f == 'thin')
        col = color_var.get()
        try:
            rgb = resolve_color(col)
            hexcol = col if col.startswith('#') else '#%02x%02x%02x' % rgb
        except Exception:
            hexcol = '#ffffff'
        word = preview_var.get() or 'NEON'
        try:
            if f.startswith('figlet:'):
                lines = render_figlet(word, f.split(':', 1)[1], 140, 30) or []
            else:
                font = FONT_5 if hollow else FONTS.get(f, FONT_5)
                lines = generate_glyph_rows(word, font, fl, g, s, oc or hollow, cs, hollow)
        except Exception:
            lines = [word]

        # In preview text, display font core
        prev.configure(fg=hexcol,
                       font=('Courier New', 13, 'bold') if bold_var.get()
                       else ('Courier New', 13, 'normal'))
        prev.delete('1.0', 'end')
        prev.insert('end', '\n'.join(lines))

    def apply_profile(p):
        if 'font' in p:
            font_var.set(p['font'])
        if 'scale' in p:
            scale_var.set(p['scale'])
        if 'fill' in p:
            fill_var.set(p['fill'])
        if 'gap' in p:
            gap_var.set(p['gap'])
        if 'color_mode' in p:
            cmode_var.set(p['color_mode'])
        if 'outline' in p:
            outline_var.set(bool(p['outline']))
        if 'bold' in p:
            bold_var.set(bool(p['bold']))
        if 'glow' in p:
            glow_var.set(bool(p['glow']))
        if 'shadow' in p:
            shadow_var.set(bool(p['shadow']))
        if 'glow_color' in p:
            glowcol_var.set(p['glow_color'])
        if 'case' in p:
            case_var.set(p['case'])
        if 'color' in p:
            color_var.set(p['color'])
        if 'bg' in p:
            bg_var.set(p['bg'])
        if 'transition' in p:
            trans_var.set(p['transition'])
        if 'fade' in p:
            fade_var.set(p['fade'])
        if 'paused_dim' in p:
            pdim_var.set(p['paused_dim'])
        if 'player' in p and p['player']:
            player_var.set(p['player'])
        if 'plain' in p:
            plain_var.set(bool(p['plain']))
        update_preview()

    def on_font_change(*_):
        f = font_var.get()
        prof = FONT_PROFILES.get(f)
        if prof is None and f.startswith('figlet:'):
            prof = GENERIC_FIGLET_PROFILE
        if prof:
            apply_profile(prof)

    def on_theme_change(*_):
        t = theme_var.get()
        if t and t in THEMES:
            apply_profile(THEMES[t])

    def reset_defaults():
        theme_var.set('')
        apply_profile(DEFAULTS)

    def do_save():
        data = dict(cfg)
        data['font'] = font_var.get()
        data['scale'] = scale_var.get()
        data['fill'] = fill_var.get() or DEFAULT_FILL
        data['gap'] = gap_var.get()
        data['outline'] = bool(outline_var.get())
        data['bold'] = bool(bold_var.get())
        data['glow'] = bool(glow_var.get())
        data['shadow'] = bool(shadow_var.get())
        data['glow_color'] = glowcol_var.get()
        data['case'] = case_var.get()
        data['v_offset'] = voff_var.get()
        data['color_mode'] = cmode_var.get()
        data['color'] = color_var.get()
        data['bg'] = 'none' if bg_var.get() in ('', 'none') else bg_var.get()
        data['transition'] = trans_var.get()
        data['fade'] = fade_var.get() if trans_var.get() == 'fade' else 0.0
        data['paused_dim'] = pdim_var.get()
        data['player'] = '' if player_var.get() == 'auto' else player_var.get()
        data['plain'] = bool(plain_var.get())
        write_config(config_path, data)
        try:
            import tkinter.messagebox as mb
            mb.showinfo('Word Sync', f'Settings saved to:\n{config_path}\n\nRun: word-sync.py')
        except Exception:
            pass
        root.destroy()

    btns = ttk.Frame(frm)
    btns.grid(row=3, column=0, columnspan=3, pady=10)
    ttk.Button(btns, text='Save & Apply', command=do_save).pack(side='left', padx=4)
    ttk.Button(btns, text='Reset to Defaults', command=reset_defaults).pack(side='left', padx=4)
    ttk.Button(btns, text='Cancel', command=root.destroy).pack(side='left', padx=4)

    for v in (font_var, scale_var, fill_var, gap_var, outline_var, bold_var,
              glow_var, shadow_var, glowcol_var, case_var, voff_var, cmode_var,
              color_var, bg_var, trans_var, fade_var, pdim_var, player_var, plain_var, preview_var):
        v.trace_add('write', update_preview)
    font_var.trace_add('write', on_font_change)
    theme_var.trace_add('write', on_theme_change)

    update_preview()
    root.mainloop()


# --------------------------------------------------------------------------
# Main Execution Loop
# --------------------------------------------------------------------------
def main():
    global should_exit
    ap = build_parser()
    args = ap.parse_args()

    if args.selftest:
        sys.exit(run_selftest())

    config_path = args.config or os.path.expanduser('~/.config/word_sync/config.toml')
    cfg = load_config(config_path)

    if args.setup:
        run_setup(cfg, config_path)
        return

    # Precedence order: DEFAULTS < config.toml < --theme < explicit CLI flags
    active = dict(DEFAULTS)
    for k in DEFAULTS:
        if cfg.get(k) is not None:
            active[k] = cfg.get(k)
    if args.theme:
        active.update(THEMES[args.theme])
    for k in ('color', 'color_mode', 'bg', 'font', 'scale', 'fill', 'gap',
              'outline', 'bold', 'glow', 'shadow', 'glow_color', 'case', 'v_offset',
              'transition', 'paused_dim', 'fade', 'player', 'plain', 'offset',
              'lyrics_dir', 'cache_dir', 'resync', 'tick'):
        cv = getattr(args, k, None)
        if cv is not None:
            active[k] = cv

    color = active['color']
    color_mode = active['color_mode']
    bg_raw = active['bg']
    bg_rgb = None
    if bg_raw and bg_raw != 'none':
        try:
            bg_rgb = resolve_color(bg_raw)
        except argparse.ArgumentTypeError:
            bg_rgb = None

    player = active['player'] or None
    font_name = active['font']
    scale = int(_as_float(active['scale'], DEFAULTS['scale']))
    fill = active['fill']
    gap = int(_as_float(active['gap'], DEFAULTS['gap']))
    outline = bool(active['outline'])
    bold = _as_bool(active['bold'], True)
    glow = _as_bool(active['glow'], False)
    shadow = bool(active['shadow'])
    glow_color = active.get('glow_color', '#3b1261')
    case = active['case']
    v_offset = int(_as_float(active['v_offset'], 0))
    transition = active['transition']
    fade = _as_float(active['fade'], DEFAULTS['fade'])
    paused_dim = _as_float(active['paused_dim'], DEFAULTS['paused_dim'])
    resync = _as_float(active['resync'], DEFAULTS['resync'])
    tick = _as_float(active['tick'], DEFAULTS['tick'])
    lyrics_dir = active['lyrics_dir']
    cache_dir = active['cache_dir']
    global_offset = _as_float(active.get('offset', 0.0), 0.0)   # from --offset / config
    manual_offset = global_offset                                # live value, tuned with + / - / 0

    try:
        color_rgb = resolve_color(color)
    except argparse.ArgumentTypeError as e:
        sys.stderr.write(f'{e}\n')
        sys.exit(2)

    opts = {
        'color_mode': color_mode,
        'bg_rgb': bg_rgb,
        'font_name': font_name,
        'scale': scale,
        'fill': fill,
        'gap': gap,
        'outline': outline,
        'bold': bold,
        'glow': glow,
        'shadow': shadow,
        'glow_color': glow_color,
        'case': case,
        'v_offset': v_offset,
        'transition': transition,
        'paused_dim': paused_dim,
    }

    if not args.plain and not playerctl_available():
        sys.stderr.write('Warning: playerctl is not installed or not in PATH.\n')
        sys.stderr.write('Install on CachyOS / Arch:  sudo pacman -S playerctl\n')

    signal.signal(signal.SIGINT, lambda *_: _request_exit())
    signal.signal(signal.SIGTERM, lambda *_: _request_exit())

    old_term_settings = None
    if not args.plain and sys.stdin.isatty():
        try:
            import termios, tty
            old_term_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            pass
    if not args.plain:
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()

    song_key = None
    track_id = None
    events, starts = [], []
    last_word_shown = None
    last_idx = -1
    failed = {}            # song_key -> monotonic time of last failed fetch (retried after 2 min)
    fade_start = -1.0
    word_rgb = color_rgb
    dot_phase = 0
    last_dot_time = 0.0
    clock = PlaybackClock()
    audio_tracker = None
    if not getattr(args, 'no_audio_sync', False):
        try:
            audio_tracker = LiveAudioTracker(no_audio=getattr(args, 'no_audio_sync', False))
            audio_tracker.start()
        except Exception:
            audio_tracker = None
    needs_redraw = False          # repaint (e.g. brightness on pause) without re-announcing a word
    song_offsets = load_song_offsets()
    offset_dirty = False
    have_state = False
    playing = False
    poll_every = max(0.12, min(0.3, resync / 5.0))   # fast position poll (seek/pause latency)
    next_poll = 0.0

    # v3.1: explicit display state machine + live phase-locker + per-song
    # residual log. The state machine is the single source of truth for
    # "what should be on screen right now".
    display_state = DisplayStateMachine()
    phase_locker = PhaseLocker(audio_tracker=audio_tracker)
    residuals = SongResiduals()

    stretch = 1.0
    cur_lines = []                 # parsed lines of the current song (to rebuild on stretch change)
    cur_duration = 0.0
    stretch_dirty = False
    candidate_idx = 0              # v1.4: which LRCLIB candidate is active (n key cycles)
    cand_count = 1
    cur_state = None               # last player state, used to reload lyrics on 'n'
    no_lyric_until = 0.0           # keep the "no lyrics" notice up until this time, then dots

    def _on_resize(*_):
        nonlocal needs_redraw
        needs_redraw = True

    try:
        signal.signal(signal.SIGWINCH, _on_resize)   # redraw promptly on terminal resize
    except (AttributeError, ValueError):
        pass

    def _remember_offset():
        # Persist this song's tuning. Default values remove the entry so the file can't grow forever.
        if not song_key:
            return
        off = round(manual_offset - global_offset, 3)
        stc = round(stretch, 3)
        if abs(off) < 1e-9 and abs(stc - 1.0) < 1e-9:
            song_offsets.pop(song_key, None)
        else:
            song_offsets[song_key] = {'offset': off, 'stretch': stc}
        save_song_offsets(song_offsets)

    def _load_song(st, cand_idx=0):
        """Fetch + parse + build the word timeline for the current track.
        Returns (events, lines, n_candidates). v3.1: cache + auto-pick."""
        nonlocal manual_offset
        artist, title = st['artist'], st['title']
        if not args.plain:
            render_message(f'loading lyrics — {artist} – {title}')
        use_cache = getattr(args, 'cache_events', True)
        auto_offset = getattr(args, 'auto_offset', True)
        # v3.1: try the built-events cache first (skips the HTTP fetch)
        evs = None
        if use_cache and cand_idx == 0:
            evs = load_event_cache(artist, title, st.get('duration', 0.0))
            if evs is not None:
                # Apply cached intrinsic offset so the timeline lines up
                # with the player's first observed position automatically.
                co = cached_intrinsic_offset(artist, title, st.get('duration', 0.0))
                if co is not None and auto_offset:
                    manual_offset = global_offset + co
        data = None
        fk = failed.get(st['song_key'])
        if fk is None or time.monotonic() - fk > 120.0:   # v1.4: retry transient failures
            data = get_lyrics(artist, title, st['duration'], cache_dir, lyrics_dir)
            if data is None:
                failed[st['song_key']] = time.monotonic()
            else:
                failed.pop(st['song_key'], None)
        lines = []
        cands = data['candidates'] if data else []
        if cands:
            # v3.1: auto-pick the best candidate by closeness to first
            # observed position (when we know it) and word-level bonus.
            if cand_idx == 0 and auto_offset:
                first_pos = st.get('position', 0.0) or 0.0
                pick = pick_candidate(cands, first_pos)
            else:
                pick = cand_idx
            cand = cands[min(pick, len(cands) - 1)]
            # Real word timing first (lyricsfile); fall back to line-level LRC.
            if cand.get('lyricsfile'):
                lines = parse_lyricsfile(cand['lyricsfile'])
            if not lines and cand.get('synced'):
                lines = parse_lrc(cand['synced'])
        if evs is None:
            _audio_arg = None
            if audio_tracker is not None:
                try:
                    if (audio_tracker.available or len(audio_tracker.left) > 1000):
                        _audio_arg = audio_tracker.get_audio()
                except Exception:
                    _audio_arg = None
            evs = build_word_events(lines, st['duration'], stretch, audio=_audio_arg) if lines else []
            if use_cache and evs and lines and cand_idx == 0:
                save_event_cache(artist, title, st.get('duration', 0.0),
                                 evs, intrinsic_offset=
                                 (manual_offset - global_offset) if auto_offset else 0.0)
        if not evs and not args.plain:
            render_message(f'no synced lyrics found for {artist} – {title}')
        return evs, lines, len(cands)

    try:
        while not should_exit:
            now = time.monotonic()

            # ---- position / status poll (fast cadence; the clock absorbs latency) ----
            if now >= next_poll:
                fresh = get_player_state(player)
                next_poll = time.monotonic() + poll_every

                if fresh is None or fresh['status'] == 'Stopped':
                    if not args.plain and (song_key is not None or not have_state):
                        render_message('♪  waiting for playback…  ♪')
                    if offset_dirty:
                        _remember_offset()
                        offset_dirty = False
                    song_key = track_id = None
                    have_state = False
                    events, starts = [], []
                    clock.reset()
                    display_state.reset()
                    time.sleep(min(resync, 0.5))
                    continue

                have_state = True
                playing = fresh['status'] == 'Playing'
                changed = (fresh['song_key'] != song_key) or (
                    fresh['track_id'] and track_id and fresh['track_id'] != track_id
                    and fresh['song_key'] == song_key and fresh['position'] < 2.0
                    and clock.position_at(time.monotonic()) > 10.0)   # same title replayed
                if changed:
                    if offset_dirty:
                        _remember_offset()
                        offset_dirty = False
                    song_key, track_id = fresh['song_key'], fresh['track_id']
                    last_word_shown = None
                    last_idx = -1
                    fade_start = -1.0
                    clock.reset()
                    # v3.1: apply self-tuned residual offset to the
                    # song_offsets entry on song change, then load.
                    saved = song_offsets.get(song_key, {})
                    base_offset = global_offset + saved.get('offset', 0.0)
                    # blend in the learned residual from past plays
                    learned = residuals.apply_to(song_key, 0.0)
                    manual_offset = base_offset + learned
                    stretch = saved.get('stretch', 1.0)
                    stretch_dirty = False
                    candidate_idx = 0
                    events, cur_lines, cand_count = _load_song(fresh, 0)
                    cur_duration = fresh['duration']
                    no_lyric_until = (time.monotonic() + 4.0) if not events else 0.0
                    starts = [e[0] for e in events]
                    # v3.1: anchor the timeline to the player's FIRST
                    # observed position. If first_pos > line[0].start,
                    # the lyrics are late in the LRC (or the player
                    # started mid-song); shift manual_offset so the
                    # first line lands at first_pos.
                    try:
                        if events and getattr(args, 'auto_offset', True):
                            first_word_t = next((t for t, _ in events if _), None)
                            if first_word_t is not None and first_word_t > 0:
                                first_pos = fresh.get('position', 0.0) or 0.0
                                if first_pos >= 0 and abs(first_pos - first_word_t) < 8.0:
                                    manual_offset = global_offset + (first_pos - first_word_t)
                    except Exception:
                        pass
                    display_state.reset()
                    if phase_locker is not None:
                        phase_locker.notify_word_displayed(0.0)
                    # first sample after a load: the fetch may have taken seconds, so re-poll
                    fresh = get_player_state(player) or fresh
                    playing = fresh['status'] == 'Playing'
                track_id = fresh['track_id'] or track_id
                cur_state = fresh
                clock.update(fresh['position'], playing, fresh['t_before'],
                             fresh['t_after'], fresh['rate'])
                if clock.snapped:
                    needs_redraw = True   # pause dim / resume: repaint, don't re-announce

            # ---- hotkeys: + / - nudge by 25 ms, 0 resets to the global offset ----
            if not args.plain and sys.stdin.isatty():
                try:
                    import select
                    while select.select([sys.stdin], [], [], 0)[0]:
                        ch = sys.stdin.read(1)
                        if ch in ('+', '='):
                            manual_offset += 0.025
                            offset_dirty = True
                        elif ch in ('-', '_'):
                            manual_offset -= 0.025
                            offset_dirty = True
                        elif ch == '[':
                            stretch = max(0.5, round(stretch - 0.02, 3))
                            stretch_dirty = offset_dirty = True
                        elif ch == ']':
                            stretch = min(1.6, round(stretch + 0.02, 3))
                            stretch_dirty = offset_dirty = True
                        elif ch == 'n':      # v1.4: next lyric candidate for this track
                            if cand_count > 1 and cur_state is not None:
                                candidate_idx = (candidate_idx + 1) % cand_count
                                events, cur_lines, cand_count = _load_song(cur_state, candidate_idx)
                                starts = [e[0] for e in events]
                                last_word_shown = None
                                last_idx = -1
                                no_lyric_until = (time.monotonic() + 4.0) if not events else 0.0
                        elif ch == '0':
                            manual_offset = global_offset
                            stretch = 1.0
                            stretch_dirty = offset_dirty = True
                except Exception:
                    pass
            if stretch_dirty and cur_lines:
                stretch_dirty = False
                events = build_word_events(cur_lines, cur_duration, stretch)
                starts = [e[0] for e in events]
                last_idx = -1

            tick_t = time.monotonic()
            pos = max(0.0, clock.position_at(tick_t) - manual_offset) if have_state else 0.0
            if args.plain:
                needs_redraw = False

            # v3.1: PhaseLocker consult + DisplayStateMachine step.
            # The state machine decides WHAT to show; we then call the
            # same draw_block / print / fade pipeline as v3.0.
            audio_state = None
            if phase_locker is not None:
                audio_state = phase_locker.snapshot(pos)

            if events and have_state:
                decision = display_state.step(pos, events, audio_state=audio_state,
                                              now=tick_t)
                word = decision['show_word']
                is_word_change = decision['is_word_change']
                show_dots = decision['show_dots']

                # v3.1: live phase-lock on the *next* predicted word.
                if (phase_locker is not None and phase_locker.available()
                        and word is not None and not is_word_change):
                    # Find this word's predicted time
                    word_idx_in_events = next(
                        (i for i, (_, w) in enumerate(events) if w == word
                         and events[i][0] <= pos + 0.05), None)
                    if word_idx_in_events is not None:
                        t_pred = events[word_idx_in_events][0]
                        verdict = phase_locker.consult(t_pred, pos)
                        # 'snap' is logged for residual learning; the
                        # display path stays on the predicted time so the
                        # remainder of the line keeps flowing.
                        if verdict[0] == 'snap':
                            residuals.log(
                                st.get('song_key') if 'st' in dir() else None,
                                t_pred, pos, audio_vocal_t=verdict[1])

                if color_mode == 'random':
                    word_rgb = hsv_to_rgb(random.random(), 1, 1)
                else:
                    word_rgb = color_rgb

                if word is not None and word != '':
                    if is_word_change:
                        last_idx = (decision.get('active_event_i', last_idx)
                                    if 'active_event_i' in decision else last_idx)
                        last_word_shown = word
                        instant = (fade <= 0) or (transition == 'instant')
                        if instant:
                            fade_start = -1.0
                            if args.plain:
                                if word:
                                    print(word, flush=True)
                            else:
                                draw_block(word, word_rgb,
                                           intensity_for(playing, 1.0, paused_dim), opts)
                        else:
                            fade_start = tick_t
                            if args.plain:
                                if word:
                                    print(word, flush=True)
                            else:
                                draw_block(word, word_rgb,
                                           intensity_for(playing, 0.0, paused_dim), opts)
                        if phase_locker is not None:
                            phase_locker.notify_word_displayed(pos)
                    elif needs_redraw and word and not args.plain:
                        needs_redraw = False
                        draw_block(word, word_rgb, intensity_for(playing, 1.0, paused_dim), opts)
                    elif not args.plain and fade_start >= 0 and (tick_t - fade_start) < fade:
                        draw_block(word, word_rgb,
                                   intensity_for(playing, (tick_t - fade_start) / fade, paused_dim), opts)
                    elif not args.plain and not playing:
                        draw_block(word, word_rgb,
                                   intensity_for(False, 1.0, paused_dim), opts)
                else:
                    # No word to show — render dots if the state says so
                    # or if we are explicitly in a no-lyric path.
                    if not args.plain and (show_dots or not events):
                        if needs_redraw or tick_t - last_dot_time > 0.35:
                            dot_phase += 1
                            last_dot_time = tick_t
                            needs_redraw = False
                            draw_block('', word_rgb, 1.0, opts, dot_phase)
                    if is_word_change and last_word_shown not in (None, ''):
                        last_word_shown = ''
                        last_idx = -1
            elif have_state and playing and not events and not args.plain:
                # track has no synced lyrics: keep the notice up for a few seconds,
                # then idle as pulsing dots instead of a frozen message (v1.4)
                if tick_t >= no_lyric_until and (needs_redraw or tick_t - last_dot_time > 0.35):
                    dot_phase += 1
                    last_dot_time = tick_t
                    needs_redraw = False
                    draw_block('', word_rgb, 1.0, opts, dot_phase)
            time.sleep(tick)

    finally:
        if offset_dirty:
            _remember_offset()
        if old_term_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term_settings)
            except Exception:
                pass
        if not args.plain:
            sys.stdout.write(SHOW_CURSOR + CLEAR)
            sys.stdout.flush()


if __name__ == '__main__':
    main()
