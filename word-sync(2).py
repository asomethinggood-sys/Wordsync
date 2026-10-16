#!/usr/bin/env python3
"""
word_sync.py — v3.0 (hybrid prosodic + live-audio sync engine)
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
- v3.0: the four architectural bugs behind the "super slow" crawl on songs like
  Panchiko's D>E>A>T>H>M>E>T>A>L are fixed:
    1. Pace was the MEDIAN line window over the median syllable count, so a
       6-syllable phrase sung in ~1.8 s followed by 4-5 s of guitars (a 6.5 s
       window) pegged pace to the 0.822 s/syllable ceiling. `measure_cadence()`
       now anchors the True Articulation Pace on the 25-30th percentile of the
       per-line syllable rate (the *dense-phrase frontier* — a singer cannot
       articulate slower than their own gapless lines) and adapts it per line
       with a 5-line rolling window, refusing to inflate when one line's window
       is stretched far beyond its content while its neighbours are fast
       (isolated spike = post-phrase instrumental gap or a held final note;
       contiguous slow run = a genuine half-time bridge / beat switch).
    2. `estimate_line()` folded the final word's articulation *and* its held
       note into the span that spaces words[0:n-1], pushing every earlier word
       progressively later. Onsets are now spaced by an `onset_span` that
       covers only words[0:n-1] plus intra-line caesurae, while the clear point
       is placed after the last word's own articulation + hold. The post-phrase
       tail-compression threshold dropped from 8.594 s to `phrase + 0.8 s`, so
       a 3.5-5.0 s window clears on time too.
    3. Zero prosodic awareness: est_line() now runs a Tier-2.5 cadence model —
       multilingual unstressed function-word compression (0.55x, curated
       frozenset, no POS tagger), intra-line caesurae from punctuation
       (1.25x pre-boundary lengthening + a 0.14-0.32 s micro-pause scaled by
       window slack), isochronous locking of repeated hooks ("la la la",
       "yeah yeah yeah", "d-d-death"), and syncopation/anacrusis pickup
       compression ("In the DEATH..." lands the stressed word on the downbeat).
- v3.0: the offline audio engine is no longer dead code during live playback.
  `LiveAudioTracker` streams the player's monitor sink with `pw-record`
  (PipeWire) or `parec` (PulseAudio) — raw s16le stereo, 16 kHz — in a daemon
  thread, converts it with `StreamingFeatureExtractor` (same band plan and
  VAD/onset/freeze rules as the offline engine, causal and ~3 % of one core),
  and keeps a 12 s ring of song-time-stamped frames plus a per-track capture
  cache. That cache is what turns a replay into a real acoustic alignment
  (`build_word_events(..., audio=...)`, `meta['aligned']`), which is exactly
  the `audio=` argument `_load_song()` never passed before.
- v3.0: stereo Mid/Side vocal isolation (`extract_features_stereo`) — 300 Hz-
  3.4 kHz formant band on Mid and Side, 2.5-5.5 kHz consonant band on Mid,
  `isolated = max(0, Mid - gain*Side)` weighted by centre dominance — so
  distorted guitars, cymbals and synths that fool a mono band-pass VAD stop
  triggering it (verified: 0 false frames on a mix where mono VAD reports 100 %).
- v3.0: `LivePhaseLock` applies the engine causally in the playback loop:
  vocal entry gate (hold a word up to +350 ms until the real onset rises),
  causal onset snap (±160 ms onto a strong isolated-vocal attack),
  post-phrase VAD freeze & early clear (>380 ms of centre-channel silence
  releases the line so the interlude follows the schedule), and rolling
  re-alignment (a fast local `refine_word_events` over the active line's
  remaining tokens every 150 ms, self-throttling on CPU cost). 100 % fail-soft:
  no backend, no permission, or `--no-audio-sync` => the prosodic engine alone.
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
from collections import OrderedDict, deque
import sys
import threading
import time
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
    # --- v3.0 stereo Mid/Side vocal isolation ---
    # isolated = max(0, formant(Mid) - side_cancel_gain * formant(Side)).
    # 1.0 is right for a centre-panned lead over wide guitars; lower it (0.5–0.8)
    # if a track has a deliberately wide or double-tracked lead vocal.
    side_cancel_gain: float = 1.0
    cons_lo_hz: float = 2500.0        # consonant / sibilance transient band
    cons_hi_hz: float = 5500.0
    # --- v3.0 live causal phase-locking ---
    live_enabled: bool = True
    live_refine: bool = True          # rolling local re-alignment during a line
    live_refine_every_s: float = 0.150
    live_refine_budget_s: float = 0.060   # self-throttle if a refine costs more
    live_beam_width: int = 6
    entry_gate_max_s: float = 0.350   # vocal entry gate hold ceiling
    onset_snap_s: float = 0.160       # causal onset snap radius
    onset_snap_min: float = 0.500     # onset strength needed to snap
    freeze_clear_s: float = 0.380     # post-phrase centre-channel silence
    ring_s: float = 12.0              # rolling feature memory
    capture_rate: int = 16000


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
        xs = [sum(xs[i:i + nch]) / nch for i in range(0, len(xs) - nch + 1, nch)]
    return sr, xs


def write_wav(path: str, sr: int, samples: Sequence[float],
              right: Optional[Sequence[float]] = None) -> None:
    """16-bit PCM writer, used by the synthetic test harness.

    Mono by default; when `right` is supplied the two channels are interleaved
    into a stereo file (the mid-side isolation tests need real stereo).
    """
    buf = array('h')

    def _q(x):
        return int(max(-1.0, min(1.0, x)) * 32767.0)

    if right is None:
        for x in samples:
            buf.append(_q(x))
        nch = 1
    else:
        n = min(len(samples), len(right))
        for i in range(n):
            buf.append(_q(samples[i]))
            buf.append(_q(right[i]))
        nch = 2
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(nch)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(buf.tobytes())


def load_wav_stereo(path: str) -> Tuple[int, List[float], List[float]]:
    """(sr, left, right) from a PCM WAV. Mono files come back duplicated."""
    with wave.open(path, 'rb') as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sw == 2:
        vals = array('h')
        vals.frombytes(raw)
        vals = [s / 32768.0 for s in vals]
    elif sw == 1:
        vals = [(b - 128) / 128.0 for b in raw]
    elif sw == 4:
        vals = array('i')
        vals.frombytes(raw)
        vals = [s / 2147483648.0 for s in vals]
    else:
        raise ValueError('WAV sample width %d not supported (need 1/2/4)' % sw)
    if nch <= 1:
        return sr, list(vals), list(vals)
    left = vals[0::nch]
    right = vals[1::nch]
    return sr, left, right


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


class _OnePoleHP:
    """Stateful one-pole high-pass (streaming twin of `_one_pole_hp`)."""

    __slots__ = ('a', 'px', 'py')

    def __init__(self, sr: int, fc: float):
        if fc <= 0:
            self.a, self.px, self.py = 1.0, 0.0, 0.0
            return
        dt = 1.0 / sr
        rc = 1.0 / (2.0 * math.pi * fc)
        self.a = rc / (rc + dt)
        self.px = self.py = 0.0

    def __call__(self, x: float) -> float:
        self.py = self.a * (self.py + x - self.px)
        self.px = x
        return self.py


class _OnePoleLP:
    """Stateful one-pole low-pass (streaming twin of `_one_pole_lp`)."""

    __slots__ = ('a', 'py')

    def __init__(self, sr: int, fc: float):
        if fc <= 0 or fc >= sr * 0.49:
            self.a, self.py = 1.0, 0.0
            return
        dt = 1.0 / sr
        rc = 1.0 / (2.0 * math.pi * fc)
        self.a = dt / (rc + dt)
        self.py = 0.0

    def __call__(self, x: float) -> float:
        self.py = self.py + self.a * (x - self.py)
        return self.py


def _hamming(n: int) -> List[float]:
    if n <= 1:
        return [1.0] * n
    return [0.54 - 0.46 * math.cos(2.0 * math.pi * i / (n - 1)) for i in range(n)]


@dataclass
class FrameFeat:
    t: float            # timestamp at window centre + calibration offset
    rms: float          # full-band RMS (not used as the onset cue)
    formant: float      # 300 Hz–3.4 kHz energy (the vocal cue; isolated in stereo)
    onset: float        # 0..1 formant-rise onset strength
    vad: bool           # vocal activity at this frame
    freeze: bool        # pointer must not advance (long pause)
    confidence: float   # 0..1, drives corridor width
    # v3.0 stereo mid-side fields (0.0 on the mono path).
    formant_mid: float = 0.0     # 300–3.4 kHz energy of Mid = (L+R)/2
    formant_side: float = 0.0    # 300–3.4 kHz energy of Side = (L-R)/2
    consonant: float = 0.0       # 2.5–5.5 kHz sibilance/transient band on Mid
    t_mono: float = 0.0          # capture-clock stamp for live frames


@dataclass
class FeatureTrack:
    sr: int
    hop_s: float
    frame_s: float
    lookahead_s: float
    calibration_offset_s: float
    receptive_delay_s: float          # 0.5 * frame_s (window centre)
    frames: List[FrameFeat] = field(default_factory=list)
    stereo: bool = False              # True when built from Mid/Side isolation

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


# --------------------------------------------------------------------------
# Shared envelope -> frame pipeline (v3.0: one implementation for the offline
# mono path, the offline stereo path and the live streaming extractor, so a
# captured frame behaves exactly like a WAV-analysis frame).
# --------------------------------------------------------------------------

def _onset_curve(env: List[float], cfg: AlignConfig, hop_s: float) -> List[float]:
    """Soft onset: formant-energy *rise rate* over ~55 ms, not a 1-frame
    full-band spike (task 3). Breathy / piano-vocal attacks show up here."""
    rise_n = max(1, int(round(cfg.onset_rise_s / hop_s)))
    raw_onsets = [0.0] * len(env)
    for t, e in enumerate(env):
        prev = env[t - rise_n] if t >= rise_n else env[0]
        raw_onsets[t] = max(0.0, e - prev)
    pos = [x for x in raw_onsets if x > 0.0]
    if pos:
        pos.sort()
        med = pos[len(pos) // 2]
        scale = med * 2.8 if med > 1e-9 else (max(pos) or 1.0)
    else:
        scale = 1.0
    return [max(0.0, min(1.0, x / scale)) for x in raw_onsets]


def _vad_threshold(env: Sequence[float]) -> Tuple[float, float]:
    """(threshold, noise floor) from the quietest 15 % of frames, capped against
    the median so an almost-fully-voiced clip still fires."""
    ordered = sorted(env) if env else [0.0]
    floor = ordered[max(0, int(0.15 * (len(ordered) - 1)))]
    median = ordered[len(ordered) // 2]
    return max(1.5e-4, min(floor * 3.4, median * 0.40 + 1e-6)), floor


def _vad_freeze(vad_thr: float, env: Sequence[float],
                cfg: AlignConfig, hop_s: float) -> Tuple[List[bool], List[bool]]:
    """VAD with hangover, then freeze after vad_freeze_s of inactivity (task 2)."""
    hang_n = max(1, int(round(cfg.vad_hangover_s / hop_s)))
    freeze_n = max(1, int(round(cfg.vad_freeze_s / hop_s)))
    raw_vad = [e > vad_thr for e in env]
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
    return vad, freeze


def _confidence_curve(env: Sequence[float], onsets: Sequence[float],
                      freeze: Sequence[bool], floor: float) -> List[float]:
    """Local confidence: high formant SNR + not a busy onset cluster
    (busy clusters => likely syncopation / messy attacks => widen)."""
    conf = [0.0] * len(env)
    w = 6
    for t, e in enumerate(env):
        lo = max(0, t - w)
        hi = min(len(env), t + w + 1)
        snr = e / (floor + 1e-6)
        snr_term = 1.0 / (1.0 + math.exp(-(snr - 4.0)))   # sigmoid around ~12 dB
        dens = sum(1 for x in onsets[lo:hi] if x > 0.45) / float(hi - lo)
        conf[t] = max(0.0, min(1.0, snr_term * (1.0 - 0.65 * dens)))
        if freeze[t]:
            conf[t] *= 0.15
    return conf


def _isolate_vocal(formant_mid: float, formant_side: float,
                   gain: float = 1.0) -> float:
    """Stereo Mid/Side vocal isolation for one frame.

        residual = max(0, formant(Mid) - gain * formant(Side))   # wide image
        centre   = Mid^2 / (Mid^2 + Side^2)                      # 0 = all side
        vocal    = max(residual, Mid * centre)

    The subtraction alone is not enough on a loud full mix: with anti-phase
    guitars at |x| and a centre vocal at |v| < |x|, Mid carries the vocal while
    Side carries the guitars, so `Mid - Side` clamps to zero exactly when the
    singer enters. Weighting Mid by how centre-dominant the frame is keeps a
    proportional vocal envelope (enough for VAD, onsets and freeze, which all
    work on relative energy) and still cancels pure wide-instrument frames to
    zero, because there Mid itself is ~0.
    """
    residual = formant_mid - gain * formant_side
    centre = (formant_mid * formant_mid
              / (formant_mid * formant_mid + formant_side * formant_side + 1e-18))
    vocal = max(residual, formant_mid * centre)
    return max(0.0, vocal)


def _frames_from_envelopes(centres: Sequence[float], form_env: Sequence[float],
                           rms_env: Sequence[float], cfg: AlignConfig,
                           hop_s: float,
                           vad_thr: Optional[float] = None,
                           floor: Optional[float] = None,
                           mid_env: Optional[Sequence[float]] = None,
                           side_env: Optional[Sequence[float]] = None,
                           cons_env: Optional[Sequence[float]] = None,
                           t_mono: Optional[Sequence[float]] = None) -> List[FrameFeat]:
    """Envelopes -> FrameFeat list (identical rules for live and offline)."""
    onsets = _onset_curve(list(form_env), cfg, hop_s)
    if vad_thr is None or floor is None:
        vad_thr, floor = _vad_threshold(form_env)
    vad, freeze = _vad_freeze(vad_thr, form_env, cfg, hop_s)
    conf = _confidence_curve(form_env, onsets, freeze, floor)
    out = []
    for k in range(len(centres)):
        out.append(FrameFeat(
            t=centres[k], rms=rms_env[k], formant=form_env[k],
            onset=onsets[k], vad=vad[k], freeze=freeze[k],
            confidence=conf[k],
            formant_mid=mid_env[k] if mid_env else form_env[k],
            formant_side=side_env[k] if side_env else 0.0,
            consonant=cons_env[k] if cons_env else 0.0,
            t_mono=t_mono[k] if t_mono else 0.0,
        ))
    return out


def _window_exact(frame_s: float, cfg: AlignConfig, sr: int):
    """Rounding-safe (n_win, hop_s, frame_s, lookahead_s) for the analysis grid."""
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
    return n_win, n_hop, hop_s, frame_s, lookahead_s


def _band_windows(signals: Sequence[Sequence[float]], win: Sequence[float],
                  n_win: int, n_hop: int, sr: int,
                  cal: float) -> Tuple[List[List[float]], List[float]]:
    """Windowed RMS per signal over the analysis grid + window-centre stamps."""
    n = len(signals[0]) if signals else 0
    envs: List[List[float]] = [[] for _ in signals]
    centres: List[float] = []
    i = 0
    while i + n_win <= n or (i < n and not centres):
        end = min(n, i + n_win)
        # Zero-pad a trailing partial frame so we don't invent lookahead
        # past EOF — we just analyse what exists.
        wsum = 0.0
        accs = [0.0] * len(signals)
        for k in range(n_win):
            w = win[k]
            idx = i + k
            wsum += w
            for si, sig in enumerate(signals):
                x = sig[idx] if idx < n else 0.0
                accs[si] += w * x * x
        wsum = wsum or 1.0
        for si in range(len(signals)):
            envs[si].append(math.sqrt(max(0.0, accs[si] / wsum)))
        centre = (i + 0.5 * min(n_win, end - i)) / float(sr)
        centres.append(centre + cal)
        i += n_hop
        if i >= n:
            break
    return envs, centres


def extract_features(samples: Sequence[float], sr: int,
                     cfg: Optional[AlignConfig] = None) -> FeatureTrack:
    """Offline feature extractor (mono).

    Lookahead is the half-window past the frame centre, capped at 150 ms.
    Frame timestamps sit on the window centre plus `calibration_offset_s`.
    """
    cfg = cfg or AlignConfig()
    if sr != TARGET_SR and sr > TARGET_SR:
        sr, samples = _resample_mono(samples, sr, TARGET_SR)
    samples = list(samples)
    frame_s = _trim_frame_s(cfg.frame_s, cfg.max_lookahead_s)
    n_win, n_hop, hop_s, frame_s, lookahead_s = _window_exact(
        frame_s, cfg, sr)
    cal = cfg.calibration_offset_s

    # Vocal-band signal (task 3): formant energy, not full-band RMS.
    vocal = _one_pole_lp(_one_pole_hp(samples, sr, cfg.formant_lo_hz),
                         sr, cfg.formant_hi_hz)
    win = _hamming(n_win)
    (rms_env, form_env), centres = _band_windows(
        [samples, vocal], win, n_win, n_hop, sr, cal)
    vad_thr, floor = _vad_threshold(form_env)
    frames = _frames_from_envelopes(centres, form_env, rms_env, cfg, hop_s,
                                    vad_thr=vad_thr, floor=floor)
    return FeatureTrack(
        sr=sr, hop_s=hop_s, frame_s=frame_s,
        lookahead_s=lookahead_s,
        calibration_offset_s=cal,
        receptive_delay_s=lookahead_s,
        frames=frames,
    )


def extract_features_stereo(left: Sequence[float], right: Sequence[float],
                            sr: int,
                            cfg: Optional[AlignConfig] = None) -> FeatureTrack:
    """Offline feature extractor with stereo Mid/Side vocal isolation (v3.0).

    In a full mix (shoegaze guitars, snares, synths) the 300 Hz–3.4 kHz band is
    flooded with energy, so a mono band-pass VAD believes the singer is always
    singing. A lead vocal sits dead-centre, while panned guitars/pads/cymbals
    live in the side channel, so:

        Mid  = 0.5 * (L + R)     (lead vocal)
        Side = 0.5 * (L - R)     (wide instruments)

        isolated = max(0, formant_rms(Mid) - gain * formant_rms(Side))

    VAD, onsets, freeze and confidence are all driven by `isolated`; a
    2.5–5.5 kHz transient band on Mid keeps consonant attacks (sibilance,
    plosives) usable. Falls back to Mid alone if the isolation nulls everything
    (a mono-ish mix, or a fully out-of-phase lead).
    """
    cfg = cfg or AlignConfig()
    if sr != TARGET_SR and sr > TARGET_SR:
        _rs = sr
        sr, left = _resample_mono(left, _rs, TARGET_SR)
        _srx, right = _resample_mono(right, _rs, TARGET_SR)
    left, right = list(left), list(right)
    n = min(len(left), len(right))
    left, right = left[:n], right[:n]
    mid = [0.5 * (left[i] + right[i]) for i in range(n)]
    side = [0.5 * (left[i] - right[i]) for i in range(n)]

    frame_s = _trim_frame_s(cfg.frame_s, cfg.max_lookahead_s)
    n_win, n_hop, hop_s, frame_s, lookahead_s = _window_exact(
        frame_s, cfg, sr)
    cal = cfg.calibration_offset_s

    mid_band = _one_pole_lp(_one_pole_hp(mid, sr, cfg.formant_lo_hz),
                            sr, cfg.formant_hi_hz)
    side_band = _one_pole_lp(_one_pole_hp(side, sr, cfg.formant_lo_hz),
                             sr, cfg.formant_hi_hz)
    cons_band = _one_pole_lp(_one_pole_hp(mid, sr, cfg.cons_lo_hz),
                             sr, cfg.cons_hi_hz)

    win = _hamming(n_win)
    (rms_env, mid_env, side_env, cons_env), centres = _band_windows(
        [mid, mid_band, side_band, cons_band], win, n_win, n_hop, sr, cal)
    gain = cfg.side_cancel_gain
    isolated = [_isolate_vocal(m, s, gain)
                for m, s in zip(mid_env, side_env)]
    if max(isolated) <= 1e-9:
        isolated = list(mid_env)          # fail-soft: nothing survived isolation
    vad_thr, floor = _vad_threshold(isolated)
    frames = _frames_from_envelopes(centres, isolated, rms_env, cfg, hop_s,
                                    vad_thr=vad_thr, floor=floor,
                                    mid_env=mid_env, side_env=side_env,
                                    cons_env=cons_env)
    return FeatureTrack(
        sr=sr, hop_s=hop_s, frame_s=frame_s,
        lookahead_s=lookahead_s,
        calibration_offset_s=cal,
        receptive_delay_s=lookahead_s,
        frames=frames,
        stereo=True,
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

# A mono capture, a stereo (sr, left, right) capture, a WAV path, or features.
AudioIn = Union[str, Tuple[int, Sequence[float]], FeatureTrack]


def _features_from(audio: AudioIn, cfg: AlignConfig) -> FeatureTrack:
    """Accepts a WAV path, (sr, mono), (sr, left, right) or a ready FeatureTrack."""
    if isinstance(audio, FeatureTrack):
        return audio
    if isinstance(audio, str):
        try:
            sr, left, right = load_wav_stereo(audio)
            if any(abs(a - b) > 1e-9 for a, b in zip(left[::37], right[::37])):
                return extract_features_stereo(left, right, sr, cfg)
            return extract_features(left, sr, cfg)
        except Exception:
            sr, samples = load_wav(audio)
            return extract_features(samples, sr, cfg)
    if len(audio) == 3:
        sr, left, right = audio
        return extract_features_stereo(left, right, sr, cfg)
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
# LIVE AUDIO BRIDGE  (v3.0: PipeWire / PulseAudio monitor capture, stdlib only)
# ===========================================================================
# Everything here is fail-soft by construction: if the capture binary is
# missing, the monitor sink cannot be opened, permissions are denied, or the
# user passed --no-audio-sync, `LiveAudioTracker.running` stays False and the
# display runs the v3.0 prosodic heuristic engine alone — no crash, no lag.

def _pulse_monitor_candidates():
    """Best-effort monitor source names for parec (PulseAudio / PipeWire-pulse)."""
    cands = ['@DEFAULT_MONITOR@']
    for cmd in (['pactl', 'get-default-sink'], ['pactl', 'info']):
        if not shutil.which('pactl'):
            break
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=1.5).stdout
        except Exception:
            continue
        for line in out.splitlines():
            if cmd[-1] != 'info' or 'Default Sink' in line or 'Default Source' in line:
                name = line.strip().split(':', 1)[-1].strip() if ':' in line else ''
                if name and name != 'n/a':
                    cands.append(name if name.endswith('.monitor') else name + '.monitor')
    if shutil.which('pactl'):
        try:
            out = subprocess.run(['pactl', 'list', 'short', 'sources'],
                                 capture_output=True, text=True, timeout=1.5).stdout
            for line in out.splitlines():
                parts = line.split('\t')
                if len(parts) > 1 and parts[1].endswith('.monitor'):
                    cands.append(parts[1])
        except Exception:
            pass
    cands.append('@DEFAULT_SINK@.monitor')
    seen, uniq = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _active_player_monitor(hint=None):
    """Monitor source of the sink the given player is currently on (best effort)."""
    if not hint or not shutil.which('pactl'):
        return None
    sink_idx = None
    try:
        out = subprocess.run(['pactl', 'list', 'sink-inputs'], capture_output=True,
                             text=True, timeout=1.5).stdout
    except Exception:
        return None
    matched = False
    for line in out.splitlines():
        low = line.lower()
        if any(k in low for k in ('application.name',
                                  'application.process.binary')):
            matched = hint.lower() in low or matched
        if matched and low.strip().startswith('sink:'):
            try:
                sink_idx = int(line.split(':', 1)[1].strip())
            except (IndexError, ValueError):
                pass
            break
    if sink_idx is None:
        return None
    try:
        out = subprocess.run(['pactl', 'list', 'short', 'sinks'],
                             capture_output=True, text=True, timeout=1.5).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split('\t')
        if len(parts) > 1 and parts[0].strip() == str(sink_idx):
            name = parts[1].strip()
            return name if name.endswith('.monitor') else name + '.monitor'
    return None


def live_capture_commands(device=None, rate=16000, player=None):
    """Ordered capture argv candidates for this machine ([] = no backend).

    PipeWire (`pw-record` / `pw-cat --record`) and PulseAudio (`parec`) both
    publish the sink monitor as a recordable source; raw s16le stereo PCM on
    stdout keeps this dependency-free. When `player` is given, the sink that
    player is currently on is tried before the default monitor.
    """
    if device:
        targets = [device]
    else:
        targets = []
        try:
            pm = _active_player_monitor(player)
        except Exception:
            pm = None
        if pm:
            targets.append(pm)
        targets.extend([c for c in _pulse_monitor_candidates() if c not in targets])
    cmds = []
    if shutil.which('pw-record'):
        for dev in targets:
            cmds.append(['pw-record', '--rate', str(rate), '--channels', '2',
                         '--format', 's16', '--target', dev, '-'])
    elif shutil.which('pw-cat'):
        for dev in targets:
            cmds.append(['pw-cat', '--record', '--rate', str(rate),
                         '--channels', '2', '--format', 's16',
                         '--target', dev, '-'])
    if shutil.which('parec'):
        for dev in targets:
            cmds.append(['parec', '--format=s16le', '--rate=%d' % rate,
                         '--channels=2', '--device', dev, '--raw'])
    return cmds


class StreamingFeatureExtractor:
    """Causal, incremental Mid/Side feature extraction for live capture.

    Same band plan, VAD, onset, freeze and confidence rules as the offline
    extractor (`_frames_from_envelopes`), but fed one hop at a time from
    persistent filter state, so a live capture costs ~1 % of one core and
    introduces no lookahead beyond the STFT window itself. The noise floor and
    onset scale are re-estimated on a rolling window, which is what lets the
    engine follow a quiet verse after a loud chorus.
    """

    STATS_EVERY = 42          # frames between rolling-statistic refreshes (~0.4 s)

    def __init__(self, sr: int = TARGET_SR, cfg: Optional[AlignConfig] = None):
        self.cfg = cfg or AlignConfig()
        self.sr = sr
        frame_s = _trim_frame_s(self.cfg.frame_s, self.cfg.max_lookahead_s)
        n_win, n_hop, hop_s, frame_s, lookahead_s = _window_exact(
            frame_s, self.cfg, sr)
        self.n_win, self.n_hop = n_win, n_hop
        self.hop_s, self.frame_s = hop_s, frame_s
        self.lookahead_s = lookahead_s
        self.win = _hamming(n_win)
        self.n_seen = 0                       # samples consumed
        self.n_frames = 0
        self._since_hop = 0
        self._mid = deque(maxlen=n_win)       # formant band, Mid
        self._side = deque(maxlen=n_win)      # formant band, Side
        self._cons = deque(maxlen=n_win)      # consonant band, Mid
        self._rms = deque(maxlen=n_win)       # full-band Mid (diagnostics)
        self._hp_mid = _OnePoleHP(sr, self.cfg.formant_lo_hz)
        self._lp_mid = _OnePoleLP(sr, self.cfg.formant_hi_hz)
        self._hp_side = _OnePoleHP(sr, self.cfg.formant_lo_hz)
        self._lp_side = _OnePoleLP(sr, self.cfg.formant_hi_hz)
        self._hp_cons = _OnePoleHP(sr, self.cfg.cons_lo_hz)
        self._lp_cons = _OnePoleLP(sr, self.cfg.cons_hi_hz)
        self._form_ring = deque(maxlen=int(round(12.0 / max(hop_s, 1e-3))))
        self._side_ring = deque(maxlen=self._form_ring.maxlen)
        self._onsets: List[float] = []
        self._rise_n = max(1, int(round(self.cfg.onset_rise_s / hop_s)))
        self._hang_n = max(1, int(round(self.cfg.vad_hangover_s / hop_s)))
        self._freeze_n = max(1, int(round(self.cfg.vad_freeze_s / hop_s)))
        self._vad_thr = 1.5e-4
        self._floor = 1.5e-4
        self._on_scale = 1.0
        self._last_on = -10**9
        self._silent_run = 0
        self._freeze_run = 0

    def push(self, left: Sequence[float], right: Sequence[float]) -> List[FrameFeat]:
        """Feed a chunk of stereo PCM; returns any newly completed frames."""
        cfg = self.cfg
        gain = cfg.side_cancel_gain
        out: List[FrameFeat] = []
        n = min(len(left), len(right))
        for i in range(n):
            mid = 0.5 * (left[i] + right[i])
            side = 0.5 * (left[i] - right[i])
            mb = self._lp_mid(self._hp_mid(mid))
            sb = self._lp_side(self._hp_side(side))
            cb = self._lp_cons(self._hp_cons(mid))
            self._mid.append(mb)
            self._side.append(sb)
            self._cons.append(cb)
            self._rms.append(mid)
            self.n_seen += 1
            self._since_hop += 1
            if self._since_hop >= self.n_hop:
                self._since_hop = 0
                out.append(self._emit_frame(gain))
        return out

    def _win_rms(self, dq) -> float:
        acc = 0.0
        w = self.win
        # Right-aligned window: the newest sample is the frame's trailing edge.
        vals = list(dq)
        off = len(vals) - self.n_win
        for k in range(self.n_win):
            idx = k + off
            x = vals[idx] if 0 <= idx < len(vals) else 0.0
            acc += w[k] * x * x
        wsum = sum(w) or 1.0
        return math.sqrt(max(0.0, acc / wsum))

    def _emit_frame(self, gain: float) -> FrameFeat:
        hop_s = self.hop_s
        form_mid = self._win_rms(self._mid)
        form_side = self._win_rms(self._side)
        cons = self._win_rms(self._cons)
        rms = self._win_rms(self._rms)
        isolated = _isolate_vocal(form_mid, form_side, gain)
        if not self._form_ring:
            isolated = max(isolated, 1e-9)
        self._form_ring.append(isolated if isolated > 0.0 else 1e-9)
        self._side_ring.append(form_side)
        t = self.n_frames * hop_s + self.lookahead_s + self.cfg.calibration_offset_s
        self.n_frames += 1

        # --- onset: rise rate over ~55 ms, normalised on a rolling median ---
        rise_n = self._rise_n
        ring = self._form_ring
        if len(ring) > rise_n:
            rise = max(0.0, isolated - ring[-rise_n - 1])
        else:
            rise = max(0.0, isolated - ring[0])
        self._onsets.append(rise)
        if len(self._onsets) > ring.maxlen:
            self._onsets.pop(0)

        if self.n_frames % self.STATS_EVERY == 0 or self.n_frames == 1:
            vals = sorted(self._form_ring)
            if vals:
                self._floor = max(1.5e-4, vals[max(0, int(0.15 * (len(vals) - 1)))])
                median = vals[len(vals) // 2]
                self._vad_thr = max(1.5e-4, min(self._floor * 3.4,
                                                median * 0.40 + 1e-6))
            pos = sorted(x for x in self._onsets if x > 0.0)
            if pos:
                med = pos[len(pos) // 2]
                self._on_scale = med * 2.8 if med > 1e-9 else (pos[-1] or 1.0)
            else:
                self._on_scale = 1.0
        onset = max(0.0, min(1.0, rise / max(self._on_scale, 1e-9)))

        # --- VAD with hangover, freeze after a long inactive run ---
        active = isolated > self._vad_thr
        if active:
            self._last_on = self.n_frames
        vad = (self.n_frames - self._last_on) <= self._hang_n
        if vad:
            self._silent_run = 0
            self._freeze_run = 0
        else:
            self._silent_run += 1
            self._freeze_run += 1
        freeze = self._freeze_run >= self._freeze_n

        # --- causal confidence (mirrors _confidence_curve, backward window) ---
        snr = isolated / (self._floor + 1e-6)
        snr_term = 1.0 / (1.0 + math.exp(-(snr - 4.0)))
        recent = self._onsets[-13:]
        dens = (sum(1 for x in recent if x / max(self._on_scale, 1e-9) > 0.45)
                / max(1, len(recent)))
        conf = max(0.0, min(1.0, snr_term * (1.0 - 0.65 * dens)))
        if freeze:
            conf *= 0.15

        return FrameFeat(t=t, rms=rms, formant=isolated, onset=onset, vad=vad,
                         freeze=freeze, confidence=conf,
                         formant_mid=form_mid, formant_side=form_side,
                         consonant=cons, t_mono=0.0)

    def track(self, frames: Optional[Sequence[FrameFeat]] = None,
              stereo: bool = True) -> FeatureTrack:
        """Wrap (a slice of) this stream's frames as a FeatureTrack."""
        return FeatureTrack(
            sr=self.sr, hop_s=self.hop_s, frame_s=self.frame_s,
            lookahead_s=self.lookahead_s,
            calibration_offset_s=self.cfg.calibration_offset_s,
            receptive_delay_s=self.lookahead_s,
            frames=list(frames) if frames is not None else [],
            stereo=stereo,
        )


class LiveAudioTracker:
    """Zero-dependency live audio bridge for PipeWire / PulseAudio (v3.0).

    A daemon thread runs `pw-record` (PipeWire) or `parec` (PulseAudio) against
    the default output monitor, converts the raw s16le stereo stream to
    Mid/Side features with `StreamingFeatureExtractor`, and keeps a rolling
    12 s ring of song-time-stamped `FrameFeat` frames plus a per-track
    FeatureTrack cache. That cache is what turns a replay of a track into a
    real acoustic alignment (`build_word_events(..., audio=...)`, Tier 1.5):
    the engine that used to be dead code during live playback now runs on the
    first pass (rolling causal phase-lock) and on every replay (full align).
    """

    MAX_CACHED_TRACKS = 2
    MAX_CACHE_S = 480.0          # keep at most 8 minutes of features per track

    def __init__(self, cfg: Optional[AlignConfig] = None, enabled: bool = True,
                 device: Optional[str] = None, player: Optional[str] = None):
        self.cfg = cfg or AlignConfig()
        self.enabled = bool(enabled)
        self.device = device
        self.player = player
        self.running = False
        self.note = 'disabled' if not enabled else 'not started'
        self.backend = None
        self._proc = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._ext: Optional[StreamingFeatureExtractor] = None
        self._capture_start = 0.0
        self._capture_bytes = 0
        # Buffers are hard-capped deques: a stalled or fast-forwarding capture
        # device can never grow the process (or the per-feed work) without bound.
        _fs = _trim_frame_s(self.cfg.frame_s, self.cfg.max_lookahead_s)
        _nw, _nh, _hop, _f, _la = _window_exact(_fs, self.cfg, self.cfg.capture_rate)
        self._hop = max(_hop, 1e-3)
        self._ring_maxlen = max(16, int(round(self.cfg.ring_s / self._hop)) + 8)
        self._track_maxlen = max(64, int(round(self.MAX_CACHE_S / self._hop)))
        self.ring: 'deque[FrameFeat]' = deque(maxlen=self._ring_maxlen)
        self._captures: 'OrderedDict[str, FeatureTrack]' = OrderedDict()
        self._track_key: Optional[str] = None
        self._track_frames: 'deque[FrameFeat]' = deque(maxlen=self._track_maxlen)
        self._anchor_pos: Optional[float] = None
        self._anchor_t: Optional[float] = None

    # ---------------- lifecycle ----------------
    @staticmethod
    def available(device: Optional[str] = None,
                  player: Optional[str] = None) -> List[List[str]]:
        return live_capture_commands(device, player=player)

    def start(self) -> bool:
        """Spawn the capture thread. Never blocks, never raises."""
        if not self.enabled:
            self.note = 'disabled (--no-audio-sync)'
            return False
        if self.running or (self._thread is not None and self._thread.is_alive()):
            return self.running
        cmds = live_capture_commands(self.device, self.cfg.capture_rate,
                                     self.player)
        if not cmds:
            self.note = ('no audio capture backend '
                         '(install pipewire pw-record or pulseaudio parec)')
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._thread_main, args=(cmds,),
                                        name='wordsync-audio', daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        t = self._thread
        if t is not None and t.is_alive():
            try:
                t.join(timeout=0.4)
            except Exception:
                pass
        self._thread = None
        self.running = False

    def _thread_main(self, cmds):
        for argv in cmds:
            if self._stop.is_set():
                return
            try:
                proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=0)
            except Exception:
                continue
            chunk = None
            deadline = time.monotonic() + 0.60
            while time.monotonic() < deadline and not self._stop.is_set():
                try:
                    chunk = proc.stdout.read(4096)
                except Exception:
                    chunk = None
                if chunk:
                    break
                if proc.poll() is not None:
                    break
                time.sleep(0.02)
            if not chunk:
                try:
                    proc.terminate()
                    proc.wait(timeout=0.4)
                except Exception:
                    pass
                continue
            self._proc = proc
            self.backend = argv[0]
            self.running = True
            self.note = 'capturing %d Hz stereo via %s' % (self.cfg.capture_rate,
                                                           argv[0])
            try:
                self._capture_loop(proc, chunk)
            except Exception:
                pass
            self.running = False
            self.note = 'capture stopped'
            return

    def _capture_loop(self, proc, first: bytes):
        ext = StreamingFeatureExtractor(self.cfg.capture_rate, self.cfg)
        self._ext = ext
        # The samples already in `first` were captured *before* now.
        self._capture_start = (time.monotonic()
                               - (len(first) // 4) / float(self.cfg.capture_rate))
        buf = first
        while not self._stop.is_set():
            if len(buf) >= 4:
                n = len(buf) // 4 * 4
                try:
                    samples = array('h')
                    samples.frombytes(buf[:n])
                except Exception:
                    samples = array('h')
                if sys.byteorder == 'big':
                    samples.byteswap()
                left = [samples[i] / 32768.0 for i in range(0, len(samples) - 1, 2)]
                right = [samples[i] / 32768.0 for i in range(1, len(samples), 2)]
                self._feed(ext, left, right)
                buf = buf[n:]
                self._capture_bytes += n
            try:
                chunk = proc.stdout.read(4096)
            except Exception:
                chunk = b''
            if not chunk:
                if proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            buf += chunk
            if len(buf) > 1 << 20:            # never let a stalled reader grow
                buf = buf[-65536:]

    def _feed(self, ext, left, right):
        frames = ext.push(left, right)
        if not frames:
            return
        # Streaming frames carry capture-relative time; stamp them with the
        # capture clock, then map onto song time through the playback anchor.
        for f in frames:
            f.t_mono = self._capture_start + f.t
            f.t = self._song_time(f.t_mono)
        with self._lock:
            self.ring.extend(frames)
            if self._track_key is not None:
                self._track_frames.extend(frames)

    def _song_time(self, mono_t: float) -> float:
        if self._anchor_pos is None or self._anchor_t is None:
            return mono_t - (self._capture_start or mono_t)
        return self._anchor_pos + (mono_t - self._anchor_t)

    # ---------------- playback anchor / track cache ----------------
    def update_anchor(self, song_pos: Optional[float], t_mono: float):
        if song_pos is None:
            return
        self._anchor_pos = float(song_pos)
        self._anchor_t = float(t_mono)

    def feed_frames(self, frames: Sequence[FrameFeat], song_time: bool = True):
        """Inject already-extracted frames (harness, or an external capture tap).

        Real capture goes through the pw-record/parec thread; this entry point
        accepts frames that are already stamped in song time (or in capture time
        when `song_time` is False), which is also how a synthetic capture device
        is simulated in the regression suite.
        """
        frames = list(frames)
        if not frames:
            return
        with self._lock:
            for f in frames:
                if not song_time:
                    f.t_mono = f.t
                    f.t = self._song_time(f.t_mono)
                self.ring.append(f)
                if self._track_key is not None:
                    self._track_frames.append(f)

    def begin_track(self, song_key: Optional[str], duration: float = 0.0):
        """Start (or restart) the per-track feature capture for `song_key`."""
        if song_key == self._track_key:
            return
        self.end_track()
        self._track_key = song_key
        self._track_frames.clear()

    def end_track(self):
        """Freeze the current capture into the replay cache (LRU, bounded)."""
        key = self._track_key
        frames = list(self._track_frames)
        self._track_key = None
        self._track_frames = deque(maxlen=self._track_maxlen)
        if not key or len(frames) < 8:
            return
        ext = self._ext
        track = FeatureTrack(
            sr=ext.sr if ext else self.cfg.capture_rate,
            hop_s=ext.hop_s if ext else self.cfg.hop_s,
            frame_s=ext.frame_s if ext else self.cfg.frame_s,
            lookahead_s=ext.lookahead_s if ext else 0.0125,
            calibration_offset_s=self.cfg.calibration_offset_s,
            receptive_delay_s=ext.lookahead_s if ext else 0.0125,
            frames=frames, stereo=True)
        self._captures[key] = track
        self._captures.move_to_end(key)
        while len(self._captures) > self.MAX_CACHED_TRACKS:
            self._captures.popitem(last=False)

    def cached_track(self, song_key: Optional[str]) -> Optional[FeatureTrack]:
        """Features captured during a previous play-through of this track."""
        if not song_key or not self.running:
            return None
        return self._captures.get(song_key)

    def capture_summary(self) -> str:
        return self.note

    # ---------------- causal queries (song time) ----------------
    def track_view(self, t0: float, t1: float,
                   stereo: bool = True) -> Optional[FeatureTrack]:
        with self._lock:
            frames = [f for f in self.ring if t0 <= f.t <= t1]
        if len(frames) < 4:
            return None
        ext = self._ext
        return FeatureTrack(
            sr=ext.sr if ext else self.cfg.capture_rate,
            hop_s=ext.hop_s if ext else self.cfg.hop_s,
            frame_s=ext.frame_s if ext else self.cfg.frame_s,
            lookahead_s=ext.lookahead_s if ext else 0.0125,
            calibration_offset_s=0.0,
            receptive_delay_s=ext.lookahead_s if ext else 0.0125,
            frames=frames, stereo=stereo)

    def frames_between(self, t0: float, t1: float) -> List[FrameFeat]:
        with self._lock:
            return [f for f in self.ring if t0 <= f.t <= t1]

    def vad_any(self, t0: float, t1: float) -> bool:
        return any(f.vad for f in self.frames_between(t0, t1))

    def first_vocal_after(self, t: float, limit: float) -> Optional[float]:
        """First voiced centre-channel frame in (t, t+limit], or None."""
        best = None
        for f in self.frames_between(t, t + limit):
            if f.vad:
                best = f.t
                break
        return best

    def strongest_onset(self, t: float, radius: float,
                        min_strength: float = 0.0) -> Optional[float]:
        best_t, best = None, min_strength
        for f in self.frames_between(t - radius, t + radius):
            if f.onset > best:
                best_t, best = f.t, f.onset
        return best_t

    def silence_run(self, t: float) -> float:
        """Seconds of continuous centre-channel silence ending at `t`."""
        frames = self.frames_between(t - self.cfg.ring_s, t)
        if not frames:
            return 0.0
        run_start = None
        for f in reversed(frames):
            if f.vad:
                break
            run_start = f.t
        if run_start is None:
            return 0.0
        hop = self._ext.hop_s if self._ext else self.cfg.hop_s
        return max(0.0, t - run_start + hop)

    def vocal_energy(self, t: float) -> float:
        best = 0.0
        for f in self.frames_between(t - 0.1, t + 0.1):
            best = max(best, f.formant)
        return best


class LivePhaseLock:
    """Causal phase-locking of the estimated timeline onto the real vocal.

    Runs only for Tier-2 (cadence-model) events and only on events that have not
    been announced yet, so a displayed word can never jump backwards. Four
    bounded behaviours, all fail-soft (any error disables live refinement and
    the schedule continues unchanged):

      * Vocal Entry Gate   — a word scheduled while the centre channel is silent
                             is held (ceiling +350 ms) until the real vocal
                             onset rises, instead of popping up during a breath
                             or a guitar intro.
      * Causal Onset Snap  — a strong isolated-vocal onset within ±160 ms of the
                             next transition advances the word onto the attack.
      * Post-Phrase Freeze — after the line's last word, >380 ms of
                             centre-channel silence releases any pending hold so
                             the interlude/clear happens on schedule.
      * Rolling Re-Align   — every 150 ms the remaining tokens of the active
                             line are re-aligned against the captured features
                             (fast local `refine_word_events`), so fast/slow
                             rhythmic deviations self-correct in real time.
    """

    def __init__(self, cfg: Optional[AlignConfig] = None, enabled: bool = True):
        self.cfg = cfg or AlignConfig()
        self.enabled = bool(enabled and self.cfg.live_enabled)
        self.meta: dict = {}
        self.events: List[Tuple[float, str]] = []
        self.next_refine = 0.0
        self.cpu_ema = 0.0
        self.stats = {'gates': 0, 'snaps': 0, 'releases': 0, 'refines': 0}
        self._last_refine_line = -1

    def reset(self, events=(), meta=None):
        self.events = list(events)
        self.meta = dict(meta or {})
        self.next_refine = 0.0
        self.cpu_ema = 0.0
        self._last_refine_line = -1

    # ---- helpers ----
    def _flag(self, key, idx, default=False):
        lst = self.meta.get(key)
        if lst and 0 <= idx < len(lst):
            return lst[idx]
        return default

    def _line_of(self, idx):
        return self._flag('line_id', idx, -1)

    def _line_bounds(self, line_id, base_starts, idx):
        """(start, end) of the line's window, from the event list."""
        if line_id < 0:
            t = base_starts[idx] if 0 <= idx < len(base_starts) else 0.0
            return t, t + 1.0
        first = idx
        while first - 1 >= 0 and self._line_of(first - 1) == line_id:
            first -= 1
        last = idx
        while last + 1 < len(self.meta.get('line_id', [])) and                 self._line_of(last + 1) == line_id:
            last += 1
        start = base_starts[first] if 0 <= first < len(base_starts) else 0.0
        end = self._flag('line_end', idx, start + 6.0)
        return start, max(end, start + 0.2)

    def _re_align(self, base_starts, active_idx, pos, tracker):
        """Fast local re-alignment of the active line's remaining tokens."""
        line_id = self._line_of(active_idx)
        if line_id < 0:
            return {}
        idxs = [i for i in range(active_idx + 1, len(self.events))
                if self._line_of(i) == line_id and self.events[i][1]
                and self._flag('estimated', i)]
        if not idxs:
            return {}
        _start, end = self._line_bounds(line_id, base_starts, active_idx)
        t0 = max(0.0, pos - 0.30)
        t1 = min(end + 0.35, max(base_starts[idxs[-1]] + 0.35, pos + 0.5))
        if t1 <= t0 + 0.1:
            return {}
        track = tracker.track_view(t0, t1)
        if track is None:
            return {}
        cfg = AlignConfig(**{**self.cfg.__dict__, 'beam_width': self.cfg.live_beam_width})
        local = [(base_starts[i], self.events[i][1]) for i in idxs]
        try:
            refined = refine_word_events(local, track, cfg)
        except Exception:
            return {}
        if len(refined) != len(local):
            return {}
        out = {}
        prev_t = base_starts[active_idx] if 0 <= active_idx < len(base_starts) else -1.0
        for k, i in enumerate(idxs):
            t = refined[k][0]
            nxt = base_starts[idxs[k + 1]] if k + 1 < len(idxs) else min(end, t1)
            lo = prev_t + MIN_SHOW
            hi = max(lo, nxt - MIN_SHOW)
            out[i] = min(max(t, lo), hi)
            prev_t = out[i]
        return out

    def update(self, events, base_starts, active_idx, pos, t_mono,
               tracker) -> dict:
        """Return {event_index: new_time} for the upcoming events (may be empty)."""
        out: dict = {}
        if not self.enabled or tracker is None or not tracker.running:
            return out
        if not events or active_idx < -1 or active_idx >= len(events):
            return out
        cfg = self.cfg

        # ---- post-phrase VAD freeze & early clear ----
        # Once the active word is the line's last and the centre channel has
        # been silent for freeze_clear_s while instruments continue, nothing
        # further is being sung on this line: release holds so the rest/clear
        # (and the next line's first word) follow the schedule.
        silence = tracker.silence_run(pos)
        line_id = self._line_of(active_idx) if active_idx >= 0 else -2
        later_same_line = [i for i in range(active_idx + 1, len(events))
                           if self._line_of(i) == line_id]
        released = (active_idx >= 0 and silence >= cfg.freeze_clear_s
                    and not later_same_line
                    and self._flag('estimated', active_idx))
        if released:
            self.stats['releases'] += 1

        for idx in range(active_idx + 1, min(len(events), active_idx + 4)):
            if not self._flag('estimated', idx):
                continue
            t = base_starts[idx]
            if idx <= active_idx:
                continue
            t_now = base_starts[idx]
            # ---- 1. vocal entry gate ----
            gate = 0.0
            if not released and not tracker.vad_any(t_now - 0.08, t_now + 0.06):
                entry = tracker.first_vocal_after(t_now, cfg.entry_gate_max_s)
                gate = (entry - t_now) if entry is not None else cfg.entry_gate_max_s
                gate = max(0.0, min(cfg.entry_gate_max_s, gate))
                if gate > 1e-3:
                    self.stats['gates'] += 1
            cand = t_now + gate
            # ---- 2. causal onset snap ----
            snap = tracker.strongest_onset(cand, cfg.onset_snap_s,
                                           cfg.onset_snap_min)
            if snap is not None and abs(snap - cand) <= cfg.onset_snap_s + 1e-9:
                cand = snap
                self.stats['snaps'] += 1
            # ---- clamp inside the neighbours / line window ----
            prev_t = base_starts[idx - 1] if idx > 0 else None
            lo = pos - 0.02
            if prev_t is not None:
                lo = max(lo, prev_t + MIN_SHOW)
            hi = base_starts[idx + 1] - MIN_SHOW if idx + 1 < len(base_starts) else cand
            _ls, le = self._line_bounds(self._line_of(idx), base_starts, idx)
            hi = max(lo, min(hi, le - 1e-3))
            cand = min(max(cand, lo), hi)
            if abs(cand - t) > 1e-3:
                out[idx] = cand

        # ---- 4. rolling re-alignment (self-throttling) ----
        if cfg.live_refine and time.monotonic() >= self.next_refine:
            self.next_refine = time.monotonic() + cfg.live_refine_every_s
            if self.cpu_ema <= cfg.live_refine_budget_s:
                t_start = time.perf_counter()
                try:
                    patch = self._re_align(base_starts, active_idx, pos, tracker)
                except Exception:
                    patch = {}
                self.cpu_ema = 0.7 * self.cpu_ema + 0.3 * (time.perf_counter() - t_start)
                if patch:
                    self.stats['refines'] += 1
                    for i, tv in patch.items():
                        if i in out:
                            # keep the causally-snapped value, bounded by the refine
                            out[i] = min(max(out[i], tv - 0.25), tv + 0.25)
                        else:
                            out[i] = tv
        # Never hand back a non-monotonic schedule: each patched event keeps at
        # least MIN_SHOW after its (possibly already patched) predecessor.
        prev = None
        for i in sorted(out):
            lo = (prev if prev is not None
                  else (base_starts[i - 1] if i > 0 else -1e9)) + MIN_SHOW
            if out[i] < lo:
                out[i] = lo
            prev = out[i]
        return out


# ===========================================================================
# Tiny self-check (imported by --selftest / the harness)
# ===========================================================================

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

    # v3.0 dense frontier: pace follows the densest lines, never a window
    # stretched by a post-phrase instrumental gap.
    dense = [{'start': i * 2.2, 'text': 'we are the dead ones now', 'words': None}
             for i in range(4)]
    b_dense, _, _ = measure_cadence(dense, [2.2] * 4)
    gap = [{'start': i * 6.5, 'text': 'we are the dead ones now', 'words': None}
           for i in range(4)]
    b_gap, _, _ = measure_cadence(gap, [6.5] * 4)
    if not (b_dense < 0.45):
        fails.append('dense lines should anchor a fast articulation pace')
    if not (b_gap < TUNE['pace_max']):
        fails.append('all-gap song should still damp below the pace ceiling')

    # v3.0 prosody helpers.
    core, marks = split_prosody('Wait,')
    if not (core == 'Wait' and marks['boundary']):
        fails.append('split_prosody should flag a trailing comma as a boundary')
    if split_prosody('middle')[1]['boundary']:
        fails.append('split_prosody invented a boundary')
    if not (repeat_key('d-d-death') == 'death' and repeat_key('LA,') == 'la'):
        fails.append('repeat_key should collapse stutters and case')
    if not (anacrusis_pickups(['In', 'the', 'SUFFERING'], [1, 1, 3]) == 2
            and anacrusis_pickups(['Deep', 'song', 'SUFFERING'], [1, 1, 3]) == 0):
        fails.append('anacrusis pickup detector mis-fires')

    # v3.0 stereo isolation: centre vocal kept, pure side image nulled.
    if not (_isolate_vocal(0.30, 0.0) > 0.29 and _isolate_vocal(0.0, 0.30) == 0.0):
        fails.append('mid/side isolation keeps the wrong image')

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
    'audio_sync': True,  # v3.0: live monitor capture + causal phase-locking
    'lyrics_dir': os.path.expanduser('~/.local/share/word_sync/lyrics'),
    'cache_dir': os.path.expanduser('~/.cache/word_sync'),
}

VERSION = '3.0'

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


# ==========================================================================
# v3.0 PROSODIC + RHYTHMIC CADENCE MODEL  (Tier 2.5, zero dependency)
# ==========================================================================
# A single global pace scalar cannot describe a real vocal performance:
# * unstressed function words are rushed pickups into the next content word,
# * punctuation opens audible micro-pauses (caesurae),
# * repeated hooks ("la la la", "yeah yeah yeah", "no no") lock to a beat,
# * a pickup of 1–2 function words before a heavy downbeat note compresses,
# * the phrase's LAST word (and its held note) happens *after* its onset, so it
#   must never be part of the span that spaces the earlier onsets.
# Everything below is stdlib-only and language-agnostic: a compact curated
# word list, not a POS tagger, so it stays pure Python and instant.

# Unstressed English function words (articles, prepositions, conjunctions,
# pronouns, auxiliaries, demonstratives) — spoken as anacruses, not beats.
FUNCTION_WORDS = frozenset("""
the a an to in on of for and but or at by from with as into like than after
under over between through against during without within about above below
i you he she it we they me him her us them my your his its our their
is am are was were be been being do does did done have has had will would
can could should shall may might must that this these those there here
not no nor so if then when while because
""".split())

# ES/FR/PT/DE/IT/RU articles, prepositions and unstressed particles.
FUNCTION_WORDS |= frozenset("""
el la los las un una unos unas de del en con por para sin sobre entre y o u
que se le les lo mi tu su mis tus nos os al
des du une dans sur avec pour par sans sous chez et ou qui ne pas ce cet
cette ces mon ma mes ton ta tes son sa ses
o a os as um uma uns umas do da dos das em no na com sem
der die das den dem des ein eine einen einem einer und oder aber mit von zu
im auf für über unter ohne bei aus nach
il gli di per su tra fra che
и в во не что он на я с со как а то все она так его но да ты к у же вы за
бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну
""".split())

# 0.50x–0.62x: unstressed pickups are spoken quickly into the next content word.
FUNC_COMPRESS = 0.55
# Pre-boundary vowel lengthening on the word *before* punctuation.
BOUNDARY_LENGTHEN = 1.25
# Caesura micro-pause injected after a boundary (scaled by available slack).
CAESURA_MIN_S = 0.14
CAESURA_MAX_S = 0.32
# "In the DEATH…": initial pickups compress harder so the stressed content
# word lands cleanly on the downbeat.
ANACRUSIS_COMPRESS = 0.45
ANACRUSIS_MAX_PICKUPS = 2
# Parenthetical / backing-vocal phrases sit behind the lead.
PARENTHETICAL_LEVEL = 0.62
# Lowered post-phrase threshold (v3.0): when the window is longer than the
# phrase's own articulation by this much, the tail belongs to the instruments,
# so the words lock to their own span and the screen clears early.
TAIL_LOCK_SLACK = 0.80
# Dense-phrase frontier percentile: singers cannot articulate slower than their
# own gapless back-to-back lines, so the pace anchor is the *bottom* of the
# per-line rate distribution, not its median.
DENSE_FRONTIER_PCT = 0.27
# Physical articulation reference (s/syllable). Used only as a whole-song
# inflation guard: a track whose MEDIAN line is slower than this is almost
# certainly gap/sustain-inflated on every line (see measure_cadence).
PACE_ARTIC = 0.55

_STRONG_BOUNDARY_CHARS = ',;:—–?!…)]'
_STRONG_BOUNDARY_TAILS = ('...', '…', ',', ';', ':', '—', '–', '?', '!',
                          ')', ']', '-', '–')


def split_prosody(token):
    """(core, marks) for one raw lyric token.

    `marks` records punctuation that has audible consequences:
      boundary      → a pause follows (',' ';' ':' '—' '–' '-' '...' '…' '?!' ')]')
      question      → '?' (phrase-final rise)
      exclaim       → '!'
      parenthetical → '(backing vocal)' / '[ad lib]' phrase
    """
    marks = {'boundary': False, 'question': False, 'exclaim': False,
             'parenthetical': False}
    t = token.strip()
    if not t:
        return '', marks
    if t[0] in '([':
        marks['parenthetical'] = True
    while t and t[-1] in ')]':
        marks['boundary'] = True
        marks['parenthetical'] = True
        t = t[:-1]
    trailing = ''
    while t and not (t[-1].isalnum() or t[-1] in "'’"):
        trailing = t[-1] + trailing
        t = t[:-1]
    if trailing:
        if '?' in trailing:
            marks['question'] = True
        if '!' in trailing:
            marks['exclaim'] = True
        if (any(c in _STRONG_BOUNDARY_CHARS for c in trailing)
                or trailing in _STRONG_BOUNDARY_TAILS
                or trailing.endswith('…')
                or (trailing.count('.') >= 3)):
            marks['boundary'] = True
    return t, marks


def is_function_word(core):
    """Curated unstressed-particle test (multilingual, no POS tagger)."""
    c = core.lower().strip("'’")
    if not c:
        return False
    if c in FUNCTION_WORDS:
        return True
    # Elided articles/prepositions: l', d', qu', c', j', m' (FR/IT).
    return len(c) <= 2 and c in ('l', 'd', 'j', 'qu', 'c', 'm', 'n', 's', 't')


def repeat_key(raw):
    """Canonical key for isochronous repetition / stutter locking.

    'la' / 'la,' / 'LA' collapse together, and a hyphenated stutter
    ('d-d-death', 'la-la-la') collapses onto the syllable it repeats.
    """
    t = raw.strip().strip('.,;:!?…—–)]').lower()
    if not t:
        return ''
    parts = [p for p in re.split(r'[-–—]', t) if p]
    if len(parts) > 1 and all(parts[-1].startswith(p) or p == parts[-1]
                              for p in parts[:-1]):
        return parts[-1]
    return t


def _iso_keys(cores):
    """Per-token keys used to detect isochronous runs (identical repeats plus
    short alliterative hooks such as 'na na' / 'big bad')."""
    keys = []
    for c, core in enumerate(cores):
        k = repeat_key(core)
        if not k:
            keys.append('')
            continue
        if len(k) <= 6 and syllables(k) <= 2 and not is_function_word(k):
            keys.append('~%s:%d' % (k[0], syllables(k)))   # alliteration bucket
        else:
            keys.append(k)                                  # exact repeat only
    return keys


def isochronous_runs(keys):
    """Maximal runs of >= 2 consecutive equal keys (the 'la la la' detector)."""
    runs = []
    i = 0
    n = len(keys)
    while i < n:
        j = i
        while j + 1 < n and keys[j + 1] and keys[j + 1] == keys[i]:
            j += 1
        if j > i:
            runs.append((i, j))
        i = j + 1
    return runs


def anacrusis_pickups(cores, syls):
    """How many leading words are rushed pickups into a heavy downbeat word
    ('In the DEATH…' → 2). 0 when the line does not open that way."""
    if not cores:
        return 0
    k = 0
    while (k < min(ANACRUSIS_MAX_PICKUPS, len(cores) - 1)
           and is_function_word(cores[k]) and syls[k] <= 1):
        k += 1
    if k == 0:
        return 0
    heavy = syls[k] >= 2 or phone_duration_scale(cores[k]) >= 1.20
    return k if heavy else 0


def _percentile(xs, q):
    """Linear-interpolated percentile of an unsorted list (stdlib only)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def measure_cadence(lines, windows):
    """v3.0 Dense-Frontier & Local-Cadence estimator.

    Returns (base_pace, line_paces, info).

    `base_pace` is the song's True Articulation Pace anchored on the 25–30th
    percentile of the per-line raw syllable rate r_i = window_i / syllables_i
    (the dense-phrase frontier), *not* the median raw window: a 6-syllable line
    sung in ~1.8 s and followed by 4.5 s of guitars has r ≈ 1.05 s/syllable and
    must not drag the song's pace onto the 0.822 s/syllable ceiling.

    `line_paces[i]` adapts to genuine mid-song tempo changes (half-time bridge,
    beat switch, rubato intro) using a 5-line rolling window, but refuses to
    inflate when a single line's window is stretched far beyond its own content
    while the neighbouring lines (or the dense frontier) are fast — that excess
    is a post-phrase instrumental gap or a final-vowel sustain, not slow-motion
    articulation.
    """
    n = len(lines)
    info = {'raw_rate': [None] * n, 'voiced': [None] * n,
            'line_pace': [DEFAULT_PACE] * n, 'base_pace': DEFAULT_PACE,
            'frontier': None, 'inflation_guard': False,
            'frontier_from_real': 0}
    if not lines:
        return DEFAULT_PACE, [], info

    syl = []
    for ln in lines:
        ws = ln.get('words')
        if ws:
            syl.append(max(1, sum(syllables(split_prosody(w)[0])
                                  for _t, w in ws)))
        else:
            syl.append(max(1, sum(syllables(split_prosody(t)[0])
                                  for t in line_words(ln['text']))))

    rates = []
    for i in range(n):
        w = windows[i] if i < len(windows) else 0.0
        s = syl[i]
        if w <= 0.6 or s <= 0:
            rates.append(None)
            continue
        vf = _voiced_fraction(w)
        rates.append((w / s) * vf)
        info['voiced'][i] = vf
    info['raw_rate'] = rates

    # Micro-interjections ("yeah!", "oh!") carry almost no articulation
    # information: drop them once enough multi-word lines exist.
    multi = [i for i in range(n) if syl[i] >= 3]
    pool = multi if len(multi) >= 3 else list(range(n))
    vals = [rates[i] for i in pool if rates[i]]

    # Real per-word timings are *direct* measurements of articulation, so they
    # dominate the frontier when a song mixes Tier 1 and Tier 2 lines.
    for i, ln in enumerate(lines):
        ws = ln.get('words')
        if not ws or len(ws) < 2:
            continue
        span = ws[-1][0] - ws[0][0]
        units = max(1, sum(syllables(split_prosody(w)[0]) for _t, w in ws[:-1]))
        if span > 0.05:
            vals.extend([span / units] * 3)
            info['frontier_from_real'] += 1

    if vals:
        frontier = _percentile(vals, DENSE_FRONTIER_PCT)
        med = _percentile(vals, 0.5)
        # Whole-song inflation guard: only when even the median line is slower
        # than any plausible sustained articulation do we assume every window
        # hides an instrumental gap / held note, and damp toward the physical
        # articulation reference. Below that the trained pace range is kept.
        if med > 1.5 * PACE_ARTIC:
            damped = PACE_ARTIC + 0.3 * max(0.0, frontier - PACE_ARTIC)
            if damped < frontier:
                info['inflation_guard'] = True
                frontier = damped
        info['frontier'] = frontier
        base_pace = min(TUNE['pace_max'], max(TUNE['pace_min'], frontier))
    else:
        base_pace = DEFAULT_PACE
    info['base_pace'] = base_pace

    # ---- per-line local pace (5-line rolling window) ----
    W = 2
    for i in range(n):
        r_i = rates[i]
        if r_i is None:
            info['line_pace'][i] = base_pace
            continue
        loc = [rates[j] for j in range(max(0, i - W), min(n, i + W + 1))
               if rates[j] is not None]
        nb = [rates[j] for j in range(max(0, i - W), min(n, i + W + 1))
              if j != i and rates[j] is not None]
        if not loc:
            info['line_pace'][i] = base_pace
            continue
        # Lower half of the local window: a gap-robust floor.
        loc_sorted = sorted(loc)
        lower = loc_sorted[:max(1, (len(loc_sorted) + 1) // 2)]
        local_low = _percentile(lower, 0.5)

        def _similar(a):
            return a is not None and abs(a - r_i) <= 0.35 * r_i

        # Genuine tempo changes are CONTIGUOUS (half-time bridge, beat switch,
        # chorus after a rap verse), while gap/sustain-inflated windows are
        # interleaved with dense lines. So measure the maximal contiguous run of
        # similar-rate lines around i, not just how many similar ones exist.
        run = 1
        j = i - 1
        while j >= 0 and _similar(rates[j]):
            run += 1
            j -= 1
        j = i + 1
        while j < n and _similar(rates[j]):
            run += 1
            j += 1
        sim_rates = [x for x in nb if _similar(x)]
        local_ref = _percentile(sim_rates, 0.5) if sim_rates else local_low
        spike = r_i > 1.6 * base_pace
        if spike and run < 2:
            # single line stretched far past its own content: the excess is a
            # post-phrase instrumental gap or a final-vowel sustain — do NOT
            # inflate line_pace[i] (the Panchiko / shoegaze archetype).
            pace_i = base_pace
        elif spike and run == 2:
            # two consecutive stretched lines: mostly gap, mildly local.
            pace_i = 0.35 * local_ref + 0.65 * base_pace
        else:
            # Dense line, or a real tempo plateau: adapt within 1–2 lines.
            pace_i = 0.60 * local_ref + 0.40 * base_pace
        info['line_pace'][i] = min(TUNE['pace_max'],
                                   max(TUNE['pace_min'], pace_i))
        info.setdefault('run_len', []).append(run)
    return base_pace, list(info['line_pace']), info


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


def estimate_line(words, line_start, line_end, pace=DEFAULT_PACE, line_pace=None):
    """[(time, word), ...] for one line, plus a trailing (time, '') when a genuine rest follows.
    `pace` is the song's measured articulation cadence (see measure_cadence);
    `line_pace` overrides it for this line (mid-song tempo changes).

    v3.0 onset/sustain split — the bug this fixes: the span that spaces the
    onsets of words[0] … words[n-1] only has to cover the articulation of
    words[0 : n-1] plus intra-line caesurae. The final word's own articulation
    AND its held note happen *after* its onset, so folding them into that span
    pushed every earlier word progressively later ("super slow" crawl).

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

    p = line_pace if line_pace else pace
    limit = max(0.0, raw_window - TAIL_GUARD)

    # ---- prosodic analysis of the raw tokens (before rendering) ----
    cores, marks = [], []
    for w in words:
        c, m = split_prosody(w)
        cores.append(c or w)
        marks.append(m)
    syl = [max(1, syllables(c)) for c in cores]
    keys = _iso_keys(cores)
    runs = isochronous_runs(keys)
    run_of = {}
    for (a, b) in runs:
        for k in range(a, b + 1):
            run_of[k] = (a, b)
    pickups = anacrusis_pickups(cores, syl)

    # ---- per-word prosodic weights (syllables × phone mix × stress) ----
    exp = TUNE['syl_exp']
    weights = []
    for i, (c, k_) in enumerate(zip(cores, syl)):
        base = (k_ ** exp) if exp != 1.0 else float(k_)
        wgt = base * phone_duration_scale(c)
        if is_function_word(c) and i < n - 1 and not marks[i]['boundary']:
            # unstressed pickup: rushed into the next content word
            wgt *= FUNC_COMPRESS
        if marks[i]['parenthetical']:
            wgt *= PARENTHETICAL_LEVEL
        if marks[i]['boundary']:
            wgt *= BOUNDARY_LENGTHEN          # pre-boundary vowel lengthening
        if i < pickups:
            wgt *= ANACRUSIS_COMPRESS         # "In the DEATH…" pickup
        weights.append(wgt)

    # Articulation time each word needs at this line's pace.
    artic = [max(1e-3, p * w) for w in weights]

    # ---- caesura micro-pauses after boundaries (never inside an iso run) ----
    gaps = [0.0] * n
    for i in range(n - 1):
        if marks[i]['boundary'] and i not in run_of and (i + 1) not in run_of:
            gaps[i] = CAESURA_MIN_S          # scaled by slack below

    onset_raw = sum(artic[:-1]) + sum(gaps[:-1])
    if onset_raw <= 1e-6:
        onset_raw = 1e-6
    final_hold = min(TUNE['hold_cap'], TUNE['hold_a'] + TUNE['hold_b'] * p)
    phrase_total = onset_raw + artic[-1] + final_hold

    # v3.0: lowered post-phrase tail compression threshold. Whenever the window
    # is longer than the phrase's own articulation by even ~0.8 s — including a
    # 3.5 s or 5.0 s line, not just an 8.6 s verse end — the excess is
    # instrumental/rest: lock the words to their own span at the top of the line.
    gap_locked = raw_window > phrase_total + TAIL_LOCK_SLACK
    slack = max(0.0, raw_window - phrase_total)
    if slack > 0.0:
        frac = min(1.0, slack / 1.4)
        for i in range(n - 1):
            if gaps[i]:
                gaps[i] = min(CAESURA_MAX_S,
                              CAESURA_MIN_S + (CAESURA_MAX_S - CAESURA_MIN_S) * frac)
        onset_raw = sum(artic[:-1]) + sum(gaps[:-1])

    # Onsets must leave the final word at least MIN_SHOW (bounded inside the window).
    last_min = min(artic[-1], MIN_SHOW)
    onset_limit = max(0.0, limit - last_min)
    if gap_locked:
        onset_cap = onset_limit
    else:
        onset_cap = min(onset_limit, raw_window * _voiced_fraction(raw_window))
        if raw_window > TUNE['tail_on']:
            # verse-end / instrumental tail (legacy v2 curve, kept for short tails)
            onset_cap *= max(TUNE['tail_floor'],
                             1.0 - TUNE['tail_slope'] * (raw_window - TUNE['tail_on']))
    onset_span = min(onset_raw, onset_cap)

    # v1.5 held-note phrasing: the slower the song, the more the articulated words
    # bunch toward the phrase start while the final vowel is sustained (convex onset
    # curve) — so mid/late words no longer trail the vocal on ballads, while upbeat
    # songs (pace < ~0.55) stay linear.
    front = 1.0 + min(TUNE['front_max'], max(0.0, (p - TUNE['front_on']) * TUNE['front_gain']))
    cum, acc = [], 0.0
    for i in range(n - 1):
        cum.append(acc)
        acc += artic[i] + gaps[i]
    scale = onset_span / max(onset_raw, 1e-6)
    offsets = [onset_span * ((c * scale / onset_span) ** front) if onset_span > 0 else 0.0
               for c in cum]

    if n == 1:
        times = [line_start]
    else:
        gaps_n = n - 1
        floor = min(MIN_SHOW, onset_limit / gaps_n)
        g = [offsets[i + 1] - offsets[i] for i in range(gaps_n - 1)] + [onset_span - offsets[-1]]
        budget = max(onset_limit, floor * gaps_n)
        need = sum(max(x, floor) for x in g)
        if need > budget + 1e-12:
            slack_g = [max(0.0, x - floor) for x in g]
            s_tot = sum(slack_g)
            if s_tot > 1e-12:
                cut = (need - budget) / s_tot
                g = [floor + s * max(0.0, 1.0 - cut) for s in slack_g]
        g = [max(x, floor) for x in g]
        tg = sum(g)
        if tg > onset_limit and tg > 0:
            k_scale = onset_limit / tg
            g = [x * k_scale for x in g]
        times = [line_start]
        for x in g:
            times.append(times[-1] + x)
        times[0] = line_start

        # ---- isochronous repetition / stutter locking ----
        # "la la la", "yeah yeah yeah", "d-d-death": equal beat intervals
        # instead of the non-linear front exponent curve.
        for (a, b) in runs:
            if b > a and b < n:
                step = (times[b] - times[a]) / (b - a)
                if step > 1e-4:
                    for k in range(a, b + 1):
                        times[k] = times[a] + (k - a) * step

    out = list(zip(times, words))

    # v3.0 clear point: the last word's own articulation + its held note, both
    # of which start at times[-1] — never a fraction of the window (that is what
    # held a finished word on screen through a 5 s instrumental phrase).
    last_hold = min(TUNE['rest_cap'], max(0.6, p * syl[-1] + TUNE['rest_a']))
    clear_at = times[-1] + min(artic[-1] + final_hold, last_hold + artic[-1])
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
    meta     optional dict; receives {'estimated_lines': int, 'real_lines': int} plus the
             per-event tier map the live phase-lock needs:
               'line_id'   list[int]   source line index per event (-1 for rests/intro)
               'estimated' list[bool]  True when the onset came from the cadence model
               'line_end'  list[float] end of that line's window (s)
               'base_pace' float       measured true articulation pace (s/syllable)
               'line_pace' list[float] per-line local pace (tempo changes)
    audio    optional WAV path, (sr, samples) tuple, (sr, left, right) stereo tuple, or
             FeatureTrack. When supplied, the offline engine refines onsets (VAD freeze,
             formant onsets, duration-aware search). Heuristic timeline always built first;
             with a live capture this is how a replay of the same track becomes Tier 1.5.
    align_cfg optional AlignConfig (lookahead cap, calibration offset, freeze)."""
    if not lines:
        return []
    lines = normalise_line_starts(lines)
    n_est = n_real = 0
    if not total_duration or total_duration <= 0:
        total_duration = lines[-1]['start'] + 30.0

    # ---- v3.0: measure the song's TRUE articulation pace with the dense-frontier
    # estimator (25–30th percentile of per-line s/syllable), then adapt it per line
    # to genuine tempo changes while refusing to inflate on post-phrase
    # instrumental gaps (the "Panchiko / shoegaze 6.5 s window" bug).
    _wins = []
    for i, _ln in enumerate(lines):
        _end = lines[i + 1]['start'] if i + 1 < len(lines) else _ln['start'] + 6.0
        _wins.append(_end - _ln['start'])
    pace, line_paces, cadence = measure_cadence(lines, _wins)

    rows = []          # (t, word, line_id, estimated)
    if lines[0]['start'] > 2.0:
        rows.append((0.0, '', -1, False))

    for i, line in enumerate(lines):
        line_end = lines[i + 1]['start'] if i + 1 < len(lines) else max(
            total_duration, line['start'] + MIN_LINE_GAP)
        real = line.get('words')
        if real:
            # Tier 1: trust the source. Keep each word strictly inside its own line window.
            ws_ = [(t, w) for t, w in real if line['start'] - 0.5 <= t < line_end]
            if ws_:
                n_real += 1
                rows.extend((t, w, i, False) for t, w in ws_)
                if line_end - ws_[-1][0] > 3.5:
                    rows.append((ws_[-1][0] + 1.8, '', -1, False))
                continue
        words = line_words(line['text'])
        if words:
            n_est += 1
            est = estimate_line(words, line['start'], line_end,
                                line_paces[i] if line_paces else pace)
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
            rows.extend((t, w, i, True) for t, w in est)

    rows.sort(key=lambda e: e[0])
    cleaned = []
    for r in rows:
        if cleaned and cleaned[-1][1] == '' and r[1] == '':
            continue
        cleaned.append(r)
    out = [(t, w) for t, w, _l, _e in cleaned]
    if meta is not None:
        meta['estimated_lines'], meta['real_lines'] = n_est, n_real
        meta['aligned'] = False
        meta['base_pace'] = pace
        meta['line_pace'] = list(line_paces)
        meta['cadence'] = {k: v for k, v in cadence.items() if k != 'raw_rate'}
    if audio is not None and out and _HAS_ALIGN:
        try:
            refined = refine_word_events(out, audio, align_cfg)
            if len(refined) == len(out):
                out = refined
            if meta is not None:
                meta['aligned'] = True
        except Exception:
            pass
    if meta is not None and len(out) == len(cleaned):
        meta['line_id'] = [r[2] for r in cleaned]
        meta['estimated'] = [r[3] for r in cleaned]
        meta['line_end'] = [
            (lines[r[2] + 1]['start'] if r[2] >= 0 and r[2] + 1 < len(lines)
             else (lines[r[2]]['start'] + 6.0 if r[2] >= 0 else 0.0))
            for r in cleaned]
    return out


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
             'player', 'plain', 'resync', 'tick', 'lyrics_dir', 'cache_dir',
             'audio_sync')
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

    # ---- v3.0 Panchiko D>E>A>T>H>M>E>T>A>L archetype ----
    # A 6-syllable vocal phrase sung in ~1.8 s followed by 4-5 s of guitars
    # gives a 6.5 s line window. The dense-phrase frontier must spot that the
    # window is mostly instrumental: onsets stay with the vocal instead of
    # crawling across the gap at the 0.822 s/syllable ceiling.
    _LONG, _SHORT = 'we are the dead ones now', 'hold me closer tonight'
    _fx, _t = [], 0.0
    for _txt, _win in ((_LONG, 6.5), (_SHORT, 2.2), (_LONG, 6.5), (_SHORT, 2.2),
                       (_LONG, 6.5), (_SHORT, 2.2), (_LONG, 6.5)):
        _fx.append({'start': round(_t, 3), 'text': _txt, 'words': None})
        _t += _win
    _meta = {}
    _evs = build_word_events(_fx, _t + 8.0, meta=_meta)
    _syls = [sum(syllables(w) for w in ln['text'].split()) for ln in _fx]
    check(all(5 <= n <= 7 for n in _syls),
          f'panchiko fixture is 5-7 syllables/line ({_syls})')
    _first = [(tt, w) for tt, w in _evs if tt < 6.5]
    _on = [tt for tt, w in _first if w]
    _clr = [tt for tt, w in _first if not w and tt > 0.2]
    check(bool(_on) and _on[-1] <= 2.4,
          f'panchiko: 6-syllable phrase completes <= 2.4 s (got {_on[-1]:.2f})')
    check(bool(_clr) and _clr[0] < 3.5,
          f'panchiko: rest marker before 3.5 s (got {_clr[0] if _clr else -1:.2f})')
    check(_meta['base_pace'] < 0.60,
          f"panchiko: frontier pace is articulation, not the ceiling "
          f"({_meta['base_pace']:.3f} s/syllable)")
    _short_on = [tt for tt, w in _evs if w and 6.5 <= tt < 8.7]
    check(bool(_short_on) and (_short_on[-1] - _short_on[0]) > 0.9,
          'panchiko: normal-window lines still fill their window')

    # ---- v3.0 function words + caesurae ----
    _w = ['Wait,', 'in', 'the', 'middle', 'of', 'the', 'night']
    _est = estimate_line(_w, 0.0, 6.0, pace=0.5)
    _d = {word: tt for tt, word in _est if word}
    _ctl = {word: tt for tt, word in
            estimate_line(['Wait'] + _w[1:], 0.0, 6.0, pace=0.5) if word}
    check(_d['in'] - _d['Wait,'] > (_ctl['in'] - _ctl['Wait']) + 0.20,
          'caesura: "Wait," is lengthened and pauses before "in"')
    check((_d['middle'] - _d['in']) < (_d['of'] - _d['middle']),
          'function words: "in the" compressed vs "middle"')
    check((_d['night'] - _d['of']) < (_d['of'] - _d['middle']),
          'function words: "of the" compressed vs "middle"')
    _ana = {word: tt for tt, word in
            estimate_line(['In', 'the', 'SUFFERING', 'now'], 0.0, 6.0, pace=0.5)
            if word}
    _ctl2 = {word: tt for tt, word in
             estimate_line(['Deep', 'song', 'SUFFERING', 'now'], 0.0, 6.0, pace=0.5)
             if word}
    _ratio = ((_ana['SUFFERING'] - _ana['In']) / max(1e-6, _ana['now'] - _ana['SUFFERING']))
    _ratio_c = ((_ctl2['SUFFERING'] - _ctl2['Deep'])
                / max(1e-6, _ctl2['now'] - _ctl2['SUFFERING']))
    check(_ratio < 0.5 * _ratio_c,
          f'anacrusis: "In the" pickups compress into the stressed word '
          f'({_ratio:.2f} vs {_ratio_c:.2f})')
    check(anacrusis_pickups(['In', 'the', 'SUFFERING'], [1, 1, 3]) == 2
          and anacrusis_pickups(['In', 'the', 'light'], [1, 1, 1]) == 0
          and anacrusis_pickups(['Deep', 'song', 'SUFFERING'], [1, 1, 3]) == 0,
          'anacrusis pickup detection')

    # ---- v3.0 isochronous repetition / stutter locking ----
    _la = [tt for tt, w in estimate_line(['la', 'la', 'la', 'la'], 0.0, 4.0, pace=0.5)
           if w]
    _steps = [_la[i + 1] - _la[i] for i in range(len(_la) - 1)]
    check(bool(_steps) and max(_steps) - min(_steps) < 1e-6,
          'isochrony: "la la la la" locks to equal beat intervals')
    check(repeat_key('d-d-death') == 'death' and repeat_key('LA,') == 'la',
          'stutter / repeat key normalisation')

    # ---- v3.0 mixed tempo / beat switch ----
    _rap = 'i got the money and the power in my hands right'
    _ch = 'hold me closer now'
    _fx2, _t2 = [], 0.0
    for _txt, _win in [(_rap, 2.5)] * 6 + [(_ch, 4.5)] * 4 + [(_rap, 2.5)] * 4:
        _fx2.append({'start': round(_t2, 3), 'text': _txt, 'words': None})
        _t2 += _win
    _meta2 = {}
    _evs2 = build_word_events(_fx2, _t2 + 8.0, meta=_meta2)
    _lp = _meta2['line_pace']
    _verse_p = max(_lp[:6])
    _chorus_p = min(_lp[6:10])
    check(_chorus_p > 1.8 * _verse_p,
          f'mixed tempo: chorus pace adapts ({_chorus_p:.3f} vs {_verse_p:.3f})')
    _c2 = [tt for tt, w in _evs2 if w and 19.5 <= tt < 24.0]
    check(bool(_c2) and (_c2[-1] - _c2[0]) > 1.8,
          f'mixed tempo: 2nd chorus line spreads out ({(_c2[-1] - _c2[0]):.2f} s)')
    _r2 = [tt for tt, w in _evs2 if w and 0.0 <= tt < 2.5]
    check(bool(_r2) and (_r2[-1] - _r2[0]) < 2.4,
          'mixed tempo: rap verse stays inside its 2.5 s window')

    # ---- v3.0 stereo Mid/Side vocal isolation ----
    _sr = TARGET_SR
    _n = int(6.0 * _sr)
    _gtr = [0.0] * _n
    for _i in range(_n):
        _tt = _i / _sr
        _gtr[_i] = 0.30 * sum(
            _a * math.sin(2 * math.pi * _f * _tt + _p) for _f, _a, _p in
            ((310, 0.30, 0.3), (520, 0.26, 1.1), (830, 0.24, 2.0),
             (1250, 0.20, 0.7), (1900, 0.16, 1.7), (2700, 0.12, 2.6),
             (3300, 0.08, 0.2)))
    _gl = list(_gtr)
    _gr = [-x for x in _gtr]
    for _t0, _t1 in ((0.4, 0.8), (1.0, 1.4), (1.6, 2.0)):
        _m = int(round((_t1 - _t0) * _sr))
        for _k in range(_m):
            _tt = _k / _sr
            _e = 1.0
            if _tt < 0.03:
                _e = _tt / 0.03
            if _tt > (_t1 - _t0) - 0.05:
                _e = max(0.0, ((_t1 - _t0) - _tt) / 0.05)
            _v = _e * 0.26 * (0.62 * math.sin(2 * math.pi * 720.0 * _tt)
                              + 0.38 * math.sin(2 * math.pi * 1240.0 * _tt))
            _idx = int(_t0 * _sr) + _k
            if 0 <= _idx < _n:
                _gl[_idx] += _v
                _gr[_idx] += _v
    _st = extract_features_stereo(_gl, _gr, _sr)
    _svad = [f.t for f in _st.frames if f.vad]
    check(bool(_svad) and _svad[0] < 0.55, 'mid/side: centre vocal burst detected')
    check(not any(f.vad for f in _st.frames if f.t < 0.28 or f.t > 2.14),
          'mid/side: wide-panned guitars rejected (no false VAD)')
    _mono_side = extract_features([0.5 * (_gl[i] - _gr[i]) for i in range(_n)], _sr)
    check(sum(1 for f in _mono_side.frames if f.vad) > 0.5 * len(_mono_side.frames),
          'mid/side: mono band-pass VAD would false-positive on the same mix')
    _al = align_words(['i', 'will', 'be', 'dead'], [0.05, 0.20, 0.35, 0.50],
                      (_sr, _gl, _gr))
    check(all(0.35 <= o <= 2.15 for o in _al.onsets),
          'mid/side: align snaps onsets onto the isolated vocal')
    # streaming (live) extractor must agree with the offline stereo path
    _ext = StreamingFeatureExtractor(_sr)
    _lf = []
    for _i in range(0, _n, 1600):
        _lf.extend(_ext.push(_gl[_i:_i + 1600], _gr[_i:_i + 1600]))
    _lv = [f.t for f in _lf if f.vad]
    check(bool(_lv) and abs(_lv[0] - _svad[0]) < 0.08,
          'live streaming extractor matches the offline stereo path')
    check(not any(f.vad for f in _lf if f.t < 0.28 or f.t > 2.14),
          'live streaming extractor rejects the guitars too')
    check(_isolate_vocal(0.30, 0.0) > 0.29 and _isolate_vocal(0.0, 0.30) == 0.0,
          'isolation: centre vocal kept, pure side image nulled')

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
    ap.add_argument('--audio-sync', dest='audio_sync', action='store_true', default=None,
                    help='capture the player\u2019s monitor sink and phase-lock estimated '
                         'word onsets to the real vocal (default: on)')
    ap.add_argument('--no-audio-sync', dest='audio_sync', action='store_false',
                    help='disable live audio capture; use the prosodic engine alone')
    ap.add_argument('--audio-device', default=None,
                    help='capture monitor/source name for --audio-sync (default: the '
                         'player\u2019s sink, then @DEFAULT_MONITOR@)')
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
              'lyrics_dir', 'cache_dir', 'resync', 'tick', 'audio_sync'):
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
    # v3.0 live audio bridge: capture the player's monitor sink (PipeWire /
    # PulseAudio), isolate the centre vocal with Mid/Side, and phase-lock the
    # estimated onsets onto it. 100 % fail-soft: when capture is unavailable
    # (or --no-audio-sync), the prosodic engine runs alone.
    audio_sync_on = args.audio_sync if args.audio_sync is not None else \
        _as_bool(active.get('audio_sync'), True)
    live_cfg = AlignConfig()
    live_audio = LiveAudioTracker(live_cfg, enabled=bool(audio_sync_on),
                                  device=args.audio_device,
                                  player=(active.get('player') or None))
    live_lock = LivePhaseLock(live_cfg, enabled=bool(audio_sync_on))
    # Live capture runs in both render modes; --no-audio-sync (or a missing
    # backend) is the opt-out / fail-soft path.
    live_on = live_audio.start() if audio_sync_on else False
    needs_redraw = False          # repaint (e.g. brightness on pause) without re-announcing a word
    song_offsets = load_song_offsets()
    offset_dirty = False
    have_state = False
    playing = False
    poll_every = max(0.12, min(0.3, resync / 5.0))   # fast position poll (seek/pause latency)
    next_poll = 0.0

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

    event_meta = {}
    base_starts = []

    def _install_events(evs, meta=None):
        """Publish a timeline to the render loop (+ tier metadata for the lock)."""
        nonlocal events, starts, base_starts, last_idx, last_word_shown, event_meta
        events = list(evs or [])
        starts = [e[0] for e in events]
        base_starts = list(starts)
        last_idx = -1
        last_word_shown = None
        event_meta = dict(meta or {})
        live_lock.reset(events, event_meta)
        return events

    def _audio_for(song_key):
        """FeatureTrack captured on a previous play-through, when we have one."""
        try:
            return live_audio.cached_track(song_key)
        except Exception:
            return None

    def _load_song(st, cand_idx=0):
        """Fetch + parse + build the word timeline for the current track.
        Returns (events, lines, n_candidates)."""
        artist, title = st['artist'], st['title']
        if not args.plain:
            render_message(f'loading lyrics — {artist} – {title}')
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
            cand = cands[min(cand_idx, len(cands) - 1)]
            # Real word timing first (lyricsfile); fall back to line-level LRC.
            if cand.get('lyricsfile'):
                lines = parse_lyricsfile(cand['lyricsfile'])
            if not lines and cand.get('synced'):
                lines = parse_lrc(cand['synced'])
        meta = {}
        evs = build_word_events(lines, st['duration'], stretch,
                                meta=meta, audio=_audio_for(st['song_key']),
                                align_cfg=live_cfg) if lines else []
        meta['audio_sync'] = live_audio.capture_summary()
        if not evs and not args.plain:
            render_message(f'no synced lyrics found for {artist} – {title}')
        return evs, lines, len(cands), meta

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
                    live_audio.end_track()
                    song_key = track_id = None
                    have_state = False
                    events, starts = [], []
                    base_starts = []
                    clock.reset()
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
                    live_audio.end_track()          # freeze the previous capture
                    song_key, track_id = fresh['song_key'], fresh['track_id']
                    last_word_shown = None
                    last_idx = -1
                    fade_start = -1.0
                    clock.reset()
                    saved = song_offsets.get(song_key, {})
                    manual_offset = global_offset + saved.get('offset', 0.0)
                    stretch = saved.get('stretch', 1.0)
                    stretch_dirty = False
                    candidate_idx = 0
                    evs_, cur_lines, cand_count, meta_ = _load_song(fresh, 0)
                    cur_duration = fresh['duration']
                    live_audio.begin_track(song_key, cur_duration)
                    _install_events(evs_, meta_)
                    no_lyric_until = (time.monotonic() + 4.0) if not events else 0.0
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
                                evs_, cur_lines, cand_count, meta_ = _load_song(
                                    cur_state, candidate_idx)
                                _install_events(evs_, meta_)
                                no_lyric_until = (time.monotonic() + 4.0) if not events else 0.0
                        elif ch == '0':
                            manual_offset = global_offset
                            stretch = 1.0
                            stretch_dirty = offset_dirty = True
                except Exception:
                    pass
            if stretch_dirty and cur_lines:
                stretch_dirty = False
                meta_ = {}
                evs_ = build_word_events(cur_lines, cur_duration, stretch, meta=meta_,
                                        audio=_audio_for(song_key), align_cfg=live_cfg)
                _install_events(evs_, meta_)

            tick_t = time.monotonic()
            pos = max(0.0, clock.position_at(tick_t) - manual_offset) if have_state else 0.0
            if args.plain:
                needs_redraw = False

            # ---- v3.0 live causal phase-locking ----
            # Only Tier-2 events that have not been announced yet are nudged, so
            # a word already on screen can never jump backwards. Any failure
            # disables the live path and the schedule plays out unchanged.
            if live_on and events and have_state and live_audio.running:
                try:
                    live_audio.update_anchor(pos, tick_t)
                    patch = live_lock.update(events, base_starts, last_idx, pos,
                                             tick_t, live_audio)
                    if patch:
                        for i_, t_ in patch.items():
                            if 0 <= i_ < len(starts):
                                lo_ = starts[i_ - 1] + 1e-4 if i_ > 0 else -1e9
                                starts[i_] = max(t_, lo_)
                except Exception:
                    live_on = False

            if events and have_state:
                idx = bisect.bisect_right(starts, pos) - 1
                if idx >= 0:
                    word = events[min(idx, len(events) - 1)][1]
                    if color_mode == 'random':
                        word_rgb = hsv_to_rgb(random.random(), 1, 1)
                    else:
                        word_rgb = color_rgb

                    if idx != last_idx:
                        # announce per event, not per text: repeated words
                        # ("la la la") re-pulse instead of freezing (v2.1)
                        last_idx = idx
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
                    elif needs_redraw and word and not args.plain:
                        needs_redraw = False
                        draw_block(word, word_rgb, intensity_for(playing, 1.0, paused_dim), opts)
                    elif not word and not args.plain:
                        if needs_redraw or tick_t - last_dot_time > 0.35:
                            dot_phase += 1
                            last_dot_time = tick_t
                            needs_redraw = False
                            draw_block('', word_rgb, 1.0, opts, dot_phase)
                    elif not args.plain and fade_start >= 0 and (tick_t - fade_start) < fade:
                        draw_block(word, word_rgb,
                                   intensity_for(playing, (tick_t - fade_start) / fade, paused_dim), opts)
                    elif not args.plain and not playing:
                        draw_block(word, word_rgb,
                                   intensity_for(False, 1.0, paused_dim), opts)
                elif idx < 0 and last_word_shown not in (None, ''):
                    # seeked back before the first word: clear instead of leaving a stale word
                    last_word_shown = ''
                    last_idx = -1
                    if not args.plain:
                        draw_block('', word_rgb, 1.0, opts, dot_phase)
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
        live_audio.stop()
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
