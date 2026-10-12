#!/usr/bin/env python3
"""
wordsync_align.py — core offline audio-to-lyric alignment engine.

Pure stdlib. No network, no UI, no third-party packages.

This is the search / onset / duration layer that the lyric display
(`word-sync(2).py`) uses to keep word onsets on the vocal even when
the song is a slow ballad, a breathy attack, or a syncopated groove.

Pipeline
--------
  WAV / PCM
    -> band-limited feature frames  (formant 300 Hz–3.4 kHz, hop ≤ 10 ms)
    -> soft vocal onsets            (formant-energy *rise*, not full-band spikes)
    -> VAD + pause freeze           (pointer locked after >400 ms of silence)
    -> duration-aware beam search   (vowel self-loops cheap, plosive ceilings)
    -> adaptive corridor            (tight when confident, wide when not;
                                     no tempo-grid penalty on off-beat hits)

Timestamps are compensated by a configurable calibration offset that
cancels the analysis-window / hop delay. Lookahead is capped at 150 ms.
"""

from __future__ import annotations

import math
import os
import wave
from array import array
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

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
    if sr != TARGET_SR and sr > TARGET_SR:
        sr, samples = _resample_mono(samples, sr, TARGET_SR)
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

    # Vocal-band signal (task 3): formant energy, not full-band RMS.
    vocal = _one_pole_lp(_one_pole_hp(samples, sr, cfg.formant_lo_hz),
                         sr, cfg.formant_hi_hz)
    win = _hamming(n_win)

    rms_env: List[float] = []
    form_env: List[float] = []
    centres: List[float] = []
    i = 0
    while i + n_win <= n or (i < n and not centres):
        end = min(n, i + n_win)
        # Zero-pad a trailing partial frame so we don't invent lookahead
        # past EOF — we just analyse what exists.
        acc_r = 0.0
        acc_f = 0.0
        wsum = 0.0
        for k in range(n_win):
            w = win[k]
            idx = i + k
            x = samples[idx] if idx < n else 0.0
            v = vocal[idx] if idx < n else 0.0
            acc_r += w * x * x
            acc_f += w * v * v
            wsum += w
        wsum = wsum or 1.0
        rms_env.append(math.sqrt(max(0.0, acc_r / wsum)))
        form_env.append(math.sqrt(max(0.0, acc_f / wsum)))
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


if __name__ == '__main__':
    fails = engine_unit_checks()
    if fails:
        print('wordsync_align unit checks FAILED:')
        for f in fails:
            print('  -', f)
        raise SystemExit(1)
    print('wordsync_align unit checks OK')
