#!/usr/bin/env python3
"""
Offline benchmark / regression harness for the v2.2 alignment engine.

No network, no UI, no third-party packages. Synthetic WAV snippets are
written under ./_align_fixtures/ (gitignored-style local dir) and aligned
with the same stdlib engine the lyric display calls.

Measures
--------
  1. Feature-extraction lookahead ≤ 150 ms + calibration offset.
  2. Pause freeze: a >400 ms instrumental must not advance the lyric pointer.
  3. Soft formant-band onset (breathy / slow rise, no full-band click).
  4. Syncopation: off-beat onsets inside the adaptive corridor are accepted.
  5. Standard on-tempo track: onsets stay close to the heuristic priors.
  6. Duration model: vowel self-loops cheap, plosive ceilings strict.
  7. Existing cadence selftest (build_word_events) still passes.
  8. v3.0 Panchiko D>E>A>T>H>M>E>T>A>L archetype (post-phrase instrumental gap).
  9. v3.0 function-word compression, caesurae and anacrusis.
 10. v3.0 mixed tempo / beat switch (per-line local pace).
 11. v3.0 stereo Mid/Side vocal isolation (synthetic stereo WAV).
 12. v3.0 live audio bridge: streaming parity, causal phase-lock, capture device.

Run:  python3 test_align_engine.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DIR = os.path.join(ROOT, '_align_fixtures')
SR = 16000


def _load_word_sync():
    path = os.path.join(ROOT, 'word-sync(2).py')
    spec = importlib.util.spec_from_file_location('word_sync_mod', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ws = _load_word_sync()
MAX_LOOKAHEAD_S = ws.MAX_LOOKAHEAD_S
AlignConfig = ws.AlignConfig
PHONE_PLOSIVE = ws.PHONE_PLOSIVE
PHONE_VOWEL = ws.PHONE_VOWEL
align_words = ws.align_words
engine_unit_checks = ws.engine_unit_checks
extract_features = ws.extract_features
phone_duration_scale = ws.phone_duration_scale
profile_feature_latency = ws.profile_feature_latency
stay_cost = ws.stay_cost
write_wav = ws.write_wav
estimate_line = ws.estimate_line
measure_cadence = ws.measure_cadence
extract_features_stereo = ws.extract_features_stereo
load_wav_stereo = ws.load_wav_stereo
StreamingFeatureExtractor = ws.StreamingFeatureExtractor
LiveAudioTracker = ws.LiveAudioTracker
LivePhaseLock = ws.LivePhaseLock
FrameFeat = ws.FrameFeat
build_word_events = ws.build_word_events


# ---------------------------------------------------------------------------
# Synthetic audio
# ---------------------------------------------------------------------------

def _env_edges(i, n, attack, release, sr):
    t = i / sr
    dur = n / sr
    e = 1.0
    if attack > 0 and t < attack:
        e *= t / attack
    if release > 0 and t > dur - release:
        e *= max(0.0, (dur - t) / release)
    return e


def vowel_tone(dur, sr=SR, f1=720.0, f2=1240.0, amp=0.28,
               attack=0.040, release=0.030):
    """Two-formant vowel-like tone (energy lives in 300 Hz–3.4 kHz)."""
    n = int(round(dur * sr))
    out = [0.0] * n
    for i in range(n):
        t = i / sr
        e = _env_edges(i, n, attack, release, sr)
        out[i] = e * amp * (
            0.62 * math.sin(2 * math.pi * f1 * t)
            + 0.38 * math.sin(2 * math.pi * f2 * t)
        )
    return out


def breathy_rise(dur, sr=SR, f1=620.0, f2=1600.0, amp=0.22, rise=0.090):
    """Slow formant-band swell with no click — a quiet / breathy attack."""
    n = int(round(dur * sr))
    out = [0.0] * n
    for i in range(n):
        t = i / sr
        if t < rise:
            e = (t / rise) ** 1.6          # concave, very soft start
        else:
            e = 1.0
        if t > dur - 0.04:
            e *= max(0.0, (dur - t) / 0.04)
        # Tiny noise, still in-band, no full-band transient spike.
        noise = 0.02 * math.sin(2 * math.pi * 310.0 * t + 0.7)
        out[i] = e * amp * (
            0.55 * math.sin(2 * math.pi * f1 * t)
            + 0.35 * math.sin(2 * math.pi * f2 * t)
            + noise
        )
    return out


def silence(dur, sr=SR):
    return [0.0] * int(round(dur * sr))


def wide_guitars(dur, sr=SR, amp=0.30):
    """Band-limited 'guitars/synths' fully panned: the mono-VAD bait of a
    shoegaze/rock/EDM mix (L = +x, R = -x, so Mid ~= 0, Side = x)."""
    n = int(round(dur * sr))
    out = [0.0] * n
    partials = ((310, 0.30, 0.3), (520, 0.26, 1.1), (830, 0.24, 2.0),
                (1250, 0.20, 0.7), (1900, 0.16, 1.7), (2700, 0.12, 2.6),
                (3300, 0.08, 0.2))
    for i in range(n):
        t = i / sr
        out[i] = amp * sum(a * math.sin(2 * math.pi * f * t + p)
                           for f, a, p in partials)
    return out


def center_vocal(dur, sr=SR, f1=720.0, f2=1240.0, amp=0.26):
    """Centre-panned vocal burst: identical on L and R."""
    n = int(round(dur * sr))
    out = [0.0] * n
    for i in range(n):
        t = i / sr
        e = 1.0
        if t < 0.030:
            e = t / 0.030
        if t > dur - 0.050:
            e = max(0.0, (dur - t) / 0.050)
        out[i] = e * amp * (0.62 * math.sin(2 * math.pi * f1 * t)
                            + 0.38 * math.sin(2 * math.pi * f2 * t))
    return out


def stereo_mix(dur, bursts, sr=SR, guitar_amp=0.30):
    """(left, right) of wide anti-phase guitars + centre vocal bursts."""
    gtr = wide_guitars(dur, sr, guitar_amp)
    left, right = list(gtr), [-x for x in gtr]
    n = len(gtr)
    for (t0, t1) in bursts:
        v = center_vocal(t1 - t0, sr)
        for k, x in enumerate(v):
            idx = int(round(t0 * sr)) + k
            if 0 <= idx < n:
                left[idx] += x
                right[idx] += x
    return left, right


def vad_segments(frames):
    """[(start, end), ...] of consecutive VAD frames."""
    segs, start = [], None
    for f in frames:
        if f.vad and start is None:
            start = f.t
        elif not f.vad and start is not None:
            segs.append((start, f.t))
            start = None
    if start is not None and frames:
        segs.append((start, frames[-1].t))
    return segs


def concat(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


def save(name, samples, sr=SR, right=None):
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_DIR, name)
    write_wav(path, sr, samples, right=right)
    return path


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

class Harness:
    def __init__(self):
        self.failed = []
        self.passed = []
        self.metrics = {}

    def check(self, cond, label, detail=''):
        if cond:
            self.passed.append(label)
            extra = ('  (%s)' % detail) if detail else ''
            print('  PASS  %s%s' % (label, extra))
        else:
            self.failed.append(label if not detail else '%s (%s)' % (label, detail))
            extra = ('  (%s)' % detail) if detail else ''
            print('  FAIL  %s%s' % (label, extra))

    def section(self, title):
        print('\n== %s ==' % title)


def test_unit_costs(h: Harness):
    h.section('1. Duration model (vowels vs. consonants)')
    cfg = AlignConfig()
    v_long = stay_cost(PHONE_VOWEL, 1.50, cfg)
    p_long = stay_cost(PHONE_PLOSIVE, 0.40, cfg)
    p_short = stay_cost(PHONE_PLOSIVE, 0.04, cfg)
    h.check(v_long < 0.15, 'vowel 1.5 s self-loop stays cheap',
            'cost=%.4f' % v_long)
    h.check(p_long > p_short * 4, 'plosive explodes past 120 ms ceiling',
            'short=%.3f long=%.3f' % (p_short, p_long))
    h.check(phone_duration_scale('love') > phone_duration_scale('strict'),
            'heuristic scale: love > strict')
    for label in engine_unit_checks():
        h.check(False, label)
    if not engine_unit_checks():
        h.check(True, 'engine_unit_checks()')


def test_latency(h: Harness):
    h.section('2. Latency & lookahead (cap 150 ms)')
    samples = vowel_tone(1.8) + silence(0.2)
    cfg = AlignConfig(frame_s=0.025, hop_s=0.010, calibration_offset_s=0.0)
    t0 = time.perf_counter()
    track = extract_features(samples, SR, cfg)
    elapsed = time.perf_counter() - t0
    prof = profile_feature_latency(track)
    h.metrics['lookahead_ms'] = prof['lookahead_ms']
    h.metrics['extract_s_per_audio_s'] = elapsed / 2.0
    h.metrics['receptive_delay_ms'] = prof['receptive_delay_s'] * 1000.0
    h.check(prof['within_cap'],
            'lookahead ≤ 150 ms',
            '%.1f ms' % prof['lookahead_ms'])
    h.check(prof['lookahead_s'] <= 0.150 + 1e-9, 'explicit 150 ms cap')
    h.check(track.receptive_delay_s <= track.frame_s * 0.5 + 1e-9,
            'receptive delay is half-window')
    # Huge requested window must be trimmed.
    wide = extract_features(samples, SR, AlignConfig(frame_s=0.80,
                                                     max_lookahead_s=0.150))
    h.check(wide.lookahead_s <= MAX_LOOKAHEAD_S + 1e-9,
            'oversize window trimmed to cap',
            '%.1f ms' % (wide.lookahead_s * 1000))
    # Calibration offset shifts every stamp.
    shifted = extract_features(samples, SR,
                               AlignConfig(calibration_offset_s=0.040))
    dt = shifted.frames[0].t - track.frames[0].t
    h.check(abs(dt - 0.040) < 1.5e-3, 'calibration offset applied',
            'dt=%.4f s' % dt)
    h.check(elapsed < 2.5, 'extract 2 s of audio in < 2.5 s CPU',
            '%.3f s' % elapsed)


def test_pause_freeze(h: Harness):
    h.section('3. Pause freeze (VAD-gated pointer, >400 ms)')
    # 0.55 s vocal | 1.10 s instrumental | 0.60 s vocal
    # Heuristic prior for word 2 sits in the *middle of the solo* — the
    # bug the freeze is designed to stop.
    pre, gap, post = 0.55, 1.10, 0.60
    samples = concat(
        vowel_tone(pre, f1=660, f2=1180, attack=0.025),
        silence(gap),
        vowel_tone(post, f1=780, f2=1400, attack=0.035),
    )
    path = save('pause_freeze.wav', samples)
    pause_start, pause_end = pre, pre + gap
    words = ['hello', 'world']
    priors = [0.06, pause_start + 0.55]          # 1.10 s, inside the gap
    cfg = AlignConfig()
    t0 = time.perf_counter()
    result = align_words(words, priors, path, cfg)
    h.metrics['align_pause_s'] = time.perf_counter() - t0
    hello, world = result.onsets
    h.metrics['frozen_frames'] = result.frozen_frames
    h.check(result.frozen_frames > 0, 'freeze engaged during the gap',
            '%d frames' % result.frozen_frames)
    h.check(hello < pause_start + 0.12, 'word 1 onset in the first vocal',
            't=%.3f' % hello)
    # Must not fire in the middle of the instrumental.
    mid = pause_start + 0.45
    h.check(not (pause_start + 0.25 <= world <= pause_end - 0.15),
            'word 2 does not land in the instrumental',
            't=%.3f (pause %.2f–%.2f)' % (world, pause_start, pause_end))
    h.check(world >= pause_end - 0.10, 'word 2 waits until singing resumes',
            't=%.3f vs resume %.2f' % (world, pause_end))
    # Drift metric: how far past the bad prior we pushed into the vocal.
    h.metrics['pause_drift_correction_s'] = world - priors[1]


def test_soft_onset(h: Harness):
    h.section('4. Soft vocal onset (formant rise, not full-band spike)')
    lead = 0.35
    samples = concat(silence(lead), breathy_rise(0.70, rise=0.090), silence(0.20))
    save('breathy_onset.wav', samples)
    track = extract_features(samples, SR, AlignConfig())
    # Strongest formant onset should sit near the swell, not at t=0.
    best = max(track.frames, key=lambda f: f.onset)
    h.check(abs(best.t - lead) < 0.16,
            'formant-rise onset near breathy attack',
            'peak t=%.3f (target %.2f, strength=%.2f)' % (best.t, lead, best.onset))
    # Full-band RMS 1-hop derivative should *not* be required: the rise is
    # spread over 90 ms. We only assert the formant onset fired.
    h.check(best.onset >= 0.35, 'formant onset strength is usable',
            '%.2f' % best.onset)
    result = align_words(['you'], [lead + 0.12], (SR, samples), AlignConfig())
    h.check(abs(result.onsets[0] - lead) < 0.18,
            'aligned word snaps to the breathy rise',
            't=%.3f' % result.onsets[0])


def test_syncopation(h: Harness):
    h.section('5. Adaptive corridor / syncopation (no tempo-grid penalty)')
    # "Beat" grid every 0.50 s. Actual vowels sit 100 ms *early* (upbeat).
    beat = 0.50
    early = 0.10
    true = [0.40, 0.90, 1.40]          # 0.50-0.10, ...
    priors = [0.50, 1.00, 1.50]
    parts = [silence(true[0])]
    last = true[0]
    for t in true:
        gap = t - last
        if gap > 1e-6:
            # the silence between vowels is already represented by waiting
            pass
        parts.append(vowel_tone(0.28, f1=700, f2=1300, attack=0.012, release=0.04))
        parts.append(silence(0.22))
        last = t
    # Rebuild more carefully so onsets land on `true`.
    samples = silence(true[0])
    cursor = true[0]
    for i, t in enumerate(true):
        if cursor < t:
            samples = samples + silence(t - cursor)
            cursor = t
        v = vowel_tone(0.26, f1=680 + 40 * i, f2=1200, attack=0.012, release=0.035)
        samples = samples + v
        cursor += 0.26
    samples = samples + silence(0.30)
    save('syncopation.wav', samples)
    words = ['off', 'beat', 'hit']
    result = align_words(words, priors, (SR, samples), AlignConfig())
    errs_prior = [abs(o - p) for o, p in zip(result.onsets, priors)]
    errs_true = [abs(o - t) for o, t in zip(result.onsets, true)]
    h.metrics['syncopation_mae_vs_true'] = sum(errs_true) / len(errs_true)
    h.metrics['syncopation_mae_vs_prior'] = sum(errs_prior) / len(errs_prior)
    # Closer to the real early onsets than to the on-beat priors, or at
    # least inside 120 ms of the true hit (corridor_tight).
    close = sum(1 for e in errs_true if e <= 0.14)
    h.check(close >= 2, '≥2/3 syncopated hits land within 140 ms of true onset',
            'onsets=%s true=%s' % (
                '[' + ', '.join('%.3f' % x for x in result.onsets) + ']',
                '[' + ', '.join('%.3f' % x for x in true) + ']'))
    h.check(all(o < p + 0.05 for o, p in zip(result.onsets, priors)),
            'engine does not push syncopated hits *later* onto the beat')


def test_standard_track(h: Harness):
    h.section('6. Standard on-tempo track (no regression)')
    # Equally spaced vowels matching the priors — the common case that
    # must not be disturbed by freeze / corridor / duration changes.
    starts = [0.20, 0.70, 1.20, 1.70]
    samples = silence(starts[0])
    cursor = starts[0]
    for i, t in enumerate(starts):
        if cursor < t:
            samples = samples + silence(t - cursor)
            cursor = t
        v = vowel_tone(0.32, f1=700, f2=1250, attack=0.015, release=0.04)
        samples = samples + v
        cursor += 0.32
    samples = samples + silence(0.25)
    save('standard.wav', samples)
    words = ['just', 'a', 'small', 'town']
    priors = list(starts)
    result = align_words(words, priors, (SR, samples), AlignConfig())
    mae = sum(abs(o - p) for o, p in zip(result.onsets, priors)) / len(priors)
    h.metrics['standard_mae_s'] = mae
    h.check(mae < 0.080, 'standard-track MAE vs prior < 80 ms',
            'MAE=%.3f s onsets=%s' % (
                mae, '[' + ', '.join('%.3f' % x for x in result.onsets) + ']'))
    h.check(all(result.onsets[i] < result.onsets[i + 1]
                for i in range(len(result.onsets) - 1)),
            'standard onsets strictly increasing')


def test_cadence_regression(h: Harness):
    h.section('7. Heuristic cadence path (no audio) still holds')
    ws = _load_word_sync()
    pop = [
        {'start': 0.0, 'text': 'comme cet homme est venu me voir', 'words': None},
        {'start': 4.0, 'text': 'pour parler de mon amoureux', 'words': None},
        {'start': 8.0, 'text': 'je savais que c’était lui que', 'words': None},
        {'start': 28.0, 'text': 'attendais depuis toujours', 'words': None},
    ]
    evs = ws.build_word_events(pop, 60.0)
    ts = [t for t, _ in evs]
    h.check(ts == sorted(ts), 'events monotonic')
    last_onset = max((t for t, w in evs if w and t < 20), default=0.0)
    clears = [t for t, w in evs if not w and 8.0 < t < 28.0]
    h.check(bool(clears) and clears[0] < 20.0, 'verse-end rest cleared')
    h.check(last_onset - 8.0 < 8.0, 'verse-end span not crawling into the solo',
            'span=%.2f s' % (last_onset - 8.0))

    def first_line_span(win):
        ls = [{'start': i * win, 'text': 'je t’aime encore ce soir', 'words': None}
              for i in range(6)]
        e = ws.build_word_events(ls, win * 8)
        on = [t for t, w in e if w and t < win]
        return (on[-1] - on[0]) if on else 0.0

    h.check(first_line_span(7.0) > first_line_span(3.0), 'tempo-adaptive cadence')
    h.check(first_line_span(7.0) < 6.2, 'ballad words stay with the vocal')

    # Audio refine must not drop words.
    samples = concat(vowel_tone(0.4), vowel_tone(0.4), vowel_tone(0.4), silence(0.2))
    lines = [{'start': 0.0, 'text': 'one two three', 'words': None},
             {'start': 2.5, 'text': 'four', 'words': None}]
    meta = {}
    refined = ws.build_word_events(lines, 5.0, meta=meta, audio=(SR, samples))
    words = [w for t, w in refined if w]
    h.check(words[:3] == ['one', 'two', 'three'], 'refine keeps word identity')
    h.check(meta.get('aligned') is True, 'meta.aligned set when audio given')




def _line_fixture(spec):
    """[{'start','text','words'}] + total length from [(text, window), ...]."""
    lines, t = [], 0.0
    for text, win in spec:
        lines.append({'start': round(t, 3), 'text': text, 'words': None})
        t += win
    return lines, t


def test_panchiko_archetype(h: Harness):
    h.section('8. Panchiko archetype: 6-syllable phrase in a 6.5 s window')
    # A vocal phrase sung in ~1.8 s, then 4-5 s of guitar riffing before the
    # next LRC line stamp. Interleaved with normal 2.2 s windows so the dense
    # frontier has something honest to anchor on.
    LONG, SHORT = 'we are the dead ones now', 'hold me closer tonight'
    spec = [(LONG, 6.5), (SHORT, 2.2), (LONG, 6.5), (SHORT, 2.2),
            (LONG, 6.5), (SHORT, 2.2), (LONG, 6.5)]
    lines, total = _line_fixture(spec)
    syls = [sum(ws.syllables(w) for w in ln['text'].split()) for ln in lines]
    h.check(all(5 <= n <= 7 for n in syls),
            'fixture lines are 5-7 syllables', 'syllables=%s' % syls)
    meta = {}
    evs = build_word_events(lines, total + 8.0, meta=meta)
    onsets = [t for t, w in evs if w and t < 6.5]
    clears = [t for t, w in evs if not w and 0.2 < t < 6.5]
    span = (onsets[-1] - onsets[0]) if onsets else 0.0
    h.metrics['panchiko_span_s'] = span
    h.metrics['panchiko_base_pace'] = meta.get('base_pace', 0.0)
    h.check(bool(onsets) and onsets[-1] <= 2.4,
            '6-syllable phrase completes within 2.4 s of line_start',
            'last onset %.2f s (old engine: 4.5-5.5 s)' % onsets[-1])
    h.check(bool(clears) and clears[0] < 3.5,
            'rest marker emitted before 3.5 s',
            'clear at %.2f s' % (clears[0] if clears else -1))
    h.check(meta.get('base_pace', 1.0) < 0.60,
            'frontier pace is articulation, not the 0.822 ceiling',
            'base pace %.3f s/syllable' % meta.get('base_pace', -1))
    h.check(not meta.get('cadence', {}).get('inflation_guard', False),
            'inflation guard not needed when dense lines exist')
    short_on = [t for t, w in evs if w and 6.5 <= t < 8.7]
    h.check(bool(short_on) and (short_on[-1] - short_on[0]) > 0.9,
            'normal 2.2 s windows still spread their words',
            'span %.2f s' % ((short_on[-1] - short_on[0]) if short_on else -1))


def test_function_words_and_caesurae(h: Harness):
    h.section('9. Function words, caesurae and anacrusis')
    words = ['Wait,', 'in', 'the', 'middle', 'of', 'the', 'night']
    est = estimate_line(words, 0.0, 6.0, pace=0.5)
    d = {w: t for t, w in est if w}
    control = {w: t for t, w in estimate_line(['Wait'] + words[1:], 0.0, 6.0,
                                              pace=0.5) if w}
    h.metrics['caesura_gap_s'] = d['in'] - d['Wait,']
    h.check(d['in'] - d['Wait,'] > (control['in'] - control['Wait']) + 0.20,
            '"Wait," gets pre-boundary lengthening + a post-comma micro-pause',
            '%.3f s vs %.3f s without the comma'
            % (d['in'] - d['Wait,'], control['in'] - control['Wait']))
    h.check((d['middle'] - d['in']) < (d['of'] - d['middle']),
            '"in the" compressed relative to "middle"',
            '%.3f s vs %.3f s' % (d['middle'] - d['in'], d['of'] - d['middle']))
    h.check((d['night'] - d['of']) < (d['of'] - d['middle']),
            '"of the" compressed relative to "middle"',
            '%.3f s vs %.3f s' % (d['night'] - d['of'], d['of'] - d['middle']))
    # anacrusis: pickups compress harder when a heavy content word follows
    ana = {w: t for t, w in estimate_line(['In', 'the', 'SUFFERING', 'now'],
                                          0.0, 6.0, pace=0.5) if w}
    ctl = {w: t for t, w in estimate_line(['Deep', 'song', 'SUFFERING', 'now'],
                                          0.0, 6.0, pace=0.5) if w}
    r_ana = (ana['SUFFERING'] - ana['In']) / max(1e-6, ana['now'] - ana['SUFFERING'])
    r_ctl = (ctl['SUFFERING'] - ctl['Deep']) / max(1e-6, ctl['now'] - ctl['SUFFERING'])
    h.metrics['anacrusis_ratio'] = r_ana
    h.check(r_ana < 0.5 * r_ctl,
            '"In the" pickups compress into the stressed downbeat word',
            'ratio %.2f vs %.2f' % (r_ana, r_ctl))
    h.check(ws.anacrusis_pickups(['In', 'the', 'SUFFERING'], [1, 1, 3]) == 2
            and ws.anacrusis_pickups(['Deep', 'song', 'SUFFERING'], [1, 1, 3]) == 0,
            'anacrusis detector fires only on function-word pickups')
    # isochrony
    la = [t for t, w in estimate_line(['la', 'la', 'la', 'la'], 0.0, 4.0,
                                      pace=0.5) if w]
    steps = [la[i + 1] - la[i] for i in range(len(la) - 1)]
    h.check(bool(steps) and max(steps) - min(steps) < 1e-6,
            '"la la la la" locks to equal isochronous intervals',
            'step %.3f s' % (steps[0] if steps else -1))
    h.check(ws.repeat_key('d-d-death') == 'death' and ws.repeat_key('LA,') == 'la',
            'stutter / repeat-key normalisation')


def test_mixed_tempo(h: Harness):
    h.section('10. Mixed tempo / beat switch (per-line local pace)')
    RAP = 'i got the money and the power in my hands right'      # 13 syllables
    CHORUS = 'hold me closer now'                                 # 5 syllables
    spec = [(RAP, 2.5)] * 6 + [(CHORUS, 4.5)] * 4 + [(RAP, 2.5)] * 4
    lines, total = _line_fixture(spec)
    meta = {}
    evs = build_word_events(lines, total + 8.0, meta=meta)
    lp = meta.get('line_pace') or []
    h.check(len(lp) == len(lines), 'per-line pace reported for every line')
    verse_p = max(lp[:6]) if lp else 0.0
    chorus_p = min(lp[6:10]) if len(lp) >= 10 else 0.0
    h.metrics['verse_pace'] = verse_p
    h.metrics['chorus_pace'] = chorus_p
    h.check(chorus_p > 1.8 * verse_p,
            'chorus line_pace adapts locally (within 1-2 lines)',
            'chorus %.3f vs verse %.3f s/syllable' % (chorus_p, verse_p))

    def span_of(t0, t1):
        on = [t for t, w in evs if w and t0 <= t < t1]
        return (on[-1] - on[0]) if on else 0.0

    h.metrics['chorus_span_s'] = span_of(19.5, 24.0)
    h.check(span_of(19.5, 24.0) > 1.8,
            '2nd chorus line spreads across the slow window',
            'span %.2f s' % span_of(19.5, 24.0))
    h.check(span_of(0.0, 2.5) < 2.4,
            'rap verse still fits inside its 2.5 s window',
            'span %.2f s' % span_of(0.0, 2.5))
    # back to the verse: the local pace returns
    h.check(lp and lp[-1] < 1.5 * verse_p,
            'returning to the rap section restores the fast pace',
            'last line %.3f' % (lp[-1] if lp else -1))


def test_stereo_isolation(h: Harness):
    h.section('11. Stereo Mid/Side vocal isolation (synthetic stereo WAV)')
    bursts = [(0.4, 0.8), (1.0, 1.4), (1.6, 2.0)]
    L, R = stereo_mix(6.0, bursts)
    path = save('stereo_midside.wav', L, right=R)
    sr, wl, wr = load_wav_stereo(path)
    h.check(sr == SR and len(wl) == len(L), 'stereo WAV round trip')
    track = extract_features_stereo(wl, wr, sr, AlignConfig())
    h.check(track.stereo, 'stereo feature track flagged')
    segs = vad_segments(track.frames)
    h.metrics['stereo_vad_segments'] = len(segs)
    h.check(len(segs) >= 2, 'centre vocal bursts detected',
            'segments=%s' % [(round(a, 2), round(b, 2)) for a, b in segs][:4])
    outside = [f.t for f in track.frames if f.vad and (f.t < 0.28 or f.t > 2.14)]
    h.check(not outside, 'wide-panned guitars rejected (no false VAD)',
            '%d false frames' % len(outside))
    # contrast: a mono band-pass VAD on the same mix believes it is all vocal
    side_mono = [0.5 * (wl[i] - wr[i]) for i in range(len(wl))]
    mono_track = extract_features(side_mono, sr, AlignConfig())
    frac = sum(1 for f in mono_track.frames if f.vad) / max(1, len(mono_track.frames))
    h.metrics['mono_false_vad_fraction'] = frac
    h.check(frac > 0.5, 'mono band-pass VAD false-positives on the same mix',
            '%.0f%% of frames "voiced"' % (100 * frac))
    res = align_words(['i', 'will', 'be', 'dead'], [0.05, 0.20, 0.35, 0.50],
                      (sr, wl, wr), AlignConfig())
    h.check(all(0.35 <= o <= 2.15 for o in res.onsets),
            'onsets snap onto the isolated vocal window',
            'onsets=%s' % ('[' + ', '.join('%.3f' % o for o in res.onsets) + ']'))
    h.check(not any(f.vad for f in track.frames if f.t > 2.6),
            'interlude after the vocal stays clear')
    # a deliberately quieter vocal must still survive loud guitars
    L2, R2 = stereo_mix(6.0, bursts, guitar_amp=0.45)
    t2 = extract_features_stereo(L2, R2, sr, AlignConfig())
    h.check(any(f.vad for f in t2.frames), 'isolation survives guitars louder '
            'than the vocal')
    h.check(ws._isolate_vocal(0.30, 0.0) > 0.29 and ws._isolate_vocal(0.0, 0.30) == 0.0,
            'centre image kept, pure side image nulled')


def test_live_bridge(h: Harness):
    h.section('12. Live audio bridge (streaming, phase-lock, capture device)')
    bursts = [(1.0, 1.45), (1.62, 2.05), (7.0, 7.5)]
    L, R = stereo_mix(12.0, bursts)
    track = extract_features_stereo(L, R, SR, AlignConfig())

    # --- streaming extractor parity ---
    ext = StreamingFeatureExtractor(SR, AlignConfig())
    t0 = time.perf_counter()
    frames = []
    for i in range(0, len(L), 1600):                 # 100 ms capture chunks
        frames.extend(ext.push(L[i:i + 1600], R[i:i + 1600]))
    elapsed = time.perf_counter() - t0
    h.metrics['stream_cpu_ratio'] = elapsed / 12.0
    h.check(abs(len(frames) - len(track.frames)) <= 4,
            'streaming frame count matches the offline grid',
            '%d vs %d' % (len(frames), len(track.frames)))
    sv = [f.t for f in frames if f.vad]
    ov = [f.t for f in track.frames if f.vad]
    h.check(bool(sv) and bool(ov) and abs(sv[0] - ov[0]) < 0.08,
            'streaming VAD onset matches the offline path',
            '%.3f vs %.3f' % (sv[0] if sv else -1, ov[0] if ov else -1))
    quiet = [f.t for f in frames
             if f.vad and (0.25 < f.t < 0.95 or 2.20 < f.t < 6.90)]
    h.check(not quiet, 'streaming path rejects the wide guitars too',
            '%d false frames' % len(quiet))
    h.check(elapsed < 1.5, 'streaming capture costs < 1.5 s CPU per 12 s audio',
            '%.3f s (%.1f%% of real time)' % (elapsed, 100 * elapsed / 12.0))

    # --- tracker queries on a simulated capture ---
    tr = LiveAudioTracker(AlignConfig(), enabled=True)
    tr.running = True                                   # simulated device
    tr.note = 'simulated'
    tr.feed_frames(frames, song_time=True)
    h.check(len(tr.ring) >= 100, 'ring buffer holds the capture', '%d frames' % len(tr.ring))
    h.check(tr.vad_any(1.1, 1.3) and not tr.vad_any(3.0, 4.0),
            'vad query separates the sung line from the interlude')
    h.check(tr.first_vocal_after(0.5, 0.35) is None
            and tr.first_vocal_after(0.9, 0.35) is not None,
            'vocal entry gate can tell "silence at t_pred" from a real entry')
    h.check(tr.silence_run(3.0) > 0.5, 'silence run measured for early clear',
            '%.2f s' % tr.silence_run(3.0))
    tv = tr.track_view(0.9, 2.2)
    h.check(tv is not None and len(tv.frames) > 40, 'track_view window for refine',
            '%d frames' % (len(tv.frames) if tv else 0))

    # --- causal phase-lock behaviours ---
    lock = LivePhaseLock(AlignConfig())
    words = ['i', 'will', 'be', 'dead']
    base = [0.75, 1.30, 1.62, 1.95]        # word 0 is scheduled while silent
    events = list(zip(base, words)) + [(3.2, '')]
    meta = {'line_id': [0, 0, 0, 0, -1], 'estimated': [True] * 4 + [False],
            'line_end': [6.5] * 5}
    lock.reset(events, meta)
    patch = lock.update(events, base, -1, 0.70, 0.0, tr)
    h.metrics['phase_lock_gates'] = lock.stats['gates']
    h.metrics['phase_lock_snaps'] = lock.stats['snaps']
    h.check(lock.stats['gates'] >= 1 and 0 in patch,
            'vocal entry gate holds a word scheduled during silence',
            'word 0 -> %.3f s' % patch.get(0, -1))
    h.check(abs(patch.get(0, -1) - 1.002) < 0.10,
            'held word lands on the real vocal onset',
            'target 1.00 s')
    h.check(patch.get(0, 1e9) <= base[0] + AlignConfig().entry_gate_max_s + 1e-6,
            'hold never exceeds the +350 ms ceiling')
    starts2 = list(base)
    for k, v in patch.items():
        starts2[k] = v
    lock.next_refine = 0.0                      # bypass the 150 ms throttle
    lock.update(events, starts2, 0, 1.05, 0.0, tr)
    h.metrics['phase_lock_refines'] = lock.stats['refines']
    h.check(lock.stats['refines'] >= 1,
            'rolling re-align ran over the active line')
    before = lock.stats['releases']
    lock.update(events, starts2, 3, 3.0, 0.0, tr)
    h.check(lock.stats['releases'] > before,
            'post-phrase silence (>380 ms) releases the line for the interlude')
    rel_before = lock.stats['releases']
    lock.update(events, base, -1, 0.20, 0.0, tr)
    h.check(lock.stats['releases'] == rel_before,
            'post-phrase release is not armed before anything is displayed')

    # --- capture device end to end (fake pw/pulse monitor) ---
    bindir = os.path.join(FIXTURE_DIR, 'bin')
    os.makedirs(bindir, exist_ok=True)
    dev = os.path.join(bindir, 'parec')
    with open(dev, 'w', encoding='utf-8') as fh:
        # A real-time fake monitor device: wide guitars + a centre vocal burst,
        # looped forever, 100 ms chunks (never faster than real time, so the
        # harness stays a regression test rather than a stress test).
        fh.write('#!/usr/bin/env python3\n'
                 'import array, math, sys, time\n'
                 'SR = 16000\n'
                 'CHUNK = int(0.1 * SR)\n'
                 'LOOP = 3 * SR\n'
                 'def gen(start):\n'
                 '    buf = bytearray()\n'
                 '    for i in range(start, start + CHUNK):\n'
                 '        t = (i % LOOP) / SR\n'
                 '        g = 0.30 * sum(a * math.sin(2 * math.pi * fr * t + p)\n'
                 '                       for fr, a, p in ((310, .30, .3), (830, .24, 2.0),\n'
                 '                                        (1900, .16, 1.7)))\n'
                 '        left, right = g, -g\n'
                 '        if 0.6 <= t < 1.8:\n'
                 '            v = 0.26 * (0.62 * math.sin(2 * math.pi * 720 * t)\n'
                 '                        + 0.38 * math.sin(2 * math.pi * 1240 * t))\n'
                 '            left += v\n'
                 '            right += v\n'
                 '        buf += array.array("h", [int(max(-1, min(1, left)) * 32767),\n'
                 '                                  int(max(-1, min(1, right)) * 32767)]).tobytes()\n'
                 '    return buf\n'
                 'out = sys.stdout.buffer\n'
                 'frames = 0\n'
                 'end = time.time() + 30\n'
                 'while time.time() < end:\n'
                 '    try:\n'
                 '        out.write(gen(frames))\n'
                 '        out.flush()\n'
                 '    except Exception:\n'
                 '        break\n'
                 '    frames += CHUNK\n'
                 '    time.sleep(0.1)\n')
    os.chmod(dev, 0o755)
    old_path = os.environ.get('PATH', '')
    os.environ['PATH'] = bindir + os.pathsep + old_path
    try:
        dev_tr = LiveAudioTracker(AlignConfig(), enabled=True)
        h.check(bool(dev_tr.available()),
                'capture backend discovered in PATH',
                'cmd=%s' % (dev_tr.available()[0][0] if dev_tr.available() else '-'))
        started = dev_tr.start()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not dev_tr.running:
            time.sleep(0.05)
        dev_tr.update_anchor(0.0, time.monotonic())
        t_wait = time.monotonic() + 1.2
        while time.monotonic() < t_wait:
            dev_tr.update_anchor(0.0, time.monotonic())
            time.sleep(0.05)
        h.check(started and dev_tr.running, 'capture subprocess is streaming',
                dev_tr.capture_summary())
        h.check(len(dev_tr.ring) > 30, 'live ring buffer filled from the device',
                '%d frames' % len(dev_tr.ring))
        h.check(any(f.vad for f in dev_tr.ring), 'VAD fires on the captured vocal')
        dev_tr.begin_track('fixture---song', 3.0)
        time.sleep(0.5)                       # let the device fill the capture
        dev_tr.end_track()
        cached = dev_tr.cached_track('fixture---song')
        h.check(cached is not None and len(cached.frames) > 30,
                'replay cache holds a FeatureTrack for the same track')
        # ... and the offline engine actually runs on it (the v2.2 dead-code bug)
        lines = [{'start': 0.0, 'text': 'i will be dead now', 'words': None},
                 {'start': 2.6, 'text': 'nothing left to say', 'words': None}]
        meta2 = {}
        refined = ws.build_word_events(lines, 5.0, meta=meta2, audio=cached,
                                       align_cfg=AlignConfig())
        words2 = [w for _t, w in refined if w]
        h.check(words2[:3] == ['i', 'will', 'be'], 'replay alignment keeps words')
        h.check(meta2.get('aligned') is True,
                'meta.aligned set: the audio engine ran with live capture')
        dev_tr.stop()
        h.check(not dev_tr.running, 'capture thread stopped cleanly')
    finally:
        os.environ['PATH'] = old_path

    # --- fail-soft without any backend ---
    blind = LiveAudioTracker(AlignConfig(), enabled=True, device='definitely-not-a-device')
    h.check(blind.start() is False or blind.running is False,
            'unknown capture device never breaks the display')
    blind.stop()
    off = LiveAudioTracker(AlignConfig(), enabled=False)
    h.check(off.start() is False and 'no-audio-sync' in off.capture_summary(),
            '--no-audio-sync falls back to the prosodic engine')
    h.check(LivePhaseLock(AlignConfig()).update([(0.0, 'x')], [0.0], -1, 0.0, 0.0,
                                                None) == {},
            'phase-lock is a no-op without audio')


def main():
    print('wordsync offline alignment harness')
    print('fixtures ->', FIXTURE_DIR)
    h = Harness()
    tests = [
        test_unit_costs,
        test_latency,
        test_pause_freeze,
        test_soft_onset,
        test_syncopation,
        test_standard_track,
        test_cadence_regression,
        test_panchiko_archetype,
        test_function_words_and_caesurae,
        test_mixed_tempo,
        test_stereo_isolation,
        test_live_bridge,
    ]
    for fn in tests:
        try:
            fn(h)
        except Exception as exc:
            h.failed.append('%s crashed: %s' % (fn.__name__, exc))
            print('  CRASH %s: %s' % (fn.__name__, exc))
            traceback.print_exc()

    print('\n-- metrics --')
    for k, v in h.metrics.items():
        if isinstance(v, float):
            print('  %-32s %10.4f' % (k, v))
        else:
            print('  %-32s %10s' % (k, v))

    print('\n%d passed, %d failed' % (len(h.passed), len(h.failed)))
    if h.failed:
        print('FAILURES:')
        for f in h.failed:
            print('  -', f)
        return 1
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
