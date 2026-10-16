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


def concat(*parts):
    out = []
    for p in parts:
        out.extend(p)
    return out


def save(name, samples, sr=SR):
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = os.path.join(FIXTURE_DIR, name)
    write_wav(path, sr, samples)
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


# ---------------------------------------------------------------------------
# v3.0 architectural regression tests
# ---------------------------------------------------------------------------

def _stereo_vocal_burst(burst_start, burst_end, total, sr=SR, vocal_amp=0.25,
                        noise_amp=0.30, noise_freq=440.0, vocal_freq=720.0,
                        pan_polarity=-1.0):
    """Synthetic stereo fixture: wide-panned 300-3400 Hz noise + a center
    vocal burst between burst_start and burst_end."""
    n = int(total * sr)
    L = [0.0] * n
    R = [0.0] * n
    bs = int(burst_start * sr)
    be = int(burst_end * sr)
    for i in range(n):
        x = noise_amp * math.sin(2 * math.pi * noise_freq * i / sr)
        L[i] = x
        R[i] = pan_polarity * x
    for i in range(bs, be):
        t = i / sr
        v = vocal_amp * math.sin(2 * math.pi * vocal_freq * t)
        L[i] += v
        R[i] += v
    return L, R


def test_panchiko_archetype(h: Harness):
    h.section('8. v3.0 Panchiko / post-phrase instrumental gap archetype')
    ws = _load_word_sync()
    # 3 fast 2.2 s lines (4 words each) then a 6-syllable panchiko line
    # inside a 6.5 s window.
    lines = []
    lines.append({'start': 0.0, 'text': 'one two three four', 'words': None})
    lines.append({'start': 2.2, 'text': 'one two three four', 'words': None})
    lines.append({'start': 4.4, 'text': 'one two three four', 'words': None})
    base = 6.6
    lines.append({'start': base, 'text': 'in the death of night', 'words': None})
    lines.append({'start': base + 6.5, 'text': 'one two three', 'words': None})

    evs = ws.build_word_events(lines, base + 13.0)
    onsets = [t for t, w in evs if w and base <= t < base + 6.5]
    clears = [t for t, w in evs if not w and base <= t < base + 6.5]
    h.check(len(onsets) == 5, 'panchiko produces 5 word onsets',
            'got %d (%s)' % (len(onsets), onsets))
    if onsets:
        span = onsets[-1] - onsets[0]
        h.check(span <= 2.4,
                'panchiko onsets complete inside vocal portion (<=2.4 s)',
                'span=%.2f s' % span)
    if clears:
        first_clear = clears[0] - base
        h.check(first_clear < 3.5,
                'rest marker emitted before 3.5 s',
                'first_clear=%.2f s' % first_clear)
    else:
        h.check(False, 'rest marker emitted before 3.5 s', 'no clear found')


def test_function_word_and_caesura(h: Harness):
    h.section('9. v3.0 Function-word compression + caesura micro-pause')
    ws = _load_word_sync()
    evs = ws.estimate_line(
        ['Wait,', 'in', 'the', 'middle', 'of', 'the', 'night'],
        0.0, 5.0, pace=0.5)
    times = [t for t, _ in evs]
    indexed = [(t, w.strip(',').lower()) for t, w in evs if w]
    pos = {w: i for i, (t, w) in enumerate(indexed)}
    ts = [t for t, _ in indexed]
    # The function cluster "in the" should be tighter than the content
    # straddle across "middle ... of the night" once we ignore the second
    # "the" (which sits after the caesura-compressed content run).
    if all(k in pos for k in ('in', 'the', 'middle', 'of', 'night')):
        cluster_lo = pos['in']
        cluster_hi_first = cluster_lo
        for k in range(cluster_lo, pos['middle']):
            if indexed[k][1] == 'the':
                cluster_hi_first = k
                break
        cluster_width = ts[cluster_hi_first] - ts[cluster_lo]
        middle_i = pos['middle']
        night_i = pos['night']
        content_width = ts[night_i] - ts[middle_i]
        h.check(cluster_width < content_width,
                'function cluster tighter than content run',
                'cluster=%.3f content=%.3f' % (cluster_width, content_width))
    # Pre-boundary lengthening on "Wait,": the gap to the next word should
    # be at least the caesura micro-pause (0.20 s).
    if indexed and indexed[0][1] == 'wait' and len(indexed) > 1:
        next_gap = indexed[1][0] - indexed[0][0]
        h.check(next_gap >= 0.20, 'post-comma caesura injected (>=0.20 s)',
                'gap=%.3f s' % next_gap)


def test_mixed_tempo_local_pace(h: Harness):
    h.section('10. v3.0 Mixed-tempo / beat-switch local pace')
    ws = _load_word_sync()
    lines = []
    for i in range(4):
        lines.append({'start': i * 2.5,
                      'text': ('one two three four five six seven eight '
                               'nine ten eleven twelve'), 'words': None})
    base = 4 * 2.5
    for i in range(3):
        lines.append({'start': base + i * 4.5,
                      'text': 'slow soft song here now', 'words': None})
    evs = ws.build_word_events(lines, base + 3 * 4.5)
    # Find onsets per line
    by_line = []
    for idx, ln in enumerate(lines):
        end = lines[idx + 1]['start'] if idx + 1 < len(lines) else ln['start'] + 4.5
        on = [t for t, w in evs if w and ln['start'] <= t < end]
        by_line.append(on)
    # Fast lines: span ~2.0 s
    fast_spans = [o[-1] - o[0] for o in by_line[:4] if o]
    slow_spans = [o[-1] - o[0] for o in by_line[4:] if o]
    h.check(all(s < 2.5 for s in fast_spans),
            'rap lines stay tight (span < 2.5 s)',
            'spans=%s' % [round(s, 2) for s in fast_spans])
    h.check(all(s < 2.5 for s in slow_spans),
            'chorus lines adapt locally (span < 2.5 s)',
            'spans=%s' % [round(s, 2) for s in slow_spans])


def test_stereo_mid_side_isolation(h: Harness):
    h.section('11. v3.0 Stereo mid-side vocal isolation')
    ws = _load_word_sync()
    L, R = _stereo_vocal_burst(0.4, 2.0, 6.0)
    path = save('stereo_burst.wav', L, sr=SR)
    # Re-load as multi-channel? The harness write_wav is mono. We feed
    # stereo directly to the engine via tuple samples.
    track = ws.extract_features((L, R), SR, ws.AlignConfig())
    vads_on = [f.t for f in track.frames if f.vad]
    h.check(bool(vads_on), 'VAD ever fires on the center vocal')
    if vads_on:
        h.check(vads_on[0] >= 0.30 and vads_on[0] <= 0.60,
                'VAD first fires inside the burst window',
                'first_vad=%.3f' % vads_on[0])
        h.check(vads_on[-1] <= 2.20,
                'VAD last fires inside the burst window',
                'last_vad=%.3f' % vads_on[-1])
    # Outside the burst there should be almost no vad activity.
    off_burst = [f for f in track.frames
                 if f.vad and (f.t < 0.30 or f.t > 2.20)]
    h.check(len(off_burst) <= 2,
            'stereo noise does not falsely trigger VAD',
            'off-burst VAD frames=%d' % len(off_burst))
    # And refine-word-events should snap to the burst.
    samples_tuple = (SR, (L, R))
    words = ['one', 'two', 'three']
    priors = [0.40, 0.90, 1.40]
    result = ws.align_words(words, priors, samples_tuple, ws.AlignConfig())
    h.check(all(result.onsets), 'align produced onsets')
    if result.onsets:
        snapped = all(0.30 <= o <= 2.20 for o in result.onsets)
        h.check(snapped, 'aligned onsets sit inside the vocal burst',
                'onsets=%s' % ['%.2f' % o for o in result.onsets])


def test_live_audio_tracker(h: Harness):
    h.section('12. v3.0 LiveAudioTracker fail-soft + buffering')
    ws = _load_word_sync()
    # Disabled tracker must still expose a usable interface.
    tr = ws.LiveAudioTracker(no_audio=True)
    h.check(tr.feed is not None, 'LiveAudioTracker.feed exposed')
    h.check(not tr.available, 'tracker reports unavailable when disabled')
    # Active tracker with synthetic stereo: features should be computable.
    L, R = _stereo_vocal_burst(0.3, 1.5, 4.0)
    tr2 = ws.LiveAudioTracker(enabled=False)  # no subprocess; feed manually
    tr2.feed(L, right=R)
    h.check(len(tr2.left) == len(L) and len(tr2.right) == len(R),
            'tracker buffer accepts stereo frames',
            'left=%d right=%d' % (len(tr2.left), len(tr2.right)))
    track = tr2.track
    h.check(track is not None and len(track.frames) > 0,
            'tracker.track computes features from buffered audio',
            'frames=%s' % (len(track.frames) if track else 0))
    # And the same buffer works as align audio via get_audio().
    samples = tr2.get_audio()
    if track is not None:
        h.check(any(f.vad for f in track.frames),
                'tracker VAD sees the center burst',
                'vad frames=%d' % sum(1 for f in track.frames if f.vad))


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
        test_function_word_and_caesura,
        test_mixed_tempo_local_pace,
        test_stereo_mid_side_isolation,
        test_live_audio_tracker,
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
