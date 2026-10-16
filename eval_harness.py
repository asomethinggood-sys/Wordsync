#!/usr/bin/env python3
"""Deterministic timing benchmark with explicit synthetic ground truth.

This is an engine evaluation, not a claim about real-track accuracy: line anchors, spoken
syllable onsets, rests, tempo changes, and audio bursts are generated locally with no web APIs.
The printed MAE/p90/max makes regressions visible without external datasets or dependencies.
"""
import math
import re
import struct

import wordsync_v3 as ws


# Hand-labelled truth for the five fixed test phrases; deliberately not computed by the
# production syllable, function-word, or punctuation helpers.
TRUTH_SYLLABLES = {
    'we': 1, 'can': 1, 'see': 1, 'light': 1, 'tonight': 2, 'beat': 1,
    'will': 1, 'stay': 1, 'inside': 2, 'wait': 1, 'in': 1, 'the': 1,
    'middle': 2, 'of': 1, 'night': 1,
}
TRUTH_FUNCTION_WORDS = frozenset(('we', 'can', 'will', 'in', 'the', 'of'))


def _truth_offsets(words, pace):
    """Synthetic singer truth: hand-labelled syllables, pickups, and a comma caesura."""
    elapsed, offsets = 0.0, []
    for i, word in enumerate(words):
        offsets.append(elapsed)
        if i == len(words) - 1:
            break
        text = re.sub(r'[^a-z]', '', word.lower())
        syllables = TRUTH_SYLLABLES[text]
        function_scale = 0.60 if text in TRUTH_FUNCTION_WORDS else 1.0
        elapsed += pace * (syllables ** 0.82) * function_scale
        # Simulate a natural small micro-pause after a comma independently of the engine.
        if word.rstrip().endswith(','):
            elapsed += 0.16
    return offsets


def _run_scenario(name, specs):
    lines, expected, cursor = [], [], 0.0
    for text, pace, window in specs:
        words = ws.line_words(text)
        lines.append({'start': cursor, 'text': text, 'words': None})
        expected.append([cursor + dt for dt in _truth_offsets(words, pace)])
        cursor += window
    meta = {}
    ws.build_word_events(lines, cursor + 1.0, meta=meta)

    errors = []
    for spec, gold in zip(meta['line_specs'], expected):
        predicted = [t for t, word in spec['events'] if word]
        if len(predicted) != len(gold):
            raise AssertionError('{} changed token count: {} != {}'.format(
                name, len(predicted), len(gold)))
        errors.extend(abs(p - g) for p, g in zip(predicted, gold))
    errors.sort()
    if not errors:
        raise AssertionError('{} produced no word events'.format(name))
    p90 = errors[min(len(errors) - 1, int(math.ceil(0.90 * len(errors))) - 1)]
    metrics = {'name': name, 'words': len(errors), 'mae': sum(errors) / len(errors),
               'p90': p90, 'max': errors[-1], 'base_pace': meta['base_pace']}
    print('{name:28s} words={words:4d} MAE={mae:.3f}s p90={p90:.3f}s '
          'max={max:.3f}s base={base_pace:.3f}'.format(**metrics))
    return metrics, meta, expected


def _synthetic_vocal_pcm(start=0.4, end=2.0, duration=6.0, sample_rate=16000):
    """Center voice burst over wide side-only guitar-like formant noise."""
    pcm = bytearray()
    for sample in range(int(duration * sample_rate)):
        t = sample / float(sample_rate)
        side = 0.18 * (math.sin(2 * math.pi * 440 * t) +
                       0.45 * math.sin(2 * math.pi * 900 * t) +
                       0.25 * math.sin(2 * math.pi * 2800 * t))
        if start <= t < end:
            phase = t - start
            envelope = 0.55 + 0.45 * max(0.0, math.sin(2 * math.pi * 3.0 * phase))
            vocal = 0.35 * envelope * math.sin(
                2 * math.pi * (620 + 40 * math.sin(2 * math.pi * 4 * phase)) * t)
        else:
            vocal = 0.0
        left, right = side + vocal, -side + vocal
        pcm.extend(struct.pack('<hh', int(max(-1.0, min(1.0, left)) * 32767),
                               int(max(-1.0, min(1.0, right)) * 32767)))
    return bytes(pcm)


def main():
    panchiko_phrase = 'we can see light tonight'  # six syllables; rest dominates 6.5 s tags
    rap_phrase = ' '.join(['beat'] * 12)
    chorus_phrase = 'we will stay inside tonight'
    scenarios = [
        ('Panchiko-like sparse', [(panchiko_phrase, 0.30, 6.5)] * 16),
        ('Panchiko-like mixed', [(panchiko_phrase, 0.30, w) for w in
                                [6.5, 2.2, 6.4, 2.3, 6.5, 2.4, 6.3, 2.2,
                                 6.5, 2.2, 6.4, 2.3, 6.5, 2.2, 6.2, 2.4]]),
        ('Fast rap', [(rap_phrase, 0.20, 2.5)] * 16),
        ('Tempo-switch verse/chorus',
         [(rap_phrase, 0.20, 2.5)] * 8 + [(chorus_phrase, 0.48, 4.5)] * 8),
        ('Punctuation caesura', [('Wait, in the middle of the night', 0.34, 4.8)] * 12),
    ]

    all_metrics = []
    for name, specs in scenarios:
        metrics, meta, expected = _run_scenario(name, specs)
        all_metrics.append(metrics)
        if name.startswith('Panchiko'):
            for spec in meta['line_specs']:
                line_start = spec['start']
                onsets = [t for t, word in spec['events'] if word]
                clears = [t for t, word in spec['events'] if not word]
                assert onsets[-1] - line_start <= 2.4, (name, onsets[-1] - line_start)
                if spec['end'] - line_start >= 3.5:
                    assert clears and clears[0] - line_start < 3.5, (name, clears)

    # Direct acoustic-feature benchmark, independent of the line-gap-only estimator.
    audio = ws.extract_features(_synthetic_vocal_pcm(), start_position=0.0)
    active = [f.song_position for f in audio if f.vad]
    assert active and min(active) >= 0.35 and max(active) <= 2.05
    aligned = ws.align_tokens(['wait', 'for', 'the', 'light'], audio, 0.0, 6.0, 0.36)
    acoustic_onsets = [t for t, word in aligned if word]
    assert 0.35 <= acoustic_onsets[0] <= 0.50
    assert acoustic_onsets[-1] <= 2.05
    print('Synthetic stereo VAD: active={:.3f}..{:.3f}s; aligned={}'.format(
        min(active), max(active), ', '.join('{:.3f}'.format(t) for t in acoustic_onsets)))

    overall = [m for m in all_metrics]
    print('SUMMARY: {} scenarios, {} synthetic word onsets; weighted MAE={:.3f}s, worst p90={:.3f}s'
          .format(len(overall), sum(m['words'] for m in overall),
                  sum(m['mae'] * m['words'] for m in overall) /
                  sum(m['words'] for m in overall), max(m['p90'] for m in overall)))


if __name__ == '__main__':
    main()
