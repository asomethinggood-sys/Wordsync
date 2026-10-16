#!/usr/bin/env python3
"""Offline regression tests for Wordsync v3.0's prosodic and acoustic alignment engine."""
import math
import struct
import unittest

import word_sync as ws


class ProsodicCadenceTests(unittest.TestCase):
    def make_lines(self, windows, text='we can see light tonight'):
        lines, cursor = [], 0.0
        for window in windows:
            lines.append({'start': cursor, 'text': text, 'words': None})
            cursor += window
        return lines, cursor

    def test_panchiko_instrumental_gap_uses_dense_frontier(self):
        windows = [6.5, 2.2, 6.2, 2.4, 6.5, 2.1, 6.0, 2.3, 6.4, 2.2]
        lines, duration = self.make_lines(windows)
        meta = {}
        ws.build_word_events(lines, duration + 2.0, meta=meta)
        spec = meta['line_specs'][0]
        onsets = [t for t, word in spec['events'] if word]
        clears = [t for t, word in spec['events'] if not word]
        self.assertEqual(sum(ws.syllables(token) for token in ws.line_words(lines[0]['text'])), 6)
        self.assertLessEqual(onsets[-1] - spec['start'], 2.4)
        self.assertTrue(clears)
        self.assertLess(clears[0] - spec['start'], 3.5)
        self.assertLessEqual(meta['base_pace'], 0.40)
        self.assertTrue(all(meta['line_paces'][i] <= 0.38 for i in (0, 2, 4, 6, 8)))

        # Even when only two dense anchors occur in ten lines, preserve the fast frontier.
        sparse_lines, sparse_duration = self.make_lines(
            [6.5, 6.5, 2.2, 6.5, 6.5, 6.5, 2.3, 6.5, 6.5, 6.5])
        sparse_meta = {}
        ws.build_word_events(sparse_lines, sparse_duration + 2.0, meta=sparse_meta)
        self.assertLessEqual(sparse_meta['base_pace'], 0.40)
        for i in (0, 1, 3, 4, 5, 7, 8, 9):
            spec = sparse_meta['line_specs'][i]
            sparse_onsets = [t for t, word in spec['events'] if word]
            self.assertLessEqual(sparse_onsets[-1] - spec['start'], 2.4)

    def test_final_sustain_does_not_delay_word_onsets(self):
        words = ws.line_words('all the words arrive tonight')
        pace = 0.58
        events = ws.estimate_line(words, 4.0, 16.0, pace)
        times = [t for t, word in events if word]
        articulation, _ = ws._prosodic_durations(words, pace, 12.0)
        final_hold = min(ws.TUNE['hold_cap'], ws.TUNE['hold_a'] + ws.TUNE['hold_b'] * pace)
        self.assertEqual(times[0], 4.0)
        self.assertLessEqual(times[-1] - times[0], sum(articulation[:-1]) + 1e-9)
        clear = next(t for t, word in events if not word)
        self.assertAlmostEqual(clear, times[-1] + articulation[-1] + final_hold, places=6)

    def test_function_words_and_caesura(self):
        phrase = ws.line_words('Wait, in the middle of the night')
        timed = ws.estimate_line(phrase, 0.0, 4.8, 0.45)
        plain = ws.estimate_line(ws.line_words('Wait in the middle of the night'),
                                 0.0, 4.8, 0.45)
        timed_positions = [t for t, word in timed if word]
        plain_positions = [t for t, word in plain if word]
        self.assertGreater(timed_positions[1] - timed_positions[0],
                           plain_positions[1] - plain_positions[0] + 0.14)
        articulation, pauses = ws._prosodic_durations(phrase, 0.45, 4.8)
        for function_index in (1, 2, 4, 5):
            self.assertLess(articulation[function_index], articulation[3])
        self.assertGreater(pauses[0], 0.14)

        backing = ws.line_words('hold (backing vocals) the line')
        backing_articulation, backing_pauses = ws._prosodic_durations(backing, 0.45, 5.0)
        self.assertGreater(backing_pauses[0], 0.14)
        self.assertGreater(backing_articulation[0], 0.45 * (1 ** ws.TUNE['syl_exp']))

    def test_local_pace_tracks_rapid_verse_to_slow_chorus(self):
        lines, cursor = [], 0.0
        for i in range(16):
            rapid = i < 8
            lines.append({'start': cursor,
                          'text': ' '.join(['beat'] * (12 if rapid else 5)),
                          'words': None})
            cursor += 2.5 if rapid else 4.5
        meta = {}
        ws.build_word_events(lines, cursor + 2.0, meta=meta)
        paces = meta['line_paces']
        self.assertLessEqual(paces[7], 0.35)
        self.assertGreater(paces[9], paces[7] * 1.5)
        self.assertGreater(paces[9], paces[8])

    def test_repetition_uses_isochronous_intervals(self):
        events = ws.estimate_line(['la', 'la', 'la', 'fire'], 0.0, 5.0, 0.42)
        times = [t for t, word in events if word]
        self.assertAlmostEqual(times[1] - times[0], times[2] - times[1], places=7)


class StereoAudioTests(unittest.TestCase):
    @staticmethod
    def synthetic_mix(duration=6.0, sample_rate=16000):
        pcm = bytearray()
        for sample in range(int(sample_rate * duration)):
            t = sample / float(sample_rate)
            # Wide, side-only guitar-like energy across the formant band.
            side = 0.18 * (math.sin(2 * math.pi * 440 * t) +
                           0.45 * math.sin(2 * math.pi * 900 * t) +
                           0.25 * math.sin(2 * math.pi * 2800 * t))
            if 0.4 <= t < 2.0:
                phase = t - 0.4
                envelope = 0.55 + 0.45 * max(0.0, math.sin(2 * math.pi * 3.0 * phase))
                vocal = 0.35 * envelope * math.sin(
                    2 * math.pi * (620 + 40 * math.sin(2 * math.pi * 4 * phase)) * t)
            else:
                vocal = 0.0
            left, right = side + vocal, -side + vocal
            pcm.extend(struct.pack('<hh', int(max(-1.0, min(1.0, left)) * 32767),
                                   int(max(-1.0, min(1.0, right)) * 32767)))
        return bytes(pcm)

    def test_mid_side_vad_rejects_wide_instruments_and_aligns_voice(self):
        pcm = self.synthetic_mix()
        tracker = ws.LiveAudioTracker(enabled=False)
        frames = tracker.feed_pcm(pcm, song_position=0.0)
        self.assertEqual(len(tracker.snapshot()), len(frames))
        active = [f.song_position for f in frames if f.vad]
        self.assertTrue(active)
        self.assertGreaterEqual(min(active), 0.35)
        self.assertLessEqual(max(active), 2.05)
        self.assertFalse(any(f.vad for f in frames
                             if f.song_position < 0.35 or f.song_position > 2.15))
        self.assertTrue(any(f.side_rms > 0.05 for f in frames if f.song_position < 0.35))

        aligned = ws.align_tokens(['wait', 'for', 'the', 'light'], frames, 0.0, 6.0, 0.45)
        word_onsets = [t for t, word in aligned if word]
        self.assertGreaterEqual(word_onsets[0], 0.35)
        self.assertLessEqual(word_onsets[0], 0.50)
        self.assertLessEqual(word_onsets[-1], 2.05)

        # Exercise the live event hook: the event builder accepts rolling feature frames.
        lines = [{'start': 0.0, 'text': 'wait for the light', 'words': None}]
        meta = {}
        events = ws.build_word_events(lines, 6.0, meta=meta, audio=frames)
        self.assertAlmostEqual(next(t for t, word in events if word), word_onsets[0], places=6)
        tracker.stop()

    def test_live_entry_gate_waits_then_snaps_to_causal_onset(self):
        frames = ws.extract_features(self.synthetic_mix(), start_position=0.0)
        base = [(0.0, 'wait'), (0.7, 'for'), (1.2, 'the'), (1.6, 'light'), (1.8, '')]
        held = ws.refine_word_events(base, frames, 0.0, 6.0, now=0.20)
        self.assertGreater(held[0][0], 0.20)  # no early pop during the instrumental intro
        observed = ws.refine_word_events(base, frames, 0.0, 6.0, now=0.42)
        self.assertAlmostEqual(observed[0][0], 0.4, delta=0.06)
        self.assertEqual([w for _, w in observed], [w for _, w in base])
        frozen = ws.refine_word_events(base, frames, 0.0, 6.0, now=2.20)
        self.assertAlmostEqual(frozen[-1][0], 2.38, delta=0.06)

    def test_audio_can_be_disabled_without_capture_tools(self):
        tracker = ws.LiveAudioTracker(enabled=False)
        self.assertFalse(tracker.start())
        self.assertEqual(tracker.error, 'disabled')
        tracker.stop()


if __name__ == '__main__':
    unittest.main()
