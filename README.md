# Wordsync

Lyrics sync at word level. Designed and published for Linux. BETA.

Zero external Python dependencies (stdlib + `playerctl`). v2.2 adds a
**pure-offline audio-to-lyric alignment engine** for ballads, breathy
onsets, and syncopation.

## Layout

| File | Role |
|------|------|
| `word-sync(2).py` | Display app, LRC parser, trained cadence model (`estimate_line` / `build_word_events`) |
| `wordsync_align.py` | Core offline alignment: phone-duration HMM, VAD freeze, formant onsets, adaptive corridor |
| `test_align_engine.py` | Synthetic-WAV harness (latency, pause drift, onsets) |

Token decoding, duration modelling, and the search live in
`wordsync_align.py` (`align_tokens`, `stay_cost`, `advance_cost`).
Onset detection and VAD live in `extract_features`. Time-warping is the
duration-aware beam in `align_tokens`, seeded by the heuristic timeline
from `build_word_events` in `word-sync(2).py`.

## Offline engine (v2.2)

1. **Duration modelling** — vowels (`/a i u …/`) get nearly-free self-loops
   (melisma); plosives/fricatives hit a hard duration ceiling.
2. **Pause freeze** — if vocal-band energy is below threshold for >400 ms
   the lyric pointer cannot advance (instrumental breaks, reverb tails).
3. **Soft onsets** — attacks are scored on 300 Hz–3.4 kHz formant *rise
   rate*, not full-band transients (breathy vocals).
4. **Adaptive corridor** — search window tightens with confidence, widens
   on dips / syncopation. No tempo-grid penalty on off-beat hits.
5. **Lookahead** — analysis window capped at ≤150 ms; `calibration_offset_s`
   cancels STFT / hop delay.

When no audio is passed, the v2 trained cadence model is unchanged (span
and verse-end rest behaviour). Pass `audio=` to `build_word_events` to
refine:

```python
from wordsync_align import AlignConfig
events = build_word_events(lines, duration, audio='vocal.wav',
                           align_cfg=AlignConfig(calibration_offset_s=0.0))
```

## Tests

```bash
python3 word-sync\(2\).py --selftest
python3 test_align_engine.py
python3 wordsync_align.py          # unit checks only
```
