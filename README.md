# Wordsync

Lyrics sync at word level. Designed and published for Linux. BETA.

Zero external Python dependencies (stdlib + `playerctl`). Single file:
`word-sync(2).py` — display, LRC parser, trained cadence model, and the
offline audio-to-lyric alignment engine.

## Run

```bash
python3 word-sync\(2\).py              # live display (needs playerctl)
python3 word-sync\(2\).py --selftest   # offline regressions
python3 word-sync\(2\).py --plain
python3 word-sync\(2\).py --offset -0.10
python3 test_align_engine.py          # synthetic-WAV alignment harness
```

## Offline engine (v2.2)

Lives in `word-sync(2).py` (`extract_features`, `align_tokens`,
`build_word_events`).

1. **Duration modelling** — vowels get nearly-free self-loops (melisma);
   plosives/fricatives hit a hard duration ceiling.
2. **Pause freeze** — if vocal-band energy is below threshold for >400 ms
   the lyric pointer cannot advance.
3. **Soft onsets** — 300 Hz–3.4 kHz formant *rise rate*, not full-band clicks.
4. **Adaptive corridor** — tight when confident, wide on dips / syncopation.
   No tempo-grid penalty on off-beat hits.
5. **Lookahead** — analysis window ≤150 ms; `calibration_offset_s` cancels
   STFT / hop delay.

Without audio, the v2 trained cadence model is unchanged. With a WAV:

```python
events = build_word_events(lines, duration, audio='vocal.wav',
                           align_cfg=AlignConfig(calibration_offset_s=0.0))
```
