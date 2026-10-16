# Wordsync

Lyrics sync at word level. Designed and published for Linux. BETA.

Zero external Python dependencies (stdlib + `playerctl`). Single file:
`word-sync(2).py` — display, LRC parser, trained cadence model, the offline
audio-to-lyric alignment engine, and the live audio sync bridge.

## Run

```bash
python3 word-sync\(2\).py              # live display (needs playerctl)
python3 word-sync\(2\).py --selftest   # offline regressions
python3 word-sync\(2\).py --plain
python3 word-sync\(2\).py --offset -0.10
python3 word-sync\(2\).py --no-audio-sync   # prosodic heuristic only
python3 test_align_engine.py          # synthetic-WAV alignment harness
```

## v3.0 — Precision Hybrid Prosodic & Live-Audio Sync Engine

### Architecture

1. **Dense-Frontier & Local-Cadence Estimator** (`build_word_events`).
   Anchors the song's true articulation pace on the 25th–30th percentile of
   per-line raw syllable rate × voiced fraction (singer's fastest back-to-back
   gapless lines) and blends a 5-line rolling local pace for mid-song tempo
   switches. Lines whose rate spikes >1.6× the base are classified as
   post-phrase instrumental gaps / final-vowel sustain — their line pace is
   *not* inflated.

2. **Onset-vs-Sustain Math Fix** (`estimate_line`).
   `onset_span` (time from word[0] onset to word[n-1] onset) is driven by the
   cumulative prosodic durations of words 0..n-2 plus intra-line caesura
   micro-pauses, bounded by `limit - last_word_min_dur`. `clear_at` is placed
   at `times[-1] + last_word_articulation + final_hold`. The post-phrase tail
   compression threshold is dropped so the words lock to `onset_span` at the
   start of the line and a rest marker is emitted even on 3.5–6.5 s windows.

3. **Prosodic & Rhythmic Cadence Model** (Tier 2.5 heuristic, in
   `estimate_line`).
   * Multilingual **function-word compression** (EN/ES/FR/PT/DE/RU +
     particles); unstressed pickups receive a 0.50×–0.62× duration
     multiplier, *unless* they sit at clause end.
   * **Intra-line caesurae**: pre-boundary vowel lengthening (1.25×) +
     additive micro-pause (0.20 s) after `, ; : — – - … ? ! ) ]` or before
     a parenthetical.
   * **Isochronous repetition lock**: `"la la la"`, `"yeah yeah yeah"`,
     `"d-d-death"` runs collapse to equal-weight tokens so the non-linear
     front exponent cannot warp them.
   * **Syncopation / anacrusis**: 1–2 leading function words before a heavy
     multi-syllable content word are compressed to 0.50× so the stressed
     word lands cleanly on the downbeat.

4. **Live Audio Sync Bridge** (`LiveAudioTracker`).
   Zero-dependency fail-soft bridge: detects `pw-cat` / `pw-record` /
   `parec` in `PATH`; spawns a daemon thread reading raw s16le 16 kHz
   stereo PCM from `@DEFAULT_MONITOR@`. Falls back to the prosodic
   heuristic engine when capture is unavailable or `--no-audio-sync` is
   passed. `feed()` / `process()` / `track` / `get_audio()` make it usable
   from the tests and from any future tight-loop integration.

5. **Stereo Mid-Side Vocal Isolation** (in `extract_features`).
   Centre-channel vocal energy is `max(0, mid_formant −
   side_cancel_gain * excess(side_formant))`; the *excess* above mid is
   the stereo-only component, so pure-mono audio passes through untouched
   while wide-panned guitars / cymbals / synths are cancelled without
   eating the lead vocal.

## Offline engine (v3.0)

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

Without audio, the v3 trained cadence model is unchanged in spirit but
now uses the dense-frontier + per-line local pace. With a WAV:

```python
events = build_word_events(lines, duration, audio='vocal.wav',
                           align_cfg=AlignConfig(calibration_offset_s=0.0))
```

## Test coverage

`test_align_engine.py` includes Panchiko D>E>A>T>H>M>E>T>A>L archetype
(post-phrase instrumental gap), function-word / caesura, mixed-tempo /
beat-switch, stereo mid-side isolation, and LiveAudioTracker buffering
regressions on top of the v2.2 unit + cadence checks.
