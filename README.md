# Wordsync

Lyrics sync at word level. Designed and published for Linux. BETA.

Zero external Python dependencies (stdlib + `playerctl`). Single file:
`word-sync(2).py` — display, LRC parser, prosodic cadence model, offline
audio-to-lyric alignment engine, and the live monitor-capture bridge.

## Run

```bash
python3 word-sync\(2\).py              # live display (needs playerctl)
python3 word-sync\(2\).py --plain      # pipe mode (bare words)
python3 word-sync\(2\).py --no-audio-sync   # heuristic engine only
python3 word-sync\(2\).py --selftest   # offline regressions
python3 word-sync\(2\).py --offset -0.10
python3 test_align_engine.py          # synthetic-WAV alignment harness (81 checks)
python3 live_e2e_test.py              # end-to-end run vs a fake player + fake device
```

## Sync ladder (v3.0)

| Tier | Source | When |
|------|--------|------|
| 1 | real word timing (Enhanced LRC `<mm:ss>`, LRCLIB `lyricsfile`) | always preferred |
| 1.5 | acoustic alignment against a capture of this track | replay of a track already played |
| 2.5 | prosodic + rhythmic cadence model (dense-frontier pace, stress, caesurae) | every line without real timing |
| 3 | causal live phase-locking on the captured vocal | while audio sync is active |

## Cadence model (Tier 2.5)

1. **Dense-phrase frontier pace** — a singer cannot articulate *slower* than
   their own back-to-back gapless lines, so the song's True Articulation Pace is
   the 25–30th percentile of per-line `window / syllables` (weighted by the
   voiced fraction), not the median raw window. A 6-syllable phrase sung in
   ~1.8 s followed by 4–5 s of guitars no longer pegs pace to the 0.822 s/
   syllable ceiling.
2. **Per-line local pace** — a 5-line rolling window adapts to genuine tempo
   changes (half-time bridge, beat switch, rubato), while a *single* line whose
   window is stretched far beyond its content, with fast neighbours, is treated
   as a post-phrase instrumental gap / held final note and never inflates.
3. **Onset vs. sustain** — the span that spaces `words[0..n-1]` covers only
   their articulation plus caesurae; the final word's own duration and its held
   note happen *after* its onset and only decide when the screen clears.
4. **Prosody** — unstressed function words (EN/ES/FR/PT/DE/IT/RU, curated
   frozenset, no POS tagger) are rushed pickups (0.55x), punctuation opens a
   caesura (1.25x pre-boundary lengthening + a 0.14–0.32 s micro-pause scaled by
   window slack), repeated hooks lock isochronously (“la la la”,
   “yeah yeah yeah”, “d-d-death”), and 1–2 opening pickups before a heavy
   content word compress so the stressed word lands on the downbeat.

## Live audio bridge (v3.0)

`LiveAudioTracker` runs `pw-record` (PipeWire) or `parec` (PulseAudio) against
the player's sink monitor, or `@DEFAULT_MONITOR@`, in a daemon thread:

1. **Stereo Mid/Side vocal isolation** — `Mid = (L+R)/2`, `Side = (L-R)/2`,
   300 Hz–3.4 kHz formant band on both, 2.5–5.5 kHz transient band on Mid,
   `isolated = max(0, Mid − gain·Side)` weighted by centre dominance. Wide
   guitars/cymbals/synths cancel; a mono band-pass VAD hears them as continuous
   singing, this does not.
2. **Causal phase-locking** (`LivePhaseLock`) — vocal entry gate (hold a
   scheduled word up to +350 ms until the real onset rises), causal onset snap
   (±160 ms onto a strong isolated-vocal attack), post-phrase VAD freeze &
   early clear (>380 ms of centre silence releases the line for the interlude),
   and rolling re-alignment (a fast local refine over the active line's
   remaining tokens every 150 ms).
3. **Replay refinement** — the captured features are cached per track, so a
   replay (or a restart of the same track) runs the *offline* acoustic aligner:
   the `audio=` argument that live playback never used to pass.
4. **Fail-soft** — no capture binary, no permission, no sink, or
   `--no-audio-sync` => the display runs the prosodic engine alone with zero
   crashes and zero lag. Capture costs ~3 % of one core.

## Offline engine (v2.2 core, still)

`extract_features`, `extract_features_stereo`, `align_tokens`,
`refine_word_events`, `build_word_events`:

1. **Duration modelling** — vowels get nearly-free self-loops (melisma);
   plosives/fricatives hit a hard duration ceiling.
2. **Pause freeze** — if vocal-band energy is below threshold for >400 ms the
   lyric pointer cannot advance.
3. **Soft onsets** — 300 Hz–3.4 kHz formant *rise rate*, not full-band clicks.
4. **Adaptive corridor** — tight when confident, wide on dips / syncopation.
   No tempo-grid penalty on off-beat hits.
5. **Lookahead** — analysis window ≤150 ms; `calibration_offset_s` cancels
   STFT / hop delay.

```python
events = build_word_events(lines, duration, audio='vocal.wav',
                           align_cfg=AlignConfig(calibration_offset_s=0.0))
# stereo WAV / capture: (sr, left, right) -> Mid/Side isolation
events = build_word_events(lines, duration, audio=(sr, left, right))
```

## Tests

* `python3 "word-sync(2).py" --selftest` — fonts, parser, syllables, the v3.0
  Panchiko archetype, function-word/caesura, isochrony, mixed tempo, stereo
  isolation and streaming-vs-offline parity. Must print `selftest OK`.
* `python3 test_align_engine.py` — 81 checks over synthetic WAVs, including
  timing, freeze, soft onsets, syncopation, the v3.0 cadence model, stereo
  Mid/Side isolation and the live bridge (fake capture device, phase-lock).
* `python3 live_e2e_test.py` — drives the real display loop against a fake
  MPRIS player and a fake monitor device; verifies that displayed words land on
  the sung phrase with live capture and that `--no-audio-sync` still displays
  everything.
