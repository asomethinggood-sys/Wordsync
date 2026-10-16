# Wordsync

Lyrics sync at word level. Designed and published for Linux. BETA.

Zero external Python dependencies (stdlib + `playerctl`). Single file:
`word-sync(2).py` — display, LRC parser, trained cadence model, the offline
audio-to-lyric alignment engine, and the v3.1 live audio sync bridge.

## Run

```bash
python3 word-sync\(2\).py              # live display (needs playerctl)
python3 word-sync\(2\).py --selftest   # offline regressions
python3 word-sync\(2\).py --plain
python3 word-sync\(2\).py --offset -0.10
python3 word-sync\(2\).py --no-audio-sync      # prosodic heuristic only
python3 word-sync\(2\).py --no-auto-offset     # disable LRC auto-anchor
python3 word-sync\(2\).py --no-cache-events    # disable built-events cache
python3 test_align_engine.py                 # synthetic-WAV alignment harness
```

## v3.1 — Sub-100ms End-to-End Lyric Sync

v3.1 is a *closed-loop* lyric display. The LRC timeline is a *prior*;
the live audio and the live player position are the *measurements*;
the display state is the *estimate*; the rendered word is the
*actuation*.

### What changed from v3.0

1. **Display State Machine** — `DisplayStateMachine` with explicit
   `LOADING / PRELUDE / SINGING / INTERLUDE / ENDING` states and
   deterministic transitions. Replaces v3.0's brittle bisect + ad-hoc
   `if word / elif '' / elif not events` chain. The dead-zone rule
   (`DEAD_ZONE_S = 0.080`) means a word can never appear on screen
   before the timeline has actually reached it OR live audio has
   confirmed a vocal onset.

2. **Live Phase-Locker** — `PhaseLocker.consult(t_predicted, pos)`
   asks the live audio engine: "is there a strong vocal onset within
   ±160 ms of where I want to show the next word?" If yes, snap to
   it. If no, hold. After 350 ms of patience, release. This is the
   "no latency" answer — the LRC schedule is corrected in real time
   by the actual singer's vocal attack.

3. **Post-Vocal VAD Freeze** — when the centre-channel VAD has been
   silent for >380 ms, the display state drops to INTERLUDE
   immediately, regardless of the LRC's `clear_at`. Kills the
   "instrument plays for 4 s after the last word and the display
   still shows the last word" bug.

4. **First-Sample Auto-Anchor** — on every song change, the first
   observed player position becomes the truth. The timeline's
   `first_word_t` is shifted to match. This corrects the LRCLIB
   timestamp error (often 1–3 s) AND the lyrics-fetch delay.

5. **LRC Candidate Auto-Picker** — `pick_candidate(cands, first_obs)`
   scores each candidate by closeness to first observed position
   (weight 2×), line count (0.1 per line), word-level bonus (+50),
   and a soft penalty for first-word articles. Picks the best
   without user intervention. The `n` hotkey still cycles manually.

6. **Built-Events Cache** — the resolved `(t, word)` timeline is
   cached to `~/.cache/word-sync/events/<hash>.json` (30-day TTL).
   Repeat plays skip the entire HTTP fetch + parse + build pipeline.
   Caches include the learned intrinsic offset per song.

7. **Per-Song Self-Tuning** — `SongResiduals` logs
   `(t_predicted, t_displayed, audio_vocal_t)` for every word. On
   song change, the median audio-vs-prediction residual is blended
   (weight 0.5) into the manual offset for the next play. After a
   few plays the system self-calibrates to within ±100 ms.

### CLI flags (v3.1)

| Flag                      | Default | Effect |
|---------------------------|---------|--------|
| `--no-audio-sync`         | off     | Disable the live audio sync bridge (heuristic only) |
| `--auto-offset`           | on      | Anchor timeline to first observed player position |
| `--no-auto-offset`        |         | Disable the auto-anchor |
| `--cache-events`          | on      | Cache built events for instant reload |
| `--no-cache-events`       |         | Disable the cache |

## Offline engine (v3.0 — unchanged)

Lives in `word-sync(2).py` (`extract_features`, `align_tokens`,
`build_word_events`).

1. **Duration modelling** — vowels get nearly-free self-loops (melisma);
   plosives/fricatives hit a hard duration ceiling.
2. **Pause freeze** — if vocal-band energy is below threshold for >400 ms
   the lyric pointer cannot advance.
3. **Soft onsets** — 300 Hz–3.4 kHz formant *rise rate*, not full-band clicks.
4. **Adaptive corridor** — tight when confident, wide on dips / syncopation.
5. **Lookahead** — analysis window ≤150 ms; `calibration_offset_s` cancels
   STFT / hop delay.

## Test coverage

`test_align_engine.py` runs 85 offline checks across:

- v2.2 cadence engine: unit costs, latency caps, pause freeze, soft
  onsets, syncopation corridor, standard-track MAE, heuristic path
  (sections 1–7).
- v3.0 prosodic / cadence: Panchiko archetype, function-word &
  caesura, mixed-tempo / beat-switch, stereo mid-side isolation,
  LiveAudioTracker (sections 8–12).
- v3.1 state machine + live phase-locking: phantom-first-word,
  dead-zone, post-vocal VAD freeze, audio onset snap, LRC auto-pick,
  events cache roundtrip, residual self-tuning, end-to-end
  phase-locked display (sections 13–20).

The in-app `--selftest` exercises the font / script / parser /
syllable / timing paths and the engine unit checks.
