# Wordsync

Word-level lyric synchronization for Linux MPRIS players. The v3.0 engine retains the terminal themes, fonts, color modes and zero-Python-dependency footprint while adding denser-phrase cadence estimation, local tempo adaptation, function-word prosody, punctuation caesurae and a fail-soft live audio path.

## Run

```sh
python3 word_sync.py
python3 word_sync.py --selftest
python3 -m unittest -v test_align_engine.py
```

Live center-vocal tracking is automatic when `pw-cat`, `pw-record` or `parec` is available and permitted. It reads the default output monitor asynchronously; if capture is unavailable or denied, timing falls back to the prosodic estimator. Disable capture explicitly with `--no-audio-sync`.

`word-sync(2).py` remains as a compatibility launcher for the original uploaded filename.

## v3.0 timing changes

- Uses a nearest-rank 27th-percentile dense-phrase frontier for the song articulation pace, then blends a robust five-line local cadence so isolated instrumental gaps do not slow every word and sustained tempo changes can adapt quickly.
- Times word onsets from prosodic articulation, compresses common multilingual function words, lengthens pre-caesura words, adds slack-bounded micro-pauses, and keeps repeated chants isochronous.
- Separates the last word's onset from its articulation and final-note hold; long post-phrase gaps produce an early clear/rest event.
- Adds a pure-stdlib stereo mid/side formant and consonant feature extractor plus a 12-second PipeWire/PulseAudio monitor ring, causal onset snap and vocal-silence clear. It never blocks the playback/render loop and gracefully falls back to heuristic timing.
