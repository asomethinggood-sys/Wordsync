# Wordsync

Word-level lyric synchronization for Linux MPRIS players. The v3.0 engine retains the terminal themes, fonts, color modes and zero-Python-dependency footprint while adding dense-phrase cadence estimation, local tempo adaptation, function-word prosody, punctuation caesurae and a fail-soft live audio path.

## Run

```sh
python3 wordsync_v3.py
python3 wordsync_v3.py --selftest
python3 -m unittest -v test_align_engine.py
python3 eval_harness.py
```

Live center-vocal tracking is automatic when `pw-cat`, `pw-record` or `parec` is available and permitted. It reads the default output monitor asynchronously; if capture is unavailable or denied, timing falls back to the prosodic estimator. Disable capture explicitly with `--no-audio-sync`.

`word_sync.py` and `word-sync(2).py` remain compatibility module/launchers for the earlier filenames.

## v3.0 timing changes

- Uses a nearest-rank 27th-percentile dense-phrase frontier for the song articulation pace, then blends a robust five-line local cadence so isolated instrumental gaps do not slow every word and sustained tempo changes can adapt quickly.
- Times word onsets from prosodic articulation, compresses common multilingual function words, lengthens pre-caesura words, adds slack-bounded micro-pauses, and keeps repeated chants isochronous.
- Separates the last word's onset from its articulation and final-note hold; long post-phrase gaps produce an early clear/rest event.
- Adds a pure-stdlib stereo mid/side formant and consonant feature extractor plus a 12-second PipeWire/PulseAudio monitor ring, causal onset snap and vocal-silence clear. It never blocks the playback/render loop and gracefully falls back to heuristic timing.
- Sparse and ambiguous 4.5–6+ second LRC intervals use conservative pace priors instead of treating the whole gap as sung duration. Corrected consonant-plus-`le` syllable counts (for example, “middle”), and fitted phrase-sized vocal activity can lengthen genuinely slow lines without stretching across long instrument/reverb tails.
- Live event identities remain stable through re-alignment; already displayed events are locked so timeline resorting cannot repeatedly re-trigger the same word.
- `test_align_engine.py` and `eval_harness.py` use offline regression cases and generated audio only. No Panchiko recording/lyrics fixture was available, so their metrics do not validate or promise accuracy on “Deathmetal.” Centered instruments can still resemble vocals to a lightweight VAD; audio correction is optional and remains fail-soft.
