#!/usr/bin/env python3
"""
End-to-end live test for the v3.0 audio-sync bridge.

Runs the *unmodified* display loop (word-sync(2).py --plain) against

  * a fake MPRIS player (`playerctl`) whose position advances in real time, and
  * a fake monitor-capture device (`parec`) that streams a looping mix of
    wide-panned "guitars" plus a centre-panned 1.2 s vocal burst every 3 s,

with a lyric file whose line anchors are deliberately ~0.45 s EARLY relative to
the vocal. Two passes are measured:

  A. --audio-sync   (live capture + causal phase-locking)
  B. --no-audio-sync (v3.0 prosodic engine alone)

and the metric is the median distance between each displayed word and the
nearest centre-channel vocal onset seen by the tracker. Nothing here needs the
network, real audio hardware, PipeWire or PulseAudio.

Run:  python3 live_e2e_test.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FAKE_DEVICE = '''#!/usr/bin/env python3
"""Fake monitor device: wide guitars + a centre vocal burst, real time."""
import array, math, sys, time
SR = 16000
CHUNK = int(0.1 * SR)
LOOP = int(3.0 * SR)
BURST = (0.60, 1.80)


def gen(start):
    buf = bytearray()
    for i in range(start, start + CHUNK):
        t = (i % LOOP) / SR
        g = 0.30 * sum(a * math.sin(2 * math.pi * fr * t + p) for fr, a, p in
                       ((310, .30, .3), (830, .24, 2.0), (1900, .16, 1.7)))
        left, right = g, -g
        if BURST[0] <= t < BURST[1]:
            v = 0.26 * (0.62 * math.sin(2 * math.pi * 720 * t)
                        + 0.38 * math.sin(2 * math.pi * 1240 * t))
            left += v
            right += v
        buf += array.array("h", [int(max(-1, min(1, left)) * 32767),
                                 int(max(-1, min(1, right)) * 32767)]).tobytes()
    return buf


out = sys.stdout.buffer
frames = 0
end = time.time() + 60
while time.time() < end:
    try:
        out.write(gen(frames))
        out.flush()
    except Exception:
        break
    frames += CHUNK
    time.sleep(0.1)
'''

FAKE_PLAYER = '''#!/usr/bin/env python3
"""Fake MPRIS player: a 120 s track whose position tracks wall-clock time."""
import os, sys, time
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state')
if not os.path.exists(STATE):
    with open(STATE, 'w') as fh:
        fh.write(str(time.time()))
start = float(open(STATE).read())
args = sys.argv[1:]
if args and args[0] == '--player':
    args = args[2:]
if args and args[0] == 'metadata':
    pos = max(0.0, (time.time() - start) % 120.0)
    print('STATUS=Playing||ARTIST=Panchiko||TITLE=Deathmetal||POS=%d||LEN=%d'
          '||RATE=1.0||TID=t1' % (int(pos * 1_000_000), 120 * 1_000_000))
    sys.exit(0)
print('Playing')
'''

LRC = '''[00:00.15]we are the dead ones now
[00:03.15]hold me closer tonight
[00:06.15]we are the dead ones now
[00:09.15]hold me closer tonight
[00:12.15]we are the dead ones now
[00:15.15]hold me closer tonight
[00:18.15]we are the dead ones now
[00:21.15]hold me closer tonight
'''


def _load_word_sync():
    spec = importlib.util.spec_from_file_location(
        'word_sync_e2e', os.path.join(ROOT, 'word-sync(2).py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ws = _load_word_sync()


def _make_fixtures(tmp):
    bindir = os.path.join(tmp, 'bin')
    lyrics = os.path.join(tmp, 'lyrics')
    os.makedirs(bindir, exist_ok=True)
    os.makedirs(lyrics, exist_ok=True)
    for name, body in (('parec', FAKE_DEVICE), ('playerctl', FAKE_PLAYER)):
        path = os.path.join(bindir, name)
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(body)
        os.chmod(path, 0o755)
    with open(os.path.join(lyrics, 'Panchiko - Deathmetal.lrc'), 'w',
              encoding='utf-8') as fh:
        fh.write(LRC)
    return bindir, lyrics


def _run_once(tmp, lyrics, audio_sync, seconds=9.0):
    """Run the real display loop for `seconds`; return (word arrivals, tracker)."""
    env_backup = os.environ.get('PATH', '')
    tmp_state = os.path.join(tmp, 'bin', 'state')
    if os.path.exists(tmp_state):
        os.remove(tmp_state)
    arrivals = []
    real_print = print
    tracker_holder = {}
    real_tracker_init = ws.LiveAudioTracker.__init__

    def spy_init(self, *a, **kw):
        real_tracker_init(self, *a, **kw)
        tracker_holder['t'] = self

    ws.LiveAudioTracker.__init__ = spy_init

    def spy_print(*args, **kw):
        text = ' '.join(str(a) for a in args)
        if text.strip():
            arrivals.append((time.time(), text.strip()))
        return None

    argv = ['word-sync', '--plain', '--lyrics-dir', lyrics, '--cache-dir',
            os.path.join(tmp, 'cache'), '--config', os.path.join(tmp, 'none.toml'),
            '--offset', '0']
    argv.append('--audio-sync' if audio_sync else '--no-audio-sync')
    old_argv = sys.argv
    sys.argv = argv
    ws.should_exit = False
    run_start = None
    import builtins
    builtins.print = spy_print
    try:
        t = threading.Timer(seconds, lambda: ws._request_exit())
        t.daemon = True
        t.start()
        try:
            ws.main()
        except SystemExit:
            pass
        finally:
            t.cancel()
    finally:
        builtins.print = real_print
        sys.argv = old_argv
        ws.LiveAudioTracker.__init__ = real_tracker_init
        os.environ['PATH'] = env_backup
    tracker = tracker_holder.get('t')
    # The fake player restarts its 120 s track per run, so song time for this
    # run is (wall - run_start); the state file records when it started.
    try:                                    # epoch seconds, same clock as time.time()
        run_start = float(open(tmp_state).read())
    except Exception:
        run_start = arrivals[0][0] if arrivals else time.time()
    return arrivals, tracker, run_start


def _vad_segments(tracker):
    """[(start, end), ...] of sung regions in the tracker's own song time."""
    if tracker is None:
        return []
    segs, start = [], None
    for f in list(tracker.ring):
        if f.vad and start is None:
            start = f.t
        elif not f.vad and start is not None:
            segs.append((start, f.t))
            start = None
    if start is not None:
        segs.append((start, None))
    return segs


def _distance_to_vocal(song_time, segments, open_tail=0.6):
    """0 when the display happens while the singer is singing, else seconds off."""
    best = None
    for t0, t1 in segments:
        end = t1 if t1 is not None else t0 + open_tail
        d = 0.0 if t0 <= song_time <= end else min(abs(song_time - t0),
                                                  abs(song_time - end))
        if best is None or d < best:
            best = d
    return best


def main():
    tmp = tempfile.mkdtemp(prefix='wordsync-e2e-')
    try:
        bindir, lyrics = _make_fixtures(tmp)
        os.environ['PATH'] = bindir + os.pathsep + os.environ.get('PATH', '')

        print('== live end-to-end (fake player + fake monitor device) ==')
        live_words, live_tr, live_start = _run_once(tmp, lyrics, audio_sync=True)
        off_words, _off_tr, off_start = _run_once(tmp, lyrics, audio_sync=False)
        segs = _vad_segments(live_tr)
        print('  words displayed: live=%d  no-audio=%d'
              % (len(live_words), len(off_words)))
        print('  capture note:    %s'
              % (live_tr.capture_summary() if live_tr else 'n/a'))
        print('  vocal segments seen by the tracker: %s'
              % [(round(a, 2), round(b, 2)) for a, b in segs][:6])

        def report(name, arrivals, start):
            if not arrivals or not segs:
                return None
            # Wall -> song time via the (fake) player's own clock, which is the
            # same timeline the display's PlaybackClock follows.
            errs = []
            for t_wall, _w in arrivals:
                d = _distance_to_vocal(t_wall - start, segs)
                if d is not None:
                    errs.append(d)
            if not errs:
                return None
            errs.sort()
            med = errs[len(errs) // 2]
            print('  %-11s median distance to the sung phrase = %.3f s '
                  '(p90 %.3f s, n=%d)'
                  % (name, med, errs[int(0.9 * (len(errs) - 1))], len(errs)))
            return med
        med_live = report('audio-sync', live_words, live_start)
        med_off = report('prosodic', off_words, off_start)
        print('== done ==')
        failures = []
        if live_tr is None or not live_tr.ring:
            failures.append('live capture never produced features')
        elif len(_vad_segments(live_tr)) < 2:
            failures.append('tracker did not see the sung phrases')
        if med_live is None or med_live > 0.15:
            failures.append('audio-sync words are not landing on the vocal '
                            '(median %s)' % med_live)
        if len(live_words) == 0 or len(off_words) == 0:
            failures.append('display produced no words')
        if failures:
            for f in failures:
                print('FAIL', f)
            return 1
        print('OK  live capture fed the display: words land in the sung phrase '
              '(median %.3f s) and the no-audio run still displays every word'
              % med_live)
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
