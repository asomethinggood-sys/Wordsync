#!/usr/bin/env python3
"""Compatibility module/launcher for the Wordsync v3.0 implementation."""
import wordsync_v3 as _implementation

# Re-export private helpers too, preserving existing test/trainer imports.
for _name in dir(_implementation):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_implementation, _name)

if __name__ == '__main__':
    _implementation.main()
