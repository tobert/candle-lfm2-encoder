#!/usr/bin/env python3
"""Corpus validation for clause_split.py — the gate-on-artifacts step.

Runs the splitter over every mined real command (default: the 2026-08-12
mining artifact, 10,490 unique commands out of Amy's own session
transcripts) and reports:

  - exceptions from the RAW splitter (`_split`, bypassing the never-raise
    wrapper, so bugs surface instead of degrading silently) — must be 0
  - clause-count distribution vs the corpus's known 92% compound rate
  - reassembly token check: severe tokens present in the input must appear
    in some output clause (a cheap way to catch clauses being DROPPED,
    which for a severity guard is the unforgivable direction)
  - the recorded dilution fixture: the command containing
    `rm -rf -- "$d.venv"` (measured 0.540 alone / 0.047 diluted) must
    yield that rm as its own clause

Run:  python3 lfm2d/hooks/validate_split_corpus.py [commands.jsonl]

Committed per commit-the-scorer: any number this prints ships with the
code that produced it.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clause_split import _split, split_clauses  # noqa: E402

DEFAULT_CORPUS = Path.home() / '.local/share/lfm2-training-data/command-mining-2026-08-12/commands.jsonl'

# Tokens whose disappearance would mean the splitter EATS severe content.
SEVERE_TOKENS = ['rm -rf', 'rm -fr', '--force', 'reset --hard', 'kubectl delete', 'DROP TABLE']


def main() -> int:
    corpus = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CORPUS
    if not corpus.exists():
        print(f'FAIL: corpus not found at {corpus}')
        return 2

    commands = []
    for line in corpus.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cmd = row['command'] if isinstance(row, dict) else row
        commands.append(cmd)
    # The mining artifact may carry duplicates; validate unique text.
    unique = list(dict.fromkeys(commands))
    print(f'corpus: {corpus}')
    print(f'rows: {len(commands)}, unique: {len(unique)}\n')

    crashes = []
    dropped_severe = []
    counts = Counter()
    for cmd in unique:
        try:
            clauses = _split(cmd, recurse=2)
        except Exception as e:
            crashes.append((cmd[:120], repr(e)))
            continue
        counts[min(len(clauses), 10)] += 1
        joined = '\n'.join(clauses)
        for tok in SEVERE_TOKENS:
            if tok in cmd and tok not in joined:
                # Token may legitimately vanish when it sat in a comment or
                # is cut by MAX_CLAUSES truncation — record for eyeballing.
                dropped_severe.append((tok, cmd[:160]))

    print('clause-count distribution (10 = 10+):')
    total = sum(counts.values())
    for k in sorted(counts):
        print(f'  {k:>3}: {counts[k]:>6}  {100 * counts[k] / total:5.1f}%')
    multi = sum(v for k, v in counts.items() if k > 1)
    print(f'\ncompound after split (2+ clauses): {multi} ({100 * multi / total:.1f}%)')

    print(f'\nraw splitter exceptions: {len(crashes)}')
    for cmd, err in crashes[:10]:
        print(f'  CRASH {err}  on: {cmd}')

    print(f'severe-token drops: {len(dropped_severe)}')
    for tok, cmd in dropped_severe[:20]:
        print(f'  DROP {tok!r}  from: {cmd}')

    # The recorded dilution fixture, against the REAL script it came from.
    fixture_cmds = [c for c in unique if 'rm -rf -- "$d' in c and len(c) > 300]
    fixture_ok = False
    for c in fixture_cmds:
        clauses = split_clauses(c)
        hits = [cl for cl in clauses if cl.startswith('rm -rf -- "$d')]
        if hits:
            fixture_ok = True
            print(f'\ndilution fixture: OK — {hits[0]!r} emerges from its '
                  f'{len(c)}-char script as 1 of {len(clauses)} clauses')
            break
    if not fixture_cmds:
        print('\ndilution fixture: NOT FOUND in corpus (checked pattern rm -rf -- "$d)')
    elif not fixture_ok:
        print(f'\ndilution fixture: FAIL — found {len(fixture_cmds)} candidate '
              f'command(s) but the rm clause did not emerge')

    ok = not crashes and (fixture_ok or not fixture_cmds)
    print(f'\n{"VALIDATION OK" if ok else "VALIDATION FAILED"}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
