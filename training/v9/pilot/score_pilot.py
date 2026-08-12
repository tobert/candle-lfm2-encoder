#!/usr/bin/env python3
"""Score the v9 rubric pilot — pilot-gate-the-rubric, mechanized.

Reads pilot.jsonl (rows + gold + gold_source) and every raw/<family>.jsonl,
then reports per row: the family votes, agreement shape, and the verdict
class this project's law assigns:

  unanimous-right              — rubric and labelers agree; nothing to do
  unanimous-wrong              — ALL families disagree with gold the same
                                 way: a RUBRIC BUG, not a labeler failure
                                 (three blind families once made an
                                 identical 10-row unanimous error)
  split                        — 2-1 or 3-way: the design working; read the
                                 minority's reasoning before deciding
  unanimous-vs-provisional     — unanimous against an AUTHORED gold: the
                                 author is probably the one who is wrong
                                 (measure-disagreement-dont-declare-it)

Run:  python3 training/v9/pilot/score_pilot.py [raw-dir]

`raw-dir` defaults to `raw/` (round 1). Re-runs after a rubric edit go in
sibling dirs (`raw-r2/`, ...) so every round's verbatim outputs survive.

Committed with the numbers it produces, per commit-the-scorer.
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALID = {'informative', 'situation-normal', 'data-critical'}


def main() -> int:
    rows = {}
    for line in (HERE / 'pilot.jsonl').read_text().splitlines():
        r = json.loads(line)
        rows[r['id']] = r

    raw_dir = HERE / (sys.argv[1] if len(sys.argv) > 1 else 'raw')
    families = {}
    for f in sorted(raw_dir.glob('*.jsonl')):
        votes = {}
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            v = json.loads(line)
            if v['label'] not in VALID:
                print(f'INVALID label {v["label"]!r} from {f.stem} on {v["id"]}')
                return 2
            votes[v['id']] = v['label']
        missing = set(rows) - set(votes)
        if missing:
            print(f'{f.stem}: MISSING ids {sorted(missing)} — refusing to score a partial family')
            return 2
        families[f.stem] = votes

    if len(families) < 3:
        print(f'only {len(families)} families in raw/ — the gate needs 3 blind families')
        return 2

    names = sorted(families)
    print(f'families: {", ".join(names)}\n')
    print(f'{"id":<5}{"gold":<18}{"src":<6}' + ''.join(f'{n[:14]:<16}' for n in names) + 'verdict')

    tally = Counter()
    findings = []
    for rid, row in sorted(rows.items()):
        gold, src = row['gold'], row['gold_source']
        votes = [families[n][rid] for n in names]
        vc = Counter(votes)
        unanimous = len(vc) == 1
        agree_gold = [v == gold for v in votes]

        if unanimous and all(agree_gold):
            verdict = 'unanimous-right'
        elif unanimous:
            verdict = ('unanimous-vs-provisional'
                       if src == 'authored-provisional' else 'UNANIMOUS-WRONG')
        else:
            verdict = 'split'
        tally[verdict] += 1
        if verdict != 'unanimous-right':
            findings.append((rid, row, votes, verdict))

        marks = ''.join(
            f'{v + ("" if v == gold else " *"):<16}' for v in votes)
        print(f'{rid:<5}{gold:<18}{src[:5]:<6}{marks}{verdict}')

    print('\ntally: ' + ', '.join(f'{k}={v}' for k, v in sorted(tally.items())))
    print('\n— rows needing a read —')
    for rid, row, votes, verdict in findings:
        print(f'\n{rid} [{verdict}] {row["text"]!r}')
        print(f'   gold={row["gold"]} ({row["gold_source"]}) votes={votes}')
        print(f'   note: {row["note"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
