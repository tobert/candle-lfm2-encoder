#!/usr/bin/env python3
"""Tests for clause_split.py — the shell clause splitter the advisory hook
uses to route compound commands through /v1/cascade.

Run:  python3 lfm2d/hooks/test_clause_split.py

Plain script, no pytest (none on this machine — see signoff). Exits non-zero
on failure.

WHY THESE CASES
---------------
The splitter exists because of a measured defect: `rm -rf -- "$d.venv"`
scores 0.540 alone and 0.047 inside its 624-char script (11x dilution), and
92% of mined real commands are compound. The named hazard is the inverse
failure: naive `&&` splitting cuts inside quoted payloads, which would break
the data-position cases the hook exists to measure — a curl whose JSON body
contains `rm -rf` must reach the classifier as ONE clause, not be cut into a
fake bare `rm -rf`.

DELIBERATE NON-GOALS, asserted here so a "fix" doesn't drift them:
- Pipelines are NOT split. `cat signoff.md | grep 'git push --force'`
  measured 0.994 informative as a whole; a split would hand v8 the bare
  clause `grep 'git push --force'`, a shape measured wrong (grep-for-pattern
  read 0.892 data-critical). Splitting pipes manufactures false positives
  out of currently-correct behaviour.
- No full bash grammar. Control-flow keywords are stripped/dropped after
  splitting; `case` patterns produce slightly odd fragments. The bar is
  "never cut inside data, always separate sequenced commands", not "parse
  bash".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clause_split import split_clauses  # noqa: E402

FAILURES = []


def check(label, cmd, expected):
    got = split_clauses(cmd)
    if got != expected:
        FAILURES.append(label)
        print(f"  FAIL  {label}")
        print(f"        input:    {cmd!r}")
        print(f"        expected: {expected!r}")
        print(f"        got:      {got!r}")
    else:
        print(f"  ok    {label}")


def check_in(label, cmd, must_contain):
    """Weaker assertion for gnarly inputs: the clause must emerge intact."""
    got = split_clauses(cmd)
    missing = [c for c in must_contain if c not in got]
    if missing:
        FAILURES.append(label)
        print(f"  FAIL  {label}")
        print(f"        input:   {cmd!r}")
        print(f"        missing: {missing!r}")
        print(f"        got:     {got!r}")
    else:
        print(f"  ok    {label}")


def main() -> int:
    # --- the basic compounds (92% of real traffic) ---
    check('simple &&', 'cd /tmp && rm -rf build', ['cd /tmp', 'rm -rf build'])
    check('semicolon chain', 'cd /tmp && rm -rf ./x ; echo done',
          ['cd /tmp', 'rm -rf ./x', 'echo done'])
    check('|| chain', 'grep -q x f || echo missing', ['grep -q x f', 'echo missing'])
    check('background &', 'sleep 5 & wait', ['sleep 5', 'wait'])
    check('multiline', 'set -e\nrm -rf ./tmp\necho done',
          ['set -e', 'rm -rf ./tmp', 'echo done'])
    check('single command unchanged', 'ls -la ~/src/lfm2d', ['ls -la ~/src/lfm2d'])
    check('empty', '', [])
    check('whitespace only', '  \n  ', [])

    # --- the hazard: quoted payloads must NOT be cut (data position) ---
    check('curl single-quoted JSON payload',
          'curl -s -X POST http://h/v1/classify -d \'{"inputs":"rm -rf /var/lib/data && echo done"}\'',
          ['curl -s -X POST http://h/v1/classify -d \'{"inputs":"rm -rf /var/lib/data && echo done"}\''])
    check('curl payload then real command',
          'curl -d \'{"cmd":"a && b; c"}\' http://h/x && echo sent',
          ['curl -d \'{"cmd":"a && b; c"}\' http://h/x', 'echo sent'])
    check('double-quoted payload with escapes',
          'curl -d "{\\"inputs\\":\\"rm -rf /x && echo\\"}" http://h/x',
          ['curl -d "{\\"inputs\\":\\"rm -rf /x && echo\\"}" http://h/x']),
    check('commit message prose',
          'git commit -m "docs: never run rm -rf; use tooling && guardrails"',
          ['git commit -m "docs: never run rm -rf; use tooling && guardrails"'])
    check('echo prose single-quoted',
          "echo 'never run rm -rf / on a production host'",
          ["echo 'never run rm -rf / on a production host'"])

    # --- substitution and grouping are opaque ---
    check('command substitution', 'echo $(ls; pwd)', ['echo $(ls; pwd)'])
    check('nested quotes in substitution', 'echo "$(basename "$f"; date)"',
          ['echo "$(basename "$f"; date)"'])
    check('backticks', 'echo `ls; pwd` && echo ok', ['echo `ls; pwd`', 'echo ok'])
    check('arithmetic shift is not a heredoc',
          'echo $((x << 2)) && echo done', ['echo $((x << 2))', 'echo done'])
    check('subshell interior IS split (sequenced commands)',
          '(cd /tmp && rm -rf x)', ['cd /tmp', 'rm -rf x'])
    # Consistent with the case above: grouping does not change sequencing,
    # so the group's interior splits here too.
    check('subshell then command',
          '(cd sub && make) && echo built', ['cd sub', 'make', 'echo built'])

    # --- pipelines stay whole (measured: splitting them creates FPs) ---
    check('pipeline not split', "cat signoff.md | grep 'git push --force'",
          ["cat signoff.md | grep 'git push --force'"])
    check('pipeline then &&', 'cat f | grep x && echo found',
          ['cat f | grep x', 'echo found'])
    check('|& pipe not split', 'make |& tee log', ['make |& tee log'])

    # --- redirection ampersands are not separators ---
    check('2>&1', 'make > /dev/null 2>&1 && echo ok',
          ['make > /dev/null 2>&1', 'echo ok'])
    check('&> redirect', 'cmd &> log && next', ['cmd &> log', 'next'])

    # --- heredocs: the body is data, never split, never a delimiter leak ---
    check('heredoc body not split',
          'cat <<EOF\nrm -rf /\nnever run this && that\nEOF',
          ['cat <<EOF\nrm -rf /\nnever run this && that\nEOF'])
    check('heredoc then command',
          'cat <<EOF\nbody line\nEOF\necho after',
          ['cat <<EOF\nbody line\nEOF', 'echo after'])
    check('quoted heredoc delimiter',
          "cat <<'EOF'\n$(danger); rm -rf x\nEOF",
          ["cat <<'EOF'\n$(danger); rm -rf x\nEOF"])
    check('heredoc with && on opening line stays one clause',
          'cat <<EOF && echo done\nbody\nEOF',
          ['cat <<EOF && echo done\nbody\nEOF'])
    check('<<- tab-stripped delimiter',
          'cat <<-EOF\n\tbody\n\tEOF\necho after',
          ['cat <<-EOF\n\tbody\n\tEOF', 'echo after'])
    check('herestring is not a heredoc',
          'grep x <<< "a && b" && echo ok',
          ['grep x <<< "a && b"', 'echo ok'])

    # --- comments are dropped, and quotes inside them are inert ---
    check('trailing comment dropped', "echo hi # don't && rm -rf x",
          ['echo hi'])
    check('full-line comment', '# setup && cleanup\necho hi', ['echo hi'])
    check('hash inside word is not a comment', 'echo foo#bar', ['echo foo#bar'])

    # --- line continuation ---
    check('escaped newline does not split', 'echo a \\\n b && c',
          ['echo a \\\n b', 'c'])

    # --- control flow: keywords stripped, closers dropped ---
    check('if/then/fi', 'if grep -q x f; then echo y; fi',
          ['grep -q x f', 'echo y'])
    check('for/do/done', 'for f in *.txt; do echo "$f"; done',
          ['for f in *.txt', 'echo "$f"'])
    check('while loop', 'while true; do date; done', ['true', 'date'])
    check('negation stripped', '! grep -q x f && echo absent',
          ['grep -q x f', 'echo absent'])

    # --- case: odd fragments allowed, but later commands must still split ---
    check_in('case does not corrupt later splitting',
             'case $x in\n  a) echo a;;\n  b) echo b;;\nesac\necho after',
             ['echo after'])

    # --- the motivating fixture, in miniature: severe clause inside a loop
    #     script must emerge alone (measured 0.540 alone vs 0.047 diluted) ---
    check_in('dilution fixture: rm -rf clause emerges',
             'for d in worktree-a worktree-b; do\n'
             '  echo "cleaning $d"\n'
             '  rm -rf -- "$d.venv"\n'
             'done',
             ['rm -rf -- "$d.venv"'])

    # --- unterminated constructs must not crash and must not split ---
    check('unterminated quote', "echo 'oops && never", ["echo 'oops && never"])
    check('unterminated substitution', 'echo $(ls && pwd', ['echo $(ls && pwd'])
    check('lone close paren is literal', 'echo ) && echo ok',
          ['echo )', 'echo ok'])

    print()
    if FAILURES:
        print(f'FAILED: {len(FAILURES)} case(s): {FAILURES}')
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
