#!/usr/bin/env python3
"""Parity gate: the advisory hook's REGEX half must decide exactly what the
current dotfiles hook decides, on every input, byte-for-byte.

This is the only thing that makes the advisory log trustworthy. If the two
differ, a disagreement row could be a rules edit rather than a model
opinion, and the whole measurement phase is measuring the wrong thing.

Run:  python3 lfm2d/hooks/test_parity.py [path-to-dotfiles-hook]

No pytest on this machine (see signoff), so this is a plain script that
exits non-zero on failure and is honest about what it did NOT cover.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADVISORY = HERE / 'pre_command_advisory.py'
DEFAULT_BASELINE = Path.home() / 'src/tobert/dotfiles/.claude/hooks/pre-command.py'

# Each case is (label, command). Chosen to hit every branch of both hooks,
# plus the data-position shapes this whole project exists to get right.
CASES = [
    # --- BLOCKED (git hygiene) ---
    ('blocked: git add -A',        'git add -A'),
    ('blocked: git add --all',     'git add --all'),
    ('blocked: git add .',         'git add .'),
    ('blocked: git commit -a',     'git commit -am "wip"'),
    ('blocked: git stash',         'git stash push -m x'),
    # --- SOFT_BLOCKED ---
    ('soft: git amend',            'git commit --amend --no-edit'),
    ('soft: git push --force',     'git push --force origin main'),
    ('soft: git reset --hard',     'git reset --hard origin/main'),
    ('soft: rm -rf',               'rm -rf /var/lib/data'),
    ('soft: rm -fr',               'rm -fr ./build'),
    ('soft: find -exec',           'find . -name "*.tmp" -exec rm {} ;'),
    ('soft: find -execdir',        'find . -execdir echo {} ;'),
    ('soft: find -ok',             'find . -ok rm {} ;'),
    # --- WARNED ---
    ('warn: rm .local/share',      'rm ~/.local/share/thing'),
    ('warn: rm .config',           'rm ~/.config/thing'),
    ('warn: rm *.db',              'rm foo.db'),
    ('warn: rm -r',                'rm -r ./build'),
    ('warn: rm -f',                'rm -f ./file'),
    # --- ALLOW ---
    ('allow: ls',                  'ls -la ~/src/lfm2d'),
    ('allow: cd',                  'cd db/migrations'),
    ('allow: npm install',         'npm install'),
    ('allow: kubectl delete ns',   'kubectl delete namespace prod'),
    # --- data position: the false-positive shapes (both hooks should agree,
    #     because the advisory hook does NOT change the regex behaviour --
    #     these document what the regex does, which is flag them) ---
    ('data: curl carrying rm -rf', 'curl -s -X POST http://h/v1/classify -d \'{"inputs":"rm -rf /var/lib/data"}\''),
    ('data: commit msg prose',     'git commit -m "docs: note the hook blocked rm -rf in prose"'),
    ('data: echo prose',           "echo 'never run rm -rf / on a production host'"),
    ('data: grep for pattern',     "grep -rn 'rm -rf' ./hooks/pre-command.py"),
    # --- edge cases ---
    ('edge: empty command',        ''),
    ('edge: multiline',            'set -e\nrm -rf ./tmp\necho done'),
    ('edge: unicode path',         'rm -rf ./データ'),
    ('edge: semicolon chain',      'cd /tmp && rm -rf ./x ; echo done'),
]


def run_hook(hook: Path, payload: dict, cache_dir: Path) -> tuple:
    """Run one hook with an isolated cache. Returns (returncode, parsed-stdout)."""
    env = dict(os.environ)
    env['XDG_CACHE_HOME'] = str(cache_dir)
    env['LFM2D_HOOK_MODE'] = 'off'  # no network in the parity gate
    p = subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    try:
        return p.returncode, json.loads(p.stdout)
    except json.JSONDecodeError:
        return p.returncode, {'__unparseable__': p.stdout, '__stderr__': p.stderr}


def main() -> int:
    baseline = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BASELINE
    if not baseline.exists():
        print(f"FAIL: baseline hook not found at {baseline}")
        return 2
    if not ADVISORY.exists():
        print(f"FAIL: advisory hook not found at {ADVISORY}")
        return 2

    print(f"baseline: {baseline}")
    print(f"advisory: {ADVISORY}\n")

    failures = []
    for label, cmd in CASES:
        payload = {'tool_name': 'Bash', 'tool_input': {'command': cmd}}
        # Fresh, separate cache dirs: the soft-block cache is stateful, and
        # sharing it would let one hook consume the other's approval entry
        # and manufacture a difference that isn't a rules difference.
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            rc_base, out_base = run_hook(baseline, payload, Path(a))
            rc_adv, out_adv = run_hook(ADVISORY, payload, Path(b))

        if rc_base != rc_adv or out_base != out_adv:
            failures.append((label, cmd, out_base, out_adv))
            print(f"  DIFF  {label}")
            print(f"        baseline: {out_base}")
            print(f"        advisory: {out_adv}")
        else:
            decision = out_base.get('hookSpecificOutput', {}).get('permissionDecision', '?')
            print(f"  ok    {label:<28} -> {decision}")

    # Non-Bash tools must pass straight through on both.
    for tool in ('Read', 'Write', 'Edit'):
        payload = {'tool_name': tool, 'tool_input': {'file_path': '/tmp/x'}}
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            rc_base, out_base = run_hook(baseline, payload, Path(a))
            rc_adv, out_adv = run_hook(ADVISORY, payload, Path(b))
        if (rc_base, out_base) != (rc_adv, out_adv):
            failures.append((f'non-bash: {tool}', '', out_base, out_adv))
            print(f"  DIFF  non-bash {tool}")
        else:
            print(f"  ok    non-bash {tool:<21} -> passthrough")

    print()
    if failures:
        print(f"PARITY FAILED: {len(failures)} of {len(CASES) + 3} cases differ")
        return 1

    print(f"PARITY OK: {len(CASES) + 3} cases identical")
    print(
        "\nNOT covered by this gate, on purpose:\n"
        "  - the soft-block approval RETRY path (stateful across two invocations;\n"
        "    both hooks share the implementation verbatim, but that is an argument\n"
        "    from inspection, not a test — worth adding)\n"
        "  - anything about lfm2d itself: LFM2D_HOOK_MODE=off here, deliberately,\n"
        "    so this gate stays offline and deterministic"
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
