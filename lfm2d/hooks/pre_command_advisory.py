#!/usr/bin/env python3
"""PreToolUse Bash hook — the existing regex guard, plus an ADVISORY lfm2d second opinion.

WHAT THIS DOES AND DOES NOT DO
------------------------------
In advisory mode (the default, and the only mode until we have data), the
regex logic below decides EVERY outcome, byte-for-byte identically to
`dotfiles/.claude/hooks/pre-command.py`. lfm2d's verdict is recorded and
never enforced. The point of this phase is to find out, on real traffic,
where the model and the regex disagree — and which of them is right when
they do.

Set LFM2D_HOOK_MODE=enforce only after that data says something. It is
wired here so the switch is one env var rather than a rewrite, NOT because
it is ready. See "MEASURED SHORTCOMINGS" below before you flip it.

MEASURED SHORTCOMINGS (2026-08-12, kube_ordinal_v8, live pod)
-------------------------------------------------------------
The motivating pitch was "a classifier can tell a dangerous command from a
dangerous-looking STRING; a regex cannot." Measured, that is only PARTLY
true on v8 today. Probe of 14 commands against /v1/classify:

  correct, data position read as benign:
    echo 'never run rm -rf / ...'                    informative      0.953
    cat signoff.md | grep 'git push --force'         informative      0.994
    git commit -m 'docs: ...rm -rf in prose'         situation-normal 0.681

  STILL WRONG, data position read as data-critical:
    curl ... -d '{"inputs":"rm -rf /var/lib/data"}'  data-critical    0.921
    curl ... -d '{"inputs":"git push --force ..."}'  data-critical    0.895
    grep -rn 'rm -rf' .../pre-command.py             data-critical    0.892

Note the first of those scores HIGHER (0.921) than the real `rm -rf
/var/lib/data` it contains (0.835). The model is keying on the severe
tokens plus general command complexity, not on argument position.

So: v8 would NOT have prevented the false positives that motivated this
work — the actual firings were curl-with-JSON-payload, which is exactly the
shape it still gets wrong. It DOES fix the echo/cat/commit-message shape.
True positives were 5/5 (rm -rf, git push --force, git reset --hard,
kubectl delete namespace). This is a coverage gap with a clear training
answer (data-position negatives in a v9 slice), not a reason the approach
is wrong — but it IS the reason this ships advisory.

OTHER KNOWN GAPS
----------------
- The BLOCKED (git hygiene) rules are POLICY, not severity: "use explicit
  paths instead of `git add -A`" is a team convention, and v8's vocabulary
  (informative / situation-normal / data-critical) cannot express it. The
  classifier is a candidate replacement for the rm/find SOFT_BLOCKED and
  WARNED half only. Do not expect it to ever own the git half.
- No clause splitting yet. `a && b` goes to /v1/classify as one string,
  even though /v1/cascade exists precisely to rank clauses within a
  statement. Splitting a shell line correctly (quoting, subshells,
  heredocs) is its own problem; naive splitting on && would break the very
  data-position cases above by cutting inside a quoted payload.
- Command text is sent to lfm2d over the tailnet (no auth) and written to a
  local JSONL. Commands can contain secrets. The log is 0600 and local, and
  the tailnet is Amy-ruled trusted, but this is a real exposure to weigh
  before this runs in every session on every machine.
- Label vocabulary is checked per call, not assumed. `SEVERE_LABELS` is what
  the disagreement buckets key on, and if the deployed checkpoint doesn't
  speak any of them the row is bucketed `vocab_mismatch` rather than silently
  counted as "the model saw nothing." Caught by a test, after shipping the
  bug: the first version compared against a hard-coded 'data-critical', so a
  rollback to v6 would have logged every `destructive` verdict as unflagged.
- Fail-open on any lfm2d error. In advisory mode that is correct and
  invisible-by-design (the regex was always going to decide). In enforce
  mode fail-open would be a silent downgrade to regex behaviour, which is
  the house rule against silent fallbacks — hence `advisory_error` is
  always logged, and enforce mode must decide this explicitly.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# lfm2d advisory configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LFM2D_URL = os.environ.get('LFM2D_URL', 'http://lfm2d-1.taila4abc.ts.net:8088')
# Measured 75 ms mean / 80 ms max over 5 calls to the live pod on 2026-08-12.
# 400 ms is ~5x headroom: generous enough that a normal call never trips it,
# tight enough that a wedged endpoint costs the session under half a second
# per Bash call rather than hanging it.
LFM2D_TIMEOUT_S = float(os.environ.get('LFM2D_HOOK_TIMEOUT', '0.4'))
# 'advisory' (default): lfm2d is recorded, regex decides. 'off': no call at
# all. 'enforce': NOT READY — read the shortcomings above.
LFM2D_MODE = os.environ.get('LFM2D_HOOK_MODE', 'advisory')
# Which of the checkpoint's labels count as "the model flagged this", for the
# disagreement buckets only — this decides nothing, it labels rows.
#
# Configurable because the vocabulary is a property of the DEPLOYED
# checkpoint, not of this file: v8 speaks data-critical, v6 spoke
# destructive. The default matches what is deployed today; when it stops
# matching, `_disagreement` reports `vocab_mismatch` instead of quietly
# treating every verdict as unflagged.
SEVERE_LABELS = [
    l.strip() for l in os.environ.get('LFM2D_SEVERE_LABELS', 'data-critical').split(',') if l.strip()
]

CACHE_DIR = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'claude-hooks'
SOFT_BLOCK_CACHE = CACHE_DIR / 'soft-blocked.jsonl'
SOFT_BLOCK_TTL = 300  # 5 minutes
ADVISORY_LOG = CACHE_DIR / 'lfm2d-advisory.jsonl'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BLOCKED / SOFT_BLOCKED / WARNED — VERBATIM from the current hook.
# Do not "improve" these here. This file's whole claim is that the regex
# half behaves identically, so any disagreement in the log is the model's,
# not a rules edit. Port changes from the dotfiles hook, don't fork them.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCKED = {
    'git': [
        (r'\bgit\s+add\s+(-A|--all)\b', 'git add -A/--all', 'Use explicit paths or `git add -p`'),
        (r'\bgit\s+add\s+\.\s*($|[;&|])', 'git add .', 'Use explicit paths or `git add -p`'),
        (r'\bgit\s+commit\s+-[a-zA-Z]*a', 'git commit -a', 'Stage files explicitly first'),
        (r'\bgit\s+stash\b', 'git stash', 'modifies working tree state across all sessions'),
    ],
}

SOFT_BLOCKED = {
    'git': [
        (r'\bgit\s+commit\s+.*--amend\b', 'git commit --amend', 'rewrites history'),
        (r'\bgit\s+push\s+.*--force\b', 'git push --force', 'rewrites remote history'),
        (r'\bgit\s+reset\s+--hard\b', 'git reset --hard', 'discards uncommitted changes'),
    ],
    'rm': [
        (r'\brm\s+-[a-zA-Z]*r[a-zA-Z]*f', 'rm -rf', 'recursive force delete'),
        (r'\brm\s+-[a-zA-Z]*f[a-zA-Z]*r', 'rm -fr', 'recursive force delete'),
    ],
    'find': [
        (r'\bfind\b.*-exec\b', 'find -exec', 'executes commands on matches'),
        (r'\bfind\b.*-execdir\b', 'find -execdir', 'executes commands on matches'),
        (r'\bfind\b.*-ok\b', 'find -ok', 'executes commands on matches'),
    ],
}

WARNED = {
    'rm': [
        (r'\brm\s+.*\.local/share/', 'rm in ~/.local/share', 'contains app data'),
        (r'\brm\s+.*\.config/', 'rm in ~/.config', 'contains config files'),
        (r'\brm\s+.*\.db\b', 'rm *.db files', 'databases are hard to recover'),
        (r'\brm\s+-[a-zA-Z]*r', 'rm -r', 'recursive delete'),
        (r'\brm\s+-[a-zA-Z]*f', 'rm -f', 'force delete'),
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# lfm2d advisory call
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def lfm2d_classify(cmd: str) -> dict:
    """One /v1/classify call. Returns a dict that ALWAYS has 'ok'.

    Never raises: an advisory second opinion must not be able to break the
    guard it is advising. Every failure path records why, so a silent run
    of empty verdicts is distinguishable from a run of successful ones —
    the log is the whole product of this phase and a blank field in it
    would be worse than useless.
    """
    started = time.monotonic()
    body = json.dumps({'inputs': cmd}).encode()
    req = urllib.request.Request(
        f'{LFM2D_URL}/v1/classify',
        data=body,
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=LFM2D_TIMEOUT_S) as resp:
            results = json.loads(resp.read())
        r = results[0]
        return {
            'ok': True,
            'top': r['top'],
            'scores': r['scores'],
            'model_id': r['model_id'],
            # The audit pair. Label vocabularies change between checkpoints,
            # so a verdict without the hash it came from cannot be compared
            # across a redeploy — which is exactly what this log is for.
            'weight_hash': r['weight_hash'],
            'latency_ms': round((time.monotonic() - started) * 1000, 1),
        }
    except urllib.error.HTTPError as e:
        return {'ok': False, 'error': f'http {e.code}', 'detail': e.read()[:200].decode('utf-8', 'replace')}
    except Exception as e:  # timeout, DNS, connection refused, malformed body
        return {'ok': False, 'error': type(e).__name__, 'detail': str(e)[:200]}


def log_advisory(cmd: str, regex_verdict: dict, lfm2d_verdict: dict):
    """Append one comparison row. Best-effort: logging must never block a command."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        row = {
            'ts': time.time(),
            'cwd': os.getcwd(),
            'command': cmd,
            'regex': regex_verdict,
            'lfm2d': lfm2d_verdict,
            # Precomputed so analysis is a grep, not a join. 'regex_only' is
            # the false-positive candidate pile (the 6:1 ratio we are trying
            # to fix); 'lfm2d_only' is the recall pile (what the regex misses
            # and we would gain).
            'disagree': _disagreement(regex_verdict, lfm2d_verdict),
        }
        with open(ADVISORY_LOG, 'a') as f:
            f.write(json.dumps(row) + '\n')
        ADVISORY_LOG.chmod(0o600)  # commands can contain secrets
    except Exception:
        pass


def _disagreement(regex_verdict: dict, lfm2d_verdict: dict) -> str:
    """Bucket one comparison. Returns 'vocab_mismatch' rather than guessing
    when the checkpoint doesn't speak the labels we're keying on.

    fleet.md: "Label vocabularies change between checkpoints — consumers MUST
    read labels at runtime; hard-coded label strings are a live breakage."
    Comparing `top` against a fixed string is exactly that. Roll the pod back
    to v6 (informative/mutating/destructive) and every `destructive` verdict
    silently buckets as "the model did not flag it" — an entire deployment of
    confidently wrong rows, with nothing in the log to say so.

    No extra round trip is needed to avoid it: every /v1/classify response
    carries the checkpoint's full label set in `scores`, so the vocabulary is
    already in hand on every call.
    """
    if not lfm2d_verdict.get('ok'):
        return 'no_verdict'

    scores = lfm2d_verdict.get('scores') or {}
    known = [l for l in SEVERE_LABELS if l in scores]
    if not known:
        return 'vocab_mismatch'

    regex_flagged = regex_verdict['decision'] in ('deny', 'soft_deny', 'warn')
    lfm2d_flagged = lfm2d_verdict['top'] in known
    if regex_flagged and lfm2d_flagged:
        return 'agree_flag'
    if not regex_flagged and not lfm2d_flagged:
        return 'agree_clear'
    return 'regex_only' if regex_flagged else 'lfm2d_only'


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Soft-block cache — VERBATIM from the current hook
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cmd_hash(cmd: str) -> str:
    return hashlib.sha256(cmd.encode()).hexdigest()[:16]


def cache_soft_block(cmd: str, name: str):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {'hash': cmd_hash(cmd), 'name': name, 'ts': time.time()}
    with open(SOFT_BLOCK_CACHE, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def check_and_consume_approval(cmd: str) -> Optional[str]:
    if not SOFT_BLOCK_CACHE.exists():
        return None

    h = cmd_hash(cmd)
    now = time.time()
    kept, matched_name = [], None

    with open(SOFT_BLOCK_CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if now - entry.get('ts', 0) > SOFT_BLOCK_TTL:
                continue
            if entry.get('hash') == h and matched_name is None:
                matched_name = entry.get('name', 'command')
                continue
            kept.append(entry)

    if matched_name is not None or len(kept) == 0:
        if kept:
            with open(SOFT_BLOCK_CACHE, 'w') as f:
                for entry in kept:
                    f.write(json.dumps(entry) + '\n')
        else:
            SOFT_BLOCK_CACHE.unlink(missing_ok=True)

    return matched_name


def get_git_status_summary() -> Optional[dict]:
    try:
        r = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        lines = [l for l in r.stdout.strip().split('\n') if l]
        return {
            'untracked': sum(1 for l in lines if l.startswith('??')),
            'modified': sum(1 for l in lines if len(l) > 1 and l[1] in 'MD'),
            'total': len(lines),
        }
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Hook responses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def emit(decision: str, reason: Optional[str] = None):
    out = {'hookSpecificOutput': {'hookEventName': 'PreToolUse', 'permissionDecision': decision}}
    if reason is not None:
        out['hookSpecificOutput']['permissionDecisionReason'] = reason
    print(json.dumps(out))


def check_patterns(cmd: str, patterns: dict) -> Optional[tuple]:
    for category, rules in patterns.items():
        for pattern, name, hint in rules:
            if re.search(pattern, cmd):
                return (category, name, hint)
    return None


def regex_verdict_for(cmd: str) -> dict:
    """The CURRENT hook's decision, as data. Pure — no I/O, no side effects.

    Split out from the acting code so the advisory log records exactly what
    the regex half decided, and so it can be tested without a subprocess.
    """
    match = check_patterns(cmd, BLOCKED)
    if match:
        category, name, hint = match
        return {'decision': 'deny', 'rule': name, 'hint': hint, 'category': category}

    match = check_patterns(cmd, SOFT_BLOCKED)
    if match:
        category, name, reason = match
        return {'decision': 'soft_deny', 'rule': name, 'hint': reason, 'category': category}

    match = check_patterns(cmd, WARNED)
    if match:
        category, name, reason = match
        return {'decision': 'warn', 'rule': name, 'hint': reason, 'category': category}

    return {'decision': 'allow', 'rule': None, 'hint': None, 'category': None}


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        emit('allow')
        return

    if data.get('tool_name') != 'Bash':
        emit('allow')
        return

    cmd = data.get('tool_input', {}).get('command', '')

    # The approved-retry path short-circuits before any advisory call: the
    # user has already ruled on this exact command, and a model opinion on a
    # settled question is noise.
    approved = check_and_consume_approval(cmd)
    if approved:
        emit('allow', f"✅ `{approved}` approved by user, allowing retry.")
        return

    verdict = regex_verdict_for(cmd)

    lfm2d = {'ok': False, 'error': 'disabled'}
    if LFM2D_MODE != 'off':
        lfm2d = lfm2d_classify(cmd)
        log_advisory(cmd, verdict, lfm2d)

    if LFM2D_MODE == 'enforce':
        # Deliberately unimplemented. Wiring an untested enforcement path
        # and leaving it reachable is how a "temporary" mode ships. Read the
        # measured shortcomings at the top of this file; enforcement needs a
        # v9 slice covering data-position negatives first.
        print(
            'lfm2d hook: LFM2D_HOOK_MODE=enforce is not implemented — '
            'refusing to guess at a decision. Falling back to advisory.',
            file=sys.stderr,
        )

    # --- advisory: the regex decides, exactly as it does today ---
    d = verdict['decision']
    if d == 'deny':
        if verdict['category'] == 'git':
            status = get_git_status_summary()
            count = status['total'] if status else '?'
            emit('deny', f"⛔ Blocked: `{verdict['rule']}` would stage {count} files. {verdict['hint']}")
        else:
            emit('deny', f"⛔ Blocked: `{verdict['rule']}`. {verdict['hint']}")
    elif d == 'soft_deny':
        cache_soft_block(cmd, verdict['rule'])
        emit(
            'deny',
            f"🛑 `{verdict['rule']}` blocked ({verdict['hint']}). "
            f"If intentional, ask the user to confirm before retrying.",
        )
    elif d == 'warn':
        emit('allow', f"⚠️ `{verdict['rule']}` ({verdict['hint']})")
    else:
        emit('allow')


if __name__ == '__main__':
    main()
