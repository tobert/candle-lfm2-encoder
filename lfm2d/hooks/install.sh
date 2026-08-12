#!/usr/bin/env bash
# Swap Amy's PreToolUse Bash hook to the lfm2d advisory one, or back.
#
#   install.sh            # show what is installed now
#   install.sh install    # point settings at the advisory hook
#   install.sh uninstall  # point it back at the dotfiles hook
#
# The advisory hook makes the SAME decisions as the dotfiles hook -- that is
# gated by test_parity.py, 33 cases byte-for-byte -- and additionally asks
# lfm2d for a second opinion which it only writes to a log. Rollback is this
# script with `uninstall`, or editing one string in settings.json.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# python3, not `python`: both are 3.14.6 here, but the tests were run under
# python3 and a guard should not inherit whatever `python` resolves to later.
ADVISORY_CMD="python3 $REPO/lfm2d/hooks/pre_command_advisory.py"
ADVISORY="$REPO/lfm2d/hooks/pre_command_advisory.py"
# The settings entry points at ~/.claude/hooks/pre-command.py, which is a
# symlink into dotfiles (verified identical). The parity gate runs against
# the dotfiles path; the settings string is what we restore.
BASELINE="$HOME/src/tobert/dotfiles/.claude/hooks/pre-command.py"
SETTINGS="$HOME/.claude/settings.json"
# The exact command string that was installed before we touched it, so
# `uninstall` restores what was there rather than what this script guesses
# was there.
PREVIOUS="$HOME/.claude/.lfm2d-hook-previous"

die() { echo "error: $*" >&2; exit 1; }

current_command() {
  python3 - "$SETTINGS" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception:
    print(''); raise SystemExit
for group in s.get('hooks', {}).get('PreToolUse', []):
    for h in group.get('hooks', []):
        cmd = h.get('command', '')
        if 'pre-command.py' in cmd or 'pre_command_advisory.py' in cmd:
            print(cmd); raise SystemExit
print('')
PY
}

set_command() {
  python3 - "$SETTINGS" "$1" <<'PY'
import json, sys
path, new = sys.argv[1], sys.argv[2]
s = json.load(open(path))
changed = 0
for group in s.get('hooks', {}).get('PreToolUse', []):
    for h in group.get('hooks', []):
        cmd = h.get('command', '')
        if 'pre-command.py' in cmd or 'pre_command_advisory.py' in cmd:
            h['command'] = new
            changed += 1
if changed != 1:
    # Refuse rather than guess. Zero means the hook is wired somewhere this
    # script does not understand; two or more means editing one is as likely
    # to be wrong as right, and a half-swapped guard is worse than either
    # state.
    raise SystemExit(f'refusing: expected exactly 1 matching hook entry, found {changed}')
json.dump(s, open(path, 'w'), indent=2)
open(path, 'a').write('\n')
print(f'set -> {new}')
PY
}

preflight() {
  [ -f "$SETTINGS" ]  || die "no settings at $SETTINGS"
  [ -f "$ADVISORY" ]  || die "advisory hook missing at $ADVISORY"
  [ -f "$BASELINE" ]  || die "dotfiles hook missing at $BASELINE"
  python3 -c "import json;json.load(open('$SETTINGS'))" \
    || die "$SETTINGS is not valid JSON -- fix that before swapping a guard"
}

case "${1:-status}" in
  status)
    preflight
    cur="$(current_command)"
    echo "settings: $SETTINGS"
    echo "hook:     ${cur:-<none found>}"
    case "$cur" in
      *pre_command_advisory.py*) echo "state:    ADVISORY (lfm2d second opinion, logged not enforced)" ;;
      *pre-command.py*)          echo "state:    BASELINE (regex only)" ;;
      *)                         echo "state:    unrecognized" ;;
    esac
    echo
    echo "advisory log: ${XDG_CACHE_HOME:-$HOME/.cache}/claude-hooks/lfm2d-advisory.jsonl"
    ;;

  install)
    preflight
    # Parity is the whole basis for calling this swap safe, so it is a GATE,
    # not a suggestion. If the two hooks ever decide differently, this script
    # must not be the thing that finds out in production.
    echo "== parity gate =="
    python3 "$REPO/lfm2d/hooks/test_parity.py" "$BASELINE" >/dev/null \
      || die "parity gate FAILED -- not installing. Run it directly to see the diff."
    echo "33/33 identical decisions vs $BASELINE"
    echo
    cur="$(current_command)"
    case "$cur" in
      *pre_command_advisory.py*) echo "already installed; nothing to do."; exit 0 ;;
    esac
    [ -n "$cur" ] || die "no recognizable PreToolUse hook in settings -- refusing to invent one"

    cp -p "$SETTINGS" "$SETTINGS.bak-$(date +%Y%m%d-%H%M%S)"
    printf '%s\n' "$cur" > "$PREVIOUS"
    set_command "$ADVISORY_CMD"
    echo
    echo "Installed. Decisions are unchanged; lfm2d is advisory only."
    echo "Previous: $cur  (saved to $PREVIOUS)"
    echo "Backup:   $SETTINGS.bak-*"
    echo "Rollback: $0 uninstall"
    ;;

  uninstall)
    preflight
    if [ -f "$PREVIOUS" ]; then
      restore="$(cat "$PREVIOUS")"
    else
      # Nothing recorded -- fall back to the settings string that was in use
      # before this script existed, rather than an absolute dotfiles path
      # that would silently change how the hook is invoked.
      restore="python ~/.claude/hooks/pre-command.py"
      echo "note: no saved previous command; restoring the known default"
    fi
    cp -p "$SETTINGS" "$SETTINGS.bak-$(date +%Y%m%d-%H%M%S)"
    set_command "$restore"
    rm -f "$PREVIOUS"
    echo "Reverted to: $restore"
    ;;

  *) die "usage: $0 [status|install|uninstall]" ;;
esac
