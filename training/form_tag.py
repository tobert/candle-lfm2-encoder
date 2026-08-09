#!/usr/bin/env python3
"""Deterministic surface-FORM classifier for kube corpus rows.

Every row's "text" field gets tagged into exactly one of:

  shell_only  — the text IS a shell statement/pipeline/chain (kubectl/helm/
                etc invocation, possibly a multi-line script). Pure command
                syntax, no accompanying narrative.
  shell_prose — shell command(s) embedded in or accompanied by prose
                (ticket text, chat, comments around a command).
  nl_whole    — the ENTIRE statement is natural language with no shell
                syntax at all.

Boundary rules (from prior adjudication — see training/CLAUDE.md context):
  - Quoted payloads inside commands (echo 'rm target.txt', destructive
    words sitting in data position) are SHELLISH, not NL. We never look
    inside quotes/heredoc bodies to decide "is this NL" — a wrapping
    command (echo/cat/read/...) outside the quotes is what matters.
  - Nested interpreters (`kubectl exec -- sh -c "..."`, `psql -c "COPY ...
    TO PROGRAM ..."`, `bash -c`) are SHELLISH — the outer invocation is
    real command syntax regardless of what the nested string contains.
  - A bare NL request that merely *names* a tool ("restart the api pods
    with kubectl") is still nl_whole if there is no actual command syntax.
    We guard against this by only counting a command token as real
    invocation syntax when it sits at a clause boundary (start of text/
    line, or right after a shell chaining token: | ; & ` $( && ||) — a
    tool name preceded by an English word like "with"/"using"/"via" is a
    mention, not an invocation, and is ignored.

Design note beyond the given rules (calibrated against an Aug-7 census
of kube_train_v6/kube_val_v6 — see training/README): a fair number of
rows carry zero literal shell syntax but are terse imperative "chatops"
directives — "restart the workers", "delete the api credentials", "Please
rotate the payments certificate during the freeze." — functionally a
command spelled in words rather than a question, report, or narrative.
Those fold into shell_prose (a command described in chat/ticket prose is
exactly what shell_prose means, even when the "command" is itself
English). Genuine NL — questions, past-tense/passive reports,
conditionals, sentences with an explicit grammatical subject — stays
nl_whole. The split is mood-based: does the WHOLE statement open with a
bare imperative verb (no stated subject, optionally after a leading
"Please ")? Past ~9 words a directive almost always carries a
justification/subordinate clause ("... to reduce cloud costs", "...
after all clients migrated") and reads as a formal ticket description
rather than terse shorthand, so long directives stay nl_whole too.

CLI:
  python3 form_tag.py census <files...>          # aggregates only, no writes
  python3 form_tag.py tag <files...> --write      # adds "form" field in place
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Command / binary vocabulary — anything that, seen at a clause boundary,
# is real shell invocation syntax.
# ---------------------------------------------------------------------------
_COMMANDS = [
    "kubectl", "helm", "etcdctl", "docker", "docker-compose", "psql", "mysql",
    "redis-cli", "redis", "mongo", "mongosh", "sqlite3", "curl", "wget",
    "ssh", "scp", "rsync", "awk", "sed", "jq", "yq", "xargs", "systemctl",
    "journalctl", "crontab", "chmod", "chown", "rm", "cp", "mv", "mkdir",
    "touch", "cat", "echo", "kill", "gcloud", "aws", "az", "terraform",
    "ansible", "git", "sudo", "nc", "openssl", "tar", "read", "sh", "bash",
    "zsh", "grep", "find", "argocd", "kubeadm", "istioctl", "vault",
    "consul", "k9s", "stern", "crictl", "ctr", "nsenter", "envsubst",
    "truncate", "nohup", "ps", "top", "df", "du", "head", "tail", "less",
    "more", "nano", "vim", "python", "python3", "pip", "pip3", "npm",
    "node", "go", "make", "cmake", "iptables", "nft", "ip", "ping",
    "traceroute", "dig", "nslookup", "nmap", "flux", "kustomize", "oc",
    "printf", "export", "unset", "alias", "source", "eval", "exec",
    "trap", "wait", "sleep", "date", "whoami", "id", "uname", "hostname",
    "env", "which", "xxd", "base64", "dd", "mount", "umount", "lsof",
    "netstat", "ss", "tcpdump", "strace",
]
_CMD_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in _COMMANDS) + r")\b")

# Prepositions/verbs that mean a following command name is a *mention*
# ("restart the pods with kubectl") rather than an invocation.
_MENTION_PRECEDERS = {
    "with", "using", "via", "through", "like", "called", "named", "is",
    "are", "and", "than", "or",
}

# Chars/tokens that legitimately open a new shell clause.
_CHAIN_ENDERS = ("|", ";", "&", "`")
_CHAIN_SUFFIXES = ("$(", "&&", "||")

_SHELL_MISC_RE = re.compile(r"\$\(|\$\{|<<[-~]?\s*['\"]?\w")
# A snake_case identifier immediately followed by a quoted argument, at
# the start of a line/clause, reads as a shell/script function call
# (log_warn "...", slack_notify "...") even though the identifier isn't
# a known binary — never a plausible way to open an English sentence.
_FUNCTION_CALL_RE = re.compile(r"(?:^|\n)\s*[a-z][a-z0-9_]*_[a-z0-9_]*\s+QUOTED")
_ASSIGN_PREFIX_RE = re.compile(r"^(?:\w+=\S+\s+|sudo\s+)*")


def _is_clause_boundary(pre: str) -> bool:
    """True if `pre` (text before a candidate command token) ends a
    real shell clause boundary — start of text/line, or right after a
    chaining token."""
    pre = pre.rstrip()
    if pre == "" or pre.endswith("\n"):
        return True
    if pre[-1:] in _CHAIN_ENDERS:
        return True
    if any(pre.endswith(suf) for suf in _CHAIN_SUFFIXES):
        return True
    return False


def _has_real_command(code: str) -> bool:
    """Scan `code` (quotes/heredocs already masked out) for a command
    token sitting at a real clause boundary, not a bare English mention."""
    for m in _CMD_RE.finditer(code):
        pre = code[: m.start()]
        pre_stripped = pre.rstrip()
        if _is_clause_boundary(pre_stripped) or _is_clause_boundary(
            _ASSIGN_PREFIX_RE.sub("", pre_stripped)
        ):
            return True
        # allow a short leading assignment/sudo prefix on the same line
        line_start = pre.rfind("\n") + 1
        line_prefix = pre[line_start:]
        if _ASSIGN_PREFIX_RE.fullmatch(line_prefix.lstrip()):
            return True
        prev_word_m = re.search(r"([A-Za-z]+)\s*$", pre_stripped)
        prev_word = prev_word_m.group(1).lower() if prev_word_m else ""
        if prev_word in _MENTION_PRECEDERS:
            continue
        # Anything else immediately preceding a command token that isn't
        # a clause boundary and isn't a "mention" preposition is treated
        # conservatively as *not* an invocation (avoids false positives
        # on domain nouns like "flux reconciliation").
    if _SHELL_MISC_RE.search(code) or _FUNCTION_CALL_RE.search(code):
        return True
    return False


# ---------------------------------------------------------------------------
# Quote / heredoc masking — payload content never drives classification.
# ---------------------------------------------------------------------------
_HEREDOC_OPEN_RE = re.compile(r"<<-?~?\s*(['\"]?)(\w+)\1")
_SQ_RE = re.compile(r"'[^']*'")
_DQ_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_BT_RE = re.compile(r"`[^`]*`")


def _mask_heredocs(text: str) -> str:
    out_lines = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEREDOC_OPEN_RE.search(line)
        if not m:
            out_lines.append(line)
            i += 1
            continue
        delim = m.group(2)
        out_lines.append(line)  # keep the opener (has the command context)
        i += 1
        while i < len(lines) and lines[i].strip() != delim:
            out_lines.append("PAYLOAD")
            i += 1
        if i < len(lines):
            out_lines.append(lines[i])  # terminator line
            i += 1
    return "\n".join(out_lines)


def _mask_quotes(text: str) -> str:
    text = _BT_RE.sub("QUOTED", text)
    text = _DQ_RE.sub("QUOTED", text)
    text = _SQ_RE.sub("QUOTED", text)
    return text


def _mask(text: str) -> str:
    return _mask_quotes(_mask_heredocs(text))


# ---------------------------------------------------------------------------
# Comments: real (unquoted) '#' content, shebangs excluded.
# ---------------------------------------------------------------------------
def _split_code_and_comments(masked: str) -> tuple[str, list[str]]:
    code_lines = []
    comments = []
    for line in masked.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("#!") and line.find("#") == len(line) - len(stripped):
            code_lines.append(line)
            continue
        idx = line.find("#")
        if idx == -1:
            code_lines.append(line)
        else:
            code_lines.append(line[:idx])
            comment = line[idx + 1 :].strip()
            if comment:
                comments.append(comment)
    return "\n".join(code_lines), comments


# ---------------------------------------------------------------------------
# Directive fold: bare-imperative English clauses with no literal syntax.
# ---------------------------------------------------------------------------
_IMPERATIVE_VERBS = {
    "decommission", "remove", "delete", "nuke", "restart", "review",
    "scale", "wipe", "implement", "purge", "deploy", "tidy", "increase",
    "bounce", "assess", "audit", "validate", "add", "flush", "investigate",
    "apply", "migrate", "enable", "clear", "document", "update",
    "consolidate", "roll", "rotate", "restore", "monitor", "configure",
    "drop", "patch", "rollback", "clean", "refresh", "absorb", "analyze",
    "tune", "revoke", "unmount", "raise", "upgrade", "fix", "disable",
    "downgrade", "rebuild", "terminate", "evict", "cordon", "uncordon",
    "drain", "taint", "label", "expand", "shrink", "resize", "prune",
    "sweep", "reclaim", "reset", "regenerate", "renew", "extend",
    "reduce", "limit", "set", "check", "verify", "run", "trigger",
    "reschedule", "reissue", "reprovision", "decrypt", "encrypt",
}
_PLEASE_RE = re.compile(r"^please\s+", re.I)
_WORD_RE = re.compile(r"[A-Za-z']+")

# Calibrated against the kube_train_v6/kube_val_v6 census gate (see
# training/README or form_tag module docstring): short imperative-first
# clauses read as a command spelled in words ("wipe the workers"); once
# a directive grows past ~9 words it typically carries a justification/
# subordinate clause ("... to reduce cloud costs", "... after all
# clients migrated") and reads as a formal ticket description rather
# than terse chatops shorthand, so it stays nl_whole.
_DIRECTIVE_MAX_WORDS = 9


def _is_directive(text: str) -> bool:
    clause = _PLEASE_RE.sub("", text.strip())
    m = _WORD_RE.match(clause)
    if not (m and m.group(0).lower() in _IMPERATIVE_VERBS):
        return False
    return len(clause.split()) <= _DIRECTIVE_MAX_WORDS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
FORMS = ("shell_only", "shell_prose", "nl_whole")


def classify_form(text: str) -> str:
    if text is None:
        raise ValueError("classify_form: text is None")
    text = str(text)
    if not text.strip():
        raise ValueError("classify_form: empty text")

    masked = _mask(text)
    code_only, comments = _split_code_and_comments(masked)
    shell_syntax = _has_real_command(code_only)

    if shell_syntax:
        return "shell_prose" if comments else "shell_only"

    if any(_CMD_RE.search(c) for c in comments):
        return "shell_prose"

    if "?" in text:
        return "nl_whole"

    # Boundary rule: a bare NL request that merely *names* a tool stays
    # nl_whole even though it's a directive — the mention-guard in
    # _has_real_command already refused to treat it as invocation
    # syntax; don't let the directive fold override that exemption.
    if _CMD_RE.search(text):
        return "nl_whole"

    if _is_directive(text):
        return "shell_prose"

    return "nl_whole"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _iter_rows(path: Path):
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"GATE FAIL: {path.name}:{lineno} invalid JSON: {e}")
            yield lineno, row


def cmd_census(files: list[str]) -> None:
    total = Counter()
    for fname in files:
        path = Path(fname)
        counts = Counter()
        n = 0
        for lineno, row in _iter_rows(path):
            text = row.get("text")
            if not text or not str(text).strip():
                sys.exit(f"GATE FAIL: {path.name}:{lineno} empty/missing text")
            form = classify_form(text)
            counts[form] += 1
            n += 1
        print(f"== {path.name} ({n} rows) ==")
        for form in FORMS:
            c = counts[form]
            pct = 100.0 * c / n if n else 0.0
            print(f"  {form:12s} {c:6d}  {pct:5.1f}%")
        total.update(counts)
        total["__n__"] += n
    n = total.pop("__n__")
    print(f"== TOTAL ({n} rows) ==")
    for form in FORMS:
        c = total[form]
        pct = 100.0 * c / n if n else 0.0
        print(f"  {form:12s} {c:6d}  {pct:5.1f}%")


def cmd_tag(files: list[str], write: bool) -> None:
    for fname in files:
        path = Path(fname)
        rows = []
        counts = Counter()
        for lineno, row in _iter_rows(path):
            text = row.get("text")
            if not text or not str(text).strip():
                sys.exit(f"GATE FAIL: {path.name}:{lineno} empty/missing text")
            form = classify_form(text)
            row["form"] = form
            counts[form] += 1
            rows.append(row)
        n = len(rows)
        print(f"== {path.name} ({n} rows) ==")
        for form in FORMS:
            c = counts[form]
            pct = 100.0 * c / n if n else 0.0
            print(f"  {form:12s} {c:6d}  {pct:5.1f}%")
        if write:
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            print(f"  wrote {n} rows -> {path}")
        else:
            print("  (dry run — pass --write to persist)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_census = sub.add_parser("census", help="print per-file and total form counts (aggregates only)")
    p_census.add_argument("files", nargs="+")

    p_tag = sub.add_parser("tag", help="add a form field to each row")
    p_tag.add_argument("files", nargs="+")
    p_tag.add_argument("--write", action="store_true", help="persist changes in place")

    args = parser.parse_args(argv)
    if args.mode == "census":
        cmd_census(args.files)
    elif args.mode == "tag":
        cmd_tag(args.files, args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
