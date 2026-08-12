#!/usr/bin/env python3
"""Shell clause splitter for the advisory hook — stdlib only, never raises.

Splits a compound shell command into the sequenced clauses a caller would
send to lfm2d's /v1/cascade, which ranks clauses by severity WITHIN a
statement. Exists because of a measured defect: whole-command classification
dilutes severe clauses (`rm -rf -- "$d.venv"` scores 0.540 alone, 0.047
inside its 624-char script — 11x), and 92% of real mined commands are
compound.

WHAT SPLITS: top-level `&&`, `||`, `;` (and case's `;;`/`;&`), `&`
(background), and newlines.

WHAT NEVER SPLITS — each is a measured hazard, not caution:
- Inside quotes. `curl -d '{"inputs":"rm -rf /x && y"}'` must stay ONE
  clause; cutting it fabricates a bare `rm -rf` out of data. This is the
  data-position trap the whole hook exists to measure.
- Inside `$(...)`, `` `...` ``, `(...)` — embedded/substituted content.
  A clause that IS a parenthesized group `(a && b)` gets its interior
  split afterward, because those are sequenced commands, just grouped.
- Heredoc bodies, and the whole statement carrying a pending heredoc. The
  body is data. Coarse on `cat <<EOF && cmd` (stays one clause) — coarse
  beats cutting a body.
- Pipelines (`|`, `|&`). NOT an oversight: `cat f | grep 'git push
  --force'` measured 0.994 informative whole, while a bare
  grep-for-severe-pattern clause is a shape v8 misreads (0.892
  data-critical). Splitting pipes would manufacture false positives out of
  currently-correct behaviour. Revisit only with measurements in hand.

WHAT IS DROPPED: comments (they are prose — leaving them in feeds the
classifier data-position bait), leading control-flow keywords
(`if`/`then`/`do`/`while`/`!`…), and closer-only fragments (`fi`, `done`,
`esac`). No bash grammar beyond that; `case` patterns yield odd-but-harmless
fragments.

The public function is `split_clauses(cmd)`. It must never raise: the hook
falls back to whole-command /v1/classify when it returns [] or [cmd], and
an advisory helper must not be able to break the guard it advises. Internal
errors return the whole command as a single clause (logged upstream via
clause count == 1).
"""
from __future__ import annotations

# Strippable leading keywords: they prefix a real command in the same clause.
_STRIP_LEADING = {'if', 'elif', 'then', 'else', 'do', 'while', 'until', 'time', '!'}
# Fragments that are ONLY structure — no command content — dropped outright.
_DROP_ALONE = {'fi', 'done', 'esac', 'then', 'else', 'do', 'in', '{', '}'}
# Bound what one hook call can send to /v1/cascade; a 17,942-char monster
# should not turn into a 200-forward inference bill on every keystroke.
MAX_CLAUSES = 64


def split_clauses(cmd: str) -> list[str]:
    """Split `cmd` into top-level clauses. Never raises."""
    try:
        return _split(cmd, recurse=2)[:MAX_CLAUSES]
    except Exception:
        # A splitter bug must degrade to v8's status quo (whole-command
        # classify), not break the hook. Upstream logs clause_count==1.
        stripped = cmd.strip()
        return [stripped] if stripped else []


def _split(cmd: str, recurse: int) -> list[str]:
    out = []
    for raw in _scan(cmd):
        clause = _strip_keywords(raw.strip())
        if not clause or clause in _DROP_ALONE:
            continue
        # A clause that is exactly a parenthesized group holds sequenced
        # commands — `(cd /tmp && rm -rf x)` is the common idiom. Split the
        # interior. Bounded recursion; nested groups past that stay whole.
        if recurse > 0 and clause.startswith('(') and _wraps_fully(clause):
            inner = _split(clause[1:-1], recurse - 1)
            if len(inner) > 1:
                out.extend(inner)
                continue
        out.append(clause)
    return out


def _strip_keywords(clause: str) -> str:
    while True:
        parts = clause.split(None, 1)
        if len(parts) == 2 and parts[0] in _STRIP_LEADING:
            clause = parts[1]
            continue
        return clause


def _wraps_fully(clause: str) -> bool:
    """True if the opening '(' closes exactly at the last character."""
    if not clause.endswith(')'):
        return False
    end = _skip_group(clause, 1, ')')
    return end == len(clause)


def _scan(s: str) -> list[str]:
    """The state machine. Returns raw segments between top-level separators.

    Quoted/substituted regions are consumed by dedicated skippers that
    understand their own nesting, so the main loop only ever sees genuinely
    top-level characters.
    """
    segs: list[str] = []
    buf: list[str] = []
    pending_heredocs: list[tuple[str, bool]] = []  # (delimiter, strip_tabs)
    i, n = 0, len(s)

    def flush():
        segs.append(''.join(buf))
        buf.clear()

    while i < n:
        ch = s[i]

        # Escapes: the pair is opaque (covers line continuation).
        if ch == '\\':
            buf.append(s[i:i + 2])
            i += 2
            continue

        # Opaque regions — copied verbatim, separators inside are content.
        if ch == "'":
            j = _skip_single(s, i + 1)
            buf.append(s[i:j])
            i = j
            continue
        if ch == '"':
            j = _skip_double(s, i + 1)
            buf.append(s[i:j])
            i = j
            continue
        if ch == '`':
            j = _skip_backtick(s, i + 1)
            buf.append(s[i:j])
            i = j
            continue
        if s.startswith('$(', i):
            j = _skip_group(s, i + 2, ')')
            buf.append(s[i:j])
            i = j
            continue
        if ch == '(':
            j = _skip_group(s, i + 1, ')')
            buf.append(s[i:j])
            i = j
            continue

        # Comments: dropped entirely — comment prose is data-position bait,
        # and a quote character inside one must not open a quote region.
        if ch == '#' and (not buf or buf[-1][-1:] in ('', ' ', '\t', '\n', ';', '&', '|', '(')):
            while i < n and s[i] != '\n':
                i += 1
            continue

        # Heredoc operator (not <<< herestring, not part of one).
        if s.startswith('<<', i) and not s.startswith('<<<', i) and (i == 0 or s[i - 1] != '<'):
            delim, strip_tabs, j = _heredoc_delim(s, i + 2)
            if delim is not None:
                pending_heredocs.append((delim, strip_tabs))
                buf.append(s[i:j])
                i = j
                continue
            # `<<` with no parseable delimiter: literal.
            buf.append('<<')
            i += 2
            continue

        if ch == '\n':
            if pending_heredocs:
                # The bodies belong to the current clause. Consume them all;
                # no splitting inside — bodies are data.
                buf.append(ch)
                i += 1
                for delim, strip_tabs in pending_heredocs:
                    i = _consume_heredoc_body(s, i, delim, strip_tabs, buf)
                pending_heredocs.clear()
                # The delimiter line ends the statement: what follows on the
                # next line is a new command, so this is a split point.
                flush()
                continue
            flush()
            i += 1
            continue

        # Separators. While a heredoc is pending on this line, suppress
        # splitting outright: `cat <<EOF && cmd` stays one clause, because
        # cutting it would orphan the body onto the wrong clause.
        if not pending_heredocs:
            if s.startswith('&&', i) or s.startswith('||', i):
                flush()
                i += 2
                continue
            if ch == ';':
                flush()
                # absorb case terminators ;; ;& ;;&
                i += 1
                while i < n and s[i] in ';&':
                    i += 1
                continue
            if ch == '&':
                nxt = s[i + 1] if i + 1 < n else ''
                prev = ''.join(buf).rstrip()[-1:] if buf else ''
                # `2>&1` / `<&` are redirections; `&>` redirects; `|&` pipes.
                if nxt != '>' and prev not in ('>', '<', '|'):
                    flush()
                    i += 1
                    continue

        buf.append(ch)
        i += 1

    flush()
    return segs


def _skip_single(s: str, i: int) -> int:
    """Past the closing single quote. No escapes exist inside '…'."""
    while i < len(s):
        if s[i] == "'":
            return i + 1
        i += 1
    return i  # unterminated: rest of string is the region


def _skip_double(s: str, i: int) -> int:
    """Past the closing double quote, honoring \\ and nested $(…)/`…`."""
    while i < len(s):
        ch = s[i]
        if ch == '\\':
            i += 2
            continue
        if ch == '"':
            return i + 1
        if s.startswith('$(', i):
            i = _skip_group(s, i + 2, ')')
            continue
        if ch == '`':
            i = _skip_backtick(s, i + 1)
            continue
        i += 1
    return i


def _skip_backtick(s: str, i: int) -> int:
    while i < len(s):
        ch = s[i]
        if ch == '\\':
            i += 2
            continue
        if ch == '`':
            return i + 1
        i += 1
    return i


def _skip_group(s: str, i: int, closer: str) -> int:
    """Past the matching `closer`, tracking nesting and inner quotes."""
    depth = 1
    while i < len(s):
        ch = s[i]
        if ch == '\\':
            i += 2
            continue
        if ch == "'":
            i = _skip_single(s, i + 1)
            continue
        if ch == '"':
            i = _skip_double(s, i + 1)
            continue
        if ch == '`':
            i = _skip_backtick(s, i + 1)
            continue
        if s.startswith('$(', i):
            i = _skip_group(s, i + 2, ')')
            continue
        if ch == '(':
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return i


def _heredoc_delim(s: str, i: int):
    """Parse the delimiter word after `<<`. Returns (delim, strip_tabs, end).

    `delim` is None when nothing parseable follows (literal `<<`).
    """
    n = len(s)
    strip_tabs = False
    if i < n and s[i] == '-':
        strip_tabs = True
        i += 1
    while i < n and s[i] in ' \t':
        i += 1
    if i >= n:
        return None, False, i
    quote = s[i] if s[i] in ('"', "'") else None
    if quote:
        j = i + 1
        while j < n and s[j] != quote:
            j += 1
        if j >= n:
            return None, False, i
        return s[i + 1:j], strip_tabs, j + 1
    j = i
    if s[j] == '\\':  # <<\EOF — delimiter escaped, same effect as quoted
        j += 1
        i += 1
    while j < n and (s[j].isalnum() or s[j] in '_-.'):
        j += 1
    if j == i:
        return None, False, i
    return s[i:j], strip_tabs, j


def _consume_heredoc_body(s: str, i: int, delim: str, strip_tabs: bool, buf: list) -> int:
    """Copy lines into buf until the delimiter line (inclusive)."""
    n = len(s)
    while i < n:
        eol = s.find('\n', i)
        if eol == -1:
            eol = n
        line = s[i:eol]
        check = line.lstrip('\t') if strip_tabs else line
        buf.append(s[i:min(eol + 1, n)])
        i = eol + 1
        if check == delim:
            return i
    return i
