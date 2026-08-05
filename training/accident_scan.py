#!/usr/bin/env python3
"""Scan training data for forkbombs and classic destructive-accident patterns.

Read-only. Prints aggregate counts by label only — never a raw row.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

DATA = Path.home() / ".local/share/lfm2-training-data/prepared"

PATTERNS = {
    "forkbomb": re.compile(r":\(\)\s*\{\s*:\|:&\s*\}|fork\s*bomb", re.I),
    "rm -rf / (root)": re.compile(r"rm\s+(-[a-z]*\s+)*-?rf?\s+/(\s|$|\*)"),
    "rm -rf ~ or $VAR": re.compile(r"rm\s+-rf\s+(~|\$[A-Za-z_{]|\"\$)"),
    "dd to blockdev": re.compile(r"dd\s+.*of=/dev/(sd|nvme|hd|vd)"),
    "mkfs": re.compile(r"\bmkfs(\.| )"),
    "chmod -R 777 /": re.compile(r"chmod\s+-R\s+777\s+/(\s|$)"),
    "curl|sh pipe": re.compile(r"(curl|wget)[^|;]*\|\s*(sudo\s+)?(ba)?sh"),
    "redirect to blockdev": re.compile(r">\s*/dev/(sd|nvme|hd)"),
    "DROP TABLE/DATABASE": re.compile(r"\bDROP\s+(TABLE|DATABASE)\b", re.I),
    "FLUSHALL/FLUSHDB": re.compile(r"\bFLUSH(ALL|DB)\b"),
    "git force push": re.compile(r"push\s+.*--force|push\s+-f\b"),
    "kubectl delete --all": re.compile(r"delete\s+.*--all\b"),
    "delete ns kube-system": re.compile(r"delete\s+(ns|namespace)\s+kube-system"),
    "etcdctl del": re.compile(r"etcdctl\s+del\b"),
    "shutdown/reboot/halt": re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    "kill init / killall": re.compile(r"kill\s+-9\s+1\b|\bkillall\b"),
    "history -c / shred": re.compile(r"history\s+-c|\bshred\b"),
}

files = sorted(DATA.glob("kube_slice_*.jsonl")) + [
    DATA / "train_4000.jsonl", DATA / "val_4000.jsonl", DATA / "eval_realistic.jsonl"
]

grand = defaultdict(int)
for p in files:
    if not p.exists():
        continue
    hits = defaultdict(lambda: defaultdict(int))
    n = 0
    for line in open(p):
        r = json.loads(line)
        n += 1
        for name, rx in PATTERNS.items():
            if rx.search(r["text"]):
                hits[name][r.get("label", "?")] += 1
                grand[name] += 1
    print(f"{p.name} ({n} rows):" + ("" if hits else " no hits"))
    for name, d in sorted(hits.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {name}: " + " ".join(f"{k}={v}" for k, v in sorted(d.items())))

print()
print("grand totals:", dict(sorted(grand.items(), key=lambda kv: -kv[1])) or "NONE")
