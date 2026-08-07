import json

S = "/tmp/claude-1000/-home-atobey-src-candle-lfm2-encoder/0dffdd2b-cf60-42ad-b08d-bab60851664b/scratchpad"
cases = json.load(open(f"{S}/cases_scored.json"))
for i, c in enumerate(cases, 1):
    c["_n"] = i

BANDS = [
    ("undecidable", "Undecidable — the cost estimate for the whole relabel",
     "Every row here is one I think **text alone cannot settle**. That call is the "
     "entire cost estimate: each undecidable row is either a row we drop (shrinking "
     "the corpus) or a row we keep with a flat target (teaching the model to "
     "escalate). If you think I over-called this band, the relabel is cheaper than I "
     "said. If under-called, more expensive."),
    ("natural_language", "Natural language — including the open v3/v6 question",
     "The register the NL holdout is drawn from. Row 51 is the one from this morning: "
     "v3 smeared to 0.523, v6 sharpened to 0.912, and the convention for interrogative "
     "NL has never been written down."),
    ("critical", "Data-critical anchors",
     "Rows I believe are clear-cut unrecoverable. Any row here you would move is a "
     "boundary correction that matters more than a hard case."),
    ("recoverable", "Situation-normal anchors",
     "Rows I believe are clear-cut recoverable. Same weight as above."),
    ("laundered", "Laundered — severity hidden by surrounding clauses",
     "These interact with this morning's clause work: the label is for the STATEMENT, "
     "and the clause path is what finds it. Two of them also test whether the axis "
     "covers disclosure at all."),
    ("informative", "Informative anchors",
     "Reads and mentions. Included mainly to anchor you and to keep the eval balanced; "
     "I expect few disagreements."),
]

L = []
w = L.append
w("# Recoverability re-axis — adjudication worksheet")
w("")
w("58 rows. **I set `mine`, you set `gold`.** Disagree freely — the whole point of")
w("splitting it this way is that rows written by one model family and labelled by")
w("the same family teach a corpus its author's blind spots instead of the task.")
w("")
w("## The three labels")
w("")
w("| label | test |")
w("|---|---|")
w("| `informative` | reads state, or mentions a command without executing it |")
w("| `situation-normal` | changes state, and **backs out by design** |")
w("| `data-critical` | destroys data with **no obvious backout** |")
w("")
w("Judge against **default convention** — a dependency directory is regenerable by")
w("definition, a PVC is not. Blast radius (staging vs prod) deliberately stays OUT")
w("of the label; it belongs to execution context and would muddy the axis.")
w("")
w("`escalate` is a second, independent call: **should a well-calibrated model be")
w("uncertain here?** It is not \"is this hard for me\" — it is \"is the answer absent")
w("from the text\". This is the flag the haiku escalation lane would run on.")
w("")
w("## Before the rows: four things only you can settle")
w("")
w("1. **Three classes or four?** I read your \"this looks good\" as the three-class")
w("   re-axis (`destructive` renamed to `data-critical`, boundary moved to")
w("   recoverability). The four-class alternative keeps `destructive` as-is and")
w("   splits only today's `mutating` — which leaves a recursive delete of a")
w("   dependency directory flagged destructive, since it never moves. Everything")
w("   below assumes three.")
w("2. **Row 43, `DELETE FROM sessions WHERE created_at < ...`** — unrecoverable")
w("   *and* completely routine. This is where the axis strains: recoverability and")
w("   normality come apart. Does expiring-by-design data get an exemption, or does")
w("   the axis just say data-critical and let the consumer's threshold handle it?")
w("3. **Row 44, `kubectl cp prod/api-7f9d:/data/dump.sql ./`** — destroys nothing,")
w("   and the disclosure is irreversible. The axis has no opinion about")
w("   exfiltration. Should it, or does that belong entirely to the always-on")
w("   PII-Detector lane?")
w("4. **Am I over- or under-calling `escalate`?** I marked 19 of 58. That number")
w("   sets how much flat-target data a calibrated v7 would train on.")
w("")
w("## If you only have fifteen minutes")
w("")
w("Read the **Undecidable** band (12 rows) and the four questions above. Those")
w("decide the shape of the work; the anchors mostly just confirm it.")
w("")
w("The warning mark flags a row where **v6 today disagrees with my proposal** —")
w("23 of 58. Those are where the relabel actually changes something.")
w("")
w("---")
w("")

for band, title, blurb in BANDS:
    rows = [c for c in cases if c["band"] == band]
    dis = sum(1 for c in rows if not c["_agree"])
    w(f"## {title}")
    w("")
    w(f"*{len(rows)} rows, {dis} where v6 disagrees with me.* {blurb}")
    w("")
    for c in rows:
        flag = "" if c["_agree"] else "  ⚠︎"
        esc = "yes" if c["escalate"] else "no"
        w(f"**{c['_n']}.** `{c['text']}`{flag}")
        w("")
        w(f"- backout — {c['backout']}")
        w(f"- note — {c['note']}")
        w(f"- mine: **{c['proposed']}** · escalate: **{esc}** · "
          f"v6 today: {c['_v6']} {c['_v6p']:.3f}")
        w("- **gold:** &nbsp;&nbsp;&nbsp; **escalate:** &nbsp;&nbsp;&nbsp; *(notes: )*")
        w("")
    w("---")
    w("")

w("## When you are done")
w("")
w("Hand it back however is easiest — filled-in file, or just the row numbers you")
w("changed. I will fold the rulings into `cases.json` as `gold`, and then we have")
w("both an adjudicated eval slice and a measured cost for the corpus relabel.")
w("")

open("training/recoverability/ADJUDICATE.md", "w").write("\n".join(L))
print("wrote training/recoverability/ADJUDICATE.md:", len(L), "lines")
