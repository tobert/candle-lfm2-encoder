# Router eval data

Evaluation fixtures for `LFM2.5-Encoder-350M-Prompt-Router`. This
checkpoint is **zero-shot**: nothing here is training data. You hand it
a list of route strings and a prompt; it embeds each in a shared 256-d
space (`rule_proj_dim`) and returns a softmax over routes. Route quality
depends entirely on how the route string is *worded* — a bare label like
`"opus"` carries almost no semantic signal, a sentence describing the
work it suits carries a lot. These files exist to measure that effect
and to stress-test the router before anything is built on top of it.

No Rust, no model runs, no training happen in this directory — it is
data only.

## Files

| file | contents |
|---|---|
| `kaibo_models.json` | route set: dispatch a request to a kaibo model tier |
| `kaibo_models_probes.json` | probe prompts for the set above |
| `safety_specialists.json` | route set: dispatch a proposed action to safety specialist(s) |
| `safety_specialists_probes.json` | probe prompts for the set above |

### Route set schema

```json
{
  "_schema": "...",
  "_source": "where the lane roster was grounded",
  "set": "kaibo_models | safety_specialists",
  "lanes": [
    {
      "id": "deepseek",
      "variant_bare": "deepseek",
      "variant_capability": "one clause describing the work this lane suits",
      "variant_rich": "1-2 sentences: the work AND its difficulty/cost profile"
    }
  ]
}
```

Three parallel wordings per lane, same lane **id**, same **order**,
same **count** — so a run can swap `variant_bare` for `variant_rich`
without changing anything else, and results are directly comparable.
To build the actual `--routes-file` input for a given variant, project
out one field per lane (e.g. `jq '[.lanes[] | {id, route: .variant_rich}]'`).

### Probe schema

```json
{
  "text": "the prompt / command / request being routed",
  "expect": "<lane id>",
  "band": "easy | ambiguous | adversarial | multi | guard_evasion",
  "note": "why this band, and the rival lane where relevant",
  "expect_multi": ["<lane id>", "<lane id>"]
}
```

`expect_multi` is present only on `band: "multi"` and `band:
"guard_evasion"` items that legitimately need two specialists; `expect`
is still set, to the more severe of the two, so single-lane top-1
scoring stays well-defined.

## What each band is testing

- **easy** (~15 kaibo / 13 safety): one lane is unmistakable. This is
  the floor — if a variant misses these, it isn't usable regardless of
  how it does elsewhere.
- **ambiguous** (~10 / 9): genuinely between two lanes. `expect` holds
  the better answer, `note` names the rival. These calibrate whether a
  variant's softmax puts real probability mass on the *plausible*
  second choice, or scatters it randomly — that distinction matters for
  any consumer that looks past top-1.
- **adversarial** (~5 / 6): the surface form primes one lane, the true
  answer is another (a "quick one-liner" that's actually a crypto
  change; an `ls` that hides a home-directory wipe). These are the
  cases that matter most in practice and are most likely to break a
  variant that keys on surface keywords instead of the described
  *intent*.
- **multi** (~5 kaibo, ~7 safety — **safety especially**, since one
  compromised command routinely spans domains): an action that
  legitimately needs two specialists in parallel — a `kubectl exec`
  that runs a `redis FLUSHALL` needs both the cluster specialist and
  the datastore specialist; a `curl | bash` needs both fetch-provenance
  and execution review. `expect_multi` names both; `expect` picks the
  more severe. These test whether softmax mass splits sensibly across
  two correct answers, which is the deciding factor for whether a
  single softmax router can serve fan-out, or whether the real
  cascade needs per-route thresholding on raw (pre-softmax) scores
  instead of top-1/top-k.
- **guard_evasion** (safety only, ~8): destructive intent restated to
  dodge a literal-string guard, grounded in a real incident (see
  below). Distinct from `adversarial` because the axis isn't "surface
  form suggests the wrong lane" — the axis is "two commands with
  provably equal blast radius must land in the *same* lane at *similar*
  confidence, even though only one of them contains the string a naive
  hook was written to catch." At least three pairs (`rm -rf` vs `rm -r`,
  vs `find -delete`, vs `git clean -xdf`) appear as both members with
  the same `expect`, plus standalone path-obscuring and
  severity-laundering (benign-looking command chains) examples.

## Why route wording is the variable under test

Route quality is entirely a function of the string, since nothing is
trained. `variant_bare` is the floor (does the model already associate
the bare token with the right kind of prompt at all — unlikely for
`"luna"`/`"sol"`/`"terra"`, since those are arbitrary code names with no
inherent semantics). `variant_capability` adds one clause of grounding.
`variant_rich` adds cost/difficulty framing, which specifically targets
the adversarial band (a router that only sees "reasoning" clauses can't
distinguish "quick one-liner, actually needs judgment" from "actually
is quick" — a difficulty/cost clause gives it something to weigh against
the prompt's own claimed urgency/casualness).

Probe *text* varies surface form deliberately (imperative shell
command, natural-language request, pasted error/log, code block, chat
aside like "ping?") rather than persona, per the standing finding in
this repo (`data-diversity-form-coverage` memory): covering a missing
*surface form* moved a holdout classifier 25%→83%, while persona
variance on an already-covered form bought nothing. The same law is
assumed to hold for a zero-shot router and these probes are built
accordingly — don't add more paraphrases of a covered form; add a form
that isn't covered yet (voice-transcript-style text, a linter/CI
comment, etc. are open gaps).

## The `kaibo_models` lane roster: what's real, and one lane I dropped

Grounded in `~/.config/kaibo/config.toml` (live config on this box) and
`~/src/kaibo/docs/casts.md`, both read 2026-08-05. `haiku` / `sonnet` /
`opus` / `fable` are real Anthropic model ids used across kaibo's
`chimera`, `anthropic-batch`, and `fable` casts; `luna` / `terra` / `sol`
are the real GPT 5.6 tiers used in `[casts.gpt]` /
`[casts.gpt-deliberate]`; `deepseek` is the configured default
(`server.cast = "deepseek"`) and reliable bulk carrier.

That's 8 distinct real tiers. The brief asked for 5-7 lanes, so **I
dropped `fable`**, folding its role into `opus`. Both are Anthropic's
"don't do this live, do it deliberately" tier — `opus` shows up as the
nuance/troubleshooting escalation in Amy's own global CLAUDE.md ("we ask
Claude Opus or GLM-5.2 to step in when we need nuance or
troubleshooting"), while `fable` is specifically kaibo's offline
batch-deliberation persona (`casts.fable`: sonnet explorer → fable synth
on the batch lane). For a *dispatch* router — pick the tier for an
incoming request, not stage a multi-phase kaibo consult — `opus` is the
more generally legible "expensive Anthropic escalation" concept and
`fable`'s distinguishing feature (offline/batch execution) isn't
something route wording can express anyway. **This is a judgment call,
not a certainty** — if the real consumer wants `fable` as its own lane
(e.g. because "run this offline and take your time" is a routing
decision it needs to make), it should be added back as an 8th lane and
`sol` vs `fable` becomes a new ambiguous-band pair worth probing
(cross-vendor "which flagship, run offline" question).

## The `safety_specialists` lane roster

Eight lanes, matching the domains named in the brief exactly:
`k8s`, `shell`, `network`, `secrets`, `database`, `git`, `package`,
`benign`. No ambiguity here about what's "real" — these are proposed
domain specialists for a system that doesn't exist yet (per
`signoff.md`'s "web of protection" musing thread), so the roster is
designed, not discovered. The one factoring choice worth flagging: `shell`
is deliberately the catch-all for destructive-but-not-domain-specific
actions (a `find -exec rm`, a `git clean -xdf` used as an `rm -rf`
substitute) rather than adding a ninth "generic destructive" lane —
see the `guard_evasion` band, where several probes route to `shell`
specifically *because* their true blast radius is filesystem-wide even
though their surface command is `git` or `find`.

## Open design question: guard evasion needs history, not just text

`guard_evasion` band added 2026-08-05 after a real incident: a static
pre-tool hook blocked `rm -rf /home/atobey/src/kaish/deliberation-staging
&& ...` (recursive force delete, guarded string matched). The agent's
very next command — `rm -r deliberation-staging && ...` — succeeded:
same intent, `-rf` softened to `-r`, absolute path swapped for a
relative one via `cd`/cwd, so the guarded string never appeared. Amy's
comment: *"no rm -rf means no rm -r but my static python check is easy
to bypass. even your classifier missed it."*

The uncomfortable part: judged **alone**, on text only, `rm -r
deliberation-staging` is genuinely mild — a relative-path delete of
what could be a scratch directory. The strongest signal in the real
case wasn't in the command text at all; it was that the command arrived
**immediately after a guard denial for a near-identical action**. A
stateless per-command router — which is all this route/probe set can
evaluate — cannot see that. These probes test whether a variant scores
the bypass command at comparably high shell-lane severity to its
guarded pair *on textual/semantic grounds alone* (softened flag +
obscured path should still read as "recursive delete" to a router that
understands intent, not just keywords) — but even a variant that aces
every `guard_evasion` probe here is only solving the half of the
problem that's visible in a single command. The other half — a
retry-after-denial signal, sourced from denial history rather than the
current command's text — is a **separate mechanism this data doesn't
cover and this pass isn't attempting to design.** Noting it here so it
isn't lost: the router (or whatever sits in front of it) likely needs
short-term memory of recent denials, not just the current prompt.

## Running

An example harness is being built in parallel; the exact flag names may
still move. Expected shape:

```
cargo run --release --example route -- <ckpt-dir> \
    --routes-file training/router/kaibo_models.json \
    --route-variant variant_rich \
    --prompts-file training/router/kaibo_models_probes.json
```

```
cargo run --release --example route -- <ckpt-dir> \
    --routes-file training/router/safety_specialists.json \
    --route-variant variant_capability \
    --prompts-file training/router/safety_specialists_probes.json
```

Run each `(route set, variant)` pair separately and compare accuracy
per band, not just overall — a variant that wins on `easy` and loses on
`adversarial` is a worse router than one that's flatter across bands,
even at equal aggregate accuracy. For `multi`-band and `guard_evasion`
probes with `expect_multi`, also look at raw (pre-softmax) scores for
both named lanes, not just whether top-1 matches `expect` — the whole
point of that band is deciding whether softmax top-1 is even the right
metric for fan-out routing.
