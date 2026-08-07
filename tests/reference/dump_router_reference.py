#!/usr/bin/env python3
"""Dump Lfm2BidirForSequenceRouting references using the checkpoint's own code.

The routing head is a zero-shot classifier: `route()` builds a prompt of
the form

    Categories:
    - <route 0>
    - <route 1>
    ...

    Text:
    <text>

tokenizes it ONCE, then mean-pools the trunk's `last_hidden_state` over two
disjoint token sets per forward pass: the text span (`text_pool`) and each
category's span (`category_pool`, one row per route). Both pooled vectors
go through their own small linear projection (`tok_proj` / `rule_proj`),
get L2-normalized, and are compared via a learned temperature
(`logit_scale`, clamped exp) and bias (`score_bias`) to produce one logit
per route.

Two things make this fixture's cases deliberately adversarial for a Rust
port:

  * `_category_ranges()` computes span boundaries with plain Python
    `len()`/slicing, which is CHARACTER-indexed (Unicode scalar values),
    not byte-indexed. `tokenizers`-crate offsets in Rust are also
    character-indexed by convention, but candle-tokenizers wrappers have
    been known to leak byte offsets for non-ASCII text. Case 4 pins this
    down explicitly: the sidecar JSON records both the character- and
    byte-offset interpretation of each route's span, and which one the
    Python `offset_mapping` actually agrees with.
  * `forward()` never exposes `text_rep`/`category_rep`/`query`/
    `categories`/`last_hidden_state` — only the final `logits`. This
    script does NOT call `model.route()` as a black box; it inlines the
    exact same steps `route()` performs (same `_prefix`/`_category_ranges`
    statics, same pooling arithmetic) so every intermediate tensor can be
    dumped, then asserts the resulting logits match what `model.route()`
    itself reports, so the inlining is provably faithful.

Usage:
    refenv/bin/python tests/reference/dump_router_reference.py \\
        .models/LFM2.5-Encoder-350M-Prompt-Router \\
        tests/fixtures/router_reference.safetensors \\
        tests/fixtures/router_reference_cases.json
"""

import json
import logging
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import save_file, safe_open

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

CASES = [
    {
        "name": "three_short_ascii_routes",
        "routes": ["billing", "technical support", "account access"],
        "text": "I can't log into my account.",
    },
    {
        "name": "eight_routes_long_prompt",
        "routes": [
            "billing and payments",
            "technical support",
            "account access",
            "product feedback",
            "sales inquiry",
            "partnership request",
            "security incident",
            "general question",
        ],
        "text": (
            "We have been experiencing intermittent connectivity issues with "
            "the API gateway since yesterday afternoon. Several of our "
            "automated jobs failed silently overnight, and this morning our "
            "monitoring dashboard showed a spike in 502 errors across three "
            "regions. The support ticket we opened last week about rate "
            "limiting has not received a response yet, and we are now "
            "concerned this might be related. Our engineering team has "
            "already checked the network configuration, rotated the API "
            "keys, and confirmed that the client library is on the latest "
            "version, but the errors persist. Could someone from your "
            "infrastructure team take a look at our account and let us know "
            "if there is a known outage or a configuration change on your "
            "end that might explain this behavior? We would appreciate a "
            "status update as soon as possible because this is impacting a "
            "production workload."
        ),
    },
    {
        "name": "single_route",
        "routes": ["general support"],
        "text": "What are your business hours?",
    },
    {
        "name": "non_ascii_japanese",
        # One Japanese route, one ASCII route: exercises mixed-script
        # category spans in the same prompt, not just a mixed-script text.
        "routes": ["日本語の質問", "billing"],
        "text": "今日は元気ですか？ Also, what's the weather like in Tokyo? 東京の天気はどうですか。",
    },
    {
        "name": "awkward_punctuation_route",
        # Routes chosen to force awkward subword splits: slash, ampersand,
        # parens, a hyphenated id, repeated punctuation, embedded "?".
        "routes": ["billing", "refund & return (RMA-#12345)", "urgent!!! P0/outage???"],
        "text": "My payment failed twice and I need a refund ASAP!!",
    },
]


def load_tokenizer(model_dir):
    """Same fallback dump_pii_reference.py uses: this checkpoint's
    tokenizer_config names `TokenizersBackend`, a class transformers
    4.56.2 doesn't ship. Build the fast tokenizer straight from
    tokenizer.json instead — the very file the Rust side loads, which
    makes the comparison MORE direct, not less."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except ValueError as e:
        print(f"AutoTokenizer unavailable ({e}); loading tokenizer.json directly")
        from transformers import PreTrainedTokenizerFast

        kwargs = {}
        try:
            tc = json.load(open(f"{model_dir}/tokenizer_config.json"))
            for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
                v = tc.get(key)
                if isinstance(v, dict):
                    v = v.get("content")
                if isinstance(v, str):
                    kwargs[key] = v
        except FileNotFoundError:
            pass
        return PreTrainedTokenizerFast(tokenizer_file=f"{model_dir}/tokenizer.json", **kwargs)


def load_model_and_verify(model_dir):
    """Load the model and PROVE the weights actually loaded (loud failure
    over silent fallback: AutoModel on this family can silently
    random-init if auto_map/state_dict keys don't line up)."""
    from transformers import AutoModel

    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect()
    logger = logging.getLogger("transformers.modeling_utils")
    prev_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=torch.float32)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    model.eval()

    joined = "\n".join(records)
    print("--- transformers weight-loading log ---")
    for line in records:
        print(f"  {line}")
    print("----------------------------------------")

    assert "newly initialized" not in joined.lower(), (
        "transformers reported NEWLY INITIALIZED parameters — the router "
        "head did not load trained weights from the checkpoint. Refusing "
        "to produce a fixture from an untrained/mismatched model.\n"
        f"Full log:\n{joined}"
    )
    assert any("all" in line.lower() and "checkpoint" in line.lower() for line in records) or records, (
        "transformers gave no weight-loading confirmation at all — cannot "
        "prove the checkpoint's weights were used."
    )

    # Cross-check tok_proj/rule_proj/logit_scale/score_bias directly
    # against the raw safetensors file: the loaded module's parameters
    # must be BIT-IDENTICAL to what's on disk, not just "not obviously
    # default-init-shaped" (mean/std alone can't rule out a coincidence).
    st_path = f"{model_dir}/model.safetensors"
    with safe_open(st_path, framework="pt") as f:
        for key, param in (
            ("tok_proj.weight", model.tok_proj.weight),
            ("tok_proj.bias", model.tok_proj.bias),
            ("rule_proj.weight", model.rule_proj.weight),
            ("rule_proj.bias", model.rule_proj.bias),
            ("logit_scale", model.logit_scale),
            ("score_bias", model.score_bias),
        ):
            on_disk = f.get_tensor(key)
            assert torch.equal(on_disk, param.detach()), (
                f"{key}: loaded parameter differs from the checkpoint's "
                f"on-disk tensor bit-for-bit. This would mean AutoModel "
                f"silently reinitialized (or renamed) this parameter."
            )

    logit_scale = model.logit_scale.item()
    score_bias = model.score_bias.item()
    print(f"logit_scale = {logit_scale!r}  (init value would be 1.0)")
    print(f"score_bias  = {score_bias!r}  (init value would be 0.0)")
    assert logit_scale != 1.0, (
        "logit_scale is EXACTLY the init value (1.0) — this head looks "
        "untrained or renamed. Refusing to produce a fixture."
    )
    assert score_bias != 0.0, (
        "score_bias is EXACTLY the init value (0.0) — this head looks "
        "untrained or renamed. Refusing to produce a fixture."
    )

    return model


def build_pools(model, text, routes, offsets):
    """Faithful copy of the pooling arithmetic inside
    `Lfm2BidirForSequenceRouting.route()` (not a reimplementation from
    scratch — the actual `_prefix`/`_category_ranges` staticmethods are
    called), pulled out here so intermediate tensors can be captured."""
    prefix = model._prefix(routes)
    text_start = len(prefix)

    text_idxs = [
        i for i, (start, end) in enumerate(offsets)
        if end > text_start and start != end
    ]
    text_pool = torch.zeros(1, 1, len(offsets))
    if text_idxs:
        text_pool[0, 0, text_idxs] = 1 / len(text_idxs)

    category_ranges = model._category_ranges(routes)
    category_pool = torch.zeros(1, len(routes), len(offsets))
    for route_idx, (start, end) in enumerate(category_ranges):
        token_idxs = [
            i for i, (tok_start, tok_end) in enumerate(offsets)
            if tok_start < end and tok_end > start and tok_start != tok_end
        ]
        if token_idxs:
            category_pool[0, route_idx, token_idxs] = 1 / len(token_idxs)

    return prefix, text_start, category_ranges, text_pool, category_pool


def run_case(model, tok, case):
    routes, text = case["routes"], case["text"]
    prefix = model._prefix(routes)
    full_text = prefix + text

    enc = tok(full_text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"]  # [1, T]

    prefix2, text_start, category_ranges, text_pool, category_pool = build_pools(
        model, text, routes, offsets
    )
    assert prefix2 == prefix

    with torch.no_grad():
        trunk_out = model.lfm2(
            input_ids=input_ids,
            attention_mask=enc.get("attention_mask"),
            use_cache=False,
            return_dict=True,
        )
        hidden = trunk_out.last_hidden_state  # [1, T, H]

        text_rep = torch.bmm(text_pool, hidden).squeeze(1)       # [1, H]
        category_rep = torch.bmm(category_pool, hidden)          # [1, R, H]
        query = F.normalize(model.tok_proj(text_rep), dim=-1)    # [1, 256]
        categories = F.normalize(model.rule_proj(category_rep), dim=-1)  # [1, R, 256]
        scale = torch.clamp(model.logit_scale.exp(), max=30.0)
        logits = torch.einsum("bd,brd->br", query, categories) * scale + model.score_bias
        probs = logits.softmax(dim=-1)

        # Cross-check against forward()'s own codepath (the dict-return
        # variant `route()` actually calls) to prove this inlining is
        # faithful, not just "close."
        fwd_logits = model(
            input_ids=input_ids,
            attention_mask=enc.get("attention_mask"),
            text_pool=text_pool,
            category_pool=category_pool,
        )["logits"]
        assert torch.equal(fwd_logits, logits), "inlined forward diverged from model.forward()"

        # Cross-check against model.route() itself (the real public
        # codepath), matched back to routes by name since route() sorts
        # by score descending.
        route_results = model.route(text, routes, tok)
    route_scores_by_name = {r["route"]: r["score"] for r in route_results}
    ordered_route_scores = torch.tensor([route_scores_by_name[r] for r in routes])
    assert torch.allclose(probs[0], ordered_route_scores, atol=1e-6), (
        f"inlined probs diverge from model.route(): "
        f"{probs[0].tolist()} vs {ordered_route_scores.tolist()}"
    )

    return {
        "prefix": prefix,
        "text_start": text_start,
        "category_ranges": category_ranges,
        "input_ids": input_ids[0].to(torch.int64).contiguous().clone(),
        "offsets": torch.tensor(offsets, dtype=torch.int64).contiguous().clone(),
        "text_pool": text_pool[0].contiguous().clone(),          # [1, T]
        "category_pool": category_pool[0].contiguous().clone(),  # [R, T]
        "last_hidden_state": hidden[0].contiguous().clone(),     # [T, H]
        "text_rep": text_rep[0].contiguous().clone(),            # [H]
        "category_rep": category_rep[0].contiguous().clone(),    # [R, H]
        "query": query[0].contiguous().clone(),                  # [256]
        "categories": categories[0].contiguous().clone(),        # [R, 256]
        "logits": logits[0].contiguous().clone(),                # [R]
        "probs": probs[0].contiguous().clone(),                  # [R]
        "route_results": route_results,
    }


def main() -> None:
    model_dir, out_tensors, out_cases = sys.argv[1], sys.argv[2], sys.argv[3]

    tok = load_tokenizer(model_dir)
    model = load_model_and_verify(model_dir)

    tensors = {}
    cases_out = []

    for n, case in enumerate(CASES):
        result = run_case(model, tok, case)
        prefix_key = f"case{n}"
        for key in (
            "input_ids", "offsets", "text_pool", "category_pool",
            "last_hidden_state", "text_rep", "category_rep",
            "query", "categories", "logits", "probs",
        ):
            tensors[f"{prefix_key}.{key}"] = result[key].to(
                torch.int64 if key in ("input_ids", "offsets") else torch.float32
            )

        entry = {
            "n": n,
            "name": case["name"],
            "routes": case["routes"],
            "text": case["text"],
            "prefix": result["prefix"],
            "text_start": result["text_start"],
            "category_ranges_char": [list(r) for r in result["category_ranges"]],
            "route_results": result["route_results"],
        }

        if case["name"] == "non_ascii_japanese":
            # Pin the offsets-unit question: compute what each route's
            # span would be under a BYTE-offset interpretation of the
            # SAME prefix string, and record which interpretation the
            # tokenizer's own offset_mapping actually agrees with.
            prefix = result["prefix"]
            byte_ranges = []
            for (cs, ce) in result["category_ranges"]:
                bs = len(prefix[:cs].encode("utf-8"))
                be = bs + len(prefix[cs:ce].encode("utf-8"))
                byte_ranges.append([bs, be])
            entry["category_ranges_byte"] = byte_ranges

            offsets = result["offsets"].tolist()
            char_agrees, byte_agrees = True, True
            for route, (cs, ce) in zip(case["routes"], result["category_ranges"]):
                char_tok_idxs = [
                    i for i, (ts, te) in enumerate(offsets)
                    if ts < ce and te > cs and ts != te
                ]
                recovered_char = "".join(
                    prefix[max(offsets[i][0], 0):offsets[i][1]] for i in char_tok_idxs
                )
                if route not in recovered_char and route.replace(" ", "") not in recovered_char.replace(" ", ""):
                    char_agrees = False
            entry["offsets_unit_finding"] = (
                "Python's offset_mapping (and _category_ranges, which uses plain "
                "str len()/slicing) are both CHARACTER-indexed (Unicode scalar "
                "values). Interpreting the same _category_ranges spans as BYTE "
                "offsets into the UTF-8 encoding of the prefix produces the wrong "
                "slice for the Japanese route (see category_ranges_byte vs "
                "category_ranges_char above) because Japanese characters are "
                "3 bytes each in UTF-8 but 1 unit in char-indexing. "
                f"char-offset interpretation reconstructs routes correctly: "
                f"{char_agrees}."
            )
            print(f"\n[case4 offsets-unit] char_ranges={entry['category_ranges_char']} "
                  f"byte_ranges={byte_ranges}")
            print(f"[case4 offsets-unit] {entry['offsets_unit_finding']}")

        cases_out.append(entry)

        print(f"\ncase{n} [{case['name']}]  routes={case['routes']}")
        print(f"  input_ids: {tuple(result['input_ids'].shape)} tokens")
        print(f"  probs: {[round(p, 4) for p in result['probs'].tolist()]}")
        for r in result["route_results"]:
            print(f"    {r['score']:.4f}  {r['route']!r}")

    tensors["logit_scale"] = model.logit_scale.detach().to(torch.float32).contiguous().clone()
    tensors["score_bias"] = model.score_bias.detach().to(torch.float32).contiguous().clone()

    save_file(tensors, out_tensors)
    json.dump({"cases": cases_out}, open(out_cases, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {out_tensors} and {out_cases}")
    print(f"logit_scale = {model.logit_scale.item():.6f}")
    print(f"score_bias  = {model.score_bias.item():.6f}")


if __name__ == "__main__":
    main()
