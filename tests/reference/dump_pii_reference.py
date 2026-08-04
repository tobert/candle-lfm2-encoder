#!/usr/bin/env python3
"""Dump PII token-classification references using the checkpoint's own code.

Ground truth is `pii_hybrid_decode.model_spans` — the PURE MODEL decode,
not the hybrid. The shipped product decode wraps the model in a regex tier
that OWNS every credential type outright (`if t in _AUTH_TYPES: continue`),
so testing the Rust head against hybrid output would be testing regexes,
not the model.

Texts are drawn from tests/data/pii_detection_eval.json so the fixture
exercises cases we actually care about.

Usage:
    refenv/bin/python tests/reference/dump_pii_reference.py \\
        .models/LFM2.5-Encoder-350M-PII-Detector \\
        tests/fixtures/pii_reference.safetensors \\
        tests/fixtures/pii_reference_spans.json
"""

import json
import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModelForTokenClassification, AutoTokenizer

# Case ids chosen to span the categories: leaked credentials of each kind,
# personal data, and clean text that must produce NO spans.
CASE_IDS = None  # filled from the eval file; see pick_cases()
N_CASES = 10


def pick_cases(eval_path):
    data = json.load(open(eval_path))
    by_cat = {}
    for c in data["cases"]:
        by_cat.setdefault(c["category"], []).append(c)
    picked = []
    # Deterministic: first few of each category, credentials first.
    for cat in ("leaked-credential", "personal-data", "clean", "tricky-negative"):
        picked.extend(by_cat.get(cat, [])[: N_CASES // 4 + 1])
    return picked[:N_CASES]


def main() -> None:
    model_dir, out_tensors, out_spans = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, model_dir)
    from pii_hybrid_decode import model_spans  # noqa: E402

    # This checkpoint's tokenizer_config names `TokenizersBackend`, a class
    # newer than the transformers we pin to the family's saved version. Fall
    # back to building the fast tokenizer straight from tokenizer.json —
    # which is the very file the Rust side loads, so this makes the
    # comparison MORE direct rather than less.
    try:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except ValueError as e:
        print(f"AutoTokenizer unavailable ({e}); loading tokenizer.json directly")
        from transformers import PreTrainedTokenizerFast

        # This checkpoint ships no special_tokens_map.json; the specials are
        # baked into tokenizer.json's post-processor, and model_spans()
        # encodes one unpadded text at a time, so none of the pad/mask
        # plumbing is exercised anyway.
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
        tok = PreTrainedTokenizerFast(
            tokenizer_file=f"{model_dir}/tokenizer.json", **kwargs
        )
    model = AutoModelForTokenClassification.from_pretrained(
        model_dir, trust_remote_code=True, dtype=torch.float32
    ).eval()

    print(f"num_labels: {model.config.num_labels}  (label_schema.json says 109 — it is STALE)")

    cases = pick_cases("tests/data/pii_detection_eval.json")
    tensors = {}
    spans_out = []

    for n, case in enumerate(cases):
        text = case["text"]
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=2048)
        off = enc.pop("offset_mapping")[0]
        with torch.no_grad():
            logits = model(**enc).logits[0]

        tensors[f"case.{n}.input_ids"] = enc["input_ids"][0].to(torch.int64).contiguous().clone()
        tensors[f"case.{n}.offsets"] = off.to(torch.int64).contiguous().clone()
        tensors[f"case.{n}.logits"] = logits.to(torch.float32).contiguous().clone()
        tensors[f"case.{n}.argmax"] = logits.argmax(-1).to(torch.int64).contiguous().clone()

        ms = model_spans(text, tok, model)
        spans_out.append({
            "n": n,
            "id": case["id"],
            "category": case["category"],
            "text": text,
            "model_spans": ms,
        })
        print(f"case.{n} [{case['category']:<18}] {len(ms)} span(s)  {case['id']}")
        for s in ms:
            print(f"    {s['type']:<34} {text[s['start']:s['end']]!r}")

    save_file(tensors, out_tensors)
    json.dump({"cases": spans_out}, open(out_spans, "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {out_tensors} and {out_spans}")


if __name__ == "__main__":
    main()
