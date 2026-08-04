#!/usr/bin/env python3
"""Dump end-to-end text -> embedding references for LFM2.5-Embedding-350M.

Where `dump_trunk_reference.py` pins the *model* using raw token ids, this
pins the whole pipeline: tokenizer -> trunk -> CLS pool. It exists to catch
the failures that live between those stages, which are the ones that fail
silently:

  * the asymmetric prompt prefixes ("query: " / "document: ") from
    config_sentence_transformers.json — using the wrong one degrades
    retrieval without erroring
  * add_bos_token=true, so CLS pooling reads the <|startoftext|> position
  * truncation at max_seq_length=512 (sentence_bert_config.json)

Pooling is CLS per 1_Pooling/config.json. There is NO Normalize module, so
raw pooled vectors are dumped un-normalized; similarity_fn_name is cosine,
which normalizes at comparison time.

Usage:
    refenv/bin/python tests/reference/dump_embedding_reference.py \\
        .models/LFM2.5-Embedding-350M \\
        tests/fixtures/embedding_reference.safetensors
"""

import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

# Prefixes are LiquidAI's, from config_sentence_transformers.json.
QUERY_PREFIX = "query: "
DOCUMENT_PREFIX = "document: "

# Deliberately mixed: ASCII, punctuation/casing, non-Latin script (Amy's
# 日本語 practice earns its keep as a tokenizer test), and an emoji, so a
# byte-level/BPE mismatch on the Rust side shows up as an id mismatch
# rather than a vague embedding drift.
TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "Rust's borrow checker: not a suggestion.",
    "日本語（にほんご）を勉強中です。",
    "vector search 🔍 with a 350M encoder",
]


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    tensors = {}
    print(f"{'case':<28} {'n_tok':>5}  text")
    for i, text in enumerate(TEXTS):
        for kind, prefix in (("query", QUERY_PREFIX), ("document", DOCUMENT_PREFIX)):
            enc = tok(prefix + text, return_tensors="pt", truncation=True, max_length=512)
            ids = enc["input_ids"]
            with torch.no_grad():
                hidden = model(input_ids=ids).last_hidden_state
            pooled = hidden[:, 0, :]  # CLS

            key = f"{kind}.{i}"
            tensors[f"{key}.input_ids"] = ids.to(torch.int64).clone()
            tensors[f"{key}.pooled"] = pooled.clone().contiguous()
            print(f"{key:<28} {ids.shape[1]:>5}  {prefix + text}")

    # The asymmetry is the point: same text under the two prefixes must NOT
    # produce the same vector. Pinned so a Rust side that drops the prefix
    # fails instead of quietly returning decent-looking embeddings.
    q = tensors["query.0.pooled"]
    d = tensors["document.0.pooled"]
    cos = torch.nn.functional.cosine_similarity(q, d).item()
    print(f"\ncos(query.0, document.0) = {cos:.6f}  (asymmetric: must not be 1.0)")

    save_file(tensors, out_path)
    print(f"wrote {out_path}  ({len(tensors)} tensors)")


if __name__ == "__main__":
    main()
