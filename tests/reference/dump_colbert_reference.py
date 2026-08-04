#!/usr/bin/env python3
"""Dump ColBERT multi-vector references using PyLate itself.

PyLate is the ground truth here on purpose. ColBERT's pipeline has several
conventions that are easy to get subtly wrong and impossible to verify by
reading a config: where the [Q]/[D] marker lands relative to BOS, whether
query expansion pads with MASK, whether skiplist punctuation vectors are
dropped or zeroed, and whether vectors are L2-normalized. Reimplementing
those in Python from my own reading and calling it a reference would just
encode my assumptions twice.

So this drives `pylate.models.ColBERT` and also dumps the tokenized ids,
which is what lets the Rust side be checked stage by stage.

Usage:
    refenv/bin/python tests/reference/dump_colbert_reference.py \\
        .models/LFM2.5-ColBERT-350M tests/fixtures/colbert_reference.safetensors
"""

import sys

import torch
from safetensors.torch import save_file
from pylate import models

QUERIES = [
    "how do I stop two threads corrupting shared state",
    "日本語のテキスト検索",
]
DOCUMENTS = [
    "Arc<Mutex<T>> is the idiomatic way to share mutable state across threads in Rust.",
    "Preheat the oven to 200C, then butter a cake tin.",
    "Punctuation!! Should, be; skipped: right? (yes)",
]


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]

    model = models.ColBERT(model_name_or_path=model_dir, trust_remote_code=True)
    # Per the checkpoint's README.
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.eval()

    tensors = {}
    meta = {}

    for kind, texts, is_query in (
        ("query", QUERIES, True),
        ("document", DOCUMENTS, False),
    ):
        embs = model.encode(
            texts,
            batch_size=1,
            is_query=is_query,
            show_progress_bar=False,
            convert_to_numpy=False,
        )
        for i, emb in enumerate(embs):
            e = emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
            e = e.detach().to(torch.float32).contiguous().clone()
            tensors[f"{kind}.{i}.vectors"] = e
            meta[f"{kind}.{i}"] = tuple(e.shape)

            # Norms tell us immediately whether PyLate normalized per token.
            norms = e.norm(dim=-1)
            print(
                f"{kind}.{i}: {tuple(e.shape)}  "
                f"norm min={norms.min():.4f} max={norms.max():.4f}  {texts[i][:44]!r}"
            )

    # Tokenization, so the Rust side can be verified stage by stage rather
    # than only at the end.
    # ONE TEXT AT A TIME. Tokenizing the list together pads every sequence
    # to the batch maximum, which would bake a batching artifact into the
    # fixture and make correct single-document tokenization look wrong.
    # (It did exactly that on the first pass.)
    for kind, texts, is_query in (
        ("query", QUERIES, True),
        ("document", DOCUMENTS, False),
    ):
        for i, text in enumerate(texts):
            toks = model.tokenize([text], is_query=is_query)
            ids = toks["input_ids"][0]
            mask = toks["attention_mask"][0]
            tensors[f"{kind}.{i}.input_ids"] = ids.to(torch.int64).contiguous().clone()
            tensors[f"{kind}.{i}.attention_mask"] = mask.to(torch.int64).contiguous().clone()
            if i == 0:
                print(f"\n{kind} ids[0]: {ids.tolist()}")
                print(f"{kind} mask[0]: {mask.tolist()}")

    # MaxSim scores: sum over query tokens of the best matching doc token.
    q = tensors["query.0.vectors"]
    print("\nMaxSim query.0 vs each document:")
    for i in range(len(DOCUMENTS)):
        d = tensors[f"document.{i}.vectors"]
        score = (q @ d.T).max(dim=-1).values.sum()
        tensors[f"maxsim.0.{i}"] = score.reshape(1).contiguous().clone()
        print(f"  document.{i}: {score:.6f}")

    save_file(tensors, out_path)
    print(f"\nwrote {out_path} ({len(tensors)} tensors)")
    print("special tokens:", {
        "mask": model.tokenizer.mask_token,
        "mask_id": model.tokenizer.mask_token_id,
        "pad": model.tokenizer.pad_token,
        "pad_id": model.tokenizer.pad_token_id,
        "bos": model.tokenizer.bos_token,
    })


if __name__ == "__main__":
    main()
