#!/usr/bin/env python3
"""Dump reference trunk activations from the checkpoint's OWN modeling code.

This is the ground truth our Rust trunk is tested against. It runs
LiquidAI's `modeling_lfm2_bidirectional.py` (via trust_remote_code) in
float32/eager and writes hidden states to a safetensors fixture that
`tests/trunk_parity.rs` loads.

Deliberately uses raw token ids, not text: this pins the *model*, leaving
tokenizer differences to a separate test. Any mismatch is then unambiguously
ours.

Usage:
    uv venv --python 3.12 refenv
    uv pip install --python refenv/bin/python torch --index-url \\
        https://download.pytorch.org/whl/cpu
    uv pip install --python refenv/bin/python transformers==4.56.2 safetensors
    refenv/bin/python tests/reference/dump_trunk_reference.py \\
        .models/LFM2.5-Embedding-350M tests/fixtures/trunk_reference.safetensors

transformers is pinned to 4.56.2 — the `transformers_version` the checkpoint
was saved with.
"""

import sys

import torch
from safetensors.torch import save_file
from transformers import AutoModel

# Arbitrary but fixed ids, well inside the 65536 vocab. Includes id 0 (the
# pad id) as a *content* token in the unpadded case, so a mask bug that
# keys off the id rather than the mask shows up.
SEQ = [1, 4919, 271, 8123, 40, 0, 65535, 12, 99, 3, 7715, 60123, 2, 7]
# Two rows of different real lengths, right-padded. Exercises the additive
# pad mask AND the deliberately unmasked short-conv (a padded row's real
# tokens do NOT reproduce the unpadded run bit-for-bit — that is the
# checkpoint's trained behaviour, and the Rust side must match the same way).
BATCH = [
    [1, 4919, 271, 8123, 40, 6, 65535, 12],
    [1, 777, 31, 4, 0, 0, 0, 0],
]
BATCH_MASK = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 0],
]


def main() -> None:
    model_dir, out_path = sys.argv[1], sys.argv[2]

    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    tensors = {}

    ids = torch.tensor([SEQ], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids).last_hidden_state
    tensors["single.input_ids"] = ids.to(torch.int64)
    tensors["single.hidden_states"] = out.contiguous().clone()
    # CLS pooling, per 1_Pooling/config.json (pooling_mode_cls_token: true).
    tensors["single.pooled_cls"] = out[:, 0, :].clone().contiguous()

    ids_b = torch.tensor(BATCH, dtype=torch.long)
    mask_b = torch.tensor(BATCH_MASK, dtype=torch.long)
    with torch.no_grad():
        out_b = model(input_ids=ids_b, attention_mask=mask_b).last_hidden_state
    tensors["batch.input_ids"] = ids_b.to(torch.int64)
    tensors["batch.attention_mask"] = mask_b.to(torch.int64)
    tensors["batch.hidden_states"] = out_b.contiguous().clone()

    save_file(tensors, out_path)

    print(f"wrote {out_path}")
    for k, v in tensors.items():
        print(f"  {k:28s} {tuple(v.shape)} {v.dtype}")
    print(f"\nsingle hidden[0,0,:6] = {out[0, 0, :6].tolist()}")
    print(f"single hidden mean/std = {out.mean():.6f} / {out.std():.6f}")


if __name__ == "__main__":
    main()
