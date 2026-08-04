# Fine-tuning: LFM2.5 encoder → whole-sequence classifier

This is the training side of a fixed contract shared with the Rust
inference side (`src/`). If you change what gets written to `--out`, the
Rust loader breaks — coordinate before changing key names, shapes, or
`config.json` fields.

## Export contract (fixed)

```
out/
  model.safetensors
    lfm2.embed_tokens.weight        [vocab, hidden]
    lfm2.embedding_norm.weight      [hidden]
    lfm2.layers.<N>....             (whatever the base checkpoint ships)
    classifier.weight               [num_labels, hidden]
    classifier.bias                 [num_labels]
  config.json                       base config + architectures override
                                     + id2label / label2id (id2label keys
                                     are STRINGS "0", "1", ... )
  tokenizer.json                    copied verbatim from the base dir
```

Pooling is **CLS** — the hidden state at sequence position 0 — never mean
pooling. That position is the tokenizer's BOS (`<|startoftext|>`, id 1),
which every encoding starts with regardless of padding side, so it's a
stable choice across a padded batch.

`architectures` is set to `["Lfm2BidirForSequenceClassification"]`, which
matches `EncoderArch::SequenceClassification` in `src/config.rs` — that
variant was already anticipated there ("none shipped yet") before this
export existed.

## Environment

Verified on "zorak": AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151), ROCm
7.2.4, 125 GB unified memory.

A working ROCm PyTorch venv (torch 2.13.0+rocm7.2, gfx1151 kernels
present) is assumed to already exist. Add the training deps to it:

```
uv pip install --python <venv>/bin/python transformers==4.56.2 safetensors
```

`transformers==4.56.2` is pinned because that's the version these
checkpoints were saved with — a newer transformers changing
`create_causal_mask`'s signature or `Lfm2Attention` internals would break
the checkpoint's custom `modeling_lfm2_bidirectional.py`, which monkeypatches
both.

`torch.cuda` is the ROCm device on this machine — use
`torch.device("cuda")` normally, there is no separate ROCm API surface to
learn. Confirmed on this box: `torch.cuda.is_bf16_supported()` is `True`,
so the script autocasts in **bf16** by default (fp32 measured ~11x slower
here). fp16 with `GradScaler` is the fallback for hardware without bf16.

## Two checkpoint-loading gotchas (fixture-verified 2026-08-04)

Both are handled inside `finetune_sequence_classifier.py`
(`dot_free_checkpoint_dir` and `load_backbone`) — read this section before
changing either function.

**1. The checkpoint directory name cannot contain a dot.** Every LFM2.5
checkpoint directory is named `LFM2.5-...`. `transformers`' dynamic-module
loader (for `trust_remote_code=True`) turns the directory's basename into
a Python module path under `transformers_modules.<basename>...`, and a
literal `.` in that basename produces an invalid module path. The
workaround: symlink the checkpoint to a dot-free name in a temp directory
and load from the symlink, never the original path. `dot_free_checkpoint_dir()`
does this automatically for any `--base` whose name contains `.`.

**2. `AutoModel.from_pretrained(dir, trust_remote_code=True)` silently
re-initializes every backbone weight — do not use it for this checkpoint.**
This is the gotcha that actually cost time to find, because nothing raises:

- The checkpoint's own `model.safetensors` stores keys with a `lfm2.`
  prefix (verified directly against the safetensors header) — it was saved
  as the `Lfm2BidirectionalForMaskedLM` wrapper class, whose
  `state_dict()` naturally prefixes its `self.lfm2 = Lfm2BidirectionalModel(...)`
  submodule that way.
- `config.json`'s `auto_map["AutoModel"]` points at the *bare*
  `Lfm2BidirectionalModel` class instead. That class never overrides
  `base_model_prefix`, so it inherits transformers' generic default,
  `"model"`.
- `transformers`' automatic checkpoint-prefix-stripping only fires when
  the checkpoint's saved prefix matches the *target* model's
  `base_model_prefix`. `"lfm2."` != `"model."`, so the strip never
  triggers and every one of the 148 backbone tensors comes back freshly
  (randomly) initialized.
- The only signal is a log line — `Some weights ... were newly
  initialized` — not an exception. Verified directly: loading via
  `AutoModel` and checking `embedding_norm.weight.mean()` gives ~1.0
  (fresh RMSNorm init); loading via the path below gives ~2.39 (trained).
  A script that doesn't specifically grep its own stdout for that log
  line has no way to notice it just fine-tuned a randomly initialized
  network while believing it was the pretrained checkpoint.

The fix: `Lfm2BidirectionalForMaskedLM.base_model_prefix = "lfm2"` **is**
set explicitly in the checkpoint's own custom code, so loading through
`AutoModelForMaskedLM` (which `auto_map` resolves to that wrapper class)
loads correctly — and `.lfm2` on the result is exactly the submodule the
export contract wants, under that same attribute name. So:

```python
mlm = AutoModelForMaskedLM.from_pretrained(base_dir, trust_remote_code=True)
backbone = mlm.lfm2  # correctly loaded, prefix-matched
```

**Always load via `AutoModelForMaskedLM` and take `.lfm2`. Never load the
backbone via bare `AutoModel` on this checkpoint family.**

## Usage

```
python finetune_sequence_classifier.py \
  --train path/to/train.jsonl \
  --val path/to/val.jsonl \
  --base /path/to/LFM2.5-Encoder-350M \
  --out  /path/to/output-dir \
  --epochs 3 --batch-size 16 --lr 2e-5 --max-len 256 --seed 42
```

Input JSONL format, one object per line:

```
{"text": "...", "label": "some_string_label"}
```

The label set is derived from `--train` (sorted, not hand-specified) and
recorded as `id2label`/`label2id` in the exported `config.json`. A label
appearing in `--val` but never in `--train` is a hard error, not a warning
— training data. Never guess a class the model never saw.

The best-validation-accuracy checkpoint is exported (overwriting `--out`)
every time validation improves, so a crash after epoch N still leaves the
best checkpoint through epoch N-1 (or N) on disk — not full step-level
resume, but never leaves `--out` empty or worse-than-best.

Before writing, the script asserts:
- `classifier.weight.shape[0] == len(id2label)`,
- the exported `state_dict()`'s non-backbone keys are *exactly*
  `{"classifier.weight", "classifier.bias"}` (everything else must be
  prefixed `lfm2.`).

Both are `assert`s that abort the export rather than write a checkpoint
that silently disagrees with its own label map or the Rust loader's
contract.

### A note on determinism

Seeds are set for `random`, `numpy`, and `torch` (CPU + all CUDA/ROCm
devices), and `DataLoader` shuffling uses a seeded generator. This is
**not** wrapped in `torch.use_deterministic_algorithms(True)` — several
ops used here (embedding backward, grouped conv backward) don't have
deterministic ROCm kernels yet, and forcing it would trade a training
script that runs for one that raises. Runs are seeded and reproducible in
practice, not bit-identical by contract.

The base repo's own README documents that this architecture's short conv
is intentionally unmasked in the reference implementation (pad states
bleed one position into their real neighbour). That's a property of the
pretrained weights' own training regime, not something this script
introduces or should "fix" by masking differently than the reference does.

## Smoke test

`make_smoke_data.py` generates a synthetic 2-class dataset — shell command
strings labeled `safe`/`dangerous` — with no real dataset required:

```
python make_smoke_data.py --out-dir training/data --seed 42
```

**Every "dangerous" string is DATA.** They are Python string literals
written into a JSONL file for a text classifier to read; nothing in
either script ever executes any of them as a shell command. They're
deliberately generic (`example.com`, no real infra, no real hostnames) so
this is unambiguously a synthetic smoke fixture, not a security dataset.

Generated 121 examples (69 safe / 52 dangerous), split 97 train / 24 val.

### Smoke run results (2026-08-04, this machine)

```
python finetune_sequence_classifier.py \
  --train training/data/train.jsonl --val training/data/val.jsonl \
  --base .models/LFM2.5-Encoder-350M --out <scratch-dir> \
  --epochs 1 --batch-size 8 --lr 2e-5 --max-len 256 --seed 42
```

```
device=cuda  autocast_dtype=torch.bfloat16
labels (2): ['dangerous', 'safe']
train examples: 97  val examples: 24
epoch 1/1  train_loss=0.6847  val_acc=0.6250
               dangerous  precision=0.529  recall=0.900  support=10
                    safe  precision=0.857  recall=0.429  support=14
  -> new best val_acc=0.6250; exporting to <scratch-dir>
done. best val_acc=0.6250. export at <scratch-dir>
```

Wall time ~11-12.5s end to end (venv activation, checkpoint load,
tokenization, 1 epoch over 13 batches, validation, and writing a 1.35 GiB
`model.safetensors`). Peak GPU memory 6.69 GiB allocated / 6.84 GiB
reserved for `--batch-size 8`. 62.5% val accuracy on 1 epoch over 97
synthetic examples is exactly what you'd expect from this being a
plumbing test, not a learning test — this is not a claim the model
learned "dangerous shell command" as a concept from 97 examples.

Export verified with a short snippet reading the safetensors header
directly (no torch/candle needed to check shapes):

```
total keys: 150
non-backbone keys: ['classifier.bias', 'classifier.weight']
  classifier.bias   [2]       F32
  classifier.weight [2, 1024] F32
lfm2.embed_tokens.weight   shape [65536, 1024]
lfm2.embedding_norm.weight shape [1024]
```

`config.json`: `architectures: ["Lfm2BidirForSequenceClassification"]`,
`id2label: {"0": "dangerous", "1": "safe"}`,
`label2id: {"dangerous": 0, "safe": 1}`. `tokenizer.json` byte-identical
(md5) to the base checkpoint's.

### Known cosmetic leftover

The exported `config.json` still carries the base checkpoint's `auto_map`
(pointing at `modeling_lfm2_bidirectional.Lfm2BidirectionalModel` /
`...ForMaskedLM`) because the contract says "copy the base checkpoint's
config" and only specifies `architectures`/`id2label`/`label2id` as
deltas. It's inert: the Rust loader's config parsing ignores unknown
fields, and the exported directory doesn't carry the custom `.py` file
anyway, so a Python `trust_remote_code=True` load against the exported dir
would fail loudly (`ImportError`) rather than silently load the wrong
class. Flagging it here rather than silently stripping it, since the
export contract didn't ask for that and it's not this script's call to
widen scope unasked.
