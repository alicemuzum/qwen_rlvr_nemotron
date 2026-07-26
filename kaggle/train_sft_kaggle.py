"""Stage-2 SFT on the synthetic step-tag corpus, for a Kaggle/Colab GPU.

Teaches Qwen the *output format* -- `<think>`-wrapped `<step type="...">`
traces ending in a bare `\\boxed{}` line -- from `synth_sft.jsonl` (14000
verified gold traces, 2000 per category). It is not meant to teach the model
to solve the tasks well; that's stage 3 (GRPO, see
`kaggle/train_grpo_cipher_kaggle.py`).

Loss is computed on the trace tokens only: the prompt is masked with -100, so
the model is never trained to reproduce the puzzle statement.

--- Design notes (the non-obvious choices) ---

* **Plain `transformers.Trainer`, not TRL's `SFTTrainer`.** Rows are
  pre-tokenized here and fed as `input_ids`/`labels`/`attention_mask`. TRL's
  SFT dataset API (prompt-completion columns, `assistant_only_loss`, packing)
  has moved across the versions Kaggle preinstalls; hand-rolled masking is a
  dozen lines and pins the behaviour exactly.

* **Over-length rows are dropped, never truncated.** Truncation cuts the
  trace's *tail*, which is precisely the `<step type="conclusion">` +
  `\\boxed{}` this stage exists to teach -- a right-cut silently trains the
  model to never commit to an answer. `MAX_SEQ_LEN` defaults to 8704, above
  the corpus's own 8192-token generation cap, so the chat-template wrapper
  and system prompt don't push the longest cipher/gravity/bit_manipulation
  rows over. Whatever gets dropped is reported per category before training.

* **Prompt format is a knob (`Cfg.prompt_style`), not a silent default.**
  `"chat"` renders the tokenizer's chat template with a short generic system
  prompt; `"raw"` feeds the bare `train.csv`-style prompt text with no
  template. This is a real cross-stage decision: the corpus rows are raw
  text, while `train_grpo_cipher_kaggle.py` prompts with chat messages and a
  cipher-specific one-shot system demo. **If you intend to chain SFT ->
  GRPO, align the two prompt formats first** -- this script deliberately
  does not edit the GRPO script's `SYSTEM_PROMPT` on your behalf.

* **Val split caveat.** The split is a stratified random split of a purely
  synthetic corpus, so train and val share underlying generated rules
  (ciphers, conversion factors, symbol mappings). Val loss is a *format*
  learning proxy, not a task-generalization measure. For the latter, hold out
  real `train.csv` rows instead -- `reasoners_train_csv_eval.csv` makes
  assembling that slice cheap.

--- Local dev/testing ---
From the project root: `uv sync`, then
    uv run python kaggle/train_sft_kaggle.py --dry-run     # data pipeline only, no GPU
    uv run python kaggle/train_sft_kaggle.py               # actually train

--- Kaggle setup ---
1. Create a Kaggle Dataset containing just `synth_sft.jsonl` (~93 MB). No
   code from this repo is needed -- this script is self-contained.
2. Attach it; set INPUT_DIR below to the mount path
   (typically /kaggle/input/<dataset-slug>).
3. Turn on a GPU accelerator. T4 x1 is enough for the 0.6B default with LoRA
   + gradient checkpointing at ~8k sequence length. The OOM risk here is the
   loss, not the weights (see `_make_chunked_loss_trainer`), which is why the
   head is applied in chunks and `per_device_train_batch_size` defaults to 1
   with 16 accumulation steps. Verified end-to-end on a 5.7 GB GPU including
   7k-token rows, so a 16 GB T4 has headroom to raise the batch size.
   bf16 is used when the GPU supports it (A100/L4), fp16 otherwise (T4/P100).
   The base model is loaded in 4-bit (QLoRA: nf4 + double quant) by default
   (`Cfg.load_in_4bit`, needs a CUDA GPU -- silently falls back to full
   precision otherwise); pass `--full-precision` to disable it.
4. Run the install cell, then this script:

    !pip install -q -U transformers peft accelerate datasets bitsandbytes

5. Session limits: Kaggle caps sessions around 9-12h. The adapter is
   checkpointed every SAVE_STEPS steps to OUTPUT_DIR and the script
   auto-resumes from the newest checkpoint found there (same pattern as the
   GRPO script). To survive a session ending, save OUTPUT_DIR as a Kaggle
   Dataset output and copy it back before the next run.

Usage (command line):
    python train_sft_kaggle.py                     # train with the defaults below
    python train_sft_kaggle.py --dry-run           # build + report the data, no GPU
    python train_sft_kaggle.py --limit 50 --max-steps 5   # end-to-end smoke run
    python train_sft_kaggle.py --help              # the rest of the CLI overrides

Usage (notebook): the flags above are a *command-line* interface -- a kernel's
argv belongs to the launcher, not to you, so pasting this into a cell parses no
flags at all (`in_notebook()` handles that; without it argparse dies on the
kernel's own `-f .../kernel-xxx.json`). Set options on CFG instead:

    CFG.category_caps = {c: 500 for c in ["cipher", "gravity", "bit_manipulation"]}
    CFG.num_train_epochs = 1
    main(dry_run=True)   # then main() for real

or, if you uploaded this file rather than pasting it, use the CLI properly with
    %run train_sft_kaggle.py --limit 50 --max-steps 5
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# --- point this at wherever you mounted the Kaggle Dataset from step 1/2 ---
INPUT_DIR = Path("/kaggle/input/synth-sft")  # <-- change to your dataset slug
if not INPUT_DIR.exists():
    # fall back to running straight out of this project checkout (local testing).
    # `__file__` is undefined when this is pasted into a notebook cell, so fall
    # back again to the working directory in that case.
    try:
        INPUT_DIR = Path(__file__).resolve().parent.parent
    except NameError:
        INPUT_DIR = Path.cwd()


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class Cfg:
    model_name: str = "Qwen/Qwen3-0.6B"  # matches the GRPO script's default
    data_file: str = "synth_sft.jsonl"
    output_dir: str = "/kaggle/working/sft_reasoners"

    prompt_style: str = "chat"  # "chat" (tokenizer chat template) | "raw"

    # Sequence budget. The corpus was generated under an 8192-token cap
    # (max observed 8188), so this sits above it to leave room for the
    # template + system prompt. Lower it only if VRAM forces you to, and read
    # the per-category drop report that gets printed when you do.
    max_seq_len: int = 8704

    # Split
    val_fraction: float = 0.05
    max_val_per_category: int = 50  # eval at 8k tokens is slow; keep it small
    seed: int = 0

    # Optional per-category cap applied *before* the split, e.g.
    # {"bit_manipulation": 500} -- its traces are ~6900-token bit-column dumps
    # that eat a disproportionate share of the token budget teaching
    # table-dumping. Empty means use every row.
    category_caps: dict[str, int] = field(default_factory=dict)

    # Quantization: QLoRA-style 4-bit base model (bitsandbytes nf4 + double
    # quant), LoRA adapters trained on top in fp32. Falls back to full
    # precision automatically if no CUDA device is found (bitsandbytes 4-bit
    # needs a GPU) -- see main().
    load_in_4bit: bool = True

    # LoRA
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Optimization
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    per_device_eval_batch_size: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    num_train_epochs: float = 2.0
    max_steps: int = -1  # >0 overrides num_train_epochs; handy for a smoke run
    loss_chunk_size: int = 1024  # see _make_chunked_loss_trainer()
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True

    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2


CFG = Cfg()


SYSTEM_PROMPT = """You are solving a hidden-rule reasoning puzzle from Alice's \
Wonderland. You are given example input -> output pairs that demonstrate a secret \
rule, then a new question to answer.

Think inside a single <think>...</think> block, written as a sequence of XML step \
tags:
  <step type="plan">...</step>          -- what you intend to do
  <step type="analysis">...</step>      -- inspect the examples, look for structure
  <step type="state_update">...</step>  -- record a fact you have deduced
  <step type="execution">...</step>     -- apply a deduced fact / do arithmetic
  <step type="verification">...</step>  -- replay a deduction against a given example
  <step type="conclusion">...</step>    -- state the final answer

After closing </think>, output the final answer on its own line as \\boxed{answer} \
and nothing else."""


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def apply_category_caps(rows: list[dict], caps: dict[str, int], seed: int) -> list[dict]:
    if not caps:
        return rows
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    rng = random.Random(seed)
    kept: list[dict] = []
    for cat, cat_rows in by_cat.items():
        cap = caps.get(cat)
        if cap is not None and cap < len(cat_rows):
            cat_rows = rng.sample(cat_rows, cap)
        kept.extend(cat_rows)
    rng.shuffle(kept)
    return kept


# --------------------------------------------------------------------------
# Prompt rendering + tokenization
# --------------------------------------------------------------------------


def _chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    try:
        # Qwen3 tokenizers accept this; when enable_thinking is False the
        # template injects its own empty `<think></think>` pair, which would
        # nest inside the trace's own `<think>`. Ask for the thinking path
        # explicitly, and assert_no_nested_think() below verifies it held.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=True,
        )
    except TypeError:
        # Qwen2.5 and most other templates reject the kwarg outright.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def render_prompt(tokenizer, prompt_text: str) -> str:
    """Render one row's prompt into the exact string the model is conditioned on."""
    if CFG.prompt_style == "raw":
        return prompt_text + "\n"
    if CFG.prompt_style != "chat":
        raise ValueError(f"unknown prompt_style: {CFG.prompt_style!r}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    return _chat_template(tokenizer, messages, add_generation_prompt=True)


def chat_template_think_prefix(tokenizer) -> str:
    """Literal `<think>...` prefix the chat template's generation prompt
    injects (Qwen3-style: an open, unclosed `<think>\\n`), or "" if it injects
    none (e.g. Qwen2.5, or prompt_style="raw").

    Every trace (see reasoners/store_types.py's wrap_trace_with_think) starts
    with its own literal `<think>\\n`. If the template's generation prompt
    already opens a `<think>` tag too, concatenating prompt + trace verbatim
    would duplicate/nest it -- so build_examples() strips this returned
    prefix off the front of each trace instead. Only an *unclosed* injected
    tag can be reconciled this way (the trace's own tag becomes the single
    real opening tag); a *closed* empty pair (`<think>...</think>`) would
    leave two separate think blocks with no clean way to merge them, so that
    case still raises. Only the generation-prompt *suffix* is inspected (the
    delta between rendering with and without it), so a `<think>` mentioned in
    SYSTEM_PROMPT itself doesn't false-positive.
    """
    if CFG.prompt_style != "chat":
        return ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "probe"},
    ]
    with_gen = _chat_template(tokenizer, messages, add_generation_prompt=True)
    without_gen = _chat_template(tokenizer, messages, add_generation_prompt=False)
    suffix = with_gen.removeprefix(without_gen)
    if "<think>" not in suffix:
        print(f"generation prompt suffix: {suffix!r} (no <think> injected)")
        return ""
    if "</think>" in suffix:
        raise RuntimeError(
            "This tokenizer's generation prompt injects a *closed* <think> "
            "block, which can't be reconciled with the trace's own leading "
            "<think> by prefix-stripping (it would leave two separate think "
            "blocks). Handle this template specifically, or set "
            f"Cfg.prompt_style='raw'.\n--- suffix ---\n{suffix!r}"
        )
    prefix = suffix[suffix.index("<think>"):]
    print(
        f"generation prompt suffix: {suffix!r} -- opens {prefix!r}; "
        "stripping this from each trace's leading text so it isn't duplicated"
    )
    return prefix


def completion_eos_id(tokenizer) -> int:
    """Token to terminate the completion with, so the model learns to stop."""
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    raise RuntimeError("tokenizer has neither eos_token_id nor pad_token_id")


def build_examples(tokenizer, rows: list[dict]) -> tuple[list[dict], Counter]:
    """Tokenize rows into {input_ids, labels, attention_mask, length, category}.

    Returns the kept examples plus a per-category count of rows dropped for
    exceeding `Cfg.max_seq_len` (dropped, never truncated -- see the module
    docstring).
    """
    think_prefix = chat_template_think_prefix(tokenizer)
    prompts = [render_prompt(tokenizer, r["prompt"]) for r in rows]
    traces = [r["trace"] for r in rows]
    if think_prefix:
        for i, t in enumerate(traces):
            if not t.startswith(think_prefix):
                raise RuntimeError(
                    f"expected trace to start with {think_prefix!r} (from "
                    f"wrap_trace_with_think), got {t[:40]!r}"
                )
            traces[i] = t[len(think_prefix):]

    eos_id = completion_eos_id(tokenizer)
    prompt_ids_all = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    trace_ids_all = tokenizer(traces, add_special_tokens=False)["input_ids"]

    kept: list[dict] = []
    dropped: Counter = Counter()
    for row, prompt_ids, trace_ids in zip(rows, prompt_ids_all, trace_ids_all):
        completion_ids = list(trace_ids) + [eos_id]
        input_ids = list(prompt_ids) + completion_ids
        if len(input_ids) > CFG.max_seq_len:
            dropped[row["category"]] += 1
            continue
        kept.append(
            {
                "input_ids": input_ids,
                "labels": [-100] * len(prompt_ids) + completion_ids,
                "attention_mask": [1] * len(input_ids),
                "length": len(input_ids),
                "category": row["category"],
            }
        )
    return kept, dropped


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------


def stratified_split(
    examples: list[dict], val_fraction: float, max_val_per_category: int, seed: int
) -> tuple[list[dict], list[dict]]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for ex in examples:
        by_cat[ex["category"]].append(ex)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        rng.shuffle(cat_rows)
        n_val = min(round(len(cat_rows) * val_fraction), max_val_per_category)
        val.extend(cat_rows[:n_val])
        train.extend(cat_rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def report(train: list[dict], val: list[dict], dropped: Counter) -> None:
    lengths: dict[str, list[int]] = defaultdict(list)
    for ex in train:
        lengths[ex["category"]].append(ex["length"])
    val_counts = Counter(ex["category"] for ex in val)

    print(f"\n{'category':<20} {'train':>7} {'val':>5} {'p50':>7} {'p95':>7} "
          f"{'max':>7} {'dropped':>8}")
    print("-" * 66)
    for cat in sorted(lengths):
        lens = sorted(lengths[cat])
        p95 = lens[min(len(lens) - 1, int(0.95 * len(lens)))]
        print(f"{cat:<20} {len(lens):>7} {val_counts[cat]:>5} "
              f"{int(statistics.median(lens)):>7} {p95:>7} {max(lens):>7} "
              f"{dropped[cat]:>8}")
    print("-" * 66)
    total_tokens = sum(sum(v) for v in lengths.values())
    print(f"{'TOTAL':<20} {len(train):>7} {len(val):>5} "
          f"{'':>7} {'':>7} {'':>7} {sum(dropped.values()):>8}")
    print(f"train tokens/epoch: {total_tokens:,} (loss is on completion tokens only)")
    if dropped:
        print(
            f"NOTE: {sum(dropped.values())} row(s) exceeded max_seq_len="
            f"{CFG.max_seq_len} and were dropped, not truncated."
        )


# --------------------------------------------------------------------------
# Collator
# --------------------------------------------------------------------------


def _make_chunked_loss_trainer(trainer_cls):
    """Trainer subclass that never materializes the full-sequence logits.

    The OOM at these sequence lengths is not the weights, it's the loss: an
    8k-token row against Qwen's ~152k vocab is a 2.5 GB bf16 logits tensor
    that `ForCausalLMLoss` then upcasts to fp32 (5 GB) and keeps a gradient
    for. Measured: that path OOMs a 6 GB GPU on a single row, and is tight
    even on a 16 GB T4.

    So the head is applied in `Cfg.loss_chunk_size`-token slices, each wrapped
    in `torch.utils.checkpoint` -- the chunk's logits are freed immediately and
    recomputed during backward, making peak loss memory proportional to the
    chunk, not the sequence. Everything else (LoRA, gradient checkpointing on
    the transformer body) is unchanged.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    class ChunkedLossTrainer(trainer_cls):
        _checked_gc = False

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            inputs = dict(inputs)
            labels = inputs.pop("labels")

            # Reach past the PEFT wrapper to the causal-LM, then run its body
            # only -- calling the causal-LM itself would compute the very
            # logits tensor this class exists to avoid. LoRA layers live inside
            # the body's modules, so they still apply.
            causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
            body = causal_lm.model
            lm_head = causal_lm.lm_head

            if not self._checked_gc:
                # We call the body directly, bypassing the causal-LM forward, so
                # confirm Trainer's gradient_checkpointing_enable actually landed
                # on this module -- if it silently didn't, activation memory
                # balloons and long rows OOM.
                self._checked_gc = True
                print(
                    "gradient checkpointing active on transformer body: "
                    f"{getattr(body, 'gradient_checkpointing', None)}"
                )

            hidden = body(**inputs).last_hidden_state
            shift_hidden = hidden[:, :-1, :]
            shift_labels = labels[:, 1:]

            def chunk_loss(h, y):
                logits = lm_head(h).float()
                return F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

            total = shift_hidden.new_zeros((), dtype=torch.float32)
            step = CFG.loss_chunk_size
            for start in range(0, shift_hidden.size(1), step):
                h = shift_hidden[:, start : start + step, :]
                y = shift_labels[:, start : start + step]
                if (y != -100).sum() == 0:
                    continue  # all-prompt / all-padding slice
                if torch.is_grad_enabled() and h.requires_grad:
                    total = total + checkpoint(chunk_loss, h, y, use_reentrant=False)
                else:
                    total = total + chunk_loss(h, y)

            # When Trainer passes num_items_in_batch it expects a sum-normalized
            # loss and skips its own gradient-accumulation division; when it
            # doesn't, it expects a per-token mean. Honour both.
            if num_items_in_batch is not None:
                loss = total / num_items_in_batch
            else:
                n_tokens = (shift_labels != -100).sum().clamp(min=1)
                loss = total / n_tokens

            return (loss, {"loss": loss}) if return_outputs else loss

    return ChunkedLossTrainer


class PadCollator:
    """Pad a batch to its own longest row. Labels pad with -100, not pad_token_id."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        import torch

        width = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = width - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [-100] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# --------------------------------------------------------------------------
# Training entrypoint
# --------------------------------------------------------------------------


def find_latest_checkpoint(output_dir: str) -> str | None:
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(
        out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    return str(checkpoints[-1]) if checkpoints else None


def build_training_arguments(
    training_arguments_cls, *, bf16: bool, fp16: bool, has_eval: bool = True
):
    """Build TrainingArguments, dropping kwargs this transformers version lacks.

    Kaggle's preinstalled transformers moves around (4.x there, 5.x locally at
    time of writing) and `TrainingArguments` fields come and go with it --
    `group_by_length` and `evaluation_strategy` are both gone in 5.x. Unknown
    kwargs are dropped with a warning rather than crashing a notebook run
    mid-session; none of the droppable ones change correctness, only batching
    efficiency and eval cadence.
    """
    import dataclasses

    kwargs = {
        "output_dir": CFG.output_dir,
        "per_device_train_batch_size": CFG.per_device_train_batch_size,
        "per_device_eval_batch_size": CFG.per_device_eval_batch_size,
        "gradient_accumulation_steps": CFG.gradient_accumulation_steps,
        "gradient_checkpointing": CFG.gradient_checkpointing,
        "learning_rate": CFG.learning_rate,
        "lr_scheduler_type": CFG.lr_scheduler_type,
        "warmup_ratio": CFG.warmup_ratio,
        "weight_decay": CFG.weight_decay,
        "num_train_epochs": CFG.num_train_epochs,
        "max_steps": CFG.max_steps,
        "max_grad_norm": CFG.max_grad_norm,
        "bf16": bf16,
        "fp16": fp16,
        "logging_steps": CFG.logging_steps,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": CFG.eval_steps,
        "save_strategy": "steps",
        "save_steps": CFG.save_steps,
        "save_total_limit": CFG.save_total_limit,
        # Batch rows of similar length together. The corpus is strongly bimodal
        # (equation_numeric ~600 tokens vs bit_manipulation ~7100), so random
        # batching pads short rows out to 8k -- wasted compute and an OOM
        # source. Only bites once per_device_train_batch_size > 1; it needs the
        # `length` column, hence remove_unused_columns=False (which also stops
        # Trainer choking on the string `category` column).
        "group_by_length": True,
        "length_column_name": "length",
        "remove_unused_columns": False,
        # eval only reports loss, so don't gather (huge) logits across the set
        "prediction_loss_only": True,
        "seed": CFG.seed,
        "report_to": [],
    }

    supported = {f.name for f in dataclasses.fields(training_arguments_cls)}
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    unsupported = sorted(k for k in kwargs if k not in supported)
    for key in unsupported:
        kwargs.pop(key)
    if unsupported:
        print(
            "WARNING: this transformers version does not support "
            f"{unsupported}; proceeding without them."
        )
    return training_arguments_cls(**kwargs)


def prepare_data(tokenizer, limit: int | None):
    data_path = INPUT_DIR / CFG.data_file
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Set INPUT_DIR to your Kaggle Dataset mount "
            "path, or run from the project checkout."
        )
    print(f"Loading {data_path} ...")
    rows = load_rows(data_path)
    caps = dict(CFG.category_caps)
    if limit is not None:
        # the jsonl is grouped by category, so a head-of-file cut would be
        # single-category -- cap per category instead
        caps = {r["category"]: limit for r in rows}
    rows = apply_category_caps(rows, caps, CFG.seed)
    print(f"{len(rows)} rows; tokenizing with {CFG.model_name} "
          f"(prompt_style={CFG.prompt_style!r}) ...")

    examples, dropped = build_examples(tokenizer, rows)
    train, val = stratified_split(
        examples, CFG.val_fraction, CFG.max_val_per_category, CFG.seed
    )
    report(train, val, dropped)
    return train, val


def preview(tokenizer, example: dict) -> None:
    """Print the masking boundary of one row -- cheap check that labels line up."""
    n_masked = sum(1 for t in example["labels"] if t == -100)
    prompt_tail = tokenizer.decode(example["input_ids"][max(0, n_masked - 60):n_masked])
    target_head = tokenizer.decode(example["input_ids"][n_masked:n_masked + 60])
    target_tail = tokenizer.decode(example["input_ids"][-40:])
    print("\n--- masking preview (category: {}) ---".format(example["category"]))
    print(f"[prompt tail, masked]  ...{prompt_tail!r}")
    print(f"[target head, trained] {target_head!r}...")
    print(f"[target tail, trained] ...{target_tail!r}")
    print("---\n")


def main(dry_run: bool = False, limit: int | None = None) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # training; flip to "left" for generation

    train_rows, val_rows = prepare_data(tokenizer, limit)
    preview(tokenizer, train_rows[0])

    if dry_run:
        print("--dry-run: data pipeline only, stopping before model load.")
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    fp16 = cuda and not bf16  # T4/P100: no bf16
    print(f"precision: {'bf16' if bf16 else 'fp16' if fp16 else 'fp32 (cpu)'}")

    compute_dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32

    use_4bit = CFG.load_in_4bit and cuda
    if CFG.load_in_4bit and not cuda:
        print(
            "WARNING: load_in_4bit requested but no CUDA device found; "
            "bitsandbytes 4-bit needs a GPU -- falling back to full precision."
        )

    if use_4bit:
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        print(
            "loading base model in 4-bit (QLoRA: nf4 + double quant, "
            f"compute dtype {compute_dtype})"
        )
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            CFG.model_name, quantization_config=quant_config, device_map="auto"
        )
        model.config.use_cache = False
        # Casts non-quantized modules (layernorms etc.) to fp32 and enables
        # input-require-grads + gradient checkpointing together -- doing this
        # by hand for a kbit model is easy to get subtly wrong.
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=CFG.gradient_checkpointing
        )
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                CFG.model_name, dtype=compute_dtype, device_map="auto"
            )
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            model = AutoModelForCausalLM.from_pretrained(
                CFG.model_name, torch_dtype=compute_dtype, device_map="auto"
            )
        model.config.use_cache = False
        if CFG.gradient_checkpointing:
            model.enable_input_require_grads()  # needed for LoRA + checkpointing

    peft_config = LoraConfig(
        r=CFG.lora_rank,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if fp16:
        # fp16 AMP needs fp32 master weights for the trainable params, else the
        # grad scaler raises "Attempting to unscale FP16 gradients" on T4/P100.
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.float()
    model.print_trainable_parameters()

    args = build_training_arguments(
        TrainingArguments, bf16=bf16, fp16=fp16, has_eval=bool(val_rows)
    )

    trainer = _make_chunked_loss_trainer(Trainer)(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows) if val_rows else None,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    resume_from = find_latest_checkpoint(CFG.output_dir)
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(CFG.output_dir)
    tokenizer.save_pretrained(CFG.output_dir)
    print(f"Done. Final adapter saved to {CFG.output_dir}")


def in_notebook() -> bool:
    """True when running inside a Jupyter/Kaggle/Colab kernel.

    A pasted-into-a-cell copy of this file still has `__name__ == "__main__"`,
    but `sys.argv` belongs to the kernel launcher (`-f .../kernel-xxx.json`),
    which argparse would reject with SystemExit(2). So parse an empty argv
    there and take settings from `CFG` instead.
    """
    return "ipykernel" in sys.modules or "google.colab" in sys.modules


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build + report the dataset, then stop before loading the model",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="keep at most N rows per category (quick runs; the jsonl is "
        "grouped by category, so this caps per category, not head-of-file)",
    )
    parser.add_argument("--model", default=None, help=f"default {CFG.model_name}")
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="disable 4-bit QLoRA quantization; load the base model at bf16/fp16",
    )
    parser.add_argument("--prompt-style", choices=["chat", "raw"], default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument(
        "--max-steps", type=int, default=None, help=">0 overrides --epochs"
    )
    parser.add_argument("--output-dir", default=None)

    if argv is None:
        argv = [] if in_notebook() else sys.argv[1:]
    ns = parser.parse_args(argv)

    for cfg_attr, cli_value in [
        ("model_name", ns.model),
        ("prompt_style", ns.prompt_style),
        ("max_seq_len", ns.max_seq_len),
        ("num_train_epochs", ns.epochs),
        ("max_steps", ns.max_steps),
        ("output_dir", ns.output_dir),
    ]:
        if cli_value is not None:
            setattr(CFG, cfg_attr, cli_value)
    if ns.full_precision:
        CFG.load_in_4bit = False

    main(dry_run=ns.dry_run, limit=ns.limit)


if __name__ == "__main__":
    # No sys.exit(): in a notebook that raises SystemExit and prints a spurious
    # traceback even on a clean finish.
    cli()
