"""Stage-2 SFT on the synthetic step-tag corpus, via the Tinker hosted API.

Same job as `train_sft_kaggle.py` (teach Qwen the `<think>`-wrapped
`<step type="...">` output format from `synth_sft.jsonl`, not to solve the
tasks -- that's stage 3, GRPO), but training runs on Thinking Machines'
Tinker service instead of a local GPU. You pay per token instead of
provisioning hardware; see the cost estimate this script prints before it
spends anything.

--- Design notes (the non-obvious choices) ---

* **Data prep is reused, not reimplemented.** This script imports
  `train_sft_kaggle` as a sibling module and calls its `load_rows`,
  `apply_category_caps`, `build_examples`, and `stratified_split` directly.
  `build_examples` already solves the hard part correctly (prompt/completion
  tokenization, the `<think>`-prefix duplication check against the chat
  template, EOS placement, drop-not-truncate for over-length rows) --
  reimplementing it against an unfamiliar renderer would risk silently
  getting the prompt/completion boundary wrong, which you would only
  discover as garbage generations after paying for a run. The dependency
  is one-directional: `train_sft_kaggle.py` stays untouched and remains
  self-contained (its own docstring promises a Kaggle Dataset needs only
  `synth_sft.jsonl`, no other code from this repo) -- this script imports
  from it, never the reverse. Because `build_examples`/`render_prompt`/
  `chat_template_think_prefix` read `train_sft_kaggle.CFG` as a module
  global rather than taking it as a parameter, `main()` below sets the
  handful of fields they actually consult (`prompt_style`, `max_seq_len`)
  on that imported module's `CFG` singleton before calling them.

* **`-100` labels become a 0.0/1.0 weight tensor.** Tinker has no label
  sentinel; masking is expressed as a `weights` tensor aligned to
  `target_tokens = tokens[1:]` (0.0 for prompt tokens, 1.0 for completion
  tokens including the trailing EOS), with the model input being `tokens[:-1]`.
  Verified constructing against the installed `tinker` 0.23.4.

  Careful with second-hand references here: an earlier draft of this script
  took its call shapes from a sibling project's *wrapper* around the SDK, which
  added methods the real `TrainingClient` doesn't have (notably
  `save_checkpoint_async(name, log_path)` -- that name is a
  `tinker_cookbook.checkpoint_utils` module function, not a client method) and
  a `micro_batch_size` parameter on `forward_backward_async` that 0.23.4 does
  not expose. Every SDK call below is against the real client; see
  `save_checkpoint()` for the primitives that replaced it.

* **Every `*_async` returning an `APIFuture` needs `.result_async()`.**
  `forward_backward_async`, `optim_step_async`, `save_state_async`, and
  `save_weights_for_sampler_async` all hand back a future, not a response --
  so `.path` on the un-awaited return value silently yields nothing useful
  instead of raising. `SamplingClient.sample_async` is the exception: it
  returns a `SampleResponse` directly, and requires `num_samples`.

* **No local OOM-avoidance machinery.** `train_sft_kaggle.py`'s
  `_make_chunked_loss_trainer` exists solely because that script computes
  loss locally on a few-GB GPU (an 8k-token row against Qwen's ~152k-token
  vocab is a multi-GB fp32 logits tensor). Tinker computes forward/backward
  server-side and the caller never materializes logits, so there is nothing
  to chunk. The one part of that trainer worth keeping -- per-category
  loss/token and min-logprob bookkeeping -- is reproduced here from
  `fwd_bwd_result.loss_fn_outputs[i]["logprobs"].data`, written to
  `metrics.jsonl` under `--log-dir`.

* **The cheap default caps long categories harder than short ones.** A flat
  per-category row cap is not a flat cost cap: per-row token counts are
  wildly bimodal (see CLAUDE.md's SFT readiness audit -- ~450 tokens/row for
  equation_numeric vs ~6900 for bit_manipulation, a ~15x spread). A uniform
  300/category would still spend ~90% of the bill on 4 of 7 categories.
  `Cfg.category_caps` below caps the four long categories harder so the
  per-category token spend is closer to even; see the comment on that field
  for the actual numbers. Override it (or pass `--epochs`/`--limit`) for a
  bigger run once you've validated the format looks right.

* **Cost is estimated and confirmed before any paid call.** `main()` builds
  and tokenizes the dataset, prints the total train-meter token count and
  its dollar estimate against `TRAIN_PRICE_PER_M_TOKENS`, and requires
  either an interactive "y" or `--yes` before calling
  `forward_backward_async` for the first time. `--dry-run` stops even
  earlier, before the Tinker client is even constructed -- no network call,
  no charge, matching `train_sft_kaggle.py --dry-run`'s contract.

* **Nothing after the training loop can forfeit the run.** By the time the
  last batch returns, every token is billed, so `run_all()` saves a checkpoint
  on both the success and the exception path (including `KeyboardInterrupt`),
  saves periodically every `Cfg.save_every_steps` steps mid-loop, and wraps
  both the smoke test and the export so neither can prevent the other or
  discard a saved checkpoint. Paths land in `<log_dir>/checkpoints.jsonl`.

* **Export step is best-effort.** The final HF-export call
  (`tinker_cookbook.weights.download` + `build_hf_model`, both keyword-only,
  verified against cookbook 0.5.2) wasn't exercised against a live run -- if
  it fails, `export_for_local_use` catches it, prints the checkpoint
  identifier so you can export manually, and does not raise.

* **Val rows are for the smoke test, not a loss curve.** There is deliberately
  no eval pass: an eval forward on this corpus bills like training, and val
  loss on synthetic gold traces measures format learning rather than task
  generalization (CLAUDE.md's audit makes this point about the rule-leakage in
  the split). `smoke_test()` is what actually tells you whether the format
  took. Set `Cfg.val_fraction = 0.0` if you don't want to hold rows back.

--- Setup ---

1. `uv sync --extra tinker` (installs `tinker` + `tinker-cookbook` on top of
   this repo's base deps; the base install has no torch/peft requirement
   for this path -- Tinker runs the model server-side).
2. Set `TINKER_API_KEY` (note: **not** `TINKER_API` -- if your `.env` has a
   key named `TINKER_API`, the Tinker SDK will not see it; rename it).
   Either `export TINKER_API_KEY=...` yourself, or leave it in `.env` under
   the correct name and this script's `_resolve_api_key()` will pick it up.
3. `uv run python kaggle/train_sft_tinker.py --dry-run` -- builds + tokenizes
   the dataset and prints the cost estimate, no network call.
4. `uv run python kaggle/train_sft_tinker.py` -- asks for confirmation, then
   trains for real. `--yes` skips the confirmation prompt (for non-
   interactive runs); the cost estimate is still printed either way.

Usage:
    uv run python kaggle/train_sft_tinker.py --dry-run
    uv run python kaggle/train_sft_tinker.py --yes
    uv run python kaggle/train_sft_tinker.py --model Qwen/Qwen3-8B --epochs 2
    uv run python kaggle/train_sft_tinker.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sft_kaggle as local

# --------------------------------------------------------------------------
# Pricing (train meter, $/M tokens) -- verify against
# https://tinker-docs.thinkingmachines.ai/tinker/models/ before a real run;
# Tinker's prices change (these reflect the 2026-07-17 increase). A model
# not listed here still trains, but no cost estimate/confirmation gate can
# be shown for it.
# --------------------------------------------------------------------------
TRAIN_PRICE_PER_M_TOKENS: dict[str, float] = {
    "Qwen/Qwen3.5-4B": 0.737,
    "Qwen/Qwen3-8B": 0.44,
}

# The seven categories synth_sft.jsonl can contain (train.csv's plain unsplit
# names -- not Problem.category's _deduce/_guess variants; see CLAUDE.md). Used
# only to build a --limit cap dict without re-reading the corpus.
CATEGORIES: tuple[str, ...] = (
    "bit_manipulation",
    "cipher",
    "cryptarithm",
    "equation_numeric",
    "gravity",
    "numeral",
    "unit_conversion",
)


@dataclass
class Cfg:
    model_name: str = "Qwen/Qwen3.5-4B"
    data_file: str = "synth_sft.jsonl"
    log_dir: str = field(default_factory=lambda: f"sft_tinker_{time.strftime('%m%d_%H%M')}")

    prompt_style: str = "chat"  # "chat" | "raw" -- see train_sft_kaggle.Cfg
    max_seq_len: int = 8704

    val_fraction: float = 0.05
    max_val_per_category: int = 50
    seed: int = 0

    # Cheap-by-default: caps the four long categories harder than the three
    # short ones so token spend doesn't concentrate in a handful of
    # categories. Per-row median tokens (CLAUDE.md's audit): equation_numeric
    # ~450, numeral ~535, cryptarithm ~1571, cipher ~4673, unit_conversion
    # ~5258, gravity ~5995, bit_manipulation ~6873. At these caps, one epoch
    # is roughly 3.8M train tokens (~$2.80 at Qwen3.5-4B's $0.737/M) instead
    # of ~7.6M (~$5.60) for a flat 300/category, and no single category eats
    # more than ~25% of the bill. Override for a bigger run once the format
    # looks right in the smoke test.
    category_caps: dict[str, int] = field(
        default_factory=lambda: {
            "equation_numeric": 300,
            "numeral": 300,
            "cryptarithm": 300,
            "cipher": 150,
            "unit_conversion": 150,
            "gravity": 150,
            "bit_manipulation": 100,
        }
    )

    num_epochs: int = 1
    batch_size: int = 64

    # Checkpoint cadence, in optimizer steps. At the default caps one epoch is
    # only ~22 steps, so a failure near the end would otherwise forfeit the
    # whole spend -- see save_checkpoint(). 0 disables periodic saves (the
    # final/interrupted save still happens either way).
    save_every_steps: int = 10

    lora_rank: int = 32
    train_mlp: bool = True
    train_attn: bool = True
    # Unlike train_sft_kaggle.py's LoraConfig (which never targets
    # lm_head/embed_tokens), Tinker exposes unembedding as its own knob and
    # bills per token, not per trainable parameter -- so there's no cost
    # reason to withhold it. Default True; set False if you specifically
    # want parity with the local script's adapter shape.
    train_unembed: bool = True

    learning_rate: float = 2e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------


def _resolve_api_key() -> str:
    key = os.environ.get("TINKER_API_KEY")
    if key:
        return key

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name == "TINKER_API_KEY" and value:
                os.environ["TINKER_API_KEY"] = value
                return value
            if name == "TINKER_API" and value:
                raise RuntimeError(
                    f".env defines TINKER_API but the Tinker SDK reads the "
                    f"environment variable TINKER_API_KEY -- rename that key "
                    f"in {env_path} (or export TINKER_API_KEY yourself) "
                    "before running."
                )

    raise RuntimeError(
        "TINKER_API_KEY is not set. Export it, or add "
        "`TINKER_API_KEY=...` to .env at the project root."
    )


# --------------------------------------------------------------------------
# Data prep (reuses train_sft_kaggle's tokenization/masking, converts the
# result -100 labels into a 0/1 Tinker weight tensor)
# --------------------------------------------------------------------------


def prepare_examples(cfg: Cfg, tokenizer, data_path: Path):
    # build_examples/render_prompt/chat_template_think_prefix read these off
    # the imported module's CFG singleton, not a parameter -- see docstring.
    local.CFG.prompt_style = cfg.prompt_style
    local.CFG.max_seq_len = cfg.max_seq_len

    rows = local.load_rows(data_path)
    rows = local.apply_category_caps(rows, cfg.category_caps, cfg.seed)
    examples, dropped = local.build_examples(tokenizer, rows)
    train_examples, val_examples = local.stratified_split(
        examples, cfg.val_fraction, cfg.max_val_per_category, cfg.seed
    )
    local.report(train_examples, val_examples, dropped)
    return train_examples, val_examples


def estimate_cost(train_examples: list[dict], cfg: Cfg) -> tuple[int, float | None]:
    total_tokens = sum(len(e["input_ids"]) for e in train_examples) * cfg.num_epochs
    price = TRAIN_PRICE_PER_M_TOKENS.get(cfg.model_name)
    if price is None:
        return total_tokens, None
    return total_tokens, total_tokens / 1_000_000 * price


def to_datum(example: dict, tinker_module) -> object:
    """Convert one build_examples() row into a tinker.Datum.

    Mirrors huikang_nemotron/train_common.py:build_datum -- tokens[:-1] are
    the model input, tokens[1:] are the prediction targets, and the -100
    label sentinel becomes a 0.0/1.0 weight aligned to those targets.
    """
    tokens = example["input_ids"]
    labels = example["labels"]
    model_input = tinker_module.ModelInput(
        chunks=[tinker_module.types.EncodedTextChunk(tokens=tokens[:-1])]
    )
    target_tokens = tokens[1:]
    weights = [0.0 if lbl == -100 else 1.0 for lbl in labels[1:]]
    return tinker_module.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": tinker_module.TensorData(
                data=target_tokens, dtype="int64", shape=[len(target_tokens)]
            ),
            "weights": tinker_module.TensorData(
                data=weights, dtype="float32", shape=[len(weights)]
            ),
        },
    )


async def save_checkpoint(training_client, name: str, log_dir: Path) -> dict[str, str]:
    """Persist both checkpoint kinds and append a record to checkpoints.jsonl.

    `TrainingClient` has no `save_checkpoint_async` -- that name belongs to
    `tinker_cookbook.checkpoint_utils` (a module-level function taking a
    `loop_state` dict) and to the wrapper class in the huikang_nemotron repo.
    The SDK primitives are `save_state_async` (resumable training state) and
    `save_weights_for_sampler_async` (what a sampling client or the HF export
    path loads); both return an `APIFuture`, so the returned path only exists
    after `.result_async()` -- the same unwrap `forward_backward_async`
    already gets in train(). Verified against tinker 0.23.4.
    """
    state_future = await training_client.save_state_async(name=name)
    sampler_future = await training_client.save_weights_for_sampler_async(name=name)
    paths = {
        "state_path": (await state_future.result_async()).path,
        "sampler_path": (await sampler_future.result_async()).path,
    }

    log_dir.mkdir(parents=True, exist_ok=True)
    record = {"name": name, "time": time.strftime("%m-%d-%H-%M"), **paths}
    with open(log_dir / "checkpoints.jsonl", "a") as f:  # noqa: ASYNC230 -- one-line local JSONL append, the two network saves above dominate
        f.write(json.dumps(record) + "\n")
    print(f"saved checkpoint {name!r}: {paths['sampler_path']}")
    return paths


def stratified_batches(
    examples: list[dict], batch_size: int, rng: random.Random
) -> list[list[int]]:
    """Equal-sized batches with categories spread evenly across each batch.

    Ported from huikang_nemotron/train_sft.py's _stratified_batches.
    """
    n = len(examples)
    n_batches = math.ceil(n / batch_size)

    by_cat: dict[str, list[int]] = {}
    for i, ex in enumerate(examples):
        by_cat.setdefault(ex["category"], []).append(i)
    for idx_list in by_cat.values():
        rng.shuffle(idx_list)

    batches: list[list[int]] = [[] for _ in range(n_batches)]
    batch_order = list(range(n_batches))
    rng.shuffle(batch_order)
    assigned = 0
    for cat in sorted(by_cat.keys()):
        for idx in by_cat[cat]:
            batches[batch_order[assigned % n_batches]].append(idx)
            assigned += 1
    return batches


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


async def create_training_client(cfg: Cfg):
    """Built separately from train() so run_all() can own the try/finally that
    guarantees a checkpoint save even when the loop raises partway through."""
    import tinker

    service_client = tinker.ServiceClient()
    return await service_client.create_lora_training_client_async(
        base_model=cfg.model_name,
        rank=cfg.lora_rank,
        seed=cfg.seed,
        train_mlp=cfg.train_mlp,
        train_attn=cfg.train_attn,
        train_unembed=cfg.train_unembed,
    )


async def train(cfg: Cfg, training_client, train_examples: list[dict], log_dir: Path) -> int:
    """Run the training loop. Returns the number of optimizer steps completed."""
    import tinker
    from tinker import types

    n_batches = math.ceil(len(train_examples) / cfg.batch_size)
    total_steps = n_batches * cfg.num_epochs
    step = 0

    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = log_dir / "metrics.jsonl"

    with open(metrics_path, "w") as metrics_file:  # noqa: ASYNC230 -- small local JSONL write, network calls dominate
        for epoch in range(cfg.num_epochs):
            rng = random.Random(cfg.seed + epoch)
            batches = stratified_batches(train_examples, cfg.batch_size, rng)

            for batch_idxs in batches:
                batch = [train_examples[i] for i in batch_idxs]
                data = [to_datum(e, tinker) for e in batch]

                batch_start = time.time()
                fwd_bwd_future = await training_client.forward_backward_async(
                    data, loss_fn="cross_entropy", loss_fn_config={}
                )
                optim_future = await training_client.optim_step_async(
                    types.AdamParams(
                        learning_rate=cfg.learning_rate,
                        beta1=cfg.adam_beta1,
                        beta2=cfg.adam_beta2,
                        eps=cfg.adam_eps,
                        weight_decay=cfg.weight_decay,
                        grad_clip_norm=cfg.grad_clip_norm,
                    )
                )
                fwd_bwd_result = await fwd_bwd_future.result_async()
                optim_result = await optim_future.result_async()
                elapsed = time.time() - batch_start

                # Per-category loss/token + min-logprob, mirroring what
                # train_sft_kaggle.py's chunked-loss trainer logs locally.
                logprobs_list = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
                cat_loss: dict[str, float] = {}
                cat_tokens: dict[str, int] = {}
                cat_min_lp: dict[str, float] = {}
                for i, example in enumerate(batch):
                    cat = example["category"]
                    mask = [0.0 if lbl == -100 else 1.0 for lbl in example["labels"][1:]]
                    lp_data = logprobs_list[i].data
                    unmasked = [v for v, m in zip(lp_data, mask) if m]
                    if unmasked:
                        cat_loss[cat] = cat_loss.get(cat, 0.0) + sum(-v for v in unmasked)
                        cat_tokens[cat] = cat_tokens.get(cat, 0) + len(unmasked)
                        cat_min = min(unmasked)
                        if cat not in cat_min_lp or cat_min < cat_min_lp[cat]:
                            cat_min_lp[cat] = cat_min

                record = {
                    "epoch": epoch,
                    "step": step,
                    "n": len(data),
                    "elapsed": round(elapsed, 2),
                    "time": time.strftime("%m-%d-%H-%M"),
                }
                record.update({f"fwd/{k}": v for k, v in fwd_bwd_result.metrics.items()})
                if optim_result.metrics:
                    record.update({f"optim/{k}": v for k, v in optim_result.metrics.items()})
                total_cat_tokens = sum(cat_tokens.values())
                if total_cat_tokens > 0:
                    record["_loss_per_token"] = sum(cat_loss.values()) / total_cat_tokens
                for cat in sorted(cat_loss):
                    if cat_tokens[cat] > 0:
                        record[f"_loss_per_token/{cat}"] = cat_loss[cat] / cat_tokens[cat]
                for cat in sorted(cat_min_lp):
                    record[f"_min_logprob/{cat}"] = round(cat_min_lp[cat], 4)
                metrics_file.write(json.dumps(record) + "\n")
                metrics_file.flush()

                print(
                    f"epoch={epoch} step={step}/{total_steps} "
                    f"n={len(data)} t={elapsed:.1f}s"
                )
                step += 1

                # Periodic save: at the default caps one epoch is ~22 steps, so
                # dying at step 21 without this forfeits the entire spend. Skip
                # the last step -- run_all() saves "final" right after.
                if (
                    cfg.save_every_steps
                    and step % cfg.save_every_steps == 0
                    and step < total_steps
                ):
                    await save_checkpoint(training_client, f"step{step}", log_dir)

    return step


async def smoke_test(
    cfg: Cfg,
    training_client,
    tokenizer,
    val_examples: list[dict],
    log_dir: Path,
    fallback_examples: list[dict] | None = None,
) -> None:
    """Sample from the trained model on a few held-out prompts and check the
    emitted format -- val loss on a synthetic corpus doesn't tell you the model
    actually emits `<step type=...>`/`</think>`/a trailing `\\boxed{}`; this
    does. Closes the "no generation smoke test" gap noted in CLAUDE.md for the
    local script.

    Two things here are easy to get subtly wrong:

    * **`<think>` lives in the *prompt*, not the completion.** The chat
      template's generation prompt ends with an unclosed `<think>\\n`, and
      train_sft_kaggle.build_examples strips exactly that prefix off each
      trace so it isn't duplicated. So a *perfect* generation contains
      `</think>` but never `<think>`: requiring the opening tag in the sampled
      text is a guaranteed false negative, and re-emitting it is a defect worth
      reporting (`reopened_think`). Nor can this be fixed by testing
      prompt+completion together -- SYSTEM_PROMPT names both tags in its
      instructions, so a joined test is True before the model emits anything.
      Every check below therefore runs on the completion alone.
    * **A bare `\\boxed{` substring test is nearly vacuous.** Every gold trace's
      *plan* step contains the boilerplate "I will put my final answer inside
      \\boxed{}" ~150 chars in (measured across all seven categories), so
      `"\\boxed{" in completion` passes on a model that emitted one opening
      sentence and then rambled forever. The check that means something is
      whether a boxed answer appears *after* `</think>` -- the shape
      store_types.wrap_trace_with_think actually emits.

    * **`max_tokens` has to clear the trace length**, for the stop behaviour
      rather than the tags: median completions run 5.3k-7.1k tokens for
      cipher/gravity/unit_conversion/bit_manipulation, and SFT's whole reason
      for appending EOS is to teach the model to stop. Capped at 512 every
      sample ends in a length cutoff and `stop_reason` can never show a natural
      EOS. We budget the remaining context and prefer short categories so this
      stays cheap.

    Sampling bills on its own meter, separate from the training estimate.
    """
    examples = val_examples
    if not examples:
        # A small --limit run rounds val down to zero rows, which would skip
        # exactly the path a cheap rehearsal exists to exercise. Fall back to
        # train rows: this is a format check, not a generalization check, so a
        # seen prompt still answers "does it emit the tags and stop".
        examples = fallback_examples or []
        if not examples:
            print("No examples available; skipping generation smoke test.")
            return
        print("No val rows (small --limit?); smoke-testing on train rows instead.")

    import tinker
    from tinker import types

    # save_weights_for_sampler_async returns an APIFuture -- the path only
    # exists after result_async(), same as forward_backward_async in train().
    save_future = await training_client.save_weights_for_sampler_async(name="smoke")
    sampler_path = (await save_future.result_async()).path

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        model_path=sampler_path
    )

    # Prefer the cheap categories: a full equation_numeric/numeral trace is
    # ~450-950 tokens vs ~7k for bit_manipulation, and the format being checked
    # is identical across all seven.
    cheap = {"equation_numeric", "numeral", "cryptarithm"}
    rng = random.Random(0)
    pool = [e for e in examples if e["category"] in cheap] or examples
    for example in rng.sample(pool, min(3, len(pool))):
        prompt_len = sum(1 for lbl in example["labels"] if lbl == -100)
        prompt_ids = example["input_ids"][:prompt_len]
        model_input = tinker.ModelInput(
            chunks=[types.EncodedTextChunk(tokens=prompt_ids)]
        )
        result = await sampling_client.sample_async(
            prompt=model_input,
            num_samples=1,  # required, no default
            sampling_params=types.SamplingParams(
                max_tokens=max(256, cfg.max_seq_len - len(prompt_ids)),
                temperature=0.0,  # a format check shouldn't be sampled noisily
            ),
        )
        sequence = result.sequences[0]
        completion = tokenizer.decode(sequence.tokens)
        # Tag checks run on prompt+completion: see the docstring on why
        # `<think>` is never in the completion alone. Computed outside the
        # f-string -- a backslash in an f-string expression is a SyntaxError
        # before Python 3.12, and this project targets >=3.11.
        # All checks run on the completion alone. Joining the prompt back on
        # would contaminate them: SYSTEM_PROMPT spells out both `<think>` and
        # `</think>` as instructions, so any test against prompt+completion is
        # True before the model emits a single token.
        closed_think = "</think>" in completion  # the tag it must produce
        reopened_think = "<think>" in completion  # should NOT: prompt opened it
        has_step = "<step type=" in completion
        # A boxed answer only counts after the think block is closed, and only
        # if it was actually closed -- str.rpartition returns the whole string
        # in [2] when the separator is missing, which would silently fall back
        # to the vacuous whole-completion test the docstring warns about.
        _, sep, tail = completion.rpartition("</think>")
        has_boxed = bool(sep) and "\\boxed{" in tail
        print(
            f"[smoke:{example['category']}] closed_think={closed_think} "
            f"step={has_step} final_boxed={has_boxed} "
            f"reopened_think={reopened_think} stop={sequence.stop_reason} "
            f"tokens={len(sequence.tokens)}\n"
            f"...{completion[-200:]!r}\n"
        )

    (log_dir / "smoke_sampler_path.txt").write_text(str(sampler_path))


async def export_for_local_use(training_client, cfg: Cfg, log_dir: Path) -> None:
    """Best-effort export to a local HF-loadable directory, for a downstream
    GRPO script to load the way it loads a local LoRA checkpoint today. See
    the module docstring's note on why this is defensive: the checkpoint is
    already saved server-side by the time this runs, so a wrong guess here
    costs a manual export step, not the training run itself.
    """
    try:
        from tinker_cookbook import weights

        # APIFuture -> response.path, same unwrap as save_checkpoint().
        save_future = await training_client.save_weights_for_sampler_async(name="export")
        sampler_path = (await save_future.result_async()).path

        adapter_dir = log_dir / "adapter"
        weights.download(tinker_path=sampler_path, output_dir=str(adapter_dir))

        merged_dir = log_dir / "merged_hf_model"
        weights.build_hf_model(
            base_model=cfg.model_name,
            adapter_path=str(adapter_dir),
            output_path=str(merged_dir),
        )
        print(f"Exported merged HF model to {merged_dir}")
    except Exception as exc:  # noqa: BLE001 -- see docstring: never lose the run over this
        print(
            "Export step failed (training + checkpoint already succeeded, "
            f"only local export is affected): {exc!r}\n"
            "Check https://tinker-docs.thinkingmachines.ai/cookbook/api-reference/weights/ "
            "and export manually from the 'final' checkpoint saved under "
            f"{log_dir}."
        )


async def run_all(
    cfg: Cfg,
    tokenizer,
    train_examples: list[dict],
    val_examples: list[dict],
    log_dir: Path,
) -> None:
    """Own the whole paid lifecycle in a single event loop.

    Saving on both branches is the load-bearing part: by the time the loop
    raises, the tokens are already billed, so an exception must still leave a
    loadable checkpoint behind rather than forfeiting the run. `except
    BaseException` is deliberate -- a Ctrl-C mid-run is the most likely way this
    ends early, and it isn't an `Exception`. The smoke test is separately
    wrapped so a sampling-path failure can't take down the export after it.
    """
    training_client = await create_training_client(cfg)

    steps_done = 0
    try:
        steps_done = await train(cfg, training_client, train_examples, log_dir)
    except BaseException as exc:  # includes KeyboardInterrupt/CancelledError
        print(f"\nTraining loop raised {exc!r} -- saving what has been trained so far.")
        try:
            await save_checkpoint(training_client, "interrupted", log_dir)
        except Exception as save_exc:  # noqa: BLE001
            print(f"Interrupted-save also failed: {save_exc!r}")
        raise
    else:
        await save_checkpoint(training_client, "final", log_dir)

    if steps_done == 0:
        print("No optimizer steps ran; skipping smoke test and export.")
        return

    try:
        await smoke_test(
            cfg, training_client, tokenizer, val_examples, log_dir,
            fallback_examples=train_examples,
        )
    except Exception as exc:  # noqa: BLE001 -- must not block the export below
        print(f"Generation smoke test failed (checkpoint is already saved): {exc!r}")

    await export_for_local_use(training_client, cfg, log_dir)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="build + tokenize + report + estimate cost, no network call")
    parser.add_argument("--yes", action="store_true", help="skip the cost-confirmation prompt")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per category (overrides category_caps)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--prompt-style", choices=["chat", "raw"], default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = cli(argv)
    cfg = Cfg()
    if ns.model:
        cfg.model_name = ns.model
    if ns.epochs is not None:
        cfg.num_epochs = ns.epochs
    if ns.prompt_style:
        cfg.prompt_style = ns.prompt_style
    if ns.log_dir:
        cfg.log_dir = ns.log_dir
    if ns.limit is not None:
        # Cap every category the corpus can contain. Enumerated from the
        # Problem.category vocabulary rather than by reading the 97 MB corpus a
        # second time just to collect its distinct category values;
        # apply_category_caps ignores caps for categories that aren't present.
        cfg.category_caps = {cat: ns.limit for cat in CATEGORIES}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_path = _data_path(cfg)
    train_examples, val_examples = prepare_examples(cfg, tokenizer, data_path)

    total_tokens, est_cost = estimate_cost(train_examples, cfg)
    if est_cost is not None:
        print(
            f"Estimated train-meter tokens: {total_tokens:,} "
            f"(~${est_cost:.2f} at ${TRAIN_PRICE_PER_M_TOKENS[cfg.model_name]}/M "
            f"for {cfg.model_name}, {cfg.num_epochs} epoch(s))"
        )
    else:
        print(
            f"Estimated train-meter tokens: {total_tokens:,} "
            f"(no pricing on file for {cfg.model_name} -- check "
            "https://tinker-docs.thinkingmachines.ai/tinker/models/ yourself)"
        )

    if ns.dry_run:
        return

    _resolve_api_key()

    if not ns.yes:
        reply = input("Proceed with this Tinker training run? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted, nothing charged.")
            return

    log_dir = Path.cwd() / "sft_tinker_runs" / cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "time": time.strftime("%Y-%m-%d %H:%M"),
                "n_train": len(train_examples),
                "n_val": len(val_examples),
                "train_meter_tokens": total_tokens,
                "estimated_usd": est_cost,
            },
            indent=2,
        )
    )

    # One event loop for the whole paid lifecycle, so the try/finally in
    # run_all() can guarantee a checkpoint save.
    asyncio.run(run_all(cfg, tokenizer, train_examples, val_examples, log_dir))


def _data_path(cfg: Cfg) -> Path:
    path = local.INPUT_DIR / cfg.data_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put {cfg.data_file!r} at the project root "
            "(same lookup rule as train_sft_kaggle.py's INPUT_DIR fallback)."
        )
    return path


if __name__ == "__main__":
    main()
