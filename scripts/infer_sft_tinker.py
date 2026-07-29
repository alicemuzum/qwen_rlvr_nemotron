"""Sample one completion from a finished Tinker SFT run and compare it to gold.

Picks a random row out of `synth_sft.jsonl`, prints the prompt, samples the
trained checkpoint on it, prints what the model emitted, then prints the
dataset's own gold trace for the same row so the two can be eyeballed
side by side.

This is the "did the format take?" check that `train_sft_tinker.py`'s
`smoke_test()` runs at the end of training, except pointed at a run
directory after the fact and printing the full trace rather than a
200-char tail.

--- The non-obvious bits ---

* **Everything about the prompt comes from the run's own `config.json`**
  (`model_name`, `prompt_style`, `max_seq_len`), not from this file's
  defaults. That's the artifact recording what the checkpoint was actually
  trained under; hardcoding it here would silently drift the moment a run
  uses different settings.

* **The prompt is rendered by `train_sft_kaggle.render_prompt`, not
  re-implemented.** Same one-directional import `train_sft_tinker.py` uses
  (that module stays self-contained; we import from it, never the reverse),
  and for the same reason: `_chat_template` passes `enable_thinking=True`
  with a `TypeError` fallback, and getting that wrong changes whether the
  template injects a `<think>` tag at all. Tokenization uses
  `add_special_tokens=False`, matching `build_examples` exactly -- the chat
  template already emits its own special tokens, and a one-token difference
  from training means sampling off-distribution from the thing you paid for.

* **A correct completion contains `</think>` but never `<think>`.** The chat
  template's generation prompt ends with an unclosed `<think>\\n`, and
  `build_examples` strips that same prefix off every trace before training on
  it. So the model's first emitted token is already *inside* the think block.
  The gold trace on disk, by contrast, still carries its leading `<think>`
  (from `store_types.wrap_trace_with_think`). To keep the two printed blocks
  comparable, the gold trace is printed with that prefix stripped -- exactly
  what training did to it -- and both panes are labelled with what was
  stripped.

* **Local inference is deliberately not offered.** The merged HF model under
  `<run_dir>/merged_hf_model` is a ~9 GB bf16 4B checkpoint; generating a
  multi-thousand-token trace from it needs more VRAM than a laptop GPU has
  and would take many minutes on CPU. Sampling server-side is the practical
  path, and it bills on Tinker's sampling meter (a single short-category
  row is cents; `bit_manipulation` traces run ~7k tokens).

Usage:
    uv run python scripts/infer_sft_tinker.py --dry-run          # no spend: pick a row, render + print the prompt, print gold
    uv run python scripts/infer_sft_tinker.py                    # sample the 'final' checkpoint on a random row
    uv run python scripts/infer_sft_tinker.py --category numeral # cheap category (~500-900 tokens)
    uv run python scripts/infer_sft_tinker.py --seed 7 --checkpoint step20
    uv run python scripts/infer_sft_tinker.py --run-dir sft_tinker_runs/full_0727
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kaggle"))
import train_sft_kaggle as local  # imported after the sys.path line above

DEFAULT_RUN_DIR = "sft_tinker_runs/full_0727"


# --------------------------------------------------------------------------
# Run artifacts
# --------------------------------------------------------------------------


def load_run(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found -- is {run_dir} a train_sft_tinker.py run dir?"
        )
    return json.loads(cfg_path.read_text())


def sampler_path_for(run_dir: Path, name: str) -> str:
    """The `tinker://.../sampler_weights/<name>` path for one saved checkpoint.

    `checkpoints.jsonl` records both kinds train_sft_tinker.save_checkpoint()
    writes; the sampler weights are the ones a SamplingClient loads
    (`state_path` is resumable *training* state and will not sample).
    """
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- did the run save a checkpoint?")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for record in records:
        if record["name"] == name:
            return record["sampler_path"]
    available = ", ".join(r["name"] for r in records) or "(none)"
    raise KeyError(f"no checkpoint named {name!r} in {path}; available: {available}")


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


def sample_row(data_path: Path, category: str | None, seed: int) -> dict:
    """Reservoir-sample one row, streaming.

    `local.load_rows` would pull all 14000 rows (~93 MB of JSON) into memory
    to hand back one of them; a single pass keeping one candidate is the same
    uniform draw at constant memory.
    """
    rng = random.Random(seed)
    chosen: str | None = None
    seen = 0
    with data_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if category is not None and f'"category": "{category}"' not in line:
                # Cheap prefilter on the raw line; the authoritative check is
                # the parsed compare below, so a false positive here is fine.
                continue
            row_category = json.loads(line)["category"] if category else None
            if category is not None and row_category != category:
                continue
            seen += 1
            if rng.randrange(seen) == 0:
                chosen = line
    if chosen is None:
        raise ValueError(
            f"no rows in {data_path}"
            + (f" with category={category!r}" if category else "")
        )
    return json.loads(chosen)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


async def sample_completion(
    sampler_path: str,
    prompt_ids: list[int],
    tokenizer,
    max_tokens: int,
    temperature: float,
):
    """Returns (completion_text, stop_reason, n_tokens).

    `sample_async` is the one Tinker call that returns a response directly
    rather than an APIFuture needing `.result_async()`, and `num_samples` has
    no default -- see train_sft_tinker.py's docstring.
    """
    # Same .env fallback the training script uses -- the SDK reads
    # TINKER_API_KEY from the environment and won't find a key that only
    # exists in .env (and raises a clear error if .env misnames it). Resolved
    # before ServiceClient() is constructed.
    import tinker
    import train_sft_tinker
    from tinker import types

    train_sft_tinker._resolve_api_key()

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        model_path=sampler_path
    )
    result = await sampling_client.sample_async(
        prompt=tinker.ModelInput(chunks=[types.EncodedTextChunk(tokens=prompt_ids)]),
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=max_tokens, temperature=temperature
        ),
    )
    sequence = result.sequences[0]
    return tokenizer.decode(sequence.tokens), sequence.stop_reason, len(sequence.tokens)


def format_checks(completion: str) -> dict[str, bool]:
    """The same checks train_sft_tinker.smoke_test() applies, and for the same
    reasons: they run on the completion *alone* (SYSTEM_PROMPT names both think
    tags, so any test including the prompt is True before the model emits a
    token), and a boxed answer only counts if it appears after a `</think>`
    that was actually emitted -- `rpartition` returns the whole string when the
    separator is missing, which would silently degrade to the vacuous
    whole-completion test (every gold trace's *plan* step mentions
    `\\boxed{}` in its boilerplate ~150 chars in).
    """
    _, sep, tail = completion.rpartition("</think>")
    return {
        "closed_think": "</think>" in completion,
        "has_step_tags": "<step type=" in completion,
        "final_boxed": bool(sep) and "\\boxed{" in tail,
        "reopened_think": "<think>" in completion,  # should be False
    }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--data-file", default=None, help="default: the run config's data_file")
    parser.add_argument("--checkpoint", default="final", help="name in checkpoints.jsonl")
    parser.add_argument("--category", default=None, help="restrict the random draw to one category")
    parser.add_argument("--seed", type=int, default=None, help="reproducible row draw (default: random)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="default: the run's max_seq_len minus the prompt")
    parser.add_argument("--dry-run", action="store_true",
                        help="pick the row and render the prompt, but don't sample (no spend)")
    parser.add_argument("--emit-prompt", default=None, metavar="FILE",
                        help="also write {prompt_text, prompt_ids, ...} as JSON, to assert "
                             "token-for-token equality against another harness's rendering")
    ns = parser.parse_args(argv)

    run_dir = Path(ns.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    cfg = load_run(run_dir)
    seed = ns.seed if ns.seed is not None else random.randrange(1 << 30)

    data_path = REPO / (ns.data_file or cfg.get("data_file", "synth_sft.jsonl"))
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")

    print(
        f"run={run_dir}  model={cfg['model_name']}  "
        f"prompt_style={cfg['prompt_style']}  max_seq_len={cfg['max_seq_len']}\n"
        f"data={data_path.name}  seed={seed}"
    )

    # Prompt rendering reads these off the imported module's CFG singleton
    # (render_prompt/chat_template_think_prefix take no config parameter),
    # same as train_sft_tinker.prepare_examples does.
    local.CFG.prompt_style = cfg["prompt_style"]
    local.CFG.max_seq_len = cfg["max_seq_len"]

    row = sample_row(data_path, ns.category, seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    think_prefix = local.chat_template_think_prefix(tokenizer)
    prompt_text = local.render_prompt(tokenizer, row["prompt"])
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    rule(f"PROMPT  [id={row['id']}  category={row['category']}  "
         f"{len(prompt_ids)} tokens]")
    print(prompt_text)

    if ns.emit_prompt:
        # For diffing against a different inference harness. Text alone isn't
        # enough: a visually identical string can tokenize differently (the
        # chat template emits its own specials, hence add_special_tokens=False
        # above), and it's the ids the model actually sees.
        out = Path(ns.emit_prompt)
        out.write_text(json.dumps({
            "id": row["id"],
            "category": row["category"],
            "model_name": cfg["model_name"],
            "prompt_style": cfg["prompt_style"],
            "raw_prompt": row["prompt"],
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "think_prefix": think_prefix,
        }, indent=1))
        print(f"\n(wrote rendered prompt + {len(prompt_ids)} token ids to {out})")

    max_tokens = ns.max_tokens or max(256, cfg["max_seq_len"] - len(prompt_ids))

    if ns.dry_run:
        rule("MODEL OUTPUT")
        print(f"(--dry-run: skipped sampling; would allow max_tokens={max_tokens})")
    else:
        sampler_path = sampler_path_for(run_dir, ns.checkpoint)
        completion, stop_reason, n_tokens = asyncio.run(
            sample_completion(
                sampler_path, prompt_ids, tokenizer, max_tokens, ns.temperature
            )
        )
        checks = format_checks(completion)
        rule(f"MODEL OUTPUT  [checkpoint={ns.checkpoint}  {n_tokens} tokens  "
             f"stop_reason={stop_reason}]")
        if think_prefix:
            print(f"(the prompt already opened {think_prefix!r}, so the completion "
                  "starts inside the think block)")
        print(completion)
        print(
            "\nformat: "
            + "  ".join(f"{k}={v}" for k, v in checks.items())
            + ("\n  NOTE: reopened_think=True -- the model re-emitted a <think> "
               "tag the prompt had already opened." if checks["reopened_think"] else "")
            # stop_reason == "length" already means the cap was hit; don't
            # also gate on n_tokens, which would suppress the note if the
            # server's accounting is off by one from ours.
            + ("\n  NOTE: stop_reason is a length cutoff, not EOS -- the trace was "
               "cut off, so the boxed-answer check above is not meaningful."
               if stop_reason == "length" else "")
        )

    gold = row["trace"]
    stripped = ""
    if think_prefix and gold.startswith(think_prefix):
        gold = gold[len(think_prefix):]
        stripped = f"  (leading {think_prefix!r} stripped, as in training)"
    rule(f"DATASET TRACE (gold){stripped}")
    print(gold)
    print(f"\ngold answer: {row['answer']!r}")


if __name__ == "__main__":
    main()
