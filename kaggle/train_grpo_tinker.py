"""Stage-3 GRPO on top of the `sft_tinker_runs/full_0727` SFT checkpoint, via
Thinking Machines' Tinker hosted RL loop (`tinker_cookbook.rl`).

Stage 2 (SFT) is done: `sft_tinker_runs/full_0727/` holds a rank-32 LoRA on
`Qwen/Qwen3.5-4B` that learned the `<think>` + six-tag + `\\boxed{}` format
(loss/token 0.334 -> 0.016; all 3 smoke completions closed `</think>` and
boxed correctly -- see CLAUDE.md's "The Tinker SFT training script"). This
script is stage 3: score the SFT policy's own sampled completions against
`reasoners/reward_<task>.py` and iterate with GRPO.

Platform is forced, not a choice: the local GPU is 5.67 GiB and the merged
model is 9.3 GB bf16, so this runs on Tinker
(`tinker_cookbook.rl.train.Config` + `main()`), not TRL's `GRPOTrainer` the
way `kaggle/train_grpo_cipher_kaggle.py` does for a 0.6B model. That script
is referenced below only for the reward-adapter shape, not as a base to
extend -- it targets a different stack and a different model.

--- Traps that would silently produce a paid run with zero learning signal ---
Each is invisible at runtime; every one below was verified against the
*installed* tinker/tinker_cookbook (0.23.4 / 0.5.2), not assumed from docs.

1. Do NOT subclass `tinker_cookbook.rl.problem_env.ProblemEnv`.
   `ProblemEnv.step` (rl/problem_env.py:121-122) calls
   `renderers.get_text_content(message)`, which strips `ThinkingPart`
   content (renderers/base.py:867-886). The entire `<step>` trace lives
   inside `<think>...</think>` (see store_types.wrap_trace_with_think), so
   the reward would see only the trailing bare `\\boxed{}` line, find zero
   `<step>` tags, and every `evaluate_structured_trace` would hit its
   "no conclusion tag" terminal penalty on every rollout. Uniform reward ->
   zero advantage after group-centering (rl/data_processing.py:39,
   `rewards_G - rewards_G.mean()`, no std division) -> no gradient, at full
   sampling cost. `ReasonerEnv` below subclasses `rl.types.Env` directly.

2. Reward text must be `think_prefix + tokenizer.decode(stripped_action)`,
   where `think_prefix` is *derived*, not hardcoded. Qwen3.5's renderer puts
   an unclosed `<think>\\n` in the generation prompt
   (renderers/qwen3_5.py:59-67: `_get_generation_suffix` appends
   `"<think>\\n"`), and SFT's `train_sft_kaggle.build_examples` stripped
   that exact prefix off every trace before tokenizing (so the model was
   never trained to re-emit it). `train_sft_kaggle.chat_template_think_prefix`
   is the function that computed what to strip at SFT time; call it again
   here (`ReasonerDatasetBuilder.__call__` does, then asserts it matches the
   tail of `renderer.build_generation_prompt(...)`'s own tokens via
   `_check_prompt_token_identity` in `--dry-run`) rather than re-deriving or
   hardcoding `"<think>\\n"` a second time, which would silently drift if
   either side changes.

3. The stop token is likely IN the sampled action, not stripped for you.
   `renderer.get_stop_sequences()` returns a *token id*
   (`[self._end_message_token]`, renderers/qwen3.py:210-216) used as
   Tinker's `SamplingParams.stop`, and `Qwen3Renderer.parse_response`'s own
   docstring says it "decodes the response, strips the `<|im_end|>` stop
   token" -- i.e. that token is present in the returned token list when
   `stop_reason == "stop"`, or `parse_response` wouldn't need to strip it.
   Left in place, `tokenizer.decode` renders it as literal `<|im_end|>` text
   glued onto the end of the completion. This does NOT break
   `evaluate_structured_trace`'s own correctness grading (it locates the
   boxed answer via `re.findall(r"\\boxed\\{(.*?)\\}", content)` on the
   *conclusion tag's captured content*, bounded by that step's own
   `</step>`, so trailing garbage after `</step>` is inert) -- but it WOULD
   break any extraction that requires the string to end in `}`
   (`scripts/eval_train_csv.last_boxed`, used below only for the `correct`
   *metric*, not the reward). `ReasonerEnv.step` therefore strips a trailing
   stop-id token before decoding, mirroring what `parse_response` does
   internally, rather than string-matching after the fact.

4. Reward functions take `list[tuple[str, str]]`, not `list[Example]`.
   `PARSERS[category][0](prompt)` (imported from `scripts/eval_train_csv.py`)
   returns `Example` objects; convert exactly as every `monitor_*.py` does:
   `[(e.input_value, e.output_value) for e in examples]`.

5. Wrap the reward call in `try/except Exception -> raw = -5.0`.
   `reward_gravity.py`/`reward_unit_conversion.py` call bare `float(...)` on
   regex-captured (adversarial, untrusted) trace content in several places
   (e.g. `_k_for_example`, `_factor_for_example`, the conclusion check) and
   can raise `ValueError` on malformed model output. An uncaught exception
   inside `Env.step` kills the whole rollout group under the default
   FailFast strategy.

6. Cipher is excluded. `reward_cipher.evaluate_structured_trace(text,
   oracle_map, expected_words)` has a different signature (no `examples`
   arg) and grades letter claims via `oracle_map.get(cipher) == plain` --
   a key missing from a partial oracle scores as *wrong*, not "unknown". A
   real prompt's handful of example sentences never covers the full
   26-letter bijection reward_cipher.py needs, so pointing it at a
   real-corpus row would falsely penalize every letter outside the visible
   examples (see CLAUDE.md's "Practical consequence for the GRPO plan").
   Cipher needs a bijection-first synthetic construction
   (`monitor_cipher.py`'s pattern), which is future work, not this script.

Pre-flight checks (all $0, all behind `--dry-run`, no Tinker client
constructed): `_check_gold_trace_rewards`, `_check_prompt_token_identity`,
`_check_trace_length_percentiles`, `_check_sft_holdout`. See their
docstrings.

Usage:
    uv run python kaggle/train_grpo_tinker.py --dry-run
    uv run python kaggle/train_grpo_tinker.py --yes --max-steps 2 --groups-per-batch 4 --group-size 4
    uv run python kaggle/train_grpo_tinker.py --yes
    uv run python kaggle/train_grpo_tinker.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sft_kaggle as local  # reuse load_rows/apply_category_caps/render_prompt/SYSTEM_PROMPT/CFG

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))
# uv sync --extra tinker installs these; imported eagerly (unlike
# train_sft_tinker.py's function-local imports) because none of this touches
# the network or needs TINKER_API_KEY at import time -- only
# tinker.ServiceClient(...) construction in run_training() does, and that's
# already gated behind --dry-run/--yes.
import chz
import eval_train_csv as eval_csv  # PARSERS, last_boxed -- scripts/ has no __init__.py
import tinker
from tinker_cookbook import checkpoint_utils, cli_utils, renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.train import Config as RLConfig
from tinker_cookbook.rl.train import main as rl_main
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from reasoners.reward_bit_manipulation import (
    evaluate_structured_trace as reward_bit_manipulation,
)
from reasoners.reward_cryptarithm import evaluate_structured_trace as reward_cryptarithm
from reasoners.reward_equation_numeric import (
    evaluate_structured_trace as reward_equation_numeric,
)
from reasoners.reward_gravity import evaluate_structured_trace as reward_gravity
from reasoners.reward_numeral import evaluate_structured_trace as reward_numeral
from reasoners.reward_unit_conversion import (
    evaluate_structured_trace as reward_unit_conversion,
)

# All 6 non-cipher categories take the same 3-arg (response_xml, examples,
# expected_answer) shape -- reward_cryptarithm's oracle_map/oracle_ops are
# optional kwargs (see its docstring: used only to gate alphabet validity,
# never to grade a claimed value), so calling it positionally with 3 args
# works identically to the others. Cipher is deliberately absent (trap 6).
REWARDS = {
    "numeral": reward_numeral,
    "equation_numeric": reward_equation_numeric,
    "cryptarithm": reward_cryptarithm,
    "gravity": reward_gravity,
    "unit_conversion": reward_unit_conversion,
    "bit_manipulation": reward_bit_manipulation,
}

# The SFT run this script's default checkpoint continues from. category_caps
# and seed are copied from sft_tinker_runs/full_0727/config.json (not read
# at runtime, so this script doesn't depend on that file surviving disk
# cleanup) -- used only to reproduce which rows train_sft_tinker.py's
# apply_category_caps(seed=0) would have handed to build_examples, so GRPO
# can roll out on the rest. See _check_sft_holdout's docstring for why this
# is a conservative superset (also excludes the ~5% that became SFT's val
# split) rather than an exact train/test partition.
SFT_LOG_DIR = "sft_tinker_runs/full_0727"
SFT_HOLDOUT_CATEGORY_CAPS: dict[str, int] = {
    "equation_numeric": 300,
    "numeral": 300,
    "cryptarithm": 300,
    "cipher": 150,
    "unit_conversion": 150,
    "gravity": 150,
    "bit_manipulation": 100,
}
SFT_HOLDOUT_SEED = 0


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class Cfg:
    model_name: str = "Qwen/Qwen3.5-4B"  # must match the SFT adapter's base model
    lora_rank: int = 32  # must match sft_tinker_runs/full_0727/config.json's lora_rank
    renderer_name: str | None = None  # None = resolve from the checkpoint (do not hardcode "qwen3_5")

    data_file: str = "synth_sft.jsonl"
    sft_checkpoints_file: str = f"{SFT_LOG_DIR}/checkpoints.jsonl"
    load_checkpoint_path: str | None = None  # None = read the "final" state_path from sft_checkpoints_file

    # ROADMAP Step 7: cheapest categories first, ~450-650 tokens/trace vs
    # ~5-7k for cipher/gravity/unit_conversion/bit_manipulation -- a ~10x
    # faster and cheaper iteration loop, and both take the 3-arg no-oracle
    # reward signature so they score any prompt, real or synthetic.
    categories: tuple[str, ...] = ("numeral", "equation_numeric")

    log_dir: str = field(default_factory=lambda: f"grpo_{time.strftime('%m%d_%H%M')}")
    # "delete" | "resume" | "ask" | "raise". "ask" (matching math_rl/train.py's
    # own CLIConfig default) blocks on stdin if log_dir already exists -- the
    # same footgun CLAUDE.md records for train_sft_tinker.py under
    # nohup/tmux. log_dir defaults to a timestamp so a collision is rare, but
    # pass --log-dir-behavior raise explicitly for any non-interactive run.
    log_dir_behavior: str = "ask"
    seed: int = 0

    group_size: int = 8
    groups_per_batch: int = 8  # small by default; raise for the real run
    learning_rate: float = 1e-5  # RL, not SFT's 2e-4
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    remove_constant_reward_groups: bool = True
    save_every: int = 5
    eval_every: int = 0  # no held-out RLDataset (see ReasonerDatasetBuilder.__call__); disabled
    max_steps: int | None = None
    num_substeps: int = 1

    # max_tokens must clear the trace length or rollouts truncate, lose the
    # conclusion tag, and take the terminal no-conclusion penalty -- that's
    # a different failure from trap 3 but reads identically (uniform low
    # reward). Measured with the Qwen3.5-4B tokenizer via
    # _check_trace_length_percentiles (2000 gold rows/category): numeral
    # p50=496 p95=615 max=702; equation_numeric p50=342 p95=415 max=469.
    # 1280 sits ~1.8x the observed max (not just p95 -- an early-training
    # policy rambles past gold length, and truncation-as-confound is worse
    # than a few wasted tokens). Re-check this comment if categories change.
    max_tokens: int = 1280

    # raw reward range is roughly -5 to +20 with an unbounded negative tail
    # (CLAUDE.md: penalties are never capped, up to _MAX_STEPS=500 steps);
    # compute_advantages only centers (no std division), so raw magnitude
    # scales the gradient directly. Squash via clip(raw, -reward_clip,
    # reward_clip) / reward_clip. Measured via _check_gold_trace_rewards
    # (200 gold rows/category, all non-negative, 0 stop_text_mismatches):
    # numeral min=19.95 p50=22.15 max=23.85; equation_numeric min=2.90
    # p50=17.95 max=21.00. 25.0 keeps gold numeral traces just under the
    # ceiling (~0.95) instead of saturating at 1.0, so the advantage signal
    # can still distinguish "very good" from "gold-quality" -- re-derive if
    # Cfg.categories changes.
    reward_clip: float = 25.0

    # Tinker sampling-meter price ($/M tokens); unlike the SFT script's
    # TRAIN_PRICE_PER_M_TOKENS this repo has no verified sampling price on
    # file, so it defaults unset and the cost estimate says so explicitly
    # rather than guessing (same contract as train_sft_tinker.py's unlisted-
    # model path).
    sample_price_per_m_tokens: float | None = None

    limit_per_category: int | None = None  # cap GRPO rows/category (smoke runs)
    gold_check_rows_per_category: int = 200  # for _check_gold_trace_rewards


# --------------------------------------------------------------------------
# Row loading / parsing (shared by the dataset builder and the dry-run checks)
# --------------------------------------------------------------------------


def _data_path(cfg: Cfg) -> Path:
    path = local.INPUT_DIR / cfg.data_file
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put {cfg.data_file!r} at the project root.")
    return path


def parse_row_examples(category: str, prompt: str) -> tuple[list[tuple[str, str]], str | None]:
    """(examples, question) via scripts/eval_train_csv.py's PARSERS, examples
    converted to list[tuple[str,str]] -- reward functions take tuples, not
    Example objects (trap 4)."""
    parse_fn = eval_csv.PARSERS[category][0]
    examples, question = parse_fn(prompt)
    return [(e.input_value, e.output_value) for e in examples], question


def sft_holdout_keys(data_path: Path) -> set[tuple[str, str]]:
    """(category, id) keys of every row train_sft_tinker.py's
    apply_category_caps(seed=SFT_HOLDOUT_SEED) would have handed to
    build_examples for the full_0727 SFT run.

    This is a conservative superset of the actual SFT *train* split: of the
    1376 rows that run trained on, this returns that plus the 73 rows that
    landed in its val split plus the 1 row build_examples dropped for
    exceeding max_seq_len (config.json: n_train=1376, n_val=73) -- i.e. it
    over-excludes by ~74 rows out of ~1450 for the two enabled categories.
    That's deliberate: stratified_split's row identity is lost by the time
    build_examples finishes tokenizing (its output dicts carry no id), so
    reconstructing the exact train/val boundary would require re-tokenizing
    with the SFT tokenizer for no real benefit -- a few dozen extra held-out
    rows costs nothing GRPO needs, whereas rolling out on a row SFT actually
    trained on would make "learned the format" indistinguishable from
    "memorized this row".
    """
    rows = local.load_rows(data_path)
    capped = local.apply_category_caps(rows, SFT_HOLDOUT_CATEGORY_CAPS, SFT_HOLDOUT_SEED)
    keys = {(r["category"], r["id"]) for r in capped}
    if len(keys) != len(capped):
        raise RuntimeError(
            f"(category, id) is not unique across the {len(capped)} capped SFT rows "
            f"({len(keys)} distinct keys) -- holdout subtraction would silently under-exclude."
        )
    return keys


def latest_sft_state_path(checkpoints_path: Path) -> str:
    """The 'final' record's state_path (fresh-optimizer weights load path,
    NOT sampler_path) from an SFT run's checkpoints.jsonl -- this is what
    rl/train.py:1956's create_training_client_from_state_async consumes.
    Falls back to the last record if a "final" entry isn't present (e.g. a
    run that only got as far as a periodic/interrupted save).
    """
    if not checkpoints_path.exists():
        raise FileNotFoundError(
            f"{checkpoints_path} not found -- pass --load-checkpoint-path explicitly, "
            "or point --sft-checkpoints at an SFT run's checkpoints.jsonl."
        )
    records = [json.loads(line) for line in checkpoints_path.read_text().splitlines() if line.strip()]
    if not records:
        raise RuntimeError(f"{checkpoints_path} has no checkpoint records.")
    for rec in reversed(records):
        if rec.get("name") == "final":
            return rec["state_path"]
    return records[-1]["state_path"]


# --------------------------------------------------------------------------
# Env / Dataset (see traps 1-4 above)
# --------------------------------------------------------------------------

# gravity/unit_conversion grade their boxed answer with a 1e-2 numeric
# tolerance, not exact string equality (CLAUDE.md: k_fit/factor_fit are
# median-of-truncated-divisions estimates, not exact recoveries). The other
# four categories' answers are Roman-numeral/symbol/bit/plain strings graded
# exactly by their own reward function -- a numeric tolerance would accept
# wrong answers there or crash on non-numeric content, so this metric-only
# `correct` check mirrors that same per-category split rather than a single
# rule for all six.
_TOLERANT_CATEGORIES = {"gravity", "unit_conversion"}
_CORRECT_TOLERANCE = 1e-2


def _boxed_matches(boxed_answer: str | None, expected_answer: str, category: str) -> bool:
    if boxed_answer is None:
        return False
    if boxed_answer == expected_answer:
        return True
    if category in _TOLERANT_CATEGORIES:
        try:
            return abs(float(boxed_answer) - float(expected_answer)) <= _CORRECT_TOLERANCE
        except ValueError:
            return False
    return False


class ReasonerEnv(Env):
    """Single-turn env for one reasoner-task prompt. Subclasses `rl.types.Env`
    directly (NOT `ProblemEnv` -- trap 1) so the reward sees the model's raw
    decoded text, `<think>` block included, rather than ProblemEnv's
    `get_text_content`-stripped version.
    """

    def __init__(
        self,
        *,
        renderer,
        convo_prefix: list[dict],
        question_prompt: str,
        reward_fn,
        examples: list[tuple[str, str]],
        expected_answer: str,
        category: str,
        think_prefix: str,
        reward_clip: float,
    ):
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.question_prompt = question_prompt
        self.reward_fn = reward_fn
        self.examples = examples
        self.expected_answer = expected_answer
        self.category = category
        self.think_prefix = think_prefix
        self.reward_clip = reward_clip
        self._stop_ids = {t for t in renderer.get_stop_sequences() if isinstance(t, int)}

    async def initial_observation(self):
        convo = self.convo_prefix + [{"role": "user", "content": self.question_prompt}]
        model_input = self.renderer.build_generation_prompt(convo)
        return model_input, self.renderer.get_stop_sequences()

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        stop_reason = (extra or {}).get("stop_reason", "stop")
        # Strip a trailing stop-id token before decoding (trap 3) -- mirrors
        # what Qwen3Renderer.parse_response does internally, but we need the
        # raw <think>-inclusive text, not its get_text_content()-stripped
        # Message, so we can't just call parse_response ourselves (trap 1).
        toks = action
        if stop_reason == "stop" and toks and toks[-1] in self._stop_ids:
            toks = toks[:-1]
        completion = self.renderer.tokenizer.decode(toks)
        text = self.think_prefix + completion

        reward_exception = False
        try:
            raw, _logs = self.reward_fn(text, self.examples, self.expected_answer)
        except Exception:  # noqa: BLE001 -- trap 5: reward fns can raise on adversarial content
            raw = -self.reward_clip
            reward_exception = True
        reward = max(-self.reward_clip, min(self.reward_clip, raw)) / self.reward_clip

        closed_think = "</think>" in completion
        reopened_think = "<think>" in completion
        has_step = "<step type=" in completion
        _, sep, tail = completion.rpartition("</think>")
        boxed_answer = eval_csv.last_boxed(tail) if sep else None
        has_boxed = boxed_answer is not None
        clean_stop = stop_reason == "stop"
        format_ok = closed_think and not reopened_think and has_step and has_boxed and clean_stop
        correct = _boxed_matches(boxed_answer, self.expected_answer, self.category)

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
            metrics={
                "format": float(format_ok),
                "correct": float(correct),
                "truncated": float(not clean_stop),
                "reopened_think": float(reopened_think),
                # A caught exception (trap 5) and a genuinely terrible
                # completion both produce reward_raw == -reward_clip --
                # without this flag they're indistinguishable in the logs,
                # and "the policy is bad" vs "the reward function is
                # crashing on every rollout" need different fixes.
                "reward_exception": float(reward_exception),
                "reward_raw": raw,
            },
        )


class ReasonerRLDataset(RLDataset):
    """Batches of ProblemGroupBuilder over pre-parsed, pre-shuffled rows."""

    def __init__(
        self,
        rows: list[dict],
        *,
        renderer,
        convo_prefix: list[dict],
        think_prefix: str,
        group_size: int,
        batch_size: int,
        reward_clip: float,
    ):
        self.rows = rows
        self.renderer = renderer
        self.convo_prefix = convo_prefix
        self.think_prefix = think_prefix
        self.group_size = group_size
        self.batch_size = batch_size
        self.reward_clip = reward_clip

    def get_batch(self, index: int) -> Sequence:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.rows))
        builders = []
        for row in self.rows[start:end]:
            builders.append(
                ProblemGroupBuilder(
                    env_thunk=partial(
                        ReasonerEnv,
                        renderer=self.renderer,
                        convo_prefix=self.convo_prefix,
                        question_prompt=row["prompt"],
                        reward_fn=REWARDS[row["category"]],
                        examples=row["examples_tuples"],
                        expected_answer=row["answer"],
                        category=row["category"],
                        think_prefix=self.think_prefix,
                        reward_clip=self.reward_clip,
                    ),
                    num_envs=self.group_size,
                    dataset_name=row["category"],
                )
            )
        return builders

    def __len__(self) -> int:
        return math.ceil(len(self.rows) / self.batch_size)


def _prepare_rows(cfg: Cfg, data_path: Path) -> list[dict]:
    """Load synth_sft.jsonl, subtract the SFT holdout, filter to enabled
    categories, parse each prompt's examples, shuffle. Rows that fail to
    parse (PARSERS returns no examples or no question) are dropped.
    """
    all_rows = local.load_rows(data_path)
    held_out = sft_holdout_keys(data_path)
    rng = random.Random(cfg.seed)

    by_cat: dict[str, list[dict]] = {}
    for r in all_rows:
        if r["category"] not in cfg.categories:
            continue
        if (r["category"], r["id"]) in held_out:
            continue
        by_cat.setdefault(r["category"], []).append(r)

    kept: list[dict] = []
    dropped_parse = 0
    for cat, cat_rows in by_cat.items():
        if cfg.limit_per_category is not None and len(cat_rows) > cfg.limit_per_category:
            cat_rows = rng.sample(cat_rows, cfg.limit_per_category)
        for row in cat_rows:
            examples, question = parse_row_examples(cat, row["prompt"])
            if not examples or question is None:
                dropped_parse += 1
                continue
            kept.append({**row, "examples_tuples": examples})

    rng.shuffle(kept)
    if dropped_parse:
        print(f"WARNING: {dropped_parse} row(s) dropped for failing to parse via PARSERS.")
    return kept


async def build_dataset_and_renderer(cfg: Cfg):
    """Shared by the dry-run checks and the real run so both see identical
    rows/renderer/think_prefix -- returns (dataset, renderer, tokenizer,
    think_prefix, rows).
    """
    local.CFG.prompt_style = "chat"  # must match full_0727's SFT config

    tokenizer = get_tokenizer(cfg.model_name)
    # "qwen3_5" is correct for full_0727 (a Qwen3.5-4B checkpoint) and lets
    # the dry-run checks run with no Tinker API call; the real run instead
    # resolves this from the checkpoint's own metadata (run_training below),
    # since only that path is allowed to touch the network.
    renderer_name = cfg.renderer_name or "qwen3_5"
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    think_prefix = local.chat_template_think_prefix(tokenizer)

    data_path = _data_path(cfg)
    rows = _prepare_rows(cfg, data_path)
    convo_prefix = [{"role": "system", "content": local.SYSTEM_PROMPT}]
    dataset = ReasonerRLDataset(
        rows,
        renderer=renderer,
        convo_prefix=convo_prefix,
        think_prefix=think_prefix,
        group_size=cfg.group_size,
        batch_size=cfg.groups_per_batch,
        reward_clip=cfg.reward_clip,
    )
    return dataset, renderer, tokenizer, think_prefix, rows


# --------------------------------------------------------------------------
# chz dataset builder (consumed by tinker_cookbook.rl.train.Config) --
# rebuilds everything build_dataset_and_renderer does, since the real run
# needs the resolved (checkpoint-derived) renderer_name, not the dry-run's
# guess.
# --------------------------------------------------------------------------


@chz.chz
class ReasonerDatasetBuilder(RLDatasetBuilder):
    data_path: str
    model_name_for_tokenizer: str
    renderer_name: str
    categories: tuple[str, ...]
    group_size: int
    groups_per_batch: int
    reward_clip: float
    seed: int = 0
    limit_per_category: int | None = None

    async def __call__(self) -> tuple[RLDataset, None]:
        local.CFG.prompt_style = "chat"
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        think_prefix = local.chat_template_think_prefix(tokenizer)

        cfg = Cfg(
            categories=self.categories,
            seed=self.seed,
            limit_per_category=self.limit_per_category,
        )
        rows = _prepare_rows(cfg, Path(self.data_path))
        convo_prefix = [{"role": "system", "content": local.SYSTEM_PROMPT}]
        dataset = ReasonerRLDataset(
            rows,
            renderer=renderer,
            convo_prefix=convo_prefix,
            think_prefix=think_prefix,
            group_size=self.group_size,
            batch_size=self.groups_per_batch,
            reward_clip=self.reward_clip,
        )
        return dataset, None


# --------------------------------------------------------------------------
# Pre-flight checks -- all $0, no Tinker client constructed
# --------------------------------------------------------------------------


def _check_gold_trace_rewards(cfg: Cfg, data_path: Path, tokenizer, renderer) -> None:
    """Score gold synth_sft.jsonl traces (guaranteed-correct by the
    foolproof contract -- see CLAUDE.md) against each enabled category's
    reward function and assert none score negative; a negative score on a
    gold trace means the reward function itself has a bug, not the model.

    Also scores each trace a second time with the renderer's stop-id token
    text appended (simulating trap 3 if left unhandled) -- if this changes
    the score, `evaluate_structured_trace`'s conclusion-tag parsing is more
    sensitive to trailing content than its own regex construction implies,
    and ReasonerEnv's stop-token stripping is load-bearing, not defensive
    paranoia.
    """
    stop_ids = renderer.get_stop_sequences()
    stop_text = tokenizer.decode(stop_ids) if stop_ids else ""

    print(f"\n=== gold-trace reward test (n<={cfg.gold_check_rows_per_category}/category) ===")
    by_cat: dict[str, list[dict]] = {}
    for r in local.load_rows(data_path):
        if r["category"] in cfg.categories:
            by_cat.setdefault(r["category"], []).append(r)

    rng = random.Random(cfg.seed)
    any_negative = False
    for cat in cfg.categories:
        rows = by_cat.get(cat, [])
        sample = rng.sample(rows, min(cfg.gold_check_rows_per_category, len(rows)))
        rewards, rewards_with_stop, mismatches = [], [], 0
        for row in sample:
            examples, _ = parse_row_examples(cat, row["prompt"])
            raw, _ = REWARDS[cat](row["trace"], examples, row["answer"])
            raw_stop, _ = REWARDS[cat](row["trace"] + stop_text, examples, row["answer"])
            rewards.append(raw)
            rewards_with_stop.append(raw_stop)
            if raw != raw_stop:
                mismatches += 1
        neg = sum(1 for r in rewards if r < 0)
        any_negative = any_negative or neg > 0
        print(
            f"{cat:<18} n={len(sample):>4} min={min(rewards):>7.2f} "
            f"p50={statistics.median(rewards):>7.2f} max={max(rewards):>7.2f} "
            f"negative={neg} stop_text_mismatches={mismatches}"
        )
    if any_negative:
        print(
            "WARNING: at least one gold trace scored negative -- a foolproof-contract "
            "trace is always correct, so this points at a reward-function bug, not "
            "a bad row. Do not proceed to a paid run until this is understood."
        )


def _check_prompt_token_identity(cfg: Cfg, data_path: Path, tokenizer, renderer) -> None:
    """Diff token IDs between train_sft_kaggle.render_prompt (the HF chat
    template SFT actually trained on) and the tinker_cookbook renderer's
    build_generation_prompt for the same messages. The renderer claims HF
    parity; verify it for Qwen3.5 specifically. A mismatch means every
    rollout starts off-distribution from the SFT checkpoint being loaded --
    this is the single check most likely to explain a smoke run with high
    reward variance but zero improvement.

    Also verifies chat_template_think_prefix's returned prefix is literally
    the tail of the renderer's own generation-prompt tokens (trap 2) --
    the two are independently derived and should agree by construction, but
    "should" is exactly what this function exists to stop assuming.
    """
    print("\n=== prompt token-identity check ===")
    local.CFG.prompt_style = "chat"
    row = next(r for r in local.load_rows(data_path) if r["category"] in cfg.categories)

    sft_prompt_str = local.render_prompt(tokenizer, row["prompt"])
    sft_ids = tokenizer(sft_prompt_str, add_special_tokens=False)["input_ids"]

    convo = [{"role": "system", "content": local.SYSTEM_PROMPT}, {"role": "user", "content": row["prompt"]}]
    renderer_ids = renderer.build_generation_prompt(convo).to_ints()

    if sft_ids == renderer_ids:
        print(f"MATCH: {len(sft_ids)} tokens identical.")
    else:
        n = min(len(sft_ids), len(renderer_ids))
        first_diff = next((i for i in range(n) if sft_ids[i] != renderer_ids[i]), n)
        print(
            f"MISMATCH at token {first_diff} "
            f"(sft len={len(sft_ids)}, renderer len={len(renderer_ids)}):"
        )
        window = 20
        print(f"  sft      ...{tokenizer.decode(sft_ids[max(0, first_diff - window):first_diff + window])!r}")
        print(f"  renderer ...{tokenizer.decode(renderer_ids[max(0, first_diff - window):first_diff + window])!r}")
        print(
            "  Inspect whether this is confined to system/user body whitespace "
            "(proceed) or falls in the assistant header / <think> region "
            "(blocking -- fix which prompt the GRPO side builds, not this check)."
        )

    think_prefix = local.chat_template_think_prefix(tokenizer)
    renderer_tail = tokenizer.decode(renderer_ids[-8:])
    if think_prefix and not renderer_tail.endswith(think_prefix):
        print(
            f"WARNING: chat_template_think_prefix()={think_prefix!r} does not match "
            f"the renderer's own generation-prompt tail {renderer_tail!r} -- reward "
            "text would be built with the wrong prefix (trap 2)."
        )
    else:
        print(f"think_prefix={think_prefix!r} confirmed as the renderer's generation-prompt tail.")


def _check_trace_length_percentiles(cfg: Cfg, data_path: Path, tokenizer) -> None:
    """Per-category p50/p95/max trace-only (not prompt+trace) token length
    with the model's own tokenizer -- CLAUDE.md's percentile table is
    Qwen2.5/Qwen3-0.6B and does not carry over. max_tokens is a completion
    budget, so it's measured against the trace alone. Set Cfg.max_tokens
    generously above the printed max (not p95 -- see the Cfg field comment
    on why truncation is worse than a few wasted tokens at iteration 0).
    """
    print("\n=== trace-length percentiles (this tokenizer) ===")
    by_cat: dict[str, list[int]] = {}
    for r in local.load_rows(data_path):
        if r["category"] in cfg.categories:
            n = len(tokenizer(r["trace"], add_special_tokens=False)["input_ids"])
            by_cat.setdefault(r["category"], []).append(n)

    for cat in cfg.categories:
        lens = sorted(by_cat.get(cat, []))
        if not lens:
            continue
        p50 = lens[len(lens) // 2]
        p95 = lens[min(len(lens) - 1, int(0.95 * len(lens)))]
        print(f"{cat:<18} n={len(lens):>5} p50={p50:>6} p95={p95:>6} max={lens[-1]:>6}")
    max_over_categories = max((max(v) for v in by_cat.values() if v), default=0)
    print(f"current Cfg.max_tokens={cfg.max_tokens}; observed max={max_over_categories}")
    if cfg.max_tokens < max_over_categories:
        print(
            f"WARNING: max_tokens ({cfg.max_tokens}) is below the observed max trace "
            f"length ({max_over_categories}) -- rollouts will truncate and lose the "
            "conclusion tag. Raise --max-tokens before a real run."
        )


def _check_sft_holdout(cfg: Cfg, data_path: Path) -> None:
    """Prints kept/held-out row counts per category after subtracting the
    rows sft_tinker_runs/full_0727 saw (see sft_holdout_keys' docstring for
    why this is a conservative superset of the exact SFT train split)."""
    print("\n=== SFT-holdout subtraction ===")
    all_rows = local.load_rows(data_path)
    held_out = sft_holdout_keys(data_path)
    by_cat_total: dict[str, int] = {}
    by_cat_held: dict[str, int] = {}
    for r in all_rows:
        if r["category"] not in cfg.categories:
            continue
        by_cat_total[r["category"]] = by_cat_total.get(r["category"], 0) + 1
        if (r["category"], r["id"]) in held_out:
            by_cat_held[r["category"]] = by_cat_held.get(r["category"], 0) + 1
    for cat in cfg.categories:
        total = by_cat_total.get(cat, 0)
        held = by_cat_held.get(cat, 0)
        print(f"{cat:<18} total={total:>5} sft_held_out={held:>5} available_for_grpo={total - held:>5}")


async def _check_env_step_closed_loop(
    cfg: Cfg, data_path: Path, tokenizer, renderer, think_prefix: str
) -> None:
    """Drives a real `ReasonerEnv` through `initial_observation()` + `step()`
    on a gold trace turned back into a synthetic sampled `action`.

    Every check above calls `REWARDS[cat](...)` directly -- none of them
    exercise `ReasonerEnv` itself: stop-token stripping, the
    `think_prefix + decode(...)` concatenation, `rpartition("</think>")`,
    `eval_csv.last_boxed(tail)`, or `StepResult` construction. A gold trace
    is correct by the foolproof contract, so round-tripping it through the
    actual env must come back `format=1.0 correct=1.0 truncated=0.0
    reward>0` for every category -- anything else is a bug in `ReasonerEnv`
    or its metric extraction, not the data or the (not-yet-sampled) model.
    """
    print("\n=== env.step() closed-loop check (gold trace round-tripped through ReasonerEnv) ===")
    by_cat: dict[str, list[dict]] = {}
    for r in local.load_rows(data_path):
        if r["category"] in cfg.categories:
            by_cat.setdefault(r["category"], []).append(r)

    convo_prefix = [{"role": "system", "content": local.SYSTEM_PROMPT}]
    stop_ids = renderer.get_stop_sequences()
    any_failed = False
    for cat in cfg.categories:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        row = rows[0]
        examples, _ = parse_row_examples(cat, row["prompt"])

        env = ReasonerEnv(
            renderer=renderer,
            convo_prefix=convo_prefix,
            question_prompt=row["prompt"],
            reward_fn=REWARDS[cat],
            examples=examples,
            expected_answer=row["answer"],
            category=cat,
            think_prefix=think_prefix,
            reward_clip=cfg.reward_clip,
        )

        obs, _stop_cond = await env.initial_observation()
        sft_prompt_str = local.render_prompt(tokenizer, row["prompt"])
        sft_ids = tokenizer(sft_prompt_str, add_special_tokens=False)["input_ids"]
        if obs.to_ints() != sft_ids:
            print(f"{cat:<18} FAIL: env.initial_observation() tokens != SFT prompt tokens")
            any_failed = True
            continue

        # Simulate a perfect rollout: the completion a model would have to
        # emit is the gold trace with think_prefix stripped -- exactly what
        # build_examples tokenized as the SFT completion target -- plus the
        # trailing stop token, exactly as a "stop"-terminated sample arrives
        # from the sampler (trap 3).
        completion_text = row["trace"].removeprefix(think_prefix) if think_prefix else row["trace"]
        completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        action = completion_ids + list(stop_ids)

        result = await env.step(action, extra={"stop_reason": "stop"})
        m = result.metrics
        ok = result.reward > 0 and m["format"] == 1.0 and m["correct"] == 1.0 and m["truncated"] == 0.0
        print(
            f"{cat:<18} {'OK' if ok else 'FAIL'} reward={result.reward:.3f} "
            f"format={m['format']} correct={m['correct']} truncated={m['truncated']} "
            f"reopened_think={m['reopened_think']}"
        )
        any_failed = any_failed or not ok

    if any_failed:
        print(
            "WARNING: a gold trace round-tripped through ReasonerEnv.step() did not "
            "score as a perfect rollout -- this is a bug in ReasonerEnv (stop-token "
            "handling, think_prefix, or metric extraction), not the data. Do not "
            "proceed to a paid run until every category above is OK."
        )


def estimate_rollout_cost(cfg: Cfg, avg_prompt_tokens: int) -> tuple[int, float | None]:
    steps = cfg.max_steps if cfg.max_steps is not None else 1
    if cfg.max_steps is None:
        print("NOTE: --max-steps not set; cost estimate below is for ONE iteration only.")
    tokens_per_group = cfg.group_size * (avg_prompt_tokens + cfg.max_tokens)
    total_tokens = tokens_per_group * cfg.groups_per_batch * steps
    if cfg.sample_price_per_m_tokens is None:
        return total_tokens, None
    return total_tokens, total_tokens / 1_000_000 * cfg.sample_price_per_m_tokens


# --------------------------------------------------------------------------
# CLI / entrypoint
# --------------------------------------------------------------------------


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="run pre-flight checks only, no network call")
    parser.add_argument("--yes", action="store_true", help="skip the cost-confirmation prompt")
    parser.add_argument("--categories", type=str, default=None, help="comma-separated, e.g. numeral,equation_numeric")
    parser.add_argument("--limit", type=int, default=None, help="cap GRPO rows per category (smoke runs)")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--load-checkpoint-path", type=str, default=None, help="override the SFT state_path to resume weights from")
    parser.add_argument("--sft-checkpoints", type=str, default=None, help="path to an SFT run's checkpoints.jsonl (default: sft_tinker_runs/full_0727/checkpoints.jsonl)")
    parser.add_argument("--renderer-name", type=str, default=None, help="override renderer auto-detection")
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--groups-per-batch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--reward-clip", type=float, default=None)
    parser.add_argument("--sample-price-per-m-tokens", type=float, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--log-dir-behavior", choices=["delete", "resume", "ask", "raise"], default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ns = cli(argv)
    cfg = Cfg()
    if ns.categories:
        cfg.categories = tuple(ns.categories.split(","))
    if ns.limit is not None:
        cfg.limit_per_category = ns.limit
    if ns.model:
        cfg.model_name = ns.model
    if ns.load_checkpoint_path:
        cfg.load_checkpoint_path = ns.load_checkpoint_path
    if ns.sft_checkpoints:
        cfg.sft_checkpoints_file = ns.sft_checkpoints
    if ns.renderer_name:
        cfg.renderer_name = ns.renderer_name
    if ns.group_size is not None:
        cfg.group_size = ns.group_size
    if ns.groups_per_batch is not None:
        cfg.groups_per_batch = ns.groups_per_batch
    if ns.learning_rate is not None:
        cfg.learning_rate = ns.learning_rate
    if ns.max_tokens is not None:
        cfg.max_tokens = ns.max_tokens
    if ns.max_steps is not None:
        cfg.max_steps = ns.max_steps
    if ns.save_every is not None:
        cfg.save_every = ns.save_every
    if ns.reward_clip is not None:
        cfg.reward_clip = ns.reward_clip
    if ns.sample_price_per_m_tokens is not None:
        cfg.sample_price_per_m_tokens = ns.sample_price_per_m_tokens
    if ns.log_dir:
        cfg.log_dir = ns.log_dir
    if ns.log_dir_behavior:
        cfg.log_dir_behavior = ns.log_dir_behavior

    for unknown in set(cfg.categories) - set(REWARDS):
        raise ValueError(f"unknown/unsupported category {unknown!r}; choose from {sorted(REWARDS)} (cipher excluded, trap 6)")

    data_path = _data_path(cfg)

    print("Building dataset + renderer for pre-flight checks ...")
    _dataset, renderer, tokenizer, think_prefix, rows = asyncio.run(build_dataset_and_renderer(cfg))
    print(f"think_prefix={think_prefix!r}")
    print(f"{len(rows)} rows available for GRPO across {cfg.categories} (post SFT-holdout subtraction, post parse filter)")

    _check_sft_holdout(cfg, data_path)
    _check_gold_trace_rewards(cfg, data_path, tokenizer, renderer)
    _check_prompt_token_identity(cfg, data_path, tokenizer, renderer)
    _check_trace_length_percentiles(cfg, data_path, tokenizer)
    asyncio.run(_check_env_step_closed_loop(cfg, data_path, tokenizer, renderer, think_prefix))

    avg_prompt_tokens = (
        statistics.mean(
            len(tokenizer(local.render_prompt(tokenizer, r["prompt"]), add_special_tokens=False)["input_ids"])
            for r in rows[: min(50, len(rows))]
        )
        if rows
        else 0
    )
    total_tokens, est_cost = estimate_rollout_cost(cfg, int(avg_prompt_tokens))
    if est_cost is not None:
        print(
            f"\nEstimated sample-meter tokens: {total_tokens:,} "
            f"(~${est_cost:.2f} at ${cfg.sample_price_per_m_tokens}/M)"
        )
    else:
        print(
            f"\nEstimated sample-meter tokens: {total_tokens:,} "
            "(no sampling price on file -- check "
            "https://tinker-docs.thinkingmachines.ai/tinker/models/ and pass "
            "--sample-price-per-m-tokens yourself)"
        )

    if ns.dry_run:
        return

    if not ns.yes:
        reply = input("Proceed with this Tinker GRPO run? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted, nothing charged.")
            return

    asyncio.run(run_training(cfg, data_path))


async def run_training(cfg: Cfg, data_path: Path) -> None:
    checkpoint_path = cfg.load_checkpoint_path or latest_sft_state_path(
        Path(cfg.sft_checkpoints_file) if Path(cfg.sft_checkpoints_file).is_absolute()
        else _ROOT / cfg.sft_checkpoints_file
    )
    print(f"Loading SFT weights from {checkpoint_path}")

    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cfg.model_name,
        explicit_renderer_name=cfg.renderer_name,
        load_checkpoint_path=checkpoint_path,
    )
    print(f"Resolved renderer_name={renderer_name!r}")

    log_path = str(_ROOT / "grpo_tinker_runs" / cfg.log_dir)
    cli_utils.check_log_dir(log_path, behavior_if_exists=cfg.log_dir_behavior)

    dataset_builder = ReasonerDatasetBuilder(
        data_path=str(data_path),
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=renderer_name,
        categories=cfg.categories,
        group_size=cfg.group_size,
        groups_per_batch=cfg.groups_per_batch,
        reward_clip=cfg.reward_clip,
        seed=cfg.seed,
        limit_per_category=cfg.limit_per_category,
    )

    config = RLConfig(
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        recipe_name="recipe_qwen_rlvr",
        renderer_name=renderer_name,
        lora_rank=cfg.lora_rank,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        log_path=log_path,
        load_checkpoint_path=checkpoint_path,
        kl_penalty_coef=cfg.kl_penalty_coef,
        remove_constant_reward_groups=cfg.remove_constant_reward_groups,
        num_substeps=cfg.num_substeps,
        eval_every=cfg.eval_every,
        save_every=cfg.save_every,
        max_steps=cfg.max_steps,
    )
    await rl_main(config)


if __name__ == "__main__":
    main()
