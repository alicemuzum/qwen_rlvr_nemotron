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

6. Cipher was excluded for this reason (fixed, kept for context): the
   original `reward_cipher.evaluate_structured_trace(text, oracle_map,
   expected_words)` grades letter claims via `oracle_map.get(cipher) ==
   plain` -- a key missing from a partial oracle scored as *wrong*, not
   "unknown", and a real prompt's handful of example sentences never
   covers the full 26-letter bijection that function needs (see CLAUDE.md's
   "Practical consequence for the GRPO plan"). `reward_cipher.py` itself is
   untouched (still what `monitor_cipher.py`/`train_grpo_cipher_kaggle.py`
   use with a real bijection-first oracle).
   `reasoners/reward_cipher_partial.py` is a separate, superset
   implementation: identical score to `reward_cipher.py` when the oracle
   IS complete (verified: 0/300 parity mismatches on gold traces), and for
   any letter missing from the oracle it grades self-consistency only
   (first claim neutral, reaffirm small/capped, contradiction still
   penalized) rather than assuming "wrong". Its
   `evaluate_structured_trace_from_examples(text, examples,
   expected_answer)` matches every other category's 3-arg shape and
   reconstructs the oracle from only the prompt's own visible examples
   (`partial_oracle_from_examples`, the same first-mapping-wins rule
   `cipher.py` uses internally, reimplemented so this has no dependency on
   the data-generation module) -- so cipher now scores any prompt, real or
   synthetic, the same way the other six do. See that file's module
   docstring for the full design rationale.

Pre-flight checks (all $0, all behind `--dry-run`, no Tinker client
constructed): `_check_sft_holdout`, `_check_gold_trace_rewards`,
`_check_reward_discrimination`, `_check_prompt_token_identity`,
`_check_trace_length_percentiles`, `_check_policy_length_distribution`,
`_check_env_step_closed_loop`. See their docstrings.

The one thing those cannot establish is whether the policy has anything left
to learn -- GRPO's gradient comes from rollouts inside a group *disagreeing*,
and measuring that needs real samples. `--probe-headroom` does it for ~3% of a
full run's cost and prints a go/no-go verdict per category; run it before
committing to a long run, not after.

Usage:
    uv run python kaggle/train_grpo_tinker.py --dry-run
    uv run python kaggle/train_grpo_tinker.py --probe-headroom          # PAID, ~1.1M tokens
    uv run python kaggle/train_grpo_tinker.py --yes --max-steps 2 --groups-per-batch 4 --group-size 4
    uv run python kaggle/train_grpo_tinker.py --yes
    uv run python kaggle/train_grpo_tinker.py --resume-latest --yes     # after a crash
    uv run python kaggle/train_grpo_tinker.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import re
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
from reasoners.reward_cipher_partial import (
    evaluate_structured_trace_from_examples as reward_cipher,
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

# All 7 categories take the same 3-arg (response_xml, examples,
# expected_answer) shape -- reward_cryptarithm's oracle_map/oracle_ops are
# optional kwargs (see its docstring: used only to gate alphabet validity,
# never to grade a claimed value), so calling it positionally with 3 args
# works identically to the others. Cipher's entry is
# reward_cipher_partial.evaluate_structured_trace_from_examples, not
# reward_cipher.py's original (see trap 6): it reconstructs a partial oracle
# from the prompt's own visible examples rather than needing a real
# bijection, at the cost of not being able to verify claims about letters
# absent from those examples (graded on self-consistency instead -- see
# reward_cipher_partial.py's module docstring).
REWARDS = {
    "numeral": reward_numeral,
    "equation_numeric": reward_equation_numeric,
    "cryptarithm": reward_cryptarithm,
    "gravity": reward_gravity,
    "unit_conversion": reward_unit_conversion,
    "bit_manipulation": reward_bit_manipulation,
    "cipher": reward_cipher,
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
    # "delete" | "resume" | "ask" | "raise". math_rl/train.py's own CLIConfig
    # defaults to "ask", which blocks on stdin if log_dir already exists -- the
    # same footgun CLAUDE.md records for train_sft_tinker.py under nohup/tmux.
    # Default to "raise" instead: failing loudly in two seconds beats hanging
    # silently for hours, and --resume-latest covers the case where continuing
    # an existing run is actually what you wanted.
    log_dir_behavior: str = "raise"
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
    # reward).
    #
    # This was 1280, derived from GOLD trace lengths (_check_trace_length_
    # percentiles: numeral p50=496 max=702; equation_numeric p50=342 max=469)
    # at "~1.8x the observed max". That is the wrong distribution: gold traces
    # are not what generates rollouts, the SFT policy is, and the policy's own
    # output is far longer. Measured on sft_tinker_train_csv_eval.csv (the
    # full_0727 checkpoint, temperature 0, 250 rows/category -- see
    # _check_policy_length_distribution, which re-derives this at --dry-run):
    #
    #   category          clean-stop p95 / max   length-stopped   max among CORRECT
    #   numeral                    457 /   759        0/250              759
    #   equation_numeric          3217 /  7093       32/250             1545
    #
    # So the policy already loops on ~13% of equation_numeric samples at greedy
    # decoding, before any RL pressure, and temperature 1.0 will be worse. But
    # the max over *correct* completions is only 759/1545: length above ~1600
    # empirically means the model is rambling, not reasoning. 2048 therefore
    # cannot truncate a right answer (1.3x the longest one ever observed) while
    # capping the spend on loops at ~25% of what 8192 would cost -- truncation
    # above this is a deliberate penalty on degenerate output, not a confound.
    # Re-derive from _check_policy_length_distribution if categories change.
    max_tokens: int = 2048

    # Saved rollouts of the SFT policy itself, used by
    # _check_policy_length_distribution to justify max_tokens against real
    # model output rather than gold traces. Produced by
    # scripts/eval_sft_tinker_train_csv.py; the check skips if it's absent.
    policy_eval_csv: str = "sft_tinker_train_csv_eval.csv"

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

    # Tinker meter prices for Qwen/Qwen3.5-4B ($/M tokens), read off
    # https://tinker-docs.thinkingmachines.ai/tinker/models/ on 2026-07-29
    # (checked: the 4B is NOT one of the models carrying the limited-time 50%
    # discount, so these are final). The train figure matches
    # train_sft_tinker.TRAIN_PRICE_PER_M_TOKENS.
    #
    # Three rates, not one, and getting this wrong dominates any other error in
    # the estimate:
    #  - prompt (prefill) tokens are $0.33, cheaper than generated tokens, and
    #    only $0.066 when cached -- likely here, since group_size rollouts share
    #    one prompt. The estimate does NOT claim the cache discount.
    #  - generated tokens are $1.005.
    #  - GRPO then bills the TRAIN meter too, on every rollout it keeps: each
    #    sampled sequence is fed to forward_backward. So the run costs roughly
    #    tokens x (sample + train), not tokens x sample.
    # `remove_constant_reward_groups` (train.py:1822) drops groups before that
    # train step, but only on *exact* reward equality -- with a continuous
    # reward even an all-wrong group usually varies, so do not count on it to
    # discount the train meter. probe_headroom measures the real rate.
    prefill_price_per_m_tokens: float | None = 0.33
    sample_price_per_m_tokens: float | None = 1.005
    train_price_per_m_tokens: float | None = 0.737

    limit_per_category: int | None = None  # cap GRPO rows/category (smoke runs)
    gold_check_rows_per_category: int = 200  # for _check_gold_trace_rewards
    discrimination_rows_per_category: int = 60  # for _check_reward_discrimination
    closed_loop_rows_per_category: int = 5  # for _check_env_step_closed_loop
    headroom_prompts_per_category: int = 30  # for probe_headroom (PAID)


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


def latest_sft_sampler_path(checkpoints_path: Path) -> str:
    """The 'final' record's sampler_path -- what create_sampling_client_async
    consumes, and NOT what latest_sft_state_path returns.

    train_sft_tinker.save_checkpoint writes both per record: `state_path` is
    the trainable state (weights + optimizer slots) that rl/train.py loads to
    *continue training* from, `sampler_path` is the inference-only export that
    a SamplingClient serves. probe_headroom needs the latter -- passing a
    state_path to create_sampling_client_async is a different object entirely.
    """
    if not checkpoints_path.exists():
        raise FileNotFoundError(
            f"{checkpoints_path} not found -- probe_headroom needs an SFT run's "
            "checkpoints.jsonl to find the sampler weights."
        )
    records = [json.loads(line) for line in checkpoints_path.read_text().splitlines() if line.strip()]
    if not records:
        raise RuntimeError(f"{checkpoints_path} has no checkpoint records.")
    for rec in reversed(records):
        if rec.get("name") == "final" and rec.get("sampler_path"):
            return rec["sampler_path"]
    for rec in reversed(records):
        if rec.get("sampler_path"):
            return rec["sampler_path"]
    raise RuntimeError(f"no record in {checkpoints_path} carries a sampler_path.")


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
                # Padding is currently free: duplicating every pre-conclusion
                # step in a gold trace leaves the reward flat-to-slightly-up
                # (equation_numeric p50 17.90 -> 18.50, measured by
                # _check_reward_discrimination). The ceilings bound how far
                # that can go, and there is no evidence the policy actually
                # pads under RL pressure -- so these two are instruments, not
                # a fix. If n_step_tags climbs while `correct` stays flat,
                # that's the moment to add a length penalty, with real data to
                # set it from.
                "completion_tokens": float(len(action)),
                "n_step_tags": float(completion.count('<step type=')),
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


def _answer_is_gradable(answer: str) -> bool:
    """False if the row's answer contains a brace, which every reward function's
    boxed-answer extraction mis-parses.

    Each `reward_<task>.py` locates the conclusion's answer with a non-greedy
    `re.findall(r"\\boxed\\{(.*?)\\}", content)`, which stops at the FIRST closing
    brace. CLAUDE.md documents this hazard for cryptarithm (whose symbol pool can
    contain literal braces) and prescribes the structural `last_boxed()` parse
    instead -- but the reward functions themselves still use the regex, and the
    hazard is not confined to cryptarithm: 34/2000 equation_numeric rows in
    synth_sft.jsonl have answers like '5}', '20{', '{26'.

    On those rows the extraction truncates ('5}' -> '5'), so even a *gold* trace
    is graded as having the wrong answer: measured gold reward p50 4.97 against
    17.90 for brace-free rows. For GRPO that is worse than a mis-scored row -- it
    is an unwinnable prompt. No completion can score well, so the group's reward
    is uniformly depressed, the centred advantage is noise, and the policy is
    pushed by whatever irrelevant variation remains. Excluding them costs 1.7% of
    equation_numeric and removes the noise entirely.

    Fixing the reward functions' own extraction is the deeper fix, but those are
    validated files (see CLAUDE.md's "Validating reward functions") and this
    script has no business patching six of them as a side effect of a GRPO run.
    """
    return "{" not in answer and "}" not in answer


def _prepare_rows(cfg: Cfg, data_path: Path) -> list[dict]:
    """Load synth_sft.jsonl, subtract the SFT holdout, filter to enabled
    categories, parse each prompt's examples, shuffle. Rows that fail to
    parse (PARSERS returns no examples or no question) are dropped, as are rows
    whose answer contains a brace (see _answer_is_gradable -- the reward function
    cannot grade those, so they are unwinnable prompts, not merely hard ones).
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
    dropped_brace = 0
    for cat, cat_rows in by_cat.items():
        if cfg.limit_per_category is not None and len(cat_rows) > cfg.limit_per_category:
            cat_rows = rng.sample(cat_rows, cfg.limit_per_category)
        for row in cat_rows:
            if not _answer_is_gradable(row["answer"]):
                dropped_brace += 1
                continue
            examples, question = parse_row_examples(cat, row["prompt"])
            if not examples or question is None:
                dropped_parse += 1
                continue
            kept.append({**row, "examples_tuples": examples})

    rng.shuffle(kept)
    if dropped_parse:
        print(f"WARNING: {dropped_parse} row(s) dropped for failing to parse via PARSERS.")
    if dropped_brace:
        print(
            f"{dropped_brace} row(s) dropped: brace in the answer, which the reward "
            "function's boxed-answer regex mis-parses (see _answer_is_gradable) -- "
            "those prompts are unwinnable and would inject noise into the advantage."
        )
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


def _policy_length_stats(cfg: Cfg) -> dict[str, dict[str, float]]:
    """Per-category completion-length stats of the SFT policy itself, read from
    cfg.policy_eval_csv. Returns {} if the file is absent.

    Keys per category: n, n_correct, clean_p50/p95/max (stop_reason=="stop"),
    correct_max, n_length_stopped, mean (all completions).
    """
    path = _ROOT / cfg.policy_eval_csv
    if not path.exists():
        return {}
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["category"] in cfg.categories]
    out: dict[str, dict[str, float]] = {}
    for cat in cfg.categories:
        rs = [r for r in rows if r["category"] == cat]
        if not rs:
            continue
        clean = sorted(int(r["n_tokens"]) for r in rs if r["stop_reason"] == "stop")
        corr = sorted(int(r["n_tokens"]) for r in rs if r["correct"].strip().lower() in ("true", "1"))
        allt = [int(r["n_tokens"]) for r in rs]
        out[cat] = {
            "n": len(rs),
            "n_correct": len(corr),
            "clean_p50": clean[len(clean) // 2] if clean else 0,
            "clean_p95": clean[int(0.95 * len(clean))] if clean else 0,
            "clean_max": clean[-1] if clean else 0,
            "correct_max": corr[-1] if corr else 0,
            "n_length_stopped": sum(1 for r in rs if r["stop_reason"] != "stop"),
            "mean": statistics.mean(allt),
        }
    return out


def _check_policy_length_distribution(cfg: Cfg) -> None:
    """Completion lengths of the ACTUAL SFT policy, split by stop_reason and by
    correctness -- the distribution max_tokens has to be set from.

    _check_trace_length_percentiles measures *gold* traces, which are only a
    lower bound on what the policy emits: gold equation_numeric tops out at 469
    tokens while the policy's clean-stop p95 is 3217 and 13% of its samples hit
    the sampler's cap outright, at temperature 0, before any RL pressure. Sizing
    max_tokens off gold (the original 1280) truncates real rollouts.

    The number that actually settles it is the max over *correct* completions
    (759 numeral / 1545 equation_numeric): no right answer has ever been longer,
    so a cap above that cannot cost a correct rollout, and everything above it
    is the model looping -- which is worth truncating and penalising rather than
    paying 4-5x to sample in full. See Cfg.max_tokens' comment.
    """
    print("\n=== policy completion-length distribution (SFT eval, temp 0) ===")
    stats = _policy_length_stats(cfg)
    if not stats:
        print(
            f"SKIPPED: {_ROOT / cfg.policy_eval_csv} not found -- max_tokens is then "
            "justified only by gold-trace lengths, which are a lower bound. Run "
            "scripts/eval_sft_tinker_train_csv.py, or use --probe-headroom (paid) to "
            "measure the policy directly on the real GRPO row pool."
        )
        return
    worst_correct = 0
    for cat, s in stats.items():
        worst_correct = max(worst_correct, int(s["correct_max"]))
        print(
            f"{cat:<18} n={int(s['n']):>4} correct={int(s['n_correct']):>4} "
            f"({s['n_correct'] / s['n']:>4.0%}) clean_p50={int(s['clean_p50']):>5} "
            f"clean_p95={int(s['clean_p95']):>5} clean_max={int(s['clean_max']):>5} "
            f"CORRECT_max={int(s['correct_max']):>5} length_stopped={int(s['n_length_stopped'])}"
        )
    if worst_correct and cfg.max_tokens < int(worst_correct * 1.2):
        print(
            f"WARNING: max_tokens={cfg.max_tokens} is below 1.2x the longest CORRECT "
            f"completion ({worst_correct}) -- real answers will truncate and take the "
            "no-conclusion penalty. Raise --max-tokens."
        )
    else:
        print(
            f"OK: max_tokens={cfg.max_tokens} clears the longest correct completion "
            f"({worst_correct}) with margin; longer rollouts are loops, truncated by design."
        )


def _mutate_wrong_box(trace: str, answer: str) -> str:
    return trace.replace("\\boxed{" + answer + "}", "\\boxed{" + answer + "X}")


def _mutate_minimal(answer: str) -> str:
    return (
        f'<think>\n<step type="conclusion">The answer is \\boxed{{{answer}}}</step>\n'
        f"</think>\n\\boxed{{{answer}}}"
    )


def _mutate_no_verification(trace: str) -> str:
    return re.sub(r'<step type="verification">.*?</step>\n?', "", trace, flags=re.DOTALL)


def _mutate_duplicate_steps(trace: str, answer: str) -> str | None:
    """Every pre-conclusion step repeated once, conclusion left intact."""
    if "<think>\n" not in trace or "</think>" not in trace:
        return None
    body = trace.split("<think>\n", 1)[1].rsplit("</think>", 1)[0]
    i = body.rfind('<step type="conclusion">')
    if i < 0:
        return None
    return "<think>\n" + body[:i] + body[:i] + body[i:] + f"</think>\n\\boxed{{{answer}}}"


def _check_reward_discrimination(cfg: Cfg, data_path: Path) -> None:
    """Does the reward SEPARATE right from wrong -- not merely score gold well?

    _check_gold_trace_rewards proves the reward function doesn't crash and rates
    correct traces highly. It does not prove the thing GRPO actually depends on:
    that a wrong completion scores *below* a right one for the same prompt. If it
    doesn't, every group of `group_size` rollouts has near-identical reward,
    compute_advantages centres it to ~0 (rl/data_processing.py:39, no std
    division), and the run buys nothing at full sampling cost.

    Five mutations of the same gold trace, per row:
      wrong_box   gold with only the boxed answer corrupted   MUST be < gold
      minimal_ok  bare conclusion, correct answer             expected << gold
      minimal_bad bare conclusion, wrong answer               MUST be < 0
      no_verify   verification steps deleted                  MUST be < gold
      duplicated  every pre-conclusion step repeated once     expected <= gold

    Baseline measured 2026-07-29, 60 rows/category (p50):
      numeral          gold 22.05 | wrong_box  7.05 | minimal_ok 10.00
                       minimal_bad -5.00 | no_verify 13.05 | duplicated 22.10
      equation_numeric gold 17.90 | wrong_box  2.93 | minimal_ok 10.00
                       minimal_bad -5.00 | no_verify 10.95 | duplicated 18.50
    Correctness is worth ~15 points, and deleting verification never once raised
    the score (0/120 rows) -- the reward rewards being right and being careful,
    not merely looking the part. `duplicated` landing slightly ABOVE gold is the
    one soft spot: padding is free. That is reported, not failed on; see the
    n_step_tags/completion_tokens metrics in ReasonerEnv.step.
    """
    print(f"\n=== reward discrimination (n<={cfg.discrimination_rows_per_category}/category) ===")
    by_cat: dict[str, list[dict]] = {}
    for r in local.load_rows(data_path):
        if r["category"] in cfg.categories:
            by_cat.setdefault(r["category"], []).append(r)

    rng = random.Random(cfg.seed)
    blocking = 0
    for cat in cfg.categories:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        sample = rng.sample(rows, min(cfg.discrimination_rows_per_category, len(rows)))
        cols: dict[str, list[float]] = {k: [] for k in
                                        ("gold", "wrong_box", "minimal_ok", "minimal_bad",
                                         "no_verify", "duplicated")}
        viol = {
            "wrong_box>gold": 0,
            "wrong_box==gold": 0,
            "minimal_bad>=0": 0,
            "no_verify>gold": 0,
            "duplicated>gold": 0,
        }
        for row in sample:
            examples, _ = parse_row_examples(cat, row["prompt"])
            ans, trace = row["answer"], row["trace"]

            def score(
                text: str,
                cat: str = cat,
                examples: list[tuple[str, str]] = examples,
                ans: str = ans,
            ) -> float:
                try:
                    return REWARDS[cat](text, examples, ans)[0]
                except Exception:  # noqa: BLE001 -- a raising mutation is a failure, not a crash
                    return float("nan")

            g = score(trace)
            wb = score(_mutate_wrong_box(trace, ans))
            mo = score(_mutate_minimal(ans))
            mb = score(_mutate_minimal(ans + "X"))
            nv = score(_mutate_no_verification(trace))
            dup_text = _mutate_duplicate_steps(trace, ans)
            dp = score(dup_text) if dup_text else g

            for k, v in zip(cols, (g, wb, mo, mb, nv, dp)):
                cols[k].append(v)
            # Separate "corrupting the answer made things BETTER" (a real
            # discrimination failure, blocking) from "corrupting the answer
            # changed nothing" (the reward never read the answer correctly to
            # begin with -- a brace-in-answer row that _answer_is_gradable now
            # filters out of the GRPO pool; reported, not blocking).
            if wb > g:
                viol["wrong_box>gold"] += 1
            elif wb == g:
                viol["wrong_box==gold"] += 1
            if mb >= 0:
                viol["minimal_bad>=0"] += 1
            if nv > g:
                viol["no_verify>gold"] += 1
            if dp > g:
                viol["duplicated>gold"] += 1

        print(f"{cat:<18} n={len(sample)}")
        for k, v in cols.items():
            print(f"    {k:<12} min={min(v):>7.2f} p50={statistics.median(v):>7.2f} max={max(v):>7.2f}")
        print(f"    violations: {viol}")
        if viol["wrong_box>gold"] or viol["minimal_bad>=0"] or viol["no_verify>gold"]:
            blocking += 1
        if viol["wrong_box==gold"]:
            print(
                f"    NOTE: corrupting the boxed answer changed nothing on "
                f"{viol['wrong_box==gold']}/{len(sample)} rows -- the reward never parsed "
                "the answer on those (brace-in-answer rows; _prepare_rows drops them from "
                "the GRPO pool, but this check samples the raw corpus)."
            )
        if viol["duplicated>gold"]:
            print(
                f"    NOTE: padding raised the reward on {viol['duplicated>gold']}/{len(sample)} rows "
                "-- expected (ceilings bound the gain, nothing penalises length). Watch the "
                "n_step_tags metric during the run rather than adding a length penalty blind."
            )

    if blocking:
        print(
            "WARNING: the reward failed to separate a wrong completion from a right one "
            "on at least one category. Every rollout group would then have near-uniform "
            "reward and zero advantage -- GRPO cannot learn. Do NOT start a paid run."
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

    Runs `Cfg.closed_loop_rows_per_category` randomly-drawn rows rather than one
    fixed row (the first row of a category is the same row every run, so a bug
    that only shows on some prompt shapes would never surface), and additionally
    drives one deliberately *truncated* action through the env with
    `stop_reason="length"`. Nothing else exercises that path, yet it is not an
    edge case: the SFT policy hits the sampler's cap on ~13% of equation_numeric
    samples (see _check_policy_length_distribution), so a real run takes it
    constantly, and it must come back visibly bad -- truncated=1.0, format=0.0,
    and a reward below the gold rollout's -- or the loop teaches nothing by it.
    """
    print("\n=== env.step() closed-loop check (gold trace round-tripped through ReasonerEnv) ===")
    by_cat: dict[str, list[dict]] = {}
    for r in local.load_rows(data_path):
        if r["category"] in cfg.categories:
            by_cat.setdefault(r["category"], []).append(r)

    convo_prefix = [{"role": "system", "content": local.SYSTEM_PROMPT}]
    stop_ids = renderer.get_stop_sequences()
    rng = random.Random(cfg.seed)
    any_failed = False
    for cat in cfg.categories:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        sample = rng.sample(rows, min(cfg.closed_loop_rows_per_category, len(rows)))

        def make_env(row: dict, examples: list[tuple[str, str]], cat: str = cat) -> ReasonerEnv:
            # Env is documented single-use ("create it, run one episode, then
            # discard"), so build a fresh one per step() rather than reusing.
            return ReasonerEnv(
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

        n_ok = 0
        gold_reward = 0.0
        last_ids: list[int] = []
        last_row: dict = {}
        last_examples: list[tuple[str, str]] = []
        for row in sample:
            examples, _ = parse_row_examples(cat, row["prompt"])
            env = make_env(row, examples)

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
            n_ok += int(ok)
            any_failed = any_failed or not ok
            gold_reward, last_ids = result.reward, completion_ids
            last_row, last_examples = row, examples
            if not ok:
                print(
                    f"{cat:<18} FAIL id={row.get('id')} reward={result.reward:.3f} "
                    f"format={m['format']} correct={m['correct']} truncated={m['truncated']} "
                    f"reopened_think={m['reopened_think']}"
                )

        print(f"{cat:<18} gold rollouts {n_ok}/{len(sample)} OK (last reward={gold_reward:.3f})")

        # The failure path: a length-truncated rollout must be visibly bad.
        if last_ids:
            trunc_action = last_ids[: max(1, int(0.6 * len(last_ids)))]
            res = await make_env(last_row, last_examples).step(
                trunc_action, extra={"stop_reason": "length"}
            )
            tm = res.metrics
            t_ok = (
                tm["truncated"] == 1.0
                and tm["format"] == 0.0
                and tm["correct"] == 0.0
                and res.reward < gold_reward
            )
            any_failed = any_failed or not t_ok
            print(
                f"{cat:<18} truncated rollout {'OK' if t_ok else 'FAIL'} "
                f"reward={res.reward:.3f} (gold={gold_reward:.3f}) format={tm['format']} "
                f"correct={tm['correct']} truncated={tm['truncated']}"
            )

    if any_failed:
        print(
            "WARNING: a gold trace round-tripped through ReasonerEnv.step() did not "
            "score as a perfect rollout, or a truncated one did not score as a bad "
            "one -- this is a bug in ReasonerEnv (stop-token handling, think_prefix, "
            "or metric extraction), not the data. Do not proceed to a paid run until "
            "every line above is OK."
        )


def estimate_rollout_cost(
    cfg: Cfg,
    avg_prompt_tokens: int,
    n_batches: int,
    avg_completion_tokens: int | None = None,
) -> tuple[int, float | None]:
    """Sampling tokens for the WHOLE run, not one iteration.

    This previously assumed `steps = 1` whenever `--max-steps` was unset -- but
    unset is the *default*, and rl/train.py then runs a full pass over the
    dataset (`end_batch = min(max_steps, len(dataset)) if max_steps is not None
    else len(dataset)`, train.py:1985). At the shipped defaults that is ~425
    steps x 64 rollouts, so the confirmation prompt was quoting roughly 1/425th
    of what the operator was agreeing to. A gate that under-reports the spend by
    two orders of magnitude is not a gate.

    Reports two figures: the worst case (every rollout runs to max_tokens) and,
    when the SFT policy's measured mean completion length is available, the
    likely case. The gap between them is large, because most completions stop
    early -- max_tokens is a cap, not a per-rollout cost.
    """
    steps = min(cfg.max_steps, n_batches) if cfg.max_steps is not None else n_batches
    rollouts = cfg.group_size * cfg.groups_per_batch * steps
    worst = rollouts * (avg_prompt_tokens + cfg.max_tokens)
    print(
        f"{steps} steps x {cfg.groups_per_batch} groups x {cfg.group_size} samples = "
        f"{rollouts:,} rollouts "
        f"({'full epoch' if cfg.max_steps is None else f'--max-steps {cfg.max_steps}'}, "
        f"dataset has {n_batches} batches)"
    )
    def dollars(completion_tokens: int) -> str:
        """Prompt tokens prefill at one rate, generated tokens at another, and
        GRPO then trains on both -- three rates, see Cfg's pricing comment."""
        p, s, t = (
            cfg.prefill_price_per_m_tokens,
            cfg.sample_price_per_m_tokens,
            cfg.train_price_per_m_tokens,
        )
        if p is None or s is None or t is None:
            return ""
        prefill_m = rollouts * avg_prompt_tokens / 1_000_000
        gen_m = rollouts * completion_tokens / 1_000_000
        prefill_cost, gen_cost = prefill_m * p, gen_m * s
        train_cost = (prefill_m + gen_m) * t
        return (
            f"  =  ${prefill_cost + gen_cost + train_cost:7.2f}"
            f"  (prefill ${prefill_cost:.2f} + generate ${gen_cost:.2f}"
            f" + train ${train_cost:.2f})"
        )

    print(f"  worst case (every rollout hits max_tokens={cfg.max_tokens}): "
          f"{worst:>12,} tok{dollars(cfg.max_tokens)}")
    if avg_completion_tokens is not None:
        likely = rollouts * (avg_prompt_tokens + avg_completion_tokens)
        print(f"  likely (SFT policy mean completion {avg_completion_tokens} tok): "
              f"{likely:>12,} tok{dollars(avg_completion_tokens)}")
    if cfg.sample_price_per_m_tokens is None:
        return worst, None
    return worst, worst / 1_000_000 * cfg.sample_price_per_m_tokens


async def probe_headroom(cfg: Cfg, data_path: Path) -> None:
    """PAID pre-flight: does the SFT policy actually disagree with itself?

    Every other check in this file is $0 and validates plumbing. This one
    validates the premise, and cannot be done for free because it needs real
    samples from the real policy on the real GRPO row pool.

    Why it matters more than any plumbing check: GRPO's gradient comes entirely
    from reward *variance inside a group*. compute_advantages centres each
    group's rewards and does not divide by std (rl/data_processing.py:39), so a
    group whose `group_size` rollouts all agree contributes nothing -- you paid
    to sample it and learned zero. The two default categories were chosen on
    cost (ROADMAP Step 7), and the SFT checkpoint's own eval says numeral is at
    90.1% and equation_numeric at 9.3% on real train.csv rows: one near the
    ceiling, one near the floor, both the shapes where a group of 8 agrees.
    (Caveat: that eval is on train.csv, while GRPO rolls out on synth_sft.jsonl
    rows, which are in-distribution for SFT and will score better -- the
    direction is indicative, the magnitude is not. Hence measuring rather than
    extrapolating.)

    Cost at the defaults: 2 categories x 30 prompts x 8 samples ~= 480 rollouts,
    ~1.1M sampling tokens -- roughly 3% of a full run.

    DECISION RULE, stated before the numbers so this is a gate and not just
    another table: if fewer than ~20% of groups are mixed (some rollouts right,
    some wrong) for a category, do not run GRPO on it. Switch to a category with
    real headroom -- unit_conversion (60.8%) or gravity (41.4%) -- and accept the
    higher per-step cost. A cheap run that teaches nothing costs more than an
    expensive run that teaches something.
    """
    _dataset, renderer, _tokenizer, think_prefix, rows = await build_dataset_and_renderer(cfg)
    sampler_path = latest_sft_sampler_path(
        Path(cfg.sft_checkpoints_file)
        if Path(cfg.sft_checkpoints_file).is_absolute()
        else _ROOT / cfg.sft_checkpoints_file
    )
    print(f"\nSampling from SFT weights at {sampler_path}")
    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(model_path=sampler_path)
    params = tinker.types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,  # must match the real run, not 0.0
        stop=renderer.get_stop_sequences(),
    )
    convo_prefix = [{"role": "system", "content": local.SYSTEM_PROMPT}]
    rng = random.Random(cfg.seed)

    n_per_cat = cfg.headroom_prompts_per_category
    print(f"\n=== iteration-0 headroom probe (PAID, ~{n_per_cat * len(cfg.categories) * cfg.group_size} rollouts) ===")
    for cat in cfg.categories:
        pool = [r for r in rows if r["category"] == cat]
        if not pool:
            continue
        all_correct = all_wrong = mixed = 0
        spreads: list[float] = []
        lengths: list[int] = []
        n_trunc = 0
        for row in rng.sample(pool, min(n_per_cat, len(pool))):

            def make_env(row: dict = row, cat: str = cat) -> ReasonerEnv:
                # Env is single-use: one per sampled sequence, not one per group.
                return ReasonerEnv(
                    renderer=renderer,
                    convo_prefix=convo_prefix,
                    question_prompt=row["prompt"],
                    reward_fn=REWARDS[cat],
                    examples=row["examples_tuples"],
                    expected_answer=row["answer"],
                    category=cat,
                    think_prefix=think_prefix,
                    reward_clip=cfg.reward_clip,
                )

            obs, _stop = await make_env().initial_observation()
            resp = await sampling_client.sample_async(
                prompt=obs, num_samples=cfg.group_size, sampling_params=params
            )
            rewards: list[float] = []
            corrects: list[float] = []
            for seq in resp.sequences:
                res = await make_env().step(
                    list(seq.tokens), extra={"stop_reason": seq.stop_reason}
                )
                rewards.append(res.reward)
                corrects.append(res.metrics["correct"])
                lengths.append(len(seq.tokens))
                n_trunc += int(res.metrics["truncated"])
            spreads.append(max(rewards) - min(rewards))
            k = sum(corrects)
            if k == len(corrects):
                all_correct += 1
            elif k == 0:
                all_wrong += 1
            else:
                mixed += 1

        n = all_correct + all_wrong + mixed
        if not n:
            continue
        lengths.sort()
        # `remove_constant_reward_groups` (train.py:1822, applied BEFORE the
        # train step at :1824) keys on *exact* reward equality, not on
        # mixed/all-correct/all-wrong. With a continuous-ish reward an
        # all-wrong group still varies on process quality and so is NOT
        # dropped -- it is billed on the train meter while teaching nothing
        # about correctness. This fraction is the only thing that discounts
        # train-meter tokens below sample-meter tokens, so measure it rather
        # than assuming the filter fires often.
        n_constant = sum(s == 0.0 for s in spreads)
        print(
            f"{cat:<18} groups={n:>3} mixed={mixed / n:>5.0%} all_correct={all_correct / n:>5.0%} "
            f"all_wrong={all_wrong / n:>5.0%} constant_reward={n_constant / n:>5.0%} "
            f"median_reward_spread={statistics.median(spreads):.3f} "
            f"completion_p95={lengths[int(0.95 * len(lengths))]} "
            f"max={lengths[-1]} truncated={n_trunc / len(lengths):.0%}"
        )
        if mixed / n < 0.20:
            print(
                f"    VERDICT: fewer than 20% of {cat} groups carry any gradient. GRPO here "
                "will mostly buy nothing -- switch to a category with real headroom "
                "(unit_conversion 60.8%, gravity 41.4% at SFT) and raise --max-tokens."
            )
        else:
            print(f"    VERDICT: {mixed / n:.0%} of {cat} groups carry gradient -- worth training.")


# --------------------------------------------------------------------------
# CLI / entrypoint
# --------------------------------------------------------------------------


def cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="run pre-flight checks only, no network call")
    parser.add_argument("--yes", action="store_true", help="skip the cost-confirmation prompt")
    parser.add_argument(
        "--probe-headroom",
        action="store_true",
        help="PAID (~1.1M tokens): sample the SFT policy on the real GRPO rows and report "
        "how many groups actually disagree, then exit without training",
    )
    parser.add_argument("--headroom-prompts", type=int, default=None, help="prompts/category for --probe-headroom (default 30)")
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="continue the newest grpo_tinker_runs/ dir that has checkpoints (sets --log-dir-behavior resume)",
    )
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
    if ns.headroom_prompts is not None:
        cfg.headroom_prompts_per_category = ns.headroom_prompts
    if ns.log_dir:
        cfg.log_dir = ns.log_dir
    if ns.log_dir_behavior:
        cfg.log_dir_behavior = ns.log_dir_behavior
    if ns.resume_latest:
        # rl/train.py already resumes: it reads the last checkpoint out of
        # config.log_path and picks start_batch up from it (train.py:1924). The
        # feature was simply unreachable, because Cfg.log_dir defaults to a
        # fresh timestamp -- so a re-run after a crash made a new empty dir,
        # found no checkpoint, and paid from step 0 again.
        runs = sorted(
            (_ROOT / "grpo_tinker_runs").glob("*/checkpoints.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        if not runs:
            raise SystemExit("--resume-latest: no run under grpo_tinker_runs/ has checkpoints yet.")
        cfg.log_dir = runs[-1].parent.name
        cfg.log_dir_behavior = "resume"
        print(f"--resume-latest: continuing {cfg.log_dir}")

    for unknown in set(cfg.categories) - set(REWARDS):
        raise ValueError(f"unknown/unsupported category {unknown!r}; choose from {sorted(REWARDS)}")

    data_path = _data_path(cfg)

    print("Building dataset + renderer for pre-flight checks ...")
    dataset, renderer, tokenizer, think_prefix, rows = asyncio.run(build_dataset_and_renderer(cfg))
    print(f"think_prefix={think_prefix!r}")
    print(f"{len(rows)} rows available for GRPO across {cfg.categories} (post SFT-holdout subtraction, post parse filter)")

    _check_sft_holdout(cfg, data_path)
    _check_gold_trace_rewards(cfg, data_path, tokenizer, renderer)
    _check_reward_discrimination(cfg, data_path)
    _check_prompt_token_identity(cfg, data_path, tokenizer, renderer)
    _check_trace_length_percentiles(cfg, data_path, tokenizer)
    _check_policy_length_distribution(cfg)
    asyncio.run(_check_env_step_closed_loop(cfg, data_path, tokenizer, renderer, think_prefix))

    avg_prompt_tokens = (
        statistics.mean(
            len(tokenizer(local.render_prompt(tokenizer, r["prompt"]), add_special_tokens=False)["input_ids"])
            for r in rows[: min(50, len(rows))]
        )
        if rows
        else 0
    )
    stats = _policy_length_stats(cfg)
    avg_completion = (
        int(statistics.mean(s["mean"] for s in stats.values())) if stats else None
    )
    print("\n=== cost estimate ===")
    total_tokens, est_cost = estimate_rollout_cost(
        cfg, int(avg_prompt_tokens), len(dataset), avg_completion
    )
    if est_cost is None:
        print(
            f"\nEstimated sample-meter tokens: {total_tokens:,} "
            "(no sampling price on file -- check "
            "https://tinker-docs.thinkingmachines.ai/tinker/models/ and pass "
            "--sample-price-per-m-tokens yourself)"
        )
    elif cfg.max_steps is None:
        print(
            "\nThis is a FULL EPOCH. Bound it with --max-steps to spend a fixed "
            "amount -- the single-epoch property that keeps the `correct` metric "
            "honest survives capping steps (each row is still seen at most once)."
        )

    if ns.dry_run:
        return

    if ns.probe_headroom:
        n = cfg.headroom_prompts_per_category * len(cfg.categories) * cfg.group_size
        probe_tokens = n * (int(avg_prompt_tokens) + (avg_completion or cfg.max_tokens))
        # Sample meter only -- the probe never calls forward_backward.
        probe_gen = n * (avg_completion or cfg.max_tokens)
        probe_cost = (
            f" = ~${(n * int(avg_prompt_tokens) * cfg.prefill_price_per_m_tokens + probe_gen * cfg.sample_price_per_m_tokens) / 1_000_000:.2f}"
            " -- sample meter only, no training"
            if cfg.sample_price_per_m_tokens is not None
            else ""
        )
        print(
            f"\n--probe-headroom will sample ~{n} rollouts (~{probe_tokens:,} tokens"
            f"{probe_cost}) and then EXIT without training."
        )
        if not ns.yes:
            reply = input("Run the (paid) headroom probe? [y/N] ").strip().lower()
            if reply != "y":
                print("Aborted, nothing charged.")
                return
        asyncio.run(probe_headroom(cfg, data_path))
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

    # Print the recovery command up front: rl/train.py resumes from the last
    # checkpoint in log_path (train.py:1924), but only if a re-run is pointed at
    # the same directory -- and cfg.log_dir defaults to a fresh timestamp. If
    # this run dies at 3am, the way back is then on screen in the log.
    print(
        "If this run is interrupted, resume it with:\n"
        f"  uv run python kaggle/train_grpo_tinker.py --yes --log-dir {cfg.log_dir} "
        "--log-dir-behavior resume\n"
        "  (or --resume-latest, which finds the newest run with checkpoints)"
    )

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
