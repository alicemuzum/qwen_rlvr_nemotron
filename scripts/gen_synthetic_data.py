"""Bulk synthetic SFT-data generator across all 7 reasoning categories.

Stage-1 data generation (see CLAUDE.md "Project goal"): construct problems
top-down from a fully known ground truth, run each category's deterministic
gold-trace generator, keep only verified traces, and write a JSONL dataset of
``{id, category, prompt, question, answer, trace}`` rows for SFT.

Design decisions (all from CLAUDE.md / the advisor review):

* The synthetic *problem* constructors are imported, not reimplemented, from
  each ``monitor_<task>.py`` (importing a monitor runs only its top-level code,
  not ``main()``), so this stays bit-identical to the monitors -- the
  must-not-drift discipline the codebase repeats everywhere. Cipher is the one
  exception (it builds inline in ``monitor_cipher.main`` with no reusable
  function), so it's assembled here from the same pieces the monitor uses.

* Per-category verification is intentionally *not* uniform:
    - The 5 foolproof generators (cryptarithm, gravity, numeral,
      unit_conversion, equation_numeric) already guarantee ``boxed == answer``
      whenever they return non-None, so a non-None trace is accepted as-is.
      We never re-parse their boxed answer -- that sidesteps cryptarithm's
      literal ``{``/``}`` symbols and unit_conversion's trailing-zero format
      differences.
    - cipher and bit_manipulation have no foolproof contract, so their traces
      are accepted only if the last ``\\boxed{...}`` equals ``problem.answer``.
      Both answer types are brace-free (plaintext words / pure binary), so a
      naive boxed regex is safe *for these two only*.

* Prompts match the real ``train.csv`` phrasing (the actual task-time
  distribution). NOTE: the kaggle GRPO stage uses a different prompt shape
  (system one-shot demo + ``Examples:/Question:`` body), so SFT-on-this ->
  GRPO-on-kaggle is a cross-stage prompt shift; align them later if desired.

* Every ``reasoning_<task>`` generator now wraps its emitted trace in a
  single ``<think>...</think>`` block (via ``store_types.wrap_trace_with_think``),
  with the final ``\\boxed{}`` line duplicated outside the tags as the
  model's visible answer -- see that helper's docstring. This script does
  not need to do any of that wrapping itself; it just writes whatever
  ``reasoning_<task>`` already returns.

* Rows whose ``prompt + trace`` exceeds 8192 Qwen2.5 tokens are **dropped
  during generation, not truncated** (see ``_token_count``/``_MAX_TOKENS``)
  -- matching CLAUDE.md's "SFT readiness audit" finding that truncation cuts
  off exactly the trailing \\boxed{} line stage-2 SFT exists to teach.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import uuid
from functools import lru_cache
from pathlib import Path

from transformers import AutoTokenizer

from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.cipher import _load_wonderland, reasoning_cipher
from reasoners.cryptarithm import reasoning_cryptarithm
from reasoners.equation_numeric import reasoning_equation_numeric
from reasoners.gravity import reasoning_gravity
from reasoners.monitor_bit_manipulation import make_synthetic_problem as _mk_bit
from reasoners.monitor_cryptarithm import make_synthetic_problem as _mk_cryptarithm
from reasoners.monitor_equation_numeric import make_synthetic_problem as _mk_equation
from reasoners.monitor_gravity import make_synthetic_problem as _mk_gravity
from reasoners.monitor_numeral import make_synthetic_problem as _mk_numeral
from reasoners.monitor_unit_conversion import make_synthetic_problem as _mk_unit
from reasoners.numeral import reasoning_numeral
from reasoners.store_types import Example, Problem
from reasoners.unit_conversion import reasoning_unit_conversion

CATEGORIES = [
    "bit_manipulation",
    "cipher",
    "cryptarithm",
    "equation_numeric",
    "gravity",
    "numeral",
    "unit_conversion",
]

# Categories whose generator honours the foolproof contract (non-None => the
# emitted boxed answer already equals problem.answer). The other two (cipher,
# bit_manipulation) need an explicit boxed==answer check.
_FOOLPROOF = {
    "cryptarithm",
    "equation_numeric",
    "gravity",
    "numeral",
    "unit_conversion",
}

_BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)

# Rows over this many Qwen2.5 tokens (prompt + trace) are dropped, not
# truncated -- truncation would cut off the trailing \boxed{} line stage-2
# SFT exists to teach (see CLAUDE.md's "SFT readiness audit" note on why
# dropping over-length rows beats truncating them).
_MAX_TOKENS = 8192
_TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=1)
def _tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(_TOKENIZER_NAME)


def _token_count(prompt: str, trace: str) -> int:
    return len(_tokenizer()(prompt + "\n" + trace).input_ids)


# --------------------------------------------------------------------------
# Problem construction -- each returns a Problem or None (retry on None)
# --------------------------------------------------------------------------


def _make_cipher_problem(rng: random.Random) -> Problem:
    """Assemble a cipher Problem the way monitor_cipher.main does, but seedable.

    monitor_cipher builds this inline with the global ``random``; we replicate
    the pieces here with the threaded rng so the whole run is reproducible.
    """
    words = _load_wonderland()
    alphabet = list(string.ascii_lowercase)
    shuffled = list(alphabet)
    rng.shuffle(shuffled)
    plain_to_cipher = dict(zip(alphabet, shuffled))

    def encrypt(text: str) -> str:
        return "".join(plain_to_cipher.get(c, c) for c in text)

    examples = []
    for _ in range(4):
        n_words = rng.randint(2, 4)
        plain = " ".join(rng.sample(words, n_words))
        examples.append(Example(encrypt(plain), plain))

    question_plain = " ".join(rng.sample(words, rng.randint(3, 5)))
    return Problem(
        id="synthetic",
        category="cipher",
        examples=examples,
        question=encrypt(question_plain),
        answer=question_plain,
    )


def _construct(category: str, rng: random.Random) -> Problem | None:
    if category == "bit_manipulation":
        return _mk_bit(rng)
    if category == "cipher":
        return _make_cipher_problem(rng)
    if category == "cryptarithm":
        built = _mk_cryptarithm(rng)
        return built[0] if built else None
    if category == "equation_numeric":
        return _mk_equation(rng)
    if category == "gravity":
        built = _mk_gravity(rng)
        return built[0] if built else None
    if category == "numeral":
        return _mk_numeral(rng)
    if category == "unit_conversion":
        built = _mk_unit(rng)
        return built[0] if built else None
    raise ValueError(f"unknown category {category!r}")


_REASONERS = {
    "bit_manipulation": reasoning_bit_manipulation,
    "cipher": reasoning_cipher,
    "cryptarithm": reasoning_cryptarithm,
    "equation_numeric": reasoning_equation_numeric,
    "gravity": reasoning_gravity,
    "numeral": reasoning_numeral,
    "unit_conversion": reasoning_unit_conversion,
}


# --------------------------------------------------------------------------
# Prompt building -- match real train.csv phrasing per category
# --------------------------------------------------------------------------


def _build_prompt(category: str, p: Problem) -> str:
    ex = p.examples
    if category == "bit_manipulation":
        lines = "\n".join(f"{e.input_value} -> {e.output_value}" for e in ex)
        return (
            "In Alice's Wonderland, a secret bit manipulation rule transforms "
            "8-bit binary numbers. The transformation involves operations like "
            "bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or "
            "choice functions.\n\n"
            "Here are some examples of input -> output:\n"
            f"{lines}\n\n"
            f"Now, determine the output for: {p.question}"
        )
    if category == "cipher":
        lines = "\n".join(f"{e.input_value} -> {e.output_value}" for e in ex)
        return (
            "In Alice's Wonderland, secret encryption rules are used on text. "
            "Here are some examples:\n"
            f"{lines}\n"
            f"Now, decrypt the following text: {p.question}"
        )
    if category == "numeral":
        lines = "\n".join(f"{e.input_value} -> {e.output_value}" for e in ex)
        return (
            "In Alice's Wonderland, numbers are secretly converted into a "
            "different numeral system. Some examples are given below:\n"
            f"{lines}\n"
            f"Now, write the number {p.question} in the Wonderland numeral system."
        )
    if category == "unit_conversion":
        lines = "\n".join(f"{e.input_value} m becomes {e.output_value}" for e in ex)
        return (
            "In Alice's Wonderland, a secret unit conversion is applied to "
            "measurements. For example:\n"
            f"{lines}\n"
            f"Now, convert the following measurement: {p.question} m"
        )
    if category == "gravity":
        lines = "\n".join(
            f"For t = {e.input_value}s, distance = {e.output_value} m" for e in ex
        )
        return (
            "In Alice's Wonderland, the gravitational constant has been secretly "
            "changed. Here are some example observations:\n"
            f"{lines}\n"
            f"Now, determine the falling distance for t = {p.question}s given "
            "d = 0.5*g*t^2."
        )
    if category in ("cryptarithm", "equation_numeric"):
        lines = "\n".join(f"{e.input_value} = {e.output_value}" for e in ex)
        return (
            "In Alice's Wonderland, a secret set of transformation rules is "
            "applied to equations. Below are a few examples:\n"
            f"{lines}\n"
            f"Now, determine the result for: {p.question}"
        )
    raise ValueError(f"unknown category {category!r}")


def _verify(category: str, p: Problem, trace: str | None) -> bool:
    if trace is None:
        return False
    if category in _FOOLPROOF:
        return True
    # cipher / bit_manipulation: no foolproof contract -> check boxed==answer.
    matches = _BOXED_RE.findall(trace)
    return bool(matches) and matches[-1] == p.answer


# --------------------------------------------------------------------------
# Collection driver
# --------------------------------------------------------------------------


def collect(
    category: str, n: int, rng: random.Random, attempts_per_row: int = 300
) -> tuple[list[dict], int, int]:
    rows: list[dict] = []
    attempts = 0
    dropped_long = 0
    cap = max(n * attempts_per_row, 2000)
    while len(rows) < n and attempts < cap:
        attempts += 1
        problem = _construct(category, rng)
        if problem is None:
            continue
        trace = _REASONERS[category](problem)
        if not _verify(category, problem, trace):
            continue
        prompt = _build_prompt(category, problem)
        if _token_count(prompt, trace) > _MAX_TOKENS:
            dropped_long += 1
            continue
        rows.append(
            {
                "id": uuid.uuid4().hex[:8],
                "category": category,
                "prompt": prompt,
                "question": problem.question,
                "answer": problem.answer,
                "trace": trace,
            }
        )
    return rows, attempts, dropped_long


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-category", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("reasoners_synthetic_sft.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--categories",
        nargs="+",
        default=CATEGORIES,
        choices=CATEGORIES,
        help="subset of categories to generate (default: all 7)",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_rows: list[dict] = []
    print(f"{'category':<18} | {'rows':>6} | {'attempts':>8} | {'>8192tok':>8} | yield")
    print("-" * 68)
    for category in args.categories:
        rows, attempts, dropped_long = collect(category, args.per_category, rng)
        all_rows.extend(rows)
        yield_pct = (len(rows) / attempts * 100) if attempts else 0.0
        flag = "" if len(rows) == args.per_category else "  <-- short"
        print(
            f"{category:<18} | {len(rows):>6} | {attempts:>8} | {dropped_long:>8} | {yield_pct:5.1f}%{flag}"
        )

    with args.out.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("-" * 68)
    print(f"wrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
