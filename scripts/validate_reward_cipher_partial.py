"""Validation for reasoners/reward_cipher_partial.py, following this repo's
standard reward-function validation pattern (see CLAUDE.md's "Validating
reward functions"): generate many synthetic gold traces via the same
bijection-first construction monitor_cipher.py uses, assert none score
negative, then hand-check known adversarial trace shapes.

This additionally checks two things specific to this file, since it changes
reasoners/reward_cipher.py's scoring in two independent ways (see its module
docstring):
  1. Full-oracle parity: with a COMPLETE oracle, gold-trace scores should
     match reward_cipher.py's original almost everywhere -- the one allowed
     divergence is the dashed-reconstruction adjacency fix (see below), which
     only ever REMOVES a spurious penalty, never adds one relative to the
     original.
  2. The reward correctly separates a correct trace from corrupted variants
     even when the oracle is only reconstructed from the prompt's own visible
     examples (not the full generating bijection) -- this is the property
     kaggle/train_grpo_tinker.py's own _check_reward_discrimination depends
     on for cipher to be GRPO-usable at all.

Run with: uv run python scripts/validate_reward_cipher_partial.py
"""

from __future__ import annotations

import random
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoners.cipher import reasoning_cipher
from reasoners.reward_cipher import evaluate_structured_trace as reward_full_original
from reasoners.reward_cipher_partial import (
    evaluate_structured_trace as reward_partial,
)
from reasoners.reward_cipher_partial import (
    evaluate_structured_trace_from_examples,
    partial_oracle_from_examples,
)
from reasoners.store_types import Example, Problem

_WONDERLAND_PATH = Path(__file__).resolve().parent.parent / "reasoners" / "wonderland.txt"


def _load_words() -> list[str]:
    with _WONDERLAND_PATH.open() as f:
        return [w.strip() for w in f if w.strip()]


def _make_random_cipher(rng: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    alphabet = list(string.ascii_lowercase)
    shuffled = list(alphabet)
    rng.shuffle(shuffled)
    plain_to_cipher = {p: c for p, c in zip(alphabet, shuffled)}
    cipher_to_plain = {c: p for p, c in zip(alphabet, shuffled)}
    return plain_to_cipher, cipher_to_plain


def _encrypt(text: str, plain_to_cipher: dict[str, str]) -> str:
    return "".join(plain_to_cipher.get(c, c) for c in text)


def _build_gold(rng: random.Random, words: list[str]):
    """Mirrors monitor_cipher.py's construction: a full bijection first, then
    real dictionary words encrypted with it -- so the boxed answer is correct
    by construction (not merely trusted from cipher.py's own, ungated,
    output).
    """
    plain_to_cipher, oracle_map = _make_random_cipher(rng)
    example_sentences = [rng.sample(words, 4), rng.sample(words, 3), rng.sample(words, 4)]
    examples: list[Example] = []
    example_tuples: list[tuple[str, str]] = []
    for sent_words in example_sentences:
        plain_text = " ".join(sent_words)
        cipher_text = _encrypt(plain_text, plain_to_cipher)
        examples.append(Example(cipher_text, plain_text))
        example_tuples.append((cipher_text, plain_text))
    question_words = rng.sample(words, 5)
    question_plain = " ".join(question_words)
    question_cipher = _encrypt(question_plain, plain_to_cipher)
    problem = Problem(
        id="validate",
        category="cipher",
        examples=examples,
        question=question_cipher,
        answer=question_plain,
        prompt="",
    )
    trace = reasoning_cipher(problem)
    return trace, oracle_map, question_words, example_tuples, question_plain


def _mutate_wrong_box(trace: str, answer: str) -> str:
    return trace.replace("\\boxed{" + answer + "}", "\\boxed{" + answer + "X}")


def _mutate_no_verification(trace: str) -> str:
    return re.sub(r'<step type="verification">.*?</step>\n?', "", trace, flags=re.DOTALL)


def check_gold_and_parity(n: int = 300) -> bool:
    words = _load_words()
    rng = random.Random(0)

    n_traces = 0
    parity_mismatch = 0
    partial_negative = 0
    adapter_mismatch = 0
    n_partial_had_gap = 0
    n_full_score_regression = 0  # partial-fix score < original on a gold trace (must be 0)

    for _ in range(n):
        trace, oracle_map, question_words, example_tuples, question_plain = _build_gold(rng, words)
        if not trace:
            continue
        m = re.findall(r"\\boxed\{(.*?)\}", trace, re.DOTALL)
        if not m or m[-1].strip() != question_plain:
            continue
        n_traces += 1

        full_original, _ = reward_full_original(trace, oracle_map, question_words)
        full_fixed, _ = reward_partial(trace, oracle_map, question_words)
        if full_fixed < full_original:
            n_full_score_regression += 1
        if full_fixed != full_original:
            parity_mismatch += 1

        recon_oracle = partial_oracle_from_examples(example_tuples)
        if len(recon_oracle) < 26:
            n_partial_had_gap += 1
        partial_score, _ = reward_partial(trace, recon_oracle, question_words)
        if partial_score < 0:
            partial_negative += 1
            print(f"NEGATIVE on partial oracle: {partial_score:.2f}")

        adapter_score, _ = evaluate_structured_trace_from_examples(trace, example_tuples, question_plain)
        if adapter_score != partial_score:
            adapter_mismatch += 1

    print(f"gold traces tested: {n_traces}")
    print(f"full-oracle score differs from reward_cipher.py's original: {parity_mismatch}/{n_traces} "
          f"(expected: the dashed-reconstruction fix only removes spurious penalties, never adds one)")
    print(f"full-oracle score REGRESSED below original (should be 0): {n_full_score_regression}/{n_traces}")
    print(f"rows where the reconstructed (example-only) oracle had gaps: {n_partial_had_gap}/{n_traces}")
    print(f"partial-oracle NEGATIVE scores (should be 0): {partial_negative}/{n_traces}")
    print(f"adapter/direct-call mismatches (should be 0): {adapter_mismatch}/{n_traces}")

    return partial_negative == 0 and adapter_mismatch == 0 and n_full_score_regression == 0


def check_no_verify_discrimination(n: int = 150) -> bool:
    """The specific regression this file's second fix targets: deleting every
    verification step from a gold trace must never score HIGHER than the
    intact trace, whether graded with a full or a reconstructed oracle.
    """
    words = _load_words()
    rng = random.Random(1)
    n_tested = 0
    full_violations = 0
    partial_violations = 0

    for _ in range(n):
        trace, oracle_map, question_words, example_tuples, _question_plain = _build_gold(rng, words)
        if not trace:
            continue
        m = re.findall(r"\\boxed\{(.*?)\}", trace, re.DOTALL)
        if not m:
            continue
        n_tested += 1
        nv_trace = _mutate_no_verification(trace)

        gold_full, _ = reward_partial(trace, oracle_map, question_words)
        nv_full, _ = reward_partial(nv_trace, oracle_map, question_words)
        if nv_full > gold_full:
            full_violations += 1

        recon_oracle = partial_oracle_from_examples(example_tuples)
        gold_partial, _ = reward_partial(trace, recon_oracle, question_words)
        nv_partial, _ = reward_partial(nv_trace, recon_oracle, question_words)
        if nv_partial > gold_partial:
            partial_violations += 1

    print(f"\nno-verification-step discrimination (n={n_tested}):")
    print(f"  full oracle:    no_verify > gold in {full_violations}/{n_tested} (should be 0)")
    print(f"  partial oracle: no_verify > gold in {partial_violations}/{n_tested} (should be 0)")
    return full_violations == 0 and partial_violations == 0


def check_adversarial() -> bool:
    expected_words = ["cat", "dog"]

    def trace_of(*steps: str) -> str:
        body = "\n".join(steps)
        return f"<think>\n{body}\n</think>\n\\boxed{{cat dog}}"

    oracle = {"a": "c", "c": "a", "t": "t"}
    ok = True

    _s, logs = reward_partial(
        trace_of('<step type="state_update">a->q</step>', '<step type="conclusion">\\boxed{cat dog}</step>'),
        oracle, expected_words,
    )
    if not any("P_fatal" in row["reason"] for row in logs):
        print("FAIL: wrong claim on a known oracle letter was not penalized as P_fatal")
        ok = False

    _s, logs = reward_partial(
        trace_of('<step type="state_update">x->q</step>', '<step type="conclusion">\\boxed{cat dog}</step>'),
        oracle, expected_words,
    )
    if logs[0]["reward_delta"] != 0.0:
        print(f"FAIL: first unverifiable claim should be neutral, got {logs[0]['reward_delta']}")
        ok = False

    _s, logs = reward_partial(
        trace_of(
            '<step type="state_update">x->q</step>',
            '<step type="state_update">x->r</step>',
            '<step type="conclusion">\\boxed{cat dog}</step>',
        ),
        oracle, expected_words,
    )
    if logs[1]["reward_delta"] >= 0:
        print(f"FAIL: contradicting an unverifiable claim should be penalized, got {logs[1]['reward_delta']}")
        ok = False

    print(f"\nadversarial checks: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ok = True
    ok &= check_gold_and_parity()
    ok &= check_no_verify_discrimination()
    ok &= check_adversarial()
    print(f"\n{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
