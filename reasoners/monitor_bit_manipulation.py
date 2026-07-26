"""Synthetic bit_manipulation problem generator + reward monitor.

Mirrors monitor_equation_numeric.py: build the problem top-down from a fully
known ground-truth per-bit rule vector (one rule per output bit, picked
first), derive every example output and the question's answer from that
vector, then run the deterministic generator afterward to produce a
demonstration trace.

bit_manipulation.py's generator does not itself re-verify its derived answer
against a known ground truth before returning (see its module docstring and
CLAUDE.md's "Current gaps" -- this is the one generator that still violates
the foolproof contract other tasks' generators follow). Ambiguity in which
per-bit rule the visible examples pin down means the generator can
confidently re-derive a *different*, still example-consistent rule that
disagrees with our synthetic ground truth on the question. Rather than
change the generator (out of scope here), this monitor applies the same
foolproof guard inline: construct, run the generator, and discard/retry
whenever the emitted boxed answer doesn't match the known answer -- the same
retry loop cryptarithm/gravity/equation_numeric use to absorb their own
generators' ambiguity misses.
"""

from __future__ import annotations

import random
import re
import sys

from reasoners.bit_manipulation import (
    N_BITS,
    evaluate_rule_expr,
    reasoning_bit_manipulation,
)
from reasoners.reward_bit_manipulation import evaluate_structured_trace
from reasoners.store_types import Example, Problem

_PAIR_FAMILIES = ("XOR", "OR", "AND", "AND-NOT", "XOR-NOT", "OR-NOT")
_BOXED_RE = re.compile(r"\\boxed\{([01]+)\}")


def _random_rule_expr(rng: random.Random) -> str:
    choice = rng.random()
    if choice < 0.15:
        return rng.choice(["C0", "C1"])
    if choice < 0.35:
        fam = rng.choice(("I", "NOT"))
        p = rng.randrange(N_BITS)
        return f"{fam}{p}"
    fam = rng.choice(_PAIR_FAMILIES)
    a = rng.randrange(N_BITS)
    b = rng.randrange(N_BITS)
    while b == a:
        b = rng.randrange(N_BITS)
    return f"{fam}{a}{b}"


def _apply_true_rule(bits: str, rule_exprs: list[str]) -> str:
    return "".join(evaluate_rule_expr(bits, expr) for expr in rule_exprs)


def _random_bits(rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(N_BITS))


def make_synthetic_problem(rng: random.Random, num_examples: int = 8) -> Problem:
    rule_exprs = [_random_rule_expr(rng) for _ in range(N_BITS)]

    distinct: set[str] = set()
    while len(distinct) < num_examples + 1:
        distinct.add(_random_bits(rng))
    bit_strings = list(distinct)

    examples = [
        Example(bits, _apply_true_rule(bits, rule_exprs)) for bits in bit_strings[:num_examples]
    ]
    question = bit_strings[num_examples]
    answer = _apply_true_rule(question, rule_exprs)

    return Problem(
        id="synthetic",
        category="bit_manipulation",
        examples=examples,
        question=question,
        answer=answer,
    )


def main() -> None:
    rng = random.Random()
    problem = None
    trace = None
    for _ in range(400):
        candidate = make_synthetic_problem(rng)
        candidate_trace = reasoning_bit_manipulation(candidate)
        if candidate_trace is None:
            continue
        matches = _BOXED_RE.findall(candidate_trace)
        if matches and matches[-1] == candidate.answer:
            problem, trace = candidate, candidate_trace
            break

    if problem is None or trace is None:
        print("Could not construct a solvable synthetic problem in 400 tries.")
        sys.exit(1)

    print("--- Problem ---")
    for ex in problem.examples:
        print(f"  {ex.input_value} -> {ex.output_value}")
    print(f"  Q: {problem.question}  (answer {problem.answer!r})")
    print("\n--- Evaluating trace... ---\n")

    examples = [(ex.input_value, ex.output_value) for ex in problem.examples]
    total_reward, step_logs = evaluate_structured_trace(trace, examples, problem.answer)

    print(f"{'Step Type':<16} | {'Delta':<8} | {'Total':<8} | {'Reason':<52} | Content")
    print("-" * 140)
    for log in step_logs:
        tag = log["tag_type"]
        content = log["content"].replace("\n", " ")
        if len(content) > 34:
            content = content[:31] + "..."
        delta = log["reward_delta"]
        total = log["total_reward"]
        reason = log["reason"]
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        if delta > 0:
            delta_str = f"\033[92m{delta_str}\033[0m"
        elif delta < 0:
            delta_str = f"\033[91m{delta_str}\033[0m"
        else:
            delta_str = f"\033[90m{delta_str}\033[0m"
        print(f"{tag:<16} | {delta_str:<17} | {total:<8.2f} | {reason:<52} | {content}")

    print("-" * 140)
    print(f"FINAL REWARD: {total_reward:.2f}")


if __name__ == "__main__":
    main()
