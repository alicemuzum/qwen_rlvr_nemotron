"""Synthetic bit_manipulation problem generator + reward monitor.

Mirrors monitor_equation_numeric.py: build the problem top-down from a fully
known ground-truth per-bit rule vector (one rule per output bit, picked
first), derive every example output and the question's answer from that
vector, then run the deterministic generator afterward to produce a
demonstration trace.

bit_manipulation.py's generator now follows the foolproof contract (gate
added, see its module docstring): it re-derives a per-bit rule vector from
only the visible examples, and since a handful of examples don't always pin
every bit's rule uniquely (the same ambiguity cryptarithm/equation_numeric
document), that re-derived rule can disagree with our synthetic ground truth
on the question even though it reproduces every example. The generator
itself now returns None in that case rather than emit an unverified trace,
so this monitor just retries like every other category's monitor -- no
extra boxed-answer re-check needed here.
"""

from __future__ import annotations

import random
import sys

from reasoners.bit_manipulation import (
    N_BITS,
    evaluate_rule_expr,
    reasoning_bit_manipulation,
)
from reasoners.reward_bit_manipulation import evaluate_structured_trace
from reasoners.store_types import Example, Problem

# Rule-vector shape, measured from the solver's own derived vectors on 338 real
# train.csv rows it verifiably solves (parsed out of the emitted execution step,
# so these are rules that actually reproduce the hidden answer, not guesses):
#
#   consecutive same-family runs per problem   1:31  2:149  3:157  4:1
#   within a run, (primary - bit_index) % 8    constant in 526/533 (98.7%)
#   within a run, (secondary - bit_index) % 8  constant in 258/260 (99.2%)
#   family, counted per output bit             I 1126, const 356, AND 268,
#                                              XOR 253, XOR-NOT 192, OR 175,
#                                              OR-NOT 157, NOT 94, AND-NOT 83
#
# That constant-difference property is the whole game: a real problem is a
# handful of consecutive bit runs where the *input* position advances in
# lockstep with the output position, which is exactly the left/right stride-run
# extrapolation `bit_manipulation.py` searches for. Drawing each bit's rule
# independently (what this module used to do) produces vectors with no such
# progression, so the solver could re-derive the rule on only ~6.5% of them
# versus 84.6% of real rows -- and the survivors were precisely the random
# draws that happened to look structured, making the kept corpus doubly
# unrepresentative. Sampling runs instead brings the yield to ~73%.
_RUN_COUNTS = (1, 2, 3)
# NOT the raw real frequencies (31/149/158). The foolproof gate does not discard
# uniformly: measured survival is 100% / 97% / 42% by run count, because a
# 3-run vector gives each run fewer bits to pin its family and offset from, so
# the solver more often lands on a different-but-example-consistent rule. Left
# uncompensated, the *kept* corpus came out 27/58/15 against a real 9/44/46 --
# i.e. systematically easier than the real task, the same selection bias that
# made the old sampler's surviving 6.5% unrepresentative. These weights are the
# real frequencies divided by those survival rates, so the rows that actually
# reach the corpus match real train.csv. Costs overall yield (~74% -> ~61%),
# which is the right trade: generation is free, biased training data is not.
_RUN_WEIGHTS = (6, 28, 67)
_RULE_FAMILIES = ("I", "C", "AND", "XOR", "XOR-NOT", "OR", "OR-NOT", "NOT", "AND-NOT")
# Counted per output *bit*, but consumed per *run* (one family is drawn and then
# painted across the run's whole span), so the realized per-bit mix is not this
# vector by construction. It was checked empirically instead and lands within
# 1-3pp of real on all nine families -- don't "fix" the apparent unit mismatch
# without re-measuring, it would break a match that currently holds.
_RULE_FAMILY_WEIGHTS = (1126, 356, 268, 253, 192, 175, 157, 94, 83)
# Real prompts carry 7-10 examples (measured: 116/114/139/131 over 500 rows);
# a fixed count narrowed the format distribution the model sees for no reason.
_EXAMPLE_COUNTS = (7, 8, 9, 10)


def _random_rule_vector(rng: random.Random) -> list[str]:
    """One rule expression per output bit, as consecutive stride-+1 runs.

    Each run picks a family and a fixed offset, then walks both the primary and
    secondary input positions forward with the output position.
    """
    n_runs = rng.choices(_RUN_COUNTS, weights=_RUN_WEIGHTS)[0]
    cuts = sorted(rng.sample(range(1, N_BITS), n_runs - 1))
    bounds = [0, *cuts, N_BITS]

    exprs = [""] * N_BITS
    for run in range(n_runs):
        fam = rng.choices(_RULE_FAMILIES, weights=_RULE_FAMILY_WEIGHTS)[0]
        span = range(bounds[run], bounds[run + 1])
        if fam == "C":
            const = rng.choice(["C0", "C1"])
            for i in span:
                exprs[i] = const
            continue
        primary_offset = rng.randrange(N_BITS)
        if fam in ("I", "NOT"):
            for i in span:
                exprs[i] = f"{fam}{(primary_offset + i) % N_BITS}"
            continue
        secondary_offset = rng.randrange(N_BITS)
        while secondary_offset == primary_offset:
            secondary_offset = rng.randrange(N_BITS)
        for i in span:
            a = (primary_offset + i) % N_BITS
            b = (secondary_offset + i) % N_BITS
            exprs[i] = f"{fam}{a}{b}"
    return exprs


def _apply_true_rule(bits: str, rule_exprs: list[str]) -> str:
    return "".join(evaluate_rule_expr(bits, expr) for expr in rule_exprs)


def _random_bits(rng: random.Random) -> str:
    return "".join(rng.choice("01") for _ in range(N_BITS))


def make_synthetic_problem(
    rng: random.Random, num_examples: int | None = None
) -> Problem:
    if num_examples is None:
        num_examples = rng.choice(_EXAMPLE_COUNTS)
    rule_exprs = _random_rule_vector(rng)

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
        if candidate_trace is not None:
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
