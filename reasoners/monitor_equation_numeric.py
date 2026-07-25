"""Synthetic equation_numeric problem generator + reward monitor.

Mirrors monitor_cryptarithm.py / monitor_gravity.py: build the problem
top-down from a fully known ground-truth rule (operation + reverse flags +
sign-carrying format, all picked first), derive every example output and the
question's answer from that rule via the shared ``apply_rule``, then run the
deterministic generator afterward to produce a demonstration trace.

Expect a lower solve rate than numeral's ~100%: a handful of examples don't
always pin the operation uniquely (the same ambiguity cryptarithm documents),
so the generator sometimes re-derives a *different* valid rule whose answer
disagrees with ours, and its foolproof check returns None. That is correct
behavior, not a bug -- the retry loop absorbs it, same as cryptarithm/gravity.
"""

from __future__ import annotations

import random
import sys

from reasoners.equation_numeric import apply_rule, reasoning_equation_numeric
from reasoners.reward_equation_numeric import evaluate_structured_trace
from reasoners.store_types import Example, Problem

# Operator symbols are cosmetic (the operation is fixed by the rule, not the
# symbol). Avoid digits and '-' (the latter has special-cased sign detection).
_OP_SYMBOLS = list("/|\\*{}~`><^%!@#&")

# Non-negative operations render cleanly with fmt="num".
_CLEAN_OPS = [
    "addition",
    "absolute difference",
    "multiplication",
    "concatenation",
    "reverse concatenation",
]
# Signed operations exercise the neg_suffix/neg_prefix format path.
_SIGNED_OPS = ["subtraction (a-b)", "reverse subtraction (b-a)"]


def make_synthetic_problem(rng: random.Random, num_examples: int = 4) -> Problem | None:
    op_char = rng.choice(_OP_SYMBOLS)
    signed = rng.random() < 0.35
    if signed:
        op_name = rng.choice(_SIGNED_OPS)
        fmt = rng.choice(["neg_suffix", "neg_prefix"])
    else:
        op_name = rng.choice(_CLEAN_OPS)
        fmt = "num"
    rev_ops = rng.random() < 0.5
    rev_res = rng.random() < 0.5

    # Distinct 2-digit operand pairs (matches real train.csv), one per example
    # plus the question. Distinctness matters -- the reward keys examples by
    # their input string (same reason numeral draws distinct integers).
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < num_examples + 1:
        pairs.add((rng.randint(10, 99), rng.randint(10, 99)))
    pair_list = list(pairs)

    def out_for(a: int, b: int) -> str | None:
        return apply_rule(op_name, rev_ops, rev_res, fmt, op_char, str(a), str(b))

    examples: list[Example] = []
    for a, b in pair_list[:num_examples]:
        out = out_for(a, b)
        if out is None:
            return None
        examples.append(Example(f"{a}{op_char}{b}", out))

    qa, qb = pair_list[num_examples]
    answer = out_for(qa, qb)
    if answer is None:
        return None

    # For the signed format to be detectable, at least one example output must
    # actually carry the sign symbol; otherwise the generator sees plain
    # numbers and infers fmt="num", which is a different (still-valid) rule.
    if signed and not any(
        ex.output_value.endswith(op_char) or ex.output_value.startswith(op_char)
        for ex in examples
    ):
        return None

    return Problem(
        id="synthetic",
        category="equation_numeric_deduce",
        examples=examples,
        question=f"{qa}{op_char}{qb}",
        answer=answer,
    )


def main() -> None:
    rng = random.Random()
    problem = None
    trace = None
    for _ in range(400):
        candidate = make_synthetic_problem(rng)
        if candidate is None:
            continue
        candidate_trace = reasoning_equation_numeric(candidate)
        if candidate_trace is not None:
            problem, trace = candidate, candidate_trace
            break

    if problem is None or trace is None:
        print("Could not construct a solvable synthetic problem in 400 tries.")
        sys.exit(1)

    print("--- Problem ---")
    for ex in problem.examples:
        print(f"  {ex.input_value} = {ex.output_value}")
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
