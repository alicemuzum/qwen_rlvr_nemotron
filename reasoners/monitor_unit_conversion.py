"""Synthetic unit_conversion problem generator + reward monitor.

Mirrors monitor_gravity.py's approach: build the problem top-down from a
fully known ground truth (a conversion factor picked first), then run the
deterministic generator to produce a demonstration trace.

Inputs and outputs are rounded to 2dp here, matching the precision real
train.csv rows use (factor range ~0.5-2.0, per CLAUDE.md's measured stats).
This deliberately reintroduces the same per-example rounding noise real
data has (reasoning_unit_conversion's median-of-truncated-divisions
estimate does not always reproduce it exactly) rather than sidestepping it
with full-precision synthetic values -- see the retry loop in main(), which
already tolerates reasoning_unit_conversion returning None on a fraction of
attempts (the same pattern monitor_gravity.py uses).
"""

from __future__ import annotations

import random
import sys

from reasoners.reward_unit_conversion import evaluate_structured_trace
from reasoners.store_types import Example, Problem
from reasoners.unit_conversion import reasoning_unit_conversion


def make_synthetic_problem(
    rng: random.Random, num_examples: int | None = None
) -> tuple[Problem, str] | None:
    """Return (problem, factor_true_str).

    Example input values are drawn distinct (including the question): both
    reasoners/unit_conversion.py's tags and reward_unit_conversion.py's
    parsing identify an example solely by its input value, so a problem
    with a repeated input is not well-formed for this trace format.
    """
    if num_examples is None:
        num_examples = rng.randint(3, 5)

    factor_true = round(rng.uniform(0.5, 2.0), 3)
    factor_str = f"{factor_true:.3f}"

    def make_example(inp: float) -> Example:
        out = round(factor_true * inp, 2)
        return Example(f"{inp:.2f}", f"{out:.2f}")

    inputs = rng.sample(
        [round(i / 100, 2) for i in range(500, 5000)], num_examples + 1
    )
    examples = [make_example(inp) for inp in inputs[:num_examples]]
    q_ex = make_example(inputs[num_examples])

    problem = Problem(
        id="synthetic",
        category="unit_conversion",
        examples=examples,
        question=q_ex.input_value,
        answer=q_ex.output_value,
    )
    return problem, factor_str


def main() -> None:
    rng = random.Random()
    problem = None
    for _ in range(200):
        built = make_synthetic_problem(rng)
        if built is None:
            continue
        candidate_problem, _factor_true = built
        trace = reasoning_unit_conversion(candidate_problem)
        if trace is not None:
            problem = candidate_problem
            break

    if problem is None:
        print("Could not construct a solvable synthetic problem in 200 tries.")
        sys.exit(1)

    print(
        f"--- Generating trace for input = {problem.question} "
        f"(answer {problem.answer}) ---"
    )
    trace = reasoning_unit_conversion(problem)
    assert trace is not None

    print("--- Evaluating trace... ---\n")
    examples = [(ex.input_value, ex.output_value) for ex in problem.examples]
    total_reward, step_logs = evaluate_structured_trace(trace, examples, problem.answer)

    print(f"{'Step Type':<16} | {'Delta':<8} | {'Total':<8} | {'Reason':<50} | Content")
    print("-" * 130)
    for log in step_logs:
        tag = log["tag_type"]
        content = log["content"].replace("\n", " ")
        if len(content) > 30:
            content = content[:27] + "..."
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
        print(f"{tag:<16} | {delta_str:<17} | {total:<8.2f} | {reason:<50} | {content}")

    print("-" * 130)
    print(f"FINAL REWARD: {total_reward:.2f}")


if __name__ == "__main__":
    main()
