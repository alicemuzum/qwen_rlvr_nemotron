"""Synthetic gravity problem generator + reward monitor.

Mirrors monitor_cipher.py / monitor_cryptarithm.py's approach: build the
problem top-down from a fully known ground truth (a gravitational constant
k picked first), then run the deterministic generator to produce a
demonstration trace.

Distances are rounded to 2dp here, matching the precision real train.csv
rows use. This deliberately reintroduces the same per-example rounding
noise real data has (reasoning_gravity's median-of-truncated-divisions
estimate does not always reproduce it exactly) rather than sidestepping it
with full-precision synthetic distances -- see the retry loop in main(),
which already tolerates reasoning_gravity returning None on a fraction of
attempts (the same pattern monitor_cryptarithm.py uses for its much lower
solve rate).
"""

from __future__ import annotations

import random
import sys

from reasoners.gravity import reasoning_gravity
from reasoners.reward_gravity import evaluate_structured_trace
from reasoners.store_types import Example, Problem


def make_synthetic_problem(
    rng: random.Random, num_examples: int = 5
) -> tuple[Problem, str] | None:
    """Return (problem, k_true_str).

    Example t values are drawn distinct (including the question): both
    reasoners/gravity.py's tags and reward_gravity.py's parsing identify an
    example solely by its t value, so a problem with a repeated t is not
    well-formed for this trace format.
    """
    k_true = round(rng.uniform(1.0, 20.0), 3)
    k_str = f"{k_true:.3f}"

    def make_example(t: float) -> Example:
        d = round(k_true * t * t, 2)
        return Example(f"{t:.2f}", f"{d:.2f}")

    ts = rng.sample([round(x, 2) for x in [i / 100 for i in range(100, 1000)]], num_examples + 1)
    examples = [make_example(t) for t in ts[:num_examples]]
    q_ex = make_example(ts[num_examples])

    problem = Problem(
        id="synthetic",
        category="gravity",
        examples=examples,
        question=q_ex.input_value,
        answer=q_ex.output_value,
    )
    return problem, k_str


def main() -> None:
    rng = random.Random()
    problem = None
    for _ in range(200):
        built = make_synthetic_problem(rng)
        if built is None:
            continue
        candidate_problem, _k_true = built
        trace = reasoning_gravity(candidate_problem)
        if trace is not None:
            problem = candidate_problem
            break

    if problem is None:
        print("Could not construct a solvable synthetic problem in 200 tries.")
        sys.exit(1)

    print(f"--- Generating trace for t = {problem.question} (answer {problem.answer}) ---")
    trace = reasoning_gravity(problem)
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
