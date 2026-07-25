"""Synthetic numeral problem generator + reward monitor.

Mirrors monitor_gravity.py's approach: build the problem top-down from a
fully known ground truth (arabic integers picked first), then run the
deterministic generator to produce a demonstration trace. Unlike gravity's
k (estimated) or cryptarithm's symbol map (structurally ambiguous),
numeral's answer is exactly recomputable from the question integer via a
fixed universal table, so reasoning_numeral should never return None here --
the retry loop is kept anyway for robustness/symmetry with its siblings.
"""

from __future__ import annotations

import random
import sys

from reasoners.numeral import _to_roman, reasoning_numeral
from reasoners.reward_numeral import evaluate_structured_trace
from reasoners.store_types import Example, Problem


def make_synthetic_problem(
    rng: random.Random, num_examples: int = 4
) -> Problem | None:
    """Pick distinct arabic integers (examples + question) in 1..3999, the
    canonical range a single Roman numeral can represent.

    Distinctness matters -- the reward parser keys examples by their arabic
    value (same reason gravity draws distinct t's; see monitor_gravity.py's
    docstring).
    """
    ns = rng.sample(range(1, 4000), num_examples + 1)
    examples = [Example(str(n), _to_roman(n)) for n in ns[:num_examples]]
    q = ns[num_examples]

    return Problem(
        id="synthetic",
        category="numeral",
        examples=examples,
        question=str(q),
        answer=_to_roman(q),
    )


def main() -> None:
    rng = random.Random()
    problem = None
    for _ in range(200):
        candidate_problem = make_synthetic_problem(rng)
        if candidate_problem is None:
            continue
        trace = reasoning_numeral(candidate_problem)
        if trace is not None:
            problem = candidate_problem
            break

    if problem is None:
        print("Could not construct a solvable synthetic problem in 200 tries.")
        sys.exit(1)

    print(f"--- Generating trace for n = {problem.question} (answer {problem.answer}) ---")
    trace = reasoning_numeral(problem)
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
