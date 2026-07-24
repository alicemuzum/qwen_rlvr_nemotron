"""Synthetic cryptarithm problem generator + reward monitor.

Mirrors monitor_cipher.py's approach: build the puzzle top-down from a fully
known ground truth (a bijective symbol->digit map over all ten digits, plus
an operator->operation assignment), so the oracle handed to the reward
function is always complete -- generation never depends on the CSP solver in
reasoners/cryptarithm.py successfully reverse-engineering it (that solver only
clears ~8.5% of the real train.csv rows, which is fine for grading a fixed
dataset but useless as a source of dense oracles for RL).

reasoning_cryptarithm() is still used here to produce a demonstration gold
trace, purely so this script has something realistic to feed the reward
function; a real GRPO loop would score the *policy's* completion against the
same oracle instead.
"""

from __future__ import annotations

import random
import sys

from reasoners.cryptarithm import OP_NAMES, _op_apply, _result_digits, reasoning_cryptarithm
from reasoners.reward_cryptarithm import evaluate_structured_trace
from reasoners.store_types import Example, Problem

_SYMBOL_POOL = list(dict.fromkeys(
    "!@#$%^&*+=~<>?/|;:.,'`_[]{}()abcdefghij"
    "¡§¶¿†‡•⁄№℗"
))
assert len(set(_SYMBOL_POOL)) == len(_SYMBOL_POOL), "duplicate symbol in pool"


def make_random_cryptarithm(
    rng: random.Random, num_op_symbols: int = 2
) -> tuple[dict[str, int], dict[str, str]]:
    """Return (oracle_map, oracle_ops): a full digit-symbol bijection over
    0-9 plus a random assignment of operations to a handful of operator
    symbols, all drawn from a disjoint symbol pool."""
    symbols = rng.sample(_SYMBOL_POOL, 10 + num_op_symbols)
    digit_symbols, op_symbols = symbols[:10], symbols[10:]
    shuffled_digits = list(range(10))
    rng.shuffle(shuffled_digits)
    oracle_map = {s: d for s, d in zip(digit_symbols, shuffled_digits)}
    oracle_ops = {s: rng.choice(OP_NAMES) for s in op_symbols}
    return oracle_map, oracle_ops


def _make_equation(
    rng: random.Random, digit_to_sym: dict[int, str], op_symbols: list[str], oracle_ops: dict[str, str]
) -> tuple[str, str] | None:
    a0, a1, b0, b1 = (rng.randint(0, 9) for _ in range(4))
    op_sym = rng.choice(op_symbols)
    op_id = OP_NAMES.index(oracle_ops[op_sym])
    left, right = a0 * 10 + a1, b0 * 10 + b1
    value = _op_apply(op_id, left, right)
    digits = _result_digits(op_id, value)
    if digits is None:
        return None
    input_str = digit_to_sym[a0] + digit_to_sym[a1] + op_sym + digit_to_sym[b0] + digit_to_sym[b1]
    output_str = "".join(digit_to_sym[d] for d in digits)
    return input_str, output_str


def make_synthetic_problem(
    rng: random.Random, num_examples: int = 4, num_op_symbols: int = 2
) -> tuple[Problem, dict[str, int], dict[str, str]] | None:
    oracle_map, oracle_ops = make_random_cryptarithm(rng, num_op_symbols)
    digit_to_sym = {d: s for s, d in oracle_map.items()}
    op_symbols = list(oracle_ops.keys())

    examples = []
    for _ in range(num_examples):
        eq = _make_equation(rng, digit_to_sym, op_symbols, oracle_ops)
        if eq is None:
            return None
        examples.append(Example(eq[0], eq[1]))

    q = _make_equation(rng, digit_to_sym, op_symbols, oracle_ops)
    if q is None:
        return None
    question, answer = q

    problem = Problem(
        id="synthetic",
        category="cryptarithm_deduce",
        examples=examples,
        question=question,
        answer=answer,
    )
    return problem, oracle_map, oracle_ops


def main() -> None:
    rng = random.Random()
    problem = oracle_map = oracle_ops = None
    for _ in range(200):
        built = make_synthetic_problem(rng)
        if built is None:
            continue
        candidate_problem, candidate_map, candidate_ops = built
        trace = reasoning_cryptarithm(candidate_problem)
        if trace is not None:
            problem, oracle_map, oracle_ops = candidate_problem, candidate_map, candidate_ops
            break

    if problem is None:
        print("Could not construct a solvable synthetic problem in 200 tries.")
        sys.exit(1)

    print(f"--- Generating trace for question 【{problem.question}】 (answer {problem.answer}) ---")
    trace = reasoning_cryptarithm(problem)
    assert trace is not None

    print("--- Evaluating trace... ---\n")
    examples = [(ex.input_value, ex.output_value) for ex in problem.examples]
    total_reward, step_logs = evaluate_structured_trace(
        trace, examples, problem.answer, oracle_map, oracle_ops
    )

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
