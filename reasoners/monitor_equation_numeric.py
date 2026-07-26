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

Real train.csv equation_numeric prompts are overwhelmingly multi-operator:
measured across all 732 real rows, only 44 (6%) use a single operator symbol
throughout, 380 (52%) use two, and 308 (42%) use three -- each operator its
own independent hidden rule, sharing one prompt (see CLAUDE.md's "SFT
readiness audit"). ``make_synthetic_problem`` reproduces that shape: it picks
1-3 operators with those measured weights, gives each its own rule, and
distributes examples across them so per-operator example counts vary the way
real data's do (mostly 1-2 examples per operator, occasionally more). The
question's operator is drawn from the same set ~81% of the time (also
measured on real data); the other ~19% it's a novel operator absent from
every example, exercising ``reasoning_equation_numeric``'s fallback path --
these frequently fail to verify and get discarded by the caller's retry
loop, mirroring real data's lower solve rate on unseen-operator questions.
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict

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

# Weights measured on the 732 real train.csv equation_numeric rows (see
# module docstring): {1: 44, 2: 380, 3: 308} distinct operators per prompt.
_NUM_OPERATORS_CHOICES = [1, 2, 3]
_NUM_OPERATORS_WEIGHTS = [44, 380, 308]

# Measured on the same 732 rows: 3/4/5 example lines occur roughly equally
# often (252/239/241).
_TOTAL_EXAMPLES_CHOICES = [3, 4, 5]

# Measured on the same 732 rows: the question's operator appears among the
# examples 596/732 of the time.
_QUESTION_OP_SEEN_RATE = 596 / 732


def _make_rule(rng: random.Random) -> tuple[str, bool, bool, str]:
    signed = rng.random() < 0.35
    if signed:
        op_name = rng.choice(_SIGNED_OPS)
        fmt = rng.choice(["neg_suffix", "neg_prefix"])
    else:
        op_name = rng.choice(_CLEAN_OPS)
        fmt = "num"
    rev_ops = rng.random() < 0.5
    rev_res = rng.random() < 0.5
    return op_name, rev_ops, rev_res, fmt


def make_synthetic_problem(rng: random.Random) -> Problem | None:
    num_operators = rng.choices(_NUM_OPERATORS_CHOICES, weights=_NUM_OPERATORS_WEIGHTS)[0]
    op_chars = rng.sample(_OP_SYMBOLS, num_operators)
    rules: dict[str, tuple[str, bool, bool, str]] = {op: _make_rule(rng) for op in op_chars}

    total_examples = max(rng.choice(_TOTAL_EXAMPLES_CHOICES), num_operators)

    # Every operator gets at least one example; remaining examples are
    # assigned to a random operator from the set, so per-operator example
    # counts come out uneven, matching real data.
    assigned_ops = list(op_chars)
    assigned_ops += [rng.choice(op_chars) for _ in range(total_examples - num_operators)]
    rng.shuffle(assigned_ops)

    # Question operator: usually one seen in the examples; otherwise a novel
    # operator with its own hidden rule (see module docstring).
    unseen_available = [c for c in _OP_SYMBOLS if c not in op_chars]
    if unseen_available and rng.random() >= _QUESTION_OP_SEEN_RATE:
        q_op = rng.choice(unseen_available)
        rules[q_op] = _make_rule(rng)
    else:
        q_op = rng.choice(op_chars)

    # Distinct 2-digit operand pairs across the whole problem (matches real
    # train.csv), one per example plus the question. Distinctness matters --
    # the reward keys examples by their full input string.
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < total_examples + 1:
        pairs.add((rng.randint(10, 99), rng.randint(10, 99)))
    pair_list = list(pairs)

    def out_for(op_char: str, a: int, b: int) -> str | None:
        op_name, rev_ops, rev_res, fmt = rules[op_char]
        return apply_rule(op_name, rev_ops, rev_res, fmt, op_char, str(a), str(b))

    examples: list[Example] = []
    outputs_by_op: dict[str, list[str]] = defaultdict(list)
    for op_char, (a, b) in zip(assigned_ops, pair_list[:total_examples]):
        out = out_for(op_char, a, b)
        if out is None:
            return None
        examples.append(Example(f"{a}{op_char}{b}", out))
        outputs_by_op[op_char].append(out)

    # For a signed format to be detectable, at least one of that operator's
    # example outputs must actually carry the sign symbol; otherwise the
    # generator sees plain numbers and infers fmt="num", a different (still
    # valid) rule.
    for op_char in op_chars:
        _, _, _, fmt = rules[op_char]
        if fmt in ("neg_suffix", "neg_prefix") and not any(
            o.endswith(op_char) or o.startswith(op_char) for o in outputs_by_op[op_char]
        ):
            return None

    qa, qb = pair_list[total_examples]
    answer = out_for(q_op, qa, qb)
    if answer is None:
        return None

    return Problem(
        id="synthetic",
        category="equation_numeric_deduce",
        examples=examples,
        question=f"{qa}{q_op}{qb}",
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
