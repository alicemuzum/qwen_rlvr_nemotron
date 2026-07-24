"""Dense, state-aware reward function for cryptarithm reasoning traces.

Scores a ``<step type="...">...</step>`` trace produced by (or in the style
of) ``reasoners/cryptarithm.py`` against the problem's ground truth: the set
of real example equations, the expected final answer, and (for validity
checks only, see below) a symbol->digit / operator->operation assignment.

Why the oracle mapping is *not* used to grade individual digit/operator
claims: a cryptarithm puzzle with only a handful of example equations is
frequently ambiguous -- multiple internally-consistent symbol->digit
assignments can satisfy every given equation while disagreeing on symbols
that concat/rev_concat-style operators don't pin down. A trace using such an
alternate-but-valid witness is not wrong; grading it against one arbitrarily
chosen oracle assignment would reward or punish traces based on luck, which
is actively harmful noise for a GRPO advantage estimate. (Substitution cipher
has no such ambiguity -- the oracle there is a full 26-letter bijection
checked against real dictionary words -- which is why reward_cipher.py *does*
grade against its oracle directly.)

Instead, the ambiguity-invariant ground truth is the problem data itself:
the example equations (given, unfalsifiable) and the expected answer (given,
unfalsifiable). state_update/execution digit and operator claims are graded
on (a) whether the symbol is even part of this puzzle's alphabet, and (b)
self-consistency with the trace's own earlier claims -- never against a
single fixed "correct" value. Real correctness credit flows from the
verification steps, which replay the claimed values against the actual
given equations and the actual expected answer.

Design mirrors reasoners/reward_cipher.py:

  * "execution" steps are scored against what the trace itself has already
    established via "state_update" (flow awareness): using a fact before
    declaring it, or declaring one thing and later using another, is a flaw.
  * Verification steps grade the *label* (match/mismatch) for honesty
    separately from whether the underlying claim is actually true -- a
    trace that claims "match" on a wrong reconstruction is penalized harder
    than one that honestly admits a mismatch. Critically, a per-equation
    verification only pays out if the equation it claims to verify is one of
    the *real* example equations -- otherwise a trace could fabricate an
    equation whose "output" it chooses to match its own wrong derivation and
    farm free "verified" reward with zero grounding.
  * Categories that are cheap to repeat (re-affirmations, narrative
    "analysis" commentary, candidate-operator scratch work) are capped with
    a hard per-trace ceiling. Categories that are expensive to fake (a
    correctly verified real equation, the final boxed answer) are naturally
    bounded by problem size and are never capped.
  * Penalties are never capped or deduplicated.
  * A trace that never reaches a conclusion tag is terminally penalized.
"""

from __future__ import annotations

import re
from typing import Any

from reasoners.cryptarithm import OP_NAMES, _op_apply, _result_digits

_CEILINGS: dict[str, float] = {
    "plan": 0.3,
    "digit_reaffirm": 1.0,
    "op_reaffirm": 1.0,
    "exec_apply": 2.0,
    "analysis_parsing": 0.1,
    "analysis_candidate": 1.0,
    "analysis_recap": 0.3,
    "exec_arith": 1.0,
    "eq_verify_reaffirm": 0.5,
}

_MAX_STEPS = 500

_ARROW_RE = re.compile(r"^(\S+)\s*->\s*(\S+)$")
_LEFT_RIGHT_RE = re.compile(
    r"^left = 10\*(-?\d+)\+(-?\d+) = (-?\d+), right = 10\*(-?\d+)\+(-?\d+) = (-?\d+)$"
)
_OP_APPLY_RE = re.compile(r"^(\w+)\((-?\d+), (-?\d+)\) = (-?\d+) -> digits (\(.*?\))$")
_FINAL_VERIFY_RE = re.compile(
    r"^(\w+)\((-?\d+), (-?\d+)\) = (-?\d+) -> digits (\(.*?\)) -> symbols (.+)$"
)
_PER_EQ_VERIFY_RE = re.compile(r"^(.+?) -> reconstructed (.+?): (match|mismatch)$")
_SOLVING_MARKER_RE = re.compile(r"^Solving 【(.+?)】 = 【(.+?)】:$")
_CANDIDATE_OP_RE = re.compile(r"^Trying (\w+)\((-?\d+), (-?\d+)\) = (.+?): (consistent|contradiction)$")


def _apply_ceiling(totals: dict[str, float], key: str, delta: float) -> float:
    if delta <= 0:
        return delta
    cap = _CEILINGS[key]
    used = totals.get(key, 0.0)
    granted = max(0.0, min(delta, cap - used))
    totals[key] = used + granted
    return granted


def _parse_digit_tuple(s: str) -> tuple[int, ...] | None:
    inner = s.strip()
    if not (inner.startswith("(") and inner.endswith(")")):
        return None
    inner = inner[1:-1].strip()
    if not inner:
        return ()
    parts = [p.strip() for p in inner.split(",")]
    try:
        return tuple(int(p) for p in parts if p != "")
    except ValueError:
        return None


def _is_digit_value(value: str) -> bool:
    return len(value) == 1 and value.isdigit()


def evaluate_structured_trace(
    response_xml: str,
    examples: list[tuple[str, str]],
    expected_answer: str,
    oracle_map: dict[str, int] | None = None,
    oracle_ops: dict[str, str] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Evaluates a generated reasoning trace for the cryptarithm task.

    Args:
        response_xml: The raw XML/text output from the model.
        examples: The problem's real (input_str, output_str) example equation
            pairs -- unfalsifiable ground truth used to validate that
            per-equation verification claims refer to a real equation, not a
            fabricated one.
        expected_answer: The ground truth answer string for the question.
        oracle_map: Optional symbol->digit assignment, used *only* to gate
            whether a digit claim's symbol is a real puzzle symbol at all
            (never to grade the claimed digit value -- see module docstring
            on why per-symbol grading against a single oracle is unsound for
            this task). If omitted, the alphabet is inferred from `examples`
            and the question symbols implied by `expected_answer`'s length.
        oracle_ops: Optional operator-symbol->operation-name assignment, used
            the same way as oracle_map but for operator symbols.

    Returns:
        tuple: (total_reward, step_logs)
    """
    total_reward = 0.0
    known_digits: dict[str, int] = {}
    known_ops: dict[str, str] = {}
    ceiling_totals: dict[str, float] = {}
    seen_eq_markers: set[str] = set()
    verified_real_equations: set[str] = set()
    real_outputs = {out for _inp, out in examples}
    num_equations = len(examples)
    equations_verified_correctly = 0
    final_verification_correct = False
    saw_conclusion = False
    step_logs: list[dict[str, Any]] = []

    valid_digit_symbols: set[str] | None = set(oracle_map) if oracle_map is not None else None
    valid_op_symbols: set[str] | None = set(oracle_ops) if oracle_ops is not None else None

    steps = re.findall(r'<step type="(.*?)">(.*?)</step>', response_xml, re.DOTALL)

    if not steps:
        step_logs.append(
            {
                "tag_type": "ERROR",
                "content": "Total format failure (no steps found)",
                "reward_delta": -5.0,
                "total_reward": -5.0,
                "reason": "Missing steps",
            }
        )
        return -5.0, step_logs

    conclusion_indices = [idx for idx, (t, _) in enumerate(steps) if t == "conclusion"]
    last_conclusion_idx = conclusion_indices[-1] if conclusion_indices else None

    for i, (tag_type, content) in enumerate(steps):
        content = content.strip()
        prev_reward = total_reward
        reason = ""

        if tag_type == "conclusion" and i != last_conclusion_idx:
            step_logs.append(
                {
                    "tag_type": tag_type,
                    "content": content if len(content) < 50 else content[:47] + "...",
                    "reward_delta": 0.0,
                    "total_reward": total_reward,
                    "reason": "Superseded by a later conclusion tag; ignored",
                }
            )
            continue

        if tag_type != "conclusion" and i >= _MAX_STEPS:
            step_logs.append(
                {
                    "tag_type": tag_type,
                    "content": content if len(content) < 50 else content[:47] + "...",
                    "reward_delta": 0.0,
                    "total_reward": total_reward,
                    "reason": "Step budget exceeded; no further reward considered",
                }
            )
            continue

        if tag_type == "plan":
            granted = _apply_ceiling(ceiling_totals, "plan", 0.15)
            total_reward += granted
            reason = "Plan step present" if granted > 0 else "Plan step (ceiling reached)"

        elif tag_type == "state_update":
            m = _ARROW_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                sym, value = m.group(1), m.group(2)
                if _is_digit_value(value):
                    digit = int(value)
                    if valid_digit_symbols is not None and sym not in valid_digit_symbols:
                        total_reward -= 1.0
                        reason = "Hallucinated symbol: not part of this puzzle's alphabet"
                    elif sym in known_digits and known_digits[sym] != digit:
                        total_reward -= 1.5
                        reason = "Self-contradiction: digit conflicts with its own earlier claim"
                    elif sym in known_digits:
                        granted = _apply_ceiling(ceiling_totals, "digit_reaffirm", 0.05)
                        total_reward += granted
                        reason = "R_consistency (re-affirmed own claim)" if granted else "R_consistency (ceiling reached)"
                    else:
                        known_digits[sym] = digit
                        total_reward += 0.3
                        reason = "New digit declared for a valid symbol (correctness judged at verification)"
                elif value in OP_NAMES:
                    if valid_op_symbols is not None and sym not in valid_op_symbols:
                        total_reward -= 1.0
                        reason = "Hallucinated symbol: not a real operator in this puzzle"
                    elif sym in known_ops and known_ops[sym] != value:
                        total_reward -= 1.5
                        reason = "Self-contradiction: operator conflicts with its own earlier claim"
                    elif sym in known_ops:
                        granted = _apply_ceiling(ceiling_totals, "op_reaffirm", 0.05)
                        total_reward += granted
                        reason = "R_consistency (re-affirmed own claim)" if granted else "R_consistency (ceiling reached)"
                    else:
                        known_ops[sym] = value
                        total_reward += 0.3
                        reason = "New operator meaning declared (correctness judged at verification)"
                else:
                    total_reward -= 0.5
                    reason = "Formatting error: value is neither a digit nor a known op name"

        elif tag_type == "execution":
            m_lr = _LEFT_RIGHT_RE.match(content)
            m_op = _OP_APPLY_RE.match(content)
            m_simple = _ARROW_RE.match(content)

            if m_lr:
                a, b, left, c, d, right = (int(x) for x in m_lr.groups())
                ok = (10 * a + b == left) and (10 * c + d == right)
                if ok:
                    granted = _apply_ceiling(ceiling_totals, "exec_arith", 0.1)
                    total_reward += granted
                    reason = "Arithmetic self-consistent" if granted else "Arithmetic self-consistent (ceiling reached)"
                else:
                    total_reward -= 1.0
                    reason = "Arithmetic error (stated left/right don't match stated digits)"

            elif m_op:
                op_name, a, b, value, digits_str = m_op.groups()
                a, b, value = int(a), int(b), int(value)
                digits = _parse_digit_tuple(digits_str)
                if op_name not in OP_NAMES:
                    total_reward -= 0.5
                    reason = "Unknown operation name"
                else:
                    true_value = _op_apply(OP_NAMES.index(op_name), a, b)
                    true_digits = _result_digits(OP_NAMES.index(op_name), true_value)
                    if value == true_value and digits == true_digits:
                        granted = _apply_ceiling(ceiling_totals, "exec_arith", 0.1)
                        total_reward += granted
                        reason = "Operation applied correctly" if granted else "Operation applied correctly (ceiling reached)"
                    else:
                        total_reward -= 1.0
                        reason = "Operation misapplied (arithmetic doesn't check out)"

            elif m_simple:
                sym, value = m_simple.groups()
                if _is_digit_value(value):
                    digit = int(value)
                    if known_digits.get(sym) == digit:
                        granted = _apply_ceiling(ceiling_totals, "exec_apply", 0.2)
                        total_reward += granted
                        reason = "R_apply (used its own previously declared digit)" if granted else "R_apply (ceiling reached)"
                    elif sym in known_digits:
                        total_reward -= 1.5
                        reason = "Self-contradiction: digit conflicts with its own earlier claim"
                    elif valid_digit_symbols is not None and sym not in valid_digit_symbols:
                        total_reward -= 1.0
                        reason = "Hallucinated symbol: not part of this puzzle's alphabet"
                    else:
                        total_reward -= 0.5
                        reason = "Used a digit for this symbol before declaring it via state_update"
                elif value in OP_NAMES:
                    if known_ops.get(sym) == value:
                        granted = _apply_ceiling(ceiling_totals, "exec_apply", 0.2)
                        total_reward += granted
                        reason = "R_apply (used its own previously declared operator)" if granted else "R_apply (ceiling reached)"
                    elif sym in known_ops:
                        total_reward -= 1.5
                        reason = "Self-contradiction: operator conflicts with its own earlier claim"
                    elif valid_op_symbols is not None and sym not in valid_op_symbols:
                        total_reward -= 1.0
                        reason = "Hallucinated symbol: not a real operator in this puzzle"
                    else:
                        total_reward -= 0.5
                        reason = "Used an operator meaning before declaring it via state_update"
                else:
                    total_reward -= 0.5
                    reason = "Formatting error"
            else:
                total_reward -= 0.5
                reason = "Formatting error"

        elif tag_type == "analysis":
            if content.startswith("Parsing the examples:"):
                granted = _apply_ceiling(ceiling_totals, "analysis_parsing", 0.1)
                total_reward += granted
                reason = "Initial parsing overview" if granted else "Parsing overview (ceiling reached)"

            elif _SOLVING_MARKER_RE.match(content):
                eq_in = _SOLVING_MARKER_RE.match(content).group(1)
                if eq_in not in seen_eq_markers:
                    seen_eq_markers.add(eq_in)
                    total_reward += 0.05
                    reason = "Started solving a distinct equation"
                else:
                    reason = "Repeated equation marker (no additional credit)"

            elif (m := _CANDIDATE_OP_RE.match(content)):
                op_name, a, b, _shown_value, verdict = m.groups()
                a, b = int(a), int(b)
                if op_name in OP_NAMES:
                    true_value = _op_apply(OP_NAMES.index(op_name), a, b)
                    # Only the arithmetic component is independently checkable
                    # here (the consistency verdict itself depends on which
                    # result symbols are already mapped, not present in this
                    # line's content) -- ground what's groundable, cap the
                    # rest so repeating scratch work can't be farmed.
                    true_digits = _result_digits(OP_NAMES.index(op_name), true_value)
                    plausible = true_digits is not None
                    if verdict == "contradiction" and not plausible:
                        granted = _apply_ceiling(ceiling_totals, "analysis_candidate", 0.02)
                        total_reward += granted
                        reason = "Correctly ruled out an infeasible candidate operation"
                    elif verdict in ("consistent", "contradiction"):
                        granted = _apply_ceiling(ceiling_totals, "analysis_candidate", 0.02)
                        total_reward += granted
                        reason = "Candidate operation scratch work (capped)"
                    else:
                        total_reward -= 0.3
                        reason = "Invalid verdict label"
                else:
                    total_reward -= 0.3
                    reason = "Unknown operation name in candidate analysis"

            elif content.startswith("Mapping so far") and "Operators so far" in content:
                lines = content.split("\n")
                mismatch = False
                consistent_lines = 0
                for line in lines:
                    m2 = _ARROW_RE.match(line.strip())
                    if not m2:
                        continue
                    sym, value = m2.groups()
                    if _is_digit_value(value):
                        digit = int(value)
                        if sym in known_digits and known_digits[sym] != digit:
                            mismatch = True
                        elif sym in known_digits:
                            consistent_lines += 1
                        # A recap can legitimately cite a digit that was never
                        # individually state_update'd (the generator only ever
                        # declares operand symbols that way, never result
                        # symbols) -- can't grade an undeclared symbol's value
                        # without an oracle, so it's neutral either way.
                    elif value in OP_NAMES:
                        if sym in known_ops and known_ops[sym] != value:
                            mismatch = True
                        elif sym in known_ops:
                            consistent_lines += 1
                if mismatch:
                    total_reward -= 0.5
                    reason = "Recap contradicts its own earlier declared state"
                elif consistent_lines and not reason:
                    granted = _apply_ceiling(ceiling_totals, "analysis_recap", 0.02 * consistent_lines)
                    total_reward += granted
                    reason = "Recap self-consistent with earlier state"

        elif tag_type == "verification":
            m_eq = _PER_EQ_VERIFY_RE.match(content)
            m_final = _FINAL_VERIFY_RE.match(content)

            if m_eq:
                dashed_output, _reconstructed, label = m_eq.groups()
                claimed_output = dashed_output.replace("–", "").strip()

                matching_examples = [(inp, out) for inp, out in examples if out == claimed_output]
                if not matching_examples:
                    total_reward -= 1.0
                    reason = "P_fabricated_equation: claims to verify an equation not in the problem"
                else:
                    # Ground truth here is NOT "does the trace's own reconstructed
                    # string equal the string it's checking against" (both are
                    # attacker-controlled and the real output is visible in the
                    # prompt, so that check is trivially copyable). Ground truth
                    # is: does the trace's OWN previously-declared mapping, when
                    # actually applied to this equation's real input, reproduce
                    # this real output? Only that can't be faked without solving.
                    # Distinct equations can share the same output string, so a
                    # candidate must be checked (and credited) per input, not
                    # per output -- otherwise verifying two different real
                    # equations that happen to produce the same output would
                    # under-count the second as a mere "reaffirmation".
                    verified_input: str | None = None
                    verified_fresh = False
                    insufficient_state = True
                    ordered_candidates = sorted(
                        matching_examples, key=lambda p: p[0] in verified_real_equations
                    )
                    for inp, out in ordered_candidates:
                        if len(inp) != 5:
                            continue
                        a0, a1, op, b0, b1 = inp[0], inp[1], inp[2], inp[3], inp[4]
                        if not all(s in known_digits for s in (a0, a1, b0, b1)) or op not in known_ops:
                            continue
                        insufficient_state = False
                        left = known_digits[a0] * 10 + known_digits[a1]
                        right = known_digits[b0] * 10 + known_digits[b1]
                        op_id = OP_NAMES.index(known_ops[op])
                        value = _op_apply(op_id, left, right)
                        digits = _result_digits(op_id, value)
                        if digits is None or len(digits) != len(out):
                            continue
                        sym_to_digit: dict[str, int] = {}
                        consistent = True
                        for sym_k, dig_k in zip(out, digits):
                            if sym_k in known_digits and known_digits[sym_k] != dig_k:
                                consistent = False
                                break
                            if sym_k in sym_to_digit and sym_to_digit[sym_k] != dig_k:
                                consistent = False
                                break
                            other = next((s for s, d in known_digits.items() if d == dig_k and s != sym_k), None)
                            if other is not None:
                                consistent = False
                                break
                            sym_to_digit[sym_k] = dig_k
                        if not consistent:
                            continue
                        verified_input = inp
                        verified_fresh = inp not in verified_real_equations
                        for sym_k, dig_k in sym_to_digit.items():
                            known_digits.setdefault(sym_k, dig_k)
                        break

                    verified_ok = verified_input is not None
                    if verified_ok and label == "match":
                        if not verified_fresh:
                            granted = _apply_ceiling(ceiling_totals, "eq_verify_reaffirm", 0.1)
                            total_reward += granted
                            reason = "Re-verified an already-credited equation (capped)"
                        else:
                            verified_real_equations.add(verified_input)
                            total_reward += 1.0
                            equations_verified_correctly += 1
                            reason = "Its own established mapping genuinely reproduces this real equation"
                    elif verified_ok and label == "mismatch":
                        total_reward -= 1.0
                        reason = "Dishonest/confused: its own mapping actually reproduces this equation but labeled mismatch"
                    elif (not verified_ok) and label == "match" and insufficient_state:
                        total_reward -= 0.5
                        reason = "Claimed a verified match without having established the required digits/operator"
                    elif (not verified_ok) and label == "match":
                        total_reward -= 2.0
                        reason = "P_false_verify: claimed a match its own established mapping doesn't actually produce"
                    else:
                        total_reward -= 0.5
                        reason = "Honestly flagged an incorrect/unverifiable reconstruction"

            elif m_final:
                op_name, a, b, value, digits_str, computed = m_final.groups()
                a, b, value = int(a), int(b), int(value)
                digits = _parse_digit_tuple(digits_str)
                arith_ok = False
                if op_name in OP_NAMES:
                    true_value = _op_apply(OP_NAMES.index(op_name), a, b)
                    true_digits = _result_digits(OP_NAMES.index(op_name), true_value)
                    arith_ok = value == true_value and digits == true_digits
                computed = computed.strip()
                if computed == expected_answer:
                    if final_verification_correct:
                        granted = _apply_ceiling(ceiling_totals, "eq_verify_reaffirm", 0.1)
                        total_reward += granted
                        reason = "Re-verified the final answer (capped, already credited)"
                    else:
                        total_reward += 2.0
                        final_verification_correct = True
                        reason = "Correct final answer derived and verified"
                else:
                    total_reward -= 2.0
                    reason = "Incorrect final answer at verification checkpoint"
                if not arith_ok:
                    total_reward -= 1.0
                    reason += " (and the arithmetic shown doesn't check out)"

        elif tag_type == "conclusion":
            saw_conclusion = True
            matches = re.findall(r"\\boxed\{(.*?)\}", content, re.DOTALL)
            final = matches[-1].strip() if matches else None
            if final is not None and final == expected_answer:
                total_reward += 10.0
                reason = "R_terminal_win (correct boxed answer)"
            else:
                total_reward -= 5.0
                reason = "R_terminal_fail (incorrect or unboxed answer)"

        step_logs.append(
            {
                "tag_type": tag_type,
                "content": content if len(content) < 50 else content[:47] + "...",
                "reward_delta": total_reward - prev_reward,
                "total_reward": total_reward,
                "reason": reason if reason else "No specific reward triggered",
            }
        )

    if not saw_conclusion:
        prev_reward = total_reward
        total_reward -= 5.0
        step_logs.append(
            {
                "tag_type": "NO_CONCLUSION",
                "content": "Trace ended without a conclusion tag",
                "reward_delta": total_reward - prev_reward,
                "total_reward": total_reward,
                "reason": "P_no_commit (never produced a final boxed answer)",
            }
        )

    horizon_delta = 0.0
    horizon_reason = ""
    if equations_verified_correctly == num_equations and final_verification_correct and num_equations:
        horizon_delta = 5.0
        horizon_reason = "Long horizon: all equations and the final question verified correctly"
    elif equations_verified_correctly > 0:
        horizon_delta = (equations_verified_correctly / num_equations) * 2.0
        horizon_reason = f"Long horizon: partial success ({equations_verified_correctly}/{num_equations} equations)"

    if horizon_delta > 0:
        total_reward += horizon_delta
        step_logs.append(
            {
                "tag_type": "HORIZON_EVAL",
                "content": "End of trace evaluation",
                "reward_delta": horizon_delta,
                "total_reward": total_reward,
                "reason": horizon_reason,
            }
        )

    return total_reward, step_logs
