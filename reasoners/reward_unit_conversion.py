"""Dense, state-aware reward function for unit_conversion (output = factor*input)
reasoning traces.

Scores a ``<step type="...">...</step>`` trace produced by (or in the style
of) ``reasoners/unit_conversion.py`` against the problem's ground truth: the
given (input, output) example pairs and the expected final answer.

Assumes example input values are distinct within a problem: both this trace
format's tags (``factor[input]->...``, ``out(input) = ...``) and
reasoners/unit_conversion.py's own generator identify an example solely by
its input value, with no way to disambiguate a repeated input.

This mirrors reward_gravity.py's discipline (see that module's docstring for
the full rationale): the trace's final fitted factor is *not* graded against
a single "true" factor, because it is recovered by taking the median of
several per-example ``output / input`` divisions, each truncated (not
rounded) to 3 decimal places -- an estimate, not an exact recovery. Grading
a claimed ``factor_fit`` against an external oracle would reward or punish a
trace based on rounding luck. Instead:

  * Each *per-example* local factor (``factor[input]->...``) is graded
    directly, because it -- unlike the final fitted factor -- *is* exactly
    and deterministically recomputable from that example's own visible
    (input, output) pair using the same truncating arithmetic
    reasoners/unit_conversion.py itself uses.
  * The final factor_fit is graded only two ways: (a) self-consistency --
    does it equal the median of the trace's own previously-declared
    per-example factor values; and (b) a "reproduce the givens"
    verification pass -- does factor_fit, applied back to each given input,
    actually reproduce that example's given output (recomputed
    independently by this function, never trusting the trace's
    self-reported "reproduced" value).
  * "execution" steps that show arithmetic work (division, final
    multiplication) are graded for self-consistency and capped.
  * Penalties are never capped or deduplicated. A trace that never reaches a
    conclusion tag is terminally penalized. Only the *last* conclusion tag
    counts.

**The one deliberate divergence from reward_gravity.py:** the conclusion's
boxed answer is graded with a ``1e-2`` numeric tolerance instead of exact
string equality. Unlike gravity's k, the factor here is confirmed (measured
against 300 real train.csv rows) to only exactly reproduce the stored
answer ~82.7% of the time even with fully correct-method estimation; the
remaining ~17% land within 0.01-0.02 due to rounding noise in 2dp-rounded
example inputs/outputs. Exact-match grading here would inject that same
rounding-luck noise into the advantage estimate that this codebase's design
philosophy (see gravity/cryptarithm docstrings) explicitly avoids.
"""

from __future__ import annotations

import re
from decimal import InvalidOperation
from typing import Any

from reasoners.store_types import (
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    round_2dp,
    truncate_3dp,
)

_CEILINGS: dict[str, float] = {
    "plan": 0.3,
    "exec_arith": 1.0,
    "factor_reaffirm": 0.5,
    "ffit_reaffirm": 0.3,
    "analysis_recap": 0.3,
    "verify_reaffirm": 0.5,
    "honest_mismatch": 0.3,
}

_MAX_STEPS = 500

_MARKER_RE = re.compile(r"^input = (\S+), output = (\S+):$")
_FVALUES_RE = re.compile(r"^factor values: (.+)$")
_FSORTED_RE = re.compile(r"^factor values \(sorted\): (.+)$")
_EXEC_FACTOR_RE = re.compile(r"^factor = (\S+)/(\S+) = (\S+)$")
_EXEC_ANSWER_RE = re.compile(r"^answer = (\S+)\*(\S+) = (\S+)$")
_STATE_FACTOR_LOCAL_RE = re.compile(r"^factor\[(\S+)\]->(\S+)$")
_STATE_FFIT_RE = re.compile(r"^factor_fit->(\S+)$")
_VERIFY_RE = re.compile(
    r"^out\((\S+)\) = (\S+)\*(\S+) = (\S+) vs (\S+): (match|mismatch)$"
)

_TOLERANCE = 1e-2 + 1e-9


def _safe_2dp(s: str) -> str | None:
    """round_2dp, but returns None instead of raising on non-numeric input.

    Needed anywhere the compared value is regex-captured straight out of the
    trace (untrusted, possibly adversarial) rather than out of the problem's
    own data.
    """
    try:
        return round_2dp(s)
    except (InvalidOperation, ValueError):
        return None


def _apply_ceiling(totals: dict[str, float], key: str, delta: float) -> float:
    if delta <= 0:
        return delta
    cap = _CEILINGS[key]
    used = totals.get(key, 0.0)
    granted = max(0.0, min(delta, cap - used))
    totals[key] = used + granted
    return granted


def _factor_for_example(inp_str: str, out_str: str) -> str:
    """Exactly replicates reasoners.unit_conversion's per-example factor derivation.

    max_decimal_digits=6 must stay bit-identical to the generator's call --
    see the comment there for why the 3dp default is wrong for this task's
    factor magnitude.
    """
    inp_t, out_t = truncate_3dp(inp_str), truncate_3dp(out_str)
    inp_cast, out_cast, _, _ = cast_dp_pair(inp_t, out_t)
    _, factor_str = long_division_lines(out_cast, inp_cast, max_decimal_digits=6)
    return factor_str


def _reproduce_out(factor_fit_str: str, inp_str: str) -> str:
    """Exactly replicates reasoners.unit_conversion's verification-pass multiplication."""
    factor_display = factor_fit_str.rstrip("0").rstrip(".") or "0"
    _, mult_result = long_multiplication_lines(factor_display, inp_str)
    return round_2dp(mult_result)


def _expected_median(factor_strs: list[str]) -> str | None:
    if not factor_strs:
        return None
    paired = sorted((float(s), s) for s in factor_strs)
    n = len(paired)
    idx = n // 2 - 1 if (n % 2 == 0 and n >= 2) else n // 2
    return paired[idx][1]


def evaluate_structured_trace(
    response_xml: str,
    examples: list[tuple[str, str]],
    expected_answer: str,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Evaluates a generated reasoning trace for the unit_conversion
    (output = factor*input) task.

    Args:
        response_xml: The raw XML/text output from the model.
        examples: The problem's real (input, output) example pairs --
            unfalsifiable ground truth used both to recompute exact
            per-example factor values and to validate that verification
            claims refer to a real given example, not a fabricated one.
        expected_answer: The ground truth answer string for the question.

    Returns:
        tuple: (total_reward, step_logs)
    """
    total_reward = 0.0
    known_factors: dict[str, str] = {}  # trace's own claimed per-example factor, by input
    known_ffit: str | None = None
    ceiling_totals: dict[str, float] = {}
    seen_markers: set[str] = set()
    verified_examples: set[str] = set()
    honestly_assessed: set[str] = set()
    example_map = {inp: out for inp, out in examples}
    num_examples = len(examples)
    saw_conclusion = False
    step_logs: list[dict[str, Any]] = []

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
            m_local = _STATE_FACTOR_LOCAL_RE.match(content)
            m_fit = _STATE_FFIT_RE.match(content)

            if m_local:
                inp_sym, factor_claimed = m_local.groups()
                if inp_sym not in example_map:
                    total_reward -= 1.0
                    reason = "Hallucinated input: not one of this problem's given examples"
                elif inp_sym in known_factors and known_factors[inp_sym] != factor_claimed:
                    total_reward -= 1.5
                    reason = "Self-contradiction: factor for this input conflicts with its own earlier claim"
                elif inp_sym in known_factors:
                    granted = _apply_ceiling(ceiling_totals, "factor_reaffirm", 0.05)
                    total_reward += granted
                    reason = "R_consistency (re-affirmed own claim)" if granted else "R_consistency (ceiling reached)"
                else:
                    expected = _factor_for_example(inp_sym, example_map[inp_sym])
                    known_factors[inp_sym] = factor_claimed
                    if factor_claimed == expected:
                        total_reward += 0.5
                        reason = "R_discovery (correctly derived this example's factor)"
                    else:
                        total_reward -= 2.0
                        reason = "P_fatal (arithmetic error: derived factor doesn't match output/input)"

            elif m_fit:
                factor_claimed = m_fit.group(1)
                if known_ffit is not None and known_ffit != factor_claimed:
                    total_reward -= 1.5
                    reason = "Self-contradiction: factor_fit conflicts with its own earlier claim"
                elif known_ffit is not None:
                    granted = _apply_ceiling(ceiling_totals, "ffit_reaffirm", 0.05)
                    total_reward += granted
                    reason = "R_consistency (re-affirmed factor_fit)" if granted else "R_consistency (ceiling reached)"
                else:
                    expected = _expected_median(list(known_factors.values()))
                    known_ffit = factor_claimed
                    if expected is not None and factor_claimed == expected:
                        total_reward += 1.0
                        reason = "Median self-consistent with its own declared per-example factor values"
                    else:
                        total_reward -= 1.5
                        reason = "factor_fit doesn't match the median of its own declared per-example factor values"
            else:
                total_reward -= 0.5
                reason = "Formatting error inside tag"

        elif tag_type == "execution":
            m_factor = _EXEC_FACTOR_RE.match(content)
            m_answer = _EXEC_ANSWER_RE.match(content)

            if m_factor:
                out_op, inp_op, claimed = m_factor.groups()
                expected = None
                # Ground the division against the real (input, output) pair
                # this claim must be describing.
                if example_map.get(inp_op) == out_op:
                    expected = _factor_for_example(inp_op, out_op)
                if expected is not None:
                    if claimed == expected:
                        granted = _apply_ceiling(ceiling_totals, "exec_arith", 0.1)
                        total_reward += granted
                        reason = "Division arithmetic correct" if granted else "Division arithmetic correct (ceiling reached)"
                    else:
                        total_reward -= 1.0
                        reason = "Arithmetic error: factor derivation doesn't check out"
                else:
                    total_reward -= 0.5
                    reason = "Unable to ground division claim against a real given example"

            elif m_answer:
                factor_op, question_op, claimed = m_answer.groups()
                try:
                    factor_display = factor_op.rstrip("0").rstrip(".") or "0"
                    _, mult_result = long_multiplication_lines(factor_display, question_op)
                    expected = round_2dp(mult_result)
                except (ValueError, ArithmeticError):
                    expected = None
                if expected is not None and claimed == expected:
                    if known_ffit is not None and factor_op != known_ffit:
                        total_reward -= 0.5
                        reason = "Arithmetic checks out, but used a factor that doesn't match its declared factor_fit"
                    else:
                        granted = _apply_ceiling(ceiling_totals, "exec_arith", 0.1)
                        total_reward += granted
                        reason = "Final multiplication arithmetic correct" if granted else "Final multiplication arithmetic correct (ceiling reached)"
                else:
                    total_reward -= 1.0
                    reason = "Arithmetic error: answer = factor*input doesn't check out"
            else:
                total_reward -= 0.5
                reason = "Formatting error"

        elif tag_type == "analysis":
            m_marker = _MARKER_RE.match(content)
            m_fvalues = _FVALUES_RE.match(content)
            m_fsorted = _FSORTED_RE.match(content)

            if m_marker:
                inp_sym, out_sym = m_marker.groups()
                if content in seen_markers:
                    reason = "Repeated example marker (no additional credit)"
                elif example_map.get(inp_sym) == out_sym:
                    seen_markers.add(content)
                    total_reward += 0.05
                    reason = "Started analyzing a distinct, real given example"
                else:
                    total_reward -= 0.5
                    reason = "Fabricated example: (input, output) pair not in this problem"

            elif m_fvalues:
                listed = [s.strip() for s in m_fvalues.group(1).split(",")]
                current = list(known_factors.values())
                if listed == current:
                    granted = _apply_ceiling(ceiling_totals, "analysis_recap", 0.1)
                    total_reward += granted
                    reason = "Recap matches its own declared per-example factor values"
                else:
                    total_reward -= 0.5
                    reason = "Recap contradicts its own earlier declared per-example factor values"

            elif m_fsorted:
                listed = [s.strip() for s in m_fsorted.group(1).split(",")]
                expected_sorted = sorted(known_factors.values(), key=float)
                if listed == expected_sorted:
                    granted = _apply_ceiling(ceiling_totals, "analysis_recap", 0.1)
                    total_reward += granted
                    reason = "Sorted recap self-consistent"
                else:
                    total_reward -= 0.5
                    reason = "Sorted recap doesn't match a sort of its own declared factor values"

        elif tag_type == "verification":
            m = _VERIFY_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                inp_sym, ffit_op, _mult_operand, _reproduced_claim, out_op, label = m.groups()

                if inp_sym not in example_map:
                    total_reward -= 1.0
                    reason = "P_fabricated_equation: claims to verify an input not in the problem"
                elif _safe_2dp(out_op) != _safe_2dp(example_map[inp_sym]):
                    total_reward -= 1.0
                    reason = "P_fabricated_equation: claimed output doesn't match this input's real given output"
                elif known_ffit is None or ffit_op != known_ffit:
                    total_reward -= 0.5
                    reason = "Used a factor_fit that doesn't match its own declared state (or none declared yet)"
                else:
                    expected_reproduced = _reproduce_out(known_ffit, inp_sym)
                    really_matches = expected_reproduced == _safe_2dp(example_map[inp_sym])
                    if really_matches and label == "match":
                        if inp_sym in verified_examples:
                            granted = _apply_ceiling(ceiling_totals, "verify_reaffirm", 0.1)
                            total_reward += granted
                            reason = "Re-verified an already-credited example (capped)"
                        else:
                            verified_examples.add(inp_sym)
                            honestly_assessed.add(inp_sym)
                            total_reward += 1.0
                            reason = "Its own declared factor_fit genuinely reproduces this real example's output"
                    elif really_matches and label == "mismatch":
                        total_reward -= 1.0
                        reason = "Dishonest/confused: its own factor_fit actually reproduces this output but labeled mismatch"
                    elif (not really_matches) and label == "match":
                        total_reward -= 2.0
                        reason = "P_false_verify: claimed a match its own factor_fit doesn't actually produce"
                    else:
                        # factor_fit is a median fit, not an exact inverse --
                        # it can legitimately fail to reproduce some examples
                        # even in a fully correct trace. Honestly reporting
                        # that is correct behavior, not a mistake, so it must
                        # not be penalized; it earns a small capped credit
                        # for the honest bookkeeping.
                        if inp_sym not in honestly_assessed:
                            honestly_assessed.add(inp_sym)
                            granted = _apply_ceiling(ceiling_totals, "honest_mismatch", 0.05)
                            total_reward += granted
                            reason = "Honestly flagged that factor_fit doesn't reproduce this example (expected for a median fit)"
                        else:
                            reason = "Re-flagged an already-assessed mismatch (no additional credit)"

        elif tag_type == "conclusion":
            saw_conclusion = True
            matches = re.findall(r"\\boxed\{(.*?)\}", content, re.DOTALL)
            final_str = matches[-1].strip() if matches else None
            final_val: float | None
            try:
                final_val = float(final_str) if final_str is not None else None
            except ValueError:
                final_val = None
            if final_val is not None and abs(final_val - float(expected_answer)) <= _TOLERANCE:
                total_reward += 10.0
                reason = "R_terminal_win (boxed answer within tolerance)"
            else:
                total_reward -= 5.0
                reason = "R_terminal_fail (incorrect, unboxed, or non-numeric answer)"

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

    # Bonus is earned for correctly *assessing* every example against
    # factor_fit, not for factor_fit genuinely reproducing every one -- a
    # median fit over truncated per-example divisions can honestly fail to
    # reproduce some examples even when the trace is entirely correct (see
    # the verification handler above), so requiring all-match here would
    # make full credit unreachable on gold traces.
    horizon_delta = 0.0
    horizon_reason = ""
    if len(honestly_assessed) == num_examples and num_examples:
        horizon_delta = 5.0
        horizon_reason = "Long horizon: every given example was correctly assessed against factor_fit"
    elif honestly_assessed:
        horizon_delta = (len(honestly_assessed) / num_examples) * 2.0
        horizon_reason = f"Long horizon: partial success ({len(honestly_assessed)}/{num_examples} examples correctly assessed)"

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
