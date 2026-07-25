"""Dense, state-aware reward function for numeral (Arabic->Roman) reasoning traces.

Scores a ``<step type="...">...</step>`` trace produced by (or in the style
of) ``reasoners/numeral.py`` against the problem's ground truth: the given
(arabic, roman) example pairs and the expected final answer.

Why numeral needs no oracle parameter (unlike reward_cipher.py, which grades
directly against a per-problem 26-letter oracle map): a Roman numeral's
correct form is not estimated or structurally ambiguous like gravity's k or
cryptarithm's symbol->digit assignment -- it's exactly and deterministically
recomputable from the arabic integer alone, against a *fixed universal*
table (``ROMAN_VALUES``) that never changes between problems. So every claim
in a numeral trace -- state_update atoms, verification reconstructions,
execution arithmetic, the final boxed answer -- can be graded directly
against that fixed table or recomputed from the problem's own visible data,
with no per-problem oracle to pass in. The signature is therefore the same
3-arg shape as reward_gravity.py's.

The examples are near-decorative here (a model that knows Roman numerals
doesn't need them to answer correctly), so the real, ungameable signal is
(a) the conclusion's boxed answer, (b) verification-label honesty, and (c)
execution arithmetic self-consistency -- not "did it use the examples."
Categories that are cheap to repeat (state_update reaffirms, analysis
markers, plan, execution arithmetic, verification re-checks) are capped
with a hard per-trace ceiling via ``_CEILINGS``/``_apply_ceiling`` so
spamming a category cannot generate unbounded reward. Penalties are never
capped or deduplicated. A trace that never reaches a conclusion tag is
terminally penalized. Only the *last* conclusion tag counts.
"""

from __future__ import annotations

import re
from typing import Any

from reasoners.numeral import ROMAN_VALUES, _roman_to_int, _to_roman

_ROMAN_ATOMS: set[tuple[int, str]] = set(ROMAN_VALUES)

_CEILINGS: dict[str, float] = {
    "plan": 0.3,
    "atom_reaffirm": 0.5,
    "exec_arith": 1.0,
    "verify_reaffirm": 0.5,
    "honest_mismatch": 0.3,
    "analysis_marker": 0.3,
}

_MAX_STEPS = 500

_ANALYSIS_RE = re.compile(r"^(\S+) -> (\S+):$")
_VERIFY_RE = re.compile(r"^(\S+) -> (\S+) vs (\S+): (match|mismatch)$")
_STATE_RE = re.compile(r"^(\S+)->(\S+)$")
_EXEC_RE = re.compile(r"^(\S+) >= (\S+) -> (\S+), remainder (\S+)$")


def _safe_int(s: str) -> int | None:
    """int(), but returns None instead of raising on non-numeric input.

    Needed anywhere the compared value is regex-captured straight out of the
    trace (untrusted, possibly adversarial) rather than out of the
    problem's own data.
    """
    try:
        return int(s)
    except ValueError:
        return None


def _apply_ceiling(totals: dict[str, float], key: str, delta: float) -> float:
    if delta <= 0:
        return delta
    cap = _CEILINGS[key]
    used = totals.get(key, 0.0)
    granted = max(0.0, min(delta, cap - used))
    totals[key] = used + granted
    return granted


def evaluate_structured_trace(
    response_xml: str,
    examples: list[tuple[str, str]],
    expected_answer: str,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Evaluates a generated reasoning trace for the numeral (Arabic->Roman) task.

    Args:
        response_xml: The raw XML/text output from the model.
        examples: The problem's real (arabic, roman) example pairs --
            unfalsifiable ground truth used to validate that analysis/
            verification claims refer to a real given example, not a
            fabricated one.
        expected_answer: The ground truth Roman numeral for the question.

    Returns:
        tuple: (total_reward, step_logs)
    """
    total_reward = 0.0
    known_atoms: dict[int, str] = {}  # trace's own claimed val->sym atoms
    ceiling_totals: dict[str, float] = {}
    seen_markers: set[str] = set()
    verified_examples: set[str] = set()
    honestly_assessed: set[str] = set()
    example_map = {n: roman for n, roman in examples}
    num_examples = len(examples)
    saw_conclusion = False
    step_logs: list[dict[str, Any]] = []
    prev_remainder: int | None = None
    question_int = _roman_to_int(expected_answer)

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

        elif tag_type == "analysis":
            m = _ANALYSIS_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                n_sym, roman_sym = m.groups()
                if content in seen_markers:
                    reason = "Repeated example marker (no additional credit)"
                elif example_map.get(n_sym) == roman_sym:
                    seen_markers.add(content)
                    granted = _apply_ceiling(ceiling_totals, "analysis_marker", 0.05)
                    total_reward += granted
                    reason = (
                        "Started analyzing a distinct, real given example"
                        if granted > 0
                        else "Started analyzing a real example (ceiling reached)"
                    )
                else:
                    total_reward -= 0.5
                    reason = "Fabricated example: (arabic, roman) pair not in this problem"

        elif tag_type == "state_update":
            m = _STATE_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                val_str, sym = m.groups()
                val = _safe_int(val_str)
                if val is None:
                    total_reward -= 2.0
                    reason = "P_fatal (val is not an integer)"
                elif val in known_atoms:
                    if known_atoms[val] == sym:
                        granted = _apply_ceiling(ceiling_totals, "atom_reaffirm", 0.05)
                        total_reward += granted
                        reason = (
                            "R_consistency (re-affirmed own claim)"
                            if granted
                            else "R_consistency (ceiling reached)"
                        )
                    else:
                        total_reward -= 1.5
                        reason = "Self-contradiction: symbol for this value conflicts with its own earlier claim"
                else:
                    known_atoms[val] = sym
                    if (val, sym) in _ROMAN_ATOMS:
                        total_reward += 0.5
                        reason = "R_discovery (correct Roman atom)"
                    else:
                        total_reward -= 2.0
                        reason = "P_fatal (val, sym) is not a valid Roman numeral atom"

        elif tag_type == "execution":
            m = _EXEC_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                remaining_str, val_str, sym, new_str = m.groups()
                remaining_v = _safe_int(remaining_str)
                val = _safe_int(val_str)
                new_v = _safe_int(new_str)
                if prev_remainder is None:
                    chain_ok = question_int is None or remaining_v == question_int
                else:
                    chain_ok = remaining_v == prev_remainder
                ok = (
                    remaining_v is not None
                    and val is not None
                    and new_v is not None
                    and (val, sym) in _ROMAN_ATOMS
                    and remaining_v >= val
                    and new_v == remaining_v - val
                    and chain_ok
                )
                if ok:
                    granted = _apply_ceiling(ceiling_totals, "exec_arith", 0.1)
                    total_reward += granted
                    reason = (
                        "Greedy subtraction step correct and chain-consistent"
                        if granted
                        else "Greedy subtraction step correct (ceiling reached)"
                    )
                    prev_remainder = new_v
                else:
                    total_reward -= 1.0
                    reason = "Arithmetic error or chain break in greedy subtraction step"
                    if new_v is not None:
                        prev_remainder = new_v

        elif tag_type == "verification":
            m = _VERIFY_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                n_sym, _reconstructed_claim, given, label = m.groups()
                if n_sym not in example_map:
                    total_reward -= 1.0
                    reason = "P_fabricated_example: claims to verify an arabic value not in the problem"
                elif given != example_map[n_sym]:
                    total_reward -= 1.0
                    reason = "P_fabricated_example: claimed roman doesn't match this value's real given roman"
                else:
                    n_int = _safe_int(n_sym)
                    really_matches = n_int is not None and _to_roman(n_int) == example_map[n_sym]
                    if really_matches and label == "match":
                        if n_sym in verified_examples:
                            granted = _apply_ceiling(ceiling_totals, "verify_reaffirm", 0.1)
                            total_reward += granted
                            reason = "Re-verified an already-credited example (capped)"
                        else:
                            verified_examples.add(n_sym)
                            honestly_assessed.add(n_sym)
                            total_reward += 1.0
                            reason = "Genuinely reconstructs this real example's Roman numeral"
                    elif really_matches and label == "mismatch":
                        total_reward -= 1.0
                        reason = "Dishonest/confused: this example genuinely reconstructs but labeled mismatch"
                    elif (not really_matches) and label == "match":
                        total_reward -= 2.0
                        reason = "P_false_verify: claimed a match that doesn't actually reconstruct"
                    else:
                        if n_sym not in honestly_assessed:
                            honestly_assessed.add(n_sym)
                            granted = _apply_ceiling(ceiling_totals, "honest_mismatch", 0.05)
                            total_reward += granted
                            reason = "Honestly flagged a genuine mismatch"
                        else:
                            reason = "Re-flagged an already-assessed mismatch (no additional credit)"

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
    if len(honestly_assessed) == num_examples and num_examples:
        horizon_delta = 5.0
        horizon_reason = "Long horizon: every given example was correctly assessed"
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
