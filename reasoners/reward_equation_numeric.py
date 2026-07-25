"""Dense, state-aware reward function for equation_numeric reasoning traces.

Scores a ``<step type="...">...</step>`` trace produced by (or in the style
of) ``reasoners/equation_numeric.py`` against the problem's ground truth: the
given ``A op B = R`` example equations and the expected final answer.

Why no oracle parameter (3-arg shape, like reward_numeral.py / reward_gravity.py
rather than reward_cipher.py / reward_cryptarithm.py): equation_numeric shares
cryptarithm's structural-ambiguity risk -- with only a handful of examples a
coincidentally-fitting *wrong* operation often can't be ruled out, so multiple
internally-consistent rules can reproduce every given example while disagreeing
on the question. Grading a trace's declared operation against one arbitrarily
chosen "true" rule would reward or punish it based on luck, which is actively
harmful noise for a GRPO advantage estimate. So the declared rule gets only
validity + self-consistency credit (recognized operation name, recognized
format, no contradiction with its own earlier claim) and is *never* graded
against a generating rule. Real correctness credit flows entirely from:

  * verification steps that replay the trace's OWN declared rule against the
    real given example inputs and check it independently reproduces the real
    given output (the ambiguity-invariant, unfalsifiable signal), and
  * the conclusion's boxed answer.

Critically -- and this is the load-bearing constraint for this task -- the
real example outputs are *visible in the prompt*, so a naive verification
check ("I reconstructed X, and X matches the given Y") is trivially satisfied
by copying prompt text. This function therefore never trusts a trace's
self-reported "reconstructed" or "given" string: it re-parses the claimed
input, confirms it is one of the real given inputs, recomputes the output
from the trace's own declared rule via the shared ``apply_rule`` (bit-identical
to the generator), and compares *that* to the real given output. Only that
can't be faked without actually having derived a working rule.

Design mirrors reward_cryptarithm.py / reward_numeral.py:

  * "execution" steps are graded against what the trace itself has already
    declared via "state_update" (flow awareness): using a rule before
    declaring it is a flaw.
  * Verification grades the match/mismatch *label* for honesty separately
    from whether the reconstruction is actually correct.
  * Cheap-to-repeat categories are capped with a hard per-trace ceiling;
    penalties are never capped or deduplicated.
  * A trace that never reaches a conclusion tag is terminally penalized.
  * Only the *last* conclusion tag counts.
"""

from __future__ import annotations

import re
from typing import Any

from reasoners.equation_numeric import VALID_FMTS, _all_candidates, apply_rule

# Full set of recognized operation names (used only to gate whether a declared
# operation is even a real member of the family -- never to pick a "correct"
# one). A 2-digit sample exposes the digit-wise operations too.
_ALL_OP_NAMES: frozenset[str] = frozenset(n for n, _ in _all_candidates(12, 34, "12", "34"))

_CEILINGS: dict[str, float] = {
    "plan": 0.3,
    "analysis_parsing": 0.1,
    "example_marker": 0.3,
    "rule_reaffirm": 0.5,
    "exec_apply": 1.0,
    "verify_reaffirm": 0.5,
    "honest_mismatch": 0.5,
}

_MAX_STEPS = 500

_EXAMPLE_RE = re.compile(r"^Example: (\d+)(\D)(\d+) = (.+)$")
_RULE_RE = re.compile(
    r"^(\S+) => op=(.+?); rev_ops=(True|False); rev_res=(True|False); fmt=(\w+)$"
)
_EXEC_RE = re.compile(r"^f\((\d+) (\S+) (\d+)\) = (.+)$")
_VERIFY_RE = re.compile(r"^(\d+)(\D)(\d+) -> (.*?) vs (.*?): (match|mismatch)$")


def _apply_ceiling(totals: dict[str, float], key: str, delta: float) -> float:
    if delta <= 0:
        return delta
    cap = _CEILINGS[key]
    used = totals.get(key, 0.0)
    granted = max(0.0, min(delta, cap - used))
    totals[key] = used + granted
    return granted


def _recompute(rule: tuple[str, bool, bool, str], op_char: str, a: str, b: str) -> str | None:
    """Replay a declared rule against real inputs via the shared generator
    routine, so the reward can never drift from what the generator emits."""
    op_name, rev_ops, rev_res, fmt = rule
    return apply_rule(op_name, rev_ops, rev_res, fmt, op_char, a, b)


def evaluate_structured_trace(
    response_xml: str,
    examples: list[tuple[str, str]],
    expected_answer: str,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Evaluates a generated reasoning trace for the equation_numeric task.

    Args:
        response_xml: The raw XML/text output from the model.
        examples: The problem's real (input_str, output_str) example equation
            pairs -- unfalsifiable ground truth. Used both to validate that
            a claimed example/verification refers to a real given equation
            (not a fabricated one) and, for verification, to recompute the
            expected output from the trace's own declared rule.
        expected_answer: The ground truth answer string for the question.

    Returns:
        tuple: (total_reward, step_logs)
    """
    total_reward = 0.0
    known_rules: dict[str, tuple[str, bool, bool, str]] = {}
    ceiling_totals: dict[str, float] = {}
    seen_markers: set[str] = set()
    verified_inputs: set[str] = set()  # real inputs freshly, correctly verified
    example_map: dict[str, str] = {inp: out for inp, out in examples}
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

        elif tag_type == "analysis":
            if content.startswith("Parsing the examples"):
                granted = _apply_ceiling(ceiling_totals, "analysis_parsing", 0.1)
                total_reward += granted
                reason = "Initial parsing overview" if granted else "Parsing overview (ceiling reached)"
            elif (m := _EXAMPLE_RE.match(content)):
                a, op, b, out = m.groups()
                inp = f"{a}{op}{b}"
                if example_map.get(inp) != out:
                    total_reward -= 0.5
                    reason = "Fabricated example: this A op B = R pair is not in the problem"
                elif inp in seen_markers:
                    reason = "Repeated example marker (no additional credit)"
                else:
                    seen_markers.add(inp)
                    granted = _apply_ceiling(ceiling_totals, "example_marker", 0.05)
                    total_reward += granted
                    reason = (
                        "Referenced a distinct, real given example"
                        if granted
                        else "Referenced a real example (ceiling reached)"
                    )
            else:
                # Free-form commentary (e.g. the fallback note). Neutral: not
                # rewardable, but a legitimate gold trace shouldn't be punished
                # for it either.
                reason = "Narrative analysis (no specific reward)"

        elif tag_type == "state_update":
            m = _RULE_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                op_char, op_name, rev_ops_s, rev_res_s, fmt = m.groups()
                rule = (op_name, rev_ops_s == "True", rev_res_s == "True", fmt)
                if op_name not in _ALL_OP_NAMES or fmt not in VALID_FMTS:
                    total_reward -= 1.0
                    reason = "Invalid rule: unrecognized operation name or format tag"
                elif op_char in known_rules and known_rules[op_char] != rule:
                    total_reward -= 1.5
                    reason = "Self-contradiction: rule conflicts with its own earlier claim for this operator"
                elif op_char in known_rules:
                    granted = _apply_ceiling(ceiling_totals, "rule_reaffirm", 0.05)
                    total_reward += granted
                    reason = "R_consistency (re-affirmed own rule)" if granted else "R_consistency (ceiling reached)"
                else:
                    known_rules[op_char] = rule
                    total_reward += 0.3
                    reason = "New rule declared for this operator (correctness judged at verification)"

        elif tag_type == "execution":
            m = _EXEC_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                a, op_char, b, claimed = m.groups()
                if op_char not in known_rules:
                    total_reward -= 0.5
                    reason = "Applied a rule for this operator before declaring it via state_update"
                else:
                    recomputed = _recompute(known_rules[op_char], op_char, a, b)
                    if recomputed is not None and recomputed == claimed:
                        granted = _apply_ceiling(ceiling_totals, "exec_apply", 0.2)
                        total_reward += granted
                        reason = (
                            "R_apply (result matches its own declared rule)"
                            if granted
                            else "R_apply (ceiling reached)"
                        )
                    else:
                        total_reward -= 1.0
                        reason = "Execution result doesn't match applying its own declared rule"

        elif tag_type == "verification":
            m = _VERIFY_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                a, op_char, b, _claimed_recon, _claimed_given, label = m.groups()
                inp = f"{a}{op_char}{b}"
                if inp not in example_map:
                    total_reward -= 1.0
                    reason = "P_fabricated_example: claims to verify an equation not in the problem"
                elif op_char not in known_rules:
                    if label == "match":
                        total_reward -= 0.5
                        reason = "Claimed a verified match without having declared a rule for this operator"
                    else:
                        reason = "Flagged a mismatch with no rule declared (no credit)"
                else:
                    real_out = example_map[inp]
                    recomputed = _recompute(known_rules[op_char], op_char, a, b)
                    really_matches = recomputed is not None and recomputed == real_out
                    if really_matches and label == "match":
                        if inp in verified_inputs:
                            granted = _apply_ceiling(ceiling_totals, "verify_reaffirm", 0.1)
                            total_reward += granted
                            reason = "Re-verified an already-credited example (capped)"
                        else:
                            verified_inputs.add(inp)
                            total_reward += 1.0
                            reason = "Its own declared rule genuinely reproduces this real example"
                    elif really_matches and label == "mismatch":
                        total_reward -= 1.0
                        reason = "Dishonest/confused: its own rule reproduces this example but labeled mismatch"
                    elif (not really_matches) and label == "match":
                        total_reward -= 2.0
                        reason = "P_false_verify: claimed a match its own declared rule doesn't actually produce"
                    else:
                        granted = _apply_ceiling(ceiling_totals, "honest_mismatch", 0.05)
                        total_reward += granted
                        reason = "Honestly flagged a genuine mismatch"

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

    # Long-horizon bonus over the examples the trace's declared rules actually
    # govern (those whose operator it committed to). A trace that declares no
    # rule, or verifies none of its governed examples, earns nothing here.
    verifiable = {
        inp for inp in example_map if (mm := re.match(r"^\d+(\D)\d+$", inp)) and mm.group(1) in known_rules
    }
    num_verifiable = len(verifiable)
    correctly = len(verified_inputs & verifiable)
    horizon_delta = 0.0
    horizon_reason = ""
    if num_verifiable and correctly == num_verifiable:
        horizon_delta = 5.0
        horizon_reason = "Long horizon: every example governed by a declared rule was verified"
    elif correctly:
        horizon_delta = (correctly / num_verifiable) * 2.0
        horizon_reason = f"Long horizon: partial success ({correctly}/{num_verifiable} governed examples verified)"

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
