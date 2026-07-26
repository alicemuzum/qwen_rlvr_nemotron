"""Dense, state-aware reward function for bit_manipulation reasoning traces.

Scores a ``<step type="...">...</step>`` trace produced by (or in the style
of) ``reasoners/bit_manipulation.py`` against the problem's ground truth: the
given input/output 8-bit example pairs and the expected final 8-bit answer.

Why no oracle parameter (3-arg shape, like reward_equation_numeric.py /
reward_numeral.py / reward_gravity.py rather than reward_cipher.py /
reward_cryptarithm.py): bit_manipulation's hidden rule is a *per-bit*
assignment drawn from a family (identity, NOT, constant, AND/OR/XOR and their
-NOT variants) applied to one or two input bit positions. With only a
handful of examples, more than one internally-consistent per-bit rule can
often reproduce every given example while disagreeing on the question --
exactly the structural-ambiguity risk reward_cryptarithm.py's and
reward_equation_numeric.py's docstrings describe (the generator's own
left/right-run heuristic exists precisely because multiple candidate columns
routinely match). Grading a trace's declared per-bit rule against one
arbitrarily-chosen "true" generating rule would reward or punish it based on
luck, which is harmful noise for a GRPO advantage estimate. So the declared
rule gets only validity + self-consistency credit, never graded against a
generating rule.

Because this generator's tags are pure wrapping around its pre-existing
narrative (see bit_manipulation.py's module docstring -- no verification
step was added), there is no per-example replay-against-the-real-examples
check the way equation_numeric's and cryptarithm's verification tags provide.
The real, ungameable correctness signal here is entirely the conclusion's
boxed answer against ``expected_answer`` (never visible in the prompt). The
execution tag's per-bit credit only confirms *self-consistency* -- that the
applied rule matches what state_update declared, and that evaluating it
against the trace's own self-reported input bits reproduces the trace's own
claimed output bit -- not that the declared rule is actually correct. This is
intentionally the same trust boundary reward_equation_numeric.py accepts for
its execution tag's ``a``/``b`` operands: the terminal check is the anchor,
not the execution self-consistency check.

Design mirrors reward_equation_numeric.py / reward_cryptarithm.py:

  * "execution" is graded against what the trace itself already declared via
    "state_update" (flow awareness): applying a bit's rule before declaring
    it, or applying a rule that doesn't match the declared one, is a flaw.
  * Cheap-to-repeat categories are capped with a hard per-trace ceiling;
    penalties are never capped or deduplicated.
  * A trace that never reaches a conclusion tag is terminally penalized.
  * Only the *last* conclusion tag counts.
  * "analysis" content that echoes the real given examples (the narrative's
    "Output i: ..." / "Input i: ..." lines) is cross-checked against the
    real examples and penalized if fabricated -- this content already existed
    in the pre-tagging narrative, so checking it doesn't add new trace
    content, only a grading rule.
"""

from __future__ import annotations

import re
from typing import Any

from reasoners.bit_manipulation import N_BITS, evaluate_rule_expr, parse_rule_expr

_CEILINGS: dict[str, float] = {
    "plan": 0.3,
    "analysis_presence": 0.3,
    "example_marker": 0.3,
    "rule_reaffirm": 0.5,
    "exec_apply": 1.6,
}

_MAX_STEPS = 500

_OUTPUT_EX_RE = re.compile(rf"^Output (\d+): ([01]{{{N_BITS}}})$", re.MULTILINE)
_INPUT_EX_RE = re.compile(rf"^Input (\d+): ([01]{{{N_BITS}}})$", re.MULTILINE)
_SELECTED_LINE_RE = re.compile(r"^(\d+) (.+)$", re.MULTILINE)
_INPUT_BIT_RE = re.compile(r"^(\d+) ([01])$", re.MULTILINE)
_OUTPUT_APPLY_RE = re.compile(r"^(\d+) (.*?) = .*(\d)$", re.MULTILINE)
_BOXED_RE = re.compile(r"\\boxed\{([01]+)\}")


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
    Evaluates a generated reasoning trace for the bit_manipulation task.

    Args:
        response_xml: The raw XML/text output from the model.
        examples: The problem's real (input_bits, output_bits) example pairs,
            in the same order the generator numbers them -- unfalsifiable
            ground truth, used only to check whether an "analysis" tag's
            echoed example lines are real (not fabricated).
        expected_answer: The ground truth 8-bit answer string for the
            question.

    Returns:
        tuple: (total_reward, step_logs)
    """
    total_reward = 0.0
    known_rules: dict[int, str] = {}
    ceiling_totals: dict[str, float] = {}
    seen_markers: set[str] = set()
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
            granted = _apply_ceiling(ceiling_totals, "analysis_presence", 0.1)
            total_reward += granted
            n_fabricated = 0
            n_new_marker = 0
            for idx_s, bits in _OUTPUT_EX_RE.findall(content):
                idx = int(idx_s)
                marker = f"out{idx}"
                if idx >= len(examples) or examples[idx][1] != bits:
                    total_reward -= 0.5
                    n_fabricated += 1
                elif marker not in seen_markers:
                    seen_markers.add(marker)
                    n_new_marker += 1
                    total_reward += _apply_ceiling(ceiling_totals, "example_marker", 0.05)
            for idx_s, bits in _INPUT_EX_RE.findall(content):
                idx = int(idx_s)
                marker = f"in{idx}"
                if idx >= len(examples) or examples[idx][0] != bits:
                    total_reward -= 0.5
                    n_fabricated += 1
                elif marker not in seen_markers:
                    seen_markers.add(marker)
                    n_new_marker += 1
                    total_reward += _apply_ceiling(ceiling_totals, "example_marker", 0.05)
            reason = "Analysis block present" if granted else "Analysis block (ceiling reached)"
            if n_new_marker:
                reason += f"; referenced {n_new_marker} distinct real example line(s)"
            if n_fabricated:
                reason += f"; {n_fabricated} example line(s) don't match the real given examples"

        elif tag_type == "state_update":
            n_new = 0
            n_reaffirm = 0
            n_contradict = 0
            n_invalid = 0
            for idx_s, expr in _SELECTED_LINE_RE.findall(content):
                idx = int(idx_s)
                if idx >= N_BITS:
                    total_reward -= 1.0
                    n_invalid += 1
                    continue
                try:
                    rule = parse_rule_expr(expr)
                except (ValueError, IndexError):
                    total_reward -= 1.0
                    n_invalid += 1
                    continue
                if any(
                    opnd is not None and not (0 <= opnd < N_BITS)
                    for opnd in (rule.primary, rule.secondary)
                ):
                    total_reward -= 1.0
                    n_invalid += 1
                elif idx in known_rules and known_rules[idx] != expr:
                    total_reward -= 1.5
                    n_contradict += 1
                elif idx in known_rules:
                    total_reward += _apply_ceiling(ceiling_totals, "rule_reaffirm", 0.05)
                    n_reaffirm += 1
                else:
                    known_rules[idx] = expr
                    total_reward += 0.3
                    n_new += 1
            if n_new + n_reaffirm + n_contradict + n_invalid == 0:
                total_reward -= 0.5
                reason = "Formatting error inside tag (no parseable per-bit rule lines)"
            else:
                reason = f"Declared {n_new} new bit rule(s)"
                if n_reaffirm:
                    reason += f", {n_reaffirm} reaffirmed"
                if n_contradict:
                    reason += f", {n_contradict} self-contradicting"
                if n_invalid:
                    reason += f", {n_invalid} invalid"

        elif tag_type == "execution":
            input_bits = ["?"] * N_BITS
            for idx_s, bit in _INPUT_BIT_RE.findall(content):
                idx = int(idx_s)
                if idx < N_BITS:
                    input_bits[idx] = bit
            input_str = "".join(input_bits)
            input_ok = "?" not in input_str

            n_ok = 0
            n_undeclared = 0
            n_inconsistent = 0
            n_wrong = 0
            for idx_s, expr, claimed in _OUTPUT_APPLY_RE.findall(content):
                idx = int(idx_s)
                if idx >= N_BITS:
                    total_reward -= 0.5
                    n_wrong += 1
                    continue
                if idx not in known_rules:
                    total_reward -= 0.5
                    n_undeclared += 1
                    continue
                if known_rules[idx] != expr:
                    total_reward -= 1.0
                    n_inconsistent += 1
                    continue
                expected = None
                if input_ok:
                    try:
                        expected = evaluate_rule_expr(input_str, expr)
                    except (ValueError, IndexError):
                        expected = None
                if expected is not None and expected == claimed:
                    granted = _apply_ceiling(ceiling_totals, "exec_apply", 0.2)
                    total_reward += granted
                    if granted:
                        n_ok += 1
                else:
                    total_reward -= 1.0
                    n_wrong += 1

            reason = f"Applied {n_ok}/{N_BITS} bit(s) consistently with own declared rule"
            if n_undeclared:
                reason += f"; {n_undeclared} applied before declaring"
            if n_inconsistent:
                reason += f"; {n_inconsistent} inconsistent with declared rule"
            if n_wrong:
                reason += f"; {n_wrong} malformed or wrong"
            if not input_ok:
                reason += "; input bits incomplete"

        elif tag_type == "conclusion":
            saw_conclusion = True
            matches = _BOXED_RE.findall(content)
            final = matches[-1] if matches else None
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

    return total_reward, step_logs
