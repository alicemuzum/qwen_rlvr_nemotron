"""Dense, state-aware reward function for gravity (d = k*t^2) reasoning traces.

Scores a ``<step type="...">...</step>`` trace produced by (or in the style
of) ``reasoners/gravity.py`` against the problem's ground truth: the given
(t, d) example pairs and the expected final answer.

Assumes example t values are distinct within a problem: both this trace
format's tags (``k[t]->...``, ``d(t) = ...``) and reasoners/gravity.py's own
generator identify an example solely by its t value, with no way to
disambiguate a repeated t.

Why the trace's final fitted k is *not* graded against a single "true" k
(unlike reward_cipher.py, which does grade directly against its oracle):
gravity's k is recovered by taking the median of several per-example
``d / t^2`` divisions, each truncated (not rounded) to 3 decimal places.
That's an estimate, not an exact recovery -- rounding noise in the given d
values means the median can differ slightly from whatever "true" constant
generated the problem. Grading a claimed k_fit against an external oracle
value would reward or punish a trace based on that rounding luck, which is
the same kind of harmful advantage-estimate noise reward_cryptarithm.py's
docstring warns about (there the cause is structural ambiguity; here it's
estimation/rounding error, but the fix is the same: don't grade against an
oracle that isn't exactly recoverable).

Instead:

  * Each *per-example* local k (``k[t]->...``) is graded directly, because
    it -- unlike the final fitted k_fit -- *is* exactly and deterministically
    recomputable from that example's own visible (t, d) pair using the same
    truncating arithmetic reasoners/gravity.py itself uses (imported from
    store_types, so the recomputation is bit-for-bit identical). There is no
    ambiguity to protect against here.
  * The final k_fit is graded only two ways: (a) self-consistency -- does it
    equal the median of the trace's own previously-declared per-example k
    values (flow awareness, mirrors both sibling reward functions); and (b)
    a "reproduce the givens" verification pass -- does k_fit, applied back to
    each given t, actually reproduce that example's given d (recomputed
    independently by this function, never trusting the trace's self-reported
    "reproduced" value, since a trace could otherwise just copy the visible d
    -- same anti-copy discipline as reward_cryptarithm.py's per-equation
    verification).
  * "execution" steps that show arithmetic work (squaring, division,
    multiplication) are graded for self-consistency and capped, since
    they're cheap to pad with correct-looking-but-repeated work.
  * Penalties are never capped or deduplicated. A trace that never reaches a
    conclusion tag is terminally penalized. Only the *last* conclusion tag
    counts.

The conclusion's boxed answer is graded with the competition's own tolerance
rule -- ``math.isclose(rel_tol=1e-2, abs_tol=1e-5)``, i.e. a *relative*
1e-2 tolerance, not a fixed absolute one -- rather than exact string
equality, mirroring reward_unit_conversion.py: k_fit is a
median-of-truncated-divisions estimate, not an exact recovery, so the final
d = k_fit*t^2 can land close to but not byte-identical with the stored
answer even on a fully correct-method trace. Exact-match grading would
inject that same rounding-luck noise into the GRPO advantage estimate that
this module's k_fit-grading discipline above already avoids. A fixed
*absolute* 0.01 tolerance was tried first and is wrong at gravity's scale:
``reasoning_gravity`` itself now gates on the same relative rule (see its
module docstring and CLAUDE.md's "gravity/unit_conversion tolerance-gate
fix"), so its own gold traces can legitimately land up to ~1% off in
absolute terms for larger d -- measured directly, up to 0.28 absolute on an
answer of 1144.60, which a fixed-0.01-tolerance check would have wrongly
scored ``R_terminal_fail`` on 114/1500 (7.6%) sampled gold traces before this
was caught and fixed.
"""

from __future__ import annotations

import math
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
    "k_reaffirm": 0.5,
    "kfit_reaffirm": 0.3,
    "analysis_recap": 0.3,
    "verify_reaffirm": 0.5,
    "honest_mismatch": 0.3,
}

_MAX_STEPS = 500

_REL_TOLERANCE = 1e-2
_ABS_TOLERANCE = 1e-5

_MARKER_RE = re.compile(r"^t = (\S+)s, d = (\S+)m:$")
_KVALUES_RE = re.compile(r"^k values: (.+)$")
_KSORTED_RE = re.compile(r"^k values \(sorted\): (.+)$")
_EXEC_TSQ_RE = re.compile(r"^t\^2 = (\S+)\*(\S+) = (\S+)$")
_EXEC_K_RE = re.compile(r"^k = (\S+)/(\S+) = (\S+)$")
_EXEC_D_RE = re.compile(r"^d = (\S+)\*(\S+) = (\S+)$")
_STATE_K_LOCAL_RE = re.compile(r"^k\[(\S+)\]->(\S+)$")
_STATE_KFIT_RE = re.compile(r"^k_fit->(\S+)$")
_VERIFY_RE = re.compile(
    r"^d\((\S+)\) = (\S+)\*(\S+) = (\S+) vs (\S+): (match|mismatch)$"
)


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


def _t_squared_full(t_str: str) -> str:
    t = float(t_str)
    return str(round(t * t, 4))


def _k_for_example(t_str: str, d_str: str) -> str:
    """Exactly replicates reasoners.gravity's per-example k derivation."""
    t_sq_full = _t_squared_full(t_str)
    t_sq_str = truncate_3dp(t_sq_full)
    d_trunc = truncate_3dp(d_str)
    d_cast, tsq_cast, _, _ = cast_dp_pair(d_trunc, t_sq_str)
    _, k_str = long_division_lines(d_cast, tsq_cast)
    return k_str


def _reproduce_d(k_fit_str: str, t_str: str) -> str:
    """Exactly replicates reasoners.gravity's verification-pass multiplication."""
    k_display = k_fit_str.rstrip("0").rstrip(".") or "0"
    t_sq_full = _t_squared_full(t_str)
    _, mult_result = long_multiplication_lines(k_display, t_sq_full)
    return round_2dp(mult_result)


def _final_d(k_fit_str: str, t_sq_str: str) -> str:
    """Exactly replicates reasoners.gravity's final answer multiplication."""
    k_display = k_fit_str.rstrip("0").rstrip(".") or "0"
    _, mult_result = long_multiplication_lines(k_display, t_sq_str)
    return round_2dp(mult_result)


def _expected_median(k_strs: list[str]) -> str | None:
    if not k_strs:
        return None
    paired = sorted((float(s), s) for s in k_strs)
    n = len(paired)
    idx = n // 2 - 1 if (n % 2 == 0 and n >= 2) else n // 2
    return paired[idx][1]


def evaluate_structured_trace(
    response_xml: str,
    examples: list[tuple[str, str]],
    expected_answer: str,
) -> tuple[float, list[dict[str, Any]]]:
    """
    Evaluates a generated reasoning trace for the gravity (d = k*t^2) task.

    Args:
        response_xml: The raw XML/text output from the model.
        examples: The problem's real (t, d) example pairs -- unfalsifiable
            ground truth used both to recompute exact per-example k values
            and to validate that verification claims refer to a real given
            example, not a fabricated one.
        expected_answer: The ground truth answer string for the question.

    Returns:
        tuple: (total_reward, step_logs)
    """
    total_reward = 0.0
    known_k: dict[str, str] = {}  # trace's own claimed per-example k, by t
    known_kfit: str | None = None
    ceiling_totals: dict[str, float] = {}
    seen_markers: set[str] = set()
    verified_examples: set[str] = set()
    honestly_assessed: set[str] = set()
    example_map = {t: d for t, d in examples}
    num_examples = len(examples)
    examples_matched = 0
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
            m_local = _STATE_K_LOCAL_RE.match(content)
            m_fit = _STATE_KFIT_RE.match(content)

            if m_local:
                t_sym, k_claimed = m_local.groups()
                if t_sym not in example_map:
                    total_reward -= 1.0
                    reason = "Hallucinated t: not one of this problem's given examples"
                elif t_sym in known_k and known_k[t_sym] != k_claimed:
                    total_reward -= 1.5
                    reason = "Self-contradiction: k for this t conflicts with its own earlier claim"
                elif t_sym in known_k:
                    granted = _apply_ceiling(ceiling_totals, "k_reaffirm", 0.05)
                    total_reward += granted
                    reason = "R_consistency (re-affirmed own claim)" if granted else "R_consistency (ceiling reached)"
                else:
                    expected = _k_for_example(t_sym, example_map[t_sym])
                    known_k[t_sym] = k_claimed
                    if k_claimed == expected:
                        total_reward += 0.5
                        reason = "R_discovery (correctly derived this example's k)"
                    else:
                        total_reward -= 2.0
                        reason = "P_fatal (arithmetic error: derived k doesn't match d/t^2)"

            elif m_fit:
                k_claimed = m_fit.group(1)
                if known_kfit is not None and known_kfit != k_claimed:
                    total_reward -= 1.5
                    reason = "Self-contradiction: k_fit conflicts with its own earlier claim"
                elif known_kfit is not None:
                    granted = _apply_ceiling(ceiling_totals, "kfit_reaffirm", 0.05)
                    total_reward += granted
                    reason = "R_consistency (re-affirmed k_fit)" if granted else "R_consistency (ceiling reached)"
                else:
                    expected = _expected_median(list(known_k.values()))
                    known_kfit = k_claimed
                    if expected is not None and k_claimed == expected:
                        total_reward += 1.0
                        reason = "Median self-consistent with its own declared per-example k values"
                    else:
                        total_reward -= 1.5
                        reason = "k_fit doesn't match the median of its own declared per-example k values"
            else:
                total_reward -= 0.5
                reason = "Formatting error inside tag"

        elif tag_type == "execution":
            m_tsq = _EXEC_TSQ_RE.match(content)
            m_k = _EXEC_K_RE.match(content)
            m_d = _EXEC_D_RE.match(content)

            if m_tsq:
                a, b, claimed = m_tsq.groups()
                if a != b:
                    total_reward -= 0.5
                    reason = "Formatting error: squaring claim has mismatched operands"
                else:
                    try:
                        # Numeric tolerance, not exact-string recompute: this
                        # checkpoint is used both for per-example squaring
                        # (rounded to 4dp) and the question's squaring (exact
                        # long multiplication), which are numerically equal
                        # but differ in trailing-zero formatting.
                        ok = abs(float(claimed) - float(a) * float(b)) < 1e-6
                    except ValueError:
                        ok = False
                    if ok:
                        granted = _apply_ceiling(ceiling_totals, "exec_arith", 0.1)
                        total_reward += granted
                        reason = "Squaring arithmetic correct" if granted else "Squaring arithmetic correct (ceiling reached)"
                    else:
                        total_reward -= 1.0
                        reason = "Arithmetic error: t^2 doesn't check out"

            elif m_k:
                d_op, t_sq_op, claimed = m_k.groups()
                expected = None
                # Ground the division against the real (t, d) pair this claim
                # must be describing: a t whose given d matches d_op and whose
                # true t^2 matches the t_sq_op shown alongside it.
                t_sym = next(
                    (
                        t
                        for t, d in example_map.items()
                        if d == d_op and _t_squared_full(t) == t_sq_op
                    ),
                    None,
                )
                if t_sym is not None:
                    expected = _k_for_example(t_sym, d_op)
                if expected is not None:
                    if claimed == expected:
                        granted = _apply_ceiling(ceiling_totals, "exec_arith", 0.1)
                        total_reward += granted
                        reason = "Division arithmetic correct" if granted else "Division arithmetic correct (ceiling reached)"
                    else:
                        total_reward -= 1.0
                        reason = "Arithmetic error: k derivation doesn't check out"
                else:
                    total_reward -= 0.5
                    reason = "Unable to ground division claim against a real given example"

            elif m_d:
                k_op, t_sq_op, claimed = m_d.groups()
                try:
                    k_display = k_op.rstrip("0").rstrip(".") or "0"
                    _, mult_result = long_multiplication_lines(k_display, t_sq_op)
                    expected = round_2dp(mult_result)
                except (ValueError, ArithmeticError):
                    expected = None
                if expected is not None and claimed == expected:
                    if known_kfit is not None and k_op != known_kfit:
                        total_reward -= 0.5
                        reason = "Arithmetic checks out, but used a k that doesn't match its declared k_fit"
                    else:
                        granted = _apply_ceiling(ceiling_totals, "exec_arith", 0.1)
                        total_reward += granted
                        reason = "Final multiplication arithmetic correct" if granted else "Final multiplication arithmetic correct (ceiling reached)"
                else:
                    total_reward -= 1.0
                    reason = "Arithmetic error: d = k*t^2 doesn't check out"
            else:
                total_reward -= 0.5
                reason = "Formatting error"

        elif tag_type == "analysis":
            m_marker = _MARKER_RE.match(content)
            m_kvalues = _KVALUES_RE.match(content)
            m_ksorted = _KSORTED_RE.match(content)

            if m_marker:
                t_sym, d_sym = m_marker.groups()
                if content in seen_markers:
                    reason = "Repeated example marker (no additional credit)"
                elif example_map.get(t_sym) == d_sym:
                    seen_markers.add(content)
                    total_reward += 0.05
                    reason = "Started analyzing a distinct, real given example"
                else:
                    total_reward -= 0.5
                    reason = "Fabricated example: (t, d) pair not in this problem"

            elif m_kvalues:
                listed = [s.strip() for s in m_kvalues.group(1).split(",")]
                current = list(known_k.values())
                if listed == current:
                    granted = _apply_ceiling(ceiling_totals, "analysis_recap", 0.1)
                    total_reward += granted
                    reason = "Recap matches its own declared per-example k values"
                else:
                    total_reward -= 0.5
                    reason = "Recap contradicts its own earlier declared per-example k values"

            elif m_ksorted:
                listed = [s.strip() for s in m_ksorted.group(1).split(",")]
                expected_sorted = sorted(known_k.values(), key=float)
                if listed == expected_sorted:
                    granted = _apply_ceiling(ceiling_totals, "analysis_recap", 0.1)
                    total_reward += granted
                    reason = "Sorted recap self-consistent"
                else:
                    total_reward -= 0.5
                    reason = "Sorted recap doesn't match a sort of its own declared k values"

        elif tag_type == "verification":
            m = _VERIFY_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                t_sym, kfit_op, t_sq_op, _reproduced_claim, d_op, label = m.groups()

                if t_sym not in example_map:
                    total_reward -= 1.0
                    reason = "P_fabricated_equation: claims to verify a t not in the problem"
                elif _safe_2dp(d_op) != _safe_2dp(example_map[t_sym]):
                    total_reward -= 1.0
                    reason = "P_fabricated_equation: claimed d doesn't match this t's real given d"
                elif known_kfit is None or kfit_op != known_kfit:
                    total_reward -= 0.5
                    reason = "Used a k_fit that doesn't match its own declared state (or none declared yet)"
                else:
                    expected_reproduced = _reproduce_d(known_kfit, t_sym)
                    really_matches = expected_reproduced == _safe_2dp(example_map[t_sym])
                    if really_matches and label == "match":
                        if t_sym in verified_examples:
                            granted = _apply_ceiling(ceiling_totals, "verify_reaffirm", 0.1)
                            total_reward += granted
                            reason = "Re-verified an already-credited example (capped)"
                        else:
                            verified_examples.add(t_sym)
                            examples_matched += 1
                            honestly_assessed.add(t_sym)
                            total_reward += 1.0
                            reason = "Its own declared k_fit genuinely reproduces this real example's d"
                    elif really_matches and label == "mismatch":
                        total_reward -= 1.0
                        reason = "Dishonest/confused: its own k_fit actually reproduces this d but labeled mismatch"
                    elif (not really_matches) and label == "match":
                        total_reward -= 2.0
                        reason = "P_false_verify: claimed a match its own k_fit doesn't actually produce"
                    else:
                        # k_fit is a median fit, not an exact inverse -- it can
                        # legitimately fail to reproduce some examples even in
                        # a fully correct trace (truncation is non-invertible).
                        # Honestly reporting that is correct behavior, not a
                        # mistake, so it must not be penalized; it earns a
                        # small capped credit for the honest bookkeeping.
                        if t_sym not in honestly_assessed:
                            honestly_assessed.add(t_sym)
                            granted = _apply_ceiling(ceiling_totals, "honest_mismatch", 0.05)
                            total_reward += granted
                            reason = "Honestly flagged that k_fit doesn't reproduce this example (expected for a median fit)"
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
            if final_val is not None and math.isclose(
                final_val,
                float(expected_answer),
                rel_tol=_REL_TOLERANCE,
                abs_tol=_ABS_TOLERANCE,
            ):
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

    # Bonus is earned for correctly *assessing* every example against k_fit,
    # not for k_fit genuinely reproducing every one -- a median fit over
    # truncated per-example divisions can honestly fail to reproduce some
    # examples even when the trace is entirely correct (see the verification
    # handler above), so requiring all-match here would make full credit
    # unreachable on gold traces.
    horizon_delta = 0.0
    horizon_reason = ""
    if len(honestly_assessed) == num_examples and num_examples:
        horizon_delta = 5.0
        horizon_reason = "Long horizon: every given example was correctly assessed against k_fit"
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
