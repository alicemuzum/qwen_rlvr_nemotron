"""Dense, state-aware reward function for substitution-cipher reasoning traces.

Scores a ``<step type="...">...</step>`` trace produced by (or in the style
of) ``reasoners/cipher.py`` against an oracle cipher->plain mapping and the
expected plaintext words for the question.

Design principles (see reasoners/reward_cryptarithm.py for the sibling task):

  * Every positive reward is grounded against either the oracle mapping or
    the agent's *own* previously-declared state -- never against surface
    keywords alone. A step that merely contains the word "match" earns
    nothing; the claim has to be checked.
  * The agent's claimed knowledge is tracked sequentially (``known``), so
    "execution" steps are scored against what the trace itself has already
    established, not the oracle -- using a fact before declaring it (or
    declaring one thing and using another) is a flow violation even when the
    fact happens to be correct.
  * Categories that are cheap to repeat (re-affirming a known mapping,
    admitting ignorance, narrative "analysis" commentary) are capped with a
    hard per-trace ceiling, so spamming them cannot generate unbounded
    reward. Categories that are expensive to fake (a *new* correct
    discovery, a correctly identified dictionary word, the final boxed
    answer) are naturally bounded by problem size and are never capped.
  * Penalties are never capped or deduplicated -- repeating a mistake is
    still a mistake.
  * A trace that never reaches a conclusion tag is terminally penalized, so
    "rack up partial credit and never commit" is not a viable strategy.
"""

from __future__ import annotations

import re
from typing import Any

# Per-trace ceilings on categories that are cheap to repeat. Only positive
# deltas are subject to these; penalties always apply in full.
_CEILINGS: dict[str, float] = {
    "state_reaffirm": 1.0,
    "exec_apply": 2.0,
    "exec_ignore": 1.0,
    "exec_same": 1.0,
    "analysis": 1.0,
    "sentence_tracking": 0.5,
}

# Steps beyond this index earn no further reward or penalty (compute/anti-bloat
# guard; a well-formed trace never gets close to this).
_MAX_STEPS = 500

_ARROW_RE = re.compile(r"^\(?([^\s()【】]+)\)?\s*->\s*\(?([^\s()【】]+)\)?$")
_SAME_RE = re.compile(r"^【([^\W\d_])】\s*->\s*【([^\W\d_])】\s*same$")
_NEW_RE = re.compile(r"^【\(([^\W\d_])\)】\s*->\s*【([^\W\d_])】\s*new$")
_SIMPLE_RE = re.compile(r"^([^\W\d_])\s*->\s*([^\W\d_]|\?)$")


def _apply_ceiling(totals: dict[str, float], key: str, delta: float) -> float:
    """Grant at most enough of *delta* to keep totals[key] under its ceiling.

    Only meant for positive, farmable deltas -- never call with delta <= 0.
    """
    cap = _CEILINGS[key]
    used = totals.get(key, 0.0)
    granted = max(0.0, min(delta, cap - used))
    totals[key] = used + granted
    return granted


def evaluate_structured_trace(
    response_xml: str, oracle_map: dict[str, str], expected_words: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    """
    Evaluates a generated reasoning trace for the substitution cipher task.
    Provides a dense reward signal incorporating long horizon evaluation, state awareness,
    and penalties for unwanted actions (e.g., hallucinated mappings, skipping steps).

    Args:
        response_xml: The raw XML/text output from the model.
        oracle_map: The ground truth cipher-to-plain character mapping.
        expected_words: The list of ground truth decrypted words for the target question.

    Returns:
        tuple: (total_reward, step_logs)
            - total_reward (float): The computed reward score.
            - step_logs (list): Detailed list of dictionaries tracking reward at each step.
    """
    total_reward = 0.0
    known: dict[str, str] = {}  # agent's own claimed cipher->plain state (flow tracking)
    ceiling_totals: dict[str, float] = {}
    verified_word_count = 0
    last_best_match_word: str | None = None
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

        # Only the *last* conclusion tag ever counts, and it always counts --
        # committing to a final answer must never be blocked by the anti-padding
        # step budget below, otherwise padding the trace with junk would let a
        # policy dodge the terminal check entirely.
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
            reason = "Step budget exceeded; no further reward considered"
            step_logs.append(
                {
                    "tag_type": tag_type,
                    "content": content if len(content) < 50 else content[:47] + "...",
                    "reward_delta": 0.0,
                    "total_reward": total_reward,
                    "reason": reason,
                }
            )
            continue

        if tag_type == "state_update":
            m = _ARROW_RE.match(content)
            if not m:
                total_reward -= 0.5
                reason = "Formatting error inside tag"
            else:
                cipher, plain = m.group(1), m.group(2)
                if oracle_map.get(cipher) == plain:
                    if cipher not in known:
                        known[cipher] = plain
                        total_reward += 0.5
                        reason = "R_discovery (correctly deduced a new mapping)"
                    else:
                        granted = _apply_ceiling(ceiling_totals, "state_reaffirm", 0.05)
                        total_reward += granted
                        reason = (
                            "R_consistency (re-affirmed known mapping)"
                            if granted > 0
                            else "R_consistency (ceiling reached, no further credit)"
                        )
                else:
                    total_reward -= 2.0
                    reason = "P_fatal (Logic error: deduced an incorrect mapping)"

        elif tag_type == "execution":
            m_same = _SAME_RE.match(content)
            m_new = _NEW_RE.match(content)
            m_simple = _SIMPLE_RE.match(content)

            if m_same:
                a, b = m_same.groups()
                if a == b:
                    granted = _apply_ceiling(ceiling_totals, "exec_same", 0.05)
                    total_reward += granted
                    reason = (
                        "Self-consistent re-confirmation"
                        if granted > 0
                        else "Self-consistent (ceiling reached, no further credit)"
                    )
                else:
                    total_reward -= 1.0
                    reason = "Claimed 'same' but the two values differ (dishonest claim)"

            elif m_new:
                cipher, plain = m_new.groups()
                if oracle_map.get(cipher) == plain:
                    if cipher not in known:
                        known[cipher] = plain
                        total_reward += 0.5
                        reason = "R_discovery (new mapping declared inline via execution)"
                    else:
                        granted = _apply_ceiling(ceiling_totals, "state_reaffirm", 0.05)
                        total_reward += granted
                        reason = "R_consistency (inline re-affirmation)"
                else:
                    total_reward -= 2.0
                    reason = "P_fatal (inline: deduced an incorrect mapping)"

            elif m_simple:
                cipher, plain = m_simple.groups()
                if plain == "?":
                    if cipher in known:
                        total_reward -= 1.0
                        reason = "Ignored its own established knowledge"
                    else:
                        granted = _apply_ceiling(ceiling_totals, "exec_ignore", 0.1)
                        total_reward += granted
                        reason = (
                            "Correctly admitted ignorance (state awareness)"
                            if granted > 0
                            else "Admitted ignorance (ceiling reached, no further credit)"
                        )
                else:
                    if known.get(cipher) == plain:
                        granted = _apply_ceiling(ceiling_totals, "exec_apply", 0.2)
                        total_reward += granted
                        reason = (
                            "R_apply (correctly applied known knowledge)"
                            if granted > 0
                            else "R_apply (ceiling reached, no further credit)"
                        )
                    elif known.get(cipher) is not None:
                        total_reward -= 1.5
                        reason = "Self-contradiction: used a value that conflicts with its own earlier claim"
                    elif oracle_map.get(cipher) == plain:
                        total_reward -= 0.5
                        reason = "Hallucination / Skipped step: got right answer without deduction"
                    else:
                        total_reward -= 1.0
                        reason = "Hallucination and wrong answer"
            else:
                total_reward -= 0.5
                reason = "Formatting error"

        elif tag_type == "analysis":
            delta = 0.0
            if content.startswith("Checking word:"):
                delta = 0.05
                reason = "Initiated dictionary check"
            elif any(
                k in content
                for k in ("match", "unmatchable", "consistent", "contradiction")
            ):
                delta = 0.01
                reason = "Dictionary check string evaluation"
            if delta:
                granted = _apply_ceiling(ceiling_totals, "analysis", delta)
                total_reward += granted
                if granted < delta:
                    reason += " (ceiling reached, reduced/no credit)"

        elif tag_type == "verification":
            if "Best match:" in content:
                m = re.search(r"【(.*?)】", content)
                if m:
                    word = m.group(1).strip()
                    last_best_match_word = word
                    if (
                        verified_word_count < len(expected_words)
                        and word == expected_words[verified_word_count]
                    ):
                        total_reward += 1.0
                        verified_word_count += 1
                        reason = "Correctly identified the word via dictionary"
                    else:
                        total_reward -= 1.5
                        reason = "Identified the wrong word"

            elif "The sentence currently is" in content:
                lines = content.split("\n")
                if len(lines) >= 2:
                    sentence = lines[-1].strip()
                    current_expected = " ".join(expected_words[:verified_word_count])
                    if (
                        sentence.startswith(current_expected)
                        or current_expected in sentence
                    ):
                        granted = _apply_ceiling(ceiling_totals, "sentence_tracking", 0.2)
                        total_reward += granted
                        reason = "Good sentence tracking"
                    else:
                        total_reward -= 0.5
                        reason = "Poor sentence tracking"

            elif "->" in content and "–" in content:
                # Dashed reconstruction line following a "Best match" claim, e.g.
                # 【c-o-l-o-r-(w)-u-l】->【c-o-l-o-r-f-u-l】. Groundable only as a
                # self-consistency check against the word just claimed above.
                tail = content.rsplit("->", 1)[1]
                reconstructed = tail.replace("【", "").replace("】", "").replace("–", "").strip()
                if last_best_match_word is not None and reconstructed == last_best_match_word:
                    total_reward += 0.1
                    reason = "Reconstruction consistent with claimed best match"
                elif last_best_match_word is not None:
                    total_reward -= 0.5
                    reason = "Reconstruction contradicts its own claimed best match"

            elif "->" in content:
                parts = content.split("->")
                if len(parts) >= 2:
                    derived_word = parts[-1].strip()
                    if "(" not in derived_word:
                        if verified_word_count < len(expected_words):
                            if derived_word == expected_words[verified_word_count]:
                                total_reward += 1.0
                                reason = "Subgoal achieved (derived expected word)"
                                verified_word_count += 1
                            else:
                                total_reward -= 1.0
                                reason = "Incorrect full derivation"

        elif tag_type == "conclusion":
            saw_conclusion = True
            matches = re.findall(r"\\boxed\{(.*?)\}", content, re.DOTALL)
            final = matches[-1].strip() if matches else None
            expected = " ".join(expected_words)
            if final is not None and re.sub(r"\s+", " ", final) == expected:
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

    # Long horizon evaluation modifiers
    horizon_delta = 0.0
    horizon_reason = ""
    if verified_word_count == len(expected_words) and expected_words:
        horizon_delta = 5.0
        horizon_reason = "Long horizon: Successfully verified all required subgoals"
    elif verified_word_count > 0:
        horizon_delta = (verified_word_count / len(expected_words)) * 2.0
        horizon_reason = f"Long horizon: Partial success ({verified_word_count}/{len(expected_words)} subgoals)"

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
