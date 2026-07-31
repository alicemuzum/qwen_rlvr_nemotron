"""Cipher reward variant that tolerates a PARTIAL cipher->plain oracle.

`reasoners/reward_cipher.py` (still the reference implementation used by
`monitor_cipher.py` / `train_grpo_cipher_kaggle.py`, and now carrying the two
shared bugfixes described below alongside this file -- see "Two shared fixes"
below) grades every `state_update`/`execution` mapping claim directly against
a complete 26-letter `oracle_map`: `oracle_map.get(cipher) == plain`. That is
only sound when the oracle is genuinely complete (built top-down from a full
random bijection, `monitor_cipher.py`'s pattern) -- a missing key there
always means "wrong", never "unknown". Any prompt whose oracle is instead
*reconstructed* from a handful of visible example sentences (a real
`train.csv` row, or a synthetic row scored outside its own generation
session) will have gaps: a letter absent from every visible example has no
entry, and the ``.get()`` returning `None` gets compared `!= plain`, so every
claim about that letter scores as an outright wrong deduction (P_fatal,
-2.0) or a hallucination penalty, regardless of whether the model actually
got it right. This oracle-completeness fix is what this file exists for, and
it is the ONE difference that remains between this file and
`reward_cipher.py` after the two shared fixes below were ported into both:
what happens when `oracle_map.get(cipher)` is `None` -- previously
indistinguishable from "wrong" (and, in `reward_cipher.py`, still is, since a
full oracle never has a missing key), now its own third state here:

  * A **first** claim about a letter outside the oracle earns nothing (it is
    not gradable for correctness, so it must not be a free way to farm
    reward) but is recorded in `known` as the trace's own claimed state, so
    it CAN still be checked for self-consistency later.
  * **Repeating** that same claim later earns the same small
    reaffirmation-style credit `reward_cipher.py` grants for a verified
    reaffirmation, but under a *separate* ceiling
    (`state_reaffirm_unverified`) so it can never eat into the budget a
    verified reaffirmation would otherwise have.
  * **Contradicting** an earlier unverified claim (declaring a different
    plain letter for the same cipher letter later) is still a flow
    violation and is still penalized, exactly the same way contradicting a
    verified claim would be -- self-consistency is gradable without an
    oracle, unlike correctness.
  * Using an out-of-oracle letter in `execution` without ever declaring it
    (skipping the deduction step) is still penalized -- skipping a required
    step is a flow violation independent of whether the skipped value turns
    out to be right -- but at the milder of `reward_cipher.py`'s two
    skipped-step penalties, since (unlike the complete-oracle case) this
    file cannot confirm the value is actually wrong.

Word-level scoring (`analysis`/`verification`/`conclusion` tags, all graded
against `expected_words` -- the hidden plaintext -- rather than the letter
oracle) is otherwise untouched: it already works from only the question's
own hidden answer, which exists for any prompt, real or synthetic, complete
oracle or not.

Two shared fixes (found while wiring cipher into
`kaggle/train_grpo_tinker.py`'s `_check_reward_discrimination`, which
`reward_cipher.py` had never been exercised against before -- both are
pre-existing bugs in `reward_cipher.py`'s original design, unrelated to
oracle completeness, and both are now applied identically in both files):

  1. **Stale `last_best_match_word` across unrelated words.** The
     "verification" tag's dashed-reconstruction branch (`elif "->" in
     content and "–" in content`) is meant to grade the ONE line that
     immediately follows a "Best match:" dictionary-lookup claim, checking
     it reconstructs that same word. But `reasoners/cipher.py` emits a
     structurally identical "->"-plus-"–" verification line for every
     ordinary, already-fully-known question word too (no dictionary lookup
     involved). In practice this branch turned out to be provably inert in
     `cipher.py`'s own gold traces (loop 1 emits every known-word line
     strictly before loop 2 ever sets `last_best_match_word`), so this fix
     alone did not change any measured gold-trace score -- but it is real
     defense-in-depth against the case (a duplicated/reordered/adversarial
     trace) where it could fire, and costs nothing since it can only ever
     suppress a delta, never add one. The fix gates the branch on adjacency
     in the raw step list (`steps[i-1]` must itself be a "Best match:"
     verification), matching how `cipher.py` actually emits the two lines
     back-to-back with nothing between them.
  2. **`verified_word_count` used as a word-position index (the actual
     cause of the discrimination failure).** The "Best match:" and the
     (for `cipher.py`'s own traces, unreachable) generic-derivation branch
     both graded the resolved word against `expected_words[verified_word_
     count]` -- treating "how many dictionary lookups have succeeded so
     far" as if it were "this word's position in the question". Those are
     only the same number when every word before it also needed a
     dictionary lookup. The moment an earlier question word was already
     decodable straight from the visible examples (extremely common, and
     never itself increments `verified_word_count`), every later
     dictionary lookup got graded against the WRONG expected word and
     failed even when it was actually correct. Measured directly: with the
     original indexing and a full/real oracle, 40/80 sampled gold traces
     scored HIGHER after every verification step was deleted than intact --
     a foolproof-contract gold trace must never score worse for including
     more honest steps, and this was blocking `_check_reward_
     discrimination`'s gate ("do NOT start a paid run") for cipher
     outright. The fix tracks each question word's absolute position
     directly: every ordinary per-word verification line (fix 1's branch,
     when NOT itself a reconstruction of a prior Best-match) is one word,
     encountered in strict question order, so its position is recorded as
     it's seen; if that line still has an unresolved letter (a "("
     placeholder), its position is queued (`unknown_position_queue`) and
     consumed in order by the next "Best match:"/derivation line, so that
     line is graded against the expected word it actually corresponds to.
     Falls back to the old `verified_word_count`-based index when the queue
     is empty (a malformed trace that never emitted the expected per-word
     lines), the same degraded behavior the original had for every trace,
     rather than crashing or creating a new free-reward path. Re-measured
     post-fix (`scripts/validate_reward_cipher_partial.py`): 0/150
     no-verify-scores-higher-than-gold violations, both with a full and a
     reconstructed oracle.

Two public entry points:

  * `evaluate_structured_trace(response_xml, oracle_map, expected_words)` --
    same 3-positional-arg shape as `reward_cipher.py`'s function of the same
    name, for direct use with either a full or a partial oracle.
  * `evaluate_structured_trace_from_examples(response_xml, examples,
    expected_answer)` -- the `(text, examples, expected_answer)` shape every
    other `reward_<task>.py` in this repo uses (see
    `kaggle/train_grpo_tinker.py`'s `ReasonerEnv`), so cipher can be wired
    into that same uniform GRPO harness. It reconstructs a *partial* oracle
    from only the prompt's own visible examples via
    `partial_oracle_from_examples` below (the same first-mapping-wins rule
    `reasoners/cipher.py` itself uses internally, reimplemented here rather
    than imported so this file has no dependency on the data-generation
    module) and splits `expected_answer` into words.
"""

from __future__ import annotations

import re
from typing import Any

# Per-trace ceilings on categories that are cheap to repeat. Only positive
# deltas are subject to these; penalties always apply in full. Identical to
# reward_cipher.py's, plus one new key (see module docstring) so unverified
# reaffirmations can never crowd out the budget for verified ones.
_CEILINGS: dict[str, float] = {
    "state_reaffirm": 1.0,
    "state_reaffirm_unverified": 1.0,
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


def partial_oracle_from_examples(examples: list[tuple[str, str]]) -> dict[str, str]:
    """Reconstruct a (possibly incomplete) cipher->plain map from visible
    example sentence pairs alone -- no access to the full generating
    bijection.

    First-mapping-wins per cipher letter, mirroring `reasoners/cipher.py`'s
    own `cipher_to_plain` construction (its `reasoning_cipher`, lines
    ~110-141) exactly, reimplemented rather than imported so this reward
    module carries no dependency on the data-generation module. Only
    same-length word pairs contribute a mapping, same restriction
    `cipher.py` itself applies (a length mismatch means the sentence
    tokenization is off, not that a letter mapping can be read off it).
    """
    mapping: dict[str, str] = {}
    for cipher_text, plain_text in examples:
        cipher_words = str(cipher_text).split()
        plain_words = str(plain_text).split()
        if len(cipher_words) != len(plain_words):
            continue
        for cw, pw in zip(cipher_words, plain_words):
            if len(cw) != len(pw):
                continue
            for cc, pc in zip(cw, pw):
                mapping.setdefault(cc, pc)
    return mapping


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
    Evaluates a generated reasoning trace for the substitution cipher task,
    tolerating an INCOMPLETE oracle_map (see module docstring for the
    tri-state correct/wrong/unverifiable handling this adds over
    reward_cipher.py's complete-oracle-only original).

    Args:
        response_xml: The raw XML/text output from the model.
        oracle_map: cipher->plain map for as many letters as are actually
            known (may cover all 26, or only the letters visible in a
            prompt's own examples).
        expected_words: The list of ground truth decrypted words for the
            target question.

    Returns:
        tuple: (total_reward, step_logs)
    """
    total_reward = 0.0
    known: dict[str, str] = {}  # agent's own claimed cipher->plain state (flow tracking)
    ceiling_totals: dict[str, float] = {}
    verified_word_count = 0
    last_best_match_word: str | None = None
    saw_conclusion = False
    step_logs: list[dict[str, Any]] = []

    # Third fix (see module docstring): tracks each question word's ABSOLUTE
    # position, separate from verified_word_count (which only counts
    # confirmed-correct dictionary lookups, and drifts the moment any earlier
    # word in the question didn't need one).
    word_position = 0
    unknown_position_queue: list[int] = []

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
                oracle_plain = oracle_map.get(cipher)
                if oracle_plain is not None:
                    if oracle_plain == plain:
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
                elif cipher not in known:
                    known[cipher] = plain
                    reason = "R_unverifiable (new claim outside the visible oracle; not gradable for correctness)"
                elif known[cipher] == plain:
                    granted = _apply_ceiling(ceiling_totals, "state_reaffirm_unverified", 0.05)
                    total_reward += granted
                    reason = (
                        "R_consistency (re-affirmed own unverifiable claim)"
                        if granted > 0
                        else "R_consistency (ceiling reached, no further credit)"
                    )
                else:
                    total_reward -= 1.5
                    reason = "Self-contradiction: conflicts with its own earlier (unverifiable) claim"

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
                oracle_plain = oracle_map.get(cipher)
                if oracle_plain is not None:
                    if oracle_plain == plain:
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
                elif cipher not in known:
                    known[cipher] = plain
                    reason = "R_unverifiable (inline new mapping outside the visible oracle)"
                elif known[cipher] == plain:
                    granted = _apply_ceiling(ceiling_totals, "state_reaffirm_unverified", 0.05)
                    total_reward += granted
                    reason = (
                        "R_consistency (inline re-affirmation, unverifiable)"
                        if granted > 0
                        else "R_consistency (ceiling reached, no further credit)"
                    )
                else:
                    total_reward -= 1.5
                    reason = "Self-contradiction (inline): conflicts with its own earlier (unverifiable) claim"

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
                    else:
                        oracle_plain = oracle_map.get(cipher)
                        if oracle_plain is not None:
                            if oracle_plain == plain:
                                total_reward -= 0.5
                                reason = "Hallucination / Skipped step: got right answer without deduction"
                            else:
                                total_reward -= 1.0
                                reason = "Hallucination and wrong answer"
                        else:
                            total_reward -= 0.5
                            reason = "Hallucination: used an undeclared, unverifiable mapping (skipped the deduction step)"
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
                    # Grade against this word's own absolute question
                    # position (popped from unknown_position_queue below),
                    # not verified_word_count -- see the module docstring's
                    # third fix. Falls back to verified_word_count if the
                    # queue is empty (a malformed/adversarial trace that
                    # never emitted the expected per-word verification
                    # lines), matching the original's only-available index
                    # for that case rather than crashing or skipping.
                    idx = unknown_position_queue.pop(0) if unknown_position_queue else verified_word_count
                    if idx < len(expected_words) and word == expected_words[idx]:
                        total_reward += 1.0
                        verified_word_count = min(verified_word_count + 1, len(expected_words))
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
                # self-consistency check against the word just claimed above --
                # and only when THIS line is the one that actually follows that
                # claim. cipher.py emits an unrelated dashed verification line
                # of the same "->"+"–" shape for every already-fully-known
                # question word too (no dictionary lookup involved), and once
                # any earlier word in the question needed a "Best match:", a
                # later, unrelated known-word line would otherwise be compared
                # against that stale last_best_match_word and almost never
                # match -- a spurious penalty on a correct trace (verified: the
                # unguarded version scores 40/80 sampled gold traces HIGHER
                # after deleting every verification step than intact, which a
                # foolproof-contract gold trace must never do). Gate on
                # adjacency in the raw step list, matching how cipher.py
                # actually emits these two lines back-to-back with nothing
                # between them.
                prev_is_best_match = (
                    i > 0
                    and steps[i - 1][0] == "verification"
                    and steps[i - 1][1].strip().startswith("Best match:")
                )
                if prev_is_best_match:
                    tail = content.rsplit("->", 1)[1]
                    reconstructed = tail.replace("【", "").replace("】", "").replace("–", "").strip()
                    if last_best_match_word is not None and reconstructed == last_best_match_word:
                        total_reward += 0.1
                        reason = "Reconstruction consistent with claimed best match"
                    elif last_best_match_word is not None:
                        total_reward -= 0.5
                        reason = "Reconstruction contradicts its own claimed best match"
                else:
                    # Not adjacent to a Best-match claim: this is the ordinary
                    # one-per-question-word verification line cipher.py emits
                    # for every word (already-known words decoded directly, or
                    # an unknown word's still-unresolved placeholder) --
                    # bookkeeping only, no reward delta. Its order of
                    # appearance IS the word's absolute position in the
                    # question (loop 1 emits exactly one such line per word,
                    # strictly in question order, before any dictionary
                    # lookup happens), so record that position -- and, if this
                    # word still has an unresolved letter (a "(" placeholder
                    # marker), queue it so the matching LATER "Best match:"
                    # line (loop 2, also in question order, but only over the
                    # unknown subset) can be graded against the right index
                    # in expected_words instead of a naive successes-so-far
                    # counter (see the module docstring's third fix).
                    pos = word_position
                    word_position += 1
                    if "(" in content:
                        unknown_position_queue.append(pos)

            elif "->" in content:
                parts = content.split("->")
                if len(parts) >= 2:
                    derived_word = parts[-1].strip()
                    if "(" not in derived_word:
                        idx = unknown_position_queue.pop(0) if unknown_position_queue else verified_word_count
                        if idx < len(expected_words):
                            if derived_word == expected_words[idx]:
                                total_reward += 1.0
                                reason = "Subgoal achieved (derived expected word)"
                                verified_word_count = min(verified_word_count + 1, len(expected_words))
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


def evaluate_structured_trace_from_examples(
    response_xml: str, examples: list[tuple[str, str]], expected_answer: str
) -> tuple[float, list[dict[str, Any]]]:
    """Adapter matching every other `reward_<task>.py`'s
    `(response_xml, examples, expected_answer)` signature (see
    `kaggle/train_grpo_tinker.py`'s `ReasonerEnv`/`REWARDS`), so cipher can be
    scored by the same uniform GRPO harness as the other six categories --
    from only the prompt's own visible examples and hidden answer, exactly
    what exists for any prompt (real `train.csv` row or synthetic), without
    needing `monitor_cipher.py`'s bijection-first construction.
    """
    oracle_map = partial_oracle_from_examples(examples)
    expected_words = expected_answer.split()
    return evaluate_structured_trace(response_xml, oracle_map, expected_words)
