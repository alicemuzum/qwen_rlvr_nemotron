"""Cipher: substitution cipher reasoning generator."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from reasoners.store_types import Problem, wrap_trace_with_think

_WONDERLAND_PATH = Path(__file__).parent / "wonderland.txt"


@lru_cache(maxsize=1)
def _load_wonderland() -> list[str]:
    """Load the Wonderland word list (sorted)."""
    with _WONDERLAND_PATH.open() as f:
        words = [line.strip() for line in f if line.strip()]
    return sorted(words)


def _word_pattern(word: str) -> tuple[int, ...]:
    seen: dict[str, int] = {}
    pattern: list[int] = []
    for char in word:
        if char not in seen:
            seen[char] = len(seen)
        pattern.append(seen[char])
    return tuple(pattern)


def _candidate_words_for_partial(
    partial: str,
    cipher_to_plain: dict[str, str],
    plain_to_cipher: dict[str, str],
    cipher_word: str,
) -> list[str]:
    """Find wonderland words matching a partial decryption with unknowns.

    Candidates must be consistent with cipher_to_plain (bijective mapping).
    """
    candidates: list[str] = []
    target_len = len(partial)
    target_pattern = _word_pattern(cipher_word)

    for word in _load_wonderland():
        if len(word) != target_len:
            continue
        if _word_pattern(word) != target_pattern:
            continue
        match = True
        for i, ch in enumerate(partial):
            if ch != "?" and ch != word[i]:
                match = False
                break
        if not match:
            continue
        # Check forward consistency only (cipher->plain)
        consistent = True
        for cc, wc in zip(cipher_word, word):
            if cc in cipher_to_plain and cipher_to_plain[cc] != wc:
                consistent = False
                break
        if consistent:
            candidates.append(word)

    candidates.sort()
    return candidates


def reasoning_cipher(problem: Problem) -> str | None:
    dash = "–"
    lines: list[str] = []
    lines.append(
        '<step type="plan">We need to find the encryption mapping from the examples. '
        "It looks like a substitution cipher.\n"
        "I will put my final answer inside \\boxed{}.</step>"
    )

    # List all input words (examples + question)
    lines.append("")
    lines.append('<step type="input_parsing">Listing the input words:')
    for ex in problem.examples:
        cipher_words = str(ex.input_value).split()
        lines.append("")
        lines.append(f"【{ex.input_value}】")
        for j, w in enumerate(cipher_words):
            lines.append(f"{'' if j == 0 else ' '}{w}")
    question_words_list = problem.question.split()
    lines.append("")
    lines.append(f"【 {problem.question}】")
    for w in question_words_list:
        lines.append(f" {w}")

    # Break down into characters (examples + question)
    lines.append("")
    lines.append("Breaking down into characters:")
    for ex in problem.examples:
        lines.append("")
        lines.append(f"【{ex.input_value}】")
        for w in str(ex.input_value).split():
            lines.append(f"{dash.join(w)}")
    lines.append("")
    lines.append(f"【 {problem.question}】")
    for w in question_words_list:
        lines.append(f"{dash.join(w)}")
    lines.append("</step>")

    wonderland_words = _load_wonderland()

    cipher_to_plain: dict[str, str] = {}
    for ex in problem.examples:
        cipher_words = str(ex.input_value).split()
        plain_words = str(ex.output_value).split()
        if len(cipher_words) != len(plain_words):
            continue

        lines.append("")
        lines.append("")
        plain_quoted = " ".join(f"【{w}】" for w in plain_words)
        lines.append(
            f'<step type="analysis">【{ex.input_value}】 -> 【{ex.output_value}】 / {plain_quoted}:</step>'
        )
        lines.append("")

        for wi, (cw, pw) in enumerate(zip(cipher_words, plain_words)):
            if len(cw) != len(pw):
                continue
            cw_display = f" {cw}" if wi > 0 else cw
            cipher_dashed = dash.join(cw)
            plain_dashed = dash.join(pw)

            if wi > 0:
                lines.append("")
            lines.append(f"【{cw_display}】->【{pw}】")
            lines.append(
                f'<step type="deduction">{cipher_dashed}->{plain_dashed}</step>'
            )
            for cc, pc in zip(cw, pw):
                if cc not in cipher_to_plain:
                    cipher_to_plain[cc] = pc
                lines.append(f'<step type="state_update">{cc}->{pc}</step>')

    lines.append("")
    all_mappings = "\n".join(
        f"{c}->{cipher_to_plain.get(c, '?')}" for c in "abcdefghijklmnopqrstuvwxyz"
    )
    plain_to_cipher_all = {v: k for k, v in cipher_to_plain.items()}
    inverse_mappings = "\n".join(
        f"{c}->{plain_to_cipher_all.get(c, '?')}" for c in "abcdefghijklmnopqrstuvwxyz"
    )
    lines.append(
        f'<step type="analysis">Mapping so far\n{all_mappings}\nInverse mapping\n{inverse_mappings}</step>'
    )

    unknown_mapping_chars = [
        c for c in "abcdefghijklmnopqrstuvwxyz" if c not in cipher_to_plain
    ]
    mapped_targets = set(cipher_to_plain.values())
    unmapped_targets = sorted(
        c for c in "abcdefghijklmnopqrstuvwxyz" if c not in mapped_targets
    )

    lines.append("")
    nl_unknown = "\n".join(unknown_mapping_chars)
    if nl_unknown:
        lines.append(f'<step type="analysis">Unknown characters\n{nl_unknown}</step>')
    else:
        lines.append('<step type="analysis">Unknown characters</step>')

    lines.append("")
    nl_unmapped = "\n".join(unmapped_targets)
    if nl_unmapped:
        lines.append(
            f'<step type="analysis">Unmapped target letters\n{nl_unmapped}</step>'
        )
    else:
        lines.append('<step type="analysis">Unmapped target letters</step>')

    lines.append("")
    question_words = problem.question.split()
    plain_to_cipher: dict[str, str] = {v: k for k, v in cipher_to_plain.items()}
    decoded_words: list[str] = [""] * len(question_words)
    unknown_words: list[
        tuple[int, str, str, str, str]
    ] = []  # (index, cipher_word, partial, display_partial, orig_dashed)

    for i, cw in enumerate(question_words):
        decrypted_chars: list[str] = []
        display_chars: list[str] = []
        mapping_steps: list[str] = []
        has_unknown = False
        for cc in cw:
            if cc in cipher_to_plain:
                decrypted_chars.append(cipher_to_plain[cc])
                display_chars.append(cipher_to_plain[cc])
                mapping_steps.append(
                    f'<step type="execution">{cc}->{cipher_to_plain[cc]}</step>'
                )
            else:
                decrypted_chars.append("?")
                display_chars.append(f"({cc})")
                mapping_steps.append(f'<step type="execution">{cc}->?</step>')
                has_unknown = True

        partial = "".join(decrypted_chars)
        display_partial = "".join(display_chars)
        step_str = "\n".join(mapping_steps)
        cipher_dashed = dash.join(cw)
        plain_dashed = dash.join(display_chars)

        if i > 0:
            lines.append("")
        if i == 0:
            lines.append(
                f'<step type="plan">Now decrypting 【 {problem.question}】:\n【 {cw}】</step>'
            )
        else:
            lines.append(f'<step type="plan">【 {cw}】</step>')

        if has_unknown:
            lines.append(f'<step type="analysis">{cipher_dashed}</step>')
            lines.append(step_str)
            lines.append(
                f'<step type="verification">{plain_dashed}->【{display_partial}】-> {display_partial}</step>'
            )

            orig_dashed = dash.join(display_chars)
            unknown_words.append((i, cw, partial, display_partial, orig_dashed))
        else:
            lines.append(f'<step type="analysis">{cipher_dashed}</step>')
            lines.append(step_str)
            lines.append(
                f'<step type="verification">{plain_dashed}->【{partial}】-> {partial}</step>'
            )
            decoded_words[i] = partial

    # Collect all unknown cipher chars across all question words
    all_unknown_chars: set[str] = set()
    for _, cw, _, _, _ in unknown_words:
        for cc in cw:
            if cc not in cipher_to_plain:
                all_unknown_chars.add(cc)

    # Show current sentence state
    sentence_parts = []
    for i, cw in enumerate(question_words):
        if decoded_words[i]:
            sentence_parts.append(decoded_words[i])
        else:
            display = dash.join(
                f"({cc})" if cc not in cipher_to_plain else cipher_to_plain[cc]
                for cc in cw
            )
            sentence_parts.append(display)
    lines.append("")
    lines.append(
        f'<step type="verification">The sentence currently is\n{" ".join(sentence_parts)}</step>'
    )

    lines.append("")
    if all_unknown_chars:
        iter_parts = "\n".join(
            f"{c} {'yes' if c in all_unknown_chars else 'no'}"
            for c in sorted(unknown_mapping_chars)
        )
        unknown_in_question = sorted(all_unknown_chars)
        nl_unknown_q = "\n".join(unknown_in_question)
        lines.append(
            f'<step type="analysis">Iterating over the unknown letters to see if they are in the question\n{iter_parts}\n\nThe unknown letters\n{nl_unknown_q}</step>'
        )
        lines.append("")
        lines.append("Let me find the best matching wonderland words:")
    else:
        lines.append(
            '<step type="analysis">Iterating over the unknown letters to see if they are in the question: no unknown letters</step>'
        )

    if unknown_words:
        wonderland_set = set(wonderland_words)
        initial_c2p = dict(cipher_to_plain)

        for idx, cw, _partial_orig, display_partial, orig_dashed in unknown_words:
            # Recompute partial with current mappings (may have new letters from previous words)
            partial = "".join(
                cipher_to_plain[cc] if cc in cipher_to_plain else "?" for cc in cw
            )
            candidates = _candidate_words_for_partial(
                partial, cipher_to_plain, plain_to_cipher, cw
            )
            display_candidates = sorted(candidates)
            if not display_candidates:
                return None

            display_dashed = dash.join(
                f"({cc})" if cc not in cipher_to_plain else cipher_to_plain[cc]
                for cc in cw
            )
            accumulated_new = [
                f"【({cc})】->【{cipher_to_plain[cc]}】"
                for cc in sorted(cipher_to_plain)
                if cc not in initial_c2p
            ]
            lines.append("")
            lines.append(f"【{orig_dashed}】")
            if accumulated_new:
                lines.append(f"New mappings: {', '.join(accumulated_new)}")
            else:
                lines.append("New mappings: none")
            lines.append(f"【{display_dashed}】")

            # Iterate over all wonderland words, checking each against the partial
            target_len = len(cw)
            lines.append(f"The length of the word is {target_len}.")
            for word in wonderland_words:
                wlen = len(word)
                if wlen != target_len:
                    lines.append(f"{word} {wlen} length")
                    continue
                # Letter-by-letter comparison with early stop on mismatch
                word_dashed = dash.join(word)
                comparisons: list[str] = []
                mismatch_found = False
                tentative: dict[str, str] = {}
                mapped_plain = set(cipher_to_plain.values())
                for pos, (wi_char, cc) in enumerate(zip(word, cw)):
                    if cc in cipher_to_plain:
                        pc = cipher_to_plain[cc]
                        if wi_char == pc:
                            comparisons.append(
                                f'<step type="analysis">{pos}【{wi_char}】【{pc}】match</step>'
                            )
                        else:
                            comparisons.append(
                                f'<step type="analysis">{pos}【{pc}】【{wi_char}】unmatchable</step>'
                            )
                            mismatch_found = True
                            break
                    else:
                        if cc in tentative:
                            if tentative[cc] == wi_char:
                                comparisons.append(
                                    f'<step type="analysis">{pos}【{wi_char}】【({cc})】consistent</step>'
                                )
                            else:
                                comparisons.append(
                                    f'<step type="analysis">{pos}【{wi_char}】【({cc})】contradiction</step>'
                                )
                                mismatch_found = True
                                break
                        else:
                            if wi_char in mapped_plain:
                                comparisons.append(
                                    f'<step type="analysis">{pos}【{wi_char}】【({cc})】untargeted</step>'
                                )
                                mismatch_found = True
                                break
                            tentative[cc] = wi_char
                            comparisons.append(
                                f'<step type="analysis">{pos}【{wi_char}】【({cc})】matchable</step>'
                            )

                lines.append(
                    f'<step type="analysis">Checking word: {word} {wlen} 【{word_dashed}】</step>'
                )
                lines.extend(comparisons)
                if not mismatch_found:
                    lines.append(f'<step type="analysis">{len(cw)} all match</step>')

            # Filter candidates: reject those mapping unknown cipher letters to already-mapped plain letters
            unmapped = {
                c
                for c in "abcdefghijklmnopqrstuvwxyz"
                if c not in cipher_to_plain.values()
            }
            remaining = []
            for c in display_candidates:
                bad = False
                for ci, wi in zip(cw, c):
                    if ci not in cipher_to_plain and wi not in unmapped:
                        bad = True
                        break
                if not bad:
                    remaining.append(c)

            # Prefer wonderland words among remaining
            wonderland_remaining = [c for c in remaining if c in wonderland_set]
            if wonderland_remaining:
                chosen = wonderland_remaining[0]
            elif remaining:
                chosen = remaining[0]
            else:
                return None
            best_match_lines = []
            best_match_lines.append(
                f'<step type="verification">Best match: 【{chosen}】</step>'
            )
            decoded_words[idx] = chosen

            # Show resolved dashed display
            resolved_dashed = dash.join(chosen)
            best_match_lines.append(
                f'<step type="verification">【{display_dashed}】->【{resolved_dashed}】</step>'
            )

            # Per-letter breakdown: same vs new mapping
            new_mappings: list[str] = []
            pending_mappings: list[tuple[str, str]] = []
            for cc, pc in zip(cw, chosen):
                if cc in cipher_to_plain:
                    known_plain = cipher_to_plain[cc]
                    best_match_lines.append(
                        f'<step type="execution">【{known_plain}】->【{pc}】same</step>'
                    )
                else:
                    best_match_lines.append(
                        f'<step type="execution">【({cc})】->【{pc}】 new</step>'
                    )
                    if (cc, pc) not in pending_mappings:
                        pending_mappings.append((cc, pc))
                        new_mappings.append(
                            f'<step type="state_update">{cc}->{pc}</step>'
                        )
            for cc, pc in pending_mappings:
                cipher_to_plain[cc] = pc
                plain_to_cipher[pc] = cc

            nl_best_match = "\n".join(best_match_lines)
            lines.append(nl_best_match)

            if new_mappings:
                lines.append('<step type="analysis">Added mappings</step>')
                lines.extend(new_mappings)

    if any(w == "" for w in decoded_words):
        return None

    computed = " ".join(decoded_words)
    lines.append("")
    lines.append(
        f'<step type="conclusion">I will now return the answer in \\boxed{{}}\n'
        f"The answer in \\boxed{{–}} is \\boxed{{{computed}}}</step>"
    )
    return wrap_trace_with_think("\n".join(lines))
