"""Numeral: Arabic to Roman reasoning generator.

Emits a ``<step type="...">...</step>`` trace using the same six-tag
vocabulary as ``cipher.py``/``cryptarithm.py``/``gravity.py`` (plan,
analysis, verification, state_update, execution, conclusion), so
``reward_numeral.py`` can score it the same way.

Follows the foolproof contract: the boxed answer is checked against
``problem.answer`` before returning, so a caller never receives an
unverified trace.
"""

from __future__ import annotations

from reasoners.store_types import Problem

ROMAN_VALUES: list[tuple[int, str]] = [
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
]


def _to_roman(n: int) -> str:
    parts: list[str] = []
    remaining = n
    for val, sym in ROMAN_VALUES:
        while remaining >= val:
            parts.append(sym)
            remaining -= val
    return "".join(parts)


def _roman_to_int(s: str) -> int | None:
    """Parse a canonical Roman numeral built purely from ROMAN_VALUES atoms.

    ROMAN_VALUES is already ordered largest-value-first with each
    subtractive pair (CM, CD, XC, XL, IX, IV) ahead of the single-letter
    atom it could otherwise be mistaken as a prefix of (C, X, I), so a
    first-match-wins scan of the table at each position parses correctly
    without extra canonicalization logic. Returns None on any leftover
    unmatched character (invalid or empty input).
    """
    if not s:
        return None
    total = 0
    pos = 0
    while pos < len(s):
        for val, sym in ROMAN_VALUES:
            if s.startswith(sym, pos):
                total += val
                pos += len(sym)
                break
        else:
            return None
    return total


def reasoning_numeral(problem: Problem) -> str | None:
    try:
        n = int(problem.question)
    except ValueError:
        return None

    computed = _to_roman(n)
    if computed != problem.answer:
        return None

    lines: list[str] = []
    lines.append(
        '<step type="plan">We need to convert the number to the Wonderland numeral '
        "system. It looks like Roman numerals. I will put my final answer inside "
        "\\boxed{}.</step>"
    )
    lines.append("")

    for ex in problem.examples:
        try:
            ex_n = int(ex.input_value)
        except ValueError:
            continue
        reconstructed = _to_roman(ex_n)
        label = "match" if reconstructed == ex.output_value else "mismatch"
        lines.append(
            f'<step type="analysis">{ex.input_value} -> {ex.output_value}:</step>'
        )
        lines.append(
            f'<step type="verification">{ex.input_value} -> {reconstructed} vs '
            f"{ex.output_value}: {label}</step>"
        )
    lines.append("")

    remaining = n
    exec_lines: list[str] = []
    atom_order: list[tuple[int, str]] = []
    seen_atoms: set[tuple[int, str]] = set()
    for val, sym in ROMAN_VALUES:
        while remaining >= val:
            new_remaining = remaining - val
            exec_lines.append(
                f'<step type="execution">{remaining} >= {val} -> {sym}, '
                f"remainder {new_remaining}</step>"
            )
            if (val, sym) not in seen_atoms:
                seen_atoms.add((val, sym))
                atom_order.append((val, sym))
            remaining -= val

    for val, sym in atom_order:
        lines.append(f'<step type="state_update">{val}->{sym}</step>')
    lines.append("")
    lines.extend(exec_lines)
    lines.append("")

    lines.append(
        '<step type="conclusion">I will now return the answer in \\boxed{}\n'
        f"The answer in \\boxed{{–}} is \\boxed{{{problem.answer}}}</step>"
    )
    return "\n".join(lines)
