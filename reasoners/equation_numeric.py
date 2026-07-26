"""Equation numeric reasoning generator.

Puzzle form: each example is ``A <op> B = R`` where A and B are integers, op
is a single non-digit operator character, and R is the transformed result.
The hidden rule for a given operator is one operation drawn from a fixed
family (see ``_common_candidates``/``_rare_candidates``), optionally with the
operands reversed, the result reversed, and/or a sign-carrying operator
prefix/suffix format (e.g. a negative result rendered as ``"17/"``). The
rule must be inferred from the examples, then applied to the question.

Emits a ``<step type="...">...</step>`` trace using the same six-tag
vocabulary as ``cipher.py``/``cryptarithm.py``/``numeral.py`` (plan,
analysis, verification, state_update, execution, conclusion), so
``reward_equation_numeric.py`` can score it the same way.

Follows the foolproof contract: the applied rule's answer is recomputed via
``apply_rule`` and checked against ``problem.answer`` before returning, so a
caller never receives an unverified trace. Because a handful of examples
don't always pin the operation uniquely (the same ambiguity risk
``cryptarithm.py`` documents), the deterministic search may land on a rule
that reproduces the examples but not the question's answer -- in that case we
``return None`` rather than emit an unsound trace.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from reasoners.store_types import Problem, wrap_trace_with_think

_EXPR_RE = re.compile(r"^(\d+)(\D)(\d+)$")

# Recognized format tags for a rule's sign-carrying operator prefix/suffix.
VALID_FMTS: frozenset[str] = frozenset({"num", "neg_suffix", "neg_prefix", "pre"})


def _common_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Common operations tried first."""
    out: list[tuple[str, str]] = []
    out.append(("concatenation", sa + sb))
    out.append(("reverse concatenation", sb + sa))
    out.append(("addition", str(a + b)))
    out.append(("absolute difference", str(abs(a - b))))
    out.append(("negated absolute difference", str(-abs(a - b))))
    out.append(("subtraction (a-b)", str(a - b)))
    out.append(("reverse subtraction (b-a)", str(b - a)))
    out.append(("multiplication", str(a * b)))
    return out


def _rare_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """Rare operations tried if common ones don't match."""
    out: list[tuple[str, str]] = []
    out.append(("multiply+1", str(a * b + 1)))
    out.append(("multiply-1", str(a * b - 1)))
    out.append(("add+1", str(a + b + 1)))
    out.append(("add-1", str(a + b - 1)))
    out.append(("sub+1", str(a - b + 1)))
    out.append(("sub-1", str(a - b - 1)))
    if a != 0 and b != 0:
        big, small = max(a, b), min(a, b)
        out.append(("max mod min", str(big % small)))
    if b != 0:
        out.append(("integer division (a/b)", str(a // b)))
        out.append(("modulo (a mod b)", str(a % b)))
    if a != 0:
        out.append(("reverse division (b/a)", str(b // a)))
        out.append(("reverse modulo (b mod a)", str(b % a)))
    if len(sa) == 2 and len(sb) == 2:
        d1, d2, d3, d4 = int(sa[0]), int(sa[1]), int(sb[0]), int(sb[1])
        out.append(("digit absolute diff", str(abs(d1 - d3)) + str(abs(d2 - d4))))
        out.append(("digit add mod10", str((d1 + d3) % 10) + str((d2 + d4) % 10)))
        out.append(("digit sub mod10", str((d1 - d3) % 10) + str((d2 - d4) % 10)))
        out.append(("cross multiply", str(d1 * d3 + d2 * d4)))
        out.append(("cross multiply rev", str(d1 * d4 + d2 * d3)))
        out.append(("digit multiply", str(d1 * d3) + str(d2 * d4)))
        out.append(("digit multiply rev", str(d1 * d4) + str(d2 * d3)))
        out.append(("digit sum diff", str((d1 + d2) - (d3 + d4))))
        out.append(("digit sum sum", str((d1 + d2) + (d3 + d4))))
        out.append(("digit product diff", str(d1 * d2 - d3 * d4)))
        out.append(("digit product sum", str(d1 * d2 + d3 * d4)))
        det_val = d1 * d4 - d2 * d3
        out.append(("determinant", str(det_val)))
        out.append(("abs determinant", str(abs(det_val))))
    return out


def _all_candidates(a: int, b: int, sa: str, sb: str) -> list[tuple[str, str]]:
    """All candidates: common first, then rare."""
    return _common_candidates(a, b, sa, sb) + _rare_candidates(a, b, sa, sb)


def _rev(s: str) -> str:
    if s.startswith("-"):
        return "-" + s[1:][::-1]
    return s[::-1]


@dataclass
class FoundOp:
    op_name: str
    rev_ops: bool
    rev_res: bool
    fmt: str
    op_char: str


def _raw_candidate(op_name: str, a: int, b: int, sa: str, sb: str) -> str | None:
    """Look up the raw (pre-reverse, pre-format) result string for *op_name*."""
    for name, res in _all_candidates(a, b, sa, sb):
        if name == op_name:
            return res
    return None


def apply_rule(
    op_name: str,
    rev_ops: bool,
    rev_res: bool,
    fmt: str,
    op_char: str,
    a_str: str,
    b_str: str,
) -> str | None:
    """Pure computation of a rule's output string for inputs (a_str, b_str).

    The declared rule a trace commits to is exactly
    ``(op_name, rev_ops, rev_res, fmt, op_char)``; this function is the single
    source of truth for what that rule produces. It is imported and called
    **bit-identically** by both ``reasoning_equation_numeric`` (trace
    emission) and ``reward_equation_numeric`` (verification replay), so the
    generator and the reward can never drift -- the same must-stay-identical
    discipline the rest of this codebase follows.

    Returns ``None`` if ``op_name`` isn't a recognized operation for these
    operands (or the inputs aren't integer strings), so a malformed rule
    degrades to "doesn't reproduce" rather than raising.
    """
    ta = a_str[::-1] if rev_ops else a_str
    tb = b_str[::-1] if rev_ops else b_str
    try:
        raw = _raw_candidate(op_name, int(ta), int(tb), ta, tb)
    except ValueError:
        return None
    if raw is None:
        return None
    final = _rev(raw) if rev_res else raw
    if fmt == "pre":
        final = op_char + final
    elif fmt == "neg_suffix" and final.startswith("-"):
        final = final[1:] + op_char
    elif fmt == "neg_prefix" and final.startswith("-"):
        final = op_char + final[1:]
    return final


def _detect_format(op_char: str, group: list[tuple[str, str, str]]) -> tuple[str, list[tuple[str, str, str]]]:
    """Reproduce the original format detection for one operator's group.

    Returns (fmt, transformed_group) where transformed_group's outputs have
    any sign-carrying operator prefix/suffix rewritten to a leading ``-`` so
    the numeric search can match against a plain signed value.
    """
    any_neg_suffixed = op_char != "-" and any(
        out.endswith("-") and len(out) > 1 for _, _, out in group
    )
    any_neg_prefixed = op_char != "-" and any(
        out.startswith("-") and len(out) > 1 for _, _, out in group
    )
    any_suffixed = any(out.endswith(op_char) and len(out) > 1 for _, _, out in group)
    any_prefixed = any(out.startswith(op_char) and len(out) > 1 for _, _, out in group)

    fmt = "num"
    transformed = list(group)
    if any_neg_suffixed:
        fmt = "neg_suffix"
        transformed = [
            (a, b, "-" + out[:-1] if out.endswith("-") and len(out) > 1 else out)
            for a, b, out in group
        ]
    elif any_neg_prefixed:
        fmt = "neg_prefix"
    elif any_suffixed:
        fmt = "neg_suffix"
        transformed = [
            (
                a,
                b,
                "-" + out[: -len(op_char)] if out.endswith(op_char) and len(out) > 1 else out,
            )
            for a, b, out in group
        ]
    elif any_prefixed:
        fmt = "neg_prefix"
        transformed = [
            (
                a,
                b,
                "-" + out[len(op_char) :] if out.startswith(op_char) and len(out) > 1 else out,
            )
            for a, b, out in group
        ]
    return fmt, transformed


def _find_rule_for_op(transformed: list[tuple[str, str, str]]) -> tuple[str, bool, bool] | None:
    """Search the operation family x reverse combinations for a rule that maps
    every ``(a, b)`` in *transformed* to its (sign-normalized) expected output.

    Tries common operations before rare ones, and the reverse-combo order the
    original narrative generator used. Returns ``(op_name, rev_ops, rev_res)``
    of the first fully-matching rule, or ``None``.
    """
    if not transformed:
        return None
    for cand_fn in (_common_candidates, _rare_candidates):
        for rev_ops, rev_res in ((True, True), (False, False), (True, False), (False, True)):
            a0, b0, _ = transformed[0]
            ta0 = a0[::-1] if rev_ops else a0
            tb0 = b0[::-1] if rev_ops else b0
            for cand_name, _ in cand_fn(int(ta0), int(tb0), ta0, tb0):
                all_pass = True
                for ax, bx, exp_x in transformed:
                    rax = ax[::-1] if rev_ops else ax
                    rbx = bx[::-1] if rev_ops else bx
                    raw = _raw_candidate(cand_name, int(rax), int(rbx), rax, rbx)
                    if raw is None:
                        all_pass = False
                        break
                    fin = _rev(raw) if rev_res else raw
                    if fin != exp_x:
                        all_pass = False
                        break
                if all_pass:
                    return cand_name, rev_ops, rev_res
    return None


def _rule_repr(op_char: str, rule: FoundOp) -> str:
    return (
        f"{op_char} => op={rule.op_name}; rev_ops={rule.rev_ops}; "
        f"rev_res={rule.rev_res}; fmt={rule.fmt}"
    )


def reasoning_equation_numeric(problem: Problem) -> str | None:
    parsed: list[tuple[str, str, str, str]] = []
    for ex in problem.examples:
        m = _EXPR_RE.fullmatch(str(ex.input_value))
        if not m:
            continue
        a, op, b = m.group(1), m.group(2), m.group(3)
        parsed.append((a, op, b, str(ex.output_value)))

    if not parsed:
        return None

    by_op: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for a, op, b, out in parsed:
        by_op[op].append((a, b, out))

    q_match = _EXPR_RE.fullmatch(str(problem.question))
    if not q_match:
        return None
    qa, q_op, qb = q_match.group(1), q_match.group(2), q_match.group(3)

    # Decide which operator's examples drive the rule. If the question's
    # operator never appears in the examples, fall back to the most common
    # example operator (mirroring the original generator) and apply absolute
    # difference to the question as a last resort.
    fallback = q_op not in by_op
    source_op = max(by_op, key=lambda o: len(by_op[o])) if fallback else q_op

    fmt, transformed = _detect_format(source_op, by_op[source_op])

    if fallback:
        applied = FoundOp(
            op_name="absolute difference",
            rev_ops=False,
            rev_res=False,
            fmt=fmt,
            op_char=q_op,
        )
    else:
        found = _find_rule_for_op(transformed)
        if found is None:
            return None
        op_name, rev_ops, rev_res = found
        applied = FoundOp(
            op_name=op_name,
            rev_ops=rev_ops,
            rev_res=rev_res,
            fmt=fmt,
            op_char=q_op,
        )

    answer = apply_rule(
        applied.op_name, applied.rev_ops, applied.rev_res, applied.fmt, applied.op_char, qa, qb
    )
    # Foolproof contract: never emit a trace whose derived answer we haven't
    # verified against the ground truth. equation_numeric answers are plain
    # strings ("17/", "6644"), not rounded decimals, so this is a direct ==.
    if answer is None or answer != str(problem.answer):
        return None

    # ---- Emit the tagged trace ----
    lines: list[str] = []
    lines.append(
        '<step type="plan">Each example is A op B = R. For the question\'s '
        "operator I need to infer a single rule -- an operation on the two "
        "numbers, possibly with the operands reversed, the result reversed, "
        "and/or a sign-carrying operator prefix/suffix -- that reproduces "
        "every example using that operator, then apply it to the question. "
        "I will put my final answer inside \\boxed{}.</step>"
    )

    lines.append("")
    lines.append('<step type="analysis">Parsing the examples:</step>')
    for a, op, b, out in parsed:
        lines.append("")
        lines.append(f'<step type="analysis">Example: {a}{op}{b} = {out}</step>')

    lines.append("")
    if fallback:
        lines.append(
            f'<step type="analysis">The question operator 【{q_op}】 does not '
            f"appear in the examples; falling back to absolute difference "
            f"for it.</step>"
        )

    lines.append("")
    lines.append(f'<step type="state_update">{_rule_repr(q_op, applied)}</step>')

    # Verification: replay the declared rule against every example that uses
    # the question's operator, and honestly report whether it reproduces the
    # real given output. (In the fallback case there are none.)
    lines.append("")
    for a, op, b, out in parsed:
        if op != q_op:
            continue
        recomputed = apply_rule(
            applied.op_name, applied.rev_ops, applied.rev_res, applied.fmt, applied.op_char, a, b
        )
        label = "match" if recomputed == out else "mismatch"
        lines.append(
            f'<step type="verification">{a}{op}{b} -> {recomputed} vs {out}: {label}</step>'
        )
        lines.append("")

    # Execution + conclusion for the question.
    lines.append(
        f'<step type="execution">f({qa} {q_op} {qb}) = {answer}</step>'
    )
    lines.append("")
    lines.append(
        '<step type="conclusion">I will now return the answer in \\boxed{}\n'
        f"The answer in \\boxed{{–}} is \\boxed{{{answer}}}</step>"
    )
    return wrap_trace_with_think("\n".join(lines))
