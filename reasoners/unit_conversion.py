"""Unit conversion: output = factor * input reasoning generator.

Emits a ``<step type="...">...</step>`` trace using the same six-tag
vocabulary as ``gravity.py``/``cipher.py``/``cryptarithm.py`` (plan,
analysis, execution, state_update, verification, conclusion), so
``reward_unit_conversion.py`` can score it the same way. This is
structurally a near-clone of ``gravity.py`` with the ``t^2`` squaring step
removed: the per-example constant here is derived directly from the input
(``factor = output / input``) rather than from its square.

Follows the foolproof contract: the boxed answer is checked against
``problem.answer`` before returning, so a caller never receives an
unverified trace.
"""

from __future__ import annotations

from reasoners.store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    round_2dp,
    truncate_3dp,
    wrap_trace_with_think,
)


def reasoning_unit_conversion(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        '<step type="plan">We need to find a conversion rule that maps the '
        "inputs to outputs. Let me check if it's a linear factor. "
        "I will put my final answer inside \\boxed{}.</step>"
    )
    lines.append("")
    factor_strs: list[str] = []
    per_example: list[tuple[str, str]] = []  # (input, output) for the verification pass below
    for ex in problem.examples:
        inp = float(ex.input_value)
        if inp != 0:
            out_str = truncate_3dp(ex.output_value)
            inp_str = truncate_3dp(ex.input_value)

            lines.append(
                f'<step type="analysis">input = {ex.input_value}, '
                f"output = {ex.output_value}:</step>"
            )
            inp_cast, out_cast, inp_dp, out_dp = cast_dp_pair(inp_str, out_str)
            lines.append(
                f"Casting input to {inp_dp} decimal places, "
                f"output to {out_dp} decimal places: "
                f"{inp_cast} -> {out_cast}"
            )
            lines.append(f"factor = {out_cast} / {inp_cast}")
            # Truncating the factor to only 3 decimal places (long_division_lines'
            # default) is fine for gravity's k (magnitude ~1-20) but is a
            # systematic *undershoot* bias for unit_conversion's factor
            # (magnitude ~0.5-2.0): 3dp there is only 3-4 significant figures,
            # and truncation-toward-zero means every per-example factor -- and
            # therefore the median and the final answer -- is biased low.
            # 6dp keeps the same truncating-division style but removes enough
            # of that bias to match real train.csv's answer precision.
            div_lines, factor_str = long_division_lines(
                out_cast, inp_cast, max_decimal_digits=6
            )
            lines.extend(div_lines)
            lines.append(f"= {factor_str}")
            lines.append(
                f'<step type="execution">factor = {ex.output_value}/'
                f"{ex.input_value} = {factor_str}</step>"
            )
            lines.append(
                f'<step type="state_update">factor[{ex.input_value}]->{factor_str}</step>'
            )
            factor_strs.append(factor_str)
            per_example.append((ex.input_value, ex.output_value))
            lines.append("")

    if not factor_strs:
        return None

    factors = [float(s) for s in factor_strs]

    # List factor values and pick median (for even count, use the smaller middle value)
    f_list_str = ", ".join(factor_strs)
    lines.append(f'<step type="analysis">factor values: {f_list_str}</step>')
    paired = sorted(zip(factors, factor_strs))
    sorted_str = ", ".join(s for _, s in paired)
    lines.append(f'<step type="analysis">factor values (sorted): {sorted_str}</step>')
    if len(paired) % 2 == 0 and len(paired) >= 2:
        _, med_factor_str = paired[len(paired) // 2 - 1]
    else:
        mid = len(paired) // 2
        _, med_factor_str = paired[mid]
    lines.append(f'<step type="state_update">factor_fit->{med_factor_str}</step>')
    factor_display = med_factor_str.rstrip("0").rstrip(".") or "0"

    # Verification pass: replay the chosen factor_fit against every given
    # example to show it actually reproduces the observed outputs (not just
    # that it was picked as a median).
    lines.append("")
    for inp_in, out_in in per_example:
        _, mult_result = long_multiplication_lines(factor_display, inp_in)
        reproduced = round_2dp(mult_result)
        # Compare at 2dp (real output values are only given to 2dp
        # precision, so matching beyond that would be spurious), but display
        # the example's own raw output string so it stays byte-identical to
        # the given example value for any downstream cross-check against it.
        label = "match" if reproduced == round_2dp(out_in) else "mismatch"
        lines.append(
            f'<step type="verification">out({inp_in}) = {med_factor_str}*{inp_in} = '
            f"{reproduced} vs {out_in}: {label}</step>"
        )

    q_str = problem.question
    lines.append("")
    lines.append(f"Converting {q_str}:")
    lines.append(f"{q_str} * {factor_display}:")
    mult_lines, mult_result = long_multiplication_lines(q_str, factor_display)
    lines.extend(mult_lines)
    # Round to 2 decimal places (real answers are only given to 2dp;
    # rounding rather than truncating matches how the given output values
    # were produced).
    boxed_answer = round_2dp(mult_result)
    lines.append(f"= {boxed_answer}")
    lines.append(
        f'<step type="execution">answer = {med_factor_str}*{q_str} = {boxed_answer}</step>'
    )

    if boxed_answer != round_2dp(problem.answer):
        return None

    lines.append("")
    lines.append(
        '<step type="conclusion">I will now return the answer in \\boxed{}\n'
        f"The answer in \\boxed{{–}} is \\boxed{{{problem.answer}}}</step>"
    )
    return wrap_trace_with_think("\n".join(lines))
