"""Gravity: d = k * t^2 reasoning generator.

Emits a ``<step type="...">...</step>`` trace using the same six-tag
vocabulary as ``cipher.py``/``cryptarithm.py`` (plan, analysis, execution,
state_update, verification, conclusion), so ``reward_gravity.py`` can score
it the same way. The long multiplication/division breakdowns stay untagged
scratch work (mirroring cipher.py's untagged "Breaking down into
characters" preamble); only the compact checkpoint claims are tagged.

Follows the foolproof contract, calibrated to the competition's actual
grading rule rather than exact string equality: the official evaluation
metric (see the competition's "Evaluation" page) accepts a prediction "that
matches the ground truth either exactly as a string or within a relative
numerical tolerance of 10^-2", and this repo's original author (huikang,
see CLAUDE.md) grades this exact generator's output the same way in his
published solution. The boxed answer -- our own computed estimate, not a
copy of ``problem.answer`` -- is checked against ``problem.answer`` with
that same ``rel_tol=1e-2`` tolerance before returning, so a caller never
receives a trace whose claimed answer would actually fail the competition's
metric, while still accepting the k-estimation noise the metric itself
tolerates (see CLAUDE.md's gravity solve-rate note for the empirical
before/after: 60.4% -> 100% on real train.csv rows).
"""

from __future__ import annotations

import math

from reasoners.store_types import (
    Problem,
    cast_dp_pair,
    long_division_lines,
    long_multiplication_lines,
    round_2dp,
    truncate_3dp,
    wrap_trace_with_think,
)


def reasoning_gravity(problem: Problem) -> str | None:
    lines: list[str] = []
    lines.append(
        '<step type="plan">We need to determine the falling distance using d = k*t^2. '
        "Let me find k from the examples. "
        "I will put my final answer inside \\boxed{}.</step>"
    )
    lines.append("")
    k_strs: list[str] = []
    per_example: list[tuple[str, str]] = []  # (t, d) for the verification pass below
    for ex in problem.examples:
        t = float(ex.input_value)
        if t > 0:
            t_squared = round(t * t, 4)
            t_sq_full = str(t_squared)
            t_sq_str = truncate_3dp(t_sq_full)
            d_str = truncate_3dp(ex.output_value)

            lines.append(
                f'<step type="analysis">t = {ex.input_value}s, d = {ex.output_value}m:</step>'
            )
            lines.append(f"t^2 = {ex.input_value} * {ex.input_value}:")
            sq_lines, sq_result = long_multiplication_lines(
                ex.input_value, ex.input_value
            )
            lines.extend(sq_lines)
            if sq_result != t_sq_full:
                lines.append(f"= {t_sq_full}")
            lines.append(
                f'<step type="execution">t^2 = {ex.input_value}*{ex.input_value} = {t_sq_full}</step>'
            )
            d_cast, tsq_cast, _, _ = cast_dp_pair(d_str, t_sq_str)
            lines.append(
                f"k = {ex.output_value} / {ex.input_value}^2 "
                f"= {d_str} / {t_sq_full} = {d_cast} / {tsq_cast}"
            )
            div_lines, k_str = long_division_lines(d_cast, tsq_cast)
            lines.extend(div_lines)
            lines.append(f"= {k_str}")
            lines.append(
                f'<step type="execution">k = {ex.output_value}/{t_sq_full} = {k_str}</step>'
            )
            lines.append(f'<step type="state_update">k[{ex.input_value}]->{k_str}</step>')
            k_strs.append(k_str)
            per_example.append((ex.input_value, ex.output_value))
            lines.append("")

    if not k_strs:
        return None

    k_values = [float(s) for s in k_strs]

    # List k values and pick median (for even count, use the smaller middle value)
    k_list_str = ", ".join(k_strs)
    lines.append(f'<step type="analysis">k values: {k_list_str}</step>')
    paired = sorted(zip(k_values, k_strs))
    sorted_k_str = ", ".join(s for _, s in paired)
    lines.append(f'<step type="analysis">k values (sorted): {sorted_k_str}</step>')
    if len(paired) % 2 == 0 and len(paired) >= 2:
        _, k_fit_str = paired[len(paired) // 2 - 1]
    else:
        mid = len(paired) // 2
        _, k_fit_str = paired[mid]
    lines.append(f'<step type="state_update">k_fit->{k_fit_str}</step>')
    k_display = k_fit_str.rstrip("0").rstrip(".") or "0"

    # Verification pass: replay the chosen k_fit against every given example
    # to show it actually reproduces the observed distances (not just that it
    # was picked as a median).
    lines.append("")
    for t_in, d_in in per_example:
        t_sq_full = str(round(float(t_in) * float(t_in), 4))
        _, mult_result = long_multiplication_lines(k_display, t_sq_full)
        reproduced = round_2dp(mult_result)
        # Compare at 2dp (real d values are only given to 2dp precision, so
        # matching beyond that would be spurious), but display the example's
        # own raw d string so it stays byte-identical to the given example
        # value for any downstream cross-check against it.
        label = "match" if reproduced == round_2dp(d_in) else "mismatch"
        lines.append(
            f'<step type="verification">d({t_in}) = {k_fit_str}*{t_sq_full} = '
            f"{reproduced} vs {d_in}: {label}</step>"
        )

    lines.append("")
    lines.append(f"For t = {problem.question}:")
    lines.append(f"t^2 = {problem.question} * {problem.question}:")
    sq_lines, t_sq_str = long_multiplication_lines(problem.question, problem.question)
    lines.extend(sq_lines)
    lines.append(f"= {t_sq_str}")
    lines.append(
        f'<step type="execution">t^2 = {problem.question}*{problem.question} = {t_sq_str}</step>'
    )
    lines.append("")
    lines.append(f"d = {k_display} * {t_sq_str}:")
    mult_lines, mult_result = long_multiplication_lines(k_display, t_sq_str)
    lines.extend(mult_lines)
    # Round to 2 decimal places (real answers are only given to 2dp; rounding
    # rather than truncating matches how the given d values were produced).
    boxed_answer = round_2dp(mult_result)
    lines.append(f"= {boxed_answer}")
    lines.append(
        f'<step type="execution">d = {k_fit_str}*{t_sq_str} = {boxed_answer}</step>'
    )

    if not math.isclose(
        float(boxed_answer), float(problem.answer), rel_tol=1e-2, abs_tol=1e-5
    ):
        return None

    lines.append("")
    lines.append(
        '<step type="conclusion">I will now return the answer in \\boxed{}\n'
        f"The answer in \\boxed{{–}} is \\boxed{{{boxed_answer}}}</step>"
    )
    return wrap_trace_with_think("\n".join(lines))
