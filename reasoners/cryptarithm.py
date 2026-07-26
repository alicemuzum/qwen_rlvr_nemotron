"""Cryptarithm: symbol-substitution arithmetic reasoning generator.

Puzzle form: A0 A1 <op> B0 B1 = R.  A0, A1, B0, B1 and the characters of R
are symbols drawn from a per-problem alphabet; each symbol stands for a
unique digit 0-9.  The operator symbol (position 2) stands for one of a
small family of operations, consistent for every equation that uses it:
addition, absolute difference, multiplication, concatenation, or reverse
concatenation of the two 2-digit numbers A0A1 and B0B1.

Both the digit mapping and the operator meanings must be inferred jointly
from the examples via constraint search; a single equation never pins them
down on its own.  Because that search is a heuristic over a fixed operation
family, it will not solve every problem in the dataset.  We therefore only
ever emit a trace when the derived answer is checked against
``problem.answer`` -- if the search fails, or lands on the wrong answer, we
return ``None`` rather than emit an unsound trace (mirroring
``reasoning_cipher``'s contract).
"""

from __future__ import annotations

from dataclasses import dataclass

from reasoners.store_types import Problem, wrap_trace_with_think

OP_NAMES: tuple[str, ...] = ("add", "abs_diff", "mul", "concat", "rev_concat")
_MAX_SOLUTIONS = 500
_MAX_NODES = 400_000


def _op_apply(op_id: int, a: int, b: int) -> int:
    if op_id == 0:
        return a + b
    if op_id == 1:
        return abs(a - b)
    if op_id == 2:
        return a * b
    if op_id == 3:
        return a * 100 + b
    return b * 100 + a  # op_id == 4: rev_concat


def _result_digits(op_id: int, value: int) -> tuple[int, ...] | None:
    if op_id in (3, 4):
        if value < 0 or value >= 10000:
            return None
        return (value // 1000, (value // 100) % 10, (value // 10) % 10, value % 10)
    if value < 0:
        return None
    if value == 0:
        return (0,)
    digits: list[int] = []
    while value > 0:
        digits.append(value % 10)
        value //= 10
    return tuple(reversed(digits))


def _feasible_ops(result_len: int) -> list[int]:
    ops: list[int] = []
    if result_len <= 3:
        ops.append(0)  # add
    if result_len <= 2:
        ops.append(1)  # abs_diff
    if result_len <= 4:
        ops.append(2)  # mul
    if result_len == 4:
        ops.extend([3, 4])  # concat, rev_concat
    return ops


@dataclass
class _Eq:
    a0: str
    a1: str
    op: str
    b0: str
    b1: str
    result: tuple[str, ...]
    input_str: str
    output_str: str


def _parse_eq(input_value: str, output_value: str) -> _Eq | None:
    if len(input_value) != 5:
        return None
    return _Eq(
        a0=input_value[0],
        a1=input_value[1],
        op=input_value[2],
        b0=input_value[3],
        b1=input_value[4],
        result=tuple(output_value),
        input_str=input_value,
        output_str=output_value,
    )


class _Solver:
    """Backtracking search over (symbol->digit, operator->operation)."""

    def __init__(self, equations: list[_Eq]):
        self.equations = equations
        self.mapping: dict[str, int] = {}
        self.used: set[int] = set()
        self.op_assign: dict[str, int] = {}
        self.solutions: list[tuple[dict[str, int], dict[str, int]]] = []
        self.nodes = 0
        self.truncated = False

    def solve(self) -> None:
        self._process(0)

    def _vals(self, sym: str) -> tuple[int, ...]:
        if sym in self.mapping:
            return (self.mapping[sym],)
        return tuple(d for d in range(10) if d not in self.used)

    def _assign(self, sym: str, dig: int) -> bool | None:
        if sym in self.mapping:
            return False if self.mapping[sym] == dig else None
        if dig in self.used:
            return None
        self.mapping[sym] = dig
        self.used.add(dig)
        return True

    def _undo(self, sym: str, was_new: bool | None) -> None:
        if was_new is True:
            self.used.discard(self.mapping[sym])
            del self.mapping[sym]

    def _process(self, idx: int) -> None:
        if len(self.solutions) >= _MAX_SOLUTIONS or self.truncated:
            return
        if self.nodes > _MAX_NODES:
            self.truncated = True
            return
        if idx == len(self.equations):
            self.solutions.append((dict(self.mapping), dict(self.op_assign)))
            return

        eq = self.equations[idx]
        feasible = _feasible_ops(len(eq.result))
        if not feasible:
            return

        for d0 in self._vals(eq.a0):
            n0 = self._assign(eq.a0, d0)
            if n0 is None:
                continue
            for d1 in self._vals(eq.a1):
                n1 = self._assign(eq.a1, d1)
                if n1 is None:
                    continue
                left = d0 * 10 + d1
                for d3 in self._vals(eq.b0):
                    n3 = self._assign(eq.b0, d3)
                    if n3 is None:
                        continue
                    for d4 in self._vals(eq.b1):
                        n4 = self._assign(eq.b1, d4)
                        if n4 is None:
                            continue
                        right = d3 * 10 + d4
                        self.nodes += 1

                        candidates = (
                            [self.op_assign[eq.op]]
                            if eq.op in self.op_assign
                            else feasible
                        )
                        for op_id in candidates:
                            value = _op_apply(op_id, left, right)
                            digits = _result_digits(op_id, value)
                            if digits is None or len(digits) != len(eq.result):
                                continue

                            assigns: list[tuple[str, bool | None]] = []
                            ok = True
                            for sym, dig in zip(eq.result, digits):
                                was_new = self._assign(sym, dig)
                                if was_new is None:
                                    ok = False
                                    break
                                assigns.append((sym, was_new))

                            if ok:
                                op_is_new = eq.op not in self.op_assign
                                if op_is_new:
                                    self.op_assign[eq.op] = op_id
                                self._process(idx + 1)
                                if op_is_new:
                                    del self.op_assign[eq.op]

                            for sym, was_new in reversed(assigns):
                                self._undo(sym, was_new)

                            if (
                                len(self.solutions) >= _MAX_SOLUTIONS
                                or self.truncated
                            ):
                                self._undo(eq.b1, n4)
                                self._undo(eq.b0, n3)
                                self._undo(eq.a1, n1)
                                self._undo(eq.a0, n0)
                                return
                        self._undo(eq.b1, n4)
                    self._undo(eq.b0, n3)
                self._undo(eq.a1, n1)
            self._undo(eq.a0, n0)


def _solved(
    problem: Problem, equations: list[_Eq], q: tuple[str, str, str, str, str]
) -> tuple[dict[str, int], dict[str, int]] | None:
    """Search for a witness (mapping, op_assign) whose predicted answer for
    the question matches ``problem.answer``.  Returns None if the search
    space is exhausted (or truncated) without finding one."""
    solver = _Solver(equations)
    solver.solve()

    qa0, qa1, qop, qb0, qb1 = q
    for mapping, op_assign in solver.solutions:
        if qop not in op_assign:
            continue
        if not all(s in mapping for s in (qa0, qa1, qb0, qb1)):
            continue
        left = mapping[qa0] * 10 + mapping[qa1]
        right = mapping[qb0] * 10 + mapping[qb1]
        value = _op_apply(op_assign[qop], left, right)
        digits = _result_digits(op_assign[qop], value)
        if digits is None:
            continue
        digit_to_sym = {d: s for s, d in mapping.items()}
        if not all(d in digit_to_sym for d in digits):
            continue
        predicted = "".join(digit_to_sym[d] for d in digits)
        if predicted == problem.answer:
            return mapping, op_assign
    return None


def _q(s: str) -> str:
    return f"【{s}】"


def _dashed(chars: str) -> str:
    return "–".join(chars)


def _render_eq_structure(eq: _Eq) -> str:
    return (
        f"{_q(eq.input_str)} = {_q(eq.output_str)}\n"
        f"  left operand: {_q(eq.a0)}{_q(eq.a1)}\n"
        f"  operator: {_q(eq.op)}\n"
        f"  right operand: {_q(eq.b0)}{_q(eq.b1)}\n"
        f"  result: {_q(eq.output_str)}"
    )


def reasoning_cryptarithm(problem: Problem) -> str | None:
    equations: list[_Eq] = []
    for ex in problem.examples:
        eq = _parse_eq(str(ex.input_value), str(ex.output_value))
        if eq is None:
            return None
        equations.append(eq)
    if not equations:
        return None

    q_str = str(problem.question)
    if len(q_str) != 5:
        return None
    q = (q_str[0], q_str[1], q_str[2], q_str[3], q_str[4])

    witness = _solved(problem, equations, q)
    if witness is None:
        return None
    mapping, op_assign = witness

    lines: list[str] = []

    lines.append(
        '<step type="plan">Each equation has the form A0 A1 op B0 B1 = R. '
        "A0, A1, B0, B1 and the characters of R are symbols that each stand "
        "for a unique digit 0-9, and the operator symbol stands for one of: "
        "addition, absolute difference, multiplication, concatenation, or "
        "reverse concatenation of the two 2-digit numbers A0A1 and B0B1. "
        "I need to find the digit for every symbol and the operation for "
        "every operator symbol, consistent with all examples, then apply "
        "that to the question. I will put my final answer inside "
        "\\boxed{}.</step>"
    )

    lines.append("")
    lines.append('<step type="analysis">Parsing the examples:')
    for eq in equations:
        lines.append("")
        lines.append(_render_eq_structure(eq))
    lines.append("</step>")

    known_syms: set[str] = set()
    known_ops: set[str] = set()

    for eq in equations:
        lines.append("")
        lines.append(f'<step type="analysis">Solving 【{eq.input_str}】 = 【{eq.output_str}】:</step>')

        for sym in (eq.a0, eq.a1, eq.b0, eq.b1):
            if sym not in known_syms:
                known_syms.add(sym)
                lines.append("")
                lines.append(f'<step type="state_update">{sym}->{mapping[sym]}</step>')
            lines.append("")
            lines.append(f'<step type="execution">{sym}->{mapping[sym]}</step>')

        left = mapping[eq.a0] * 10 + mapping[eq.a1]
        right = mapping[eq.b0] * 10 + mapping[eq.b1]
        lines.append("")
        lines.append(
            f'<step type="execution">left = 10*{mapping[eq.a0]}+{mapping[eq.a1]} = {left}, '
            f'right = 10*{mapping[eq.b0]}+{mapping[eq.b1]} = {right}</step>'
        )

        op_id = op_assign[eq.op]
        op_name = OP_NAMES[op_id]
        if eq.op not in known_ops:
            known_ops.add(eq.op)
            lines.append("")
            for cand_id in _feasible_ops(len(eq.result)):
                cand_value = _op_apply(cand_id, left, right)
                cand_digits = _result_digits(cand_id, cand_value)
                consistent = cand_digits is not None and len(cand_digits) == len(
                    eq.result
                )
                if consistent:
                    for sym, dig in zip(eq.result, cand_digits or ()):
                        if sym in mapping and mapping[sym] != dig:
                            consistent = False
                            break
                verdict = "consistent" if consistent else "contradiction"
                shown_value = cand_value if cand_digits is not None else "n/a"
                lines.append(
                    f'<step type="analysis">Trying {OP_NAMES[cand_id]}({left}, {right}) '
                    f"= {shown_value}: {verdict}</step>"
                )
            lines.append("")
            lines.append(f'<step type="state_update">{eq.op}->{op_name}</step>')

        value = _op_apply(op_id, left, right)
        digits = _result_digits(op_id, value)
        assert digits is not None
        lines.append("")
        lines.append(
            f'<step type="execution">{op_name}({left}, {right}) = {value} -> digits {digits}</step>'
        )

        digit_to_sym_partial = {d: s for s, d in mapping.items()}
        reconstructed = "".join(digit_to_sym_partial.get(d, "?") for d in digits)
        lines.append("")
        lines.append(
            f'<step type="verification">{_dashed(eq.output_str)} -> reconstructed {reconstructed}: '
            f'{"match" if reconstructed == eq.output_str else "mismatch"}</step>'
        )

    lines.append("")
    mapping_table = "\n".join(f"{s}->{d}" for s, d in sorted(mapping.items()))
    op_table = "\n".join(f"{s}->{OP_NAMES[o]}" for s, o in sorted(op_assign.items()))
    lines.append(
        f'<step type="analysis">Mapping so far\n{mapping_table}\nOperators so far\n{op_table}</step>'
    )

    qa0, qa1, qop, qb0, qb1 = q
    lines.append("")
    lines.append(
        f'<step type="plan">Now applying to the question 【{q_str}】:</step>'
    )

    for sym in (qa0, qa1, qb0, qb1):
        lines.append("")
        lines.append(f'<step type="execution">{sym}->{mapping[sym]}</step>')

    q_left = mapping[qa0] * 10 + mapping[qa1]
    q_right = mapping[qb0] * 10 + mapping[qb1]
    lines.append("")
    lines.append(
        f'<step type="execution">left = 10*{mapping[qa0]}+{mapping[qa1]} = {q_left}, '
        f'right = 10*{mapping[qb0]}+{mapping[qb1]} = {q_right}</step>'
    )

    q_op_id = op_assign[qop]
    q_op_name = OP_NAMES[q_op_id]
    lines.append("")
    lines.append(f'<step type="execution">{qop}->{q_op_name}</step>')

    q_value = _op_apply(q_op_id, q_left, q_right)
    q_digits = _result_digits(q_op_id, q_value)
    if q_digits is None:
        return None
    digit_to_sym = {d: s for s, d in mapping.items()}
    if not all(d in digit_to_sym for d in q_digits):
        return None
    computed = "".join(digit_to_sym[d] for d in q_digits)

    lines.append("")
    lines.append(
        f'<step type="verification">{q_op_name}({q_left}, {q_right}) = {q_value} -> '
        f"digits {q_digits} -> symbols {computed}</step>"
    )

    if computed != problem.answer:
        return None

    lines.append("")
    lines.append(
        f'<step type="conclusion">I will now return the answer in \\boxed{{}}\n'
        f"The answer in \\boxed{{–}} is \\boxed{{{computed}}}</step>"
    )
    return wrap_trace_with_think("\n".join(lines))
