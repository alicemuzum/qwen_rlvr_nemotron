"""Evaluate every deterministic reasoner against real train.csv rows.

For each row, parses the prompt into a Problem matching that category's
prompt format, runs the matching reasoning_<category> generator, extracts the
\\boxed{} answer from the emitted trace (or None if the generator declined /
errored), and compares it against the row's ground-truth answer column.

Writes reasoners_train_csv_eval.csv (repo root) with columns:
sample_id, category, prompt, reasoning, reasoning_answer, ground_truth, correct

Usage: uv run python scripts/eval_train_csv.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reasoners.bit_manipulation import reasoning_bit_manipulation
from reasoners.cipher import reasoning_cipher
from reasoners.cryptarithm import reasoning_cryptarithm
from reasoners.equation_numeric import reasoning_equation_numeric
from reasoners.gravity import reasoning_gravity
from reasoners.numeral import reasoning_numeral
from reasoners.store_types import Example, Problem
from reasoners.unit_conversion import reasoning_unit_conversion

_BOXED_MARKER = "\\boxed{"


def last_boxed(trace: str | None) -> str | None:
    """Extract the final \\boxed{...} answer from a trace.

    Cryptarithm answers are drawn from a symbol alphabet that can itself
    include literal "{"/"}" characters, so a brace-balancing regex like
    r"\\boxed\{([^{}]*)\}" mis-parses those traces. Instead this relies on
    the fixed conclusion template every reasoning_<task> function emits:
    the trace always *ends* with (optionally) "</step>" immediately after
    the final boxed answer's closing brace, with nothing following it. That
    lets the wrapper's closing "}" be identified positionally rather than
    by brace-matching, so arbitrary brace content inside the answer itself
    is preserved untouched.
    """
    if trace is None:
        return None
    body = trace.rstrip()
    if body.endswith("</step>"):
        body = body[: -len("</step>")]
    if not body.endswith("}"):
        return None
    body = body[:-1]
    idx = body.rfind(_BOXED_MARKER)
    if idx == -1:
        return None
    return body[idx + len(_BOXED_MARKER) :]


# ---- Per-category prompt parsers: prompt text -> (examples, question) ----

_CIPHER_EX_RE = re.compile(r"^([a-z]+(?: [a-z]+)*) -> ([a-z]+(?: [a-z]+)*)$", re.MULTILINE)
_CIPHER_Q_RE = re.compile(r"Now, decrypt the following text: (.+)$", re.MULTILINE)


def parse_cipher(prompt: str) -> tuple[list[Example], str | None]:
    examples = [Example(i, o) for i, o in _CIPHER_EX_RE.findall(prompt)]
    m = _CIPHER_Q_RE.search(prompt)
    return examples, (m.group(1).strip() if m else None)


_NUMERAL_EX_RE = re.compile(r"^(\d+) -> (\S+)$", re.MULTILINE)
_NUMERAL_Q_RE = re.compile(r"Now, write the number (\d+) in the Wonderland numeral system\.")


def parse_numeral(prompt: str) -> tuple[list[Example], str | None]:
    examples = [Example(i, o) for i, o in _NUMERAL_EX_RE.findall(prompt)]
    m = _NUMERAL_Q_RE.search(prompt)
    return examples, (m.group(1) if m else None)


_UNIT_EX_RE = re.compile(r"^([\d.]+) m becomes ([\d.]+)$", re.MULTILINE)
_UNIT_Q_RE = re.compile(r"Now, convert the following measurement: ([\d.]+) m")


def parse_unit_conversion(prompt: str) -> tuple[list[Example], str | None]:
    examples = [Example(i, o) for i, o in _UNIT_EX_RE.findall(prompt)]
    m = _UNIT_Q_RE.search(prompt)
    return examples, (m.group(1) if m else None)


_GRAVITY_EX_RE = re.compile(r"For t = ([\d.]+)s, distance = ([\d.]+) m")
_GRAVITY_Q_RE = re.compile(r"Now, determine the falling distance for t = ([\d.]+)s given")


def parse_gravity(prompt: str) -> tuple[list[Example], str | None]:
    examples = [Example(t, d) for t, d in _GRAVITY_EX_RE.findall(prompt)]
    m = _GRAVITY_Q_RE.search(prompt)
    return examples, (m.group(1) if m else None)


_CRYPT_EX_RE = re.compile(r"^(.{5}) = (.+)$", re.MULTILINE)
_CRYPT_Q_RE = re.compile(r"Now, determine the result for: (.+)$", re.MULTILINE)


def parse_cryptarithm(prompt: str) -> tuple[list[Example], str | None]:
    examples = [Example(i, o) for i, o in _CRYPT_EX_RE.findall(prompt)]
    m = _CRYPT_Q_RE.search(prompt)
    return examples, (m.group(1).strip() if m else None)


_EQNUM_EX_RE = re.compile(r"^(\d+\D\d+) = (.+)$", re.MULTILINE)
_EQNUM_Q_RE = re.compile(r"Now, determine the result for: (\d+\D\d+)$", re.MULTILINE)


def parse_equation_numeric(prompt: str) -> tuple[list[Example], str | None]:
    examples = [Example(i, o) for i, o in _EQNUM_EX_RE.findall(prompt)]
    m = _EQNUM_Q_RE.search(prompt)
    return examples, (m.group(1) if m else None)


_BIT_EX_RE = re.compile(r"^([01]{8}) -> ([01]{8})$", re.MULTILINE)
_BIT_Q_RE = re.compile(r"Now, determine the output for: ([01]+)$", re.MULTILINE)


def parse_bit_manipulation(prompt: str) -> tuple[list[Example], str | None]:
    examples = [Example(i, o) for i, o in _BIT_EX_RE.findall(prompt)]
    m = _BIT_Q_RE.search(prompt)
    return examples, (m.group(1) if m else None)


# category -> (parser, reasoning_fn, placeholder Problem.category label)
PARSERS = {
    "cipher": (parse_cipher, reasoning_cipher, "cipher"),
    "cryptarithm": (parse_cryptarithm, reasoning_cryptarithm, "cryptarithm_deduce"),
    "gravity": (parse_gravity, reasoning_gravity, "gravity"),
    "numeral": (parse_numeral, reasoning_numeral, "numeral"),
    "unit_conversion": (parse_unit_conversion, reasoning_unit_conversion, "unit_conversion"),
    "equation_numeric": (parse_equation_numeric, reasoning_equation_numeric, "equation_numeric_deduce"),
    "bit_manipulation": (parse_bit_manipulation, reasoning_bit_manipulation, "bit_manipulation"),
}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    in_path = root / "train.csv"
    out_path = root / "reasoners_train_csv_eval.csv"

    stats = {cat: {"total": 0, "parse_fail": 0, "solved": 0, "correct": 0} for cat in PARSERS}
    errors: dict[str, int] = {cat: 0 for cat in PARSERS}

    with in_path.open(newline="") as f_in, out_path.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(
            ["sample_id", "category", "prompt", "reasoning", "reasoning_answer", "ground_truth", "correct"]
        )

        for row in reader:
            cat = row["category"]
            sid = row["id"]
            prompt = row["prompt"]
            answer = row["answer"]

            if cat not in PARSERS:
                continue
            stats[cat]["total"] += 1
            parse_fn, reasoning_fn, category_label = PARSERS[cat]
            examples, question = parse_fn(prompt)

            if not examples or question is None:
                stats[cat]["parse_fail"] += 1
                writer.writerow([sid, cat, prompt, "", "", answer, False])
                continue

            problem = Problem(
                id=sid,
                category=category_label,
                examples=examples,
                question=question,
                answer=answer,
            )

            try:
                trace = reasoning_fn(problem)
            except Exception:
                trace = None
                errors[cat] += 1

            reasoning_answer = last_boxed(trace)
            if trace is not None:
                stats[cat]["solved"] += 1

            correct = reasoning_answer is not None and reasoning_answer == answer
            if correct:
                stats[cat]["correct"] += 1

            writer.writerow(
                [sid, cat, prompt, trace or "", reasoning_answer or "", answer, correct]
            )

    grand_total = sum(s["total"] for s in stats.values())
    grand_correct = sum(s["correct"] for s in stats.values())
    grand_solved = sum(s["solved"] for s in stats.values())

    print(f"{'category':<18} {'total':>7} {'solved':>7} {'correct':>7} {'accuracy':>9} {'errors':>7}")
    print("-" * 65)
    for cat, s in sorted(stats.items()):
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        print(f"{cat:<18} {s['total']:>7} {s['solved']:>7} {s['correct']:>7} {acc:>8.1%} {errors[cat]:>7}")
    print("-" * 65)
    grand_acc = grand_correct / grand_total if grand_total else 0.0
    print(f"{'TOTAL':<18} {grand_total:>7} {grand_solved:>7} {grand_correct:>7} {grand_acc:>8.1%}")
    print(f"\nWrote {grand_total} rows to {out_path}")


if __name__ == "__main__":
    main()
