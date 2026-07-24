# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone extract of synthetic-reasoning-task solvers and reward functions
(originally pulled from a `huikang_nemotron` parent repo), plus a GRPO
training script meant to run on a Kaggle/Colab GPU. It's a self-contained
`uv`-managed project with no dependency on the parent repo.

## Commands

```
uv sync                                          # install deps into .venv (editable install of `reasoners`)
uv run python kaggle/train_grpo_cipher_kaggle.py # GRPO training, runs locally (falls back off /kaggle/input)
uv run python -m reasoners.monitor_cipher        # generate+score one random cipher trace, print step-by-step reward log
uv run python -m reasoners.monitor_cryptarithm   # generate+score one random cryptarithm trace, same style
uv run python reasoners/run_cipher.py            # minimal example: build a Problem, call reasoning_cipher, print the trace
uv run ruff check .                              # lint (ruff is a dev dependency; no repo-specific ruff config file)
```

There is no test suite yet (`pytest` is declared as a dev dependency in
`pyproject.toml` but no `test_*.py` files exist). Validate changes to a
reasoner or reward function empirically instead, e.g. by running the
relevant `monitor_*` script or writing a throwaway script that loops over
many synthetic problems and checks the reward distribution (see "Validating
reward functions" below).

## Architecture

### The Problem/Example data model

Every solver operates on `reasoners/store_types.py`'s `Problem`/`Example`
dataclasses: a `Problem` has `examples: list[Example]` (each an
`input_value`/`output_value` string pair demonstrating the task's hidden
rule), a `question`, and the ground-truth `answer`. `store_types.py` also
holds shared numeric-formatting helpers (`truncate_3dp`, `pad_dp`,
`cast_dp_pair`, `long_multiplication_lines`, `long_division_lines`) used by
the `gravity` and `unit_conversion` solvers.

`Problem.category` is typed as a `Literal` covering nine values, including
split `_deduce`/`_guess` variants for `cryptarithm` and `equation_numeric`.
**`train.csv`'s `category` column does not follow this** -- it uses the
plain unsplit names (`cryptarithm`, `equation_numeric`, plus
`bit_manipulation`, `cipher`, `numeral`, `unit_conversion`, `gravity`).
The type hint isn't runtime-enforced, but don't assume the two line up when
loading real data; check what a given CSV actually contains.

### Two kinds of reasoner module

Every `reasoners/<task>.py` exports a `reasoning_<task>(problem: Problem) ->
str | None`: a deterministic trace generator, not a solver a model runs at
inference time. All of them follow a **foolproof contract**: if the
generator can't derive an answer it's actually verified against
`problem.answer`, it returns `None` rather than emit a trace that looks
plausible but might be wrong. Never make a generator emit an unverified
trace to "improve coverage" -- a wrong answer in training data is worse than
a missing one.

- **Plain narrative generators** (`numeral.py`, `gravity.py`,
  `unit_conversion.py`, `equation_numeric.py`, `bit_manipulation.py`): free-text
  reasoning, no tag structure, no reward function written for them yet.
  Extending GRPO to one of these means designing a reward function from
  scratch, not reusing the tag vocabulary below.
- **Tagged generators** (`cipher.py`, `cryptarithm.py`): emit
  `<step type="...">...</step>` traces using only six semantic tags --
  `plan`, `analysis`, `verification`, `execution`, `state_update`,
  `conclusion` -- shared across both so a similarly-shaped reward function
  can score either. These are the only two tasks with a matching
  `reward_<task>.py`. Keep any new tagged generator to this same vocabulary
  rather than inventing new tag types, so the reward-scoring approach stays
  transferable.

### How training data is actually generated (don't use train.csv as an oracle source)

`train.csv` (9500 rows, columns `id, prompt, answer, category`) is a fixed
dataset for evaluating solvers, not a source of dense per-step ground truth.
Its cryptarithm rows in particular are frequently unsolvable by the
deterministic CSP solver in `cryptarithm.py` (~8.5% solve rate) since a
handful of example equations often doesn't uniquely pin the symbol->digit
mapping. Per-category prompt formats differ and need to be parsed
accordingly: cipher examples are `"ciphertext" -> "plaintext"` lines;
cryptarithm examples are `SYMBOLS = SYMBOLS` lines (`A0 A1 op B0 B1 = R`);
both end with a `Now, ...: <question>` line.

Both `monitor_cipher.py` and `monitor_cryptarithm.py` instead **construct
synthetic problems top-down** from a fully known ground truth, then run the
deterministic generator to produce a demonstration trace:
- `monitor_cipher.py` / `kaggle/train_grpo_cipher_kaggle.py`: picks a full
  random 26-letter substitution bijection first (`make_random_cipher`),
  then encrypts real dictionary words with it. The oracle handed to
  `reward_cipher.py` is always the complete alphabet mapping.
- `monitor_cryptarithm.py`: picks a full 10-digit symbol bijection plus a
  random operator->operation assignment first, then generates equations from
  it (`make_random_cryptarithm`). The CSP solver in `cryptarithm.py` is only
  used afterward, to produce a demo trace -- it is not the source of truth.

This matters for anyone building a GRPO harness for a new task: don't try to
reverse-engineer an oracle from a solved dataset row; construct the problem
from a known ground truth instead, the same way these two scripts do.

### Why the two reward functions are shaped differently

Both `reward_cipher.py` and `reward_cryptarithm.py` parse the six-tag
vocabulary sequentially, track the trace's own claimed state (flow
awareness -- using a fact before declaring it, or contradicting an earlier
declaration, is penalized even when the fact is correct), grade
match/mismatch verification labels for *honesty* separately from whether
the claim is actually true, and cap cheap-to-repeat reward categories with a
hard per-trace ceiling (`_CEILINGS` dict + `_apply_ceiling` helper) so
farming a category by spamming tags is bounded to a small constant rather
than unbounded. Penalties are never capped or deduplicated. A trace that
never reaches a `conclusion` tag is terminally penalized. Only the *last*
`conclusion` tag in a trace counts, evaluated regardless of how much padding
precedes it (padding is already worthless via the ceilings, so this can't be
exploited by pushing the real answer past a step budget).

Where they differ, and why -- **this is the load-bearing design fact for
this codebase, don't collapse it if you touch either file**:

- `reward_cipher.py` grades individual `state_update`/`execution` claims
  (a cipher-letter -> plain-letter mapping) directly against `oracle_map`,
  because the oracle there is unambiguous: a full 26-letter bijection
  checked against real dictionary words has exactly one correct mapping.
- `reward_cryptarithm.py` does **not** grade individual digit/operator
  claims against `oracle_map`/`oracle_ops` (those params, when passed, are
  used *only* to gate whether a claimed symbol is even part of the puzzle's
  alphabet -- never to judge the claimed value). A cryptarithm puzzle with
  only a handful of example equations is frequently ambiguous: multiple
  internally-consistent symbol->digit assignments can satisfy every given
  equation while disagreeing on symbols that operators like `concat`/
  `rev_concat` don't pin down. Grading against one arbitrarily-chosen oracle
  assignment would reward or punish a trace based on luck -- noise that's
  actively harmful to a GRPO advantage estimate. Instead, per-symbol claims
  are graded on alphabet validity + self-consistency with the trace's own
  earlier claims, and real correctness credit flows from the verification
  steps, which replay the trace's *own* claimed digits/operators against the
  actual given example equations (`examples` param, required) and the
  hidden `expected_answer` (also required) -- both unfalsifiable regardless
  of which valid witness the trace is using.

A consequence of this: `reward_cryptarithm.py`'s per-equation verification
handler deliberately does **not** trust a trace's self-reported
"reconstructed" string against its self-reported "target" string, since both
would be attacker-controlled and the real example outputs are visible in
the prompt -- copying them verbatim would trivially pass a naive check. It
instead recomputes the equation's result from the trace's own
previously-declared `state_update` digits/operator and requires that to
independently reproduce the real given output before awarding credit. Any
new verification-style check added to either reward function should apply
the same test: can this check be satisfied by copying visible prompt text,
without the model having actually derived anything? If so, ground it in
something the model doesn't already have for free (the trace's own prior
declarations, arithmetic that must independently check out, or a value
that's genuinely hidden from the prompt).

### Validating reward functions

There's no test suite, so the pattern used to validate both reward
functions (see git history / prior session work on `reward_cipher.py` and
`reward_cryptarithm.py`) is: generate on the order of a few hundred
synthetic gold traces via the relevant `monitor_*` construction helpers,
score each, and assert none score negative (a gold trace, by the foolproof
contract, is always fully correct, so a negative score means the reward
function itself has a bug). Separately, hand-write adversarial traces to
confirm known exploit shapes score negative/flat: fabricated claims,
repetition-spam of any single tag, self-contradiction, copying visible
prompt text into a self-referential check. Any new reward function should
be validated the same two ways before being trusted.

## Current gaps (from README.md, still accurate as of the last full read)

Only `cipher` and `cryptarithm` have reward functions and `monitor_*`
scripts; only `cipher` has a GRPO training script
(`kaggle/train_grpo_cipher_kaggle.py`). Extending GRPO to `equation_numeric`,
`bit_manipulation`, `gravity`, `numeral`, or `unit_conversion` means writing
both a tagged trace generator (if going the same route) and a matching
`reward_<task>.py` from scratch -- there's no shared reward infrastructure
beyond the pattern described above. `reasoners/dictionary.txt` is present
but not currently imported by any solver -- `wonderland.txt` (77 words) is
the word list actually used by `cipher.py`/`reward_cipher.py`.
