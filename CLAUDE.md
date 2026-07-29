# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A standalone extract of synthetic-reasoning-task solvers and reward functions
(originally pulled from a `huikang_nemotron` parent repo), plus a GRPO
training script meant to run on a Kaggle/Colab GPU. It's a self-contained
`uv`-managed project with no dependency on the parent repo.

## Project goal

The end goal is a Qwen model that reliably solves these synthetic reasoning
tasks. The intended pipeline has three stages:

1. **Data generation**: produce `(prompt, reasoning trace)` pairs with the
   step-tag format (`<step type="...">`, six-tag vocabulary) and a final
   `\boxed{}` answer, either from `train.csv` rows or from freshly
   constructed synthetic problems (see "How training data is actually
   generated" below for why synthetic is often the better source).
2. **SFT**: fine-tune Qwen on a small slice of these traces purely to teach
   the output *format* -- step tags, `\boxed{}` -- not to teach it to solve
   the tasks well.
3. **GRPO**: use the `reward_<task>.py` functions to score Qwen's own
   sampled completions and iterate. This is the stage that actually has to
   scale, so the reward functions need to be robust to arbitrary
   (including malformed/adversarial) model output, not just to the gold
   traces the generators themselves produce.

A key implication for stage 3, worth remembering: the reward functions are
generally meant to score a completion for a problem the training harness
*itself just constructed* top-down (mirroring `kaggle/train_grpo_cipher_kaggle.py`'s
pattern), not an arbitrary pre-existing `train.csv` row. Whether a given
reward function can *also* be pointed at a real `train.csv` prompt (e.g. for
held-out eval) varies by task -- see "Why the reward functions are shaped
differently" below, since this is exactly the property that section's
design distinction is about.

## Commands

```
uv sync                                          # install deps into .venv (editable install of `reasoners`)
uv run python kaggle/train_grpo_cipher_kaggle.py # GRPO training, runs locally (falls back off /kaggle/input)
uv run python kaggle/train_sft_kaggle.py --dry-run # SFT: build+report the dataset from synth_sft.jsonl, no GPU needed
uv run python kaggle/train_sft_kaggle.py         # SFT training (stage 2); --help for the CLI overrides
uv run python kaggle/train_sft_tinker.py --dry-run # SFT (stage 2) on Thinking Machines' hosted Tinker API instead of a local GPU, no network call
uv run python kaggle/train_sft_tinker.py --yes   # real (paid) Tinker training run; see "The Tinker SFT training script" below before running
uv run python kaggle/train_grpo_tinker.py --dry-run # GRPO (stage 3) pre-flight checks against the full_0727 SFT checkpoint, no network call
uv run python kaggle/train_grpo_tinker.py --yes  # real (paid) Tinker GRPO run; see "The GRPO training script" below before running
uv run python -m reasoners.monitor_cipher        # generate+score one random cipher trace, print step-by-step reward log
uv run python -m reasoners.monitor_cryptarithm   # generate+score one random cryptarithm trace, same style
uv run python -m reasoners.monitor_gravity       # generate+score one random gravity trace, same style
uv run python -m reasoners.monitor_numeral       # generate+score one random numeral trace, same style
uv run python -m reasoners.monitor_unit_conversion # generate+score one random unit_conversion trace, same style
uv run python -m reasoners.monitor_equation_numeric # generate+score one random equation_numeric trace, same style
uv run python -m reasoners.monitor_bit_manipulation # generate+score one random bit_manipulation trace, same style (WIP -- see "Current gaps")
uv run python reasoners/run_cipher.py            # minimal example: build a Problem, call reasoning_cipher, print the trace
uv run python scripts/eval_train_csv.py          # run every reasoner against all 9500 real train.csv rows, write reasoners_train_csv_eval.csv + print per-category accuracy
uv run python scripts/gen_synthetic_data.py      # bulk synthetic SFT-trace generator across all 7 categories, writes reasoners_synthetic_sft.jsonl (see "Bulk synthetic SFT data generation" below)
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

- **Plain narrative generators**: none remain -- every `reasoners/<task>.py`
  has now been converted to the tagged format below. If you're looking at
  old context/history that calls `bit_manipulation.py` (or `gravity.py`,
  `numeral.py`, `unit_conversion.py`, `equation_numeric.py`) a "plain
  narrative generator with no reward function," that's stale.
- **Tagged generators** (`cipher.py`, `cryptarithm.py`, `gravity.py`,
  `numeral.py`, `unit_conversion.py`, `equation_numeric.py`,
  `bit_manipulation.py`): emit `<step type="...">...</step>`
  traces using six semantic tags -- `plan`, `analysis`,
  `verification`, `execution`, `state_update`, `conclusion` -- shared
  across all seven so a similarly-shaped reward function can score any of
  them (`bit_manipulation.py` uses only five of the six -- no
  `verification` tag, see below).

  **`cipher.py` is the one generator that goes outside this vocabulary**
  (measured over all 1500 cipher rows in `synth_sft.jsonl`): it also emits
  `deduction` and `input_parsing` tags. `reward_cipher.py` has no handler
  for either, so they fall through to the "No specific reward triggered"
  branch -- 0 delta, *not* a penalty -- meaning it's a cross-stage
  inconsistency (SFT teaches a tag shape GRPO ignores) rather than a
  scoring bug. Related headroom concern, same measurement: gold cipher
  traces run **184 steps at p50 and 439 at max against
  `reward_cipher.py`'s `_MAX_STEPS = 500`** -- 88% of the budget consumed
  by the gold shape itself, so a GRPO policy that pads even slightly will
  start hitting the step budget and having real steps zeroed out. Every
  other category is far below: bit_manipulation 7, equation_numeric 13,
  numeral 21, unit_conversion 22, gravity 32, cryptarithm 60. Either drop
  the two extra tags from `cipher.py` (and shorten its per-letter step
  granularity) or raise `_MAX_STEPS` before relying on cipher GRPO
  rollouts.

  Each has a matching `reward_<task>.py`
  (`reward_cipher.py`, `reward_cryptarithm.py`, `reward_gravity.py`,
  `reward_numeral.py`, `reward_unit_conversion.py`,
  `reward_equation_numeric.py`, `reward_bit_manipulation.py`) and a
  `monitor_<task>.py`. Keep any new tagged generator to this same
  vocabulary rather than inventing new tag types, so the reward-scoring
  approach stays transferable.

  **All seven generators now wrap their returned trace in a single
  `<think>...</think>` block** via `store_types.wrap_trace_with_think`
  (added this session): every `reasoning_<task>` function's final `return`
  calls this helper instead of returning its assembled `<step>`-tagged
  string directly. The wrapper leaves the original trace (all `<step>`
  tags, including the internal `<step type="conclusion">...\boxed{}...
  </step>`) completely untouched *inside* `<think>`, and additionally
  appends the same final answer as a bare `\boxed{answer}` line *outside*
  the tags -- so the emitted shape is always
  `<think>\n<original trace>\n</think>\n\boxed{answer}`. This needed **no
  changes to any `reward_<task>.py`**: every reward function locates step
  tags via `re.findall(r'<step type="(.*?)">(.*?)</step>', ...)`, which
  matches regardless of what non-step text wraps around them. The trailing
  answer is extracted structurally (last `\boxed{` marker to the trace's
  own final closing brace, exactly `scripts/eval_train_csv.py`'s
  `last_boxed()` technique) rather than via a brace-balancing regex, so it
  handles cryptarithm's brace-containing symbol answers correctly (verified
  directly: a symbol answer of `"]d}"` round-trips through the wrapper to a
  correct trailing `\boxed{]d}}` line).

  **`bit_manipulation.py` is the one exception to the foolproof contract
  above, and its tag conversion is still in-progress (uncommitted in git as
  of this writing).** Unlike every other tagged generator,
  `reasoning_bit_manipulation` does not gate its emitted trace on
  `answer == problem.answer` before returning -- it was wrapped in
  `<step>` tags without adding that check, so it can still confidently
  emit a wrong boxed answer (see the solve-rate table and "Current gaps"
  below). `reward_bit_manipulation.py` and `monitor_bit_manipulation.py`
  both exist and work (same `evaluate_structured_trace(response_xml,
  examples, expected_answer)` 3-arg shape as `reward_numeral.py`/
  `reward_gravity.py` -- no oracle param, for the same ambiguity reason as
  cryptarithm/equation_numeric, see "Why the reward functions are shaped
  differently"), and `monitor_bit_manipulation.py` routes around the
  missing foolproof check with its own retry loop (construct, run the
  generator, discard/retry until the emitted boxed answer matches the
  known synthetic answer) -- the same pattern cryptarithm/gravity use to
  absorb their own generators' ambiguity misses, just needed here for a
  different reason (a missing check, not inherent ambiguity). Fixing
  `reasoning_bit_manipulation` itself to add the missing gate is still
  open work.

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

`monitor_cipher.py`, `monitor_cryptarithm.py`, and `monitor_gravity.py`
instead **construct synthetic problems top-down** from a fully known ground
truth, then run the deterministic generator to produce a demonstration
trace:
- `monitor_cipher.py` / `kaggle/train_grpo_cipher_kaggle.py`: picks a full
  random 26-letter substitution bijection first (`make_random_cipher`),
  then encrypts real dictionary words with it. The oracle handed to
  `reward_cipher.py` is always the complete alphabet mapping.
- `monitor_cryptarithm.py`: picks a full 10-digit symbol bijection plus a
  random operator->operation assignment first, then generates equations from
  it (`make_random_cryptarithm`). The CSP solver in `cryptarithm.py` is only
  used afterward, to produce a demo trace -- it is not the source of truth.
- `monitor_gravity.py`: picks a gravitational constant `k` first, then
  derives every example distance and the question's answer as
  `round(k * t^2, 2)` -- 2dp, matching real `train.csv` precision (see the
  rounding-fix note below). `reasoning_gravity` is only used afterward to
  produce a demo trace.
- `monitor_unit_conversion.py`: picks a linear conversion `factor` first
  (uniform in 0.5-2.0, matching real `train.csv`'s measured factor range),
  then derives every example output and the question's answer as
  `round(factor * input, 2)` -- 2dp, same precision discipline as gravity.
  `reasoning_unit_conversion` is only used afterward to produce a demo
  trace; same retry loop, since the median-of-truncated-divisions estimate
  doesn't always land exactly on the 2dp-rounded target.

This matters for anyone building a GRPO harness for a new task: don't try to
reverse-engineer an oracle from a solved dataset row; construct the problem
from a known ground truth instead, the same way these scripts do.

**Measured solve rates of the deterministic generators against real
`train.csv` rows** (each generator's foolproof contract means it returns
`None`, rather than a wrong trace, whenever it can't verify -- so these
numbers are "how much of `train.csv` could source *gold SFT traces* via
this generator today," not an accuracy metric). Originally measured on
300-row samples per category; `scripts/eval_train_csv.py` (see "Commands")
later re-ran all six against the **full 9500-row dataset** and every number
below held within a point or two of the sampled figures:

| category | solve rate on real rows | notes |
|---|---|---|
| cipher | ~100% (300/300 sampled; 1576/1576 full dataset) | dictionary-fallback logic handles letters absent from the visible examples. No foolproof check against `problem.answer` (see "Two kinds of reasoner module"), but 0 wrong answers observed on the full dataset |
| cryptarithm | ~8.5% (823-row full dataset: 70/823 = 8.5%, 0 wrong) | documented above; example equations rarely pin a unique symbol->digit assignment |
| gravity | ~62% (300/300 sampled, post rounding-fix; was 0/300 before -- see below); 964/1597 = 60.4% on the full dataset, 0 wrong | remaining misses are genuine estimation noise from the median-of-truncated-divisions approach on 2dp-rounded inputs, not a bug |
| unit_conversion | ~81% (243/300 sampled); 1322/1594 = 82.9% on the full dataset | median-of-truncated-divisions estimation noise, same as gravity -- but unlike gravity, the per-example division needs 6dp precision (not the 3dp default) to avoid a *systematic* bias; see the note below the table. **Zero wrong-answer traces** among the solved rows -- the foolproof contract holds |
| equation_numeric | ~77% (230/300 sampled); 559/732 = 76.4% on the full dataset | post foolproof-fix + tag conversion; the rest return `None` (a few examples don't always pin a unique operation -- same ambiguity risk as cryptarithm), **zero wrong-answer traces** among the solved rows. Was previously N/A: pre-fix the generator confidently emitted 69/300 wrong answers (see "Current gaps") |
| numeral | 1576/1576 = 100% on the full dataset, 0 wrong | not previously sampled at 300; deterministically recomputed from a fixed universal Roman-numeral table (see "Why the reward functions are shaped differently"), so it isn't ambiguity- or estimation-limited the way cryptarithm/gravity are |
| bit_manipulation | 1602/1602 "solved" (a trace is emitted almost always) but only **1364/1602 = 85.1% correct** (238 confidently wrong boxed answers) | **the one category with no foolproof contract** -- `reasoning_bit_manipulation` never checks its derived rule against `problem.answer` before returning, so unlike every other category, "solved" and "correct" are different numbers here. This is the only generator that emits genuinely wrong training traces today. It does now have a tagged trace format, `reward_bit_manipulation.py`, and `monitor_bit_manipulation.py` (uncommitted, see "Two kinds of reasoner module"); see "Current gaps" |

For the five foolproof generators (cryptarithm, gravity, numeral,
unit_conversion, equation_numeric), `scripts/eval_train_csv.py` confirmed
`solved == correct` exactly on the full dataset -- i.e. zero wrong answers
among emitted traces, as the foolproof contract promises. `cipher` has no
such internal check but also measured 0 wrong on the full dataset.
`bit_manipulation` is the only exception, and its 238 wrong-answer rows are
all cases where the generator emitted a full trace (not a decline).

**Gotcha for anyone extracting `\boxed{...}` content programmatically:**
a naive brace-balancing regex like `\boxed\{([^{}]*)\}` will mis-parse
cryptarithm traces, because cryptarithm's symbol alphabet is drawn from a
pool that can itself include literal `{`/`}` characters (see
`monitor_cryptarithm.py`'s `_SYMBOL_POOL`) -- the answer content can contain
unbalanced braces that a bracket-matching regex can't delimit correctly.
`scripts/eval_train_csv.py`'s `last_boxed()` instead locates the answer
structurally: every generator's conclusion is the literal end of the
returned string (optionally followed by `</step>`), so the boxed content is
whatever lies between the last `\boxed{` substring and the trace's final
closing brace, no brace-matching required.

**Why unit_conversion needs 6dp division precision and gravity doesn't:**
`long_division_lines` (in `store_types.py`) truncates toward zero at
`max_decimal_digits` places (3 by default). For gravity's `k` (magnitude
~1-20) that's plenty of significant figures and the truncation error is
negligible. For unit_conversion's `factor` (magnitude ~0.5-2.0), 3dp is
only 3-4 significant figures, and truncation-toward-zero is a *systematic*
downward bias, not noise that cancels out: every per-example factor
undershoots, so the median undershoots, so the final answer undershoots.
Measured directly: at the 3dp default, 224 of 225 real-row misses were
undershoots (only 1 overshoot) -- a strong bias signature, not estimation
noise. Solve rate at 3dp was ~25% (75/300); raising
`long_division_lines`'s `max_decimal_digits` to 6 for unit_conversion's
per-example division (both in `reasoning_unit_conversion` and
`reward_unit_conversion.py`'s `_factor_for_example` -- **these two calls
must stay bit-identical**, same discipline as everywhere else in this
codebase) removed the bias and brought the real-row solve rate to ~81%
(243/300), matching gravity's ballpark. The remaining ~19% misses at 6dp
are genuine median-of-truncated-divisions estimation noise (rounding in
the 2dp example values themselves), the same kind gravity's docstring
describes -- not a further precision issue: precision beyond 6dp measured
no further improvement (81% flat through 10dp).

Given this spread, **the right SFT data source is a per-category decision,
not a blanket policy**: cipher can lean on real rows directly; cryptarithm
and gravity need synthetic-primary generation (real rows are too sparse or,
for gravity pre-fix, entirely unusable) but should still keep a slice of
real solvable rows around as a distribution-shift check (vocabulary,
sentence length, number precision) against purely synthetic data.

### Why the reward functions are shaped differently

`reward_cipher.py`, `reward_cryptarithm.py`, and `reward_gravity.py` all
parse the six-tag vocabulary sequentially, track the trace's own claimed
state (flow awareness -- using a fact before declaring it, or contradicting
an earlier declaration, is penalized even when the fact is correct), grade
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
this codebase, don't collapse it if you touch any of these files**:

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
- `reward_gravity.py` similarly does **not** take a true-`k` oracle
  parameter at all, and for an analogous but distinct reason: gravity's `k`
  is *recovered by estimation* (median of several per-example `d/t^2`
  divisions), not looked up, so even the "true" generating `k` wouldn't be
  exactly recoverable from rounded example distances -- grading against it
  would inject the same kind of luck-driven noise cryptarithm's docstring
  warns about, just from rounding error instead of structural ambiguity.
  Per-example `k` values *are* graded directly (`_k_for_example`), because
  those, unlike the fitted `k_fit`, are exactly and deterministically
  recomputable from that example's own visible `(t, d)` pair. `k_fit` is
  graded only on (a) self-consistency with the trace's own declared
  per-example `k`s, and (b) whether it honestly reproduces the given `d`
  values when replayed -- recomputed independently by the reward function,
  never trusting the trace's self-reported reproduction.
- `reward_numeral.py` is the odd one out: it takes **no oracle parameter at
  all**, for a reason distinct from both cryptarithm's and gravity's.
  Numeral's correct answer isn't structurally ambiguous (cryptarithm) or
  estimated with rounding noise (gravity) -- it's **deterministically
  recomputable** from the question integer alone, against a *fixed
  universal* table (`ROMAN_VALUES`, unchanged across every problem, unlike
  cipher's per-problem 26-letter secret). So every claim in a numeral trace
  -- `state_update` atoms, `verification` reconstructions, `execution`
  arithmetic, the final boxed answer -- is graded directly against that
  fixed table or recomputed from the problem's own visible data; there's no
  per-problem oracle to pass in, so the signature is the same 3-arg shape as
  `reward_gravity.py`'s (`evaluate_structured_trace(response_xml, examples,
  expected_answer)`). The catch: numeral's examples are near-decorative -- a
  model that already knows Roman numerals doesn't need them to answer
  correctly -- so the real, ungameable signal is the conclusion's boxed
  answer, verification-label honesty, and execution arithmetic
  self-consistency, not "did it correctly use the examples." Don't
  over-engineer a "deduce the system from examples" grader for this task;
  the ceilings on cheap-to-farm categories are what bound padding, the same
  as everywhere else.
- `reward_unit_conversion.py` is graded with the **same discipline as
  `reward_gravity.py`** (unit_conversion is structurally gravity with the
  `t^2` squaring step removed -- see the generator's docstring): the fitted
  `factor_fit` takes no oracle parameter and is graded only on
  self-consistency with the trace's own declared per-example factors plus
  an honest reproduce-the-givens verification pass; per-example factors
  *are* graded exactly, since they're deterministically recomputable from
  each example's own visible `(input, output)` pair. The conclusion's boxed
  answer is graded with a `1e-2` numeric tolerance rather than exact string
  equality, because the factor is still not exactly recoverable even at the
  corrected 6dp division precision (see the solve-rate table above and the
  truncation-bias note beneath it) -- exact-match grading would inject
  rounding-luck noise into the GRPO advantage estimate for correct-method
  completions that land close but not byte-identical. Measured directly: of
  the 57/300 real rows the generator couldn't exactly verify at 6dp, 56
  (98%) land within the reward's `1e-2` tolerance anyway (only one outlier
  at 0.02) -- so the tolerance is well-calibrated to the actual estimation
  noise, not an arbitrary buffer. **`reward_gravity.py` uses the identical
  `1e-2` tolerance on its own conclusion check for the same reason** (k_fit
  is likewise a median-of-truncated-divisions estimate, not an exact
  recovery) -- this was originally an exact-string check and was widened to
  match unit_conversion's discipline once it became clear any reward
  function grading an estimated (not structurally-ambiguous, not
  fixed-table) numeric answer needs the same tolerance, not just this one.
- `reward_equation_numeric.py` takes **no oracle parameter** (same 3-arg
  shape as `reward_numeral.py`/`reward_gravity.py`), but its *motivation* is
  cryptarithm's, not numeral's: with only a handful of examples a
  coincidentally-fitting wrong operation often can't be ruled out, so the
  declared rule (`op_name`, `rev_ops`, `rev_res`, `fmt` for one operator) is
  graded **only** on validity (recognized operation name + format tag) and
  self-consistency -- *never* against the generating rule, exactly the
  ambiguity-invariance argument cryptarithm's docstring makes. Real
  correctness credit flows from `verification` steps that replay the trace's
  own declared rule against each real given example and check it
  independently reproduces the real output, plus the conclusion's boxed
  answer. The load-bearing subtlety (equation_numeric's version of the
  copy-visible-text exploit): the real example outputs are visible in the
  prompt, so the reward **never trusts a trace's self-reported
  "reconstructed"/"given" strings** -- it recomputes via the shared
  `apply_rule` (imported bit-identically from `equation_numeric.py`, same
  must-stay-identical discipline as unit_conversion's `_factor_for_example`)
  and compares to the real given output. Boxed answers are graded by exact
  string match (unlike gravity's/unit_conversion's `1e-2` tolerance) because
  equation_numeric answers are plain strings (`"17/"`, `"6644"`), not
  rounded decimals -- there's no estimation error to tolerate, and a numeric
  tolerance would silently break on the non-numeric prefix/suffix formats
  (`"17/"` isn't parseable as a float) while adding nothing for the purely
  integer results. The same reasoning keeps `reward_cryptarithm.py`'s boxed
  answer (a symbol string over the puzzle's alphabet, not a decimal number)
  and `reward_numeral.py`'s (a Roman numeral string) and
  `reward_bit_manipulation.py`'s (a bit string) on exact string match too --
  none of them are estimated numeric quantities, so a numeric tolerance has
  nothing to buy and would just accept wrong answers or crash on non-numeric
  content. Validated the standard two ways: 600 synthetic gold
  traces (incl. 186 exercising the sign-carrying prefix/suffix format path)
  all non-negative, plus 230/230 solvable real `train.csv` rows (227
  multi-operator, 4 fallback-path) all non-negative; adversarial checks
  (fabricated example, false verification, tag-spam, self-contradiction,
  copy-visible-output with a wrong final answer) all score negative.

**Practical consequence for the GRPO plan** (see "Project goal" above):
`reward_cryptarithm.py`, `reward_gravity.py`, `reward_numeral.py`,
`reward_unit_conversion.py`, `reward_equation_numeric.py`, and
`reward_bit_manipulation.py` need only the prompt's own visible examples
plus the hidden final answer to score a completion -- both of which exist
for *any* prompt, real `train.csv` row or synthetic. `reward_cipher.py`
genuinely needs the
complete 26-letter oracle to grade `state_update`/`execution` claims, and
that oracle only exists if you
generated the cipher yourself -- it's never recoverable from a real
`train.csv` row (a handful of example sentences won't cover all 26
letters). So `reward_cipher.py` is reliable for GRPO iff the harness always
constructs its own cipher before scoring (as `kaggle/train_grpo_cipher_kaggle.py`
already does); pointing it at a pre-existing real prompt with only a
partial reconstructed oracle would falsely penalize letters outside the
visible examples. This isn't a bug to fix so much as a property to respect
when deciding what feeds the GRPO loop for each task.

A consequence of this: `reward_cryptarithm.py`'s per-equation verification
handler deliberately does **not** trust a trace's self-reported
"reconstructed" string against its self-reported "target" string, since both
would be attacker-controlled and the real example outputs are visible in
the prompt -- copying them verbatim would trivially pass a naive check. It
instead recomputes the equation's result from the trace's own
previously-declared `state_update` digits/operator and requires that to
independently reproduce the real given output before awarding credit.
`reward_gravity.py`'s verification handler applies the identical discipline
for the same reason (its own docstring calls this out explicitly). Any new
verification-style check added to any of these reward functions should
apply the same test: can this check be satisfied by copying visible prompt
text, without the model having actually derived anything? If so, ground it
in something the model doesn't already have for free (the trace's own
prior declarations, arithmetic that must independently check out, or a
value that's genuinely hidden from the prompt).

### The gravity 2dp-rounding fix (why gravity's real-row solve rate went 0% -> 62%)

`reasoning_gravity` originally computed its final answer by *truncating* a
long-multiplication result to 3 decimal places and comparing that string
exactly against `problem.answer`. This double-failed against real
`train.csv` rows: their `answer` column is 2dp, not 3dp (a pure format
mismatch that alone would always fail the equality check), and the
median-of-truncated-divisions `k` estimate carries genuine rounding noise
from 2dp-rounded example distances (measured directly: one real row
computed `154.631` against a true answer of `154.62`). Real `train.csv`
example distances also frequently drop trailing zeros inconsistently (e.g.
`"12.4"` vs `"12.40"` for the same value), which was silently breaking
several raw string-equality checks in `reward_gravity.py` that compared a
trace's claim against `example_map` values.

The fix, all routed through a new `round_2dp()` helper in `store_types.py`
(`Decimal(...).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)`):
- `reasoning_gravity`'s final answer and per-example verification pass now
  round to 2dp instead of truncating to 3dp, and the emitted `\boxed{}`
  echoes `problem.answer`'s *literal* string (not the internally-rounded
  value) so it stays byte-identical to whatever format the eval/grading
  step expects.
- `monitor_gravity.py`'s synthetic example distances are now generated at
  `round(k * t^2, 2)` -- 2dp, matching real data's precision -- rather than
  3dp precision chosen specifically to dodge this rounding noise. The
  existing retry loop (already used to route around cryptarithm's much
  lower solve rate) absorbs the cases that still don't verify.
- `reward_gravity.py`'s independent recomputations (`_reproduce_d`,
  `_final_d`, the `m_d` execution-tag check) were updated to the same 2dp
  convention, and every place that compared a trace's claimed `d` against
  `example_map` was switched to compare via `round_2dp()` rather than raw
  string equality, plus a `_safe_2dp()` wrapper for values regex-captured
  out of (untrusted) trace content so malformed claims degrade to "doesn't
  match" instead of raising.

Result: real-row solve rate went from 0/300 to 186/300 (62%); 300 fresh
synthetic gold traces still all score non-negative; adversarial checks
(plan-spam, fabricated example, missing conclusion, wrong final answer)
still behave as designed post-fix.

### unit_conversion's generator fixes (three, not two -- don't assume it's a pure gravity clone)

`unit_conversion.py` was originally a plain-narrative generator (no
`<step>` tags, no reward function) with the same two latent bugs gravity
had pre-fix: it never re-verified its computed answer against
`problem.answer` before returning (foolproof-contract violation -- `return
None` only fired when there were no usable examples), and it truncated the
final product to 3dp by string-slicing the decimal point instead of
rounding to 2dp (which also raised `ValueError` outright whenever the
product happened to be integer-valued, since there was no decimal point to
slice at). Converting it to the tagged-generator trio (this session) fixed
both, routed through the same `round_2dp()` helper gravity uses: the
foolproof check (`boxed_answer != round_2dp(problem.answer): return None`)
now gates every emitted trace, and the final answer is rounded to 2dp
instead of truncated to 3dp.

That alone left real-row solve rate at only ~25% (75/300) -- a **third,
unit_conversion-specific bug**, distinct from anything gravity had: at
`long_division_lines`'s 3dp truncation default, unit_conversion's smaller
factor magnitude (0.5-2.0 vs gravity's 1-20) turns truncation-toward-zero
into a systematic downward bias rather than noise (224/225 real-row misses
were undershoots -- see the solve-rate table's note above for the full
measurement). Raising the per-example division's `max_decimal_digits` to 6
(in both `reasoning_unit_conversion` and `reward_unit_conversion.py`'s
`_factor_for_example`, which must stay bit-identical) fixed it: solve rate
rose to ~81% (243/300), with **zero wrong-answer traces** among the solved
rows throughout, confirming the foolproof contract held even while the
division-precision bug was still present. `reward_unit_conversion.py` was
validated the same way described below (300 fresh synthetic gold traces, all
non-negative; adversarial checks for fabrication, tag-spam,
self-contradiction, dishonest verification labels, and the boxed-answer
tolerance all behave as designed).

### Validating reward functions

There's no test suite, so the pattern used to validate reward functions
(see git history / prior session work on `reward_cipher.py`,
`reward_cryptarithm.py`, `reward_gravity.py`, and
`reward_unit_conversion.py`) is: generate on the order of a few hundred
synthetic gold traces via the relevant `monitor_*` construction helpers,
score each, and assert none score negative (a gold trace, by the
foolproof contract, is always fully correct, so a negative score means the
reward function itself has a bug). Separately, hand-write adversarial
traces to confirm known exploit shapes score negative/flat: fabricated
claims, repetition-spam of any single tag, self-contradiction, copying
visible prompt text into a self-referential check. Any new reward function
should be validated the same two ways before being trusted.
`reward_gravity.py`, `reward_unit_conversion.py`, and
`reward_equation_numeric.py` have all been through this validation
(`reward_equation_numeric.py` as of this session -- 600 synthetic gold +
230/230 solvable real rows non-negative, adversarials negative; see its
bullet under "Why the reward functions are shaped differently").

## Current gaps

- **`equation_numeric.py`'s foolproof-contract bug is fixed (this
  session).** It previously never re-verified its derived answer against
  `problem.answer`, emitting 69/300 (23%) confidently wrong boxed answers.
  It now recomputes the applied rule's answer via the shared `apply_rule`
  and gates on `answer == problem.answer` before returning (a plain `==`,
  no rounding caveats -- equation_numeric answers are plain strings). It was
  also converted from a plain-narrative generator to the six-tag vocabulary
  this session, and now has `reward_equation_numeric.py` +
  `monitor_equation_numeric.py`. Measured: 230/300 real rows solved, 0
  wrong.
- All seven categories now have reward functions and `monitor_*` scripts
  (see "Two kinds of reasoner module"). `cipher` has a local/Kaggle GRPO
  script (`kaggle/train_grpo_cipher_kaggle.py`, TRL `GRPOTrainer`); the
  other six now have a hosted-Tinker GRPO script
  (`kaggle/train_grpo_tinker.py`, see "The GRPO training script" below)
  that continues training from the `full_0727` SFT checkpoint, defaulting
  to `numeral` + `equation_numeric` per ROADMAP Step 7. Neither script has
  had a real paid run yet -- `train_grpo_tinker.py --dry-run` is verified
  (see its section), but the actual `forward_backward`/`optim_step` loop
  has not been exercised against the live service.
- `bit_manipulation` now has a tagged generator, `reward_bit_manipulation.py`,
  and `monitor_bit_manipulation.py` (**all uncommitted in git as of this
  writing** -- `reasoners/bit_manipulation.py` is modified but not
  committed, and the other two are untracked new files). **It's still the
  only generator that violates the foolproof contract**:
  `scripts/eval_train_csv.py` measured 1364/1602 (85.1%) correct on the full
  `train.csv` dataset, with 238 rows where `reasoning_bit_manipulation`
  emitted a fully-formed trace and boxed answer that was simply wrong (it
  never checks its derived rule against `problem.answer`). The tag-wrapping
  work wrapped the existing narrative in `<step>` tags without adding that
  check; `monitor_bit_manipulation.py` routes around it with its own
  retry loop (discard/retry until the emitted trace happens to match a
  known synthetic answer), but `reasoning_bit_manipulation` itself still
  needs the `answer == problem.answer` gate added before it can be trusted
  as an SFT source the way `equation_numeric.py`'s fix was.
- **Two SFT training scripts now exist** (stage 2 of the three-stage plan
  under "Project goal"): `kaggle/train_sft_kaggle.py` (local/Kaggle GPU) and
  `kaggle/train_sft_tinker.py` (Thinking Machines' hosted Tinker API), both
  uncommitted in git as of this writing. See "The SFT training script" and
  "The Tinker SFT training script" below. The local script's remaining
  stage-2 gap is not the script but the synthetic-vs-real equation_numeric
  operator-count mismatch found while auditing (fixed in the generator, see
  the audit section). The Tinker script has since completed a full default
  run against the live service (`sft_tinker_runs/full_0727/`) confirming the
  step-tag format is actually learnable, not just internally consistent;
  its one open gap is no resume-from-checkpoint path (see its section).
- `reasoners/dictionary.txt` is present but not currently imported by any
  solver -- `wonderland.txt` (77 words) is the word list actually used by
  `cipher.py`/`reward_cipher.py`.
- `cipher.py` emits two tags outside the shared six-tag vocabulary and its
  gold traces sit at 88% of `reward_cipher.py`'s step budget -- see "Two
  kinds of reasoner module" for the measurement and both fix options.

## Bulk synthetic SFT data generation

`scripts/gen_synthetic_data.py` (see "Commands") is a single driver that
generates stage-1 SFT data (see "Project goal") across all 7 categories in
one run, rather than eyeballing one trace at a time via the `monitor_*`
scripts. For each category it imports that category's `monitor_<task>.py`
problem constructor directly (never reimplements it, to stay bit-identical
-- cipher is the one exception, since `monitor_cipher.py` builds its
problem inline in `main()` with no reusable function, so this script
reassembles it from the same pieces with a threaded `random.Random` for
reproducibility), runs the matching `reasoning_<task>` generator, and keeps
only verified traces:
- The 5 foolproof generators (cryptarithm, gravity, numeral,
  unit_conversion, equation_numeric) are trusted as-is whenever they return
  non-`None` -- their own `answer == problem.answer` gate already
  guarantees correctness, and re-parsing their boxed answer here would hit
  the same brace-balancing hazard the "Gotcha" note above describes for
  cryptarithm.
- cipher and bit_manipulation (the two generators without a foolproof gate)
  are additionally checked here: the trace is kept only if its last
  `\boxed{...}` equals `problem.answer`. Both answer types are brace-free
  (plaintext words / pure binary), so a naive boxed regex is safe for these
  two specifically -- it would not be for cryptarithm.

**8192-token cap is now enforced during generation, not as a post-hoc
filter (this session).** Every candidate row's `prompt + "\n" + trace` is
tokenized with the `Qwen/Qwen2.5-0.5B-Instruct` tokenizer (cached locally;
same vocab as the rest of the Qwen2.5 family, so the 0.5B checkpoint is
fine for tokenizer-only use); rows over `_MAX_TOKENS = 8192` are **dropped
and retried, never truncated** -- truncation would cut off the trailing
`\boxed{}` line stage-2 SFT exists to teach, which is exactly why the prior
"SFT readiness audit" (below) recommended filtering over truncating in the
first place. The per-category summary table now has an `>8192tok` column
counting how many candidates each category dropped for length.

Output is written to whatever `--out` names (default
`reasoners_synthetic_sft.jsonl`, override with `--per-category`), rows of
`{id, category, prompt, question, answer, trace}`. Prompts are written to
match real `train.csv` phrasing per category. Note this is a *different*
prompt shape than `kaggle/train_grpo_cipher_kaggle.py`'s (system one-shot
demo + `Examples:/Question:` body) -- SFT data from this script and GRPO
rollouts from the kaggle script currently see different prompt formats for
the same task; align them before chaining SFT -> GRPO if that cross-stage
consistency matters.

**`synth_sft.jsonl` was regenerated this session (14000 rows, 2000/category,
~93 MB) and is the current canonical corpus -- the audit numbers in this
section are now stale against it, see the callout below the table.** The
prior 10500-row/1500-per-category run (the one the "SFT readiness audit"
below was actually measured on) is preserved as `synth_sft.legacy.jsonl`
rather than deleted. Both files are untracked and not gitignored -- this
still applies to the new file too.

Regeneration picked up every fix from this session: equation_numeric's
operator-count mismatch (the new corpus measures 1:2:3 distinct operators
per prompt at 8.4%/57.8%/33.9%, close to real `train.csv`'s 6%/52%/42%; see
"Distribution mismatch" above), the `<think>`-wrapping on every generator's
trace (see "Two kinds of reasoner module"), and an 8192-token drop-during-
generation filter (verified independently post-generation: 0/14000 rows
exceed 8192 Qwen2.5 tokens, max observed 8188). Per-category yields (rows /
attempts) on this run: bit_manipulation 5.6%, cipher 95.9%, cryptarithm
55.5%, equation_numeric 55.3%, gravity 92.1%, numeral 100%, unit_conversion
80.6% -- bit_manipulation's low yield is expected (see "Two kinds of
reasoner module"'s note on its missing foolproof gate) and cost ~35,500
attempts for its 2000 rows, still well under the collector's attempt cap.
Corpus integrity re-checked the same way as the original audit below: zero
unbalanced `<step>`/`</step>` tags, zero rows missing a `conclusion` tag,
14000/14000 unique `(prompt, trace)` pairs, exact 2000/category balance.

**Not re-run on the new corpus**: the token-length percentile table, the
train/val rule-leakage count, and the other narrower measurements in the
audit below all still describe `synth_sft.legacy.jsonl` specifically (the
qualitative filter-don't-truncate conclusion still holds, and is now
enforced automatically rather than needing a manual pass -- see "Bulk
synthetic SFT data generation" above -- but the exact percentile numbers
below have not been recomputed for the regenerated file).

## SFT readiness audit (measured on `synth_sft.legacy.jsonl`, 10500 rows -- superseded, see note above)

This audit predates `kaggle/train_sft_kaggle.py` (see "The SFT training
script" below, which enforces several of its recommendations in code).
What was checked:

**Corpus integrity -- clean.** All 10500 rows: zero boxed-answer
mismatches (using the structural `last_boxed` parse, not a brace-matching
regex -- see the "Gotcha" above), zero unbalanced `<step>`/`</step>`
counts, zero traces missing a `conclusion` tag, and 10500/10500 unique
`(prompt, trace)` pairs. Category balance is exact at 1500 each.

**Sequence length is bimodal by category -- filter, don't truncate.**
Qwen2.5 token counts over `prompt + trace`, full corpus:

| category | p50 | p95 | max |
|---|---|---|---|
| equation_numeric | 446 | 473 | 473 |
| numeral | 535 | 648 | 759 |
| cryptarithm | 1571 | 1641 | 1678 |
| cipher | 4673 | 7927 | 11467 |
| unit_conversion | 5258 | 7384 | 8373 |
| gravity | 5995 | 7509 | 8515 |
| bit_manipulation | 6873 | 7188 | 8382 |

Rows exceeding a given cap: 2048 -> 5996 (57.1%), 4096 -> 5183 (49.4%),
6144 -> 2906 (27.7%), **8192 -> 70 (0.7%)**, 12288 -> 0. Because the
distribution is bimodal, a mid-range cap is not a uniform trim: at 2048 it
removes *all* of bit_manipulation, cipher, gravity and effectively all of
unit_conversion while leaving the other three untouched. And truncation
removes the trace's **tail**, which is exactly the
`<step type="conclusion">...\boxed{}` that stage-2 SFT exists to teach --
a right-cut silently trains the model on traces that never commit to an
answer. So drop over-length rows rather than truncating them, and prefer
8192 (costs 0.7% of the corpus: 59 cipher, 6 gravity, 4 unit_conversion,
1 bit_manipulation) unless VRAM forces lower -- at 4096 you are training
three categories and abandoning four.

**Distribution mismatch: synthetic equation_numeric was single-operator,
real rows are not (generator fixed this session; `synth_sft.jsonl` is
stale for this category).** All 1500 equation_numeric rows in the
on-disk `synth_sft.jsonl` corpus (see "Two corpora exist on disk" above)
contain exactly one operator (one `state_update`, one declared rule),
because `monitor_equation_numeric.py`'s constructor only ever generated a
single operator symbol. Real `train.csv` equation_numeric prompts are 94%
multi-operator: 44 single-operator, 380 two-operator, 308 three-operator
(measured across all 732 real rows) -- consistent with this file's note
elsewhere that 227 of the 230 solvable real rows are multi-operator. SFT
on the old corpus therefore taught a strictly easier task shape than the
real eval set. **That file still reflects the pre-fix generator** -- the
1500 rows on disk were not regenerated as part of this fix, so anyone
using `synth_sft.jsonl` for equation_numeric SFT should regenerate that
slice via `scripts/gen_synthetic_data.py` first, or the mismatch persists
regardless of what this section says.

`monitor_equation_numeric.py`'s `make_synthetic_problem` now picks 1-3
distinct operators per problem with weights matching the real 44/380/308
split, gives each its own independently-drawn rule, and distributes
3-5 examples across them unevenly (mirroring real data, where per-operator
example counts are mostly 1-2). The question's operator is drawn from that
same set ~81% of the time (also measured on real rows); the other ~19% it's
a novel operator absent from every example, exercising
`reasoning_equation_numeric`'s fallback path -- these frequently fail to
verify and get discarded by the retry loop, same as the ambiguity misses
cryptarithm/gravity already absorb. Measured post-fix: over 3000 synthetic
attempts, the realized operator-count split among *verified* (non-`None`)
problems was roughly 9%/57%/34% -- close to but skewed somewhat toward
2-operator relative to the 6%/52%/42% target, because 3-operator problems
have more independent chances to fail a `None`/signed-format filter and so
are mildly under-represented among survivors; this residual skew wasn't
worth correcting for and is far smaller than the defect it replaces
(100% single-operator). Overall construct->verified yield dropped to
~55-75% (was higher pre-fix, since a single-rule problem has only one
chance to fail), so regenerating a full 1500-row equation_numeric slice
now costs proportionally more `make_synthetic_problem` attempts. As a
side effect, the fix also fixed the "fixed example count" complaint below
for this category specifically (now 3-5 examples, matching real rows) and
substantially addressed the rule-leakage concern: a 200-row post-fix
sample already showed 160 distinct declared rules (was 40 distinct rules
across the old 1500-row single-operator corpus), since varying operators
per problem naturally diversifies which rule gets declared. This was not
re-measured against the real-row solve rate: `reasoning_equation_numeric.py`
and `reward_equation_numeric.py` were untouched by this fix (the reward
function already keyed declared rules by `op_char`, so it needed no
change), so the 559/732 (76.4%) real-row solve rate in the table above
should still hold, but wasn't re-run to confirm.

Also, synthetic prompts for the *other* categories still use a fixed
example count (4, or 5 for gravity / 9 for bit_manipulation) where real
prompts vary 3-5 (8-11 for bit_manipulation); minor, but it narrows the
format distribution the model sees for those tasks.

**Open decisions before a run** (none are bugs; all are the operator's
call):
- Prompt format is now a knob rather than an unmade decision
  (`Cfg.prompt_style` in `kaggle/train_sft_kaggle.py`: `"chat"` renders the
  tokenizer chat template with a generic step-tag system prompt, `"raw"`
  feeds the bare `train.csv`-style text with no template *and no system
  prompt at all*). The cross-stage inconsistency itself is unresolved:
  `train_grpo_cipher_kaggle.py` still uses chat messages with a
  cipher-specific one-shot demo, so aligning the two is still the
  operator's call before chaining SFT -> GRPO.
- The corpora themselves still carry no train/val split; the SFT script
  builds one at load time (stratified by category, fixed seed) rather than
  materializing it on disk. The rule-leakage caveat above applies to it --
  val loss there measures format learning, not task generalization.
- The corpus is 100% synthetic, against this file's own advice to keep a
  slice of real solvable rows as a distribution-shift check.
  `reasoners_train_csv_eval.csv` makes assembling that slice cheap.
- `bit_manipulation` is a downsample candidate: its traces are bit-column
  dumps ("bitsum as hash", "default 1 = 1") -- a search log rather than
  reasoning -- at ~6900 median tokens, so it consumes a disproportionate
  share of the token budget teaching table-dumping. Its boxed answers are
  all verified correct, so this is not wrong data; but note its rows
  survive via `monitor_bit_manipulation.py`'s retry-until-match loop (see
  "Two kinds of reasoner module"), so the kept set is selection-biased
  toward the rules that generator's heuristic happens to land on, and is
  not a uniform sample of the task.

## The SFT training script

`kaggle/train_sft_kaggle.py` (added this session, **uncommitted in git as of
this writing**) is stage 2: LoRA SFT on `synth_sft.jsonl` to teach the
output *format* (the `<think>`-wrapped six-tag trace ending in a bare
`\boxed{}` line), not to teach the tasks. It's self-contained -- the Kaggle
Dataset needs only `synth_sft.jsonl`, no code from this repo -- and mirrors
`train_grpo_cipher_kaggle.py`'s conventions (`INPUT_DIR` with a local
checkout fallback, a `Cfg` dataclass, `find_latest_checkpoint` auto-resume
for Kaggle's ~9-12h session cap).

Design decisions worth not re-litigating:

- **Prompt masked to `-100`, loss on trace tokens only**, with an EOS
  appended to the completion so the model learns to stop (omit it and GRPO
  rollouts later run to `max_completion_length` and get scored on truncated
  traces). `preview()` prints the masking boundary of one row each run, so
  a labels/ids misalignment is visible before the first step.
- **Over-length rows are dropped, never truncated** -- enforcing in code the
  conclusion the audit above reached, for the same reason (a right-cut
  removes the `conclusion` tag + `\boxed{}` this stage exists to teach).
  `max_seq_len` defaults to **8704**, deliberately above the corpus's own
  8192-token generation cap, so the chat-template wrapper and system prompt
  don't push the longest rows over. Measured on the current 14000-row
  corpus with the Qwen3-0.6B tokenizer: **0 rows dropped**, 13650 train /
  350 val, max observed 8393 tokens (cipher). Set it lower only under VRAM
  pressure, and read the per-category drop table it prints.
- **Plain `transformers.Trainer` over TRL's `SFTTrainer`**, feeding
  pre-tokenized rows. TRL's SFT dataset API (prompt-completion columns,
  `assistant_only_loss`, packing) has churned across the versions Kaggle
  preinstalls; the hand-rolled masking is ~12 lines and pins the behaviour.
  For the same churn reason, `build_training_arguments()` drops
  `TrainingArguments` kwargs the installed version doesn't have (warning,
  not crash) -- `group_by_length` is gone in transformers 5.x, and it's a
  no-op at the default batch size 1 anyway.
- **The loss is chunked, and this is load-bearing, not an optimization.**
  The OOM at these sequence lengths is the loss, not the weights: an 8k-token
  row against Qwen's ~152k vocab is a 2.5 GB bf16 logits tensor that
  `ForCausalLMLoss` upcasts to fp32 and keeps a gradient for. Measured: the
  stock path OOM'd a 5.7 GB GPU on a *single* row, and is tight even on a
  16 GB T4. `_make_chunked_loss_trainer()` therefore calls the transformer
  *body* directly (bypassing the causal-LM forward, which would compute the
  very tensor being avoided -- LoRA layers live inside the body's modules so
  they still apply) and applies the head in `loss_chunk_size`-token slices
  wrapped in `torch.utils.checkpoint`, making peak loss memory proportional
  to the chunk rather than the sequence. Verified numerically equal to the
  model's own `ForCausalLMLoss` on a real row (delta 0.00e+00), and a full
  train step on 7k-token rows then fits in 5.7 GB. It honours
  `num_items_in_batch` when Trainer passes it and falls back to a per-token
  mean otherwise, which is what keeps the gradient-accumulation
  normalization correct across transformers versions.
- **Split is stratified by category with a fixed seed**, val capped at 50
  rows/category (eval at ~8k tokens is slow). Empty-val runs disable eval
  rather than erroring, so `--limit`-style smoke runs work.
- fp16 vs bf16 is chosen from `torch.cuda.is_bf16_supported()`; on the fp16
  path (T4/P100) trainable params are upcast to fp32, or the grad scaler
  raises "Attempting to unscale FP16 gradients".

Two caveats that are the operator's call, not bugs:

- **The defaults imply a run longer than one Kaggle session**: 13650 rows x
  2 epochs / 16 effective batch is ~1700 optimizer steps at ~5-6k median
  tokens. Auto-resume makes that survivable across sessions; for pure format
  learning, `Cfg.category_caps` (~300-500/category) or one epoch is
  plausibly enough.
- **SFT -> GRPO is not wired.** Beyond the prompt-format mismatch noted in
  the audit's open-decisions list, `train_grpo_cipher_kaggle.py` loads the
  stock checkpoint -- chaining the stages means merging the adapter or
  repointing its `model_name` at the SFT output dir.

Not yet validated: only the bf16 path has been exercised end-to-end (the
fp16/T4 branch is written but untested), and no generation smoke test
(sample a val prompt, eyeball the emitted format) exists in the script.

## The Tinker SFT training script

`kaggle/train_sft_tinker.py` (untracked) is a second stage-2 entry point:
the same LoRA-SFT-on-`synth_sft.jsonl` job as `train_sft_kaggle.py`, but
trained on Thinking Machines' hosted Tinker API (`tinker` 0.23.4 +
`tinker-cookbook` 0.5.2, the `tinker` extra in `pyproject.toml`) instead of a
local/Kaggle GPU -- no torch/peft needed for this path, since Tinker runs the
model server-side. It reuses `train_sft_kaggle.py`'s data prep
(`load_rows`/`build_examples`/`stratified_split`) directly rather than
reimplementing it.

It was audited before its first real run and found to mix in call shapes
from `huikang_nemotron/trainer/client.py`, a **wrapper** around the Tinker
SDK that this script was originally drafted against instead of the raw SDK
-- see [[tinker-sdk-vs-wrapper-api]] for the full drift list. All three
defects sat *after* the billed training loop, so an unfixed run would have
spent the money and persisted nothing:

1. **No working checkpoint save.** The wrapper's
   `save_checkpoint_async(name, log_path)` doesn't exist on the real
   `TrainingClient`. Fixed with a `save_checkpoint()` helper that calls the
   SDK's actual `save_state_async` + `save_weights_for_sampler_async`,
   invoked periodically (`Cfg.save_every_steps = 10`) and from a
   try/except around the whole run so a mid-run crash still saves an
   `"interrupted"` checkpoint before re-raising. Paths land in
   `<log_dir>/checkpoints.jsonl`.
2. **`APIFuture` never unwrapped** at two `save_weights_for_sampler_async`
   call sites (a `getattr(result, "path", result)` fallback silently
   returned the future itself), and `sample_async` was missing the
   required `num_samples` argument. Both fixed.
3. **The smoke test's format checks were structurally wrong** --
   `"\boxed{" in completion` is near-vacuous (every gold trace's *plan*
   step contains boilerplate "I will put my final answer inside \boxed{}"
   ~150 chars in, regardless of correctness) and `"<think>" in completion`
   is a guaranteed false negative (`build_examples` strips the chat
   template's leading `<think>\n`, so a perfect completion never contains
   an opening tag). Fixed to check the completion alone: `</think>`
   present, `<think>` absent, and `\boxed{` only counted if it falls in the
   tail *after* `</think>` (guarding `str.rpartition`, which returns the
   whole string when the separator is absent). See
   [[step-tag-format-check-pitfalls]] for the full reasoning.

All three were confirmed against the real installed SDK, not assumed from
the sibling wrapper repo (0.16.1 there vs. 0.23.4 here).

**Verified against the live service, cheapest first.** A `--limit 4 --epochs
1 --yes` rehearsal (~$0.08) exercised every post-loop path end to end: real
`tinker://` checkpoint paths, sampling, `weights.download`, and a full
`build_hf_model` merge (8.8 GB, 738 tensors, no LoRA leftovers). Then a full
default-config run completed (`sft_tinker_runs/full_0727/`, 2026-07-27):
1376 train / 73 val rows (1 dropped for length), 4,043,134 train tokens
(~$2.98), 22 optimizer steps. `_loss_per_token` dropped monotonically
0.334 -> 0.016 across the run, and hard per-category too (e.g.
`equation_numeric` 1.49 -> 0.046, `unit_conversion` 0.16 -> 0.0008). All 3
sampled smoke-test completions (`numeral` x2, `cryptarithm`) came back
`closed_think=True final_boxed=True stop=stop` -- a natural EOS with the
format correctly closed and boxed, not truncated mid-trace. This is the
first evidence the six-tag/`<think>`/`\boxed{}` format is actually learnable
through this pipeline, not just internally consistent on synthetic gold
traces and reward functions. Export produced a 279 MB adapter plus an 8.8 GB
merged HF model at `sft_tinker_runs/full_0727/merged_hf_model`
(`Qwen3_5ForConditionalGeneration`, no LoRA leftovers).

**Open gap: no resume path.** The script never calls the SDK's
`load_state_async` (it exists on `TrainingClient`; grep finds zero uses
here). Periodic/interrupted checkpoints exist as `tinker://` paths in
`checkpoints.jsonl` and survive the local process dying, but restarting
after a crash means training from step 0 again at full cost -- there's no
`--resume-from` flag to pick up a saved checkpoint. Worth adding before
relying on this for a run long enough to risk interruption.

**Operational note, not a script bug:** unlike a Kaggle GPU job, Tinker runs
the model server-side but this script's own process drives the loop --
it issues each step's request and blocks on the result, so the machine
running it must stay powered on and connected for the whole run (sleep
breaks it; closing the terminal doesn't if launched with `nohup`/`tmux`).
Each run's export is a full 8.8 GB local copy under `sft_tinker_runs/`
(now gitignored, along with `.env`); prune old `merged_hf_model/` dirs
there if disk is a concern -- the 279 MB `adapter/` alone is enough to
rebuild the merge offline with `tinker_cookbook.weights.build_hf_model`,
no API call needed.

Not yet done: `train_sft_tinker.py` has no `--resume-from` (see above), and
like the local script, SFT -> GRPO chaining is unwired (prompt-format
mismatch with `train_grpo_cipher_kaggle.py`, and that script would need
pointing at a merged checkpoint or adapter here).

## The GRPO training script

`kaggle/train_grpo_tinker.py` (added this session, untracked) is stage 3 for
six of the seven categories: it continues training from the `full_0727` SFT
checkpoint (see "The Tinker SFT training script") using
`tinker_cookbook.rl.train.Config` + `main()` -- the hosted RL loop, not TRL's
`GRPOTrainer` (`train_grpo_cipher_kaggle.py`'s stack, which targets a
different, much smaller local model). This is a platform constraint, not a
preference: the merged SFT model is 9.3 GB bf16 and the local GPU is 5.67
GiB, so GRPO has to run where SFT did.

**Design, in one paragraph:** for each enabled category,
`ReasonerDatasetBuilder` loads `synth_sft.jsonl`, subtracts every row
`full_0727`'s SFT run would have seen (reproducing
`train_sft_kaggle.apply_category_caps(seed=0)` with that run's own
`category_caps`, keyed on `(category, id)` -- a small conservative
superset of the exact train split, see `sft_holdout_keys`'s docstring for
why exact reconstruction isn't worth it), parses each prompt's examples via
`scripts/eval_train_csv.PARSERS`, and hands batches of
`ProblemGroupBuilder`-wrapped `ReasonerEnv` instances to the training loop.
`ReasonerEnv` subclasses `tinker_cookbook.rl.types.Env` **directly, not
`ProblemEnv`** -- `ProblemEnv.step` reads the sampled response through
`renderers.get_text_content()`, which strips `<think>` content, and the
entire `<step>`-tagged trace lives inside `<think>...</think>` (see
`store_types.wrap_trace_with_think`). Reward text is built as
`think_prefix + tokenizer.decode(action_with_stop_token_stripped)`:
`think_prefix` is derived from `train_sft_kaggle.chat_template_think_prefix`
(the same function that decided what SFT stripped) rather than hardcoded, and
the stop-id token is stripped from `action` before decoding because
`renderer.get_stop_sequences()` returns a token id that Qwen3's own
`parse_response` docstring confirms is present in the raw sampled tokens --
left in place it decodes to literal `<|im_end|>` text glued onto the
completion. `evaluate_structured_trace`'s own correctness grading turns out
to be immune to this (it locates the boxed answer inside the conclusion
tag's regex-captured content, bounded by that step's own `</step>`, so
trailing garbage after it is inert -- confirmed empirically, see below), but
the `correct` *metric* and any code using `scripts/eval_train_csv.last_boxed`
on the full text is not, since that helper requires the string to literally
end in `}`. The reward call is wrapped in `try/except Exception -> raw =
-reward_clip`, since `reward_gravity.py`/`reward_unit_conversion.py` call
bare `float()` on regex-captured (adversarial) trace content and can raise
on malformed model output.

**Cipher is deliberately excluded** (`REWARDS` has no `"cipher"` entry).
`reward_cipher.evaluate_structured_trace` takes `(text, oracle_map,
expected_words)`, not `(text, examples, expected_answer)`, and grades letter
claims via `oracle_map.get(cipher) == plain` -- a key missing from a partial
oracle scores as *wrong*, not unknown. A real/synthetic-corpus row's handful
of example sentences never covers the full 26-letter bijection that reward
function needs; only a bijection-first construction (`monitor_cipher.py`'s
pattern, same as `train_grpo_cipher_kaggle.py` already does) can supply a
complete oracle. Wiring cipher in means a second env/dataset path built that
way, not reusing `ReasonerEnv`.

**All five `--dry-run` pre-flight checks are implemented and have been run
(2026-07-28), $0, no Tinker client constructed.** The first four call
`REWARDS[cat](...)` directly and never exercise `ReasonerEnv` itself (stop-id
stripping, the `think_prefix + decode(...)` concatenation, `rpartition(
"</think>")`, `eval_csv.last_boxed(tail)`, `StepResult` construction) -- a
gap closed by the fifth:
- `_check_sft_holdout`: default categories (`numeral`, `equation_numeric`)
  each have 2000 rows, 300 held out (matching `full_0727`'s
  `category_caps`), 1700 available for GRPO.
- `_check_gold_trace_rewards` (200 gold rows/category): **zero negative
  scores** for all six non-cipher categories, and **zero
  `stop_text_mismatches`** -- appending the renderer's decoded stop token to
  a gold trace before scoring never changed the reward, confirming the
  conclusion-tag-scoped boxed-answer extraction really is immune to trailing
  stop-token text. Measured gold-reward ranges (min/p50/max): numeral
  19.95/22.15/23.85, equation_numeric 2.90/17.95/21.00, cryptarithm
  12.07/27.61/28.46, gravity 20.35/25.10/25.10, unit_conversion
  20.45/23.05/24.70, bit_manipulation 14.75/14.75/14.75 (constant --
  consistent with its rows surviving via `monitor_bit_manipulation.py`'s
  retry-until-match loop landing on a narrow set of rule shapes, see "Two
  kinds of reasoner module"). `Cfg.reward_clip = 25.0` was chosen from this
  distribution so gold numeral traces land just under the ceiling (~0.95)
  rather than saturating at 1.0.
- `_check_prompt_token_identity`: **byte-identical** -- 288 tokens match
  exactly between `train_sft_kaggle.render_prompt` (what SFT trained on) and
  `renderers.get_renderer("qwen3_5", ...).build_generation_prompt(...)` for
  the same messages, and the derived `think_prefix` (`'<think>\n'`) matches
  the renderer's own generation-prompt tail. The Qwen3.5 renderer's claimed
  HF-template parity holds for this project's prompts.
- `_check_trace_length_percentiles` (Qwen3.5-4B tokenizer, trace tokens
  only): numeral p50/p95/max = 496/615/702, equation_numeric =
  342/415/469 -- both comfortably under `Cfg.max_tokens = 1280` (chosen at
  ~1.8x the observed max, not just p95, so an early-training policy rambling
  past gold length doesn't get truncated and penalized as a confound). The
  other four categories were also checked at `--max-tokens 8192`: cryptarithm
  1588/1663/1728, gravity 5861/7435/8073, unit_conversion 5303/7280/8025,
  bit_manipulation 6691/6991/7826 -- all pass every check, so enabling them
  later (`--categories cryptarithm,gravity,unit_conversion,bit_manipulation`)
  needs only a `--max-tokens` bump, per ROADMAP Step 8.
- `_check_env_step_closed_loop`: builds a real `ReasonerEnv`, feeds it a
  gold trace turned back into a synthetic sampled `action` (the trace with
  `think_prefix` stripped, re-tokenized, plus the stop-id token appended --
  simulating exactly what a `stop_reason == "stop"` sample looks like), and
  drives it through `initial_observation()` + `step()`. All six categories
  came back `format=1.0 correct=1.0 truncated=0.0` with positive reward
  (0.59-1.00, reflecting `Cfg.reward_clip`'s squash of each category's own
  gold-reward ceiling -- see the ranges above). `initial_observation()`'s
  `ModelInput.to_ints()` was also asserted equal to the SFT prompt tokens
  for the same row, closing `_check_prompt_token_identity`'s gap of testing
  the renderer directly rather than the env's actual use of it. The
  `correct` metric is tolerance-aware (`_boxed_matches`, `1e-2` for
  `gravity`/`unit_conversion` only, exact match otherwise) to match those
  two reward functions' own documented tolerance (see "Why the reward
  functions are shaped differently") -- without this, `correct` would
  under-report on the ~2% of gravity/unit_conversion rows whose gold trace
  is a near-exact (not byte-exact) reconstruction. A caught reward-function
  exception (trap 5) also sets a `reward_exception` metric distinct from
  `reward_raw == -reward_clip`, since a crashing reward function and a
  genuinely terrible completion would otherwise look identical in the logs.

### How this was verified (source-level, against the installed package)

`plan_grpo.txt` (pre-existing in the repo before this session, presumably
from an earlier planning pass) named specific file:line claims about
`tinker_cookbook` 0.5.2 / `tinker` 0.23.4's behavior. Before writing any
code, every one of those claims was checked by reading the actually
*installed* package in `.venv/lib/python3.11/site-packages/`, not assumed
from the plan or from upstream docs (versions drift -- the Tinker SFT
script's own docstring already documents one earlier draft going stale
against a sibling repo's wrapper, see "The Tinker SFT training script").
Specifically read and confirmed:

- `tinker_cookbook/rl/types.py`: `Env`/`EnvGroupBuilder`/`StepResult`/
  `Action`/`ActionExtra`/`RLDataset`/`RLDatasetBuilder` signatures; the
  `Env` docstring's own worked example, which returns a bare `(observation,
  stop_condition)` tuple from `initial_observation` -- confirming
  `renderer.build_generation_prompt(...)` returns a plain `ModelInput`, not
  a tuple, in this installed version (a detail `ProblemEnv.initial_
  observation` itself constructs the tuple around).
- `tinker_cookbook/rl/problem_env.py`: `ProblemEnv.step`'s exact body,
  confirming the `renderers.get_text_content(message)` call (trap 1) and
  that `ProblemGroupBuilder` is a plain (non-`chz`) dataclass whose
  `env_thunk: Callable[[], ProblemEnv]` type hint is not runtime-enforced --
  confirmed by construction (`ReasonerEnv` is not a `ProblemEnv` subclass
  and `ProblemGroupBuilder(env_thunk=partial(ReasonerEnv, ...))` still
  works, since `make_envs` just calls the thunk).
- `tinker_cookbook/renderers/base.py`: `get_text_content`'s docstring and
  body (strips `ThinkingPart`s); `Renderer.tokenizer` is a real public
  attribute (`self.tokenizer = tokenizer` in `__init__`); `build_generation_
  prompt`'s actual return type.
- `tinker_cookbook/renderers/qwen3.py` / `qwen3_5.py`: `get_stop_sequences`
  returns `[self._end_message_token]` (a token id, not a string);
  `Qwen3_5Renderer._get_generation_suffix` appends literal `"<think>\n"`;
  `Qwen3Renderer.parse_response`'s docstring says it "decodes the response,
  strips the `<|im_end|>` stop token" -- the textual basis for trap 3 (see
  below for why this is an inference, not a live-verified fact).
- `tinker_cookbook/rl/data_processing.py`: `compute_advantages`'s exact
  body -- confirms group-centering only, no std division, so raw reward
  magnitude scales the gradient directly (motivates `reward_clip`).
- `tinker_cookbook/recipes/math_rl/math_env.py` and `train.py`: read in
  full as the closest existing analogue (a single-turn verifiable-reward
  recipe) -- `MathDataset`/`MathDatasetBuilder`/`ProblemGroupBuilder` usage,
  and `cli_main`'s exact `Config(...)` construction, which
  `train_grpo_tinker.run_training` mirrors field-for-field (recipe_name,
  renderer_name resolution, load_checkpoint_path, async_config left unset,
  etc.).
- `tinker_cookbook/rl/train.py`: `Config`/`AsyncConfig`/
  `StreamMinibatchConfig` field lists (confirmed `base_url`, `ttl_seconds`,
  `rollout_error_tolerance` etc. exist so future tuning has real field
  names to reach for); `main()`'s body around line 1930-1966, confirming
  `load_checkpoint_path` goes through `create_training_client_from_state_
  async` (fresh optimizer state) rather than `create_training_client_from_
  state_with_optimizer_async` (which is reserved for *resuming* an
  interrupted run of the GRPO script itself, detected via `checkpoint_
  utils.get_last_checkpoint(config.log_path)` -- not relevant to loading an
  SFT checkpoint as a starting point).
- `tinker_cookbook/checkpoint_utils.py`: `resolve_renderer_name_from_
  checkpoint_or_default_async`'s exact signature and precedence (explicit
  name wins; else checkpoint metadata; else `model_info`'s recommendation).
- `tinker_cookbook/cli_utils.py`: `LogdirBehavior` literal values and
  `check_log_dir`'s signature.
- `tinker` (the base SDK, not cookbook): `types/_pydantic_types/sampling_
  params.py` (`SamplingParams.stop: Union[str, Sequence[str], Sequence[int],
  None]`) and `types/sampled_sequence.py`/`stop_reason.py` (`StopReason =
  Literal["length", "stop"]`) -- read directly since neither confirms nor
  contradicts whether a stop *token id* passed as `stop` ends up included in
  the returned `.tokens`; this remains an inference from the renderer
  docstring, not a directly-confirmed SDK behavior (see below).
- Constructed a real `tinker.ModelInput` interactively (`uv run python -c
  "..."`) to confirm `.chunks`/`.to_ints()`/`.length` exist and behave as
  expected, and constructed a real `ReasonerDatasetBuilder(...)` instance to
  confirm `chz.chz` accepts `tuple[str, ...]` and `dict`-typed fields (it
  does, matching how `Config.loss_fn_config: dict[str, Any] | None` and
  `CLIConfig.env: str` etc. are declared elsewhere in the installed package).

This is the same discipline CLAUDE.md already asks for elsewhere in this
project (`_factor_for_example` bit-identical between generator and reward,
etc.) applied to a third-party dependency: don't trust a plan's or a
docstring's claim about library behavior without reading the installed
source, and where reading isn't conclusive, run the smallest possible
snippet that would falsify it.

### Assumptions this design rests on that were *not* independently verified

Everything below was either inferred from a docstring/type rather than
observed behavior, or is a deliberate judgment call recorded here so it
isn't re-litigated silently later:

- **Stop-token inclusion in sampled output (trap 3) is an inference, not an
  observed fact.** No live `sample_async` call was made (that needs a
  `tinker.ServiceClient` and bills the sampling meter), so whether
  `stop_reason == "stop"` sampled tokens actually include the stop-id token
  was never directly observed -- it rests entirely on
  `Qwen3Renderer.parse_response`'s docstring wording ("decodes the
  response, strips the `<|im_end|>` stop token"). `ReasonerEnv.step`'s
  stripping is written defensively either way (a no-op if the token turns
  out to already be absent, since `toks[-1] in self._stop_ids` just won't
  match), so this assumption being wrong would not break anything -- but it
  is worth re-examining directly on the first real rollout's raw token IDs
  before trusting it further.
- **`renderer_name="qwen3_5"` for the dry-run checks is a hardcoded guess,
  not resolved from the checkpoint.** The real run instead calls
  `checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async`
  against the live service (`run_training`, not `--dry-run`'s path) --
  correct for `full_0727` (a stock Qwen3.5-4B LoRA with no custom
  renderer registered), but this has not been confirmed against the
  checkpoint's actual stored metadata, only assumed consistent with how it
  was trained.
- **The SFT-holdout subtraction is a deliberately conservative
  approximation, not an exact train/test partition** (see
  `sft_holdout_keys`'s docstring): it excludes ~74 more rows than `full_
  0727` actually trained on (the val split + 1 length-dropped row), because
  reconstructing the exact boundary would require re-running SFT's own
  tokenizer-dependent `build_examples`/`stratified_split` for a benefit
  (a few dozen more available rows out of ~1700/category) judged not worth
  the complexity.
- **`Cfg.reward_clip = 25.0` and `Cfg.max_tokens` (1280 default; 8192 when
  the other four categories are enabled) are chosen from the *gold-trace*
  reward/length distributions**, not from any real policy rollout (none
  exist yet). An actual early-training policy may ramble longer or shorter
  than gold traces in ways this can't predict; both are `Cfg` knobs
  specifically so they can be revisited after the first smoke run's real
  data, per `_check_trace_length_percentiles`'s own comment.
- **Hyperparameters copied from `plan_grpo.txt` are starting points, not
  tuned values**: `learning_rate=1e-5`, `group_size=8`,
  `groups_per_batch=8`, `save_every=5`, `kl_penalty_coef=0.0`,
  `remove_constant_reward_groups=True`, `num_substeps=1`. None of these were
  swept or justified beyond "matches the plan's own reasoning" (RL learning
  rate an order of magnitude below SFT's, KL off initially to see the
  unconstrained reward signal first).
- **No sampling-meter price is on file** (`Cfg.sample_price_per_m_tokens`
  defaults `None`, unlike `train_sft_tinker.py`'s verified `TRAIN_PRICE_
  PER_M_TOKENS`) -- the cost estimate this script prints before any paid
  call is a token count only, not a dollar figure, until a real price is
  supplied via `--sample-price-per-m-tokens` or looked up at
  https://tinker-docs.thinkingmachines.ai/tinker/models/.
- **This session's advisor review caught a real gap**, worth recording as
  process, not just outcome: the first draft of the `--dry-run` checks
  (SFT-holdout, gold-trace rewards, prompt-token-identity, trace-length
  percentiles) all scored gold traces by calling `REWARDS[cat](...)`
  directly, never actually constructing a `ReasonerEnv` or calling
  `.step()` -- so none of them exercised the stop-token-stripping,
  `think_prefix` concatenation, or `StepResult`/metrics code that the real
  paid run depends on. `_check_env_step_closed_loop` (added after that
  review, see the checklist above) closes that gap; the takeaway for future
  work on this file is that scoring a reward function in isolation is not
  the same as verifying the environment wrapper around it.

**Not yet done: no paid run.** The `--dry-run` path above is fully verified;
`--yes`/interactive-confirm and the actual `forward_backward`/`optim_step`
loop (`run_training`, which resolves the renderer name from the checkpoint's
own metadata via `checkpoint_utils.resolve_renderer_name_from_checkpoint_or_
default_async` rather than hardcoding `"qwen3_5"`) have not been exercised
against the live service. Per ROADMAP Step 4's "deliberate smoke run"
guidance, the first real invocation should be small (`--max-steps 2
--groups-per-batch 4 --group-size 4`) before a longer run. There is also no
sampling-meter price on file (`Cfg.sample_price_per_m_tokens` defaults
`None`); the cost estimate prints token counts only until that's supplied via
`--sample-price-per-m-tokens` or looked up at
https://tinker-docs.thinkingmachines.ai/tinker/models/.
