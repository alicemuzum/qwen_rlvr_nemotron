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
uv run python -m reasoners.monitor_bit_manipulation # generate+score one random bit_manipulation trace, same style
uv run python reasoners/run_cipher.py            # minimal example: build a Problem, call reasoning_cipher, print the trace
uv run python scripts/eval_train_csv.py          # run every reasoner against all 9500 real train.csv rows, write reasoners_train_csv_eval.csv + print per-category accuracy
uv run python scripts/gen_synthetic_data.py      # bulk synthetic SFT-trace generator across all 7 categories, writes reasoners_synthetic_sft.jsonl (see "Bulk synthetic SFT data generation" below)
uv run python scripts/validate_reward_cipher_partial.py # validates reward_cipher_partial.py (gold-trace scoring, full-oracle parity with reward_cipher.py, adversarial checks)
uv run python scripts/score_policy_completions.py # $0: score the SFT policy's real completions through ReasonerEnv's reward path (does the reward separate right from wrong on actual model output?)
uv run python scripts/eval_sft_tinker_train_csv.py --run-dir <run> --yes # PAID: sample a SFT/GRPO checkpoint on real train.csv rows (250/category, temp 0); --categories to subset, --limit 7 for a ~$0.05 smoke, no --yes = free preview
uv run python scripts/analyze_grpo_rollouts.py grpo_tinker_runs/<run> 10 # $0: pool a GRPO run's per-rollout summaries early-vs-late, bootstrapped over groups (never read a single step)
uv run python scripts/compare_before_after_eval.py BEFORE.csv AFTER.csv  # $0: paired before/after on identical prompts -- McNemar exact + bootstrap CI
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

  **`bit_manipulation.py` now follows the foolproof contract too (gate
  added this session).** `reasoning_bit_manipulation` re-derives a per-bit
  rule vector from only the visible examples (bitsum-hash column matching +
  left/right stride-run extrapolation -- the same algorithm independently
  described in the Kaggle discussion post "How I solved bit manipulation
  problems" by huikang, this repo's original author, whose own reported
  85.1%/1364-of-1602 figure is what this file's solve-rate table below
  already measured), evaluates that vector against `question_bits` via
  `_evaluate_rule`, and now `return None`s unless the result equals
  `problem.answer` -- the same `answer == problem.answer` gate every other
  tagged generator already had. `reward_bit_manipulation.py` and
  `monitor_bit_manipulation.py` both exist and work (same
  `evaluate_structured_trace(response_xml, examples, expected_answer)`
  3-arg shape as `reward_numeral.py`/`reward_gravity.py` -- no oracle param,
  for the same ambiguity reason as cryptarithm/equation_numeric, see "Why
  the reward functions are shaped differently"); `monitor_bit_manipulation.py`
  still retries (construct, run the generator, discard on `None`) but no
  longer needs its own separate boxed-answer re-check now that the
  generator gates internally -- same pattern as
  `monitor_equation_numeric.py`. Solve rate is unchanged at 85.1%
  (`_evaluate_rule` can still land on a different, still
  example-consistent rule than the synthetic ground truth, the same
  ambiguity risk cryptarithm/equation_numeric document), but `solved` and
  `correct` are now the same number -- re-measured on the full `train.csv`:
  1364/1364, zero wrong-answer traces (see the solve-rate table below,
  which no longer needs its "no foolproof contract" caveat).

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
| gravity | **100.0% (1597/1597) on the full dataset**, post tolerance-gate fix (was 60.4%, 964/1597, before -- see "The gravity/unit_conversion tolerance-gate fix" below) | the median-of-truncated-divisions `k` estimate carries real rounding noise and rarely lands byte-exact, but the competition's own official metric accepts a 1e-2 relative tolerance -- once the generator's gate was calibrated to that (instead of exact string match), essentially every row clears it |
| unit_conversion | **100.0% (1594/1594) on the full dataset**, post tolerance-gate fix (was 82.9%, 1322/1594, before) | same fix as gravity, same tolerance -- see "The gravity/unit_conversion tolerance-gate fix" below |
| equation_numeric | ~77% (230/300 sampled); 559/732 = 76.4% on the full dataset | post foolproof-fix + tag conversion; the rest return `None` (a few examples don't always pin a unique operation -- same ambiguity risk as cryptarithm), **zero wrong-answer traces** among the solved rows. Was previously N/A: pre-fix the generator confidently emitted 69/300 wrong answers (see "Current gaps") |
| numeral | 1576/1576 = 100% on the full dataset, 0 wrong | not previously sampled at 300; deterministically recomputed from a fixed universal Roman-numeral table (see "Why the reward functions are shaped differently"), so it isn't ambiguity- or estimation-limited the way cryptarithm/gravity are |
| bit_manipulation | 1364/1364 = 100% of emitted traces correct (85.1% of the full dataset gets a trace at all: 1364/1602) | foolproof gate added this session (see "Two kinds of reasoner module") -- `reasoning_bit_manipulation` now re-checks its derived rule against `problem.answer` and returns `None` on the 238 rows it used to get wrong, instead of emitting a confidently-wrong trace. `solved == correct` now holds, same as every other category. Has a tagged trace format, `reward_bit_manipulation.py`, and `monitor_bit_manipulation.py` |

For all six foolproof generators (cryptarithm, gravity, numeral,
unit_conversion, equation_numeric, bit_manipulation), `scripts/eval_train_csv.py`
confirmed `solved == correct` exactly on the full dataset -- i.e. zero wrong
answers among emitted traces, as the foolproof contract promises. `cipher`
has no such internal check but also measured 0 wrong on the full dataset.
Re-run after the bit_manipulation fix: bit_manipulation 1364/1364, 0 wrong
(previously 238/1602 wrong before the gate was added).

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
were, at the time, genuine median-of-truncated-divisions estimation noise
under an exact-match gate (rounding in the 2dp example values themselves) --
not a further precision issue, since precision beyond 6dp measured no
further improvement (81% flat through 10dp). **That gate was later loosened
to the competition's actual 1e-2 relative tolerance** (see "The gravity/
unit_conversion tolerance-gate fix" above), which absorbed essentially all
of that remaining noise and brought both categories to 100% -- the 6dp fix
documented here is still necessary (it's what keeps the estimate biased
close enough to fall inside that tolerance at all), just no longer the
whole story.

Given this, **the right SFT data source is still a per-category decision,
not a blanket policy** -- though the spread is narrower than it used to be
now that gravity/unit_conversion both solve real rows at 100%: cipher,
gravity, and unit_conversion can all lean on real rows directly; cryptarithm
still needs synthetic-primary generation (real rows stay too sparse, ~8.5%
solvable). Keeping a slice of real solvable rows around as a distribution-
shift check (vocabulary, sentence length, number precision) against purely
synthetic data is still worthwhile regardless.

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

### The gravity/unit_conversion tolerance-gate fix (why solve rate went 60.4%/82.9% -> 100%/100%)

The user linked two Kaggle discussion posts by huikang (this repo's original
author, confirmed by pulling `reasoners/gravity.py`/`reasoners/unit_conversion.py`
directly from his GitHub, `tonghuikang/nemotron`, and diffing) and asked for a
side-by-side accuracy comparison against this repo. huikang's own solve-rate
table (discussion 689915) reports **100.0%** on both gravity (1597/1597) and
unit_conversion (1594/1594) -- against our then-measured 60.4%/82.9%. Every
other category matched his numbers exactly or within a few rows either way, so
this was the one real gap worth chasing.

Root cause: `reasoning_gravity`/`reasoning_unit_conversion`'s own foolproof
gate required **exact string match** (`boxed_answer != round_2dp(problem.answer):
return None`) against `problem.answer` before returning a trace. But the
competition's actual official metric (see its "Evaluation" page, quoted
verbatim) is looser: "A prediction is graded as correct if it matches the
ground truth either exactly as a string or within a relative numerical
tolerance of 10^-2." huikang's own `reasoning.py` driver never self-gates at
all -- it always emits its k/factor estimate and grades separately with
`math.isclose(rel_tol=1e-2, abs_tol=1e-5)`, matching that rule. Our
generators' internal gate was strictly *more conservative* than the metric
they were ultimately trying to satisfy, so they declined (`return None`) on
every row where the median-of-truncated-divisions estimate was close but not
byte-exact -- even though the competition itself would have scored that row
correct.

Verified empirically before changing anything: re-running the existing
k/factor-estimation math unchanged, just swapping the gate's comparison for
`rel_tol=1e-2, abs_tol=1e-5`, gave `1597/1597 (100.0%)` gravity and
`1594/1594 (100.0%)` unit_conversion on the real dataset -- exactly matching
huikang's reported numbers. The estimation algorithm itself was never the
problem; only the gate was stricter than the target.

Fixed in both `reasoning_gravity` and `reasoning_unit_conversion`: the gate
now uses `math.isclose(float(boxed_answer), float(problem.answer), rel_tol=1e-2,
abs_tol=1e-5)`. This required a design choice on what the emitted `\boxed{}`
should claim once exact match is no longer required -- **the user chose to
match huikang's own behavior**: emit the generator's own computed estimate
(`boxed_answer`), not a copy of `problem.answer`. Concretely this means ~1 in
6 emitted gravity/unit_conversion traces now state a final answer that is
within 1e-2 relative tolerance of the true value but not byte-identical to it
(measured on a 60-row synthetic sample: 50/60 exact, 60/60 within tolerance).
This is a deliberate tradeoff: the trace is now always *derivation-honest*
(the boxed answer is what the shown arithmetic actually produces) rather than
always *literally correct*, at the cost of occasionally teaching the model an
answer that's off by a fraction of a percent -- acceptable because that's
exactly what the real evaluation metric tolerates.

Re-verified end to end after the fix: `scripts/eval_train_csv.py` (see below
for why its own `correct` check also needed the same tolerance) now reports
`TOTAL 8336/9500 = 87.7%`, matching huikang's target table (`8333/9500 =
87.7%`) almost exactly. `scripts/gen_synthetic_data.py`'s yield for both
categories jumped from 92.1%/80.6% to 100%/100% on a smoke sample (30/category).

**`scripts/eval_train_csv.py` needed the same tolerance in its own,
independent `correct` check.** That script computes `correct` via a second,
separate exact-string comparison against `problem.answer` (not by trusting
the generator's own gate) -- reasonable when every generator echoed
`problem.answer` verbatim, but now systematically wrong for gravity/
unit_conversion, since their emitted answer can legitimately differ from
`problem.answer` in the last digit while still being correct. Added a
`compare_answer()` helper (mirroring huikang's own, and the competition's
official rule) that treats binary strings ([01]+, to avoid bit_manipulation
spuriously "matching" via naive float parsing that discards leading zeros)
as exact-match-only, and everything else as exact-match-or-1e-2-relative-
tolerance. This is now the metric the whole "accuracy" column in this file
reflects going forward.

**A second, more serious bug surfaced while validating this fix, and is now
also fixed: `reward_gravity.py`/`reward_unit_conversion.py`'s own conclusion
check used a fixed *absolute* tolerance (`abs(final_val - expected) <= 0.01`),
not the competition's *relative* one.** Once the generators started
legitimately emitting answers up to ~1% off (by design, per the choice
above), this mismatch became live: scoring 1500 fresh gold gravity traces
through `reward_gravity.py` found **114/1500 (7.6%) scored
`R_terminal_fail`** despite being correct-by-the-actual-metric -- e.g. a
boxed answer of `1144.88` against a true `1144.60` (0.28 absolute, 0.024%
relative) failed the old `0.01`-absolute check outright. This would have
silently fed wrong (negative) reward signal into GRPO for a meaningful slice
of otherwise-correct gravity rollouts. Fixed in both files: `_TOLERANCE`
replaced with `_REL_TOLERANCE = 1e-2` / `_ABS_TOLERANCE = 1e-5`, and the
conclusion check now uses `math.isclose(final_val, expected_answer,
rel_tol=_REL_TOLERANCE, abs_tol=_ABS_TOLERANCE)` -- the same call shape as
the generators' own gate. Re-verified: 1500/1500 gold traces now score
`R_terminal_win` for both categories (0 negative, 0 `R_terminal_fail`), and
a targeted adversarial check (corrupting the conclusion tag's boxed value by
+20%) still correctly scores `R_terminal_fail` on 50/50 sampled traces --
the tolerance is calibrated, not simply disabled.

**Not re-measured after this fix**: the SFT-corpus percentile tables and the
`--probe-headroom` per-category accuracy table further below in this file
predate this fix and were measured against the old, stricter generators +
the old, buggy absolute-tolerance reward functions. The GRPO dry-run's
gold-trace reward ranges/ceilings (`_check_gold_trace_rewards`,
`Cfg.reward_clip=25.0`) in particular should be re-run before relying on
them, since the 114/1500 false-negative rate measured above means the old
gravity numbers in that table understated the true gold-reward floor.

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
  to `numeral` + `equation_numeric` per ROADMAP Step 7.
  `train_grpo_tinker.py` **has now had a real paid run** (2026-07-30,
  gravity + unit_conversion, 30 steps, $9.95 -- see "The first GRPO
  training run" below); the full `forward_backward`/`optim_step` loop is
  exercised and produced a significant, paired +13.2pp on both trained
  categories. `train_grpo_cipher_kaggle.py` (the TRL script) still has not
  been run. Note the defaults are still `numeral` + `equation_numeric`,
  which the headroom analysis says are the two *worst* picks available.
- `bit_manipulation` now has a tagged generator, `reward_bit_manipulation.py`,
  and `monitor_bit_manipulation.py` (**all uncommitted in git as of this
  writing** -- `reasoners/bit_manipulation.py` is modified but not
  committed, and the other two are untracked new files). **Its
  foolproof-contract gap is fixed (this session).** `scripts/eval_train_csv.py`
  had measured 1364/1602 (85.1%) correct on the full `train.csv` dataset,
  with 238 rows where `reasoning_bit_manipulation` emitted a fully-formed
  trace and boxed answer that was simply wrong -- the tag-wrapping work had
  wrapped the existing narrative in `<step>` tags without adding the
  `answer == problem.answer` gate every other generator already had. The
  gate is now in place (`reasoning_bit_manipulation` evaluates its derived
  rule vector against `question_bits` via `_evaluate_rule` and returns
  `None` on mismatch); re-running `scripts/eval_train_csv.py` confirms
  `solved == correct` at 1364/1364, zero wrong answers.
  `monitor_bit_manipulation.py`'s own separate retry-until-match check
  (previously needed to route around the missing gate) was simplified away
  now that the generator gates internally -- it just retries on `None`,
  same as `monitor_equation_numeric.py`. See "Two kinds of reasoner module"
  and the solve-rate table above.
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
- The 6 foolproof generators (cryptarithm, gravity, numeral,
  unit_conversion, equation_numeric, bit_manipulation) are trusted as-is
  whenever they return non-`None` -- their own `answer == problem.answer`
  gate already guarantees correctness, and re-parsing their boxed answer
  here would hit the same brace-balancing hazard the "Gotcha" note above
  describes for cryptarithm.
- cipher (the one generator without a foolproof gate) is additionally
  checked here: the trace is kept only if its last `\boxed{...}` equals
  `problem.answer`. Its answers are brace-free (plaintext words), so a
  naive boxed regex is safe here specifically -- it would not be for
  cryptarithm.

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
80.6% -- bit_manipulation's low yield cost ~35,500 attempts for its 2000
rows, still well under the collector's attempt cap. **That 5.6% was long
written off as expected ("it has no foolproof gate"); both halves of that
were wrong** -- the gate had already been added, and the real cause was
that the synthetic generator drew a rule distribution the task doesn't
use. Fixed 2026-07-30, yield now ~63%; see "bit_manipulation's synthetic
generator modelled the wrong task" below. These 2000 rows predate the fix.
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
  not a uniform sample of the task. **The root cause of that bias is now
  found and fixed -- see the next section. The 2000 bit_manipulation rows
  in `synth_sft.jsonl` predate the fix and are the biased slice; regenerate
  before any bit_manipulation SFT.**

### bit_manipulation's synthetic generator modelled the wrong task (fixed 2026-07-30)

The 5.6% generation yield above and the 85.1% real-`train.csv` solve rate
looked contradictory -- same solver, same 8-bit problems, real prompts
carrying 7-10 examples against the synthetic 8. Measured side by side:
**84.6% (423/500) on real rows vs 6.5% (26/400) on synthetic.** The gap was
never solver strength; it was that `monitor_bit_manipulation.make_synthetic_
problem` generated a *different task* from the one `train.csv` poses.

The old `_random_rule_expr` drew each of the 8 output bits' rule
**independently and uniformly** -- output bit 0 might be `bit3 XOR bit7`,
bit 1 `NOT bit5`, bit 2 a constant, with no relationship between them. Eight
unrelated rules are close to uninferable from 8 examples; that's an
information limit, not a heuristic weakness. More examples barely helped
(swept 4 -> 24 examples: 4.0% -> 7.5%), which is the signature of
unidentifiability rather than under-constraint.

What real problems actually look like, measured from the solver's *own*
derived vectors on 338 real rows it verifiably solves (parsed out of the
emitted execution step, so these are rules that reproduce the hidden
answer, not brute-force guesses -- a first attempt at brute-forcing the
rule space was abandoned as unreliable: candidate ordering biased the family
counts and 61 bits had no consistent candidate at all, meaning real rules
use vocabulary outside the obvious grammar):

- consecutive same-family **runs** per problem: 1:31, 2:149, 3:157, 4:1
- within a run, `(primary - bit_index) % 8` is **constant in 526/533
  (98.7%)**; `(secondary - bit_index) % 8` in 258/260 (99.2%)
- family, per output bit: I 1126, const 356, AND 268, XOR 253, XOR-NOT 192,
  OR 175, OR-NOT 157, NOT 94, AND-NOT 83

That constant-difference property is the whole thing: a real problem is a
few consecutive bit runs where the *input* position advances in lockstep
with the output position -- precisely the left/right stride-run
extrapolation `bit_manipulation.py` searches for (and why its code is
organized around `left_run`/`right_run` at all). Confirmed causally, not
just correlationally: one family with a constant stride scores **200/200 =
100%**, while *family count* is not the driver at all -- drawing all 8 bits
from a pool of only one family, but with unstructured positions, still
yields just 8.5%.

`_random_rule_expr` is replaced by `_random_rule_vector`, which samples runs.
`make_synthetic_problem`'s `num_examples` also became `None`-defaulted and
now draws 7-10 to match real prompts (measured 116/114/139/131 over 500
rows), fixing the "fixed example count" complaint above for this category.

**`_RUN_WEIGHTS` is deliberately NOT the measured real frequency.** The
foolproof gate discards unevenly -- survival by run count is **100% / 97% /
42%**, since a 3-run vector gives each run fewer bits to pin its family and
offset from, so the solver more often lands on a different-but-example-
consistent rule. Weighting by the raw real frequencies produced a *kept*
corpus at 27/58/15 against a real 9/44/46: systematically easier than the
real task, i.e. the same class of selection bias the whole fix is about,
just an order of magnitude smaller. The weights are therefore the real
frequencies divided by those survival rates, `(6, 28, 67)`. This is the
generalizable lesson: when a generator is gated by a filter, validate the
distribution that **survives** the gate, not the one you draw.

Measured after the fix (800 draws): yield **62.9%** (from 6.5%), kept
runs/problem **11/48/41** vs real 9/44/46, kept family mix within ~1-3pp of
real on all nine families, examples/problem 7-10 matching real. The
remaining ~22pp gap to the 84.6% real-row rate is **genuine rule
ambiguity** -- the solver lands on a different rule that still reproduces
every example -- which the foolproof gate correctly discards rather than
emitting a wrong trace. That is not a residual defect and isn't worth
chasing.

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

**Cipher was excluded for this reason, and is now wired in (fixed
2026-07-29).** `reward_cipher.evaluate_structured_trace` takes `(text,
oracle_map, expected_words)`, not `(text, examples, expected_answer)`, and
grades letter claims via `oracle_map.get(cipher) == plain` -- a key missing
from a partial oracle scored as *wrong*, not unknown. A real/synthetic-corpus
row's handful of example sentences never covers the full 26-letter bijection
that reward function needs; only a bijection-first construction
(`monitor_cipher.py`'s pattern, same as `train_grpo_cipher_kaggle.py` already
does) can supply a complete oracle.

`reasoners/reward_cipher_partial.py` (new file; `reward_cipher.py` itself is
still what `monitor_cipher.py`/`train_grpo_cipher_kaggle.py` use, untouched
on this point) is a superset that grades a letter outside the oracle on
self-consistency only -- first claim neutral, reaffirm small/capped under a
separate ceiling so it can't crowd the budget for a verified reaffirmation,
contradiction still penalized -- rather than assuming "wrong". Its
`evaluate_structured_trace_from_examples(text, examples, expected_answer)`
matches the uniform 3-arg shape and reconstructs the oracle from only the
prompt's own visible examples (`partial_oracle_from_examples`, the same
first-mapping-wins rule `cipher.py` uses internally, reimplemented rather
than imported so this file has no data-generation dependency). `REWARDS` now
has a `"cipher"` entry pointing at this adapter, so `--categories cipher`
works end to end (parsing, gold-trace scoring, and
`_check_env_step_closed_loop` all pass) -- it's just not in the *default*
category tuple, for the same cost reasons Step 7 chose numeral/
equation_numeric first (cipher's median trace is ~4700 tokens, and its own
`_check_trace_length_percentiles`/`_check_policy_length_distribution` still
warn that the default `Cfg.max_tokens=2048` is too low for it -- raise
`--max-tokens` when actually enabling it, per Step 8).

While wiring this up, `_check_reward_discrimination` (never previously
exercised against cipher) surfaced **two independent, pre-existing bugs in
`reward_cipher.py`'s own scoring**, unrelated to oracle completeness, fixed
in both `reward_cipher.py` and `reward_cipher_partial.py`:
1. The dashed-reconstruction verification check could compare an unrelated
   already-known word's line against a stale `last_best_match_word` left
   over from an earlier word (same "->"+"-" shape as the genuine
   reconstruction line). Provably inert on `cipher.py`'s own gold traces
   (loop 1 always finishes before loop 2 sets `last_best_match_word`), fixed
   as defense-in-depth by gating on adjacency to the actual "Best match:"
   line it's meant to follow.
2. **The actual cause of the discrimination failure**: the "Best match:"
   dictionary-lookup check graded the resolved word against
   `expected_words[verified_word_count]` -- treating "how many dictionary
   lookups have succeeded so far" as the word's position in the question.
   Those only coincide when every earlier word also needed a lookup; the
   moment an earlier word was already decodable from the visible examples
   (common, and never increments that counter), every later lookup got
   graded against the wrong expected word and failed even when correct.
   Measured: with the original indexing, 40/80 sampled gold traces (full,
   real oracle) scored *higher* after every verification step was deleted
   than intact -- a foolproof-contract gold trace must never score worse for
   including more honest steps. Fixed by tracking each word's absolute
   position directly (a queue populated in question order, consumed by each
   dictionary lookup in turn) instead of the success counter. Re-measured
   post-fix: 0/150 violations, with both a full and a reconstructed oracle.
   See `reasoners/reward_cipher_partial.py`'s module docstring for the full
   writeup, and `scripts/validate_reward_cipher_partial.py` for the
   validation suite (gold-trace scoring, full-oracle parity with
   `reward_cipher.py`, and the adversarial checks -- all passing).

**All seven `--dry-run` pre-flight checks are implemented and have been run
($0, no Tinker client constructed; five on 2026-07-28, two added 2026-07-29 --
see "Audit fixes" below).** Most call `REWARDS[cat](...)` directly and never
exercise `ReasonerEnv` itself (stop-id stripping, the `think_prefix +
decode(...)` concatenation, `rpartition("</think>")`,
`eval_csv.last_boxed(tail)`, `StepResult` construction) -- a gap closed by
`_check_env_step_closed_loop`:
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
  342/415/469. **These are *gold* trace lengths and are only a lower bound on
  what the policy emits** -- `Cfg.max_tokens` must be sized from
  `_check_policy_length_distribution` instead (see "Audit fixes" below; the
  original 1280 came from this table and was too low). The
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
- `_check_reward_discrimination` (added 2026-07-29): see "Audit fixes" below.
- `_check_policy_length_distribution` (added 2026-07-29): see "Audit fixes".

### Audit fixes (2026-07-29) -- read this before trusting the numbers above

The script was audited after the checks above were written. Every claim it
made about `tinker_cookbook` held (`stop_reason` really is passed via
`ActionExtra`, `rollout_runner.py:412`; `FailFast` really is the default for
non-Inkling models, `rollout_presets.py:184`; `ProblemGroupBuilder.compute_
group_rewards` really does return zeros; the SFT-holdout caps/seed really do
match `full_0727/config.json`; the run really is single-epoch, `for i_batch in
range(start_batch, end_batch)`, which is what makes the training-set `correct`
metric an unbiased online estimate -- and what breaks if anyone ever runs more
than one pass). What the checks did *not* cover was the data and the policy:

- **`Cfg.max_tokens` was sized from the wrong distribution (1280 -> 2048).**
  It was derived from gold traces (max 469 for equation_numeric) at "~1.8x the
  observed max". But rollouts come from the *policy*, and
  `sft_tinker_train_csv_eval.csv` shows the `full_0727` policy at temperature 0
  already emits far longer output: equation_numeric clean-stop p95 = 3217, max
  = 7093, with 32/250 samples hitting the sampler cap outright. The number that
  settles it is the max over *correct* completions -- 759 (numeral) / 1545
  (equation_numeric): no right answer has ever been longer, so 2048 cannot
  truncate a correct rollout, while capping the spend on loops at ~25% of what
  8192 would cost. Truncation above 2048 is now a deliberate penalty on
  degenerate output, not a confound. `_check_policy_length_distribution` prints
  this table at `--dry-run` and warns if `max_tokens` drops below 1.2x the
  longest correct completion.
- **The cost gate under-reported the spend by ~425x.** `estimate_rollout_cost`
  assumed `steps = 1` when `--max-steps` was unset -- but unset is the default,
  and `rl/train.py:1985` then runs a full pass. It now defaults to
  `len(dataset)` and prints both a worst case (every rollout hits `max_tokens`)
  and a likely case from the policy's measured mean completion length: 422
  steps x 64 rollouts, 63.1M worst / 36.8M likely sampling tokens.
- **Nothing measured whether the policy has headroom -- `--probe-headroom`
  (PAID) now does.** GRPO's gradient comes entirely from reward variance
  *inside* a group (`compute_advantages` centers, no std division), so a group
  whose rollouts all agree contributes nothing. The default categories were
  picked on cost (ROADMAP Step 7), and the SFT checkpoint's own eval says
  numeral is at **90.1%** and equation_numeric at **9.3%** on real `train.csv`
  rows -- one near the ceiling, one near the floor, both shapes where a group
  of 8 agrees. (That eval is on `train.csv` while GRPO rolls out on
  `synth_sft.jsonl`, which is in-distribution for SFT and will score better --
  direction indicative, magnitude not, hence measuring rather than
  extrapolating.) The probe samples the real policy on the real GRPO pool for
  ~1.1M tokens (~3% of a full run) and prints, per category, the fraction of
  groups that are mixed / all-correct / all-wrong, with an explicit verdict:
  **under ~20% mixed, don't train that category** -- switch to
  `unit_conversion` (60.8%) or `gravity` (41.4%). Run it before committing to a
  long run.
- **Brace-in-answer rows are unwinnable and are now dropped from the GRPO
  pool.** Every `reward_<task>.py` extracts the boxed answer with a non-greedy
  `\boxed\{(.*?)\}`, which stops at the first `}`. CLAUDE.md documents this
  hazard for cryptarithm but it is not confined there: **34/2000 (1.7%)
  equation_numeric rows in `synth_sft.jsonl` have answers like `'5}'`, `'20{'`,
  `'{26'`**, and on those even a *gold* trace is graded as wrong (measured gold
  reward p50 **4.97** vs **17.90** for brace-free rows). For GRPO that is worse
  than a mis-scored row -- no completion can score well, so the group's reward
  is uniformly depressed and the centered advantage is pure noise.
  `_prepare_rows` now drops them via `_answer_is_gradable` and reports the
  count. Fixing the reward functions' own extraction (to the structural
  `last_boxed()` technique) is the deeper fix and is still open; it was left
  alone deliberately, since those are validated files.
- **`log_dir_behavior` now defaults to `"raise"`, and `--resume-latest`
  exists.** The old `"ask"` default blocks on stdin -- the same footgun
  CLAUDE.md already records for `train_sft_tinker.py` under nohup/tmux. Note
  that unlike the SFT script, **GRPO resume was never actually missing**:
  `rl/train.py:1924` reads the last checkpoint out of `log_path` and picks
  `start_batch` up from it. It was merely *unreachable*, because `Cfg.log_dir`
  defaults to a fresh timestamp, so a re-run after a crash made a new empty dir
  and paid from step 0. `--resume-latest` points at the newest run with
  checkpoints; `run_training` also prints the recovery command before the loop.
- **`_check_reward_discrimination` (new, $0).** `_check_gold_trace_rewards`
  proves the reward doesn't crash and rates gold highly; it does not prove the
  reward *separates* right from wrong, which is what decides whether GRPO has
  any signal at all. Five mutations per gold trace. Measured (p50, 60
  rows/category): numeral gold 22.05 | answer corrupted 7.05 | bare correct
  answer 10.00 | bare wrong answer -5.00 | verification steps deleted 13.05 |
  steps duplicated 22.10. equation_numeric: 17.90 / 2.93 / 10.00 / -5.00 /
  10.95 / 18.50. **Correctness is worth ~15 points, and deleting verification
  never once raised the score (0/120 rows)** -- the reward rewards being right
  and being careful, not merely looking the part. Two findings from it: the
  brace-in-answer rows above (surfaced as "corrupting the answer changed
  nothing"), and **padding is free** -- duplicating every pre-conclusion step
  raised the reward on 36/60 numeral and 60/60 equation_numeric rows. The
  ceilings bound how far that can go and there is no evidence the policy pads
  under RL pressure, so this is *instrumented, not fixed*: `ReasonerEnv.step`
  now emits `completion_tokens` and `n_step_tags` metrics. If `n_step_tags`
  climbs while `correct` stays flat, that is the moment to add a length
  penalty, with real data to set it from.
- **`_check_env_step_closed_loop` now samples 5 rows/category and exercises the
  failure path.** It previously used `rows[0]` -- the same single row every run
  -- and only ever fed a perfect completion. It now also drives a deliberately
  truncated action with `stop_reason="length"`, which is not an edge case (~13%
  of equation_numeric samples take it) and must come back `truncated=1.0
  format=0.0 correct=0.0` with a reward below the gold rollout's. Verified:
  numeral 0.194 vs gold 0.850; equation_numeric -0.182 vs gold 0.118.

**A held-out test set was deliberately NOT added.** `RLDatasetBuilder` can
return one as its second element (wired to `RLTestSetEvaluator`,
`train.py:1975`), but it would cost ~512 extra rollouts per eval to buy a
number the single-epoch property already provides for free. Revisit only if
the run is ever changed to more than one pass over the data.

### Readiness re-check (2026-07-29, later): the reward works; the default categories don't

All seven `--dry-run` checks were re-run and pass, on four different category
sets (`numeral,equation_numeric`; `cipher`; `unit_conversion,gravity`).
`ruff check .` reports 94 findings, **all pre-existing style noise** in
`reasoners/bit_manipulation.py` (76 of them, old-style `typing` annotations)
and `scripts/` -- nothing in `kaggle/` or the new cipher reward files.

**New $0 measurement: `scripts/score_policy_completions.py`.** Every reward
number quoted above comes from a *gold* trace or a mutation of one, which
shows the reward *can* discriminate, not that it does on what the policy
emits. This script scores the 250 real `full_0727` completions per category
already sitting in `sft_tinker_train_csv_eval.csv` through the exact
`ReasonerEnv.step` reward path. Results (reward p50 for really-correct vs
really-wrong completions, `agree` = `ReasonerEnv`'s `correct` metric vs the
independent eval's verdict):

| category | n | fmt | reward correct | reward wrong | agree |
|---|---|---|---|---|---|
| numeral | 250 | 100% | 19.50 | 2.85 | 250/250 |
| cipher | 250 | 81% | 23.70 | 4.80 | 250/250 |
| unit_conversion | 250 | 91% | 19.85 | 4.08 | 250/250 |
| gravity | 250 | 91% | 15.30 | -4.35 | 250/250 |
| cryptarithm | 199 | 96% | 11.43 | -5.92 | 199/199 |
| equation_numeric | 247 | 87% | 8.43 | -6.60 | 247/247 |
| bit_manipulation | 250 | 62% | 10.40 | -4.60 | 250/250 |

Three things this establishes that nothing else did: the reward separates
right from wrong on **real policy output** in all seven categories (gap
15-20 points everywhere); the trap-5 exception path fires **once in 1,696
completions** (gravity), so a crashing reward function is not a live risk;
and `ReasonerEnv`'s `correct` metric agrees with an independent eval on
**every single scored row**, which matters because that metric is what the
whole run is judged on.

**Brace-answer rates, measured across the whole GRPO pool** (`synth_sft.jsonl`,
2000 rows/category) rather than inferred from one category:
`cryptarithm` **227/2000 = 11.4%**, `equation_numeric` **34/2000 = 1.7%**,
every other category **0**. (The 51/250 = 20% figure the run above reports
for cryptarithm is from `train.csv` via the eval CSV, a *different* corpus --
don't compare the two directly.) `_answer_is_gradable` drops all of these
from the pool; enabling cryptarithm means losing an eighth of its rows, and
would have meant an eighth of its groups being pure noise without the fix.

**Trap 3 is now empirically confirmed and is no longer an assumption.** The
first draft of this script disagreed with the eval on 225/250 numeral rows.
Cause: the script fed the CSV's stored completion, which ends
`...\boxed{LVIII}<|im_end|>`, where `ReasonerEnv.step` strips the stop-token
*id* before decoding. `scripts/eval_sft_tinker_train_csv.py:299` builds that
column as a bare `tokenizer.decode(sequence.tokens)` on the raw output of a
real `sample_async` against the live service, with no stripping (its own
`extract_boxed` docstring says so explicitly) -- so **the stop token really
is present in sampled tokens**, exactly as `Qwen3Renderer.parse_response`'s
docstring implied. `ReasonerEnv.step`'s stripping is load-bearing, not
defensive, and the 225/250 disagreement is a direct demonstration of what
happens without it. The corresponding bullet under "Assumptions this design
rests on that were *not* independently verified" is superseded.

**The blocking issue is category selection, and it is not a code fix.**
GRPO's gradient comes only from reward variance *inside* a group, so a
category where the policy agrees with itself 8 times out of 8 contributes
nothing. Per-category accuracy of the `full_0727` policy on real `train.csv`
rows: numeral **90.0%**, unit_conversion **63.2%**, cipher **53.2%**,
gravity **45.2%**, equation_numeric **9.6%**, cryptarithm **0.8%**,
bit_manipulation **0.4%**. The two defaults (chosen on cost, ROADMAP Step 7)
are the **two worst headroom picks available** -- one nearly saturated, one
nearly floored. `unit_conversion`/`cipher`/`gravity` sit where group
disagreement actually lives, but cost ~4x per rollout (8k-token traces vs
~500). Do not simply swap the defaults from this table: it is single-sample
at temperature 0 on `train.csv`, whereas the run samples 8x at
`Cfg.temperature` on in-distribution `synth_sft.jsonl` rows. Measure it with
`--probe-headroom` (~3% of a full run's tokens), as **two separate
invocations**, not one:

```
--probe-headroom --categories numeral,equation_numeric
--probe-headroom --categories unit_conversion,gravity --max-tokens 10240
```

`probe_headroom` samples at `cfg.max_tokens`, so probing the 8k-token
categories at the 2048 default would truncate nearly every rollout into
`format=0 correct=0`, report uniformly-wrong groups, and return a
"don't train this" verdict on **precisely the two categories with the most
headroom**. The high cap costs nothing extra -- sampling bills emitted
tokens and the cap is only a ceiling.

**The positive reward floor on wrong answers is not a defect.** numeral,
cipher and unit_conversion all score ~2/3 of their *wrong* completions above
zero (numeral p50 +2.85, cipher +4.80, unit_conversion +3.80), because a
well-formed trace earns process credit whether or not it lands the answer.
This does not matter, because `compute_advantages` centres within a group
and never uses the absolute level -- only the *ordering* inside one group of
rollouts on one prompt does any work. Measured as a rank statistic over the
real completions above (probability a correct completion outscores a wrong
one, ties half): numeral **0.988**, cipher **0.968**, unit_conversion
**0.932**, gravity **0.927**, equation_numeric **1.000**. That is a
pessimistic bound -- these pairs come from *different* problems, so
cross-problem difficulty variance is folded in, whereas a real group shares
one prompt. Only 2-3% of wrong completions beat the *median* correct one,
and none do for numeral/cipher/equation_numeric. Ordering is sound; there is
no need to push the wrong-answer floor negative, and doing so would discard
the format signal that makes iteration 0 legible.

The one real consequence to watch: in an **all-wrong** group the surviving
variance is process-quality only, which is a gradient pointed at "look
better" rather than "be right". For equation_numeric that variance is tiny
(wrong-completion p25..p75 spans just 2.1 points, -7.60 to -5.50), so such
groups contribute nearly nothing -- which is also precisely why
equation_numeric cannot learn at 9.6% accuracy. This is the same phenomenon
`remove_constant_reward_groups=True` and the `n_step_tags` instrument
already exist for.

**Prices are now on file** (read from
https://tinker-docs.thinkingmachines.ai/tinker/models/ on 2026-07-29 for
`Qwen/Qwen3.5-4B`, which is **not** among the models carrying that page's
limited-time 50% discount, so these are final). **Three rates, not one** --
prompt/prefill **$0.33/M** ($0.066 cached), generated **$1.005/M**, training
**$0.737/M** (the last matching
`train_sft_tinker.TRAIN_PRICE_PER_M_TOKENS`). All three are now `Cfg` fields
and `estimate_rollout_cost` prints the breakdown.

The correction that dominates every other error here: **GRPO bills both
meters** -- each rollout is sampled *and then* fed to `forward_backward` --
so training is ~45% of the bill, not zero. `remove_constant_reward_groups`
(train.py:1822) does drop groups before the train step at :1824, but only on
*exact* reward equality; with a continuous reward even an all-wrong group
usually varies slightly, so it cannot be relied on to discount the train
meter. `probe_headroom` now reports a `constant_reward=` column measuring how
often it actually fires. The estimate also does not claim the 80% cached-
prefill discount, so it errs a few percent high.

| what | tokens (likely) | cost (likely) | cost (worst) |
|---|---|---|---|
| headroom probe, numeral+equation_numeric | 0.65M | **$0.56** | ~$1.0 |
| headroom probe, unit_conversion+gravity @10240 | 2.59M | **$2.50** | ~$4.9 |
| 50 steps, numeral+equation_numeric | 4.4M | **$6.96** | $12.40 |
| 50 steps, unit_conversion+gravity @10240 | 17.3M | **$29.39** | $58.19 |
| full epoch (422), numeral+equation_numeric | 36.8M | **$58.76** | $104.68 |
| full epoch (463), unit_conversion+gravity @10240 | 160.0M | **$272.14** | $538.85 |

The probe costs under 1% of the run it protects, and is sample-meter only
(it never trains). Capping with `--max-steps` does not break the
single-epoch property that makes the training-set `correct` metric an
unbiased online estimate (each row is still seen at most once); the dry run
now says so when `max_steps` is unset.

### First paid call on the GRPO path: the cryptarithm headroom probe (2026-07-29, ~$0.49)

`--probe-headroom --categories cryptarithm --max-tokens 8192`, 30 groups x 8
samples against the live `full_0727` sampler. This is the **first real
`sample_async` on the GRPO path** -- everything before it was `--dry-run`.

```
cryptarithm  groups=30  mixed=10%  all_correct=0%  all_wrong=90%
             constant_reward=0%  median_reward_spread=0.510
             completion_p95=2056  max=3889  truncated=0%
VERDICT: fewer than 20% of cryptarithm groups carry any gradient.
```

**Do not RL cryptarithm at its current checkpoint.** But read the rest before
concluding the category is hopeless -- three findings here generalize:

1. **Latent capability is far above greedy accuracy.** `sft_tinker_train_csv_
   eval.csv` puts cryptarithm at 0.8% (2/250, temperature 0). The probe found
   **3/30 groups containing at least one correct rollout** -- so pass@8 at
   temperature 1 is roughly an order of magnitude above greedy pass@1, and
   `all_correct=0%` means nothing is saturated. The model is not incapable at
   cryptarithm, it is unreliable at it. That is the shape SFT fixes, and it is
   the empirical support for an SFT-then-RL detour on this category. Caveat
   the sample size honestly: 3/30 has a 95% interval of roughly 2-27%, so this
   resolves "not 40%+", not "exactly 10%".
2. **`remove_constant_reward_groups` fires on ZERO groups** -- the
   `constant_reward=0%` column added for exactly this question. The filter
   keys on *exact* reward equality, and this reward is continuous enough that
   even 8 uniformly-wrong rollouts never tie. So it does **not** discount the
   train meter at all: the cost table above is right as written, and any hope
   that GRPO would skip paying to train on hopeless groups is dead.
3. **All-wrong groups carry a large gradient pointed away from correctness --
   this is now measured, not hypothetical.** The median group (90% of them are
   all-wrong) has a normalized reward spread of **0.510**, i.e. ~12.75 raw
   points, comparable to cryptarithm's entire gold-vs-wrong-answer gap (27.61
   vs 12.61). In an all-wrong group the only surviving variance is process
   quality, so GRPO would push hard on whatever varies there. Cryptarithm's
   format rate is already 96%, so little of that headroom is legitimate
   format learning -- and `_check_reward_discrimination` measured
   **padding raising the reward on 60/60 cryptarithm rows** (duplicated p50
   28.89 > gold 27.61). Training a *low-accuracy* category therefore does
   worse than buy nothing: it actively reinforces padding. The `n_step_tags`
   instrument was added to detect this; on evidence like the above, a length
   penalty should be settled **before** any low-accuracy category is trained,
   not after. Mid-band categories are safer precisely because correctness
   dominates their in-group variance.

Two smaller notes: `truncated=0%` with `max=3889` means **8192 was
over-provisioned for cryptarithm** -- 4096 would halve the worst-case cost
line. And a full cryptarithm epoch is only 188 batches (1502 rows survive the
brace drop and SFT holdout), so ~$43.92 likely / $175.34 worst -- cheaper
than the numeral+equation_numeric pair, for anyone revisiting after an SFT
lift.

Two smaller items found and deliberately left as knobs: **`reward_clip=25.0`
saturates on gravity** (gold p50 == max == 25.10, so every gold-quality
rollout normalizes to exactly 1.0 and advantage among correct rollouts
collapses) -- bump to ~30 if enabling gravity/unit_conversion, noting it
uniformly shrinks all normalized rewards and so acts as a small LR change.
(The former "no sampling price on file" gap is closed -- see the price table
above.)

### The first GRPO training run (2026-07-30): gravity + unit_conversion, 30 steps, $9.95

The first real `forward_backward`/`optim_step` loop against the live service.
Run dir `grpo_tinker_runs/grpo_uc_gravity_0730/`, continuing from `full_0727`.
Settings deviating from `Cfg` defaults, each for a reason recorded above:
`--categories unit_conversion,gravity` (headroom, not cost -- see the probe
below), `--max-tokens 10240`, `--reward-clip 30` (gravity saturates at 25),
`--groups-per-batch 4` (halves cost/step, buying 2x the weight updates at a
fixed budget), `--max-steps 30`.

**Headroom probe first, 20 prompts/category (~$2.3).** `unit_conversion`
mixed=95%, `gravity` mixed=75%, `all_correct=0%` for both -- decisively past
the 20% gate, unlike cryptarithm's 10%. Probe size was raised from 12 to 20
because at 12 the 20% gate falls between 2 and 3 groups, i.e. one group
flipping changes the verdict; both results landed far enough from the gate
that this turned out not to be load-bearing.

**Result, paired against the pre-existing `sft_tinker_train_csv_eval.csv`
("before") on identical real `train.csv` prompts, temperature 0:**

| category | n | before | after | delta | 95% CI | McNemar p |
|---|---|---|---|---|---|---|
| gravity | 250 | 45.2% | **58.4%** | **+13.2pp** | [+7.6, +18.8] | <0.0001 |
| unit_conversion | 250 | 63.2% | **76.4%** | **+13.2pp** | [+7.2, +19.6] | 0.0001 |
| numeral | 100 | 89.0% | **76.0%** | **-13.0pp** | [-21, -5] | **0.0044** |
| cipher | 100 | 54.0% | 53.0% | -1.0pp | | 1.0 |
| cryptarithm | 100 | 2.0% | 0.0% | -2.0pp | | 0.50 |
| equation_numeric | 100 | 8.0% | 10.0% | +2.0pp | | 0.73 |
| bit_manipulation | 100 | 0.0% | 0.0% | 0.0pp | | 1.0 |

GRPO works on this stack: both trained categories moved +13.2pp, each
individually significant. In-distribution (synthetic rollouts, temperature 1,
pooled first-10 vs last-10 iterations) the gain was **+24.7pp [+10.6, +38.4]**.
Do not read that as "half transferred": the in-distribution early block
already contains ~10 iterations of learning, whereas the real-row comparison
is pre-run vs post-run, so the true in-distribution gain from step 0 is
larger and the transfer fraction correspondingly lower than half. The safe
statement is that the training-log gain **overstates** real-row transfer, by
at least 2x.

**Part of the trained-category gain is length control, not new correctness.**
Of the wrong->right flips, **8/44 (gravity) and 12/51 (unit_conversion) were
rows the before-policy had truncated** at the cap with no `\boxed{}` -- worth
~3.2pp and ~4.8pp of the respective +13.2pp. Rows hitting the length cap fell
22->13 (gravity) and 22->5 (unit_conversion). Those flips are genuine
improvements, but the mechanism is the policy learning to finish inside the
budget, not better arithmetic; the remaining ~8-10pp per category is
correctness proper. This is the same effect visible in the smoke test, where
a unit_conversion row went 0%->100% purely because the before-completion ran
8,393 tokens and got cut off.

**Training on two categories cost a third 13 points, and this is the finding
most likely to generalize.** `numeral` fell 89% -> 76% (p=0.0044, 16 rows
right->wrong vs 3 the other way). It is **not** a format failure: 15/16
regressions kept `closed_think`+`final_boxed` and stopped naturally. What
changed is length -- on the regressed rows, completions went 305 -> 869
tokens (+185%) while step tags rose only 15 -> 18 (+20%), and the wrong
answers are wholly different values (`XXIX`->`LXXIV`, `XCII`->`LXII`), not
malformed numerals. **The inflation is specific to the rows that broke, which
is what rules out "longer output is an unrelated symptom":** regressed rows
went 305 -> 869 (x2.85) while numeral rows that stayed correct went 338 ->
362 (x1.07). A category-wide style shift would have inflated both equally.
RL on two categories whose traces run 5-6k tokens of
arithmetic derivation shifted the policy toward verbose derivation
*globally*; applied to numeral, whose gold traces are ~500 tokens and nearly
trivial, the elaboration loses track of the value. **Always pay for the
regression check on untrained categories** -- without the 100-row slice this
would have been reported as a clean win. Untrained-category format actually
*improved* nearly everywhere (equation_numeric 87%->100%, cryptarithm
93%->98%), so format and accuracy moved in opposite directions on numeral;
neither is a proxy for the other.

**Process metrics: the padding exploit never materialized.** CLAUDE.md's
`_check_reward_discrimination` measured padding raising the reward on 60/60
rows for both categories, so `ReasonerEnv.step`'s `n_step_tags` /
`completion_tokens` were watched every step against a pre-committed kill rule
(median `n_step_tags` >1.25x the step-0 baseline sustained 5 steps with
`correct` flat; plus `truncated` >25% and `frac_mixed` <20%). None fired.
`n_step_tags` ended at 23.2 against a 24.7 baseline -- *below* it -- and mean
completion ended at 4,883 against 4,884. There was a mid-run excursion
(completion peaked 7,274 at step 19, truncation 16% at step 14) that reversed
on its own. Format rose 0.94 -> 0.99 and truncation fell 6% -> 1% across the
run. On this evidence a length penalty was correctly *not* added blind.

**Do not read per-step `correct` as a measurement.** At
`groups_per_batch=4`/`group_size=8` each step is **4 distinct prompts** (2 per
category) x 8 rollouts, and rollouts in a group share a problem -- so per-step
`correct` is dominated by which 4 problems were drawn, not by 32 independent
samples. Observed per-step values swung 0.281 / 0.312 / 0.656 / 0.344 / 0.625
/ 0.719 / 0.875 with no trend readable from any single pair of points. The
whole 30-step run is 120 distinct prompts (60/category). Pool blocks of
iterations and bootstrap over **groups**, not rollouts:
`scripts/analyze_grpo_rollouts.py` does this from the per-iteration
`train_rollout_summaries.jsonl` files (which carry per-rollout `correct`,
`format`, `truncated`, `completion_tokens`, `n_step_tags` and the category
tag -- richer than `metrics.jsonl`'s per-step means).

**pass@8 barely moved: 0.90 -> 0.95 in-distribution, against +24.7pp on
per-rollout accuracy.** GRPO here sharpened the distribution toward answers
the policy could already sometimes reach rather than adding capability -- the
same latent-capability-vs-reliability distinction the cryptarithm probe
surfaced. `frac_mixed` correspondingly fell 1.00 -> 0.55 as accuracy rose, so
each further step carries less gradient; that, not budget, is the argument
against simply extending this run.

**Evaluating a GRPO checkpoint needed no new eval script, but did need two
things.** (1) A shim run dir (`grpo_eval_shim/`): a copy of
`full_0727/config.json` plus a one-record `checkpoints.jsonl` pointing at the
GRPO `sampler_path`. Reusing the SFT config is what keeps before/after
comparable -- `prompt_style`, `max_seq_len`, `seed` and `category_caps` drive
row selection, so the after-run scores byte-identical prompts. (2) A
`--categories` flag on `scripts/eval_sft_tinker_train_csv.py`, without which
the planned spend split (250 rows on the trained pair, 100 on the other five)
could not be expressed and `--per-category 250` would have sampled all 1750
rows. **That flag has a trap worth preserving**: `select_rows` shares one RNG
across categories in `CATEGORIES` order, so filtering *before* the shuffle
shifts every later category's row order and silently selects a different
subset than an unfiltered run -- breaking comparability while still producing
plausible numbers. The filter therefore narrows what is *emitted*, never what
is shuffled; verified by reproducing the before-CSV's gravity/unit_conversion
rows exactly (250/250) and confirming each 100-row set is a strict prefix of
its before-250. `scripts/compare_before_after_eval.py` does the paired
analysis (McNemar exact on discordant pairs + bootstrap CI), which is the
right test given identical prompts.

**No contamination:** `synth_sft.jsonl`'s 14,000 prompts and `train.csv`'s
9,500 share **zero** strings, so `train.csv` is genuinely held out from both
SFT and GRPO. The eval script's own exclusion logic only reproduces the *SFT*
pool from `config.json` and would not have caught a GRPO-pool collision --
this was checked corpus-wide instead, which covers any stage.

**Costs, measured not estimated** (`env/all/total_ob_tokens` /
`total_ac_tokens` per step x the three rates): GRPO **$9.95** for 30 steps
($0.332/step realized against a $0.283 step-0 rate, the difference being
mid-run length drift); after-eval + smoke **$4.07** (sample meter only);
probe ~$2.30. Total ~$16.32. Note the `--dry-run` cost estimator's "likely"
figure derives from `avg_completion` measured at **temperature 0** and so
runs low: it predicted $8.50 for 30 steps. The probe's own
`completion_p95 = max_tokens` for both categories was the early warning that
temperature-1.0 lengths differ.

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

- **Stop-token inclusion in sampled output (trap 3) — SUPERSEDED, now
  confirmed.** See "Readiness re-check" above: `sft_tinker_train_csv_eval.csv`
  is built from `tokenizer.decode(sequence.tokens)` on raw live-service
  samples with no stripping, and its completions demonstrably end
  `...\boxed{...}<|im_end|>`. The stop token *is* present; the stripping in
  `ReasonerEnv.step` is load-bearing. The original (now obsolete) reasoning
  is kept below for context. ~~It is an inference, not an
  observed fact.~~ No live `sample_async` call was made (that needs a
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
- **`Cfg.reward_clip = 25.0` is chosen from the *gold-trace* reward
  distribution**, not from any real policy rollout (none exist yet). It is a
  `Cfg` knob specifically so it can be revisited after the first smoke run's
  real data. `Cfg.max_tokens` **no longer** falls under this caveat: it was
  2048-ified from the SFT policy's own measured output lengths in the
  2026-07-29 audit (see "Audit fixes"), not from gold traces. The 8192 figure
  quoted elsewhere for enabling the other four categories *is* still
  gold-derived and should be re-checked against
  `_check_policy_length_distribution` before those are switched on.
- **Hyperparameters copied from `plan_grpo.txt` are starting points, not
  tuned values**: `learning_rate=1e-5`, `group_size=8`,
  `groups_per_batch=8`, `save_every=5`, `kl_penalty_coef=0.0`,
  `remove_constant_reward_groups=True`, `num_substeps=1`. None of these were
  swept or justified beyond "matches the plan's own reasoning" (RL learning
  rate an order of magnitude below SFT's, KL off initially to see the
  unconstrained reward signal first).
- ~~No sampling-meter price is on file~~ **SUPERSEDED**: sampling
  ($1.005/M) and training ($0.737/M) prices for `Qwen/Qwen3.5-4B` are now
  `Cfg` defaults and the estimate prints dollars on both meters. See the
  price table under "Readiness re-check" above. Re-verify at
  https://tinker-docs.thinkingmachines.ai/tinker/models/ before a large run;
  Tinker's prices have moved once already.
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

**SUPERSEDED -- the training loop has now been exercised end to end.** As of
2026-07-30 a full 30-step paid run completed (gravity + unit_conversion,
$9.95, see "The first GRPO training run" above), covering `--yes`,
`run_training`'s renderer resolution from checkpoint metadata,
`forward_backward`/`optim_step`, periodic + final checkpointing, and the
resulting checkpoint being sampled by the eval script. The original text is
kept below for the reasoning it records, which still holds for any *new*
category: the recommended first paid call on an untried category remains
`--probe-headroom`, not a training smoke run, because `--dry-run` already
covers the plumbing (including the env's own `step()`) while only the probe
answers whether training that category can accomplish anything. That ordering
was followed here and correctly redirected the run away from the default
`numeral`/`equation_numeric` pair. ~~`--probe-headroom` has
now been run once for real (cryptarithm, ~$0.49 -- see "First paid call on the
GRPO path" above), which exercised `sample_async`, `create_sampling_client_
async`, the `ReasonerEnv` round trip on live samples, and the verdict logic.
The training loop itself is still unexercised.~~ ~~There is also no
sampling-meter price on file~~ -- superseded, all three prices are `Cfg`
defaults; re-verify at
https://tinker-docs.thinkingmachines.ai/tinker/models/ before a large run.

## Evaluation: every measurement in this repo, step by step

This project has evaluation at five different layers, and they measure
genuinely different things -- a green light at one layer says nothing about
the next. In pipeline order: (1) can the deterministic generator solve the
task at all, (2) does the reward function score traces correctly, (3) did SFT
teach the output format, (4) can the trained policy actually answer real
questions, (5) did GRPO improve that. Most confusion in this repo's history
came from reading a layer-N result as evidence about layer N+1.

### What was trained, on what, with which framework

| | |
|---|---|
| base model | `Qwen/Qwen3.5-4B`, LoRA (rank in `full_0727/config.json`) |
| framework | Thinking Machines **Tinker** (`tinker` 0.23.4 + `tinker-cookbook` 0.5.2), hosted -- model runs server-side, this repo's process drives the loop |
| SFT data | **`synth_sft.jsonl`** -- 14,000 rows, 2000/category, **100% synthetic**, built by `scripts/gen_synthetic_data.py` from each `monitor_<task>.py`'s top-down problem constructor + the deterministic `reasoning_<task>` generator |
| SFT run | `sft_tinker_runs/full_0727/` -- 1376 train / 73 val rows, 4,043,134 train tokens, 22 optimizer steps, ~$2.98 |
| GRPO data | the same `synth_sft.jsonl`, **minus** every row `full_0727` could have trained on (`sft_holdout_keys`), parsed into examples via `scripts/eval_train_csv.PARSERS` |
| GRPO run | `grpo_tinker_runs/grpo_uc_gravity_0730/` -- gravity + unit_conversion, 30 steps, 960 rollouts, $9.95 |
| eval set | **`train.csv`** -- 9500 real competition rows, **never trained on at any stage** |

**No train/eval contamination, verified corpus-wide**: `synth_sft.jsonl`'s
14,000 distinct prompts and `train.csv`'s 9,500 share **zero** strings. This
matters because `eval_sft_tinker_train_csv.py`'s built-in exclusion only
reconstructs the *SFT* pool from `config.json` and would not have caught a
GRPO-pool collision. The corpus-wide check covers any stage.

The local/Kaggle alternatives (`train_sft_kaggle.py` on `transformers.Trainer`,
`train_grpo_cipher_kaggle.py` on TRL `GRPOTrainer`) exist and the SFT one is
verified end to end on bf16, but **neither produced the real runs** -- the
merged SFT model is 9.3 GB bf16 against a 5.67 GiB local GPU, so both real
stages ran on Tinker.

### Layer 1 -- deterministic generator coverage ($0)

`uv run python scripts/eval_train_csv.py` runs all seven `reasoning_<task>`
generators against all 9500 real rows and writes
`reasoners_train_csv_eval.csv`. **This measures whether a gold SFT trace can
be *sourced* for a row, not whether any model can solve it.**

Metrics: `solved` (generator returned non-`None`) and `correct` (its boxed
answer matches `train.csv`'s, via `compare_answer()`'s exact-or-1e-2-relative
rule, binary strings exact-only). For the six foolproof generators these are
equal by construction -- the contract is `return None` rather than emit an
unverified trace. **TOTAL 8336/9500 = 87.7%.** Per-category numbers and the
history behind each are in "How training data is actually generated".

### Layer 2 -- reward-function correctness ($0)

Four independent checks, because a reward function that crashes or misranks
poisons GRPO silently:

1. **`uv run python -m reasoners.monitor_<task>`** -- generate one synthetic
   problem, run the generator, score it, print the step-by-step reward log.
   Eyeball-level, one trace.
2. **Gold-trace non-negativity at scale** -- a few hundred synthetic gold
   traces per category, asserting none score negative. A gold trace is fully
   correct by the foolproof contract, so a negative score is a reward bug.
3. **Adversarial traces** -- fabricated claims, tag-spam, self-contradiction,
   dishonest verification labels, copying visible prompt text. Must score
   negative or flat.
4. **`scripts/validate_reward_cipher_partial.py`** -- the same three for
   `reward_cipher_partial.py`, plus full-oracle parity against
   `reward_cipher.py`.

`kaggle/train_grpo_tinker.py --dry-run` folds 2 and 3 into
`_check_gold_trace_rewards` and `_check_reward_discrimination`. The latter is
the one that matters: it proves the reward *separates* right from wrong
(numeral gold 22.05 vs corrupted-answer 7.05 vs bare-wrong -5.00), which
`_check_gold_trace_rewards` alone does not.

**`scripts/score_policy_completions.py` ($0) closes the last gap in this
layer**: every number above comes from a gold trace or a mutation of one. This
scores 250 *real policy completions* per category, already sitting in
`sft_tinker_train_csv_eval.csv`, through the exact `ReasonerEnv.step` reward
path. The reward separates right from wrong on real output in all seven
categories (15-20 point gap), `ReasonerEnv`'s `correct` metric agreed with the
independent eval on **1696/1696** rows, and the exception path fired once.

### Layer 3 -- did SFT teach the format? (training metrics)

Tinker writes one JSON per optimizer step to `<run>/metrics.jsonl`. The
load-bearing key is **`_loss_per_token`**, plus a per-category
`_loss_per_token/<category>` breakdown and `_min_logprob/<category>`.

`full_0727`, 22 steps:

| step | overall | equation_numeric | cryptarithm | numeral | gravity | unit_conversion |
|---|---|---|---|---|---|---|
| 0 | 0.334 | 1.492 | 0.577 | 0.542 | 0.132 | 0.162 |
| 11 | 0.056 | 0.078 | 0.056 | 0.019 | 0.011 | 0.009 |
| 21 | **0.016** | 0.046 | 0.029 | 0.008 | 0.002 | 0.001 |

Monotone decrease, overall and per category. **What this does NOT establish**
is that the model can produce the format unprompted -- loss-per-token is
teacher-forced. That needs generation, which is why the SFT script ends with a
sampling smoke test checking `</think>` present, `<think>` absent (the chat
template already opens it -- see [[step-tag-format-check-pitfalls]]), and
`\boxed{` in the tail *after* `</think>`. All 3 sampled completions came back
`closed_think=True final_boxed=True stop=stop`.

### Layer 4 -- can the policy answer real questions? (paid)

`uv run python scripts/eval_sft_tinker_train_csv.py --run-dir <run> --yes`
samples a checkpoint on real `train.csv` rows at **temperature 0**, 250 per
category (1750 rows) by default, and writes a self-documenting CSV plus a
provenance manifest. `--categories` narrows the set; `--limit 7 --yes` is the
~$0.05 smoke test; without `--yes` it previews for free.

It records **correctness** (`correct`, vs the ground-truth answer column) and
**six independent format signals**, which is what makes it useful for
diagnosing *why* a category fails:

| column | meaning |
|---|---|
| `closed_think` | emitted a closing `</think>` |
| `has_step_tags` | emitted `<step type=...>` at all |
| `n_step_tags` / `step_tag_counts` | how many, and of which types -- **the step-tag-emission metric** |
| `has_conclusion_tag` | reached a `conclusion` step |
| `unbalanced_step_tags` | open/close counts disagree (the truncation signature) |
| `unknown_step_tags` | tags outside the six-tag vocabulary |
| `reopened_think` | re-opened `<think>` after closing (never observed: 0% everywhere) |
| `stop_reason` / `n_tokens` | natural stop vs hitting the cap, and **completion length** |

**Baseline, `full_0727` SFT checkpoint, 250 rows/category, temperature 0:**

| category | acc | closed | boxed | unbal | concl | tok p50 | tok p95 | avg tags | stop=length |
|---|---|---|---|---|---|---|---|---|---|
| numeral | 90.0% | 100% | 100% | 0% | 100% | 326 | 457 | 16.1 | 0% |
| unit_conversion | 63.2% | 91% | 91% | 0% | 91% | 5473 | 8405 | 21.6 | 9% |
| cipher | 53.2% | 81% | 81% | 48% | 74% | 5403 | 8394 | 211.0 | 19% |
| gravity | 45.2% | 91% | 91% | 0% | 91% | 4302 | 8349 | 25.0 | 9% |
| equation_numeric | 9.6% | 87% | 87% | 38% | 83% | 491 | 8413 | 21.9 | 13% |
| cryptarithm | 0.8% | 96% | 95% | 7% | 94% | 1656 | 2359 | 75.4 | 4% |
| bit_manipulation | 0.4% | 62% | 62% | 39% | 61% | 7411 | 8291 | 8.1 | 38% |

Read the columns together, not the accuracy alone: `bit_manipulation` is 38%
length-stopped with 39% unbalanced tags, so much of its 0.4% is *never
finishing*, a different failure from `cryptarithm`, which finishes cleanly
(96% closed, 4% length-stopped) and is simply wrong. Only two unknown step
tags appeared across all 1750 rows (`cipher:match`, `cipher:answer`, 1 each),
so the six-tag vocabulary held up. `cipher`'s 211 average tags reflects its
per-letter granularity -- the same shape flagged under "Two kinds of reasoner
module".

### Layer 5 -- did GRPO improve it?

**Pre-flight, $0** -- seven `--dry-run` checks (`_check_sft_holdout`,
`_check_gold_trace_rewards`, `_check_reward_discrimination`,
`_check_prompt_token_identity`, `_check_trace_length_percentiles`,
`_check_policy_length_distribution`, `_check_env_step_closed_loop`). Detailed
under "The GRPO training script".

**Headroom probe, paid** -- `--probe-headroom`. The gate that decides whether
a category is trainable at all: GRPO's gradient comes only from reward
variance *inside* a group, so a category the policy gets 8/8 right or 8/8
wrong contributes nothing. Reports `mixed` / `all_correct` / `all_wrong` /
`constant_reward` / `median_reward_spread` per category, verdict at 20% mixed.
Measured: cryptarithm 10% (don't train), unit_conversion 95% and gravity 75%
(train). Use >=20 prompts/category -- at 12 the gate falls between 2 and 3
groups.

**Per-step training metrics** -- `<run>/metrics.jsonl`, keys under
`env/all/*` and `env/<category>/*`: `correct`, `format`, `truncated`,
`reopened_think`, `reward/total`, `reward_raw`, `reward_exception`,
**`completion_tokens`**, **`n_step_tags`**, `by_group/frac_mixed`,
`frac_all_good`, `frac_all_bad`, and exact `total_ob_tokens` /
`total_ac_tokens` counters (use these for cost, not means). The last two
env metrics exist specifically to detect reward hacking, since
`_check_reward_discrimination` measured padding raising the reward on 60/60
rows.

**Never read a single step as a result.** See
[[grpo-per-step-metrics-are-noisy]]: a step is `groups_per_batch` distinct
prompts (4) x `group_size` correlated rollouts, so per-step `correct` swings
wildly (observed 0.281 / 0.312 / 0.656 / 0.344 / 0.625 / 0.719 / 0.875 in one
healthy run). Use **`scripts/analyze_grpo_rollouts.py <run> <block>`**, which
pools blocks of iterations from the per-iteration
`train_rollout_summaries.jsonl` files (per-rollout `correct`, `format`,
`truncated`, `completion_tokens`, `n_step_tags` + category tag) and
bootstraps over **groups**.

**Paired before/after on real rows** --
`scripts/compare_before_after_eval.py BEFORE.csv AFTER.csv [...]`. Because
both eval runs select from the same seeded pool, every after-row has a
same-prompt before-row, so the correct test is **McNemar's exact on the
discordant pairs** plus a bootstrap CI, not two independent proportions.
Evaluating a GRPO checkpoint needs a shim run dir (`grpo_eval_shim/`: a copy
of the SFT `config.json` + a one-record `checkpoints.jsonl` pointing at the
GRPO `sampler_path`) -- reusing the SFT config is what keeps row selection
identical and the comparison honest.

**Result after 30 GRPO steps on gravity + unit_conversion** (full tables and
caveats under "The first GRPO training run"):

| category | trained? | n | acc before | acc after | delta | McNemar p |
|---|---|---|---|---|---|---|
| gravity | yes | 250 | 45.2% | 58.4% | **+13.2pp** | <0.0001 |
| unit_conversion | yes | 250 | 63.2% | 76.4% | **+13.2pp** | 0.0001 |
| numeral | no | 100 | 89.0% | **76.0%** | **-13.0pp** | 0.0044 |
| equation_numeric | no | 100 | 8.0% | 10.0% | +2.0pp | 0.73 |
| cipher | no | 100 | 54.0% | 53.0% | -1.0pp | 1.0 |
| cryptarithm | no | 100 | 2.0% | 0.0% | -2.0pp | 0.50 |
| bit_manipulation | no | 100 | 0.0% | 0.0% | 0.0pp | 1.0 |

The untrained categories are scored on **100 rows, a strict prefix of the same
seeded 250** the Layer-4 baseline table uses, so their "before" figures differ
slightly from it by subsampling alone (numeral 89.0% here vs 90.0% at n=250,
cipher 54.0% vs 53.2%, cryptarithm 2.0% vs 0.8%). Every row is still paired
against its own before-row, so the deltas are unaffected.

**What GRPO did after SFT, in the format metrics** (untrained categories at
100 rows, trained at 250):

| category | closed/boxed | unbalanced | avg tags | tok p50 | tok p95 | stop=length |
|---|---|---|---|---|---|---|
| unit_conversion | 91% -> **98%** | 0% -> 0% | 21.6 -> 22.0 | 5473 -> 5322 | 8405 -> **7535** | 9% -> **2%** |
| gravity | 91% -> **95%** | 0% -> 1% | 25.0 -> 25.9 | 4302 -> 4277 | 8349 -> 8345 | 9% -> **5%** |
| equation_numeric | 87% -> **100%** | 38% -> **0%** | 21.9 -> 13.4 | 491 -> 378 | 8413 -> **502** | 13% -> **0%** |
| cryptarithm | 96% -> 98% | 7% -> 8% | 75.4 -> 67.9 | 1656 -> 1590 | 2359 -> 2077 | 4% -> 2% |
| numeral | 100% -> 99% | 0% -> 1% | 16.1 -> **18.6** | 326 -> 353 | 457 -> **609** | 0% -> 1% |
| cipher | 81% -> 78% | 48% -> 50% | 211 -> 224 | 5403 -> 5392 | 8394 -> 8400 | 19% -> 22% |
| bit_manipulation | 62% -> 60% | 39% -> 47% | 8.1 -> 5.9 | 7411 -> 7566 | 8291 -> 8291 | 38% -> 40% |

Three things this table says that the accuracy table does not:

1. **GRPO's clearest effect was teaching the policy to stop.** Length-stopping
   collapsed on the trained pair (9%->2%, 9%->5%) and on *untrained*
   `equation_numeric` (13%->0%, p95 8413->502, unbalanced tags 38%->0%, format
   to a perfect 100%). Part of the trained-pair accuracy gain is exactly this:
   8/44 and 12/51 of the wrong->right flips were rows the before-policy had
   truncated. **But note equation_numeric's format windfall bought nothing
   measurable** -- its accuracy moved +2.0pp at p=0.73. This is the same
   global length/stopping shift that *hurt* numeral; it is not a separate
   positive spillover, and only the numeral effect is statistically
   established.
2. **The numeral regression is visible here as length**, in the opposite
   direction: p95 457 -> 609, avg tags 16.1 -> 18.6. See
   [[grpo-cross-category-regression]].
3. **Format and accuracy move independently.** `cryptarithm` improved format
   and went 2% -> 0% accuracy; `equation_numeric` hit perfect format at 10%
   accuracy. Never use format as an accuracy proxy.

### Cost limitation -- what the evaluation budget could and could not buy

Every layer-4 and layer-5 number is **metered**: Tinker bills prompt/prefill
$0.33/M, generated $1.005/M, training $0.737/M for `Qwen/Qwen3.5-4B`. GRPO
bills *both* sampling and training meters, and `remove_constant_reward_groups`
measured **0%** firing on a real probe, so it discounts nothing. This
constrained the design in ways worth stating plainly, because several numbers
above are smaller-sample than they look:

- **The whole GRPO run was 30 steps / 960 rollouts / $9.95**, against a full
  epoch of 925 steps (~$272). That is **120 distinct prompts** (60/category).
  A null result at this size would have been uninformative; the observed
  effect was large enough to clear it anyway, but per-category CIs are wide
  ([+7.6, +18.8] and [+7.2, +19.6]).
- **The SFT-then-RL detour was cut for budget** ($9-26 on its own). GRPO was
  run directly on the `full_0727` checkpoint instead, which is why
  low-accuracy categories (cryptarithm 0.8%, bit_manipulation 0.4%) were
  never trainable in this cycle -- the probe says so, and lifting them needs
  SFT first.
- **The regression check was deliberately 100 rows/category, not 250** --
  **$1.52 measured** for its 500 rows (vs $2.53 for the trained pair's 500,
  which carry much longer traces; ~$3.79 is the *scaled* estimate for a
  250-row version, and scaling is only roughly valid here since per-category
  completion length spans ~20x, numeral ~350 tokens vs bit_manipulation
  ~7,500). 100 rows is enough to detect numeral's -13pp at p=0.0044 but not
  enough to resolve a small regression; cipher's -1.0pp and
  equation_numeric's +2.0pp are indistinguishable from zero at this n and
  should not be read as real.
- **The before-eval was reused, not re-paid.** `sft_tinker_train_csv_eval.csv`
  already existed at 250 rows/category, and the `--categories` flag was
  written specifically so the after-eval could sample a comparable subset
  rather than repurchase all 1750 rows.
- **Total for the whole GRPO cycle: ~$16.32** -- probe ~$2.30, training $9.95,
  after-eval + smoke $4.07 (measured from `total_ob_tokens`/`total_ac_tokens`
  and the eval CSVs' own token columns, not estimated).
- **The `--dry-run` cost estimator runs low.** Its "likely" figure derives
  from `avg_completion` measured at temperature 0; it predicted $8.50 for the
  30 steps that actually cost $9.95. The probe's `completion_p95 == max_tokens`
  for both categories was the early warning that temperature-1.0 lengths
  differ.

The general rule this cycle established: **spend on the measurement that
decides whether to spend.** The $2.30 probe redirected the run away from the
default `numeral`/`equation_numeric` pair (90% and 9.6% accuracy -- nearly
saturated and nearly floored, the two worst headroom picks available), and the
$1.68 regression slice was the only thing that caught a 13-point loss on a
category nobody was training.
