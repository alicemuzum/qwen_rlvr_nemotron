"""Score the SFT policy's REAL completions through the exact ReasonerEnv reward path.

Every reward measurement in `kaggle/train_grpo_tinker.py --dry-run` feeds the reward a
*gold* trace or a mutation of one. That proves the reward function *can* discriminate;
it does not prove it discriminates on the text the policy actually emits, which is what
decides whether GRPO has a usable signal. `sft_tinker_train_csv_eval.csv` already holds
250 real completions per category from the `full_0727` checkpoint (temperature 0), so
this measurement is $0 and needs no network call.

Reports per category:
  - `exc`: how often the reward function raises (the try/except -> -reward_clip path in
    ReasonerEnv.step, trap 5). A high rate would mean the run is scoring noise.
  - reward for really-correct vs really-wrong completions. These must separate.
  - `agree`: whether ReasonerEnv's own `correct` metric reproduces the independent
    eval's verdict. This is the metric the whole run is judged on, so a disagreement
    is a bug in the metric, not a curiosity.

Run: uv run python scripts/score_policy_completions.py
"""

import csv
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "kaggle"), str(REPO / "scripts")]
csv.field_size_limit(10_000_000)

import eval_train_csv as eval_csv
from train_grpo_tinker import (
    REWARDS,
    Cfg,
    _answer_is_gradable,
    _boxed_matches,
    parse_row_examples,
)

# ReasonerEnv builds its reward text as think_prefix + decode(action). The eval CSV
# stores the decoded completion but keeps the literal stop-token *text*, where the env
# strips the stop-token *id* before decoding -- strip it here so the two agree.
THINK_PREFIX = "<think>\n"
STOP_TEXT = "<|im_end|>"


def summarize(xs: list[float]) -> str:
    if not xs:
        return "n/a"
    return f"p50={statistics.median(xs):7.2f} mean={statistics.fmean(xs):7.2f}"


def main() -> None:
    cfg = Cfg()
    eval_path = REPO / cfg.policy_eval_csv
    with eval_path.open() as fh:
        rows = list(csv.DictReader(fh))

    by_cat: dict[str, list[dict]] = {}
    for row in rows:
        by_cat.setdefault(row["category"], []).append(row)

    header = (
        f"{'category':18s} {'n':>4s} {'skip':>5s} {'exc':>4s} {'fmt':>5s}  "
        f"{'reward | really correct':>26s}  {'reward | really wrong':>26s}  {'agree':>9s}"
    )
    print(header)
    print("-" * len(header))
    for cat in sorted(by_cat):
        if cat not in REWARDS:
            print(f"{cat:18s} -- no reward function registered, skipping")
            continue
        reward_fn = REWARDS[cat]
        rw_correct: list[float] = []
        rw_wrong: list[float] = []
        n_exc = n_skip = n_scored = n_fmt = n_agree = 0

        for row in by_cat[cat]:
            answer = row["ground_truth"]
            if not _answer_is_gradable(answer):
                n_skip += 1
                continue
            examples, _question = parse_row_examples(cat, row["prompt"])
            if examples is None:
                n_skip += 1
                continue

            completion = row["completion"].removesuffix(STOP_TEXT)
            try:
                raw, _logs = reward_fn(THINK_PREFIX + completion, examples, answer)
            except Exception:  # noqa: BLE001 -- mirrors ReasonerEnv.step's trap-5 guard
                raw = -cfg.reward_clip
                n_exc += 1
            n_scored += 1

            # Recomputed exactly as ReasonerEnv.step does, from the same text.
            _head, sep, tail = completion.rpartition("</think>")
            boxed = eval_csv.last_boxed(tail) if sep else None
            clean_stop = row["stop_reason"] == "stop"
            fmt_ok = (
                sep != ""
                and "<think>" not in completion
                and "<step type=" in completion
                and boxed is not None
                and clean_stop
            )
            n_fmt += fmt_ok

            our_correct = _boxed_matches(boxed, answer, cat)
            eval_correct = row["correct"] in ("True", "true", "1")
            n_agree += our_correct == eval_correct
            (rw_correct if eval_correct else rw_wrong).append(raw)

        fmt_rate = f"{100 * n_fmt / n_scored:4.0f}%" if n_scored else "  n/a"
        print(
            f"{cat:18s} {n_scored:4d} {n_skip:5d} {n_exc:4d} {fmt_rate:>5s}  "
            f"{summarize(rw_correct):>26s}  {summarize(rw_wrong):>26s}  "
            f"{n_agree}/{n_scored}"
        )

    print(
        "\n'skip' = rows dropped before scoring: a brace in the answer (unwinnable, see\n"
        "_answer_is_gradable) or a prompt the category parser could not read.\n"
        "'agree' must be n/n -- it is ReasonerEnv's `correct` metric checked against an\n"
        "independent eval of the same completions."
    )


if __name__ == "__main__":
    main()
