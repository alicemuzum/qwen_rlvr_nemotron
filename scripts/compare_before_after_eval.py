"""Paired before/after comparison of two eval CSVs on identical prompts.

The before and after runs select rows from the same seeded pool (see
eval_sft_tinker_train_csv.select_rows), so every after-row has a before-row
with the same prompt. That makes this a PAIRED comparison: the right test is
McNemar's on the discordant pairs, not two independent proportions -- pairing
removes between-problem difficulty variance, which is the dominant noise term
when n is a few hundred.

Usage:
    uv run python scripts/compare_before_after_eval.py BEFORE.csv AFTER.csv [AFTER2.csv ...]
"""
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


def load(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def binom_two_sided(b, c):
    """Exact McNemar: P(|X - n/2| >= |b - n/2|) for X ~ Bin(n, 0.5), n = b + c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def boot_ci(pairs, n=10000, seed=0):
    """Bootstrap the paired accuracy difference (after - before)."""
    rng = random.Random(seed)
    if not pairs:
        return float("nan"), float("nan")
    diffs = []
    for _ in range(n):
        s = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        diffs.append(sum(a for _, a in s) / len(s) - sum(b for b, _ in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def main(argv):
    before_rows = load(argv[0])
    after_rows = []
    for p in argv[1:]:
        after_rows += load(p)

    before = {r["prompt"]: r for r in before_rows}
    by_cat = defaultdict(list)
    unmatched = 0
    for r in after_rows:
        b = before.get(r["prompt"])
        if b is None:
            unmatched += 1
            continue
        by_cat[r["category"]].append((b, r))

    if unmatched:
        print(f"WARNING: {unmatched} after-rows had no before-row with the same prompt "
              f"-- those are unpaired and excluded")

    hdr = (f"{'category':<18}{'n':>5}{'before':>8}{'after':>8}{'delta':>8}"
           f"{'95% CI':>18}{'w->r':>6}{'r->w':>6}{'McNemar p':>11}")
    print(hdr)
    print("-" * len(hdr))

    tot = defaultdict(int)
    all_pairs = []
    for cat in sorted(by_cat):
        pairs = [(b["correct"] == "True", a["correct"] == "True") for b, a in by_cat[cat]]
        all_pairs += pairs
        n = len(pairs)
        bacc = sum(x for x, _ in pairs) / n
        aacc = sum(y for _, y in pairs) / n
        w2r = sum(1 for x, y in pairs if not x and y)
        r2w = sum(1 for x, y in pairs if x and not y)
        lo, hi = boot_ci(pairs)
        p = binom_two_sided(r2w, w2r)
        star = " *" if p < 0.05 else ""
        print(f"{cat:<18}{n:>5}{bacc:>8.1%}{aacc:>8.1%}{aacc - bacc:>+8.1%}"
              f"{f'[{lo:+.1%}, {hi:+.1%}]':>18}{w2r:>6}{r2w:>6}{p:>11.4f}{star}")
        tot["n"] += n

    print("-" * len(hdr))
    n = len(all_pairs)
    bacc = sum(x for x, _ in all_pairs) / n
    aacc = sum(y for _, y in all_pairs) / n
    w2r = sum(1 for x, y in all_pairs if not x and y)
    r2w = sum(1 for x, y in all_pairs if x and not y)
    lo, hi = boot_ci(all_pairs)
    print(f"{'POOLED':<18}{n:>5}{bacc:>8.1%}{aacc:>8.1%}{aacc - bacc:>+8.1%}"
          f"{f'[{lo:+.1%}, {hi:+.1%}]':>18}{w2r:>6}{r2w:>6}"
          f"{binom_two_sided(r2w, w2r):>11.4f}")
    print("\nw->r = wrong before, right after.  r->w = the reverse (regressions).")
    print("McNemar p is exact, two-sided, on the discordant pairs only.")

    # format/process side-by-side, same pairing
    print(f"\n{'category':<18}{'fmt_before':>11}{'fmt_after':>10}{'tags_before':>12}{'tags_after':>11}"
          f"{'tok_before':>11}{'tok_after':>10}")
    print("-" * 83)
    for cat in sorted(by_cat):
        ps = by_cat[cat]

        def m(rows, key, cast=float):
            vals = [cast(r[key]) for r in rows if r.get(key) not in (None, "")]
            return sum(vals) / len(vals) if vals else float("nan")

        bs = [b for b, _ in ps]
        as_ = [a for _, a in ps]
        fmt_b = sum(b["closed_think"] == "True" and b["final_boxed"] == "True" for b in bs) / len(bs)
        fmt_a = sum(a["closed_think"] == "True" and a["final_boxed"] == "True" for a in as_) / len(as_)
        print(f"{cat:<18}{fmt_b:>11.1%}{fmt_a:>10.1%}"
              f"{m(bs, 'n_step_tags'):>12.1f}{m(as_, 'n_step_tags'):>11.1f}"
              f"{m(bs, 'n_tokens'):>11.0f}{m(as_, 'n_tokens'):>10.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
