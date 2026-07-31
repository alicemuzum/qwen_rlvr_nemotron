"""Pool a GRPO run's per-rollout summaries into an early-vs-late comparison.

Why not read metrics.jsonl: each step is only groups_per_batch prompts (4 here,
2 per category), so per-step `correct` is dominated by which problems were
drawn. Rollouts inside a group share a prompt, so the independent unit is the
GROUP, not the rollout -- every interval here is bootstrapped over groups for
that reason.

Reports per category: rollout-level accuracy, pass@8 (any rollout in the group
correct), and the group-mixing fraction that carries GRPO's whole gradient.
"""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RUN = Path(sys.argv[1])
BLOCK = int(sys.argv[2]) if len(sys.argv) > 2 else 10

groups = []  # one entry per (iteration, group): category + its rollouts
for path in sorted(RUN.glob("iteration_*/train_rollout_summaries.jsonl")):
    per = defaultdict(list)
    for line in path.open():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        m = r["steps"][0].get("metrics", {}) if r.get("steps") else {}
        per[(r["iteration"], r["group_idx"])].append(
            {
                "cat": (r.get("tags") or ["?"])[0],
                "correct": m.get("correct"),
                "fmt": m.get("format"),
                "trunc": m.get("truncated"),
                "toks": m.get("completion_tokens"),
                "tags": m.get("n_step_tags"),
                "raw": m.get("reward_raw"),
                "total": r.get("total_reward"),
            }
        )
    for (it, gi), rolls in per.items():
        groups.append({"it": it, "gi": gi, "cat": rolls[0]["cat"], "rolls": rolls})

iters = sorted({g["it"] for g in groups})
if not iters:
    raise SystemExit(f"no rollout summaries under {RUN}")
early_its = set(iters[:BLOCK])
late_its = set(iters[-BLOCK:])


def agg(gs, key):
    vals = [r[key] for g in gs for r in g["rolls"] if r[key] is not None]
    return statistics.mean(vals) if vals else float("nan")


def pass_at_k(gs):
    hits = [any(r["correct"] for r in g["rolls"]) for g in gs if g["rolls"]]
    return statistics.mean(hits) if hits else float("nan")


def mixed(gs):
    out = []
    for g in gs:
        c = [r["correct"] for r in g["rolls"] if r["correct"] is not None]
        if c:
            out.append(0 < sum(c) < len(c))
    return statistics.mean(out) if out else float("nan")


def boot_diff(a, b, key, n=4000, seed=0):
    """Bootstrap the late-minus-early difference, resampling GROUPS."""
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(agg(rb, key) - agg(ra, key))
    diffs.sort()
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


cats = sorted({g["cat"] for g in groups})
print(f"run={RUN.name}  iterations={len(iters)} ({iters[0]}..{iters[-1]})  "
      f"groups={len(groups)}  rollouts={sum(len(g['rolls']) for g in groups)}")
print(f"comparing first {len(early_its)} vs last {len(late_its)} iterations\n")

hdr = (f"{'category':<17}{'n_grp':>6}{'acc_early':>10}{'acc_late':>9}{'delta':>8}"
       f"{'95% CI':>18}{'p@8_e':>7}{'p@8_l':>7}{'mixed_l':>8}")
print(hdr)
print("-" * len(hdr))
for cat in cats + ["ALL"]:
    sel = [g for g in groups if cat == "ALL" or g["cat"] == cat]
    e = [g for g in sel if g["it"] in early_its]
    ll = [g for g in sel if g["it"] in late_its]
    if not e or not ll:
        continue
    ae, al = agg(e, "correct"), agg(ll, "correct")
    lo, hi = boot_diff(e, ll, "correct")
    print(f"{cat:<17}{len(e) + len(ll):>6}{ae:>10.3f}{al:>9.3f}{al - ae:>+8.3f}"
          f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}{pass_at_k(e):>7.2f}{pass_at_k(ll):>7.2f}"
          f"{mixed(ll):>8.2f}")

print("\n--- process / cost drift (rollout means) ---")
h2 = f"{'category':<17}{'fmt_e':>7}{'fmt_l':>7}{'tags_e':>8}{'tags_l':>8}{'tok_e':>8}{'tok_l':>8}{'trunc_e':>9}{'trunc_l':>9}"
print(h2)
print("-" * len(h2))
for cat in cats + ["ALL"]:
    sel = [g for g in groups if cat == "ALL" or g["cat"] == cat]
    e = [g for g in sel if g["it"] in early_its]
    ll = [g for g in sel if g["it"] in late_its]
    if not e or not ll:
        continue
    print(f"{cat:<17}{agg(e, 'fmt'):>7.2f}{agg(ll, 'fmt'):>7.2f}"
          f"{agg(e, 'tags'):>8.1f}{agg(ll, 'tags'):>8.1f}"
          f"{agg(e, 'toks'):>8.0f}{agg(ll, 'toks'):>8.0f}"
          f"{agg(e, 'trunc'):>9.2f}{agg(ll, 'trunc'):>9.2f}")

print("\n--- per-iteration accuracy (2 prompts/category/iter -- noisy by design) ---")
for cat in cats:
    row = []
    for it in iters:
        gs = [g for g in groups if g["cat"] == cat and g["it"] == it]
        row.append(f"{agg(gs, 'correct'):.2f}" if gs else " -- ")
    print(f"{cat:<17}" + " ".join(row))
