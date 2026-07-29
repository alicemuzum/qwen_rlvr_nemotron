"""Evaluate a Tinker-trained SFT checkpoint against real train.csv rows.

Samples --per-category (default 250) rows per category from train.csv --
1750 rows total across the 7 categories -- renders each with the exact
prompt template train_sft_kaggle.render_prompt used during training, samples
the trained checkpoint on Tinker, and checks both format (closed <think>,
step tags present, a trailing \\boxed{}, no re-opened <think> -- the same
four checks scripts/infer_sft_tinker.py's format_checks() runs) and answer
correctness against train.csv's ground-truth answer column.

Writes sft_tinker_train_csv_eval.csv (repo root) with columns:
sample_id, category, prompt, completion, model_answer, ground_truth,
correct, closed_think, has_step_tags, final_boxed, reopened_think,
stop_reason, n_tokens, prompt_tokens, max_tokens, checkpoint, model_name,
temperature, timestamp

Every row carries its own checkpoint/model/temperature/timestamp rather than
leaving them implicit, so the CSV is self-documenting even pulled out of
this repo (e.g. for a report) -- no need to cross-reference which run/config
produced it. A companion sft_tinker_train_csv_eval.manifest.json (same
directory) records the run-level provenance once: run_dir, checkpoint,
model_name, sampling params, git commit, per-category selected/eligible/
excluded-from-training counts, and start/end timestamps.

--- Never eval on a row the run trained/validated on ---

train_sft_tinker.py's run never touches train.csv directly (its data_file is
synth_sft.jsonl, entirely synthetic per CLAUDE.md), so in principle no
train.csv row could have been trained on. This script still verifies that
empirically rather than assuming it: training_prompts() reproduces
train_sft_kaggle.apply_category_caps() bit-for-bit (same seed, same
category_caps read from the run's own config.json) to recover the exact set
of synth_sft.jsonl rows the run's data prep touched (train + val both --
build_examples() only *further* trims this set by sequence length, so this
is a safe superset, never a smaller set than what was literally trained on).
Any train.csv row whose prompt string exactly matches one of those is
dropped from the eligible pool before sampling.

--- This calls a paid API ---

Tinker's sampling meter bills per generated token. Actual spend is roughly
bounded by how many tokens the model naturally emits before stopping (the
full_0727 smoke test came back stop_reason="stop" well short of the
max_tokens ceiling), not by the per-row cap this script requests, but there
is no way to predict an exact dollar figure here. Run a cheap --limit smoke
test first:

    uv run python scripts/eval_sft_tinker_train_csv.py --limit 7 --yes

(one row per category, mirroring train_sft_tinker.py's own $0.08 rehearsal
pattern) before committing to the full 1750-row run. Real spend requires
--yes; without it the script only builds the eligible pool, prints the
per-category selection + exclusion counts, and exits -- no network call.

Usage:
    uv run python scripts/eval_sft_tinker_train_csv.py                    # preview only, no spend
    uv run python scripts/eval_sft_tinker_train_csv.py --limit 7 --yes    # cheap smoke test
    uv run python scripts/eval_sft_tinker_train_csv.py --yes              # full 1750-row run
    uv run python scripts/eval_sft_tinker_train_csv.py --yes --resume     # continue an interrupted run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "kaggle"))
import train_sft_kaggle as local  # noqa: E402  (imported after the sys.path line above)

DEFAULT_RUN_DIR = "sft_tinker_runs/full_0727"
CATEGORIES = [
    "cipher",
    "cryptarithm",
    "gravity",
    "numeral",
    "unit_conversion",
    "equation_numeric",
    "bit_manipulation",
]
_TOLERANT = {"gravity", "unit_conversion"}  # 1e-2 numeric tolerance, see CLAUDE.md
_MAX_ATTEMPTS = 3


# --------------------------------------------------------------------------
# Run artifacts
# --------------------------------------------------------------------------


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run(run_dir: Path) -> dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"{cfg_path} not found -- is {run_dir} a train_sft_tinker.py run dir?")
    return json.loads(cfg_path.read_text())


def sampler_path_for(run_dir: Path, name: str) -> str:
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- did the run save a checkpoint?")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for record in records:
        if record["name"] == name:
            return record["sampler_path"]
    available = ", ".join(r["name"] for r in records) or "(none)"
    raise KeyError(f"no checkpoint named {name!r} in {path}; available: {available}")


# --------------------------------------------------------------------------
# Row selection: real train.csv rows, minus anything the run trained on
# --------------------------------------------------------------------------


def training_prompts(cfg: dict) -> set[str]:
    """Every raw prompt string the run's data prep could have touched."""
    data_path = REPO / cfg["data_file"]
    rows = local.load_rows(data_path)
    capped = local.apply_category_caps(rows, cfg.get("category_caps", {}), cfg["seed"])
    return {r["prompt"] for r in capped}


def load_eligible_pool(exclude: set[str]) -> tuple[dict[str, list[dict]], dict[str, int]]:
    pool: dict[str, list[dict]] = defaultdict(list)
    excluded_by_cat: dict[str, int] = defaultdict(int)
    with (REPO / "train.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] not in CATEGORIES:
                continue
            if row["prompt"] in exclude:
                excluded_by_cat[row["category"]] += 1
                continue
            pool[row["category"]].append(row)
    total_excluded = sum(excluded_by_cat.values())
    if total_excluded:
        print(f"excluded {total_excluded} train.csv row(s) whose prompt matched a trained-on synth_sft.jsonl row")
    return pool, dict(excluded_by_cat)


def select_rows(pool: dict[str, list[dict]], per_category: int, seed: int) -> list[dict]:
    """Round-robin across categories so --limit N is a spread smoke test, not one category."""
    rng = random.Random(seed)
    shuffled = {}
    for cat in CATEGORIES:
        rows = list(pool.get(cat, []))
        rng.shuffle(rows)
        if len(rows) < per_category:
            print(f"WARNING: only {len(rows)} eligible rows for {cat!r} (wanted {per_category}); taking all")
        shuffled[cat] = rows[:per_category]

    selected: list[dict] = []
    for i in range(per_category):
        for cat in CATEGORIES:
            if i < len(shuffled[cat]):
                selected.append(shuffled[cat][i])
    return selected


# --------------------------------------------------------------------------
# Format checks + answer extraction (same discipline as infer_sft_tinker.py)
# --------------------------------------------------------------------------


def extract_boxed(completion: str) -> str | None:
    """Structural extraction, not brace-balancing (cryptarithm answers can
    contain literal braces -- see CLAUDE.md's "Gotcha" note). Only looks
    after the completion's own </think>, and only counts a boxed marker that
    appears there -- the system prompt itself mentions \\boxed{} well before
    any real answer, so an unguarded substring search would false-positive.

    Takes the *last* "}" in the tail as the closing brace, not just the
    tail's final character: tokenizer.decode() doesn't strip special tokens,
    so a real completion ends "...\\boxed{answer}<|im_end|>" -- and
    "<|im_end|>" contains no braces, so this stays correct even for
    cryptarithm answers that legitimately contain literal "}" characters.
    """
    _, sep, tail = completion.rpartition("</think>")
    if not sep:
        return None
    marker = "\\boxed{"
    idx = tail.rfind(marker)
    if idx == -1:
        return None
    start = idx + len(marker)
    end = tail.rfind("}")
    if end <= start:
        return None
    return tail[start:end]


def format_checks(completion: str) -> dict[str, bool]:
    _, sep, tail = completion.rpartition("</think>")
    return {
        "closed_think": "</think>" in completion,
        "has_step_tags": "<step type=" in completion,
        "final_boxed": bool(sep) and "\\boxed{" in tail,
        "reopened_think": "<think>" in completion,
    }


# Shared six-tag vocabulary (CLAUDE.md's "Two kinds of reasoner module"). Cipher
# is the one generator that also emits these two -- reward_cipher.py has no
# handler for either, so seeing them from the *model* (not just gold traces)
# is expected for cipher and worth flagging as genuinely novel for anything else.
_KNOWN_STEP_TAGS = {"plan", "analysis", "state_update", "execution", "verification", "conclusion"}
_CIPHER_EXTRA_TAGS = {"deduction", "input_parsing"}

_STEP_OPEN_RE = re.compile(r'<step type="([^"]*)">')
# type group is [^"]* (not .*?): an unrestricted .*? type group -- what
# reward_<task>.py's own extraction regex uses -- will, on malformed/
# degenerate output where a step's *content* happens to contain a literal
# '">' sequence, skip past the real tag boundary and swallow multiple tags
# into one bogus "type" capture. A tag type is always a bare identifier, so
# restricting to [^"]* is strictly safer with no downside on well-formed tags.
_STEP_PAIR_RE = re.compile(r'<step type="([^"]*)">(.*?)</step>', re.DOTALL)


def step_tag_report(category: str, completion: str) -> dict:
    """Same tag extraction the reward_<task>.py functions use
    (re.findall(r'<step type="(.*?)">(.*?)</step>', ...)), applied here for
    diagnostics rather than scoring. Distinguishes *opened* tags (every
    `<step type="...">`, including ones truncated by hitting the token cap
    before their `</step>`) from *closed* pairs, so a truncated trace shows
    up as unbalanced rather than silently undercounting.
    """
    opens = _STEP_OPEN_RE.findall(completion)
    pairs = _STEP_PAIR_RE.findall(completion)
    n_closes = completion.count("</step>")
    counts = Counter(t for t, _ in pairs)
    allowed = _KNOWN_STEP_TAGS | (_CIPHER_EXTRA_TAGS if category == "cipher" else set())
    unknown = sorted(set(counts) - allowed)
    return {
        "step_tag_counts": json.dumps(dict(sorted(counts.items()))),
        "n_step_tags": len(pairs),
        "has_conclusion_tag": "conclusion" in counts,
        "unknown_step_tags": ",".join(unknown),
        "unbalanced_step_tags": len(opens) != n_closes,
    }


def is_correct(category: str, model_answer: str | None, ground_truth: str) -> bool:
    if model_answer is None:
        return False
    if model_answer == ground_truth:
        return True
    if category in _TOLERANT:
        try:
            return abs(float(model_answer) - float(ground_truth)) <= 1e-2
        except ValueError:
            return False
    return False


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


async def sample_one(sampling_client, tokenizer, prompt_ids, max_tokens, temperature, sem):
    import tinker
    from tinker import types

    async with sem:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                result = await sampling_client.sample_async(
                    prompt=tinker.ModelInput(chunks=[types.EncodedTextChunk(tokens=prompt_ids)]),
                    num_samples=1,
                    sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=temperature),
                )
                break
            except Exception:
                if attempt == _MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(2 * attempt)
    sequence = result.sequences[0]
    return tokenizer.decode(sequence.tokens), sequence.stop_reason, len(sequence.tokens)


async def run_eval(
    rows: list[dict],
    cfg: dict,
    run_dir: Path,
    checkpoint: str,
    temperature: float,
    concurrency: int,
    out_path: Path,
    resume: bool,
) -> None:
    import tinker
    import train_sft_tinker

    train_sft_tinker._resolve_api_key()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    local.CFG.prompt_style = cfg["prompt_style"]
    local.CFG.max_seq_len = cfg["max_seq_len"]

    done_ids: set[str] = set()
    file_exists = out_path.exists()
    if resume and file_exists:
        with out_path.open(newline="") as f:
            for r in csv.DictReader(f):
                done_ids.add(r["sample_id"])
        print(f"resuming: {len(done_ids)} rows already in {out_path}")

    todo = [row for row in rows if row["id"] not in done_ids]
    if not todo:
        print("nothing left to sample")
        return

    service_client = tinker.ServiceClient()
    sampler_path = sampler_path_for(run_dir, checkpoint)
    print(f"sampling {len(todo)} rows from checkpoint={checkpoint} ({sampler_path})")
    sampling_client = await service_client.create_sampling_client_async(model_path=sampler_path)

    fieldnames = [
        "sample_id", "category", "prompt", "completion", "model_answer",
        "ground_truth", "correct", "closed_think", "has_step_tags",
        "final_boxed", "reopened_think", "step_tag_counts", "n_step_tags",
        "has_conclusion_tag", "unknown_step_tags", "unbalanced_step_tags",
        "stop_reason", "n_tokens", "prompt_tokens", "max_tokens",
        "checkpoint", "model_name", "temperature", "timestamp",
    ]
    mode = "a" if (resume and file_exists) else "w"
    f_out = out_path.open(mode, newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()
        f_out.flush()

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    completed = 0
    failed = 0
    total = len(todo)

    async def process(row: dict) -> None:
        nonlocal completed, failed
        prompt_text = local.render_prompt(tokenizer, row["prompt"])
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        max_tokens = max(256, cfg["max_seq_len"] - len(prompt_ids))
        try:
            completion, stop_reason, n_tokens = await sample_one(
                sampling_client, tokenizer, prompt_ids, max_tokens, temperature, sem
            )
        except Exception as e:
            async with lock:
                failed += 1
                print(f"  FAILED id={row['id']} category={row['category']}: {e}", file=sys.stderr)
            return

        model_answer = extract_boxed(completion)
        checks = format_checks(completion)
        tag_report = step_tag_report(row["category"], completion)
        correct = is_correct(row["category"], model_answer, row["answer"])
        async with lock:
            writer.writerow(
                {
                    "sample_id": row["id"],
                    "category": row["category"],
                    "prompt": row["prompt"],
                    "completion": completion,
                    "model_answer": model_answer or "",
                    "ground_truth": row["answer"],
                    "correct": correct,
                    **checks,
                    **tag_report,
                    "stop_reason": stop_reason,
                    "n_tokens": n_tokens,
                    "prompt_tokens": len(prompt_ids),
                    "max_tokens": max_tokens,
                    "checkpoint": checkpoint,
                    "model_name": cfg["model_name"],
                    "temperature": temperature,
                    "timestamp": now_iso(),
                }
            )
            f_out.flush()
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"  {completed}/{total} sampled ({failed} failed)")

    await asyncio.gather(*(process(row) for row in todo))
    f_out.close()
    if failed:
        print(f"\n{failed} row(s) failed after {_MAX_ATTEMPTS} attempts each -- rerun with --resume to retry them")


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summarize(out_path: Path) -> None:
    stats = {
        cat: {
            "total": 0, "correct": 0, "closed_think": 0, "final_boxed": 0,
            "reopened_think": 0, "has_conclusion_tag": 0, "unbalanced_step_tags": 0,
            "n_step_tags_sum": 0,
        }
        for cat in CATEGORIES
    }
    unknown_seen: dict[str, set[str]] = defaultdict(set)
    with out_path.open(newline="") as f:
        for row in csv.DictReader(f):
            cat = row["category"]
            if cat not in stats:
                continue
            s = stats[cat]
            s["total"] += 1
            s["correct"] += row["correct"] == "True"
            s["closed_think"] += row["closed_think"] == "True"
            s["final_boxed"] += row["final_boxed"] == "True"
            s["reopened_think"] += row["reopened_think"] == "True"
            s["has_conclusion_tag"] += row["has_conclusion_tag"] == "True"
            s["unbalanced_step_tags"] += row["unbalanced_step_tags"] == "True"
            s["n_step_tags_sum"] += int(row["n_step_tags"] or 0)
            if row["unknown_step_tags"]:
                unknown_seen[cat].update(row["unknown_step_tags"].split(","))

    print(f"\n{'category':<18} {'total':>6} {'correct':>8} {'acc':>7}  {'closed':>7} {'boxed':>7} {'reopened':>9}")
    print("-" * 72)
    grand_total = grand_correct = 0
    for cat in CATEGORIES:
        s = stats[cat]
        if s["total"] == 0:
            continue
        acc = s["correct"] / s["total"]
        print(
            f"{cat:<18} {s['total']:>6} {s['correct']:>8} {acc:>6.1%}  "
            f"{s['closed_think']:>6}/{s['total']:<3} {s['final_boxed']:>4}/{s['total']:<3} "
            f"{s['reopened_think']:>6}/{s['total']:<3}"
        )
        grand_total += s["total"]
        grand_correct += s["correct"]
    print("-" * 72)
    if grand_total:
        print(f"{'TOTAL':<18} {grand_total:>6} {grand_correct:>8} {grand_correct / grand_total:>6.1%}")

    print(f"\n{'category':<18} {'avg tags':>9} {'conclusion':>11} {'unbalanced':>11}  unknown tags seen")
    print("-" * 72)
    for cat in CATEGORIES:
        s = stats[cat]
        if s["total"] == 0:
            continue
        avg_tags = s["n_step_tags_sum"] / s["total"]
        print(
            f"{cat:<18} {avg_tags:>9.1f} "
            f"{s['has_conclusion_tag']:>6}/{s['total']:<4} "
            f"{s['unbalanced_step_tags']:>6}/{s['total']:<4}  "
            f"{', '.join(sorted(unknown_seen.get(cat, ()))) or '-'}"
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default="final", help="name in checkpoints.jsonl")
    parser.add_argument("--per-category", type=int, default=250)
    parser.add_argument("--limit", type=int, default=None, help="cap total rows sampled (smoke test)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=8, help="max in-flight Tinker sample_async calls")
    parser.add_argument("--seed", type=int, default=0, help="row-selection seed")
    parser.add_argument("--out", default=None, help="default: sft_tinker_train_csv_eval.csv at repo root")
    parser.add_argument("--resume", action="store_true", help="skip sample_ids already present in --out")
    parser.add_argument("--yes", action="store_true", help="required to actually spend -- without it, preview only")
    ns = parser.parse_args(argv)

    run_dir = Path(ns.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO / run_dir
    cfg = load_run(run_dir)
    out_path = Path(ns.out) if ns.out else REPO / "sft_tinker_train_csv_eval.csv"

    print(f"run={run_dir}  model={cfg['model_name']}  checkpoint={ns.checkpoint}")

    excl = training_prompts(cfg)
    print(f"{len(excl)} distinct prompts in the run's training/val pool (excluded from eval)")

    pool, excluded_by_cat = load_eligible_pool(excl)
    eligible_by_cat = {cat: len(pool.get(cat, [])) for cat in CATEGORIES}
    rows = select_rows(pool, ns.per_category, ns.seed)
    if ns.limit is not None:
        rows = rows[: ns.limit]

    by_cat: dict[str, int] = defaultdict(int)
    for r in rows:
        by_cat[r["category"]] += 1
    print(f"\nselected {len(rows)} row(s):")
    for cat in CATEGORIES:
        print(f"  {cat:<18} {by_cat.get(cat, 0)}")

    manifest_path = out_path.with_suffix("").with_name(out_path.stem + ".manifest.json")
    manifest = {
        "script": "scripts/eval_sft_tinker_train_csv.py",
        "git_commit": git_commit(),
        "run_dir": str(run_dir),
        "checkpoint": ns.checkpoint,
        "model_name": cfg["model_name"],
        "prompt_style": cfg["prompt_style"],
        "max_seq_len": cfg["max_seq_len"],
        "temperature": ns.temperature,
        "seed": ns.seed,
        "per_category_requested": ns.per_category,
        "limit": ns.limit,
        "out_csv": str(out_path),
        "eligible_by_category": eligible_by_cat,
        "excluded_as_trained_by_category": excluded_by_cat,
        "selected_by_category": dict(by_cat),
        "selected_total": len(rows),
        "started_at": now_iso(),
        "finished_at": None,
    }

    if not ns.yes:
        print(
            "\n(preview only -- no Tinker calls made. Pass --yes to actually sample; "
            "try --limit 7 --yes first for a cheap smoke test.)"
        )
        return

    manifest_path.write_text(json.dumps(manifest, indent=2))
    asyncio.run(
        run_eval(
            rows, cfg, run_dir, ns.checkpoint, ns.temperature, ns.concurrency, out_path, ns.resume
        )
    )
    manifest["finished_at"] = now_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if out_path.exists():
        summarize(out_path)
        print(f"\nwrote results to {out_path}")
        print(f"wrote run manifest to {manifest_path}")


if __name__ == "__main__":
    main()
