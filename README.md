# reasoners-grpo-kaggle

Standalone extract of the synthetic-reasoning-task solvers/reward functions
from `huikang_nemotron`, plus a GRPO training script meant to run on a
Kaggle/Colab GPU (as opposed to that repo's Tinker-hosted `train_sft.py`).

This folder has no dependency on the parent repo -- it's a self-contained
`uv`-managed project. Move it anywhere and `uv sync` will set it up.

## Layout

```
reasoners/                  deterministic solvers + reward functions (a plain Python package)
  store_types.py            Problem/Example dataclasses + numeric-formatting helpers
  cipher.py                 deterministic solver for the substitution-cipher task
  cryptarithm.py            deterministic solver for the cryptarithm task
  equation_numeric.py       deterministic solver for the equation_numeric task
  bit_manipulation.py       deterministic solver for the bit_manipulation task
  gravity.py                deterministic solver for the gravity task
  numeral.py                deterministic solver for the numeral task
  unit_conversion.py        deterministic solver for the unit_conversion task
  reward_cipher.py          dense, per-step XML-tag reward function for cipher traces
  monitor_cipher.py         example: generate a random cipher problem, solve it, score the trace
  run_cipher.py             example: construct a Problem and call the cipher solver
  dictionary.txt            word list used by some solvers for unknown-word lookups
  wonderland.txt            small word list (77 words) used by the cipher task/reward

kaggle/
  train_grpo_cipher_kaggle.py   GRPO (TRL + PEFT LoRA) training script for the cipher task
```

Only the cipher task has a reward function (`reward_cipher.py`) and a GRPO
script wired up so far. The other solvers (`cryptarithm.py`,
`equation_numeric.py`, `bit_manipulation.py`, `gravity.py`, `numeral.py`,
`unit_conversion.py`) are deterministic trace generators only -- extending
GRPO training to them means writing an `evaluate_structured_trace`-style
reward function for each, following `reward_cipher.py` as a template.

## Setup

```
uv sync
```

This creates `.venv` with `reasoners` installed (editable) plus the ML deps
needed to run `kaggle/train_grpo_cipher_kaggle.py` locally: `torch`,
`transformers`, `trl`, `peft`, `accelerate`, `bitsandbytes`, `datasets`.

## Running locally (dev/testing on any GPU box)

```
uv run python kaggle/train_grpo_cipher_kaggle.py
```

`INPUT_DIR` in the script auto-falls-back to the local `reasoners/` folder
when `/kaggle/input/...` doesn't exist, so this works out of the box from a
checkout of this project.

## Running on Kaggle/Colab

See the docstring at the top of `kaggle/train_grpo_cipher_kaggle.py` for the
full walkthrough. Short version:

1. Create a Kaggle Dataset containing just `reasoners/reward_cipher.py` and
   `reasoners/wonderland.txt`.
2. Attach it to the notebook, point `INPUT_DIR` at its mount path.
3. Turn on a GPU accelerator.
4. `!pip install -q -U trl peft accelerate bitsandbytes datasets`
5. Run the script. It checkpoints periodically and resumes automatically
   within a session; across sessions, save/restore `OUTPUT_DIR` as a Kaggle
   Dataset output since `/kaggle/working` doesn't persist by default.
