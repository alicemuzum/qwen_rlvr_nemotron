"""GRPO training for the substitution-cipher task on a Kaggle/Colab GPU.

This script runs the whole RL loop locally against a small HF model using
TRL's GRPOTrainer + PEFT LoRA. It's meant to be dropped into a Kaggle
notebook (or Colab) as-is, or run locally via `uv run` for development.

Reuses reasoners/reward_cipher.py's `evaluate_structured_trace` unchanged as
the reward function, so a trace is scored by the exact same dense, per-step
XML-tag rubric (rewards for correct/incorrect deductions, dictionary
lookups, and the final \\boxed{} answer).

--- Local dev/testing ---
From the project root: `uv sync`, then `uv run python kaggle/train_grpo_cipher_kaggle.py`.
INPUT_DIR falls back to the local `reasoners/` folder automatically.

--- Kaggle setup ---
1. Create a Kaggle Dataset containing just two files:
     reasoners/reward_cipher.py
     reasoners/wonderland.txt
   (nothing else is needed -- this script reimplements the small amount of
   cipher-generation logic from reasoners/monitor_cipher.py inline so you
   don't have to upload the rest of the package).
2. Attach that dataset to the notebook; set INPUT_DIR below to its mount
   path (typically /kaggle/input/<dataset-slug>).
3. Turn on a GPU accelerator (T4 x1 is enough for the 0.6B default; use an
   A100/T4x2 + 4-bit loading for the 4B variant).
4. Run the pip install cell below, then run the script (or paste it cell by
   cell). Note a Kaggle/Colab notebook environment is NOT a uv-managed venv
   -- use plain pip there, uv is only for local dev on this project.

    !pip install -q -U trl peft accelerate bitsandbytes datasets

5. Session limits: Kaggle sessions cap out around 9-12h and can disconnect
   earlier. This script checkpoints the LoRA adapter every SAVE_STEPS steps
   to OUTPUT_DIR and auto-resumes from the latest checkpoint found there. To
   survive a session ending, periodically save OUTPUT_DIR as a Kaggle Dataset
   output and, on the next run, copy it back into OUTPUT_DIR before starting
   (or just set OUTPUT_DIR itself to a path under a persisted dataset/drive).

Usage:
    python train_grpo_cipher_kaggle.py
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

# --- point this at wherever you mounted the Kaggle Dataset from step 1/2 ---
INPUT_DIR = Path("/kaggle/input/cipher-reward")  # <-- change to your dataset slug
if not INPUT_DIR.exists():
    # fall back to running straight out of this project checkout (local testing)
    INPUT_DIR = Path(__file__).resolve().parent.parent / "reasoners"

sys.path.insert(0, str(INPUT_DIR.parent))
sys.path.insert(0, str(INPUT_DIR))

from reward_cipher import evaluate_structured_trace

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class Cfg:
    model_name: str = "Qwen/Qwen3-0.6B"  # swap to "Qwen/Qwen3-4B" for the bigger run
    output_dir: str = "/kaggle/working/grpo_cipher"
    lora_rank: int = 16
    lora_alpha: int = 32
    num_train_problems: int = 4000
    num_examples_per_problem: int = 3
    num_question_words: int = 3  # keep small early; the deterministic solver's
    # own dictionary-scan reward terms get harder to hit as this grows
    num_generations: int = 8  # rollouts per prompt (GRPO group size)
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 768
    max_completion_length: int = 2048
    learning_rate: float = 1e-5
    beta: float = 0.02  # KL penalty coefficient
    num_train_epochs: int = 3
    save_steps: int = 20
    logging_steps: int = 1
    seed: int = 0


CFG = Cfg()


# --------------------------------------------------------------------------
# Cipher problem generation (inlined from reasoners/monitor_cipher.py so the
# Kaggle dataset only needs reward_cipher.py + wonderland.txt)
# --------------------------------------------------------------------------

ALPHABET = list("abcdefghijklmnopqrstuvwxyz")


def load_words() -> list[str]:
    path = INPUT_DIR / "wonderland.txt"
    with path.open() as f:
        return [w.strip() for w in f if w.strip()]


def make_random_cipher(rng: random.Random) -> tuple[dict[str, str], dict[str, str]]:
    shuffled = list(ALPHABET)
    rng.shuffle(shuffled)
    plain_to_cipher = {p: c for p, c in zip(ALPHABET, shuffled)}
    cipher_to_plain = {c: p for p, c in zip(ALPHABET, shuffled)}
    return plain_to_cipher, cipher_to_plain


def encrypt(text: str, plain_to_cipher: dict[str, str]) -> str:
    return "".join(plain_to_cipher.get(c, c) for c in text)


SYSTEM_PROMPT = """You are solving a substitution cipher puzzle. Every letter in the cipher \
text maps to exactly one plaintext letter (a 1-to-1 substitution over a-z). You are given \
example cipher/plaintext sentence pairs, then a new cipher question to decrypt.

Write your reasoning as a sequence of XML step tags, then give the final answer. The tags:
  <step type="state_update">c->p</step>       -- record a letter mapping you deduced from an example
  <step type="execution">c->p</step>          -- apply a known mapping while decoding a question word
  <step type="execution">c->?</step>          -- admit a cipher letter you don't know yet
  <step type="analysis">...</step>            -- notes, e.g. dictionary checks for an unknown word
  <step type="verification">Best match: 【word】</step>  -- state the dictionary word you resolved an unknown word to
  <step type="conclusion">... \\boxed{final answer}</step>       -- final answer, space-separated words, must be the last tag

Worked example:
Examples: "wqj sajoju" -> "the clever"
Question: "wqj eplgeijh"

<step type="state_update">w->t</step>
<step type="state_update">q->h</step>
<step type="state_update">j->e</step>
<step type="state_update">s->c</step>
<step type="state_update">a->l</step>
<step type="state_update">o->v</step>
<step type="state_update">u->r</step>
<step type="execution">w->t</step>
<step type="execution">q->h</step>
<step type="execution">j->e</step>
<step type="analysis">Checking word: eplgeijh 8 decoding with known letters, e and j unknown-free letters resolved, rest unknown</step>
<step type="execution">e->?</step>
<step type="execution">p->?</step>
<step type="execution">l->?</step>
<step type="execution">g->?</step>
<step type="execution">e->?</step>
<step type="execution">i->?</step>
<step type="execution">j->e</step>
<step type="execution">h->?</step>
<step type="verification">Best match: 【imagines】</step>
<step type="conclusion">\\boxed{the imagines}</step>

Now solve the new problem below in the same tag format. Do not skip steps or guess a word \
without a verification step."""


def build_prompt_and_meta(
    problem_idx: int, words: list[str], rng: random.Random
) -> dict:
    plain_to_cipher, cipher_to_plain = make_random_cipher(rng)

    examples = []
    for _ in range(CFG.num_examples_per_problem):
        n_words = rng.randint(2, 4)
        sent_words = rng.sample(words, n_words)
        plain_text = " ".join(sent_words)
        cipher_text = encrypt(plain_text, plain_to_cipher)
        examples.append((cipher_text, plain_text))

    question_words = rng.sample(words, CFG.num_question_words)
    question_plain = " ".join(question_words)
    question_cipher = encrypt(question_plain, plain_to_cipher)

    example_lines = "\n".join(f'"{c}" -> "{p}"' for c, p in examples)
    user_prompt = (
        f"Examples:\n{example_lines}\n\n"
        f'Question: "{question_cipher}"\n\n'
        "Decrypt the question."
    )

    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "oracle_map_json": json.dumps(cipher_to_plain),
        "expected_words_json": json.dumps(question_words),
    }


def build_dataset():
    from datasets import Dataset

    words = load_words()
    rng = random.Random(CFG.seed)
    rows = [build_prompt_and_meta(i, words, rng) for i in range(CFG.num_train_problems)]
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------
# Reward function -- thin adapter around reasoners/reward_cipher.py
# --------------------------------------------------------------------------


def cipher_reward(prompts, completions, **kwargs) -> list[float]:
    oracle_maps_json = kwargs["oracle_map_json"]
    expected_words_json = kwargs["expected_words_json"]

    rewards: list[float] = []
    for completion, oracle_map_json, expected_words_str in zip(
        completions, oracle_maps_json, expected_words_json
    ):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        oracle_map = json.loads(oracle_map_json)
        expected_words = json.loads(expected_words_str)
        reward, _ = evaluate_structured_trace(text, oracle_map, expected_words)
        rewards.append(reward)
    return rewards


# --------------------------------------------------------------------------
# Training entrypoint
# --------------------------------------------------------------------------


def find_latest_checkpoint(output_dir: str) -> str | None:
    out = Path(output_dir)
    if not out.exists():
        return None
    checkpoints = sorted(
        out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
    )
    return str(checkpoints[-1]) if checkpoints else None


def main():
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    dataset = build_dataset()

    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=CFG.lora_rank,
        lora_alpha=CFG.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    args = GRPOConfig(
        output_dir=CFG.output_dir,
        per_device_train_batch_size=CFG.per_device_train_batch_size,
        gradient_accumulation_steps=CFG.gradient_accumulation_steps,
        num_generations=CFG.num_generations,
        max_prompt_length=CFG.max_prompt_length,
        max_completion_length=CFG.max_completion_length,
        learning_rate=CFG.learning_rate,
        beta=CFG.beta,
        num_train_epochs=CFG.num_train_epochs,
        save_steps=CFG.save_steps,
        logging_steps=CFG.logging_steps,
        bf16=True,
        seed=CFG.seed,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[cipher_reward],
        args=args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    resume_from = find_latest_checkpoint(CFG.output_dir)
    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
    trainer.train(resume_from_checkpoint=resume_from)

    trainer.save_model(CFG.output_dir)
    print(f"Done. Final adapter saved to {CFG.output_dir}")


if __name__ == "__main__":
    main()
