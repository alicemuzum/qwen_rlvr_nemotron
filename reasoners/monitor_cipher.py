import sys
import random
import string
from pathlib import Path

from reasoners.store_types import Problem, Example
from reasoners.cipher import reasoning_cipher
from reasoners.reward_cipher import evaluate_structured_trace


def load_words():
    path = Path(__file__).parent / "wonderland.txt"
    with path.open() as f:
        return [w.strip() for w in f if w.strip()]


def make_random_cipher():
    alphabet = list(string.ascii_lowercase)
    shuffled = list(alphabet)
    random.shuffle(shuffled)
    plain_to_cipher = {p: c for p, c in zip(alphabet, shuffled)}
    cipher_to_plain = {c: p for p, c in zip(alphabet, shuffled)}
    return plain_to_cipher, cipher_to_plain


def encrypt(text, plain_to_cipher):
    return "".join(plain_to_cipher.get(c, c) for c in text)


def main():
    words = load_words()

    # 1. Setup a new problem
    plain_to_cipher, oracle_map = make_random_cipher()

    example_sentences = [
        random.sample(words, 4),
        random.sample(words, 3),
        random.sample(words, 4),
    ]

    examples = []
    for sent_words in example_sentences:
        plain_text = " ".join(sent_words)
        cipher_text = encrypt(plain_text, plain_to_cipher)
        examples.append(Example(cipher_text, plain_text))

    question_words = random.sample(words, 5)
    question_plain = " ".join(question_words)
    question_cipher = encrypt(question_plain, plain_to_cipher)

    problem = Problem(
        id="monitor_test",
        category="cipher",
        examples=examples,
        question=question_cipher,
        answer=question_plain,
        prompt="",
    )

    # 2. Run reasoning
    print(f"--- Generating trace for new question: '{question_plain}' ---")
    trace = reasoning_cipher(problem)

    if not trace:
        print("Model failed to produce a valid trace.")
        sys.exit(1)

    # 3. Evaluate trace
    print("--- Evaluating trace... ---\n")
    total_reward, step_logs = evaluate_structured_trace(
        trace, oracle_map, question_words
    )

    # 4. Display monitoring output
    print(f"{'Step Type':<16} | {'Delta':<6} | {'Total':<6} | {'Reason':<40} | Content")
    print("-" * 120)
    for log in step_logs:
        tag = log["tag_type"]
        content = log["content"].replace("\n", " ")
        if len(content) > 30:
            content = content[:27] + "..."
        delta = log["reward_delta"]
        total = log["total_reward"]
        reason = log["reason"]

        # Color formatting
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        if delta > 0:
            delta_str = f"\033[92m{delta_str}\033[0m"  # Green
        elif delta < 0:
            delta_str = f"\033[91m{delta_str}\033[0m"  # Red
        else:
            delta_str = f"\033[90m{delta_str}\033[0m"  # Gray

        print(f"{tag:<16} | {delta_str:<15} | {total:<6.2f} | {reason:<40} | {content}")

    print("-" * 120)
    print(f"FINAL REWARD: {total_reward:.2f}")


if __name__ == "__main__":
    main()
