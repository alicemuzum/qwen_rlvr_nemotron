import re
from typing import Any


def evaluate_structured_trace(
    response_xml: str, oracle_map: dict[str, str], expected_words: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    """
    Evaluates a generated reasoning trace for the substitution cipher task.
    Provides a dense reward signal incorporating long horizon evaluation, state awareness,
    and penalties for unwanted actions (e.g., hallucinated mappings, skipping steps).

    Args:
        response_xml: The raw XML/text output from the model.
        oracle_map: The ground truth cipher-to-plain character mapping.
        expected_words: The list of ground truth decrypted words for the target question.

    Returns:
        tuple: (total_reward, step_logs)
            - total_reward (float): The computed reward score.
            - step_logs (list): Detailed list of dictionaries tracking reward at each step.
    """
    total_reward = 0.0
    agent_knowledge: dict[str, str] = {}
    verified_word_count = 0
    step_logs: list[dict[str, Any]] = []

    # Extract tags sequentially
    # Regex captures: 1) Tag type, 2) Inner content
    steps = re.findall(r'<step type="(.*?)">(.*?)</step>', response_xml, re.DOTALL)

    if not steps:
        step_logs.append(
            {
                "tag_type": "ERROR",
                "content": "Total format failure (no steps found)",
                "reward_delta": -5.0,
                "total_reward": -5.0,
                "reason": "Missing steps",
            }
        )
        return -5.0, step_logs

    for tag_type, content in steps:
        content = content.strip()
        prev_reward = total_reward
        reason = ""

        if tag_type == "state_update":
            # Expecting format "c->p"
            if "->" in content:
                cipher, plain = content.split("->", 1)
                cipher, plain = cipher.strip(), plain.strip()

                if oracle_map.get(cipher) == plain:
                    if cipher not in agent_knowledge:
                        total_reward += 0.5
                        reason = "R_discovery (correctly deduced a new mapping)"
                        agent_knowledge[cipher] = plain
                    else:
                        total_reward += 0.05
                        reason = "R_consistency (re-affirmed known mapping)"
                else:
                    total_reward -= 2.0
                    reason = "P_fatal (Logic error: deduced an incorrect mapping)"
            else:
                total_reward -= 0.5
                reason = "Formatting error inside tag"

        elif tag_type == "execution":
            # Expecting format "c->p" or "c->?" or something like "【p】->【p】same"
            if "->" in content:
                m = re.search(
                    r"(?:【)?(?:\()?([a-z])(?:\))?(?:】)?->(?:【)?([a-z?])(?:】)?",
                    content,
                )
                if m:
                    cipher, plain = m.group(1), m.group(2)

                    if plain == "?":
                        if cipher in agent_knowledge:
                            total_reward -= 1.0
                            reason = "Ignored its own established knowledge"
                        else:
                            total_reward += 0.1
                            reason = "Correctly admitted ignorance (state awareness)"
                    else:
                        if agent_knowledge.get(cipher) == plain:
                            total_reward += 0.2
                            reason = "R_apply (correctly applied known knowledge)"
                        elif oracle_map.get(cipher) == plain:
                            total_reward -= 0.5
                            reason = "Hallucination / Skipped step: got right answer without deduction"
                        else:
                            total_reward -= 1.0
                            reason = "Hallucination and wrong answer"
                else:
                    total_reward -= 0.5
                    reason = "Formatting error"
            else:
                total_reward -= 0.5
                reason = "Formatting error"

        elif tag_type == "analysis":
            # Handle dictionary checks
            if (
                "match" in content
                or "unmatchable" in content
                or "consistent" in content
                or "contradiction" in content
            ):
                total_reward += 0.01
                reason = "Dictionary check string evaluation"
            elif content.startswith("Checking word:"):
                total_reward += 0.05
                reason = "Initiated dictionary check"

        elif tag_type == "verification":
            if "Best match:" in content:
                m = re.search(r"【(.*?)】", content)
                if m:
                    word = m.group(1).strip()
                    if (
                        verified_word_count < len(expected_words)
                        and word == expected_words[verified_word_count]
                    ):
                        total_reward += 1.0
                        reason = "Correctly identified the word via dictionary"
                    else:
                        total_reward -= 1.5
                        reason = "Identified the wrong word"
            elif "The sentence currently is" in content:
                lines = content.split("\n")
                if len(lines) >= 2:
                    sentence = lines[-1].strip()
                    current_expected = " ".join(expected_words[:verified_word_count])
                    if (
                        sentence.startswith(current_expected)
                        or current_expected in sentence
                    ):
                        total_reward += 0.2
                        reason = "Good sentence tracking"
                    else:
                        total_reward -= 0.5
                        reason = "Poor sentence tracking"
            elif "->" in content:
                parts = content.split("->")
                if len(parts) >= 2:
                    derived_word = parts[-1].strip()
                    if "(" not in derived_word:
                        if verified_word_count < len(expected_words):
                            if derived_word == expected_words[verified_word_count]:
                                total_reward += 1.0
                                reason = "Subgoal achieved (derived expected word)"
                                verified_word_count += 1
                            else:
                                total_reward -= 1.0
                                reason = "Incorrect full derivation"

        elif tag_type == "conclusion":
            final_match = re.search(r"\\boxed{(.*?)}", content)
            if final_match and final_match.group(1).strip() == " ".join(expected_words):
                total_reward += 10.0
                reason = "R_terminal_win (correct boxed answer)"
            else:
                total_reward -= 5.0
                reason = "R_terminal_fail (incorrect or unboxed answer)"

        step_logs.append(
            {
                "tag_type": tag_type,
                "content": content if len(content) < 50 else content[:47] + "...",
                "reward_delta": total_reward - prev_reward,
                "total_reward": total_reward,
                "reason": reason if reason else "No specific reward triggered",
            }
        )

    # Long horizon evaluation modifiers
    horizon_delta = 0.0
    horizon_reason = ""
    if verified_word_count == len(expected_words):
        horizon_delta = 5.0
        horizon_reason = "Long horizon: Successfully verified all required subgoals"
    elif verified_word_count > 0:
        horizon_delta = (verified_word_count / len(expected_words)) * 2.0
        horizon_reason = f"Long horizon: Partial success ({verified_word_count}/{len(expected_words)} subgoals)"

    if horizon_delta > 0:
        total_reward += horizon_delta
        step_logs.append(
            {
                "tag_type": "HORIZON_EVAL",
                "content": "End of trace evaluation",
                "reward_delta": horizon_delta,
                "total_reward": total_reward,
                "reason": horizon_reason,
            }
        )

    return total_reward, step_logs
