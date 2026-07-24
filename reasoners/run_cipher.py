from reasoners.store_types import Problem, Example
from reasoners.cipher import reasoning_cipher

examples = [
    Example("wqj sajoju zeig fyxik", "the clever king found"),
    Example("kulgyi kujlph wqj qekkji wujlhxuj", "dragon dreams the hidden treasure"),
    Example("wqj lisejiw vueisjhh jtvayujh", "the ancient princess explores"),
    Example("hwxkjiw eplgeijh lryoj hsqyya", "student imagines above school"),
]

problem = Problem(
    id="test1",
    category="cipher",
    examples=examples,
    question="wqj syayufxa wjlsqju ujlkh",
    answer="",
    prompt="",
)

result = reasoning_cipher(problem)

print(result if result is not None else "None")
