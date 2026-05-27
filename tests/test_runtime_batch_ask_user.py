from __future__ import annotations

from looplet import SkillRuntime
from looplet.bundles import QuestionSpec


def test_batch_ask_user_defaults_to_sequential_ask_user_calls() -> None:
    calls: list[tuple[str, list[str]]] = []

    def ask_handler(question: str, options: list[str]) -> str:
        calls.append((question, options))
        return f"answer-{len(calls)}"

    runtime = SkillRuntime(ask_handler=ask_handler)

    answers = runtime.batch_ask_user(
        prelude="Please answer these setup questions.",
        questions=[
            {"id": "scope", "question": "Which scope?", "options": ["diff", "all"]},
            {"id": "format", "question": "Which output format?"},
        ],
    )

    assert answers == {"scope": "answer-1", "format": "answer-2"}
    assert calls == [
        ("Which scope?", ["diff", "all"]),
        ("Which output format?", []),
    ]


def test_batch_ask_user_rejects_duplicate_question_ids() -> None:
    runtime = SkillRuntime(ask_handler=lambda _question, _options: "unused")

    try:
        runtime.batch_ask_user(
            prelude="Please answer these setup questions.",
            questions=[
                {"id": "scope", "question": "Which scope?"},
                {"id": "scope", "question": "Really, which scope?"},
            ],
        )
    except ValueError as exc:
        assert "duplicate batch question id" in str(exc)
    else:
        raise AssertionError("duplicate ids should raise")


def test_runtime_adapter_can_override_batch_ask_user() -> None:
    class NativeBatchRuntime(SkillRuntime):
        def batch_ask_user(self, *, prelude: str, questions: list[QuestionSpec]) -> dict[str, str]:
            assert prelude == "One form, please."
            return {question["id"]: f"native-{index}" for index, question in enumerate(questions, 1)}

    def fail_if_called(_question: str, _options: list[str]) -> str:
        raise AssertionError("overridden batch_ask_user should bypass ask_user")

    answers = NativeBatchRuntime(ask_handler=fail_if_called).batch_ask_user(
        prelude="One form, please.",
        questions=[
            {"id": "severity", "question": "Severity threshold?", "options": ["low", "high"]},
            {"id": "format", "question": "Output format?", "options": []},
        ],
    )

    assert answers == {"severity": "native-1", "format": "native-2"}
