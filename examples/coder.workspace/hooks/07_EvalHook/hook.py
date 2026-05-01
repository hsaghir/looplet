"""EvalHook subclass exposing both evaluators and collectors via @ref.

Both kwargs round-trip through to_config so the workspace snapshot
captures them as ``@eval_evaluators`` / ``@eval_collectors`` refs.
The corresponding builders live in resources/.
"""

from looplet import EvalHook as _EvalHook


class EvalHook(_EvalHook):
    def to_config(self) -> dict:
        return {
            "evaluators": "@eval_evaluators",
            "collectors": "@eval_collectors",
        }
