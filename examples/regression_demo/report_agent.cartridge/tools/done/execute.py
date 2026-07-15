"""Completion sentinel for the report-agent demo."""


def execute(*, summary: str) -> dict:
    return {"status": "completed", "summary": summary}
