"""Done tool - completion sentinel for the planner."""


def execute(*, summary: str) -> dict:
    return {"status": "completed", "summary": summary}
