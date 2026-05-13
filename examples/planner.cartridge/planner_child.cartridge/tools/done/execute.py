"""Done tool — completion sentinel for the planner child."""


def execute(*, summary: str) -> dict:
    return {"status": "completed", "summary": summary}
