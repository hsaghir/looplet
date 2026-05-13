"""Done tool — completion sentinel."""


def execute(*, summary: str) -> dict:
    return {"status": "completed", "summary": summary}
