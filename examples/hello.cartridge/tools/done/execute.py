"""Done tool — completion sentinel."""


def execute(*, answer: str) -> dict:
    return {"status": "completed", "answer": answer}
