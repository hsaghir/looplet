"""think tool — pure reasoning checkpoint, no side effects."""


def execute(*, thought: str) -> dict:
    return {"thought": thought, "noted": True}
