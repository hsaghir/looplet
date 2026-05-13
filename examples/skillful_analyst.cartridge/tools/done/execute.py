"""Done sentinel — the loop's done_tool consumes this."""


def execute(ctx, *, summary: str) -> dict:  # noqa: ARG001
    return {"summary": summary}
