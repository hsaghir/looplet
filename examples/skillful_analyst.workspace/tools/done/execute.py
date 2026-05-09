"""Done sentinel — the loop's done_tool consumes this."""


def execute(ctx, *, answer: str) -> dict:  # noqa: ARG001
    return {"answer": answer}
