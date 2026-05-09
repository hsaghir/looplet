def execute(ctx, *, summary: str, passed: bool) -> dict:
    return {"summary": summary, "passed": passed}
