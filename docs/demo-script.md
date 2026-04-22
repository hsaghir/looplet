# Demo script — GIF recording

The hero GIF at the top of the README is produced from a deterministic
scripted run of `examples/scripted_demo.py` — no API key, no network,
no flaky timing. Re-record whenever the loop's output format changes.

## Prep

```bash
# Install the recorder + converter (one-time):
uv tool install asciinema
cargo install --git https://github.com/asciinema/agg --tag v1.5.0

# Short prompt so the recording is clean:
export PS1='$ '
clear
```

## Record

```bash
cd <repo-root>
rm -f docs/demo.cast docs/demo.gif
asciinema rec docs/demo.cast --overwrite --cols 90 --rows 22 \
    --command "uv run --quiet --active python -m looplet.examples.scripted_demo"
agg docs/demo.cast docs/demo.gif --theme monokai --cols 90 --rows 22 --font-size 18
```

Output: ~5-second, 16-frame, ~40 KB GIF that loops forever. Shows a
`DebugHook` printing per-dispatch/per-result lines, an `ApprovalHook`
pause on the destructive `delete_rows` call, and a clean resume.

## Real-LLM version (optional)

If you want a recording that hits a real model (Groq, Together, vLLM,
Ollama), substitute the command:

```bash
asciinema rec docs/demo.cast --overwrite --cols 100 --rows 16 \
    --command "python -m looplet.examples.coding_agent 'write fizzbuzz in fizz.py and test it'"
```

Requires `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_API_KEY` and a fast
(< 2 s/step) model. Timings won't be deterministic — expect slight
variation between runs.

## Re-record checklist

- Fresh working directory (no leftover `fizz.py`, `test_fizz.py`).
- The `$` prompt is the only thing before the command.
- No stderr warnings leaked into the tape (check with
  `grep -i warn docs/demo.cast`).
- GIF is under 200 KB and loops smoothly.
- Frame count roughly matches lines of output (use Pillow to inspect
  if curious).

