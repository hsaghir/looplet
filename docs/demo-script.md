# Demo script — 30-second asciinema recording

A single take, ~30 seconds, that ends up as the hero GIF on the README.
Keep it minimal — no `cd`, no editor, no sudo.

## Prep

```bash
# Terminal: wide, monospace font, dark theme, 120×30.
# Ensure prompt is short ($ only).
export PS1='$ '
clear

# Python env already set up with openharness[openai] installed.
# OPENAI_BASE_URL + OPENAI_MODEL pointed at a fast model (Groq, Together).
```

## Script

```text
$ python -m openharness.examples.coding_agent "write fizzbuzz in fizz.py and test it"
```

The agent should:

1. Call `write` — creates `fizz.py` with a FizzBuzz implementation.
2. Call `bash` — `python fizz.py` to sanity-check.
3. Call `write` — creates `test_fizz.py` with pytest tests.
4. Call `bash` — `pytest test_fizz.py` — all green.
5. Call `done` with a summary.

Each step prints via `step.pretty()` in under 80 columns:

```
#1 ✓ write(file_path='fizz.py') → 412B [182ms]
#2 ✓ bash(command='python fizz.py') → exit=0 [91ms]
#3 ✓ write(file_path='test_fizz.py') → 328B [145ms]
#4 ✓ bash(command='pytest test_fizz.py') → 3 passed [842ms]
#5 ✓ done(answer='FizzBuzz implemented and tested') → final [0ms]
```

Total: ~8 seconds of visible output.

## Record

```bash
asciinema rec demo.cast -c "python -m openharness.examples.coding_agent 'write fizzbuzz in fizz.py and test it'"

# Convert to GIF:
agg demo.cast demo.gif --theme monokai --speed 1.2 --rows 14 --cols 100
# or SVG animated:
svg-term --cast demo.cast --out demo.svg --width 100 --height 14
```

## Embed

Drop `demo.gif` (or `demo.svg`) at the top of the README, right under
the h1:

```markdown
# openharness

![demo](./docs/demo.gif)

**The tool-calling loop you can actually step through.**
...
```

## Re-record checklist

- Fresh working directory (no leftover `fizz.py`).
- API key works and the model is fast (< 2 s/step).
- No stderr warnings leaked into the tape.
- Total run ≤ 12 s so the GIF loops smoothly.
