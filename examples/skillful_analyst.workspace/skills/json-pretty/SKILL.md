---
name: json-pretty
description: Pretty-print a JSON file with sorted keys and 2-space indent. Use when the user asks to format, prettify, or "make this JSON readable".
tags: [json, formatting]
---

# JSON pretty-printer

Use this skill when the user wants a JSON file formatted for human
reading.

## Steps

1. Use `read_text(path=<json>)` to load the file.
2. Parse it with `json.loads`. If parsing fails, write the parse
   error to `<stem>.parse_error.txt` and stop — do not write a
   partial output.
3. Re-emit with `json.dumps(obj, indent=2, sort_keys=True,
   ensure_ascii=False)`.
4. Write the result back to `<stem>.pretty.json` next to the
   original. Do NOT modify the original.

## Output contract

The user expects a `<stem>.pretty.json` file. The `done(answer=...)`
summary should report the new file's path.
