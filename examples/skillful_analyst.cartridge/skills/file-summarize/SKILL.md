---
name: file-summarize
description: Produce a short Markdown summary of a text file (line count, top tokens, first/last lines). Use when the user asks "what's in this file" or wants a quick characterisation.
tags: [text, summary, exploration]
---

# File summariser

Use this skill when the user wants a quick characterisation of a text
file but does NOT want full content reproduction.

## Steps

1. Use `read_text(path=<file>)` to load the file.
2. Count: total lines, total chars, distinct words (split on
   whitespace, lowercased, stripped of `.,;:!?"'()`).
3. Identify the 10 most-frequent words, ignoring this stoplist:
   `the a an of to in for and or is are was were be been being it its
   this that these those on at by from as with`.
4. Capture the first 3 non-empty lines and the last 3 non-empty
   lines.
5. Write a Markdown report to `<stem>.summary.md` with sections:
   `## Counts`, `## Top words`, `## First lines`, `## Last lines`.

## Output contract

The user expects `<stem>.summary.md`. Report the path in
`done(answer=...)`.
