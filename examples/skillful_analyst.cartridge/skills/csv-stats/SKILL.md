---
name: csv-stats
description: Compute mean, median, min, max, count for every numeric column in a CSV file. Use when the user asks for descriptive statistics, summary stats, or "tell me about this CSV".
tags: [csv, statistics, analysis]
---

# CSV stats

Use this skill when the user wants descriptive statistics for a CSV file.

## Steps

1. Use `read_text(path=<csv>)` to load the file.
2. Parse the header row by splitting the first line on `,`.
3. For every subsequent row, split on `,` and try to coerce each cell
   to `float`. Cells that fail the coercion are treated as missing
   for that column.
4. For every column with at least one numeric value, compute:
   `count`, `mean`, `median`, `min`, `max`. Use the standard
   formulas:
   - mean = sum(values) / count
   - median = sorted middle value (or average of two middles for
     even-length lists)
5. Render a Markdown table with one row per numeric column. Headers:
   `column | count | mean | median | min | max`. Round all floats to
   3 decimal places.
6. Write the Markdown table to `<csv stem>_stats.md` next to the
   input file using `write_text`. Include a top-level heading
   `# Statistics for <filename>`.

## Output contract

The user expects a `<stem>_stats.md` file written to disk. The
`done(answer=...)` summary should mention the path.
