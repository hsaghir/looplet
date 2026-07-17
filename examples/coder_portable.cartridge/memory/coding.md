# Coding guidelines

Static distillation of the coder agent's working conventions. (The
original `coder` cartridge derived equivalent memory dynamically from
`@instructions_memory` - discovering CLAUDE.md / AGENTS.md in the
project - and `@project_memory` - git branch + step budget. Those
builders are in-process and not portable; this file is the static
substitute. Per-project instruction files and live git context are not
surfaced by the portable twin.)

## Workflow
- Read a file before editing it; never edit blind.
- Prefer `edit_file` / `multi_edit` for surgical changes over rewriting
  whole files with `write_file`.
- Use `grep` and `glob` to locate code before reading; avoid scanning
  large trees with `bash`.
- After changing code, run the project's tests (`pytest`, `npm test`,
  `cargo test`, `go test`) and confirm they pass before calling `done`.

## Discipline
- Make only the changes that are requested or clearly necessary.
- Keep diffs minimal; do not refactor or reformat untouched code.
- When stuck, `think()` to plan instead of repeating a failing action.
- Call `done` with a concise summary once the task is verified complete.

## Safety
- Never run destructive commands (`rm -rf`, `sudo`, raw device writes).
- Do not write to system paths (`/etc/`).
