"""Custom YAML reader/writer for the subset of YAML the cartridge format uses.

Dependency-free (stdlib only) — keeping looplet.cartridge a zero-dep
package is a v1 commitment. The full PyYAML dep is overkill for the
shape we need: ``key: value`` mappings, nested blocks, lists of
scalars, lists of dicts, inline flow collections (``[a, b]``,
``{a: 1}``), block scalars (``|``/``>``), comments
(including inline `` # `` on value lines), and ``${runtime.x}` /
``${py:mod:sym}`` reference grammar (resolved later in
:mod:`looplet.cartridge._resources`, kept as raw strings here).

Two public symbols: :func:`_load_yaml` and :func:`_dump_yaml`.
Both are imported into :mod:`looplet.cartridge` and re-exported via
the package's public API surface.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from looplet.cartridge._layout import CartridgeSerializationError

# ── helpers: minimal YAML (key: value, lists, nested dicts) ────


def _dump_yaml(value: Any, indent: int = 0) -> str:
    """Dependency-free YAML emitter for the JSON subset we need.

    Looplet has no third-party dependencies; we hand-emit the limited
    YAML subset we use (scalars, lists of scalars/dicts, nested dicts).
    """
    pad = "  " * indent
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        if not value or any(c in value for c in ":#\n'\"") or value.strip() != value:
            return json.dumps(value)
        return value
    if isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                rendered = _dump_yaml(item, indent + 1).rstrip()
                if "\n" in rendered:
                    lines.append(f"{pad}-")
                    lines.append(rendered)
                else:
                    lines.append(f"{pad}- {rendered.strip()}")
            else:
                lines.append(f"{pad}- {_dump_yaml(item, 0)}")
        return "\n".join(lines)
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for key, val in value.items():
            rendered = _dump_yaml(val, indent + 1)
            if isinstance(val, (dict, list)) and rendered not in ("{}", "[]"):
                lines.append(f"{pad}{key}:")
                lines.append(rendered)
            else:
                lines.append(f"{pad}{key}: {rendered}")
        return "\n".join(lines)
    raise CartridgeSerializationError(
        f"cannot serialize value of type {type(value).__name__!r} to workspace YAML"
    )


def _load_yaml(text: str, *, source_path: str | Path | None = None) -> Any:
    """Parse the YAML subset emitted by :func:`_dump_yaml`.

    Supports key: value lines, nested dicts (indent 2), lists with ``- ``,
    and JSON-style scalars (true/false/null/numbers/quoted strings). For
    anything beyond this subset we fall back to JSON parsing of the line
    value.

    The optional ``source_path`` is appended to parse errors so a
    typo in ``hooks/05_QualityGate/config.yaml`` reports its location
    instead of leaving the user to grep every workspace YAML file.
    """
    src = f" (in {source_path})" if source_path else ""
    lines = [line.rstrip() for line in text.splitlines()]
    pos = 0

    def parse_block(min_indent: int) -> Any:
        nonlocal pos
        # Detect whether the block is a list (lines starting with "- ")
        # or a dict (lines with "key: value"). Empty block → empty dict.
        while pos < len(lines) and not lines[pos].strip():
            pos += 1
        if pos >= len(lines):
            return {}
        first = lines[pos]
        first_indent = len(first) - len(first.lstrip())
        if first_indent < min_indent:
            return {}
        is_list = first.lstrip().startswith("- ") or first.lstrip() == "-"
        if is_list:
            return parse_list(first_indent)
        return parse_dict(first_indent)

    def parse_dict(indent: int) -> dict[str, Any]:
        nonlocal pos
        out: dict[str, Any] = {}
        while pos < len(lines):
            line = lines[pos]
            if not line.strip() or line.lstrip().startswith("#"):
                # Skip blank lines and full-line comments.
                pos += 1
                continue
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent < indent:
                break
            stripped = line.strip()
            if stripped.startswith("- "):
                break
            if ":" not in stripped:
                raise CartridgeSerializationError(f"unparseable workspace YAML line{src}: {line!r}")
            key, _, raw_val = stripped.partition(":")
            raw_val = _strip_inline_comment(raw_val.strip())
            pos += 1
            if raw_val in ("|", "|-", "|+", ">", ">-", ">+"):
                # YAML block scalar — gather all subsequent lines whose
                # indent exceeds ``indent``, strip the leading common
                # indent, and join with newlines (literal ``|`` style)
                # or spaces (folded ``>`` style). The trailing chomp
                # indicator (``-`` strip / ``+`` keep / default clip)
                # only matters for trailing newlines, which we always
                # strip for tool descriptions.
                style = raw_val[0]
                block_lines: list[str] = []
                while pos < len(lines):
                    nl = lines[pos]
                    if nl.strip() == "":
                        block_lines.append("")
                        pos += 1
                        continue
                    nl_indent = len(nl) - len(nl.lstrip())
                    if nl_indent <= indent:
                        break
                    block_lines.append(nl)
                    pos += 1
                # Find common leading indent among non-blank lines.
                non_blank = [bl for bl in block_lines if bl.strip()]
                common = (
                    min(len(bl) - len(bl.lstrip()) for bl in non_blank) if non_blank else indent + 2
                )
                stripped_lines = [bl[common:] if bl else "" for bl in block_lines]
                joiner = "\n" if style == "|" else " "
                value = joiner.join(stripped_lines).rstrip("\n")
                out[key.strip()] = value
            elif not raw_val:
                # Nested block follows.
                out[key.strip()] = parse_block(indent + 2)
            else:
                out[key.strip()] = _scalar(raw_val)
        return out

    def parse_list(indent: int) -> list[Any]:
        nonlocal pos
        out: list[Any] = []
        while pos < len(lines):
            line = lines[pos]
            if not line.strip() or line.lstrip().startswith("#"):
                pos += 1
                continue
            cur_indent = len(line) - len(line.lstrip())
            if cur_indent < indent:
                break
            stripped = line.strip()
            if not stripped.startswith("-"):
                break
            after = _strip_inline_comment(stripped[1:].strip())
            pos += 1
            if not after:
                out.append(parse_block(indent + 2))
            elif _is_inline_single_key_dict(after):
                # ``- key: <flow value or scalar>`` — may be a single-key
                # dict (when no follow-on fields at the same indent),
                # OR the first key of a multi-field block mapping list
                # item:
                #     - key: value
                #       other: thing
                # The follow-on fields live at the indent of the dash
                # plus 2 (the standard YAML alignment for keys after
                # ``- ``). We parse them via ``parse_block(...)`` and
                # merge into the item; absent follow-ons, the result
                # is the same single-key dict the parser produced before.
                key, _, val = after.partition(":")
                item: dict[str, Any] = {}
                if val.strip():
                    item[key.strip()] = _scalar(val.strip())
                else:
                    item[key.strip()] = parse_block(cur_indent + 2)
                # Consume any additional ``key: value`` lines at
                # ``cur_indent + 2``. ``parse_block`` returns ``{}`` if
                # nothing matches, so this is a no-op when the list
                # item is genuinely single-key.
                extra = parse_block(cur_indent + 2)
                if isinstance(extra, dict):
                    item.update(extra)
                out.append(item)
            else:
                out.append(_scalar(after))
        return out

    def _is_inline_single_key_dict(s: str) -> bool:
        # Plain ``key: value`` after the ``- ``. The colon must be
        # outside any quoted string or flow collection. Cheap-and-cheerful
        # detector: look for the FIRST top-level colon, ignoring colons
        # inside ``{...}`` / ``[...]`` / quoted strings.
        depth_curly = 0
        depth_square = 0
        in_str: str | None = None
        for i, ch in enumerate(s):
            if in_str:
                if ch == in_str and (i == 0 or s[i - 1] != "\\"):
                    in_str = None
                continue
            if ch in ('"', "'"):
                in_str = ch
                continue
            if ch == "{":
                depth_curly += 1
            elif ch == "}":
                depth_curly -= 1
            elif ch == "[":
                depth_square += 1
            elif ch == "]":
                depth_square -= 1
            elif ch == ":" and depth_curly == 0 and depth_square == 0:
                # Must be followed by space or end-of-line for YAML key
                if i + 1 == len(s) or s[i + 1] == " ":
                    # Cheap sanity: key part before colon must not
                    # itself contain spaces (would mean it's a sentence,
                    # not a key — see issue with ``- some sentence: blah``).
                    key_part = s[:i].strip()
                    return bool(key_part) and " " not in key_part
        return False

    def _strip_inline_comment(raw: str) -> str:
        """Strip ``# ...`` trailing inline comments from a YAML scalar.

        YAML allows ``key: value  # comment`` and only the ``value``
        is the actual scalar. The custom parser previously kept the
        comment glued to the value, so e.g. ``done_tool: done  # foo``
        registered the literal string ``"done  # foo"`` as the tool
        name — silently breaking sentinel matching.

        A ``#`` is only a comment delimiter when:
        * preceded by whitespace (or appears at start of value), AND
        * not inside a quoted string or flow collection ``[...]`` /
          ``{...}``. We track quote state and bracket depth as we
          scan; ``#`` outside any quote and outside any bracket and
          preceded by whitespace marks the start of the comment.
        """
        depth_curly = 0
        depth_square = 0
        in_str: str | None = None
        for i, ch in enumerate(raw):
            if in_str:
                if ch == in_str and (i == 0 or raw[i - 1] != "\\"):
                    in_str = None
                continue
            if ch in ('"', "'"):
                in_str = ch
                continue
            if ch == "[":
                depth_square += 1
                continue
            if ch == "]":
                depth_square -= 1
                continue
            if ch == "{":
                depth_curly += 1
                continue
            if ch == "}":
                depth_curly -= 1
                continue
            if ch == "#" and depth_curly == 0 and depth_square == 0:
                # Only treat as comment when preceded by whitespace
                # or at start (so URLs / hash strings don't trigger).
                if i == 0 or raw[i - 1] in " \t":
                    return raw[:i].rstrip()
        return raw

    def _scalar(raw: str) -> Any:
        if raw in ("null", "~", ""):
            return None
        if raw == "true":
            return True
        if raw == "false":
            return False
        if raw.startswith(('"', "'")):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Single-quoted YAML strings: trim the quotes by hand
                # since json.loads only accepts double-quoted strings.
                if raw.startswith("'") and raw.endswith("'") and len(raw) >= 2:
                    return raw[1:-1]
                return raw
        if raw.startswith(("[", "{")):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Strict JSON failed (likely YAML flow style with
                # unquoted keys/values). Try the tolerant flow parser.
                parsed, ok = _parse_flow(raw)
                if ok:
                    return parsed
                return raw
        try:
            if "." in raw or "e" in raw or "E" in raw:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    def _parse_flow(raw: str) -> tuple[Any, bool]:
        """Parse a YAML flow scalar (``[a, b]`` or ``{k: v, k2: v2}``).

        Tolerant of unquoted strings (the common YAML idiom). Returns
        ``(value, True)`` on success or ``(None, False)`` on failure.
        Nested flow collections are supported.
        """
        text = raw.strip()
        try:
            value, idx = _parse_flow_value(text, 0)
        except (ValueError, IndexError):
            return None, False
        # All input must be consumed (modulo trailing whitespace).
        if text[idx:].strip():
            return None, False
        return value, True

    def _skip_ws(s: str, i: int) -> int:
        while i < len(s) and s[i] in " \t":
            i += 1
        return i

    def _parse_flow_value(s: str, i: int) -> tuple[Any, int]:
        i = _skip_ws(s, i)
        if i >= len(s):
            raise ValueError("unexpected end of flow value")
        ch = s[i]
        if ch == "[":
            return _parse_flow_list(s, i)
        if ch == "{":
            return _parse_flow_map(s, i)
        if ch in ('"', "'"):
            return _parse_flow_quoted(s, i)
        # Bare scalar: read until , } ] (or end).
        start = i
        while i < len(s) and s[i] not in ",}]":
            i += 1
        token = s[start:i].strip()
        return _scalar(token), i

    def _parse_flow_list(s: str, i: int) -> tuple[list[Any], int]:
        assert s[i] == "["
        out: list[Any] = []
        i = _skip_ws(s, i + 1)
        if i < len(s) and s[i] == "]":
            return out, i + 1
        while i < len(s):
            value, i = _parse_flow_value(s, i)
            out.append(value)
            i = _skip_ws(s, i)
            if i < len(s) and s[i] == ",":
                i = _skip_ws(s, i + 1)
                continue
            if i < len(s) and s[i] == "]":
                return out, i + 1
            raise ValueError(f"expected ',' or ']' in flow list at offset {i}")
        raise ValueError("unterminated flow list")

    def _parse_flow_map(s: str, i: int) -> tuple[dict[str, Any], int]:
        assert s[i] == "{"
        out: dict[str, Any] = {}
        i = _skip_ws(s, i + 1)
        if i < len(s) and s[i] == "}":
            return out, i + 1
        while i < len(s):
            # Parse key (bare or quoted, terminate on ':').
            if s[i] in ('"', "'"):
                key_value, i = _parse_flow_quoted(s, i)
                key = str(key_value)
            else:
                start = i
                while i < len(s) and s[i] != ":":
                    if s[i] in ",}]":
                        raise ValueError("missing ':' in flow map")
                    i += 1
                key = s[start:i].strip()
            if i >= len(s) or s[i] != ":":
                raise ValueError(f"expected ':' in flow map at offset {i}")
            i = _skip_ws(s, i + 1)
            value, i = _parse_flow_value(s, i)
            out[key] = value
            i = _skip_ws(s, i)
            if i < len(s) and s[i] == ",":
                i = _skip_ws(s, i + 1)
                continue
            if i < len(s) and s[i] == "}":
                return out, i + 1
            raise ValueError(f"expected ',' or '}}' in flow map at offset {i}")
        raise ValueError("unterminated flow map")

    def _parse_flow_quoted(s: str, i: int) -> tuple[str, int]:
        quote = s[i]
        i += 1
        start = i
        while i < len(s) and s[i] != quote:
            if s[i] == "\\" and i + 1 < len(s):
                i += 2
                continue
            i += 1
        if i >= len(s):
            raise ValueError("unterminated quoted string")
        text = s[start:i]
        return text, i + 1

    return parse_block(0)
