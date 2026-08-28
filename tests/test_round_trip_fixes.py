"""Regression tests for cartridge serializer and loader symmetry."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import looplet
from looplet import cartridge_to_preset, preset_to_cartridge

REPO = Path(__file__).resolve().parents[1]


def _uses_cache(ctx):
    return {"cache": ctx.resources["cache"]}


def test_nonempty_done_tools_are_not_serialized_into_v2_cartridge(tmp_path: Path) -> None:
    preset = looplet.minimal_preset()
    preset.config.done_tools = ["escalate"]

    cartridge = preset_to_cartridge(preset, tmp_path / "agent.cartridge")

    assert any("done_tools is host-only" in warning for warning in cartridge.serialization_warnings)
    assert "done_tools" not in (cartridge.path / "config.yaml").read_text()


def test_host_only_config_is_not_serialized_into_v2_cartridge(tmp_path: Path) -> None:
    preset = looplet.minimal_preset()
    preset.config.approval_handler = lambda _prompt, _choices=None: "yes"

    cartridge = preset_to_cartridge(preset, tmp_path / "agent.cartridge")

    assert any(
        "approval_handler is host-only" in warning for warning in cartridge.serialization_warnings
    )
    assert "approval_handler" not in (cartridge.path / "config.yaml").read_text()


def test_tool_tags_are_not_serialized_into_v2_cartridge(tmp_path: Path) -> None:
    tool = looplet.ToolSpec(
        name="tagged",
        description="Tagged host-side tool.",
        parameters={},
        execute=lambda: {},
        tags=["host-category"],
    )
    preset = looplet.minimal_preset(tools=[tool])

    cartridge = preset_to_cartridge(preset, tmp_path / "agent.cartridge")

    assert any("host-side tags" in warning for warning in cartridge.serialization_warnings)
    assert "tags:" not in (cartridge.path / "tools" / "tagged" / "tool.yaml").read_text()


def test_service_directives_and_disabled_gateway_round_trip(tmp_path: Path) -> None:
    cached_tool = looplet.ToolSpec(
        name="cached",
        description="Use shared state.",
        parameters={},
        execute=_uses_cache,
        requires=["cache"],
    )
    preset = looplet.minimal_preset(tools=[cached_tool])
    preset.mcp_servers = {"tools": {"command": "fake-mcp"}}
    preset.state_services = {"cache": {"command": "fake-state"}}
    preset.llm_gateway_enabled = False
    preset.resources["cache"] = object()

    cartridge = preset_to_cartridge(preset, tmp_path / "agent.cartridge")
    config = (cartridge.path / "config.yaml").read_text()

    assert "mcp_servers:" in config
    assert "state_services:" in config
    assert "llm_gateway: false" in config
    assert not (cartridge.path / "resources" / "cache.py").exists()


def _summary(preset) -> dict:
    """Spec-pinned summary subset (mirrors the conformance tests)."""
    cfg = preset.config
    from looplet import PermissionHook  # noqa: PLC0415

    perm = next((h for h in preset.hooks if isinstance(h, PermissionHook)), None)
    perm_rules = []
    if perm is not None:
        perm_rules = [
            (r.tool, r.decision.value, r.reason, r.arg_matcher is not None)
            for r in perm.engine.rules
        ]
    return {
        "tools": sorted(preset.tools.tool_names),
        "tags": {
            n: sorted(preset.tools._tools[n].tags)
            for n in preset.tools.tool_names
            if preset.tools._tools[n].tags
        },
        "render": {
            n: dict(preset.tools._tools[n].render)
            for n in preset.tools.tool_names
            if preset.tools._tools[n].render
        },
        "perm_rules": perm_rules,
        "done_tool": cfg.done_tool,
        "done_tools": list(cfg.done_tools or []),
        "system_prompt": (cfg.system_prompt or "")[:200],
    }


def test_tool_render_hints_round_trip(tmp_path: Path) -> None:
    """Tool ``render`` hints must survive a round-trip."""
    src = tmp_path / "src.cartridge"
    src.mkdir()
    (src / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (src / "config.yaml").write_text("max_steps: 3\n")
    (src / "prompts").mkdir()
    (src / "prompts" / "system.md").write_text("test")
    done = src / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\n"
        "description: done\n"
        "parameters:\n  summary: { type: string }\n"
        "render:\n  preview: 5\n  max_chars: 800\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )

    p1 = cartridge_to_preset(str(src), strict=True)
    out = tmp_path / "rt.cartridge"
    preset_to_cartridge(p1, out, strict=False)
    p2 = cartridge_to_preset(str(out), strict=True)

    assert p1.tools._tools["done"].render == {"preview": 5, "max_chars": 800}
    assert p2.tools._tools["done"].render == p1.tools._tools["done"].render


def test_permissions_with_contains_matcher_round_trip(tmp_path: Path) -> None:
    """``permissions: deny: contains:`` rules round-trip without losing
    the matcher closure."""
    src = tmp_path / "src.cartridge"
    src.mkdir()
    (src / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (src / "config.yaml").write_text(
        "max_steps: 3\n"
        "permissions:\n"
        "  default: allow\n"
        "  deny:\n"
        "    - tool: bash\n"
        "      contains:\n"
        '        command: "rm -rf"\n'
        '      reason: "destructive shell"\n'
        "    - tool: bash\n"
        "      contains:\n"
        '        command: "sudo"\n'
        '      reason: "privilege escalation"\n'
    )
    (src / "prompts").mkdir()
    (src / "prompts" / "system.md").write_text("test")
    done = src / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n    return {'summary': summary}\n"
    )

    p1 = cartridge_to_preset(str(src), strict=True)
    out = tmp_path / "rt.cartridge"
    preset_to_cartridge(p1, out, strict=False)
    p2 = cartridge_to_preset(str(out), strict=True)

    from looplet import PermissionHook

    perm1 = next(h for h in p1.hooks if isinstance(h, PermissionHook))
    perm2 = next(h for h in p2.hooks if isinstance(h, PermissionHook))
    assert len(perm1.engine.rules) == 2
    assert len(perm2.engine.rules) == len(perm1.engine.rules), (
        f"perm rules lost on round-trip: {len(perm1.engine.rules)} → {len(perm2.engine.rules)}"
    )
    # Each rule still has a working matcher after round-trip.
    for r1, r2 in zip(perm1.engine.rules, perm2.engine.rules):
        assert r2.tool == r1.tool
        assert r2.decision == r1.decision
        assert r2.reason == r1.reason
        assert (r2.arg_matcher is None) == (r1.arg_matcher is None)
        if r1.arg_matcher is not None:
            # Both matchers should agree on a sample input.
            sample = {"command": "rm -rf /"}
            assert r1.arg_matcher(sample) == r2.arg_matcher(sample)


def test_tool_required_resource_written_back(tmp_path: Path) -> None:
    """A resource referenced via ``tool.requires`` is written back to the
    round-tripped cartridge so the reloaded tool can dispatch."""
    src = tmp_path / "src.cartridge"
    src.mkdir()
    (src / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (src / "config.yaml").write_text("max_steps: 3\n")
    (src / "prompts").mkdir()
    (src / "prompts" / "system.md").write_text("test")
    res_dir = src / "resources"
    res_dir.mkdir()
    (res_dir / "my_resource.py").write_text(
        "class MyResource:\n"
        "    def __init__(self): self.value = 42\n"
        "\n"
        "def build(): return MyResource()\n"
    )
    done = src / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\n"
        "description: done\n"
        "parameters:\n  summary: { type: string }\n"
        "requires: [my_resource]\n"
    )
    (done / "execute.py").write_text(
        "def execute(ctx, *, summary):\n"
        "    res = ctx.resources['my_resource']\n"
        "    return {'summary': summary, 'value': res.value if res else None}\n"
    )

    p1 = cartridge_to_preset(str(src), strict=True)
    out = tmp_path / "rt.cartridge"
    preset_to_cartridge(p1, out, strict=False)

    # The round-trip must have written resources/my_resource.py.
    assert (out / "resources" / "my_resource.py").is_file(), (
        "tool's required resource was not written back on round-trip; "
        f"resources/ contents: {list((out / 'resources').iterdir())}"
    )

    # Reload and verify the resource is registered.
    p2 = cartridge_to_preset(str(out), strict=True)
    assert "my_resource" in p2.resources, (
        f"reloaded preset missing my_resource; got {sorted(p2.resources.keys())}"
    )


def test_hook_kwargs_inferred_from_init_when_no_to_config(tmp_path: Path) -> None:
    """A hook whose constructor takes resource kwargs but does NOT
    implement ``to_config()`` should still round-trip those kwargs by
    introspecting the ``__init__`` signature."""
    src = tmp_path / "src.cartridge"
    src.mkdir()
    (src / "cartridge.json").write_text('{"name": "x", "schema_version": 2}\n')
    (src / "config.yaml").write_text("max_steps: 3\n")
    (src / "prompts").mkdir()
    (src / "prompts" / "system.md").write_text("test")
    res_dir = src / "resources"
    res_dir.mkdir()
    (res_dir / "my_log.py").write_text(
        "class MyLog:\n    def __init__(self): self.events = []\n\ndef build(): return MyLog()\n"
    )
    hook_dir = src / "hooks" / "00_AuditHook"
    hook_dir.mkdir(parents=True)
    (hook_dir / "config.yaml").write_text(
        'class_name: AuditHook\nkwargs:\n  log: "@my_log"\n  budget: 100\n'
    )
    (hook_dir / "hook.py").write_text(
        "class AuditHook:\n"
        "    def __init__(self, *, log=None, budget=10):\n"
        "        self.log = log\n"
        "        self.budget = budget\n"
        "    def pre_dispatch(self, *a, **k): return None\n"
        "    def post_dispatch(self, *a, **k): return None\n"
        "    def check_done(self, *a, **k): return None\n"
        "    def should_stop(self, *a, **k): return False\n"
    )
    done = src / "tools" / "done"
    done.mkdir(parents=True)
    (done / "tool.yaml").write_text(
        "name: done\ndescription: done\nparameters:\n  summary: { type: string }\n"
    )
    (done / "execute.py").write_text("def execute(ctx, *, summary): return {'summary': summary}\n")

    p1 = cartridge_to_preset(str(src), strict=True)
    audit_a = next(h for h in p1.hooks if type(h).__name__ == "AuditHook")
    assert audit_a.log is not None  # resource was injected
    assert audit_a.budget == 100

    out = tmp_path / "rt.cartridge"
    preset_to_cartridge(p1, out, strict=False)

    # Read the round-tripped hook config.
    cfg_path = out / "hooks" / "00_AuditHook" / "config.yaml"
    assert cfg_path.is_file()
    cfg_text = cfg_path.read_text()
    assert "budget: 100" in cfg_text, (
        f"hook scalar kwarg 'budget' lost on round-trip; got: {cfg_text}"
    )
    assert "@my_log" in cfg_text, (
        f"hook resource kwarg 'log' should be re-emitted as @my_log; got: {cfg_text}"
    )


def test_round_trip_preserves_summary_for_every_shipped_example() -> None:
    """The seven shipped example cartridges all round-trip with an
    EXACT spec-pinned summary match. Pins the comprehensive integration
    surface - if any single example breaks, this catches it before
    release."""
    failures = []
    for cart in sorted((REPO / "examples").glob("*.cartridge")):
        runtime = {
            "cartridge_root": str(cart),
            "workspace": "/tmp",
            "repo": str(REPO),
            "project_root": str(REPO),
        }
        try:
            p1 = cartridge_to_preset(str(cart), strict=False, runtime=runtime)
        except Exception as e:
            failures.append((cart.name, f"load_a: {e}"))
            continue
        p2 = None
        try:
            s1 = _summary(p1)
            with tempfile.TemporaryDirectory(prefix=f"rt_{cart.name}_") as tmp:
                out = Path(tmp) / "rt.cartridge"
                try:
                    preset_to_cartridge(p1, out, strict=False)
                    p2 = cartridge_to_preset(str(out), strict=False, runtime=runtime)
                except Exception as e:
                    failures.append((cart.name, f"write/load_b: {e}"))
                    continue
                s2 = _summary(p2)
                if s1 != s2:
                    diffs = []
                    for k in sorted(set(s1) | set(s2)):
                        if s1.get(k) != s2.get(k):
                            diffs.append(k)
                    failures.append((cart.name, f"summary differs in: {diffs}"))
        finally:
            if p2 is not None:
                p2.close()
            p1.close()
    assert not failures, (
        f"{len(failures)} cartridge(s) failed lossless round-trip:\n  "
        + "\n  ".join(f"{n}: {m}" for n, m in failures)
    )
