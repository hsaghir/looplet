"""Dogfood test for lossless cartridge⇄library round-trip of hooks.

The "100% lossless" claim (HOOK_CARTRIDGE_DESIGN.md §9.0) means: a
cartridge → library (preset) → cartridge → library round-trip preserves
hook *behaviour*. This test exercises the linchpin case — a declarative
``kind: lep`` hook — and asserts:

* the serialiser emits a self-contained ``kind: lep`` config (server
  script copied in, command rewritten relative);
* reloading the serialised cartridge reproduces an ``LEPHookAdapter``
  that enforces the *same* policy (rm denied, ls allowed);
* the portability classifier labels it ``portable``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import looplet
from looplet.cartridge import cartridge_to_preset, preset_to_cartridge
from looplet.hook_contract import classify_preset_hooks
from looplet.lep import LEPHookAdapter
from looplet.types import ToolCall

_SRC = str(Path(looplet.__file__).resolve().parent.parent)


def _make_lep_cartridge(root: Path) -> Path:
    """Author a minimal cartridge whose only hook is a ``kind: lep`` policy."""
    import json

    root.mkdir(parents=True, exist_ok=True)
    (root / "cartridge.json").write_text(
        json.dumps({"name": "lep-rt", "schema_version": 2}), encoding="utf-8"
    )
    (root / "config.yaml").write_text("max_steps: 5\n", encoding="utf-8")
    hook_dir = root / "hooks" / "00_policy"
    hook_dir.mkdir(parents=True, exist_ok=True)
    (hook_dir / "config.yaml").write_text(
        "kind: lep\n"
        "server: server.py\n"
        "view:\n"
        "  fields: [tool, args]\n"
        "  fidelity: digest\n"
        "on_failure: fail_closed\n",
        encoding="utf-8",
    )
    (hook_dir / "server.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {_SRC!r})\n"
        "from looplet.lep import LEPServerBase\n"
        "\n"
        "class PolicyServer(LEPServerBase):\n"
        "    def decide(self, slot, view):\n"
        "        if slot == 'check_permission' and view.get('tool') == 'rm':\n"
        "            return {'kind': 'Deny', 'block': 'rm denied'}\n"
        "        return {'kind': 'Continue'}\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    raise SystemExit(PolicyServer().serve())\n",
        encoding="utf-8",
    )
    return root


def _assert_policy(hook: LEPHookAdapter) -> None:
    hook.pre_loop(None, None, None)
    try:
        assert hook.check_permission(ToolCall(tool="rm", args={}), None) is False
        assert hook.check_permission(ToolCall(tool="ls", args={}), None) is True
    finally:
        hook.close()


def test_lep_hook_cartridge_roundtrip(tmp_path):
    cart = _make_lep_cartridge(tmp_path / "src.cartridge")

    # cartridge → library
    preset1 = cartridge_to_preset(cart, strict=True)
    lep1 = [h for h in preset1.hooks if isinstance(h, LEPHookAdapter)]
    assert len(lep1) == 1
    _assert_policy(lep1[0])

    # library → cartridge (the lossless direction under test)
    out = tmp_path / "out.cartridge"
    preset_to_cartridge(preset1, out, strict=True)

    # serialised form is a self-contained declarative kind: lep block
    cfgs = list(out.glob("hooks/*/config.yaml"))
    lep_cfgs = [p for p in cfgs if "kind: lep" in p.read_text()]
    assert len(lep_cfgs) == 1
    lep_dir = lep_cfgs[0].parent
    assert (lep_dir / "server.py").is_file(), "server not copied into snapshot"
    cfg_text = lep_cfgs[0].read_text()
    assert "- server.py" in cfg_text, cfg_text  # command rewritten relative

    # cartridge → library again; behaviour must be identical
    preset2 = cartridge_to_preset(out, strict=True)
    lep2 = [h for h in preset2.hooks if isinstance(h, LEPHookAdapter)]
    assert len(lep2) == 1
    _assert_policy(lep2[0])

    # the classifier labels the LEP hook portable
    labels = dict((name, cls.kind) for name, cls in classify_preset_hooks(preset2))
    assert labels.get("LEPHookAdapter") == "portable"


def test_lep_hook_dir_prefix_does_not_compound(tmp_path):
    """Repeated round-trips must not stack ``NN_`` index prefixes.

    The loader sets ``LEPHookAdapter._cartridge_id`` to the source hook
    dir name (which already carries an index prefix). A naive serialiser
    re-prefixes it, so ``00_policy`` would drift to ``00_00_policy`` and
    then ``00_00_00_policy`` on each cartridge→library→cartridge cycle.
    """
    cart = _make_lep_cartridge(tmp_path / "src.cartridge")
    current = cart
    for i in range(3):
        preset = cartridge_to_preset(current, strict=True)
        out = tmp_path / f"rt_{i}.cartridge"
        preset_to_cartridge(preset, out, strict=True)
        lep_dirs = [
            p.parent.name for p in out.glob("hooks/*/config.yaml") if "kind: lep" in p.read_text()
        ]
        assert lep_dirs == ["00_policy"], lep_dirs
        current = out
