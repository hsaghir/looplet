"""Public discovery contract for cartridges shipped inside Looplet."""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet import bundled_cartridge_path
from looplet.cartridge import analyse_cartridge


@pytest.mark.parametrize("name", ["agent_factory", "coder", "coder_portable"])
def test_bundled_cartridge_path_resolves_source_checkout(name: str) -> None:
    cartridge = bundled_cartridge_path(name)

    assert cartridge.name == f"{name}.cartridge"
    assert (cartridge / "cartridge.json").is_file()


def test_bundled_portable_coder_has_no_inprocess_blockers() -> None:
    report = analyse_cartridge(bundled_cartridge_path("coder_portable"))

    assert report.profile == "portable"
    assert report.blockers == ()


def test_bundled_cartridge_path_resolves_installed_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from looplet import bundled

    package_root = tmp_path / "site-packages" / "looplet"
    fake_module = package_root / "bundled.py"
    cartridge = package_root / "_bundled" / "coder_portable.cartridge"
    fake_module.parent.mkdir(parents=True)
    cartridge.mkdir(parents=True)
    (cartridge / "cartridge.json").write_text('{"schema_version": 2, "name": "coder_portable"}\n')

    monkeypatch.setattr(bundled, "__file__", str(fake_module))

    assert bundled.bundled_cartridge_path("coder_portable") == cartridge


@pytest.mark.parametrize("name", ["", ".", "../coder", "nested/coder", "nested\\coder"])
def test_bundled_cartridge_path_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="cartridge name"):
        bundled_cartridge_path(name)
