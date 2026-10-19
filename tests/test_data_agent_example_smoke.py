"""Dogfood tests for the packaged Data Agent example."""

from __future__ import annotations

from pathlib import Path

import pytest

from looplet.examples import data_agent

pytestmark = pytest.mark.smoke


class TestDataAgentExample:
    def test_live_run_uses_shared_provider_defaults(self, tmp_path, monkeypatch, capsys) -> None:
        from looplet import backends

        csv_path = tmp_path / "orders.csv"
        csv_path.write_text("order_id,customer,amount,status\n1,alice,5,cancelled\n")
        backend = data_agent.scripted_llm(str(csv_path))
        selected = []

        def resolve():
            selected.append(backend)
            return backend

        monkeypatch.setattr(backends, "make_backend", resolve)
        monkeypatch.setattr(data_agent, "_make_sample_csv", lambda: csv_path)
        monkeypatch.setattr(data_agent, "CHECKPOINT_DIR", tmp_path / "checkpoints")
        assert data_agent.main(["--auto-approve"]) == 0
        assert selected == [backend]
        assert backend.calls > 0
        assert "# tool protocol: native" in capsys.readouterr().out

    def test_build_tools_uses_decorator_schema_and_helpers(self) -> None:
        registry = data_agent.build_tools()
        info = {tool["name"]: tool for tool in registry.introspect()["tools"]}

        assert list(info) == [
            "describe_csv",
            "head_csv",
            "groupby_count",
            "delete_rows",
            "think",
            "done",
        ]
        assert info["describe_csv"]["parameters"]["required"] == ["path"]
        assert info["head_csv"]["parameters"]["properties"]["n"]["type"] == "integer"
        assert info["head_csv"]["parameters"]["required"] == ["path"]
        assert info["groupby_count"]["parameters"]["required"] == ["path", "column"]
        assert "ctx" not in info["delete_rows"]["parameters"]["properties"]
        assert info["think"]["free"] is True
        assert "summary" in info["done"]["parameters"]["properties"]

    def test_scripted_run_exercises_new_api_end_to_end(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(data_agent, "CHECKPOINT_DIR", tmp_path / "checkpoints")

        rc = data_agent.main(["--scripted", "--auto-approve", "--clean"])

        assert rc == 0
        assert any((tmp_path / "checkpoints").glob("*.json"))
        out = capsys.readouterr().out
        assert "# tool protocol: json-text" in out
        assert "describe_csv" in out
        assert "delete_rows" in out
        assert "done(summary=inspected orders.csv and removed cancellations)" in out

    def test_mock_alias_remains_compatible(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(data_agent, "CHECKPOINT_DIR", tmp_path / "checkpoints")

        rc = data_agent.main(["--mock", "--auto-approve", "--clean"])

        assert rc == 0
        assert any((tmp_path / "checkpoints").glob("*.json"))

    def test_completed_resume_preserves_data_without_repeating_tools(
        self, tmp_path, monkeypatch, capsys
    ) -> None:
        csv_path = tmp_path / "orders.csv"

        def seed():
            csv_path.write_text(
                "order_id,customer,amount,status\n1,alice,42,paid\n2,bob,8,cancelled\n"
            )
            return csv_path

        monkeypatch.setattr(data_agent, "CHECKPOINT_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(data_agent, "_make_sample_csv", seed)
        assert data_agent.main(["--scripted", "--auto-approve"]) == 0
        before = csv_path.read_bytes()
        assert b"cancelled" not in before
        capsys.readouterr()
        assert data_agent.main(["--scripted", "--auto-approve", "--resume"]) == 0
        output = capsys.readouterr().out
        assert "# resuming from step 5" in output
        assert "describe_csv(" not in output
        assert csv_path.read_bytes() == before

    @pytest.mark.parametrize("completed_steps,missing_csv", [(2, False), (4, False), (2, True)])
    def test_incomplete_resume_preserves_source_and_script_position(
        self, tmp_path, monkeypatch, capsys, completed_steps, missing_csv
    ) -> None:
        from looplet import ApprovalHook, LoopConfig, composable_loop
        from looplet.checkpoint import FileCheckpointStore

        csv_path = tmp_path / "original.csv"
        csv_path.write_text("order_id,customer,amount,status\n1,alice,42,paid\n2,bob,8,cancelled\n")
        first_store = FileCheckpointStore(str(tmp_path / "first"))
        iterator = composable_loop(
            llm=data_agent.scripted_llm(str(csv_path)),
            tools=data_agent.build_tools(),
            config=LoopConfig(
                max_steps=10,
                use_native_tools=False,
                checkpoint_dir=str(tmp_path / "first"),
                checkpoint_state=lambda _context: {"csv_path": str(csv_path)},
                approval_handler=lambda _prompt, _options: "yes",
            ),
            hooks=[ApprovalHook()],
        )
        try:
            for _index in range(completed_steps + 1):
                next(iterator)
            checkpoint = first_store.load(f"step_{completed_steps}")
        finally:
            iterator.close()
        assert checkpoint is not None and checkpoint.step_number == completed_steps
        directory = tmp_path / "recovered"
        FileCheckpointStore(str(directory)).save(checkpoint, f"step_{completed_steps}")
        monkeypatch.setattr(data_agent, "CHECKPOINT_DIR", directory)

        def refuse_reseeding():
            pytest.fail("Resume must never recreate the sample CSV")

        monkeypatch.setattr(data_agent, "_make_sample_csv", refuse_reseeding)
        if missing_csv:
            csv_path.unlink()
            before = {path.name: path.read_bytes() for path in directory.glob("*.json")}
            with pytest.raises(SystemExit) as error:
                data_agent.main(["--scripted", "--auto-approve", "--resume"])
            assert error.value.code == 2
            assert {path.name: path.read_bytes() for path in directory.glob("*.json")} == before
            assert not csv_path.exists()
        else:
            assert data_agent.main(["--scripted", "--auto-approve", "--resume"]) == 0
            output = capsys.readouterr().out
            assert f"#{completed_steps + 1} " in output
            assert "describe_csv(" not in output
            assert b"cancelled" not in csv_path.read_bytes()
            terminal = FileCheckpointStore(str(directory)).load_latest()
            assert terminal is not None and terminal.step_number == 5
            assert terminal.termination_reason == "done"

    @pytest.mark.parametrize("arguments", [["--resume"], ["--resume", "--clean"]])
    def test_resume_requires_compatible_existing_checkpoint(
        self, tmp_path, monkeypatch, arguments
    ) -> None:
        directory = tmp_path / "checkpoints"
        directory.mkdir()
        marker = directory / "keep.txt"
        marker.write_text("Existing work")
        monkeypatch.setattr(data_agent, "CHECKPOINT_DIR", directory)
        with pytest.raises(SystemExit) as error:
            data_agent.main(["--scripted", "--auto-approve", *arguments])
        assert error.value.code == 2
        assert marker.read_text() == "Existing work"
