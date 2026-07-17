"""Smoke tests for the EvalCase + load_cases + pytest_param_cases primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet.evals import (
    EvalCase,
    EvalContext,
    assert_evals_pass,
    eval_cli,
    load_cases,
    parametrize_cases,
    pytest_param_cases,
    save_case,
    save_cases,
)


class TestEvalCase:
    def test_round_trip(self) -> None:
        c = EvalCase(
            id="basic",
            task={"description": "x"},
            expected={"tests_passing": True},
            marks=["smoke"],
            notes="seeded",
        )
        assert EvalCase.from_dict(c.to_dict()) == c

    def test_minimal_dict_only_requires_id(self) -> None:
        c = EvalCase.from_dict({"id": "x"})
        assert c.id == "x"
        assert c.task == {}
        assert c.expected == {}
        assert c.marks == []
        assert c.notes == ""

    def test_missing_id_raises(self) -> None:
        with pytest.raises(ValueError, match="id"):
            EvalCase.from_dict({"task": {}})

    def test_to_dict_omits_empty_optionals(self) -> None:
        c = EvalCase(id="bare", task={"description": "x"})
        d = c.to_dict()
        assert "expected" not in d
        assert "marks" not in d
        assert "notes" not in d

    @pytest.mark.parametrize(
        "data",
        [
            {"id": "x", "task": []},
            {"id": "x", "expected": []},
            {"id": "x", "marks": [1]},
            {"id": "x", "notes": 1},
        ],
    )
    def test_malformed_field_types_raise(self, data) -> None:
        with pytest.raises(ValueError, match="EvalCase"):
            EvalCase.from_dict(data)


class TestLoadCases:
    def test_single_json_file_one_case(self, tmp_path: Path) -> None:
        f = tmp_path / "case.json"
        f.write_text(json.dumps({"id": "a", "task": {"description": "x"}}))
        cases = load_cases(f)
        assert [c.id for c in cases] == ["a"]

    def test_single_json_file_list_of_cases(self, tmp_path: Path) -> None:
        f = tmp_path / "case.json"
        f.write_text(json.dumps([{"id": "a", "task": {}}, {"id": "b", "task": {}}]))
        cases = load_cases(f)
        assert [c.id for c in cases] == ["a", "b"]

    def test_jsonl_file(self, tmp_path: Path) -> None:
        f = tmp_path / "cases.jsonl"
        f.write_text(
            json.dumps({"id": "a", "task": {}})
            + "\n# comment\n\n"
            + json.dumps({"id": "b", "task": {}})
            + "\n"
        )
        cases = load_cases(f)
        assert [c.id for c in cases] == ["a", "b"]

    def test_directory_sorted_by_filename(self, tmp_path: Path) -> None:
        (tmp_path / "z.json").write_text(json.dumps({"id": "z", "task": {}}))
        (tmp_path / "a.json").write_text(json.dumps({"id": "a", "task": {}}))
        (tmp_path / "m.jsonl").write_text(json.dumps({"id": "m", "task": {}}) + "\n")
        cases = load_cases(tmp_path)
        # Sorted by file name: a.json < m.jsonl < z.json
        assert [c.id for c in cases] == ["a", "m", "z"]

    def test_duplicate_id_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "dup.json"
        f.write_text(json.dumps([{"id": "a", "task": {}}, {"id": "a", "task": {}}]))
        with pytest.raises(ValueError, match="Duplicate"):
            load_cases(f)

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_cases(tmp_path / "nope")

    def test_malformed_json_raises_with_filename(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("{not valid")
        with pytest.raises(ValueError, match="bad.json"):
            load_cases(f)

    def test_save_case_to_dir_uses_id_as_filename(self, tmp_path: Path) -> None:
        c = EvalCase(id="foo_bar", task={"description": "x"})
        out = save_case(c, tmp_path)
        assert out == tmp_path / "foo_bar.json"
        loaded = load_cases(out)
        assert loaded[0] == c

    def test_save_case_trailing_slash_treated_as_dir(self, tmp_path: Path) -> None:
        """Regression: when ``path`` is a string ending in a path
        separator (the obvious "I want a directory" convention shown
        in docs/evals.md), ``save_case`` must create the directory
        and write ``<dir>/<id>.json`` - even if the directory does
        not yet exist. Previously the trailing-slash path was treated
        as a file because ``Path.exists()`` was False, so the case
        landed at e.g. ``evals/cases`` (a file named "cases" with
        no extension)."""
        c = EvalCase(id="foo_bar", task={})
        target = tmp_path / "evals" / "cases" / ""  # ends with separator
        # Use the raw string form so the trailing slash survives.
        out = save_case(c, str(tmp_path / "evals" / "cases") + "/")
        assert out.is_file()
        assert out.name == "foo_bar.json"
        assert out.parent.is_dir()
        loaded = load_cases(out)
        assert loaded[0] == c

    def test_save_cases_round_trips_via_load_cases(self, tmp_path: Path) -> None:
        cases = [
            EvalCase(id="alpha", task={"q": "1"}, marks=["smoke"]),
            EvalCase(id="beta", task={"q": "2"}, marks=["smoke", "slow"]),
            EvalCase(id="gamma", task={"q": "3"}),
        ]
        target = tmp_path / "evals" / "cases"
        paths = save_cases(cases, target)
        assert [p.name for p in paths] == ["alpha.json", "beta.json", "gamma.json"]
        loaded = load_cases(target)
        assert sorted(c.id for c in loaded) == ["alpha", "beta", "gamma"]

    def test_save_cases_rejects_duplicate_ids(self, tmp_path: Path) -> None:
        cases = [EvalCase(id="dup"), EvalCase(id="dup")]
        with pytest.raises(ValueError, match="duplicate case ids"):
            save_cases(cases, tmp_path / "evals")


class TestPytestParamCases:
    @pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")
    def test_returns_pytest_params_with_ids_and_marks(self) -> None:
        c1 = EvalCase(id="alpha", task={}, marks=["smoke"])
        c2 = EvalCase(id="beta", task={}, marks=["regression", "slow"])
        params = pytest_param_cases([c1, c2])
        assert [p.id for p in params] == ["alpha", "beta"]
        assert params[0].values[0] is c1
        # Marks survive - pytest stores them as MarkDecorator instances
        assert any("smoke" in repr(m) for m in params[0].marks)
        assert any("regression" in repr(m) for m in params[1].marks)

    def test_empty_marks(self) -> None:
        c = EvalCase(id="bare", task={})
        params = pytest_param_cases([c])
        assert params[0].id == "bare"
        assert list(params[0].marks) == []


class TestCasesCli:
    def _setup(self, tmp_path: Path) -> Path:
        (tmp_path / "a.json").write_text(
            json.dumps(
                {
                    "id": "alpha",
                    "task": {"description": "first"},
                    "marks": ["smoke"],
                }
            )
        )
        (tmp_path / "b.json").write_text(
            json.dumps(
                {
                    "id": "beta",
                    "task": {"description": "second"},
                    "marks": ["regression"],
                }
            )
        )
        return tmp_path

    def test_ls_lists_all_cases(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        d = self._setup(tmp_path)
        rc = eval_cli(["cases", "ls", str(d)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out
        assert "first" in out
        assert "smoke" in out

    def test_show_by_id(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        d = self._setup(tmp_path)
        rc = eval_cli(["cases", "show", str(d), "beta"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["id"] == "beta"

    def test_show_unknown_id(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        d = self._setup(tmp_path)
        rc = eval_cli(["cases", "show", str(d), "missing"])
        assert rc == 1
        assert "missing" in capsys.readouterr().out

    def test_show_directory_without_id_when_multiple(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        d = self._setup(tmp_path)
        rc = eval_cli(["cases", "show", str(d)])
        assert rc == 1
        assert "case_id" in capsys.readouterr().out

    def test_ls_missing_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = eval_cli(["cases", "ls", str(tmp_path / "nope")])
        assert rc == 1


class TestParametrizeCases:
    def test_decorator_loads_and_parametrizes(self, tmp_path: Path) -> None:
        (tmp_path / "a.json").write_text(json.dumps({"id": "a", "task": {}, "marks": ["smoke"]}))
        (tmp_path / "b.json").write_text(json.dumps({"id": "b", "task": {}}))

        decorator = parametrize_cases(tmp_path)

        seen: list[EvalCase] = []

        @decorator
        def _placeholder(case: EvalCase) -> None:
            seen.append(case)

        # The decorator stamps a parametrize marker onto the function;
        # extract the params and confirm cases + ids round-trip.
        marks = list(_placeholder.pytestmark)
        assert len(marks) == 1
        params = marks[0].args[1]
        assert [p.id for p in params] == ["a", "b"]
        assert params[0].values[0].id == "a"


def _eval_passing(ctx: EvalContext) -> bool:
    return True


def _eval_failing(ctx: EvalContext) -> bool:
    return False


_eval_failing.__doc__ = "Always fails - for assert_evals_pass tests."


class TestAssertEvalsPass:
    def _ctx(self) -> EvalContext:
        return EvalContext(steps=[])

    def test_passes_when_all_evals_pass(self) -> None:
        assert_evals_pass(self._ctx(), [_eval_passing])

    def test_raises_with_failure_messages(self) -> None:
        with pytest.raises(AssertionError) as exc:
            assert_evals_pass(self._ctx(), [_eval_passing, _eval_failing])
        assert "_eval_failing" in str(exc.value)

    def test_path_arg_uses_eval_discover(self, tmp_path: Path) -> None:
        eval_file = tmp_path / "eval_demo.py"
        eval_file.write_text("def eval_always_pass(ctx):\n    return True\n")
        # Should not raise - discovers and runs the single passing eval.
        assert_evals_pass(self._ctx(), tmp_path)


class TestEvalCliHelp:
    """Regression: `looplet eval --help` must mention the `cases` subcommand."""

    def test_help_mentions_cases_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            eval_cli(["--help"])
        out = capsys.readouterr().out
        assert "cases ls" in out
        assert "cases show" in out
