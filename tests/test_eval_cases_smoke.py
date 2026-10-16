"""Smoke tests for the EvalCase + load_cases + pytest_param_cases primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from looplet.evals import (
    EvalCase,
    EvalContext,
    EvalResult,
    assert_evals_pass,
    eval_cli,
    eval_mark,
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

    @pytest.mark.parametrize(
        "misspelled, correct",
        [("expectd", "expected"), ("taks", "task"), ("mark", "marks"), ("note", "notes")],
    )
    def test_unknown_field_suggests_correction(self, misspelled, correct) -> None:
        with pytest.raises(ValueError) as failure:
            EvalCase.from_dict({"id": "example", misspelled: {}})
        assert f"Unknown EvalCase field {misspelled!r}" in str(failure.value)
        assert f"did you mean {correct!r}?" in str(failure.value)

    def test_unknown_field_without_suggestion_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown EvalCase field 'extra'"):
            EvalCase.from_dict({"id": "example", "extra": True})

    @pytest.mark.parametrize("data", [None, 17, "id", [], True])
    def test_non_object_raises_value_error(self, data) -> None:
        with pytest.raises(ValueError, match="EvalCase must be an object"):
            EvalCase.from_dict(data)

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

    @pytest.mark.parametrize("suffix", ["json", "jsonl"])
    @pytest.mark.parametrize(
        "data, message",
        [
            (None, "EvalCase must be an object, got NoneType"),
            (17, "EvalCase must be an object, got int"),
            ("id", "EvalCase must be an object, got str"),
            ({}, "EvalCase requires 'id'"),
            ({"id": "broken", "task": []}, "EvalCase task must be an object"),
            ({"id": "broken", "expectd": {}}, "Unknown EvalCase field 'expectd'"),
        ],
    )
    def test_invalid_entry_reports_source(self, tmp_path: Path, suffix, data, message) -> None:
        path = tmp_path / f"cases.{suffix}"
        if suffix == "jsonl":
            path.write_text('{"id": "valid"}\n# comment\n\n' + json.dumps(data) + "\n")
            location = f"{path}:4:"
        else:
            path.write_text(json.dumps([{"id": "valid"}, data]))
            location = f"{path}: entry 2:"
        with pytest.raises(ValueError) as failure:
            load_cases(path)
        assert location in str(failure.value)
        assert message in str(failure.value)

    def test_invalid_single_object_reports_source(self, tmp_path: Path) -> None:
        path = tmp_path / "case.json"
        path.write_text(json.dumps({"id": "broken", "task": []}))
        with pytest.raises(ValueError) as failure:
            load_cases(path)
        assert str(path) in str(failure.value)
        assert "EvalCase task must be an object" in str(failure.value)

    @pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
    def test_jsonl_preserves_unicode_in_strings(self, tmp_path: Path, separator) -> None:
        path = tmp_path / "cases.jsonl"
        goal = f"first{separator}second"
        path.write_text(json.dumps({"id": "valid", "task": {"goal": goal}}, ensure_ascii=False))
        assert load_cases(path)[0].task["goal"] == goal

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

    @pytest.mark.parametrize(
        "marks, expected", [(["smoke"], ["alpha"]), (["smoke", "regression"], ["alpha", "beta"])]
    )
    def test_ls_filters_case_marks(self, tmp_path, capsys, marks, expected):
        directory = self._setup(tmp_path)
        arguments = ["cases", "ls", str(directory)]
        for mark in marks:
            arguments.extend(["--case-mark", mark])

        assert eval_cli(arguments) == 0
        rendered = capsys.readouterr().out
        for case_id in ("alpha", "beta"):
            assert (case_id in rendered) == (case_id in expected)
        assert f"{len(expected)} case(s)" in rendered

    def test_ls_rejects_unknown_case_mark(self, tmp_path, capsys):
        directory = self._setup(tmp_path)
        assert eval_cli(["cases", "ls", str(directory), "--case-mark", "typo"]) == 1
        assert "Unknown eval case mark" in capsys.readouterr().out

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

    @pytest.mark.parametrize("action", ["ls", "show", "check"])
    @pytest.mark.parametrize("suffix, location", [("json", "entry 2"), ("jsonl", ":4:")])
    @pytest.mark.parametrize(
        "data, message",
        [
            (None, "EvalCase must be an object, got NoneType"),
            ({"id": "broken", "expectd": {}}, "Unknown EvalCase field 'expectd'"),
        ],
    )
    def test_invalid_entry_reports_actionable_error(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        action,
        suffix,
        location,
        data,
        message,
    ) -> None:
        definitions = tmp_path / "evals"
        assert eval_cli(["init", str(definitions)]) == 0
        capsys.readouterr()
        path = definitions / "cases" / f"bad.{suffix}"
        if suffix == "jsonl":
            path.write_text('{"id": "valid"}\n# comment\n\n' + json.dumps(data) + "\n")
        else:
            path.write_text(json.dumps([{"id": "valid"}, data]))
        arguments = (
            ["check", str(definitions), "--json"]
            if action == "check"
            else ["cases", action, str(definitions)]
        )
        assert eval_cli(arguments) == 1
        output = capsys.readouterr()
        assert "Traceback" not in output.out + output.err
        error = json.loads(output.out)["errors"][0] if action == "check" else output.out
        assert str(path) in error
        assert location in error
        assert message in error


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

    def test_rejects_empty_suite_and_empty_filter(self) -> None:
        with pytest.raises(AssertionError, match="No graders selected"):
            assert_evals_pass(self._ctx(), [])
        with pytest.raises(AssertionError, match="No graders selected"):
            assert_evals_pass(self._ctx(), [_eval_passing], include=["unknown"])
        assert_evals_pass(self._ctx(), [], version=1)

    def test_metrics_alone_do_not_satisfy_a_gate(self) -> None:
        def eval_metrics(ctx):
            return EvalResult(metrics={"latency_ms": 12.0})

        with pytest.raises(AssertionError, match="no verdict-producing grader"):
            assert_evals_pass(self._ctx(), [eval_metrics])

    def test_raises_with_failure_messages(self) -> None:
        with pytest.raises(AssertionError) as exc:
            assert_evals_pass(self._ctx(), [_eval_passing, _eval_failing])
        assert "_eval_failing" in str(exc.value)

    def test_path_arg_uses_eval_discover(self, tmp_path: Path) -> None:
        eval_file = tmp_path / "eval_demo.py"
        eval_file.write_text("def eval_always_pass(ctx):\n    return True\n")
        # Should not raise - discovers and runs the single passing eval.
        assert_evals_pass(self._ctx(), tmp_path)

    def test_metric_only_result_is_not_a_release_gate(self) -> None:
        def eval_cost(ctx):
            return {"steps": 20.0}

        assert_evals_pass(self._ctx(), [_eval_passing, eval_cost])

    def test_required_metric_only_result_fails(self) -> None:
        @eval_mark("required")
        def eval_gate(ctx):
            return EvalResult(metrics={"steps": 20.0})

        with pytest.raises(AssertionError, match="eval_gate"):
            assert_evals_pass(self._ctx(), [eval_gate])

    def test_optional_judge_without_backend_is_neutral(self) -> None:
        def eval_judge(ctx, llm):
            return True

        assert_evals_pass(self._ctx(), [_eval_passing, eval_judge])

    def test_required_judge_without_backend_fails(self) -> None:
        @eval_mark("required")
        def eval_judge(ctx, llm):
            return True

        with pytest.raises(AssertionError, match="eval_judge"):
            assert_evals_pass(self._ctx(), [eval_judge])

    def test_filtered_required_grader_fails(self) -> None:
        @eval_mark("required", "release")
        def eval_gate(ctx):
            return True

        with pytest.raises(AssertionError, match="eval_gate: required grader missing"):
            assert_evals_pass(self._ctx(), [eval_gate], exclude=["release"])

    def test_numeric_score_below_pass_boundary_fails(self) -> None:
        def eval_score(ctx):
            return 0.4

        with pytest.raises(AssertionError, match="eval_score"):
            assert_evals_pass(self._ctx(), [eval_score])


class TestEvalCliHelp:
    """Regression: `looplet eval --help` must mention the `cases` subcommand."""

    def test_help_mentions_cases_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit):
            eval_cli(["--help"])
        out = capsys.readouterr().out
        assert "cases ls" in out
        assert "cases show" in out
