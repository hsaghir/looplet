"""Result-returning and explicit streaming entry points over preset execution."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Mapping
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from looplet.bundles import SkillBundle, SkillRuntime, load_skill_bundle
from looplet.presets import AgentPreset
from looplet.types import RunResult, Step

AgentSource = AgentPreset | SkillBundle | str | Path | Callable[[], AgentPreset]

__all__ = ["RunStream", "AsyncRunStream", "run", "stream", "run_async", "stream_async"]


def _build_preset(
    source: AgentSource,
    workspace: str | Path | None,
    runtime: Mapping[str, Any] | SkillRuntime | None,
) -> tuple[AgentPreset, bool, Callable[[], AbstractContextManager[Any]]]:
    if isinstance(source, AgentPreset) or callable(source):
        if workspace is not None or runtime is not None:
            raise ValueError("workspace/runtime apply to paths and bundles; configure your factory")
        preset = source if isinstance(source, AgentPreset) else source()
        if not isinstance(preset, AgentPreset):
            raise TypeError("preset factory must return AgentPreset")
        return preset, not isinstance(source, AgentPreset), nullcontext

    bundle = source if isinstance(source, SkillBundle) else None
    if bundle is None:
        if not isinstance(source, (str, Path)):
            raise TypeError("source must be a preset, bundle, path, or preset factory")
        path = Path(source).expanduser().resolve()
        if (path / "SKILL.md").is_file() and (path / "cartridge.json").is_file():
            raise ValueError("ambiguous source; pass cartridge.json or SKILL.md explicitly")
        if path.name == "SKILL.md" or (path / "SKILL.md").is_file():
            bundle = load_skill_bundle(path)
        else:
            from looplet.cartridge import cartridge_to_preset

            if isinstance(runtime, SkillRuntime):
                raise TypeError("cartridge runtime must be a mapping; SkillRuntime is for bundles")
            options = dict(runtime or {})
            if workspace is not None:
                options["project_root"] = str(Path(workspace).expanduser().resolve())
            if path.name in ("cartridge.json", "workspace.json"):
                path = path.parent
            return cartridge_to_preset(path, runtime=options), True, nullcontext

    if isinstance(runtime, SkillRuntime):
        if workspace is not None:
            raise ValueError("set workspace on SkillRuntime instead of passing it twice")
        skill_runtime = runtime
    else:
        skill_runtime = SkillRuntime(workspace=workspace or ".", options=dict(runtime or {}))
    preset = bundle.build_preset(skill_runtime)
    if not isinstance(preset, AgentPreset):
        raise TypeError("bundle build must return AgentPreset")
    return preset, True, bundle.import_context


class _ExecutionResult:
    def __init__(self, preset: AgentPreset, *, owns_preset: bool) -> None:
        self.preset = preset
        self._owns_preset = owns_preset
        self._steps: list[Step] = []
        self._result: RunResult | None = None

    @property
    def result(self) -> RunResult:
        """Return the final execution summary, never an evaluation verdict."""
        if self._result is None:
            raise RuntimeError("run is still open; consume the stream or close it first")
        return self._result

    def _finish(self) -> None:
        shutdown = self.preset.close() if self._owns_preset else None
        self._result = RunResult.from_state(self.preset.state, steps=self._steps)
        errors = getattr(shutdown, "errors", ())
        if errors:
            self._result.metadata["shutdown_errors"] = list(errors)


class RunStream(_ExecutionResult, Iterator[Step]):
    """A closeable step stream with a result after exhaustion or explicit close.

    Use a context manager when stopping iteration early. Presets built from
    paths, bundles, and factories are owned; supplied live presets are not.
    """

    def __init__(
        self,
        preset: AgentPreset,
        iterator: Any,
        *,
        owns_preset: bool,
        import_context: Callable[[], AbstractContextManager[Any]],
    ) -> None:
        super().__init__(preset, owns_preset=owns_preset)
        self._driver = iterator

        def drive() -> Any:
            with import_context():
                return (yield from iterator)

        self._iterator = drive()

    def __iter__(self) -> RunStream:
        return self

    def __next__(self) -> Step:
        if self._result is not None:
            raise StopIteration
        try:
            step = next(self._iterator)
        except BaseException:
            self.close()
            raise
        self._steps.append(step)
        return step

    def close(self) -> None:
        """Finalize the driver and release only resources owned by this stream."""
        if self._result is not None:
            return
        try:
            self._iterator.close()
        finally:
            try:
                self._driver.close()
            finally:
                self._finish()

    def __enter__(self) -> RunStream:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _setup_driver(
    source: AgentSource,
    *,
    task: str | Mapping[str, Any] | None = None,
    llm: Any = None,
    workspace: str | Path | None = None,
    runtime: Mapping[str, Any] | SkillRuntime | None = None,
    async_run: bool = False,
    **kwargs: Any,
) -> tuple[AgentPreset, Any, bool, Callable[[], AbstractContextManager[Any]]]:
    preset, owns_preset, import_context = _build_preset(source, workspace, runtime)
    try:
        if llm is None:
            if async_run:
                raise ValueError("async execution requires an explicit llm backend")
            from looplet.backends import make_backend

            llm = make_backend()
        loop_task = (
            {"description": task}
            if isinstance(task, str)
            else dict(task)
            if task is not None
            else None
        )
        with import_context():
            driver = preset.run_async if async_run else preset.run
            iterator = driver(llm, task=loop_task, **kwargs)
        return preset, iterator, owns_preset, import_context
    except BaseException:
        if owns_preset:
            preset.close()
        raise


def stream(
    source: AgentSource,
    task: str | Mapping[str, Any] | None = None,
    *,
    llm: Any = None,
    workspace: str | Path | None = None,
    runtime: Mapping[str, Any] | SkillRuntime | None = None,
    **kwargs: Any,
) -> RunStream:
    """Expose one fresh execution as a closeable step stream.

    ``task`` accepts text or a mapping; ``llm`` defaults to the environment
    backend resolver. ``workspace`` and ``runtime`` apply to paths/bundles.
    Remaining arguments go to ``AgentPreset.run``. Setup and execution
    errors propagate; ordinary stops are available in ``result``.
    """
    preset, iterator, owns_preset, import_context = _setup_driver(
        source, task=task, llm=llm, workspace=workspace, runtime=runtime, async_run=False, **kwargs
    )
    return RunStream(preset, iterator, owns_preset=owns_preset, import_context=import_context)


def run(
    source: AgentSource,
    task: str | Mapping[str, Any] | None = None,
    *,
    llm: Any = None,
    workspace: str | Path | None = None,
    runtime: Mapping[str, Any] | SkillRuntime | None = None,
    **kwargs: Any,
) -> RunResult:
    """Execute a cartridge, bundle, factory, or caller-owned preset to a result."""
    with stream(source, task, llm=llm, workspace=workspace, runtime=runtime, **kwargs) as execution:
        for _step in execution:
            pass
        return execution.result


class AsyncRunStream(_ExecutionResult, AsyncIterator[Step]):
    """Async counterpart of :class:`RunStream`; use ``async with`` on early exit."""

    def __init__(
        self,
        preset: AgentPreset,
        iterator: Any,
        *,
        owns_preset: bool,
        import_context: Callable[[], AbstractContextManager[Any]],
    ) -> None:
        super().__init__(preset, owns_preset=owns_preset)
        self._driver = iterator

        async def drive() -> AsyncGenerator[Step, None]:
            with import_context():
                async for step in iterator:
                    yield step

        self._iterator = drive()

    def __aiter__(self) -> AsyncRunStream:
        return self

    async def __anext__(self) -> Step:
        if self._result is not None:
            raise StopAsyncIteration
        try:
            step = await self._iterator.__anext__()
        except BaseException:
            await self.aclose()
            raise
        self._steps.append(step)
        return step

    async def aclose(self) -> None:
        """Finalize the async driver and release only owned resources."""
        if self._result is not None:
            return
        try:
            await self._iterator.aclose()
        finally:
            try:
                await self._driver.aclose()
            finally:
                self._finish()

    async def __aenter__(self) -> AsyncRunStream:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()


def stream_async(
    source: AgentSource,
    task: str | Mapping[str, Any] | None = None,
    *,
    llm: Any,
    workspace: str | Path | None = None,
    runtime: Mapping[str, Any] | SkillRuntime | None = None,
    **kwargs: Any,
) -> AsyncRunStream:
    """Expose async steps with the same source, task, and ownership contract."""
    preset, iterator, owns_preset, import_context = _setup_driver(
        source, task=task, llm=llm, workspace=workspace, runtime=runtime, async_run=True, **kwargs
    )
    return AsyncRunStream(preset, iterator, owns_preset=owns_preset, import_context=import_context)


async def run_async(
    source: AgentSource,
    task: str | Mapping[str, Any] | None = None,
    *,
    llm: Any,
    workspace: str | Path | None = None,
    runtime: Mapping[str, Any] | SkillRuntime | None = None,
    **kwargs: Any,
) -> RunResult:
    """Consume async steps and return the shared host-facing result."""
    async with stream_async(
        source, task, llm=llm, workspace=workspace, runtime=runtime, **kwargs
    ) as execution:
        async for _step in execution:
            pass
        return execution.result
