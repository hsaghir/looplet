"""looplet — composable tool-calling LLM agent harness.

A minimal, composable framework for building tool-calling LLM agent loops.

Only symbols listed in ``__all__`` are imported here. For everything
else, import from the relevant submodule::

    from looplet.backends import OpenAIStreamingBackend, AsyncOpenAIBackend
    from looplet.provenance import ProvenanceSink, replay_loop
    from looplet.cache import CacheBreakDetector, compute_breakpoints
"""

__version__ = "0.1.8"

# ── Public re-exports (one import per submodule, alphabetical) ──────────
# ruff: noqa: F401 — __init__.py intentionally re-exports for `from looplet import X`

from looplet.approval import ApprovalHook
from looplet.async_loop import async_composable_loop
from looplet.backends import AnthropicBackend, OpenAIBackend
from looplet.blueprints import (
    AgentBlueprint,
    BlueprintComparison,
    ClaudeSkillCompatibility,
    ComponentBlueprint,
    SourceBlueprint,
    ToolBlueprint,
    blueprint_from_bundle,
    blueprint_from_preset,
    claude_skill_compatibility,
    compare_blueprints,
    export_bundle_to_library_code,
    package_agent_factory_as_bundle,
    wrap_claude_skill_as_bundle,
)
from looplet.budget import ContextBudget, ThresholdCompactHook
from looplet.bundles import (
    BundleCard,
    BundleValidation,
    SkillBundle,
    SkillRuntime,
    discover_skill_bundles,
    load_skill_bundle,
    run_skill_bundle,
    validate_skill_bundle,
)
from looplet.cache import CachePolicy
from looplet.checkpoint import FileCheckpointStore
from looplet.compact import (
    CompactOutcome,
    CompactService,
    PruneToolResults,
    SummarizeCompact,
    TruncateCompact,
    compact_chain,
    run_compact,
)
from looplet.conversation import Conversation, Message
from looplet.done_steps import (
    is_rejected_done,
    iter_done_steps,
    last_accepted_done,
    last_rejected_done,
)
from looplet.evals import (
    EvalCase,
    EvalContext,
    EvalHook,
    EvalResult,
    assert_evals_pass,
    eval_cli,
    eval_discover,
    eval_mark,
    eval_run,
    eval_run_batch,
    load_cases,
    parametrize_cases,
    pytest_param_cases,
    save_case,
    save_cases,
)
from looplet.events import EventPayload, LifecycleEvent
from looplet.harness_snapshot import serialize_harness
from looplet.hook_decision import (
    Allow,
    Block,
    Continue,
    Deny,
    HookDecision,
    InjectContext,
    Stop,
)
from looplet.limits import BudgetWarningHook, PerToolLimitHook
from looplet.loop import DomainAdapter, LoopConfig, LoopHook, composable_loop, emit_event
from looplet.mcp import MCPToolAdapter
from looplet.memory import CallableMemorySource, StaticMemorySource
from looplet.native_tools import (
    NativeToolProbeResult,
    probe_native_tool_support,
    supports_native_tools,
)
from looplet.permissions import (
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    PermissionRule,
)
from looplet.presets import (
    AgentPreset,
    coding_agent_preset,
    minimal_preset,
    research_agent_preset,
)
from looplet.prompts import preview_prompt
from looplet.provenance import ProvenanceSink, TrajectoryRecorder, replay_loop
from looplet.resilient import ResilientBackend, RetryExhausted
from looplet.session import SessionLog
from looplet.skills import (
    FileSkillStore,
    Skill,
    SkillActivationHook,
    SkillCard,
    SkillManager,
    install_skills,
    make_skill_tools,
)
from looplet.stagnation import (
    StagnationHook,
    result_size_fingerprint,
    tool_call_fingerprint,
)
from looplet.streaming import StreamingHook
from looplet.subagent import run_sub_loop
from looplet.telemetry import MetricsCollector, MetricsHook, Tracer, TracingHook
from looplet.testing import AsyncMockLLMBackend, LLMResponsesExhausted, MockLLMBackend
from looplet.tools import (
    BaseToolRegistry,
    ToolSpec,
    excerpt_around_match,
    register_done_tool,
    suggest_similar,
    tool,
    tools_from,
)
from looplet.types import (
    CancelToken,
    DefaultState,
    ErrorKind,
    LLMBackend,
    NativeToolBackend,
    Step,
    ToolCall,
    ToolContext,
    ToolError,
    ToolResult,
    ToolValidationError,
)
from looplet.workspace import (
    Workspace,
    WorkspaceLayout,
    WorkspaceSerializationError,
    preset_to_workspace,
    resource_ref_for,
    workspace_to_preset,
)

__all__ = [
    # ── ESSENTIALS (what you need for your first agent) ──────────
    "__version__",
    "composable_loop",
    "async_composable_loop",
    "LoopConfig",
    "LoopHook",
    "Step",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "BaseToolRegistry",
    "register_done_tool",
    "Skill",
    "SkillCard",
    "SkillBundle",
    "SkillRuntime",
    "BundleCard",
    "BundleValidation",
    "AgentBlueprint",
    "SourceBlueprint",
    "ToolBlueprint",
    "ComponentBlueprint",
    "BlueprintComparison",
    "ClaudeSkillCompatibility",
    "FileSkillStore",
    "SkillManager",
    "SkillActivationHook",
    "install_skills",
    "make_skill_tools",
    "discover_skill_bundles",
    "load_skill_bundle",
    "validate_skill_bundle",
    "run_skill_bundle",
    "blueprint_from_bundle",
    "blueprint_from_preset",
    "compare_blueprints",
    "export_bundle_to_library_code",
    "package_agent_factory_as_bundle",
    "claude_skill_compatibility",
    "wrap_claude_skill_as_bundle",
    "DefaultState",
    "LLMBackend",
    "HookDecision",
    "Allow",
    "Block",
    "Continue",
    "Deny",
    "Stop",
    "InjectContext",
    "preview_prompt",
    "last_accepted_done",
    "last_rejected_done",
    "iter_done_steps",
    "is_rejected_done",
    # ── BACKENDS ─────────────────────────────────────────────────
    "OpenAIBackend",
    "AnthropicBackend",
    "NativeToolBackend",
    "MCPToolAdapter",
    "NativeToolProbeResult",
    "probe_native_tool_support",
    "supports_native_tools",
    # ── CONTEXT MANAGEMENT ──────────────────────────────────────
    "CompactService",
    "CompactOutcome",
    "TruncateCompact",
    "SummarizeCompact",
    "PruneToolResults",
    "compact_chain",
    "run_compact",
    "ContextBudget",
    "ThresholdCompactHook",
    "CachePolicy",
    "StaticMemorySource",
    "CallableMemorySource",
    # ── APPROVAL / PERMISSIONS ──────────────────────────────────
    "ApprovalHook",
    "PermissionEngine",
    "PermissionHook",
    "PermissionRule",
    "PermissionDecision",
    # ── CHECKPOINTS ─────────────────────────────────────────────
    "FileCheckpointStore",
    # ── OBSERVABILITY ───────────────────────────────────────────
    "TrajectoryRecorder",
    "ProvenanceSink",
    "replay_loop",
    "serialize_harness",
    "Workspace",
    "WorkspaceLayout",
    "WorkspaceSerializationError",
    "preset_to_workspace",
    "resource_ref_for",
    "workspace_to_preset",
    "StreamingHook",
    "Tracer",
    "TracingHook",
    "MetricsCollector",
    "MetricsHook",
    # ── EVALS ────────────────────────────────────────────────────
    "EvalHook",
    "EvalContext",
    "EvalResult",
    "EvalCase",
    "assert_evals_pass",
    "eval_discover",
    "eval_run",
    "eval_run_batch",
    "eval_mark",
    "eval_cli",
    "load_cases",
    "parametrize_cases",
    "save_case",
    "save_cases",
    "pytest_param_cases",
    # ── TESTING ─────────────────────────────────────────────────
    "MockLLMBackend",
    "AsyncMockLLMBackend",
    "LLMResponsesExhausted",
    # ── RESILIENCE & CONTROL ────────────────────────────────────
    "ResilientBackend",
    "RetryExhausted",
    "StagnationHook",
    "tool_call_fingerprint",
    "result_size_fingerprint",
    "PerToolLimitHook",
    "BudgetWarningHook",
    # ── ADVANCED (power users import from submodules directly) ──
    "DomainAdapter",
    "emit_event",
    "LifecycleEvent",
    "EventPayload",
    "CancelToken",
    "ToolContext",
    "ToolError",
    "ErrorKind",
    "ToolValidationError",
    "suggest_similar",
    "excerpt_around_match",
    "tool",
    "tools_from",
    "SessionLog",
    "Conversation",
    "Message",
    "run_sub_loop",
    # ── PRESETS (one-liner agent setup) ─────────────────────────
    "AgentPreset",
    "coding_agent_preset",
    "research_agent_preset",
    "minimal_preset",
]
