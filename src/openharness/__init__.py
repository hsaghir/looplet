"""openharness — composable tool-calling LLM agent harness.

A minimal, composable framework for building tool-calling LLM agent loops.
"""
# ruff: noqa: F401 — __init__.py intentionally re-exports for `from openharness import X`

__version__ = "0.1.6"

from openharness.approval import ApprovalHook, ApprovalRequest
from openharness.backends import (
    AnthropicBackend,
    AnthropicStreamingBackend,
    AsyncAnthropicBackend,
    AsyncOpenAIBackend,
    OpenAIBackend,
    OpenAIStreamingBackend,
)
from openharness.budget import (
    BudgetTelemetry,
    ContextBudget,
    ThresholdCompactHook,
    classify_tier,
)
from openharness.cache import (
    CacheBreakDetector,
    CacheBreakpoint,
    CacheControl,
    CachePolicy,
    compute_breakpoints,
)
from openharness.checkpoint import (
    Checkpoint,
    CheckpointHook,
    CheckpointStore,
    FileCheckpointStore,
    resume_loop_state,
)
from openharness.compact import (
    CompactOutcome,
    CompactService,
    PruneToolResults,
    SummarizeCompact,
    TruncateCompact,
    compact_chain,
    run_compact,
)
from openharness.context import ContextPressureHook
from openharness.conversation import (
    ContentBlock,
    Conversation,
    Message,
    MessageRole,
)
from openharness.evals import (
    EvalContext,
    EvalHook,
    EvalResult,
    eval_cli,
    eval_discover,
    eval_mark,
    eval_run,
    eval_run_batch,
)
from openharness.events import LIFECYCLE_EVENTS, EventPayload, LifecycleEvent  # noqa: F401
from openharness.flags import FLAGS
from openharness.history import HistoryRecorder  # noqa: F401
from openharness.hook_decision import (
    Allow,
    Block,
    Continue,
    Deny,
    HookDecision,
    InjectContext,
    Stop,
    normalize_hook_return,
)
from openharness.loop import DomainAdapter, LoopConfig, LoopHook, composable_loop
from openharness.mcp import MCPToolAdapter
from openharness.memory import (
    CallableMemorySource,
    PersistentMemorySource,
    StaticMemorySource,
    render_memory,
)
from openharness.parse import (  # noqa: F401
    parse_multi_tool_calls,
    parse_native_tool_use,
    parse_tool_call,
)
from openharness.permissions import (
    PermissionDecision,
    PermissionEngine,
    PermissionHook,
    PermissionOutcome,
    PermissionRule,
)
from openharness.prompts import build_prompt, preview_prompt
from openharness.provenance import (
    AsyncRecordingLLMBackend,
    ProvenanceSink,
    RecordingLLMBackend,
    Trajectory,
    TrajectoryRecorder,
    replay_loop,
)
from openharness.recovery import (
    RecoveryRegistry,
    build_default_registry,
)
from openharness.router import (
    CostTracker,
    FallbackRouter,
    ModelProfile,
    ModelRouter,
    RoutingLLMBackend,
    SimpleRouter,
)
from openharness.scaffolding import (
    LLMResult,
    build_parse_recovery_prompt,  # noqa: F401
    emergency_truncate,  # noqa: F401
    estimate_prompt_tokens,
    estimate_tokens,  # noqa: F401
    is_context_oversized,  # noqa: F401
    llm_call_with_retry,  # noqa: F401
    trim_results,  # noqa: F401
    truncate_tool_result,  # noqa: F401
)
from openharness.session import LogEntry, SessionLog
from openharness.skills import Skill
from openharness.streaming import (
    CallbackEmitter,
    CompositeEmitter,
    ContextPressureEvent,  # noqa: F401
    Event,
    EventEmitter,
    HookEvent,  # noqa: F401
    LLMCallEndEvent,  # noqa: F401
    LLMCallStartEvent,  # noqa: F401
    LLMChunkEvent,  # noqa: F401
    LoopEndEvent,  # noqa: F401
    LoopStartEvent,  # noqa: F401
    QueueEmitter,
    RecoveryEvent,  # noqa: F401
    StepEndEvent,  # noqa: F401
    StepStartEvent,  # noqa: F401
    StreamingHook,
    ToolDispatchEvent,  # noqa: F401
    ToolResultEvent,  # noqa: F401
)
from openharness.subagent import clone_tools_excluding, run_sub_loop
from openharness.telemetry import MetricsCollector, MetricsHook, Span, Tracer, TracingHook
from openharness.tools import BaseToolRegistry, ToolSpec, register_think_tool
from openharness.types import (
    AgentState,
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
)
from openharness.validation import (
    DoneValidator,
    FieldSpec,  # noqa: F401
    OutputSchema,
    SimpleDoneValidator,
    ValidatingToolRegistry,
    ValidationResult,  # noqa: F401
    validate_args,
)

__all__ = [
    # ── ESSENTIALS (what you need for your first agent) ──────────
    "__version__",
    "composable_loop",
    "LoopConfig",
    "LoopHook",
    "Step",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "BaseToolRegistry",
    "Skill",
    "DefaultState",
    "LLMBackend",
    "HookDecision",
    "Allow",
    "Deny",
    "Block",
    "Stop",
    "InjectContext",
    "preview_prompt",
    # ── BACKENDS ─────────────────────────────────────────────────
    "OpenAIBackend",
    "AnthropicBackend",
    "MCPToolAdapter",
    # ── CONTEXT MANAGEMENT ──────────────────────────────────────
    "CompactService",
    "TruncateCompact",
    "SummarizeCompact",
    "PruneToolResults",
    "compact_chain",
    "ContextBudget",
    "ThresholdCompactHook",
    "StaticMemorySource",
    "CallableMemorySource",
    # ── APPROVAL / PERMISSIONS ──────────────────────────────────
    "ApprovalHook",
    "PermissionEngine",
    "PermissionHook",
    "PermissionRule",
    # ── CHECKPOINTS ─────────────────────────────────────────────
    "FileCheckpointStore",
    # ── OBSERVABILITY ───────────────────────────────────────────
    "TrajectoryRecorder",
    "StreamingHook",
    # ── EVALS ────────────────────────────────────────────────────
    "EvalHook",
    "EvalContext",
    "EvalResult",
    "eval_discover",
    "eval_run",
    "eval_run_batch",
    "eval_mark",
    "eval_cli",
    # ── ADVANCED (power users import from submodules directly) ──
    "DomainAdapter",
    "LoopHook",
    "LifecycleEvent",
    "EventPayload",
    "CancelToken",
    "ToolContext",
    "ToolError",
    "ErrorKind",
    "SessionLog",
    "Conversation",
    "Message",
    "run_sub_loop",
]
