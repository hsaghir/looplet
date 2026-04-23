"""looplet — composable tool-calling LLM agent harness.

A minimal, composable framework for building tool-calling LLM agent loops.

Only symbols listed in ``__all__`` are imported here. For everything
else, import from the relevant submodule::

    from looplet.backends import OpenAIStreamingBackend, AsyncOpenAIBackend
    from looplet.provenance import ProvenanceSink, replay_loop
    from looplet.cache import CacheBreakDetector, compute_breakpoints
"""

__version__ = "0.1.7"

# ── Public re-exports (one import per submodule, alphabetical) ──────────
# ruff: noqa: F401 — __init__.py intentionally re-exports for `from looplet import X`

from looplet.approval import ApprovalHook
from looplet.backends import AnthropicBackend, OpenAIBackend
from looplet.budget import ContextBudget, ThresholdCompactHook
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
    EvalContext,
    EvalHook,
    EvalResult,
    eval_cli,
    eval_discover,
    eval_mark,
    eval_run,
    eval_run_batch,
)
from looplet.events import EventPayload, LifecycleEvent
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
from looplet.skills import Skill, install_skills
from looplet.stagnation import (
    StagnationHook,
    result_size_fingerprint,
    tool_call_fingerprint,
)
from looplet.streaming import StreamingHook
from looplet.subagent import run_sub_loop
from looplet.telemetry import MetricsCollector, MetricsHook, Tracer, TracingHook
from looplet.testing import AsyncMockLLMBackend, MockLLMBackend
from looplet.tools import BaseToolRegistry, ToolSpec, register_done_tool
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
    "register_done_tool",
    "Skill",
    "install_skills",
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
    "StreamingHook",
    "Tracer",
    "TracingHook",
    "MetricsCollector",
    "MetricsHook",
    # ── EVALS ────────────────────────────────────────────────────
    "EvalHook",
    "EvalContext",
    "EvalResult",
    "eval_discover",
    "eval_run",
    "eval_run_batch",
    "eval_mark",
    "eval_cli",
    # ── TESTING ─────────────────────────────────────────────────
    "MockLLMBackend",
    "AsyncMockLLMBackend",
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
