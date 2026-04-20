"""openharness — composable tool-calling LLM agent harness.

A minimal, composable framework for building tool-calling LLM agent loops.

Only symbols listed in ``__all__`` are imported here. For everything
else, import from the relevant submodule::

    from openharness.backends import OpenAIStreamingBackend, AsyncOpenAIBackend
    from openharness.provenance import ProvenanceSink, replay_loop
    from openharness.cache import CacheBreakDetector, compute_breakpoints
"""

__version__ = "0.1.6"

# ── Public re-exports (one import per submodule, alphabetical) ──────────
# ruff: noqa: F401 — __init__.py intentionally re-exports for `from openharness import X`

from openharness.approval import ApprovalHook
from openharness.backends import AnthropicBackend, OpenAIBackend
from openharness.budget import ContextBudget, ThresholdCompactHook
from openharness.cache import CachePolicy
from openharness.checkpoint import FileCheckpointStore
from openharness.compact import (
    CompactOutcome,
    CompactService,
    PruneToolResults,
    SummarizeCompact,
    TruncateCompact,
    compact_chain,
    run_compact,
)
from openharness.conversation import Conversation, Message
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
from openharness.events import EventPayload, LifecycleEvent
from openharness.hook_decision import (
    Allow,
    Block,
    Continue,
    Deny,
    HookDecision,
    InjectContext,
    Stop,
)
from openharness.loop import DomainAdapter, LoopConfig, LoopHook, composable_loop
from openharness.mcp import MCPToolAdapter
from openharness.memory import CallableMemorySource, StaticMemorySource
from openharness.permissions import PermissionEngine, PermissionHook, PermissionRule
from openharness.presets import (
    AgentPreset,
    coding_agent_preset,
    minimal_preset,
    research_agent_preset,
)
from openharness.prompts import preview_prompt
from openharness.provenance import TrajectoryRecorder
from openharness.session import SessionLog
from openharness.skills import Skill
from openharness.streaming import StreamingHook
from openharness.subagent import run_sub_loop
from openharness.tools import BaseToolRegistry, ToolSpec
from openharness.types import (
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
    "Skill",
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
