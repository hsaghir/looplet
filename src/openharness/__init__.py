"""openharness — composable tool-calling LLM agent harness.

Extracted from cadence. Provides the inner agent loop, hooks, context
management, streaming, checkpoints, recovery, and LLM backends.
"""

__version__ = "0.1.5"

from openharness.async_loop import AsyncLLMBackend, AsyncLoopHook, SyncToAsyncAdapter, async_composable_loop
from openharness.checkpoint import (
    Checkpoint,
    CheckpointHook,
    CheckpointStore,
    FileCheckpointStore,
    resume_loop_state,
)
from openharness.context import ContextManagerHook
from openharness.conversation import Conversation, DefaultSummarizer, Message, MessageRole
from openharness.flags import FLAGS, HARNESS_FLAGS
from openharness.loop import LoopConfig, LoopHook, composable_loop
from openharness.parse import parse_multi_tool_calls, parse_native_tool_use, parse_tool_call
from openharness.prompts import build_prompt
from openharness.recovery import (
    FailureScenario,
    RecoveryAction,
    RecoveryRecipe,
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
    PARSE_RECOVERY_MAX,
    MAX_LLM_RETRIES,
    TOOL_RESULT_MAX_CHARS,
    TOOL_RESULT_MAX_ROWS,
    DiminishingReturnsTracker,
    LLMResult,
    StepProgressTracker,
    build_parse_recovery_prompt,
    compress_session_log,
    enforce_result_budget,
    estimate_prompt_tokens,
    estimate_tokens,
    llm_call_with_retry,
    reactive_compact,
    should_compress_context,
    truncate_tool_result,
)
from openharness.session import LogEntry, SessionLog, InvestigationLog
from openharness.streaming import (
    CallbackEmitter,
    CompositeEmitter,
    Event,
    EventEmitter,
    HookEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LLMChunkEvent,
    LoopEndEvent,
    LoopStartEvent,
    QueueEmitter,
    RecoveryEvent,
    StepEndEvent,
    StepStartEvent,
    StreamingHook,
    ToolDispatchEvent,
    ToolResultEvent,
)
from openharness.backends import (
    OpenAIBackend,
    OpenAIStreamingBackend,
    AnthropicBackend,
    AnthropicStreamingBackend,
    AsyncOpenAIBackend,
    AsyncAnthropicBackend,
)
from openharness.subagent import run_sub_loop, _clone_tools_excluding
from openharness.telemetry import MetricsCollector, MetricsHook, Span, Tracer, TracingHook
from openharness.tools import BaseToolRegistry, ToolSpec, register_think_tool
from openharness.types import AgentState, DefaultState, LLMBackend, Step, ToolCall, ToolResult
from openharness.validation import (
    DoneValidator,
    FieldSpec,
    OutputSchema,
    SimpleDoneValidator,
    ValidatingToolRegistry,
    ValidationResult,
    validate_args,
)

__all__ = [
    # version
    "__version__",
    # Core loop
    "composable_loop",
    "async_composable_loop",
    "LoopConfig",
    "LoopHook",
    # Types
    "Step",
    "ToolCall",
    "ToolResult",
    "AgentState",
    "DefaultState",
    "LLMBackend",
    "LLMResult",
    # Async
    "AsyncLLMBackend",
    "AsyncLoopHook",
    "SyncToAsyncAdapter",
    # Conversation
    "Conversation",
    "DefaultSummarizer",
    "Message",
    "MessageRole",
    # Session log
    "SessionLog",
    "LogEntry",
    "InvestigationLog",  # backward-compat alias for SessionLog
    # Tools
    "ToolSpec",
    "BaseToolRegistry",
    "register_think_tool",
    "_clone_tools_excluding",
    # Streaming
    "CallbackEmitter",
    "CompositeEmitter",
    "Event",
    "EventEmitter",
    "HookEvent",
    "LLMCallEndEvent",
    "LLMCallStartEvent",
    "LLMChunkEvent",
    "LoopEndEvent",
    "LoopStartEvent",
    "QueueEmitter",
    "RecoveryEvent",
    "StepEndEvent",
    "StepStartEvent",
    "StreamingHook",
    "ToolDispatchEvent",
    "ToolResultEvent",
    # Backends (LLM adapters)
    "OpenAIBackend",
    "OpenAIStreamingBackend",
    "AnthropicBackend",
    "AnthropicStreamingBackend",
    "AsyncOpenAIBackend",
    "AsyncAnthropicBackend",
    # Checkpoint
    "Checkpoint",
    "CheckpointHook",
    "CheckpointStore",
    "FileCheckpointStore",
    "resume_loop_state",
    # Router
    "CostTracker",
    "FallbackRouter",
    "ModelProfile",
    "ModelRouter",
    "RoutingLLMBackend",
    "SimpleRouter",
    # Telemetry
    "Span",
    "Tracer",
    "TracingHook",
    "MetricsCollector",
    "MetricsHook",
    # Recovery
    "FailureScenario",
    "RecoveryAction",
    "RecoveryRecipe",
    "RecoveryRegistry",
    "build_default_registry",
    # Validation
    "FieldSpec",
    "OutputSchema",
    "ValidationResult",
    "ValidatingToolRegistry",
    "DoneValidator",
    "SimpleDoneValidator",
    "validate_args",
    # Context management
    "ContextManagerHook",
    # Scaffolding
    "DiminishingReturnsTracker",
    "StepProgressTracker",
    "build_parse_recovery_prompt",
    "compress_session_log",
    "enforce_result_budget",
    "estimate_prompt_tokens",
    "estimate_tokens",
    "llm_call_with_retry",
    "reactive_compact",
    "should_compress_context",
    "truncate_tool_result",
    "PARSE_RECOVERY_MAX",
    "MAX_LLM_RETRIES",
    "TOOL_RESULT_MAX_CHARS",
    "TOOL_RESULT_MAX_ROWS",
    # Parsing
    "parse_multi_tool_calls",
    "parse_native_tool_use",
    "parse_tool_call",
    # Prompts
    "build_prompt",
    # Sub-agents
    "run_sub_loop",
    # Flags
    "FLAGS",
    "HARNESS_FLAGS",  # backward-compat alias for FLAGS
]
