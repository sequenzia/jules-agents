# AI Agent Framework - Core Requirements Document

## 1. Overview

### 1.1 Purpose

This document defines the requirements for a flexible, extensible AI Agent framework designed to support multiple agent architectural patterns. The framework enables developers to build production-grade autonomous agents capable of reasoning, planning, tool use, and collaboration while maintaining observability and control over execution.

### 1.2 Scope

The framework provides:

- A unified interface for implementing various agent patterns
- Built-in tooling for common operations (filesystem, search, shell execution)
- Integration points for custom tools and Model Context Protocol (MCP) servers
- Hierarchical agent orchestration through subagent support
- Comprehensive context window and token management
- Compatibility with locally hosted models exposing OpenAI-compatible APIs

### 1.3 Target Users

- ML Engineers building custom agent applications
- Developers integrating agentic capabilities into existing systems
- Researchers experimenting with agent architectures and patterns

### 1.4 Related Documents

Agent pattern requirements are defined in separate documents:

| Pattern | Document |
|---------|----------|
| ReAct (Reasoning and Acting) | [pattern-react.md](./patterns/pattern-react.md) |
| Planning | [pattern-planning.md](./patterns/pattern-planning.md) |
| Reflection | [pattern-reflection.md](./patterns/pattern-reflection.md) |
| Human-in-the-Loop | [pattern-human-in-the-loop.md](./patterns/pattern-human-in-the-loop.md) |
| Multi-Agent Systems | [pattern-multi-agent.md](./patterns/pattern-multi-agent.md) |
| Meta-Controller (Router/Orchestrator) | [pattern-meta-controller.md](./patterns/pattern-meta-controller.md) |
| Blackboard System (Shared Workspace) | [pattern-blackboard.md](./patterns/pattern-blackboard.md) |

---

## 2. Goals and Non-Goals

### 2.1 Goals

- Provide a pattern-agnostic foundation that supports multiple agent architectures without opinionated constraints
- Enable seamless integration with local LLM deployments (Ollama, vLLM, llama.cpp, etc.)
- Offer first-class support for tool use, including dynamic tool registration and MCP server integration
- Maintain full observability into agent state, token consumption, and context evolution
- Support complex multi-agent workflows with proper state isolation and summarization

### 2.2 Non-Goals

- This framework will not provide pre-built, domain-specific agents (e.g., coding assistants, research agents)
- This framework will not include a graphical user interface for agent management
- This framework will not handle model hosting or deployment
- This framework will not provide built-in vector storage or RAG capabilities (these can be added via tools)

---

## 3. Functional Requirements

### 3.1 Agent Pattern Support

The framework must provide a pattern-agnostic base agent class that can be extended to support various agent patterns. Pattern implementations are defined in separate requirement documents (see Section 1.4).

#### 3.1.1 Pattern Interface Requirements

- Base `AgentPattern` abstract class that all patterns must implement
- Standard lifecycle hooks: `on_start`, `on_step`, `on_complete`, `on_error`
- Pattern-specific configuration via Pydantic models
- Composable patterns (ability to combine multiple patterns)
- Runtime pattern switching (optional, for advanced use cases)

#### 3.1.2 Pattern Registration

- Patterns registered via decorator or explicit registration
- Pattern discovery for available patterns
- Pattern validation on agent instantiation

### 3.2 Project and Dependency Management

| Requirement | Specification |
|-------------|---------------|
| Package Manager | UV (astral-sh/uv) |
| Project Structure | UV workspace support for monorepo compatibility |
| Python Version | 3.12+ |
| Lock File | `uv.lock` for reproducible builds |
| Dependency Groups | Separate groups for core, dev, and optional integrations |

### 3.3 Core Framework Dependencies

| Dependency | Purpose |
|------------|---------|
| Pydantic AI | Agent orchestration, tool definitions, structured outputs |
| Pydantic | Data validation, settings management, schema definitions |
| httpx | HTTP client for API calls and MCP communication |
| tiktoken | Token counting and context window management |

### 3.4 Model Backend Compatibility

The framework must work with any model backend exposing an OpenAI-compatible API:

- **Required Compatibility**: `/v1/chat/completions` endpoint
- **Configuration Options**:
  - Base URL (required)
  - API key (optional, for authenticated endpoints)
  - Model identifier
  - Default generation parameters (temperature, max_tokens, top_p, etc.)
- **Tested Backends**: Ollama, vLLM, llama.cpp server, LM Studio, LocalAI
- **Streaming Support**: Must support both streaming and non-streaming responses for model outputs and tool results

#### 3.4.1 Streaming Configuration

| Component | Streamable | Description |
|-----------|------------|-------------|
| Model responses | Yes | Token-by-token streaming of LLM output |
| Tool results | Yes | Streaming output from long-running tools (e.g., bash commands, file reads) |
| Subagent output | Yes | Stream subagent responses back to parent |
| Final response | Yes | Complete agent response stream |

**Streaming Configuration**:
```python
StreamingConfig(
    stream_model_responses=True,    # Stream LLM token output
    stream_tool_results=True,       # Stream tool execution output
    stream_subagent_output=False,   # Buffer subagent output (default)
    chunk_size=1024,                # Bytes per chunk for tool streaming
)
```

### 3.5 Built-in Tools

All built-in tools must follow Pydantic AI's tool definition patterns and include comprehensive error handling.

#### 3.5.1 Filesystem Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `read_file` | Read contents of a file | `path: str`, `encoding: str = "utf-8"` |
| `write_file` | Write or overwrite a file | `path: str`, `content: str`, `encoding: str = "utf-8"` |
| `append_file` | Append content to a file | `path: str`, `content: str` |
| `list_directory` | List contents of a directory | `path: str`, `recursive: bool = False`, `max_depth: int = 2` |
| `file_info` | Get file metadata | `path: str` |
| `delete_file` | Delete a file | `path: str`, `require_confirmation: bool = True` |
| `move_file` | Move or rename a file | `source: str`, `destination: str` |
| `copy_file` | Copy a file | `source: str`, `destination: str` |

**Security Requirements**:
- Configurable base directory restriction (sandbox mode)
- Path traversal prevention
- Optional file size limits for read operations
- Configurable allowed/denied file extensions

#### 3.5.2 Glob Tool

| Tool | Description | Parameters |
|------|-------------|------------|
| `glob_search` | Find files matching a pattern | `pattern: str`, `root_dir: str = "."`, `recursive: bool = True` |

**Requirements**:
- Support standard glob syntax (`*`, `**`, `?`, `[...]`)
- Return relative or absolute paths (configurable)
- Configurable result limit
- Respect gitignore patterns (optional)

#### 3.5.3 Grep Tool

| Tool | Description | Parameters |
|------|-------------|------------|
| `grep_search` | Search file contents | `pattern: str`, `path: str`, `recursive: bool = True`, `file_pattern: str = "*"`, `context_lines: int = 0`, `ignore_case: bool = False`, `regex: bool = True` |

**Requirements**:
- Support both literal string and regex patterns
- Return matched lines with file paths and line numbers
- Configurable context lines (before/after matches)
- Binary file detection and skipping
- Configurable result limit

#### 3.5.4 Bash Tool

| Tool | Description | Parameters |
|------|-------------|------------|
| `run_bash` | Execute a shell command | `command: str`, `working_dir: str = "."`, `timeout: int = 30`, `env: dict = None` |

**Requirements**:
- Capture stdout, stderr, and return code
- Configurable timeout with graceful termination
- Environment variable injection
- Working directory specification
- Optional command allowlist/denylist for security

**Security Requirements**:
- Configurable restricted mode (limit to specific commands)
- Shell injection prevention guidance in documentation
- Optional confirmation requirement for destructive commands

#### 3.5.5 Web Search Tool

| Tool | Description | Parameters |
|------|-------------|------------|
| `web_search` | Search the web | `query: str`, `num_results: int = 10`, `search_type: str = "general"` |

**Requirements**:
- Pluggable search backend (default implementation required)
- Supported backends: SearXNG (self-hosted), Tavily API, Brave Search API, SerpAPI
- Return structured results (title, URL, snippet)
- Rate limiting support
- Result caching (optional, configurable TTL)

### 3.6 Custom Tool Support

#### 3.6.1 Tool Registration

- Tools defined as decorated Python functions (following Pydantic AI patterns)
- Tools defined as Pydantic models with `__call__` method
- Dynamic tool registration at runtime
- Tool enable/disable without removal
- Tool grouping and namespacing

#### 3.6.2 MCP Server Integration

| Requirement | Specification |
|-------------|---------------|
| Transport | stdio and HTTP/SSE transports |
| Discovery | Automatic tool discovery from MCP server capabilities |
| Lifecycle | Managed server lifecycle (start, health check, shutdown) |
| Configuration | YAML or JSON configuration for server definitions |
| Multiple Servers | Support for connecting to multiple MCP servers simultaneously |
| Authentication | API key authentication for secured MCP servers |

**MCP Configuration Example**:
```yaml
mcp_servers:
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed"]
    transport: stdio
  - name: custom-api
    url: http://localhost:8080/mcp
    transport: sse
    auth:
      type: api_key
      key_env: MCP_CUSTOM_API_KEY  # Environment variable containing the API key
      header: X-API-Key            # Header name for the API key (default: Authorization)
  - name: secured-service
    url: https://api.example.com/mcp
    transport: sse
    auth:
      type: api_key
      key: ${SECURED_SERVICE_KEY}  # Direct reference to environment variable
```

**Authentication Configuration**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `type` | `str` | Authentication type (`api_key`) |
| `key_env` | `str` | Environment variable name containing the API key |
| `key` | `str` | Direct key value or environment variable reference (`${VAR_NAME}`) |
| `header` | `str` | HTTP header name for the key (default: `Authorization` with `Bearer` prefix) |

### 3.7 Subagent System

#### 3.7.1 Subagent Creation

- Programmatic subagent instantiation from parent agent
- Subagent configuration inheritance (with overrides)
- Distinct model configuration per subagent (optional)
- Subagent-specific tool restrictions

#### 3.7.2 State Management

| Requirement | Description |
|-------------|-------------|
| Isolated Context | Each subagent maintains its own message history and context window |
| State Snapshots | Ability to capture and restore subagent state |
| Memory Isolation | Subagent operations do not pollute parent context |
| Lifecycle Hooks | `on_start`, `on_complete`, `on_error` callbacks |

#### 3.7.3 Context Summarization

- Automatic summarization of subagent execution on completion
- Configurable summarization strategies:
  - LLM-based summarization (default)
  - Structured output extraction
  - Final message only
  - Full transcript (for debugging)
- Summary token budget configuration
- Summary inclusion in parent context

#### 3.7.4 Subagent Limits

Configurable limits to prevent runaway subagent spawning and excessive resource consumption.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_subagents` | `int` | `5` | Maximum number of concurrent subagents per parent agent |
| `max_depth` | `int` | `3` | Maximum nesting depth (subagents spawning subagents) |
| `max_total_subagents` | `int` | `20` | Maximum total subagents across entire agent tree |
| `subagent_timeout` | `float` | `300.0` | Default timeout in seconds for subagent execution |

**Subagent Limits Configuration**:
```python
SubagentLimitsConfig(
    max_subagents=5,           # Max concurrent subagents per parent
    max_depth=3,               # Max nesting depth
    max_total_subagents=20,    # Max across entire hierarchy
    subagent_timeout=300.0,    # 5 minute default timeout
    on_limit_exceeded="error", # "error", "queue", or "reject_oldest"
)
```

### 3.8 Token Usage Tracking

#### 3.8.1 Token Counting Implementation

Token counting uses **tiktoken** for approximate token estimation. This provides fast, local token counting without requiring model backend calls.

| Configuration | Description |
|---------------|-------------|
| `tokenizer` | Tiktoken encoding to use (default: `cl100k_base` for GPT-4 compatibility) |
| `fallback_encoding` | Fallback if model-specific encoding unavailable |
| `count_special_tokens` | Include special tokens in count (default: `True`) |

**Tokenizer Configuration**:
```python
TokenizerConfig(
    encoding="cl100k_base",      # Default tiktoken encoding
    model_mapping={              # Optional model-specific encodings
        "llama": "cl100k_base",
        "mistral": "cl100k_base",
    },
    cache_tokenizer=True,        # Cache tokenizer instance for performance
)
```

**Note**: Token counts are approximate and may vary slightly from actual model tokenization. For context window management, counts include a configurable safety margin (default: 5%) to prevent overflow.

#### 3.8.2 Metrics Captured

| Metric | Description |
|--------|-------------|
| `prompt_tokens` | Tokens in the prompt/context |
| `completion_tokens` | Tokens in the model response |
| `total_tokens` | Sum of prompt and completion tokens |
| `cached_tokens` | Tokens served from cache (if applicable) |

#### 3.8.3 Tracking Granularity

- Per-request token counts
- Per-agent cumulative totals
- Per-subagent breakdowns
- Per-tool-call attribution (when identifiable)
- Session-level aggregates

#### 3.8.3 Cost Estimation

- Configurable cost-per-token rates
- Estimated cost calculation per request and cumulative
- Cost alerts/limits (optional)

### 3.9 Context Window Management

#### 3.9.1 State Tracking

| Component | Description |
|-----------|-------------|
| System Prompt | Tracked separately, always preserved |
| Message History | Full conversation history with roles |
| Tool Definitions | Current tool schemas |
| Pending Tool Results | Tool calls awaiting results |

#### 3.9.2 Context Compaction Strategies

The framework must support configurable compaction strategies triggered by token thresholds:

| Strategy | Description |
|----------|-------------|
| `sliding_window` | Remove oldest messages beyond a count threshold |
| `summarize_older` | Summarize messages older than N turns, keep recent verbatim |
| `selective_pruning` | Remove tool call/result pairs for completed operations |
| `importance_scoring` | LLM-based importance scoring, prune lowest scored messages |
| `hybrid` | Combination of strategies with configurable weights |

#### 3.9.3 Compaction Configuration

```python
# Example configuration structure
CompactionConfig(
    strategy="summarize_older",
    trigger_threshold_tokens=100000,
    target_tokens=80000,
    preserve_recent_turns=10,
    preserve_system_prompt=True,
    summarization_model="same"  # or specify different model
)
```

### 3.10 Helper Functions and Observability

#### 3.10.1 Context Inspection

| Function | Returns |
|----------|---------|
| `get_context_state()` | Current messages, token counts, compaction history |
| `get_message_history()` | Full or filtered message list |
| `get_system_prompt()` | Current system prompt |
| `get_active_tools()` | List of currently enabled tools |

#### 3.10.2 Token Usage Inspection

| Function | Returns |
|----------|---------|
| `get_token_usage()` | Current session token metrics |
| `get_token_breakdown()` | Per-component token attribution |
| `get_cost_estimate()` | Estimated cost based on configured rates |
| `get_usage_history()` | Time-series of token usage |

#### 3.10.3 Agent State Inspection

| Function | Returns |
|----------|---------|
| `get_agent_state()` | Current execution state, pending actions |
| `get_execution_trace()` | Full trace of reasoning steps, tool calls, results |
| `get_subagent_states()` | State summaries of all child agents |

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Requirement | Target |
|-------------|--------|
| Tool execution overhead | < 10ms excluding actual tool runtime |
| Context serialization | < 100ms for 100k token context |
| Subagent spawn time | < 50ms |
| Memory per agent | < 50MB base, scaling with context size |

### 4.2 Reliability

- Graceful handling of model backend unavailability
- Automatic retry with exponential backoff for transient failures
- Tool execution timeout enforcement
- State recovery after unexpected termination (checkpoint support)

#### 4.2.1 Error Recovery Configuration

Error recovery aggressiveness is configurable on a scale of 1-3, with different retry behaviors for tool failures vs. model failures.

| Level | Description | Tool Retries | Model Retries | Backoff |
|-------|-------------|--------------|---------------|---------|
| `1` (Conservative) | Minimal retries, fail fast | 1 | 2 | Aggressive (2x multiplier) |
| `2` (Balanced) | Default behavior | 2 | 3 | Standard (1.5x multiplier) |
| `3` (Aggressive) | Maximum retry attempts | 3 | 5 | Gentle (1.2x multiplier) |

**Error Recovery Configuration**:
```python
ErrorRecoveryConfig(
    retry_level=2,                    # 1=conservative, 2=balanced (default), 3=aggressive

    # Fine-grained overrides (optional)
    tool_max_retries=None,            # Override tool retry count
    model_max_retries=None,           # Override model retry count

    # Backoff configuration
    initial_backoff_seconds=1.0,      # Initial wait before retry
    max_backoff_seconds=60.0,         # Maximum wait between retries

    # Error classification
    retryable_tool_errors=[           # Tool errors that trigger retry
        "TimeoutError",
        "ConnectionError",
        "TemporaryFailure",
    ],
    retryable_model_errors=[          # Model errors that trigger retry
        "RateLimitError",
        "ServiceUnavailable",
        "Timeout",
    ],

    # Circuit breaker
    circuit_breaker_threshold=5,      # Failures before circuit opens
    circuit_breaker_timeout=30.0,     # Seconds before retry after circuit opens
)
```

### 4.3 Security

- No execution of untrusted code without explicit sandbox configuration
- Secrets management (API keys not logged, not included in traces)
- Input sanitization for shell and filesystem operations
- Audit logging for security-sensitive operations

### 4.4 Extensibility

- Plugin architecture for custom compaction strategies
- Custom tool base classes for specialized tool categories
- Event hooks throughout agent lifecycle
- Custom serialization for state persistence

### 4.5 Testing

- Unit test coverage target: 80%+
- Integration tests for each agent pattern
- Mock model backend for deterministic testing
- Tool testing utilities included

### 4.6 Documentation

- API reference with all public interfaces
- Pattern implementation guides with examples
- Tool development guide
- MCP integration tutorial
- Migration guide from other frameworks (LangChain, AutoGen)

---

## 5. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Agent Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Agent Patterns │  │  Tool Registry  │  │ Context Manager │  │
│  │  ─────────────  │  │  ────────────   │  │ ─────────────── │  │
│  │  • ReAct        │  │  • Built-in     │  │ • State Track   │  │
│  │  • Planning     │  │  • Custom       │  │ • Compaction    │  │
│  │  • Reflection   │  │  • MCP Bridge   │  │ • Token Count   │  │
│  │  • HITL         │  │                 │  │                 │  │
│  │  • Multi-Agent  │  │                 │  │                 │  │
│  │  • Meta-Control │  │                 │  │                 │  │
│  │  • Blackboard   │  │                 │  │                 │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
│  ┌─────────────────────────────┴─────────────────────────────┐  │
│  │                      Core Agent Engine                    │  │
│  │  (Pydantic AI integration, execution loop, state mgmt)    │  │
│  └─────────────────────────────┬─────────────────────────────┘  │
│                                │                                │
│  ┌─────────────────────────────┴─────────────────────────────┐  │
│  │                     Model Backend Layer                   │  │
│  │              (OpenAI-compatible API adapter)              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│           Local Model Backends (Ollama, vLLM, etc.)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. API Design Principles

### 6.1 Consistency

- All configuration via Pydantic models
- Consistent naming conventions (snake_case for Python, following Pydantic AI patterns)
- Predictable return types (always return structured objects, not raw dicts)

### 6.2 Composability

- Agent patterns implementable via mixins or composition
- Tools usable independently of agents
- Context manager usable as standalone utility

### 6.3 Sensible Defaults

- Zero-configuration startup for simple use cases
- Progressive disclosure of advanced features
- Fail-safe defaults (e.g., sandbox mode on by default for filesystem tools)

---

## 7. Example Usage

### 7.1 Basic Agent with Tools

```python
from agent_framework import Agent
from agent_framework.patterns import ReActPattern
from agent_framework.tools import read_file, write_file, run_bash
from agent_framework.backends import LocalModelBackend

backend = LocalModelBackend(
    base_url="http://localhost:11434/v1",
    model="llama3.2"
)

agent = Agent(
    pattern=ReActPattern(max_iterations=10),
    backend=backend,
    tools=[read_file, write_file, run_bash],
    system_prompt="You are a helpful coding assistant."
)

result = await agent.run("Create a Python script that lists all .py files in the current directory")
```

### 7.2 Agent with Subagents

```python
from agent_framework import Agent
from agent_framework.patterns import PlanningPattern
from agent_framework.subagents import SubagentConfig

research_config = SubagentConfig(
    name="researcher",
    system_prompt="You are a research specialist.",
    tools=[web_search],
    summarization_strategy="structured"
)

writer_config = SubagentConfig(
    name="writer",
    system_prompt="You are a technical writer.",
    tools=[write_file]
)

orchestrator = Agent(
    pattern=PlanningPattern(),
    backend=backend,
    subagent_configs=[research_config, writer_config],
    system_prompt="You coordinate research and writing tasks."
)
```

---

## 8. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Pattern Coverage | All 7 patterns implemented with documentation |
| Tool Reliability | Built-in tools pass 100% of integration tests |
| Token Accuracy | Token counts within 5% of actual model tokenization |
| Local Model Compat | Verified working with Ollama, vLLM, llama.cpp |
| MCP Integration | Successfully connects to reference MCP servers |
| Documentation | All public APIs documented with examples |

---

## 9. Future Considerations

These items are explicitly out of scope for initial release but should inform architectural decisions:

- **Persistent Memory**: Long-term memory storage and retrieval across sessions
- **Evaluation Framework**: Built-in benchmarking and evaluation harness
- **Distributed Execution**: Multi-process or multi-node agent execution
- **Visual Workflow Builder**: Graph-based agent composition UI
- **Prompt Versioning**: Track and rollback system prompt changes
- **A/B Testing**: Compare agent configurations in production

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| Agent Pattern | A structured approach to agent behavior (e.g., ReAct, Planning) |
| Compaction | The process of reducing context size while preserving relevant information |
| HITL | Human-in-the-Loop, requiring human approval at decision points |
| MCP | Model Context Protocol, a standard for tool/resource integration |
| Subagent | A child agent spawned by a parent agent for delegated tasks |

---

## Appendix B: References

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [UV Package Manager](https://github.com/astral-sh/uv)
