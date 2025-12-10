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

## 3. Development Phases

Development is organized into two phases to establish a solid foundation before implementing pattern-specific functionality.

### 3.1 Phase 1: Core Framework and Pattern Scaffolding

Phase 1 delivers the foundational infrastructure and extensible scaffolding for agent patterns.

#### 3.1.1 Core Framework Components

| Component | Description |
|-----------|-------------|
| Project Structure | UV workspace setup, dependency management, package configuration |
| Configuration System | Pydantic Settings integration, environment variable handling, config file support |
| Model Backend Layer | OpenAI-compatible API adapter, connection management, streaming support |
| Tool System | Tool registration, execution engine, built-in tools (filesystem, bash, web search) |
| MCP Integration | MCP client implementation, server lifecycle management, tool discovery |
| Token Management | Token counting with tiktoken, usage tracking, cost estimation |
| Context Management | Message history, context window tracking, compaction strategy interface |
| Subagent Infrastructure | Subagent spawning, state isolation, lifecycle management |
| Skills System | Skill discovery, loading, registration, and execution infrastructure |
| Logging & Observability | Structured logging, request tracing, OpenTelemetry hooks |
| Error Handling | Retry logic with tenacity, circuit breakers, error classification |

#### 3.1.2 Pattern Scaffolding

Phase 1 includes the abstract base classes and interfaces for all agent patterns without implementing pattern-specific logic:

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel

class AgentPattern(ABC):
    """Base class for all agent patterns."""
    
    @abstractmethod
    async def execute(self, task: str, context: AgentContext) -> AgentResult:
        """Execute the pattern's main loop."""
        pass
    
    @abstractmethod
    def on_step(self, step: StepResult) -> StepAction:
        """Handle a single step in the pattern's execution."""
        pass
    
    # Lifecycle hooks (default implementations)
    async def on_start(self, context: AgentContext) -> None: ...
    async def on_complete(self, result: AgentResult) -> None: ...
    async def on_error(self, error: Exception) -> ErrorAction: ...


class PatternConfig(BaseModel):
    """Base configuration for pattern-specific settings."""
    max_iterations: int = 10
    timeout_seconds: float = 300.0
```

**Pattern Scaffolding Deliverables**:

| Pattern | Scaffolding Components |
|---------|------------------------|
| ReAct | `ReActPattern` class, `ReActConfig`, thought/action/observation data models |
| Planning | `PlanningPattern` class, `PlanningConfig`, plan/step data models |
| Reflection | `ReflectionPattern` class, `ReflectionConfig`, critique/revision data models |
| Human-in-the-Loop | `HITLPattern` class, `HITLConfig`, approval request/response interfaces |
| Multi-Agent | `MultiAgentPattern` class, `MultiAgentConfig`, agent communication interfaces |
| Meta-Controller | `MetaControllerPattern` class, `RouterConfig`, routing strategy interfaces |
| Blackboard | `BlackboardPattern` class, `BlackboardConfig`, knowledge source interfaces |

#### 3.1.3 Phase 1 Deliverables

- Fully functional core framework with all infrastructure components
- Abstract base classes for all 7 agent patterns
- Pattern-specific configuration models (Pydantic)
- Pattern-specific data models for inputs/outputs
- Comprehensive test suite for core components
- Documentation for extending patterns
- Example skeleton implementations demonstrating pattern interface usage

#### 3.1.4 Phase 1 Exit Criteria

| Criterion | Requirement |
|-----------|-------------|
| Core Tests | All core framework tests passing (90%+ coverage) |
| Pattern Interfaces | All 7 pattern base classes defined with complete interfaces |
| Tool System | Built-in tools functional and tested |
| MCP Integration | Successfully connects to reference MCP servers |
| Documentation | API documentation for all public interfaces |
| Type Safety | Clean ty/mypy checks on all modules |

### 3.2 Phase 2: Pattern Implementation

Phase 2 implements the full functionality for each agent pattern, building on the Phase 1 scaffolding.

#### 3.2.1 Pattern Implementation Order

Patterns are implemented in order of complexity and dependency:

| Order | Pattern | Dependencies | Complexity |
|-------|---------|--------------|------------|
| 1 | ReAct | Core only | Low |
| 2 | Planning | Core only | Medium |
| 3 | Reflection | Core only | Medium |
| 4 | Human-in-the-Loop | Core only | Medium |
| 5 | Multi-Agent | Subagent system | High |
| 6 | Meta-Controller | Model backend, routing | High |
| 7 | Blackboard | Multi-agent concepts | High |

#### 3.2.2 Per-Pattern Deliverables

Each pattern implementation includes:

- Complete pattern logic implementing the abstract interface
- Pattern-specific tool integrations (if applicable)
- Configuration validation and defaults
- Comprehensive unit tests (90%+ coverage)
- Integration tests with mock model backend
- Pattern-specific documentation with examples
- Migration guide from other frameworks (where applicable)

#### 3.2.3 Phase 2 Implementation Details

**ReAct Pattern**:
- Thought-action-observation loop implementation
- Tool selection and execution logic
- Iteration management and termination conditions
- Structured output parsing for reasoning traces

**Planning Pattern**:
- Plan generation and decomposition
- Step execution and tracking
- Plan revision on failure
- Progress monitoring and reporting

**Reflection Pattern**:
- Self-critique generation
- Response revision loop
- Quality assessment metrics
- Convergence detection

**Human-in-the-Loop Pattern**:
- Approval checkpoint system
- User input collection interfaces
- Timeout and fallback handling
- Audit trail for approvals

**Multi-Agent Pattern**:
- Agent orchestration strategies
- Inter-agent communication protocols
- Result aggregation methods
- Conflict resolution mechanisms

**Meta-Controller Pattern**:
- Query analysis and classification
- Routing strategy implementations (semantic, LLM-based, ML classifier)
- Cascading and fallback logic
- Cost/quality optimization

**Blackboard Pattern**:
- Blackboard data structure implementation
- Knowledge source activation logic
- Scheduling strategies
- Conflict detection and resolution

#### 3.2.4 Phase 2 Exit Criteria

| Criterion | Requirement |
|-----------|-------------|
| Pattern Tests | All pattern implementations tested (90%+ coverage) |
| Integration Tests | End-to-end tests for each pattern with mock backend |
| Local Model Tests | Verified working with Ollama, vLLM, llama.cpp |
| Documentation | Complete guides for each pattern with examples |
| Performance | Meets performance targets defined in Section 7.1 |
| Examples | Working example scripts for each pattern |

### 3.3 Phase Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           PHASE 1                                        │
│                  Core Framework & Scaffolding                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │   Config    │ │   Model     │ │   Tools     │ │    MCP      │        │
│  │   System    │ │  Backend    │ │   System    │ │ Integration │        │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │   Token     │ │  Context    │ │  Subagent   │ │   Skills    │        │
│  │ Management  │ │ Management  │ │   System    │ │   System    │        │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘        │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │              Pattern Base Classes & Interfaces              │        │
│  │   ReAct │ Planning │ Reflection │ HITL │ Multi │ Meta │ BB │        │
│  └─────────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           PHASE 2                                        │
│                    Pattern Implementations                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                        │
│  │  ReAct  │ │Planning │ │Reflect  │ │  HITL   │  ← Lower Complexity    │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘                        │
│  ┌─────────────┐ ┌─────────────────┐ ┌─────────────┐                    │
│  │ Multi-Agent │ │ Meta-Controller │ │  Blackboard │ ← Higher Complexity│
│  └─────────────┘ └─────────────────┘ └─────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Technology Stack

| Component | Tool | Purpose | Documentation |
|:----------|:-----|:--------|:--------------|
| **LLM Agent Framework & Interface** | `pydantic-ai` | LLM Interface, agent orchestration, tool definitions, structured outputs | [Pydantic AI](https://ai.pydantic.dev/) |
| **Validation** | `pydantic` | Runtime data validation, automatic error parsing, and strict schema enforcement | [Pydantic](https://docs.pydantic.dev/latest/) |
| **Configuration** | `pydantic-settings` | Type-safe configuration management (environment variables, `.env` files) | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings) |
| **Environment Files** | `python-dotenv` | Loads key-value pairs from a `.env` file into environment variables so your Python code can access configuration and secrets without hardcoding them | [python-dotenv](https://saurabh-kumar.com/python-dotenv/) |
| **HTTP Client** | `httpx` | Native async support, HTTP/2, strictly typed, and broadly compatible with `requests` API | [HTTPX](https://www.python-httpx.org/) |
| **Retries** | `tenacity` | Decorator-based retry logic with composable stop/wait conditions | [Tenacity](https://tenacity.readthedocs.io/en/latest/) |
| **Tokenizer** | `tiktoken` | Token counting and context window management | [tiktoken](https://github.com/openai/tiktoken) |
| **Testing** | `respx` | Specifically designed to mock `httpx` requests; superior to standard `unittest.mock` | [RESPX](https://lundberg.github.io/respx/) |
| **Package Manager** | `uv` | Fast Python package installer and resolver | [UV](https://docs.astral.sh/uv/) |
| **Formatter & Linter** | `ruff` | Extremely fast Python linter and formatter written in Rust | [Ruff](https://docs.astral.sh/ruff/) |
| **Type Checker** | `ty` | Extremely fast Python type checker and language server written in Rust | [ty](https://docs.astral.sh/ty/) |

---

## 5. Functional Requirements

### 5.1 Agent Pattern Support

The framework must provide a pattern-agnostic base agent class that can be extended to support various agent patterns. Pattern implementations are defined in separate requirement documents (see Section 1.4).

#### 5.1.1 Pattern Interface Requirements

- Base `AgentPattern` abstract class that all patterns must implement
- Standard lifecycle hooks: `on_start`, `on_step`, `on_complete`, `on_error`
- Pattern-specific configuration via Pydantic models
- Composable patterns (ability to combine multiple patterns)
- Runtime pattern switching (optional, for advanced use cases)

#### 5.1.2 Pattern Registration

- Patterns registered via decorator or explicit registration
- Pattern discovery for available patterns
- Pattern validation on agent instantiation

### 5.2 Project and Dependency Management

| Requirement | Specification |
|-------------|---------------|
| Package Manager | UV (astral-sh/uv) |
| Project Structure | UV workspace support for monorepo compatibility |
| Python Version | 3.12+ |
| Lock File | `uv.lock` for reproducible builds |
| Dependency Groups | Separate groups for core, dev, and optional integrations |

### 5.3 Model Backend Compatibility

The framework must work with any model backend exposing an OpenAI-compatible API:

- **Required Compatibility**: `/v1/chat/completions` endpoint
- **Configuration Options**:
  - Base URL (required)
  - API key (optional, for authenticated endpoints)
  - Model identifier
  - Default generation parameters (temperature, max_tokens, top_p, etc.)
- **Tested Backends**: Ollama, vLLM, llama.cpp server, LM Studio, LocalAI
- **Streaming Support**: Must support both streaming and non-streaming responses for model outputs and tool results

#### 6.3.1 Streaming Configuration

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

### 5.4 Built-in Tools

All built-in tools must follow Pydantic AI's tool definition patterns and include comprehensive error handling.

#### 6.4.1 Filesystem Tools

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

#### 6.4.2 Glob Tool

| Tool | Description | Parameters |
|------|-------------|------------|
| `glob_search` | Find files matching a pattern | `pattern: str`, `root_dir: str = "."`, `recursive: bool = True` |

**Requirements**:
- Support standard glob syntax (`*`, `**`, `?`, `[...]`)
- Return relative or absolute paths (configurable)
- Configurable result limit
- Respect gitignore patterns (optional)

#### 6.4.3 Grep Tool

| Tool | Description | Parameters |
|------|-------------|------------|
| `grep_search` | Search file contents | `pattern: str`, `path: str`, `recursive: bool = True`, `file_pattern: str = "*"`, `context_lines: int = 0`, `ignore_case: bool = False`, `regex: bool = True` |

**Requirements**:
- Support both literal string and regex patterns
- Return matched lines with file paths and line numbers
- Configurable context lines (before/after matches)
- Binary file detection and skipping
- Configurable result limit

#### 6.4.4 Bash Tool

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

#### 6.4.5 Web Search Tool

| Tool | Description | Parameters |
|------|-------------|------------|
| `web_search` | Search the web | `query: str`, `num_results: int = 10`, `search_type: str = "general"` |

**Requirements**:
- Pluggable search backend (default implementation required)
- Supported backends: SearXNG (self-hosted), Tavily API, Brave Search API, SerpAPI
- Return structured results (title, URL, snippet)
- Rate limiting support
- Result caching (optional, configurable TTL)

### 5.5 Custom Tool Support

#### 5.5.1 Tool Registration

- Tools defined as decorated Python functions (following Pydantic AI patterns)
- Tools defined as Pydantic models with `__call__` method
- Dynamic tool registration at runtime
- Tool enable/disable without removal
- Tool grouping and namespacing

#### 5.5.2 MCP Server Integration

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

### 5.6 Skills System

Skills are modular capability packages that extend agent functionality through organized folders containing instructions, scripts, and resources. Unlike tools which provide discrete actions, skills provide domain expertise and procedural knowledge that agents can discover and load dynamically.

#### 5.6.1 Skill Structure

A skill is a directory containing a `SKILL.md` file with optional supporting files:

```
my-skill/
├── SKILL.md              # Required: Main skill definition
├── reference.md          # Optional: Detailed reference documentation
├── examples.md           # Optional: Usage examples
├── scripts/
│   ├── helper.py         # Optional: Executable scripts
│   └── validate.sh
└── templates/
    └── output.jinja2     # Optional: Output templates
```

#### 5.6.2 SKILL.md Format

The `SKILL.md` file must contain YAML frontmatter followed by Markdown content:

```yaml
---
name: pdf-processor
description: >
  Extract text, fill forms, and manipulate PDF documents. 
  Use when working with PDF files, forms, or document extraction.
version: 1.0.0
author: optional-author
tags:
  - documents
  - pdf
  - forms
dependencies:
  - pypdf>=3.0.0
  - pdfplumber>=0.9.0
allowed_tools:              # Optional: Restrict available tools when skill is active
  - read_file
  - write_file
  - run_bash
---

# PDF Processor

## Instructions

Step-by-step guidance for the agent when using this skill.

## Examples

Concrete examples demonstrating skill usage.

## References

For detailed API information, see [reference.md](reference.md).
For form filling instructions, see [forms.md](forms.md).
```

#### 5.6.3 Skill Metadata Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique identifier (lowercase, hyphens, max 64 chars) |
| `description` | Yes | What the skill does and when to use it (max 1024 chars) |
| `version` | No | Semantic version string |
| `author` | No | Skill author or maintainer |
| `tags` | No | Categorization tags for discovery |
| `dependencies` | No | Required packages (pip format) |
| `allowed_tools` | No | Tools permitted when skill is active |
| `priority` | No | Numeric priority for conflict resolution (default: 0) |

#### 5.6.4 Progressive Disclosure

Skills use progressive disclosure to manage context efficiently:

| Level | Content | When Loaded |
|-------|---------|-------------|
| 1 | Name and description only | Always (in system prompt) |
| 2 | Full SKILL.md content | When skill is triggered |
| 3+ | Referenced files (reference.md, etc.) | When agent needs specific details |

This ensures agents have access to extensive documentation without consuming context window unnecessarily.

```python
# Progressive disclosure configuration
SkillConfig(
    eager_load=False,           # Don't load full content at startup
    max_context_tokens=4000,    # Max tokens to load from a single skill
    cache_loaded_skills=True,   # Cache loaded skill content
)
```

#### 5.6.5 Skill Discovery and Invocation

Skills are **agent-invoked**: the agent autonomously decides when to use them based on the current task and skill descriptions.

**Discovery Locations** (searched in order):

| Location | Scope | Description |
|----------|-------|-------------|
| `~/.agent-framework/skills/` | User | Personal skills across all projects |
| `./.agent-framework/skills/` | Project | Project-specific skills (version controlled) |
| Registered skill packages | Global | Skills from installed packages |

**Skill Registration**:

```python
from agent_framework import Agent
from agent_framework.skills import Skill, SkillRegistry

# Register skill from directory
registry = SkillRegistry()
registry.register_from_path("./skills/pdf-processor")

# Register skill programmatically
pdf_skill = Skill(
    name="pdf-processor",
    description="Extract and manipulate PDF documents",
    content="# PDF Processor\n\n## Instructions\n...",
    scripts={"extract.py": extract_script_content},
)
registry.register(pdf_skill)

# Create agent with skills
agent = Agent(
    pattern=ReActPattern(),
    backend=backend,
    tools=[read_file, write_file],
    skills=registry,  # Or pass list of Skill objects
)
```

#### 5.6.6 Skill Execution Environment

When a skill is active, the agent can:

- Read the full SKILL.md and referenced files
- Execute bundled scripts via the bash tool
- Access skill-specific templates
- Use only allowed tools (if `allowed_tools` is specified)

**Script Execution**:

```python
# Skill scripts are executed in a controlled environment
SkillExecutionConfig(
    script_timeout=60.0,              # Max execution time per script
    working_directory="skill_dir",    # Execute from skill directory
    inherit_environment=True,         # Inherit parent environment
    sandbox_mode=False,               # Enable for untrusted skills
)
```

#### 5.6.7 Built-in Skills

The framework includes optional built-in skills for common tasks:

| Skill | Description |
|-------|-------------|
| `skill-creator` | Interactive skill creation assistant |
| `code-reviewer` | Code review with best practices |
| `git-workflow` | Git operations and commit message generation |
| `documentation` | Documentation generation and formatting |

Built-in skills can be enabled selectively:

```python
from agent_framework.skills import BuiltinSkills

agent = Agent(
    # ...
    skills=[
        BuiltinSkills.SKILL_CREATOR,
        BuiltinSkills.CODE_REVIEWER,
    ],
)
```

#### 5.6.8 Skill Composition

Multiple skills can be active simultaneously. The agent coordinates their use based on the current task:

```python
# Skills compose automatically based on task requirements
agent = Agent(
    skills=[
        pdf_skill,
        excel_skill,
        documentation_skill,
    ],
)

# Agent will use PDF + documentation skills together when asked to
# "Extract data from the PDF and create a summary document"
```

**Conflict Resolution**:

When multiple skills could apply, resolution uses:
1. Explicit `priority` field (higher wins)
2. Specificity of description match
3. Order of registration (later wins)

#### 5.6.9 Skill Security

Skills can execute code and access the filesystem. Security measures include:

| Measure | Description |
|---------|-------------|
| `allowed_tools` | Restrict tools available when skill is active |
| Sandbox mode | Execute scripts in isolated environment |
| Dependency auditing | Validate skill dependencies before installation |
| Source verification | Track skill provenance and signatures |

**Security Configuration**:

```python
SkillSecurityConfig(
    allow_network_access=False,       # Block network in skill scripts
    allow_file_writes=True,           # Allow writing files
    allowed_paths=["./output"],       # Restrict write locations
    audit_logging=True,               # Log all skill actions
    require_signature=False,          # Require signed skills
)
```

#### 5.6.10 Skill Development Workflow

**Creating a New Skill**:

1. Create skill directory structure
2. Write SKILL.md with metadata and instructions
3. Add supporting files (scripts, templates, references)
4. Test with representative tasks
5. Iterate based on agent behavior

**Testing Skills**:

```python
from agent_framework.skills import SkillTester

tester = SkillTester(skill_path="./my-skill")

# Test skill discovery
assert tester.matches_query("help me with PDF forms")
assert not tester.matches_query("write Python code")

# Test skill execution
result = tester.execute_with_skill(
    "Extract all form fields from document.pdf"
)
assert result.skill_used == "pdf-processor"
```

### 5.7 Subagent System

#### 5.7.1 Subagent Creation

- Programmatic subagent instantiation from parent agent
- Subagent configuration inheritance (with overrides)
- Distinct model configuration per subagent (optional)
- Subagent-specific tool restrictions

#### 5.7.2 State Management

| Requirement | Description |
|-------------|-------------|
| Isolated Context | Each subagent maintains its own message history and context window |
| State Snapshots | Ability to capture and restore subagent state |
| Memory Isolation | Subagent operations do not pollute parent context |
| Lifecycle Hooks | `on_start`, `on_complete`, `on_error` callbacks |

#### 5.7.3 Context Summarization

- Automatic summarization of subagent execution on completion
- Configurable summarization strategies:
  - LLM-based summarization (default)
  - Structured output extraction
  - Final message only
  - Full transcript (for debugging)
- Summary token budget configuration
- Summary inclusion in parent context

#### 5.7.4 Subagent Limits

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

### 5.8 Token Usage Tracking

#### 5.8.1 Token Counting Implementation

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

#### 5.8.2 Metrics Captured

| Metric | Description |
|--------|-------------|
| `prompt_tokens` | Tokens in the prompt/context |
| `completion_tokens` | Tokens in the model response |
| `total_tokens` | Sum of prompt and completion tokens |
| `cached_tokens` | Tokens served from cache (if applicable) |

#### 5.8.3 Tracking Granularity

- Per-request token counts
- Per-agent cumulative totals
- Per-subagent breakdowns
- Per-tool-call attribution (when identifiable)
- Session-level aggregates

#### 5.8.4 Cost Estimation

- Configurable cost-per-token rates
- Estimated cost calculation per request and cumulative
- Cost alerts/limits (optional)

### 5.9 Context Window Management

#### 5.9.1 State Tracking

| Component | Description |
|-----------|-------------|
| System Prompt | Tracked separately, always preserved |
| Message History | Full conversation history with roles |
| Tool Definitions | Current tool schemas |
| Pending Tool Results | Tool calls awaiting results |

#### 5.9.2 Context Compaction Strategies

The framework must support configurable compaction strategies triggered by token thresholds:

| Strategy | Description |
|----------|-------------|
| `sliding_window` | Remove oldest messages beyond a count threshold |
| `summarize_older` | Summarize messages older than N turns, keep recent verbatim |
| `selective_pruning` | Remove tool call/result pairs for completed operations |
| `importance_scoring` | LLM-based importance scoring, prune lowest scored messages |
| `hybrid` | Combination of strategies with configurable weights |

#### 5.9.3 Compaction Configuration

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

### 5.10 Helper Functions and Observability

#### 5.10.1 Context Inspection

| Function | Returns |
|----------|---------|
| `get_context_state()` | Current messages, token counts, compaction history |
| `get_message_history()` | Full or filtered message list |
| `get_system_prompt()` | Current system prompt |
| `get_active_tools()` | List of currently enabled tools |

#### 5.10.2 Token Usage Inspection

| Function | Returns |
|----------|---------|
| `get_token_usage()` | Current session token metrics |
| `get_token_breakdown()` | Per-component token attribution |
| `get_cost_estimate()` | Estimated cost based on configured rates |
| `get_usage_history()` | Time-series of token usage |

#### 5.10.3 Agent State Inspection

| Function | Returns |
|----------|---------|
| `get_agent_state()` | Current execution state, pending actions |
| `get_execution_trace()` | Full trace of reasoning steps, tool calls, results |
| `get_subagent_states()` | State summaries of all child agents |

---

## 6. Developer Experience

### 6.1 Type Hints

- Complete type annotations for all public APIs
- Support for type checkers (mypy, pyright, pyrefly, ty)
- Generic types for flexibility with Pydantic models
- Typed dictionaries for complex parameter objects
- `py.typed` marker file for PEP 561 compliance

### 6.2 Logging

Integration with Python's standard logging module with optional structured logging support.

#### 7.2.1 Log Levels

| Level | Events |
|-------|--------|
| **DEBUG** | Full request/response details (headers, body samples) |
| **INFO** | Request method, URL, and response status |
| **WARNING** | Retries, slow responses, deprecation notices |
| **ERROR** | Fatal errors, connection failures |

#### 6.2.2 Log Configuration

```python
LoggingConfig(
    level="INFO",                          # Default log level
    structured=False,                      # Enable JSON format logging
    include_request_body=False,            # Opt-in request body logging
    include_response_body=False,           # Opt-in response body logging
    body_size_limit=1024,                  # Max bytes to log for bodies
    include_correlation_id=True,           # Include correlation IDs
    
    # Sensitive data redaction
    redact_authorization=True,             # Redact auth headers (default)
    redact_api_keys=True,                  # Redact API keys in query params
    sensitive_patterns=[                   # Custom regex patterns to redact
        r"password=\S+",
        r"secret=\S+",
    ],
)
```

### 6.3 Observability

Built-in observability features for monitoring and debugging agent behavior.

| Feature | Description |
|---------|-------------|
| Request ID Generation | Automatic ID generation (configurable: UUID4, ULID) |
| Trace Context Propagation | Support for `traceparent`, `tracestate`, `X-Request-ID` headers |
| OpenTelemetry Integration | Optional instrumentation hooks (separate dependency) |
| Correlation IDs | Structured logging with correlation ID inclusion |
| Duration Metrics | Request duration exposure via hooks |
| Custom Collectors | Support for custom metrics collectors |

#### 6.3.1 Observability Configuration

```python
ObservabilityConfig(
    request_id_format="uuid4",             # "uuid4" or "ulid"
    propagate_trace_context=True,          # Propagate trace headers
    enable_otel_instrumentation=False,     # OpenTelemetry hooks
    metrics_enabled=True,                  # Expose duration metrics
    custom_collectors=[],                  # Custom metrics collectors
)
```

### 6.4 Configuration Management

Type-safe configuration management using Pydantic Settings with support for multiple configuration sources.

#### 6.4.1 Configuration Sources

Configuration values are loaded from multiple sources in the following priority order (highest to lowest):

| Priority | Source | Description |
|----------|--------|-------------|
| 1 | Constructor arguments | Explicitly passed values |
| 2 | Environment variables | `AGENT_` prefixed variables |
| 3 | `.env` file | Local environment file |
| 4 | `config.toml` / `config.yaml` | Project configuration file |
| 5 | Default values | Defined in settings models |

#### 6.4.2 Environment Variable Mapping

All settings support environment variable configuration with the `AGENT_` prefix:

| Setting | Environment Variable | Example |
|---------|---------------------|---------|
| `model_backend.base_url` | `AGENT_MODEL_BACKEND__BASE_URL` | `http://localhost:11434/v1` |
| `model_backend.api_key` | `AGENT_MODEL_BACKEND__API_KEY` | `sk-...` |
| `logging.level` | `AGENT_LOGGING__LEVEL` | `DEBUG` |
| `retry.max_attempts` | `AGENT_RETRY__MAX_ATTEMPTS` | `3` |

Note: Nested settings use double underscore (`__`) as separator.

#### 6.4.3 Settings Models

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AgentSettings(BaseSettings):
    """Root configuration for the agent framework."""
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        toml_file="config.toml",
        extra="ignore",
    )
    
    # Model backend configuration
    model_backend: ModelBackendSettings = Field(default_factory=ModelBackendSettings)
    
    # Logging configuration
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Observability configuration
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    # Retry configuration
    retry: ErrorRecoveryConfig = Field(default_factory=ErrorRecoveryConfig)
    
    # Token tracking configuration
    token_tracking: TokenizerConfig = Field(default_factory=TokenizerConfig)
    
    # Context management configuration
    context: CompactionConfig = Field(default_factory=CompactionConfig)


class ModelBackendSettings(BaseSettings):
    """Configuration for the model backend connection."""
    
    base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for the OpenAI-compatible API endpoint"
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authenticated endpoints"
    )
    model: str = Field(
        default="llama3.2",
        description="Model identifier to use"
    )
    timeout: float = Field(
        default=30.0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed requests"
    )
```

#### 6.4.4 Configuration File Formats

**TOML Configuration** (`config.toml`):

```toml
[model_backend]
base_url = "http://localhost:11434/v1"
model = "llama3.2"
timeout = 30.0

[logging]
level = "INFO"
structured = true

[retry]
retry_level = 2
initial_backoff_seconds = 1.0

[context]
strategy = "summarize_older"
trigger_threshold_tokens = 100000
```

**YAML Configuration** (`config.yaml`):

```yaml
model_backend:
  base_url: http://localhost:11434/v1
  model: llama3.2
  timeout: 30.0

logging:
  level: INFO
  structured: true

retry:
  retry_level: 2
  initial_backoff_seconds: 1.0

context:
  strategy: summarize_older
  trigger_threshold_tokens: 100000
```

#### 6.4.5 Configuration Loading

```python
from agent_framework import AgentSettings, Agent

# Load from default sources (env vars, .env, config files)
settings = AgentSettings()

# Override specific values
settings = AgentSettings(
    model_backend={"base_url": "http://custom:8080/v1"}
)

# Load from specific file
settings = AgentSettings(_env_file="production.env")

# Create agent with settings
agent = Agent.from_settings(settings)
```

#### 6.4.6 Secrets Management

Sensitive configuration values are handled securely:

| Feature | Description |
|---------|-------------|
| `SecretStr` type | API keys use Pydantic's `SecretStr` to prevent accidental logging |
| Environment precedence | Secrets should be provided via environment variables, not config files |
| Redaction | Secret values are automatically redacted in logs and `repr()` output |
| Validation | Secrets are validated but never exposed in error messages |

```python
from pydantic import SecretStr

class ModelBackendSettings(BaseSettings):
    api_key: SecretStr | None = Field(default=None)
    
    def get_headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            # Access secret value only when needed
            headers["Authorization"] = f"Bearer {self.api_key.get_secret_value()}"
        return headers
```

#### 6.4.7 Configuration Validation

All configuration is validated at load time using Pydantic:

- Type coercion (strings to integers, booleans, etc.)
- Range validation (e.g., `retry_level` must be 1-3)
- URL validation for endpoints
- Path existence validation for file paths (optional)
- Custom validators for complex constraints

```python
from pydantic import field_validator

class AgentSettings(BaseSettings):
    @field_validator("retry")
    @classmethod
    def validate_retry_level(cls, v: ErrorRecoveryConfig) -> ErrorRecoveryConfig:
        if v.retry_level not in (1, 2, 3):
            raise ValueError("retry_level must be 1, 2, or 3")
        return v
```

---

## 7. Non-Functional Requirements

### 7.1 Performance

| Requirement | Target |
|-------------|--------|
| Tool execution overhead | < 10ms excluding actual tool runtime |
| Context serialization | < 100ms for 100k token context |
| Subagent spawn time | < 50ms |
| Memory per agent | < 50MB base, scaling with context size |

### 7.2 Reliability

- Graceful handling of model backend unavailability
- Automatic retry with exponential backoff for transient failures
- Tool execution timeout enforcement
- State recovery after unexpected termination (checkpoint support)

#### 7.2.1 Error Recovery Configuration

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

### 7.3 Security

- No execution of untrusted code without explicit sandbox configuration
- Secrets management (API keys not logged, not included in traces)
- Input sanitization for shell and filesystem operations
- Audit logging for security-sensitive operations

### 7.4 Extensibility

- Plugin architecture for custom compaction strategies
- Custom tool base classes for specialized tool categories
- Event hooks throughout agent lifecycle
- Custom serialization for state persistence

### 7.5 Testing and Quality Control

#### 7.5.1 Testability

- Design for easy mocking and testing
- Provide test utilities for common scenarios (mock server, fixtures)
- Support for request/response recording and playback
- Integration with respx for httpx mocking
- Example test patterns in documentation

#### 7.5.2 Code Quality

| Requirement | Standard |
|-------------|----------|
| Style | PEP 8 compliance enforced via ruff |
| Type Checking | Static analysis with ty |
| Formatting | Consistent formatting with ruff |
| Coverage | Minimum 90% code coverage target |
| API Testing | All public APIs must have tests |
| Integration | Integration tests against real HTTP servers |

#### 7.5.3 Test Infrastructure

- Unit test coverage target: 90%+
- Integration tests for each agent pattern
- Mock model backend for deterministic testing
- Tool testing utilities included
- Agent pattern test harnesses

### 7.6 Documentation

#### 7.6.1 Code Documentation

- Comprehensive docstrings for all public APIs following Google style
- Type hints integrated with documentation
- Code examples in docstrings
- Sphinx-compatible documentation with autodoc support

#### 7.6.2 External Documentation

| Document | Description |
|----------|-------------|
| API Reference | All public interfaces with examples |
| Pattern Guides | Implementation guides for each agent pattern |
| Tool Development | Guide for creating custom tools |
| MCP Integration | Tutorial for MCP server integration |
| Migration Guide | Guides from LangChain, AutoGen, and requests |
| Tutorials | Step-by-step guides for common use cases |

---

## 8. Architecture Overview

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
│  │                      Core Agent Engine                     │  │
│  │  (Pydantic AI integration, execution loop, state mgmt)    │  │
│  └─────────────────────────────┬─────────────────────────────┘  │
│                                │                                │
│  ┌─────────────────────────────┴─────────────────────────────┐  │
│  │                     Model Backend Layer                    │  │
│  │              (OpenAI-compatible API adapter)               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│           Local Model Backends (Ollama, vLLM, etc.)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. API Design Principles

### 9.1 Consistency

- All configuration via Pydantic models
- Consistent naming conventions (snake_case for Python, following Pydantic AI patterns)
- Predictable return types (always return structured objects, not raw dicts)

### 9.2 Composability

- Agent patterns implementable via mixins or composition
- Tools usable independently of agents
- Context manager usable as standalone utility
- Method chaining where appropriate

### 9.3 Sensible Defaults

- Zero-configuration startup for simple use cases
- Progressive disclosure: simple things simple, complex things possible
- Fail-safe defaults (e.g., sandbox mode on by default for filesystem tools)
- Minimal configuration required for common use cases

### 9.4 Pythonic Interface

- Intuitive interface following established Python conventions
- Consistent naming conventions aligned with httpx where applicable
- Clear separation between configuration and runtime behavior
- Predictable behavior matching developer expectations

---

## 10. Example Usage

### 10.1 Basic Agent with Tools

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

### 10.2 Agent with Subagents

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

## 11. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Pattern Coverage | All 7 patterns implemented with documentation |
| Tool Reliability | Built-in tools pass 100% of integration tests |
| Token Accuracy | Token counts within 5% of actual model tokenization |
| Local Model Compat | Verified working with Ollama, vLLM, llama.cpp |
| MCP Integration | Successfully connects to reference MCP servers |
| Documentation | All public APIs documented with examples |

---

## 12. Future Considerations

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
