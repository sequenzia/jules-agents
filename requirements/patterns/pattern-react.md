# Agent Pattern Requirements: ReAct (Reasoning and Acting)

## 1. Overview

### 1.1 Pattern Description

ReAct (Reasoning and Acting) is an agent pattern that interleaves reasoning traces with action execution. The agent explicitly generates reasoning steps (thoughts) before taking actions, observes the results, and continues the thought-action-observation loop until the task is complete or termination conditions are met.

### 1.2 Reference

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022)

### 1.3 Related Documents

- [Core Framework Requirements](../core-framework-prd.md)

---

## 2. Functional Requirements

### 2.1 Execution Loop

The ReAct pattern must implement a thought-action-observation loop with the following structure:

```
┌─────────────────────────────────────────────────────┐
│                    User Query                       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                    Thought                          │
│    (Reasoning about current state and next step)    │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                    Action                           │
│         (Tool call or final response)               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                  Observation                        │
│              (Tool result or feedback)              │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │   Complete?   │───Yes──▶ Final Response
              └───────┬───────┘
                      │ No
                      └──────────▶ Back to Thought
```

### 2.2 Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | `int` | `10` | Maximum number of thought-action-observation cycles |
| `thought_prompt` | `str` | (default template) | Custom prompt for generating thoughts |
| `require_thought` | `bool` | `True` | Whether to require explicit thought before each action |
| `thought_tag` | `str` | `"Thought"` | Tag/prefix used to identify thought sections |
| `action_tag` | `str` | `"Action"` | Tag/prefix used to identify action sections |
| `observation_tag` | `str` | `"Observation"` | Tag/prefix used to identify observation sections |
| `enable_scratchpad` | `bool` | `True` | Whether to maintain a visible scratchpad of reasoning |

### 2.3 Iteration Control

#### 2.3.1 Maximum Iterations

- Configurable hard limit on the number of iterations
- When limit is reached, the agent must:
  - Generate a final response summarizing progress
  - Indicate that the iteration limit was reached
  - Return partial results if available

#### 2.3.2 Early Termination Conditions

The pattern must support configurable early termination:

| Condition | Description | Configuration |
|-----------|-------------|---------------|
| Success Detection | Agent determines task is complete | `success_phrases: list[str]` or LLM-based detection |
| Failure Detection | Agent determines task cannot be completed | `failure_phrases: list[str]` or LLM-based detection |
| No Progress | Repeated similar actions without advancement | `stall_threshold: int` (consecutive similar actions) |
| Token Budget | Approaching context window limit | `token_budget: int` |
| Time Budget | Execution time limit | `timeout_seconds: float` |
| Custom Condition | User-defined termination logic | `termination_callback: Callable` |

### 2.4 Thought Generation

#### 2.4.1 Thought Structure

Each thought should address:

- Current understanding of the task state
- What has been accomplished so far
- What information is still needed
- Reasoning for the next action choice

#### 2.4.2 Thought Prompt Template

Default template (customizable):

```
Based on the current state of the task:
- What do I know so far?
- What do I still need to find out or accomplish?
- What is the most logical next step?
- Which tool or action would best accomplish this step?

Think step by step before deciding on an action.
```

#### 2.4.3 Thought Validation

- Optional validation that thoughts contain substantive reasoning
- Configurable minimum thought length
- Detection of degenerate thoughts (e.g., "I will use the tool")

### 2.5 Action Execution

#### 2.5.1 Action Types

| Action Type | Description |
|-------------|-------------|
| Tool Call | Invoke a registered tool with parameters |
| Final Answer | Provide the final response to the user |
| Request Clarification | Ask the user for more information |
| Delegate | Hand off to a subagent (if multi-agent enabled) |

#### 2.5.2 Action Validation

- Validate tool exists before execution
- Validate required parameters are provided
- Type checking on parameters (via Pydantic)
- Optional pre-execution hooks for custom validation

### 2.6 Observation Handling

#### 2.6.1 Observation Formatting

- Tool outputs formatted consistently for inclusion in context
- Large outputs truncated with configurable limits
- Error outputs clearly distinguished from success outputs

#### 2.6.2 Observation Processing

| Feature | Description |
|---------|-------------|
| Truncation | Limit observation size to `max_observation_tokens` |
| Summarization | Optional LLM summarization of large outputs |
| Error Wrapping | Wrap errors with context for agent understanding |
| Type Preservation | Maintain structured data types where possible |

### 2.7 Trace Logging

The pattern must maintain a complete execution trace:

```python
@dataclass
class ReActStep:
    iteration: int
    thought: str
    action: Action  # Tool call or final answer
    observation: str | None
    timestamp: datetime
    token_usage: TokenUsage

@dataclass
class ReActTrace:
    steps: list[ReActStep]
    final_result: str | None
    termination_reason: str
    total_iterations: int
    total_tokens: TokenUsage
```

---

## 3. Interface Requirements

### 3.1 Pattern Class

```python
class ReActPattern(AgentPattern):
    """ReAct (Reasoning and Acting) agent pattern."""

    def __init__(
        self,
        max_iterations: int = 10,
        thought_prompt: str | None = None,
        require_thought: bool = True,
        success_phrases: list[str] | None = None,
        failure_phrases: list[str] | None = None,
        stall_threshold: int = 3,
        max_observation_tokens: int = 2000,
        enable_scratchpad: bool = True,
        termination_callback: Callable[[ReActStep], bool] | None = None,
    ) -> None: ...

    async def execute(
        self,
        agent: Agent,
        query: str,
    ) -> ReActResult: ...

    def get_trace(self) -> ReActTrace: ...

    def get_current_scratchpad(self) -> str: ...
```

### 3.2 Result Structure

```python
@dataclass
class ReActResult:
    success: bool
    final_answer: str | None
    trace: ReActTrace
    termination_reason: TerminationReason
    token_usage: TokenUsage
    execution_time: float
```

### 3.3 Termination Reasons

```python
class TerminationReason(Enum):
    SUCCESS = "success"              # Task completed successfully
    MAX_ITERATIONS = "max_iterations"  # Hit iteration limit
    FAILURE_DETECTED = "failure"     # Agent detected it cannot complete task
    STALLED = "stalled"              # No progress being made
    TOKEN_BUDGET = "token_budget"    # Approaching token limit
    TIMEOUT = "timeout"              # Time limit exceeded
    USER_CANCELLED = "cancelled"     # User requested stop
    CUSTOM = "custom"                # Custom termination condition
```

---

## 4. Behavioral Requirements

### 4.1 Reasoning Quality

- Thoughts should demonstrate logical progression
- Each thought should reference relevant prior observations
- Thoughts should justify action choices

### 4.2 Action Selection

- Prefer simpler tools when multiple tools could work
- Avoid repeating failed actions without modification
- Recognize when sufficient information has been gathered

### 4.3 Error Recovery

- On tool error, generate thought analyzing the error
- Attempt alternative approaches when primary approach fails
- Recognize unrecoverable errors and terminate gracefully

### 4.4 Context Efficiency

- Avoid redundant tool calls for information already obtained
- Reference prior observations rather than re-fetching
- Summarize lengthy reasoning chains when context is constrained

---

## 5. Example Usage

### 5.1 Basic ReAct Agent

```python
from agent_framework import Agent
from agent_framework.patterns import ReActPattern
from agent_framework.tools import read_file, list_directory, grep_search

pattern = ReActPattern(
    max_iterations=15,
    require_thought=True,
    stall_threshold=3
)

agent = Agent(
    pattern=pattern,
    tools=[read_file, list_directory, grep_search],
    system_prompt="You are a code analysis assistant."
)

result = await agent.run("Find all functions that handle user authentication in this codebase")

# Access the reasoning trace
for step in result.trace.steps:
    print(f"Thought: {step.thought}")
    print(f"Action: {step.action}")
    print(f"Observation: {step.observation[:200]}...")
    print("---")
```

### 5.2 ReAct with Custom Termination

```python
def custom_termination(step: ReActStep) -> bool:
    # Stop if we find what we're looking for
    if "FOUND:" in step.observation:
        return True
    return False

pattern = ReActPattern(
    max_iterations=20,
    termination_callback=custom_termination
)
```

---

## 6. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Reasoning Coherence | Thoughts logically connect to actions in 90%+ of steps |
| Termination Accuracy | Correct termination reason identified in 95%+ of runs |
| Trace Completeness | All steps captured with full metadata |
| Error Recovery | Agent recovers from tool errors in 80%+ of cases |
| Iteration Efficiency | Tasks completed in < 50% of max_iterations on average |

---

## 7. Open Questions

1. Should thoughts be included in the context window for subsequent iterations, or only actions and observations?
2. How should the pattern handle tools that return streaming results?
3. Should there be a "confidence score" associated with the final answer?
