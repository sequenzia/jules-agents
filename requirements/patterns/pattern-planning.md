# Agent Pattern Requirements: Planning

## 1. Overview

### 1.1 Pattern Description

The Planning pattern separates task execution into two distinct phases: plan generation and plan execution. The agent first creates a structured plan decomposing the task into discrete steps, optionally validates the plan, then executes each step sequentially or in parallel. This pattern excels at complex, multi-step tasks requiring coordination and allows for human review before execution begins.

### 1.2 Key Characteristics

- Upfront decomposition of complex tasks into manageable steps
- Explicit plan representation that can be inspected and modified
- Clear separation between planning and execution phases
- Support for plan revision during execution when circumstances change

### 1.3 Related Documents

- [Core Framework Requirements](../core-framework-prd.md)

---

## 2. Functional Requirements

### 2.1 Planning Phase

#### 2.1.1 Plan Generation

The agent must generate a structured plan with the following properties:

```python
@dataclass
class PlanStep:
    id: str                          # Unique step identifier
    description: str                 # Human-readable step description
    action: str                      # Tool or action to execute
    parameters: dict[str, Any]       # Parameters for the action
    dependencies: list[str]          # IDs of steps this depends on
    expected_output: str             # Description of expected result
    fallback_action: str | None      # Alternative if primary fails
    estimated_tokens: int | None     # Estimated token cost

@dataclass
class Plan:
    goal: str                        # Overall task objective
    steps: list[PlanStep]            # Ordered list of steps
    success_criteria: str            # How to determine completion
    created_at: datetime
    estimated_total_tokens: int | None
```

#### 2.1.2 Plan Generation Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `planning_prompt` | `str` | (default template) | Custom prompt for plan generation |
| `max_steps` | `int` | `20` | Maximum number of steps in a plan |
| `require_dependencies` | `bool` | `True` | Whether steps must declare dependencies |
| `require_fallbacks` | `bool` | `False` | Whether steps must have fallback actions |
| `planning_model` | `str` | `None` | Use different model for planning (optional) |
| `structured_output` | `bool` | `True` | Use structured output for plan generation |

#### 2.1.3 Hierarchical Planning

Support for hierarchical task decomposition:

- Top-level plan with high-level steps
- Sub-plans for complex steps (recursive decomposition)
- Configurable maximum decomposition depth
- Roll-up of sub-plan results to parent plan

```
Main Plan
├── Step 1: Research topic
│   └── Sub-plan:
│       ├── Step 1.1: Search for sources
│       ├── Step 1.2: Read top 3 sources
│       └── Step 1.3: Extract key points
├── Step 2: Create outline
└── Step 3: Write content
```

### 2.2 Plan Validation

#### 2.2.1 Automatic Validation

Before execution, the plan must be validated:

| Validation | Description |
|------------|-------------|
| Tool Existence | All referenced tools exist and are available |
| Parameter Validity | Parameters match tool signatures |
| Dependency Graph | No circular dependencies, all referenced steps exist |
| Feasibility Check | Optional LLM review of plan feasibility |
| Token Budget | Estimated tokens within configured limits |

#### 2.2.2 Human Validation (Optional)

When human-in-the-loop is enabled for planning:

- Present plan to user before execution
- Accept approval, rejection, or modification
- Support inline editing of individual steps
- Support reordering and dependency modification
- Timeout handling with configurable default action

### 2.3 Execution Phase

#### 2.3.1 Execution Modes

| Mode | Description |
|------|-------------|
| `sequential` | Execute steps in order, respecting dependencies |
| `parallel` | Execute independent steps concurrently |
| `adaptive` | Start sequential, parallelize when beneficial |

#### 2.3.2 Execution Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `execution_mode` | `str` | `"sequential"` | Execution strategy |
| `max_parallel` | `int` | `3` | Maximum concurrent step executions |
| `step_timeout` | `float` | `60.0` | Timeout per step in seconds |
| `continue_on_failure` | `bool` | `False` | Continue with next steps if one fails |
| `retry_failed_steps` | `int` | `1` | Number of retries for failed steps |

#### 2.3.3 Step Execution Flow

```
┌─────────────────────────────────────────────────────┐
│                    Select Next Step                 │
│           (Based on dependencies & status)          │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                  Pre-Step Hooks                     │
│        (Logging, validation, HITL check)            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                  Execute Action                      │
└─────────────────────┬───────────────────────────────┘
                      │
              ┌───────┴───────┐
              │   Success?    │
              └───────┬───────┘
                      │
         ┌────────────┼────────────┐
         │ Yes        │ No         │
         ▼            ▼            │
┌─────────────┐ ┌─────────────┐    │
│Record Result│ │Try Fallback │    │
└──────┬──────┘ └──────┬──────┘    │
       │               │           │
       │        ┌──────┴──────┐    │
       │        │  Fallback   │    │
       │        │  Success?   │    │
       │        └──────┬──────┘    │
       │               │           │
       │    ┌──────────┴───────┐   │
       │    │ Yes              │No │
       │    ▼                  ▼   │
       │ ┌─────────────┐ ┌────────────┐
       │ │Record Result│ │Handle Error│
       │ └──────┬──────┘ └─────┬──────┘
       │        │              │
       └────────┴──────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                  Post-Step Hooks                    │
│           (Update state, check revision)            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  More Steps?  │───No──▶ Complete
              └───────┬───────┘
                      │ Yes
                      └──────────▶ Select Next Step
```

### 2.4 Plan Revision

#### 2.4.1 Revision Triggers

| Trigger | Description |
|---------|-------------|
| Step Failure | Primary and fallback actions both failed |
| Unexpected Result | Step output doesn't match expected_output |
| New Information | Execution reveals task requirements changed |
| User Request | Human requests plan modification |
| Token Budget | Running low on token budget |

#### 2.4.2 Revision Strategies

| Strategy | Description |
|----------|-------------|
| `replan_remaining` | Generate new plan for incomplete steps |
| `insert_recovery` | Add recovery steps without full replan |
| `skip_and_continue` | Mark step as skipped, continue with dependents |
| `abort` | Stop execution, return partial results |
| `ask_user` | Request human guidance |

#### 2.4.3 Revision Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_revision` | `bool` | `True` | Enable plan revision during execution |
| `max_revisions` | `int` | `3` | Maximum number of plan revisions |
| `revision_strategy` | `str` | `"replan_remaining"` | Default revision approach |
| `revision_threshold` | `float` | `0.3` | Failure rate triggering revision |

### 2.5 State Tracking

#### 2.5.1 Step States

```python
class StepStatus(Enum):
    PENDING = "pending"          # Not yet started
    RUNNING = "running"          # Currently executing
    COMPLETED = "completed"      # Successfully finished
    FAILED = "failed"            # Failed (including fallback)
    SKIPPED = "skipped"          # Skipped due to dependency failure
    BLOCKED = "blocked"          # Waiting on dependencies
    REVISED = "revised"          # Step was modified during revision
```

#### 2.5.2 Execution State

```python
@dataclass
class PlanExecutionState:
    plan: Plan
    step_statuses: dict[str, StepStatus]
    step_results: dict[str, Any]
    current_step: str | None
    revision_count: int
    revision_history: list[PlanRevision]
    started_at: datetime
    completed_at: datetime | None
```

---

## 3. Interface Requirements

### 3.1 Pattern Class

```python
class PlanningPattern(AgentPattern):
    """Planning agent pattern with plan generation and execution phases."""

    def __init__(
        self,
        max_steps: int = 20,
        execution_mode: Literal["sequential", "parallel", "adaptive"] = "sequential",
        max_parallel: int = 3,
        allow_revision: bool = True,
        max_revisions: int = 3,
        require_plan_approval: bool = False,
        continue_on_failure: bool = False,
        planning_prompt: str | None = None,
        hierarchical: bool = False,
        max_decomposition_depth: int = 2,
    ) -> None: ...

    async def generate_plan(
        self,
        agent: Agent,
        query: str,
    ) -> Plan: ...

    async def execute_plan(
        self,
        agent: Agent,
        plan: Plan,
    ) -> PlanExecutionResult: ...

    async def execute(
        self,
        agent: Agent,
        query: str,
    ) -> PlanExecutionResult: ...

    async def revise_plan(
        self,
        agent: Agent,
        current_state: PlanExecutionState,
        reason: str,
    ) -> Plan: ...

    def get_plan(self) -> Plan | None: ...

    def get_execution_state(self) -> PlanExecutionState | None: ...
```

### 3.2 Result Structure

```python
@dataclass
class PlanExecutionResult:
    success: bool
    plan: Plan
    final_plan: Plan                    # May differ from initial if revised
    step_results: dict[str, StepResult]
    final_output: str | None
    revision_count: int
    execution_state: PlanExecutionState
    token_usage: TokenUsage
    execution_time: float
```

### 3.3 Callbacks and Hooks

```python
class PlanningCallbacks:
    async def on_plan_generated(self, plan: Plan) -> Plan:
        """Called after plan generation. Can modify plan."""
        return plan

    async def on_plan_approved(self, plan: Plan) -> bool:
        """Called to get approval. Return False to reject."""
        return True

    async def on_step_start(self, step: PlanStep) -> None:
        """Called before each step execution."""
        pass

    async def on_step_complete(self, step: PlanStep, result: StepResult) -> None:
        """Called after each step completion."""
        pass

    async def on_revision_needed(
        self,
        state: PlanExecutionState,
        reason: str
    ) -> RevisionDecision:
        """Called when revision is triggered."""
        return RevisionDecision.REPLAN
```

---

## 4. Behavioral Requirements

### 4.1 Plan Quality

- Steps should be atomic and independently verifiable
- Dependencies should be minimal but complete
- Expected outputs should be specific and measurable
- Plans should be achievable within token and time budgets

### 4.2 Execution Robustness

- Graceful handling of step failures
- Intelligent use of fallback actions
- Appropriate revision decisions based on failure patterns
- Preservation of partial results on abort

### 4.3 Efficiency

- Minimize redundant steps in plans
- Parallelize independent steps when possible
- Cache intermediate results for reuse
- Avoid regenerating entire plans for minor revisions

---

## 5. Example Usage

### 5.1 Basic Planning Agent

```python
from agent_framework import Agent
from agent_framework.patterns import PlanningPattern
from agent_framework.tools import web_search, read_file, write_file

pattern = PlanningPattern(
    max_steps=15,
    execution_mode="sequential",
    allow_revision=True
)

agent = Agent(
    pattern=pattern,
    tools=[web_search, read_file, write_file],
    system_prompt="You are a research assistant."
)

result = await agent.run(
    "Research the top 3 Python web frameworks and create a comparison document"
)

# Inspect the plan that was generated
print("Generated Plan:")
for step in result.plan.steps:
    print(f"  {step.id}: {step.description}")
```

### 5.2 Planning with Human Approval

```python
pattern = PlanningPattern(
    require_plan_approval=True,
    max_steps=20
)

# Custom approval callback
class MyCallbacks(PlanningCallbacks):
    async def on_plan_approved(self, plan: Plan) -> bool:
        print("Proposed plan:")
        for step in plan.steps:
            print(f"  - {step.description}")
        response = input("Approve? (y/n): ")
        return response.lower() == "y"

agent = Agent(
    pattern=pattern,
    callbacks=MyCallbacks(),
    tools=[...],
)
```

### 5.3 Parallel Execution

```python
pattern = PlanningPattern(
    execution_mode="parallel",
    max_parallel=5,
    continue_on_failure=True  # Don't block on individual failures
)

# The agent will execute independent steps concurrently
result = await agent.run("Analyze these 10 log files for errors")
```

---

## 6. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Plan Validity | 95%+ of generated plans pass automatic validation |
| Step Atomicity | Steps are independently executable in 90%+ of cases |
| Revision Effectiveness | Plan revisions resolve issues in 80%+ of cases |
| Execution Completion | Plans execute to completion in 85%+ of runs |
| Dependency Accuracy | Declared dependencies are necessary and sufficient in 90%+ of plans |

---

## 7. Open Questions

1. Should plans support conditional steps (if-then-else branching)?
2. How should parallel execution handle steps that produce conflicting outputs?
3. Should there be a "dry-run" mode that validates a plan without executing?
4. How detailed should expected_output specifications be for reliable revision triggering?
