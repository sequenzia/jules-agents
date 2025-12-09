# Agent Pattern Requirements: Multi-Agent Systems

## 1. Overview

### 1.1 Pattern Description

The Multi-Agent Systems pattern enables coordination between multiple specialized agents working together to accomplish complex tasks. Agents can communicate, delegate tasks, share information, and collaborate through various orchestration strategies. This pattern is essential for tasks that benefit from division of labor, specialized expertise, or parallel processing.

### 1.2 Key Characteristics

- Multiple agents with distinct roles, capabilities, and system prompts
- Communication primitives for agent-to-agent interaction
- Orchestration strategies for coordinating agent activities
- Shared and isolated state management
- Hierarchical or peer-to-peer organizational structures

### 1.3 Use Cases

- Research tasks with separate research and synthesis agents
- Software development with architect, coder, and reviewer agents
- Customer service with triage, specialist, and escalation agents
- Content creation with researcher, writer, and editor agents
- Data analysis with collection, processing, and visualization agents

### 1.4 Related Documents

- [Core Framework Requirements](../core-framework-prd.md)
- [Subagent System (Core Framework Section 3.7)](../core-framework-prd.md#37-subagent-system)

---

## 2. Functional Requirements

### 2.1 Agent Definitions

#### 2.1.1 Agent Role Configuration

```python
@dataclass
class AgentRole:
    name: str                           # Unique identifier
    description: str                    # Human-readable description
    system_prompt: str                  # Role-specific system prompt
    capabilities: list[str]             # What this agent can do
    tools: list[Tool]                   # Tools available to this agent
    model: str | None = None            # Optional role-specific model
    max_tokens: int | None = None       # Token limit for this agent
    temperature: float | None = None    # Generation temperature
    delegation_allowed: bool = True     # Can delegate to other agents
    can_be_delegated_to: bool = True    # Can receive delegations
    priority: int = 0                   # For resource allocation
```

#### 2.1.2 Capability Declaration

Agents must declare their capabilities for task routing:

```python
@dataclass
class Capability:
    name: str                           # e.g., "code_review", "research"
    description: str                    # What the capability entails
    input_types: list[str]              # Types of input handled
    output_types: list[str]             # Types of output produced
    estimated_tokens: int | None        # Typical token usage
    confidence: float = 1.0             # Self-assessed capability level
```

### 2.2 Orchestration Strategies

#### 2.2.1 Strategy Types

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `sequential` | Agents work in predefined order | Pipeline workflows |
| `parallel` | Independent agents work simultaneously | Embarrassingly parallel tasks |
| `hierarchical` | Manager agent delegates to workers | Complex task decomposition |
| `consensus` | Multiple agents collaborate on single output | High-stakes decisions |
| `debate` | Agents argue different positions | Reasoning verification |
| `auction` | Agents bid for tasks based on capability | Dynamic task allocation |
| `blackboard` | Agents contribute to shared workspace | Collaborative problem-solving |
| `custom` | User-defined orchestration logic | Specialized workflows |

#### 2.2.2 Sequential Orchestration

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Agent A │───▶│ Agent B │───▶│ Agent C │───▶│ Output  │
│(Research)│    │ (Draft) │    │ (Edit)  │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │
     └──────────────┴──────────────┘
              Shared Context
```

Configuration:

```python
@dataclass
class SequentialConfig:
    agent_order: list[str]              # Agent names in execution order
    pass_full_context: bool = False     # Pass all prior outputs or just previous
    allow_skip: bool = False            # Skip agents that have no work
    early_termination: bool = True      # Stop if task completed early
```

#### 2.2.3 Parallel Orchestration

```
                    ┌─────────┐
               ┌───▶│ Agent A │───┐
               │    └─────────┘   │
┌─────────┐    │    ┌─────────┐   │    ┌───────────┐
│  Input  │───┼───▶│ Agent B │───┼───▶│ Aggregator│
└─────────┘    │    └─────────┘   │    └───────────┘
               │    ┌─────────┐   │
               └───▶│ Agent C │───┘
                    └─────────┘
```

Configuration:

```python
@dataclass
class ParallelConfig:
    agents: list[str]                   # Agents to run in parallel
    aggregation_strategy: str           # "merge", "vote", "best", "custom"
    max_concurrent: int = 5             # Maximum concurrent agents
    timeout_per_agent: float = 60.0     # Individual agent timeout
    require_all: bool = False           # Wait for all or return on first
    conflict_resolution: str = "vote"   # How to handle conflicting outputs
```

#### 2.2.4 Hierarchical Orchestration

```
                    ┌───────────┐
                    │  Manager  │
                    │   Agent   │
                    └─────┬─────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │ Worker  │   │ Worker  │   │ Worker  │
      │ Agent A │   │ Agent B │   │ Agent C │
      └─────────┘   └─────────┘   └─────────┘
```

Configuration:

```python
@dataclass
class HierarchicalConfig:
    manager: str                        # Manager agent name
    workers: list[str]                  # Worker agent names
    max_delegation_depth: int = 2       # Nested delegation limit
    manager_summarizes: bool = True     # Manager combines worker outputs
    worker_autonomy: str = "low"        # "low", "medium", "high"
    progress_reporting: bool = True     # Workers report progress to manager
```

#### 2.2.5 Consensus Orchestration

```
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Agent A │    │ Agent B │    │ Agent C │
└────┬────┘    └────┬────┘    └────┬────┘
     │              │              │
     └──────────────┼──────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Consensus   │
            │    Builder    │
            └───────────────┘
                    │
                    ▼
             ┌───────────┐
             │  Output   │
             │(Agreed)   │
             └───────────┘
```

Configuration:

```python
@dataclass
class ConsensusConfig:
    agents: list[str]                   # Participating agents
    consensus_threshold: float = 0.67   # Agreement threshold
    max_rounds: int = 3                 # Maximum discussion rounds
    consensus_method: str = "discussion"  # "vote", "discussion", "synthesis"
    tie_breaker: str | None = None      # Agent to break ties
    require_unanimous: bool = False     # All must agree
```

#### 2.2.6 Debate Orchestration

```
┌─────────────┐                    ┌─────────────┐
│  Agent A    │◀──────────────────▶│  Agent B    │
│ (Position 1)│     Debate         │ (Position 2)│
└──────┬──────┘                    └──────┬──────┘
       │                                  │
       └──────────────┬───────────────────┘
                      │
                      ▼
               ┌─────────────┐
               │    Judge    │
               │    Agent    │
               └─────────────┘
```

Configuration:

```python
@dataclass
class DebateConfig:
    debaters: list[str]                 # Debating agents
    judge: str                          # Judging agent
    positions: list[str] | None         # Assigned positions (or auto)
    max_rounds: int = 3                 # Debate rounds
    require_evidence: bool = True       # Must cite evidence
    judge_criteria: list[str]           # Evaluation criteria
```

### 2.3 Communication

#### 2.3.1 Message Types

| Type | Description | Routing |
|------|-------------|---------|
| `request` | Ask another agent to do something | Direct to agent |
| `response` | Reply to a request | Back to requester |
| `broadcast` | Send to all agents | All agents |
| `publish` | Add to shared workspace | Blackboard |
| `subscribe` | Register for updates | Event system |
| `delegate` | Transfer task ownership | Target agent |
| `report` | Status update to manager | Hierarchical |

#### 2.3.2 Message Structure

```python
@dataclass
class AgentMessage:
    id: str
    type: MessageType
    sender: str                         # Agent name
    recipient: str | list[str] | None   # Target(s) or None for broadcast
    content: str
    structured_data: dict | None = None
    priority: int = 0
    requires_response: bool = False
    response_timeout: float | None = None
    thread_id: str | None = None        # For conversation threading
    metadata: dict = field(default_factory=dict)
```

#### 2.3.3 Communication Patterns

| Pattern | Description | Implementation |
|---------|-------------|----------------|
| Request-Response | Synchronous ask and answer | Blocking await |
| Fire-and-Forget | Send without waiting | Async send |
| Publish-Subscribe | Topic-based messaging | Event emitter |
| Streaming | Continuous data flow | Async generator |

### 2.4 State Management

#### 2.4.1 State Scopes

| Scope | Visibility | Persistence |
|-------|------------|-------------|
| `agent_private` | Single agent only | Agent lifetime |
| `agent_shared` | Explicitly shared | Agent lifetime |
| `team_shared` | All agents in team | Session lifetime |
| `blackboard` | All agents, structured | Session lifetime |
| `persistent` | All agents, across sessions | Configurable |

#### 2.4.2 Blackboard System

Shared workspace for collaborative problem-solving:

```python
@dataclass
class BlackboardEntry:
    key: str
    value: Any
    author: str                         # Agent that wrote it
    timestamp: datetime
    entry_type: str                     # "fact", "hypothesis", "solution", etc.
    confidence: float = 1.0
    supersedes: str | None = None       # Key this replaces
    references: list[str] = field(default_factory=list)

class Blackboard:
    async def write(self, key: str, value: Any, entry_type: str) -> None: ...
    async def read(self, key: str) -> BlackboardEntry | None: ...
    async def query(self, entry_type: str | None = None) -> list[BlackboardEntry]: ...
    async def subscribe(self, pattern: str, callback: Callable) -> None: ...
    def get_history(self, key: str) -> list[BlackboardEntry]: ...
```

#### 2.4.3 Context Isolation

- Each agent maintains its own context window
- Shared information explicitly passed via messages or blackboard
- Agent summaries generated when context passed between agents
- Configurable context inheritance from parent/manager

### 2.5 Task Distribution

#### 2.5.1 Task Routing

| Method | Description |
|--------|-------------|
| `explicit` | Orchestrator assigns to specific agent |
| `capability_match` | Route based on declared capabilities |
| `load_balance` | Distribute based on agent workload |
| `auction` | Agents bid based on suitability |
| `round_robin` | Rotate through available agents |

#### 2.5.2 Task Structure

```python
@dataclass
class AgentTask:
    id: str
    description: str
    required_capabilities: list[str]
    assigned_to: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    deadline: datetime | None = None
    dependencies: list[str] = field(default_factory=list)
    result: Any = None
    parent_task: str | None = None      # For subtasks
```

### 2.6 Coordination Protocols

#### 2.6.1 Handoff Protocol

When one agent transfers work to another:

1. Source agent summarizes current state
2. Source agent identifies handoff target
3. Handoff request sent with context
4. Target agent acknowledges and assumes responsibility
5. Source agent marks task as handed off
6. Target agent proceeds with task

#### 2.6.2 Escalation Protocol

When an agent cannot complete a task:

1. Agent identifies inability to proceed
2. Agent packages current state and blockers
3. Escalation sent to manager or designated agent
4. Receiving agent evaluates and either:
   - Provides guidance
   - Reassigns task
   - Takes over directly
   - Escalates further

#### 2.6.3 Synchronization Points

```python
@dataclass
class SyncPoint:
    name: str
    required_agents: list[str]          # Who must reach this point
    timeout: float                      # Max wait time
    on_timeout: str = "proceed"         # "proceed", "abort", "notify"
    data_exchange: bool = True          # Share data at sync point
```

---

## 3. Interface Requirements

### 3.1 Pattern Class

```python
class MultiAgentPattern(AgentPattern):
    """Multi-agent orchestration pattern."""
    
    def __init__(
        self,
        agents: list[AgentRole],
        orchestration: OrchestrationStrategy,
        communication: CommunicationConfig | None = None,
        state_management: StateConfig | None = None,
        task_routing: str = "capability_match",
        enable_blackboard: bool = False,
        max_concurrent_agents: int = 5,
        global_timeout: float | None = None,
    ) -> None: ...
    
    async def execute(
        self,
        query: str,
    ) -> MultiAgentResult: ...
    
    async def send_message(
        self,
        message: AgentMessage,
    ) -> AgentMessage | None: ...
    
    async def broadcast(
        self,
        content: str,
        sender: str,
    ) -> None: ...
    
    def get_agent(self, name: str) -> AgentInstance | None: ...
    
    def get_blackboard(self) -> Blackboard | None: ...
    
    def get_message_history(self) -> list[AgentMessage]: ...
    
    def get_task_status(self) -> dict[str, TaskStatus]: ...
```

### 3.2 Result Structure

```python
@dataclass
class MultiAgentResult:
    success: bool
    final_output: str | None
    agent_outputs: dict[str, Any]       # Per-agent results
    message_history: list[AgentMessage]
    task_completion: dict[str, TaskStatus]
    blackboard_state: dict[str, Any] | None
    orchestration_trace: OrchestrationTrace
    token_usage: dict[str, TokenUsage]  # Per-agent token usage
    total_token_usage: TokenUsage
    execution_time: float
    agent_execution_times: dict[str, float]
```

### 3.3 Orchestration Trace

```python
@dataclass
class OrchestrationTrace:
    events: list[OrchestrationEvent]
    decisions: list[OrchestrationDecision]
    agent_activations: list[AgentActivation]
    message_flow: list[MessageFlowEntry]
    sync_points_reached: list[str]
    
@dataclass
class OrchestrationEvent:
    timestamp: datetime
    event_type: str                     # "agent_start", "message_sent", etc.
    agent: str | None
    details: dict
```

### 3.4 Agent Instance

```python
class AgentInstance:
    """Runtime instance of an agent in the multi-agent system."""
    
    @property
    def name(self) -> str: ...
    
    @property
    def role(self) -> AgentRole: ...
    
    @property
    def status(self) -> AgentStatus: ...
    
    async def run(self, task: str) -> str: ...
    
    async def receive_message(self, message: AgentMessage) -> None: ...
    
    def get_context_summary(self) -> str: ...
    
    def get_token_usage(self) -> TokenUsage: ...
```

---

## 4. Behavioral Requirements

### 4.1 Agent Coordination

- Agents should respect role boundaries
- Communication should be purposeful, not excessive
- Delegation should match task to capability
- Conflicts should be resolved through defined protocols

### 4.2 Resource Management

- Token budgets distributed fairly across agents
- Concurrent agent limit enforced
- Long-running agents do not block others
- Failed agents do not crash the system

### 4.3 Output Quality

- Final output should synthesize agent contributions
- Contradictions between agents should be resolved
- Attribution of contributions should be possible
- Quality should exceed single-agent baseline for suitable tasks

### 4.4 Failure Handling

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Agent Crash | Exception/timeout | Restart or redistribute work |
| Deadlock | Circular wait detection | Timeout and intervention |
| Livelock | No progress detection | Force resolution |
| Communication Failure | Message timeout | Retry then escalate |
| Consensus Failure | Max rounds reached | Tie-breaker or abort |

---

## 5. Example Usage

### 5.1 Sequential Pipeline

```python
from agent_framework import Agent
from agent_framework.patterns import MultiAgentPattern, SequentialConfig
from agent_framework.multi_agent import AgentRole

roles = [
    AgentRole(
        name="researcher",
        description="Gathers information from sources",
        system_prompt="You are a research specialist. Find relevant information.",
        tools=[web_search],
        capabilities=["research", "fact_finding"]
    ),
    AgentRole(
        name="writer",
        description="Creates content from research",
        system_prompt="You are a technical writer. Create clear documentation.",
        tools=[write_file],
        capabilities=["writing", "documentation"]
    ),
    AgentRole(
        name="editor",
        description="Reviews and improves content",
        system_prompt="You are an editor. Improve clarity and correctness.",
        tools=[read_file, write_file],
        capabilities=["editing", "review"]
    ),
]

pattern = MultiAgentPattern(
    agents=roles,
    orchestration=SequentialConfig(
        agent_order=["researcher", "writer", "editor"],
        pass_full_context=True
    )
)

result = await pattern.execute(
    "Create a guide explaining how to use Docker Compose"
)
```

### 5.2 Hierarchical with Manager

```python
roles = [
    AgentRole(
        name="project_manager",
        description="Coordinates the team and synthesizes results",
        system_prompt="You manage a team of specialists. Delegate appropriately.",
        capabilities=["coordination", "synthesis"]
    ),
    AgentRole(
        name="frontend_dev",
        description="Handles UI and frontend code",
        system_prompt="You are a frontend specialist.",
        tools=[read_file, write_file],
        capabilities=["frontend", "react", "css"]
    ),
    AgentRole(
        name="backend_dev",
        description="Handles API and backend code",
        system_prompt="You are a backend specialist.",
        tools=[read_file, write_file, run_bash],
        capabilities=["backend", "python", "api"]
    ),
]

pattern = MultiAgentPattern(
    agents=roles,
    orchestration=HierarchicalConfig(
        manager="project_manager",
        workers=["frontend_dev", "backend_dev"],
        manager_summarizes=True
    )
)

result = await pattern.execute(
    "Add a user profile page with API endpoint"
)
```

### 5.3 Debate for Verification

```python
roles = [
    AgentRole(name="advocate", system_prompt="Argue in favor of the proposition."),
    AgentRole(name="critic", system_prompt="Argue against the proposition."),
    AgentRole(name="judge", system_prompt="Evaluate arguments and decide."),
]

pattern = MultiAgentPattern(
    agents=roles,
    orchestration=DebateConfig(
        debaters=["advocate", "critic"],
        judge="judge",
        max_rounds=3,
        require_evidence=True
    )
)

result = await pattern.execute(
    "Should we migrate from REST to GraphQL for our API?"
)

# Access the debate transcript
for msg in result.message_history:
    print(f"{msg.sender}: {msg.content[:100]}...")
```

### 5.4 Parallel with Blackboard

```python
pattern = MultiAgentPattern(
    agents=[
        AgentRole(name="data_collector", capabilities=["data_collection"]),
        AgentRole(name="analyzer_1", capabilities=["analysis"]),
        AgentRole(name="analyzer_2", capabilities=["analysis"]),
        AgentRole(name="synthesizer", capabilities=["synthesis"]),
    ],
    orchestration=ParallelConfig(
        agents=["data_collector", "analyzer_1", "analyzer_2"],
        aggregation_strategy="custom"
    ),
    enable_blackboard=True
)

# Agents can write to shared blackboard
# blackboard.write("dataset_stats", stats, "fact")
# blackboard.write("anomalies", anomalies, "finding")
# Synthesizer reads all findings and produces final output
```

---

## 6. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Orchestration Correctness | Agents execute in correct order/parallel 100% of time |
| Communication Reliability | Messages delivered 99.9%+ of time |
| Task Completion | Multi-agent tasks complete 90%+ of time |
| Quality Improvement | Output quality exceeds single agent for suitable tasks |
| Resource Efficiency | Token usage overhead < 20% vs sequential single-agent |
| Failure Recovery | System recovers from single-agent failure 95%+ of time |

---

## 7. Open Questions

1. How to handle agents that need different model backends?
2. Should there be "meta-agents" that can create new agents dynamically?
3. How to evaluate the contribution of individual agents to the final output?
4. What's the optimal team size for different task types?
5. How to handle security when agents have different trust levels?
6. Should agents be able to refuse tasks delegated to them?
