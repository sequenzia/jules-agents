# Agent Pattern Requirements: Blackboard System (Shared Workspace)

## 1. Overview

### 1.1 Pattern Description

The Blackboard System pattern is a collaborative problem-solving architecture where multiple specialized knowledge sources (agents) work together by reading from and writing to a shared workspace called the "blackboard." Unlike direct agent-to-agent communication, all interaction occurs through this central repository, enabling asynchronous collaboration, emergent problem-solving, and opportunistic contribution from agents whose expertise becomes relevant as the solution evolves.

### 1.2 Key Characteristics

- Central shared workspace accessible to all participating agents
- Specialized knowledge sources that contribute partial solutions based on their expertise
- Control component that schedules and coordinates agent activation
- Opportunistic problem-solving where agents contribute when their expertise matches the current state
- Incremental solution building through iterative refinement
- Support for ill-defined problems where the solution path is not known in advance

### 1.3 Historical Context

The blackboard architecture originated in the 1970s with the Hearsay-II speech recognition system and was formalized in the 1980s. Classic implementations include BB1 (Stanford), GBB (Generic Blackboard), and applications in robotics, medical diagnosis, and satellite mission control. The pattern has experienced renewed interest for LLM-based multi-agent systems due to its natural fit for collaborative AI problem-solving.

### 1.4 Use Cases

- Complex reasoning tasks requiring diverse expertise (research, analysis, synthesis)
- Software development with specialized agents (architect, coder, reviewer, tester)
- Data science workflows with discovery, analysis, and visualization agents
- Medical diagnosis integrating multiple specialist perspectives
- Scientific discovery and hypothesis generation
- Game AI coordinating perception, planning, and action agents
- Any problem where the solution emerges from collaborative contribution

### 1.5 References

- [Blackboard Systems (Nii, 1986)](https://www.aaai.org/Papers/Workshops/1986/WS-86-06/WS86-06-001.pdf) - Original formalization
- [BB1: An Architecture for Control (Hayes-Roth, 1985)](https://www.sciencedirect.com/science/article/pii/0004370285900163)
- [LLM Multi-Agent Blackboard Systems (arXiv 2507.01701)](https://arxiv.org/abs/2507.01701) - Modern LLM adaptation
- [LLM-based Multi-Agent Blackboard for Data Science (arXiv 2510.01285)](https://arxiv.org/abs/2510.01285)

### 1.6 Related Documents

- [Core Framework Requirements](../core-framework-prd.md)
- [Multi-Agent Systems Pattern](./pattern-multi-agent.md)

---

## 2. Functional Requirements

### 2.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BLACKBOARD SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         CONTROL COMPONENT                           │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │    │
│  │  │   Monitor    │  │  Scheduler   │  │  Activation Manager      │   │    │
│  │  │ (State Watch)│  │  (Ordering)  │  │  (KS Selection)          │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                           BLACKBOARD                                │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                      PUBLIC SPACE                           │    │    │
│  │  │  • Problem State    • Hypotheses    • Partial Solutions     │    │    │
│  │  │  • Facts            • Evidence      • Final Solution        │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                     PRIVATE SPACES                          │    │    │
│  │  │  • Agent Debates    • Self-Reflection    • Working Memory   │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    ▲                                        │
│                    ┌───────────────┼───────────────┐                        │
│                    │               │               │                        │
│  ┌─────────────────┴───┐  ┌────────┴────────┐  ┌───┴─────────────────┐      │
│  │  Knowledge Source 1 │  │ Knowledge Source│  │  Knowledge Source N │      │
│  │  (e.g., Planner)    │  │ (e.g., Analyst) │  │  (e.g., Critic)     │      │
│  │  ┌───────────────┐  │  │ ┌─────────────┐ │  │  ┌───────────────┐  │      │
│  │  │ Preconditions │  │  │ │Preconditions│ │  │  │ Preconditions │  │      │
│  │  │ Actions       │  │  │ │Actions      │ │  │  │ Actions       │  │      │
│  │  │ Expertise     │  │  │ │Expertise    │ │  │  │ Expertise     │  │      │
│  │  └───────────────┘  │  │ └─────────────┘ │  │  └───────────────┘  │      │
│  └─────────────────────┘  └─────────────────┘  └─────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Blackboard Component

The blackboard is the central shared workspace where all problem-solving data resides.

#### 2.2.1 Blackboard Structure

```python
@dataclass
class BlackboardEntry:
    id: str                              # Unique entry identifier
    entry_type: EntryType                # Type of entry (see below)
    content: Any                         # Entry content
    author: str                          # Knowledge source that created it
    timestamp: datetime                  # Creation time
    confidence: float = 1.0              # Confidence score (0.0 to 1.0)
    supersedes: str | None = None        # ID of entry this replaces
    references: list[str] = field(default_factory=list)  # Related entry IDs
    level: int = 0                       # Abstraction level
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

class EntryType(Enum):
    PROBLEM = "problem"                  # Initial problem specification
    FACT = "fact"                        # Established fact
    HYPOTHESIS = "hypothesis"            # Proposed hypothesis
    EVIDENCE = "evidence"                # Supporting/contradicting evidence
    PARTIAL_SOLUTION = "partial"         # Partial solution component
    PLAN = "plan"                        # Plan or strategy
    QUESTION = "question"                # Question requiring answer
    ANSWER = "answer"                    # Answer to a question
    CRITIQUE = "critique"                # Critique or feedback
    CONFLICT = "conflict"                # Identified conflict
    RESOLUTION = "resolution"            # Conflict resolution
    SOLUTION = "solution"                # Final or candidate solution
    CONTROL = "control"                  # Control/meta information
```

#### 2.2.2 Blackboard Spaces

| Space | Description | Visibility |
|-------|-------------|------------|
| Public | Main workspace visible to all agents | All agents |
| Private | Agent-specific working areas | Designated agents only |
| Control | Metadata for scheduling and coordination | Control component |
| Archive | Historical entries for reference | All agents (read-only) |

#### 2.2.3 Abstraction Levels

The blackboard can be organized into hierarchical abstraction levels:

```python
@dataclass
class AbstractionLevel:
    level: int                           # Level number (0 = lowest)
    name: str                            # e.g., "raw_data", "features", "hypotheses"
    description: str
    entry_types: list[EntryType]         # Types allowed at this level

# Example levels for a research task:
# Level 0: Raw data, observations
# Level 1: Extracted facts, features
# Level 2: Hypotheses, patterns
# Level 3: Partial solutions, conclusions
# Level 4: Final solution, synthesis
```

#### 2.2.4 Blackboard Operations

```python
class Blackboard:
    # Write operations
    async def post(self, entry: BlackboardEntry) -> str: ...
    async def update(self, entry_id: str, updates: dict) -> None: ...
    async def supersede(self, old_id: str, new_entry: BlackboardEntry) -> str: ...
    async def delete(self, entry_id: str, reason: str) -> None: ...

    # Read operations
    async def get(self, entry_id: str) -> BlackboardEntry | None: ...
    async def query(
        self,
        entry_type: EntryType | None = None,
        author: str | None = None,
        level: int | None = None,
        tags: list[str] | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[BlackboardEntry]: ...
    async def get_current_state(self) -> BlackboardState: ...
    async def get_history(self, entry_id: str) -> list[BlackboardEntry]: ...

    # Subscription operations
    async def subscribe(
        self,
        callback: Callable,
        entry_types: list[EntryType] | None = None,
        levels: list[int] | None = None,
    ) -> str: ...
    async def unsubscribe(self, subscription_id: str) -> None: ...

    # Space management
    async def create_private_space(self, name: str, agents: list[str]) -> str: ...
    async def get_space(self, space_id: str) -> BlackboardSpace: ...
```

#### 2.2.5 Blackboard Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_entries` | `int` | `10000` | Maximum entries before archival |
| `enable_versioning` | `bool` | `True` | Track entry history |
| `enable_subscriptions` | `bool` | `True` | Allow event subscriptions |
| `abstraction_levels` | `list` | `None` | Custom abstraction hierarchy |
| `conflict_detection` | `bool` | `True` | Auto-detect conflicting entries |
| `archival_policy` | `str` | `"age"` | "age", "relevance", "manual" |
| `archival_threshold` | `int` | `1000` | Entries before archival triggers |

### 2.3 Knowledge Sources (Agents)

Knowledge sources are specialized agents that contribute to problem-solving.

#### 2.3.1 Knowledge Source Definition

```python
@dataclass
class KnowledgeSource:
    name: str                            # Unique identifier
    description: str                     # What this KS does
    expertise: list[str]                 # Areas of expertise

    # Activation conditions
    preconditions: list[Precondition]    # When this KS can activate
    trigger_types: list[EntryType]       # Entry types that trigger evaluation
    trigger_levels: list[int] | None     # Abstraction levels to monitor

    # Execution
    system_prompt: str                   # LLM system prompt
    tools: list[Tool] = None             # Available tools
    model: str | None = None             # Specific model (optional)

    # Scheduling hints
    priority: int = 0                    # Base priority
    estimated_cost: float = 1.0          # Relative cost estimate
    max_frequency: float | None = None   # Max activations per minute

    # Output
    output_types: list[EntryType]        # Types this KS produces
    output_levels: list[int]             # Levels this KS writes to

@dataclass
class Precondition:
    condition_type: str                  # "entry_exists", "entry_count", "pattern", "custom"
    parameters: dict[str, Any]           # Condition-specific parameters
    description: str                     # Human-readable description
```

#### 2.3.2 Precondition Types

| Type | Description | Parameters |
|------|-------------|------------|
| `entry_exists` | Specific entry type exists | `entry_type`, `min_confidence` |
| `entry_count` | Minimum entries of type | `entry_type`, `min_count` |
| `entry_absent` | Entry type does not exist | `entry_type` |
| `pattern_match` | Content matches pattern | `pattern`, `entry_type` |
| `level_complete` | Level has sufficient entries | `level`, `min_entries` |
| `conflict_exists` | Unresolved conflict present | `entry_types` |
| `time_elapsed` | Time since last activation | `min_seconds` |
| `custom` | Custom evaluation function | `function` |

#### 2.3.3 Knowledge Source Categories

| Category | Description | Examples |
|----------|-------------|----------|
| Domain Experts | Contribute domain-specific knowledge | Researcher, Analyst, Domain Specialist |
| Processors | Transform or analyze data | Summarizer, Extractor, Calculator |
| Generators | Create new hypotheses or solutions | Planner, Hypothesis Generator, Synthesizer |
| Critics | Evaluate and critique contributions | Reviewer, Validator, Fact-Checker |
| Resolvers | Handle conflicts and inconsistencies | Conflict Resolver, Arbiter, Consensus Builder |
| Meta-level | Control and coordination | Decider, Cleaner, Focus Manager |

#### 2.3.4 Built-in Knowledge Sources

| Knowledge Source | Description | Trigger |
|------------------|-------------|---------|
| `Planner` | Decomposes problems into steps | Problem entry posted |
| `Researcher` | Gathers information using tools | Question or gap identified |
| `Analyst` | Analyzes data and extracts insights | Raw data available |
| `Synthesizer` | Combines partial solutions | Multiple partial solutions exist |
| `Critic` | Evaluates hypotheses and solutions | New hypothesis or solution posted |
| `ConflictResolver` | Resolves contradictions | Conflict entry created |
| `Decider` | Determines if solution is complete | Sufficient evidence accumulated |
| `Cleaner` | Removes redundant/outdated entries | Entry count exceeds threshold |

### 2.4 Control Component

The control component manages the problem-solving process.

#### 2.4.1 Control Component Structure

```python
class ControlComponent:
    def __init__(
        self,
        blackboard: Blackboard,
        knowledge_sources: list[KnowledgeSource],
        scheduler: Scheduler,
        strategy: ControlStrategy,
    ) -> None: ...

    async def run(self, problem: str, max_rounds: int = 50) -> BlackboardResult: ...
    async def step(self) -> StepResult: ...
    async def stop(self, reason: str) -> None: ...

    # Monitoring
    def get_pending_activations(self) -> list[KSActivation]: ...
    def get_execution_history(self) -> list[ExecutionRecord]: ...
```

#### 2.4.2 Scheduling Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `priority_based` | Highest priority KS first | Clear importance hierarchy |
| `round_robin` | Cycle through applicable KSs | Fair participation |
| `opportunistic` | Most relevant to current state | Dynamic problems |
| `focus_based` | Concentrate on specific area | Directed problem-solving |
| `cost_aware` | Balance quality and cost | Resource-constrained |
| `llm_based` | LLM selects next KS | Complex decision-making |

#### 2.4.3 Scheduler Configuration

```python
@dataclass
class SchedulerConfig:
    strategy: str = "opportunistic"
    max_concurrent: int = 1              # Parallel KS executions
    activation_timeout: float = 60.0     # Timeout per KS

    # Priority modifiers
    recency_weight: float = 0.3          # Boost for recently triggered KSs
    relevance_weight: float = 0.5        # Boost for matching preconditions
    cost_weight: float = 0.2             # Penalty for expensive KSs

    # Focus control
    enable_focus: bool = True            # Allow focus directives
    focus_decay: float = 0.9             # Focus priority decay per round

    # Starvation prevention
    max_consecutive_same_ks: int = 3     # Prevent single KS dominance
    min_ks_gap_rounds: int = 2           # Minimum rounds between same KS
```

#### 2.4.4 Knowledge Source Activation Record (KSAR)

```python
@dataclass
class KSActivation:
    id: str
    knowledge_source: str
    trigger_entries: list[str]           # Entries that triggered this
    priority: float                      # Computed priority
    created_at: datetime
    status: ActivationStatus             # PENDING, RUNNING, COMPLETED, FAILED

class ActivationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
```

### 2.5 Control Strategies

#### 2.5.1 Opportunistic Control

The default strategy where the most relevant KS is selected based on current blackboard state:

```python
@dataclass
class OpportunisticStrategy:
    relevance_threshold: float = 0.5     # Minimum relevance to consider

    async def select_next(
        self,
        blackboard: Blackboard,
        pending: list[KSActivation],
    ) -> KSActivation | None:
        """Select the most relevant KS based on current state."""
        ...
```

#### 2.5.2 Focus-Directed Control

Allows directing attention to specific areas:

```python
@dataclass
class FocusDirective:
    target: str                          # What to focus on
    target_type: str                     # "level", "entry_type", "tag", "ks"
    priority_boost: float = 2.0          # Priority multiplier
    duration_rounds: int = 5             # How long to maintain focus

class FocusStrategy:
    async def set_focus(self, directive: FocusDirective) -> None: ...
    async def clear_focus(self) -> None: ...
```

#### 2.5.3 LLM-Based Control

Uses an LLM to make sophisticated scheduling decisions:

```python
@dataclass
class LLMControlStrategy:
    model: str                           # Model for control decisions
    decision_prompt: str                 # Prompt template
    include_history: bool = True         # Include execution history
    max_history_entries: int = 20        # History context limit
```

### 2.6 Termination Conditions

#### 2.6.1 Termination Types

| Condition | Description | Configuration |
|-----------|-------------|---------------|
| `solution_found` | Decider KS marks solution complete | `require_confidence: float` |
| `max_rounds` | Maximum iteration count reached | `max_rounds: int` |
| `no_progress` | No new entries for N rounds | `stall_rounds: int` |
| `token_budget` | Token limit reached | `max_tokens: int` |
| `time_limit` | Time limit reached | `timeout_seconds: float` |
| `consensus` | Agents reach agreement | `consensus_threshold: float` |
| `custom` | User-defined condition | `condition_function: Callable` |

#### 2.6.2 Termination Configuration

```python
@dataclass
class TerminationConfig:
    conditions: list[TerminationCondition]
    require_all: bool = False            # All conditions or any

    # Solution validation
    validate_solution: bool = True
    validation_ks: str | None = None     # KS to validate final solution
    min_solution_confidence: float = 0.8
```

### 2.7 Conflict Resolution

#### 2.7.1 Conflict Detection

```python
@dataclass
class Conflict:
    id: str
    entry_ids: list[str]                 # Conflicting entries
    conflict_type: ConflictType
    description: str
    detected_at: datetime
    resolved: bool = False
    resolution_id: str | None = None

class ConflictType(Enum):
    CONTRADICTION = "contradiction"       # Mutually exclusive claims
    INCONSISTENCY = "inconsistency"      # Logically inconsistent
    DISAGREEMENT = "disagreement"        # Different conclusions
    DUPLICATION = "duplication"          # Redundant entries
```

#### 2.7.2 Resolution Strategies

| Strategy | Description |
|----------|-------------|
| `voting` | KSs vote on correct entry |
| `confidence` | Highest confidence wins |
| `evidence` | Most supported by evidence wins |
| `debate` | KSs debate in private space |
| `arbiter` | Designated KS decides |
| `synthesis` | Combine into unified view |

### 2.8 Private Spaces

#### 2.8.1 Private Space Uses

| Use Case | Description |
|----------|-------------|
| Debate | Two or more KSs discuss disagreement |
| Self-Reflection | KS refines its own contribution |
| Working Memory | KS stores intermediate calculations |
| Collaboration | Subset of KSs work on subtask |

#### 2.8.2 Private Space Configuration

```python
@dataclass
class PrivateSpace:
    id: str
    name: str
    participants: list[str]              # Allowed KS names
    purpose: str
    created_at: datetime
    max_entries: int = 100
    auto_summarize: bool = True          # Summarize to public on close
    summary_ks: str | None = None        # KS to create summary
```

---

## 3. Interface Requirements

### 3.1 Pattern Class

```python
class BlackboardPattern(AgentPattern):
    """Blackboard System pattern for collaborative problem-solving."""

    def __init__(
        self,
        knowledge_sources: list[KnowledgeSource],
        blackboard_config: BlackboardConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        control_strategy: str = "opportunistic",
        termination_config: TerminationConfig | None = None,
        conflict_resolution: str = "evidence",
        enable_private_spaces: bool = True,
        built_in_ks: list[str] | None = None,  # Include built-in KSs
    ) -> None: ...

    async def execute(
        self,
        problem: str,
        initial_facts: list[dict] | None = None,
    ) -> BlackboardResult: ...

    async def step(self) -> StepResult: ...

    async def add_knowledge_source(self, ks: KnowledgeSource) -> None: ...
    async def remove_knowledge_source(self, name: str) -> None: ...

    def get_blackboard(self) -> Blackboard: ...
    def get_control(self) -> ControlComponent: ...
    def get_execution_trace(self) -> ExecutionTrace: ...
```

### 3.2 Result Structure

```python
@dataclass
class BlackboardResult:
    success: bool
    solution: str | None
    solution_entry: BlackboardEntry | None
    confidence: float

    # Execution details
    rounds: int
    termination_reason: str
    execution_trace: ExecutionTrace

    # Blackboard state
    final_state: BlackboardState
    entry_count: int
    entries_by_type: dict[EntryType, int]
    entries_by_ks: dict[str, int]

    # Conflicts
    conflicts_detected: int
    conflicts_resolved: int
    unresolved_conflicts: list[Conflict]

    # Resource usage
    token_usage: TokenUsage
    token_usage_by_ks: dict[str, TokenUsage]
    execution_time: float

@dataclass
class ExecutionTrace:
    rounds: list[RoundRecord]
    ks_activations: list[KSActivation]
    entry_timeline: list[tuple[datetime, str, EntryType]]
    focus_history: list[FocusDirective]
    conflicts: list[Conflict]

@dataclass
class RoundRecord:
    round_number: int
    activated_ks: str
    trigger_entries: list[str]
    new_entries: list[str]
    modified_entries: list[str]
    deleted_entries: list[str]
    duration_ms: float
    token_usage: TokenUsage
```

### 3.3 Callbacks

```python
class BlackboardCallbacks:
    async def on_entry_posted(
        self,
        entry: BlackboardEntry
    ) -> None:
        """Called when new entry is posted."""
        pass

    async def on_ks_activated(
        self,
        ks: str,
        activation: KSActivation
    ) -> bool:
        """Called before KS executes. Return False to skip."""
        return True

    async def on_ks_completed(
        self,
        ks: str,
        entries_created: list[BlackboardEntry],
    ) -> None:
        """Called after KS completes."""
        pass

    async def on_conflict_detected(
        self,
        conflict: Conflict,
    ) -> str:
        """Called on conflict. Return resolution strategy."""
        return "evidence"

    async def on_round_complete(
        self,
        round_record: RoundRecord,
    ) -> bool:
        """Called after each round. Return False to stop."""
        return True

    async def on_solution_proposed(
        self,
        solution: BlackboardEntry,
    ) -> bool:
        """Called when solution proposed. Return False to continue."""
        return True
```

---

## 4. Behavioral Requirements

### 4.1 Knowledge Source Behavior

- KSs should only activate when their preconditions are genuinely met
- KSs should contribute entries appropriate to their expertise
- KSs should reference relevant existing entries
- KSs should indicate confidence levels accurately
- KSs should not duplicate existing entries unnecessarily

### 4.2 Control Component Behavior

- Scheduler should prevent any single KS from dominating
- Control should detect and handle stalled states
- Focus directives should appropriately bias selection
- Conflicts should be detected and flagged for resolution
- Termination should occur when conditions are met

### 4.3 Blackboard Behavior

- Entries should be immutable once posted (use supersede for updates)
- Queries should be efficient even with many entries
- Subscriptions should trigger promptly on relevant changes
- Private spaces should maintain strict access control
- Archival should preserve ability to reference historical entries

### 4.4 Efficiency Considerations

- Avoid redundant KS activations for unchanged state
- Prune low-value entries to control blackboard size
- Summarize verbose entries when appropriate
- Cache frequently-accessed queries
- Batch related operations when possible

---

## 5. Example Usage

### 5.1 Research Task with Blackboard

```python
from agent_framework.patterns import BlackboardPattern
from agent_framework.blackboard import KnowledgeSource, Precondition

# Define knowledge sources
knowledge_sources = [
    KnowledgeSource(
        name="researcher",
        description="Searches for relevant information",
        expertise=["research", "information_gathering"],
        preconditions=[
            Precondition(
                condition_type="entry_exists",
                parameters={"entry_type": "question"},
                description="Questions exist to research"
            )
        ],
        trigger_types=[EntryType.QUESTION, EntryType.PROBLEM],
        tools=[web_search],
        system_prompt="You are a research specialist. Find relevant information.",
        output_types=[EntryType.FACT, EntryType.EVIDENCE],
    ),
    KnowledgeSource(
        name="analyst",
        description="Analyzes facts and generates hypotheses",
        expertise=["analysis", "reasoning"],
        preconditions=[
            Precondition(
                condition_type="entry_count",
                parameters={"entry_type": "fact", "min_count": 3},
                description="Sufficient facts to analyze"
            )
        ],
        trigger_types=[EntryType.FACT],
        system_prompt="You are an analyst. Identify patterns and generate hypotheses.",
        output_types=[EntryType.HYPOTHESIS, EntryType.PARTIAL_SOLUTION],
    ),
    KnowledgeSource(
        name="critic",
        description="Evaluates hypotheses and identifies weaknesses",
        expertise=["evaluation", "critique"],
        preconditions=[
            Precondition(
                condition_type="entry_exists",
                parameters={"entry_type": "hypothesis"},
                description="Hypotheses exist to evaluate"
            )
        ],
        trigger_types=[EntryType.HYPOTHESIS],
        system_prompt="You are a critical evaluator. Find weaknesses in hypotheses.",
        output_types=[EntryType.CRITIQUE, EntryType.QUESTION],
    ),
    KnowledgeSource(
        name="synthesizer",
        description="Combines partial solutions into final answer",
        expertise=["synthesis", "integration"],
        preconditions=[
            Precondition(
                condition_type="entry_count",
                parameters={"entry_type": "partial", "min_count": 2},
                description="Multiple partial solutions to combine"
            )
        ],
        trigger_types=[EntryType.PARTIAL_SOLUTION],
        system_prompt="You synthesize partial solutions into coherent answers.",
        output_types=[EntryType.SOLUTION],
    ),
]

# Create pattern
pattern = BlackboardPattern(
    knowledge_sources=knowledge_sources,
    built_in_ks=["decider", "cleaner"],  # Add built-in KSs
    control_strategy="opportunistic",
    termination_config=TerminationConfig(
        conditions=[
            TerminationCondition("solution_found", require_confidence=0.8),
            TerminationCondition("max_rounds", max_rounds=30),
        ]
    ),
)

# Execute
result = await pattern.execute(
    problem="What are the key factors affecting renewable energy adoption in developing countries?",
    initial_facts=[
        {"type": "fact", "content": "Solar panel costs have decreased 90% since 2010"},
    ]
)

# Examine the solution and trace
print(f"Solution: {result.solution}")
print(f"Confidence: {result.confidence}")
print(f"Rounds: {result.rounds}")
print(f"KS activations: {result.entries_by_ks}")
```

### 5.2 Software Development Blackboard

```python
knowledge_sources = [
    KnowledgeSource(
        name="architect",
        description="Designs system architecture",
        expertise=["architecture", "design_patterns"],
        trigger_types=[EntryType.PROBLEM],
        system_prompt="You are a software architect. Design clean, scalable solutions.",
        output_types=[EntryType.PLAN, EntryType.PARTIAL_SOLUTION],
    ),
    KnowledgeSource(
        name="coder",
        description="Implements code based on designs",
        expertise=["coding", "implementation"],
        preconditions=[
            Precondition("entry_exists", {"entry_type": "plan"}, "Design exists")
        ],
        trigger_types=[EntryType.PLAN],
        tools=[write_file, read_file],
        system_prompt="You implement clean, well-tested code.",
        output_types=[EntryType.PARTIAL_SOLUTION],
    ),
    KnowledgeSource(
        name="reviewer",
        description="Reviews code for quality and issues",
        expertise=["code_review", "best_practices"],
        trigger_types=[EntryType.PARTIAL_SOLUTION],
        system_prompt="You review code for bugs, security issues, and best practices.",
        output_types=[EntryType.CRITIQUE, EntryType.QUESTION],
    ),
    KnowledgeSource(
        name="tester",
        description="Writes and runs tests",
        expertise=["testing", "quality_assurance"],
        trigger_types=[EntryType.PARTIAL_SOLUTION],
        tools=[run_bash],
        system_prompt="You write comprehensive tests and verify code works correctly.",
        output_types=[EntryType.EVIDENCE, EntryType.CRITIQUE],
    ),
]

pattern = BlackboardPattern(
    knowledge_sources=knowledge_sources,
    enable_private_spaces=True,  # Allow debate between reviewer and coder
    conflict_resolution="debate",
)

result = await pattern.execute(
    problem="Implement a rate limiter with sliding window algorithm"
)
```

### 5.3 Blackboard with Custom Control

```python
from agent_framework.blackboard import LLMControlStrategy

# LLM-based control for sophisticated scheduling
control_strategy = LLMControlStrategy(
    model="llama3.2:8b",
    decision_prompt="""
    Given the current blackboard state and pending knowledge source activations,
    select the most appropriate knowledge source to activate next.

    Current state summary: {state_summary}
    Pending activations: {pending_activations}
    Recent history: {recent_history}

    Consider:
    - What knowledge is missing?
    - Which KS can best advance the solution?
    - Are there conflicts that need resolution?

    Select the KS name to activate:
    """,
)

pattern = BlackboardPattern(
    knowledge_sources=knowledge_sources,
    control_strategy=control_strategy,
)
```

---

## 6. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Convergence | System reaches solution in 85%+ of solvable problems |
| Efficiency | Average rounds to solution within 2x optimal |
| Conflict Resolution | 90%+ of detected conflicts resolved |
| KS Utilization | All relevant KSs activated at least once in 80%+ of runs |
| Entry Quality | < 20% of entries marked as redundant/low-value |
| Stall Prevention | < 5% of runs terminate due to no progress |

---

## 7. Open Questions

1. How to handle knowledge sources with vastly different response times?
2. Should the blackboard support typed/structured entries beyond free text?
3. How to efficiently scale to hundreds of knowledge sources?
4. What's the optimal strategy for balancing exploration vs. exploitation?
5. How to handle knowledge sources that consistently produce low-quality entries?
6. Should there be "meta-knowledge sources" that can create new KSs dynamically?
7. How to integrate external knowledge bases as passive knowledge sources?
