# Agent Pattern Requirements: Reflection

## 1. Overview

### 1.1 Pattern Description

The Reflection pattern enables agents to evaluate their own outputs, reasoning, and actions, then iteratively improve based on self-assessment. The agent generates an initial response, reflects on its quality and correctness, identifies improvements, and revises until the output meets quality criteria or iteration limits are reached.

### 1.2 Key Characteristics

- Self-evaluation of generated outputs against quality criteria
- Iterative refinement based on identified weaknesses
- Explicit articulation of what could be improved
- Convergence toward higher-quality outputs over iterations

### 1.3 Use Cases

- Code generation with self-review and bug detection
- Writing tasks requiring multiple drafts
- Complex reasoning where initial attempts may have errors
- Tasks where quality is subjective and benefits from reconsideration

### 1.4 Related Documents

- [Core Framework Requirements](../core-framework-prd.md)

---

## 2. Functional Requirements

### 2.1 Reflection Loop

#### 2.1.1 Core Loop Structure

```
┌─────────────────────────────────────────────────────┐
│                    User Query                       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│               Generate Initial Output               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                     Reflect                         │
│    (Evaluate output against quality criteria)       │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Satisfactory?│───Yes──▶ Return Output
              └───────┬───────┘
                      │ No
                      ▼
┌─────────────────────────────────────────────────────┐
│              Identify Improvements                  │
│      (Specific, actionable feedback)                │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                 Revise Output                       │
│       (Apply identified improvements)               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ Max Iterations│───Yes──▶ Return Best Output
              │   Reached?    │
              └───────┬───────┘
                      │ No
                      └──────────▶ Reflect
```

#### 2.1.2 Reflection Phases

| Phase | Description | Output |
|-------|-------------|--------|
| Generation | Produce initial output for the task | Draft output |
| Evaluation | Assess output against criteria | Quality scores, identified issues |
| Critique | Articulate specific weaknesses | List of improvement areas |
| Revision | Apply improvements to create new version | Revised output |

### 2.2 Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_iterations` | `int` | `3` | Maximum reflection-revision cycles |
| `reflection_prompt` | `str` | (default template) | Custom reflection prompt |
| `quality_criteria` | `list[str]` | (task-dependent) | Criteria for evaluation |
| `quality_threshold` | `float` | `0.8` | Score threshold to stop iterating |
| `reflection_model` | `str` | `None` | Use different model for reflection |
| `structured_reflection` | `bool` | `True` | Use structured output for reflections |
| `preserve_all_versions` | `bool` | `True` | Keep all intermediate versions |
| `reflection_triggers` | `list[str]` | `["completion"]` | When to trigger reflection |

### 2.3 Reflection Triggers

The pattern must support configurable reflection triggers:

| Trigger | Description | Configuration |
|---------|-------------|---------------|
| `completion` | Reflect after generating complete output | Always enabled |
| `step_interval` | Reflect every N steps (for multi-step tasks) | `reflection_interval: int` |
| `error` | Reflect after encountering an error | `reflect_on_error: bool` |
| `tool_result` | Reflect after receiving tool output | `reflect_on_tools: list[str]` |
| `manual` | Explicit reflection request via API | Method call |
| `quality_drop` | Reflect when quality score decreases | `quality_drop_threshold: float` |

### 2.4 Quality Evaluation

#### 2.4.1 Evaluation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| LLM Self-Evaluation | Model evaluates its own output | General purpose |
| Criteria Checklist | Structured evaluation against criteria | Consistent scoring |
| External Validator | Custom validation function | Domain-specific validation |
| Comparison | Compare against reference or prior version | Improvement tracking |

#### 2.4.2 Quality Criteria Configuration

```python
@dataclass
class QualityCriterion:
    name: str                    # e.g., "correctness", "clarity"
    description: str             # What this criterion measures
    weight: float = 1.0          # Relative importance
    threshold: float = 0.7       # Minimum acceptable score
    evaluator: str = "llm"       # "llm", "regex", "function"
    evaluator_config: dict = {}  # Evaluator-specific config
```

Default criteria sets for common task types:

| Task Type | Default Criteria |
|-----------|------------------|
| Code Generation | correctness, efficiency, readability, edge_cases |
| Writing | clarity, coherence, completeness, tone |
| Analysis | accuracy, depth, relevance, objectivity |
| Reasoning | logical_validity, completeness, clarity |

#### 2.4.3 Evaluation Output

```python
@dataclass
class ReflectionEvaluation:
    overall_score: float                      # 0.0 to 1.0
    criteria_scores: dict[str, float]         # Per-criterion scores
    satisfactory: bool                        # Meets threshold
    issues: list[str]                         # Identified problems
    suggestions: list[str]                    # Improvement suggestions
    confidence: float                         # Confidence in evaluation
```

### 2.5 Revision Process

#### 2.5.1 Revision Strategies

| Strategy | Description |
|----------|-------------|
| `full_rewrite` | Generate entirely new output incorporating feedback |
| `targeted_edit` | Make specific edits to address identified issues |
| `incremental` | Build upon previous version, adding/fixing sections |
| `hybrid` | Choose strategy based on scope of needed changes |

#### 2.5.2 Revision Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `revision_strategy` | `str` | `"hybrid"` | How to apply improvements |
| `include_prior_versions` | `bool` | `False` | Show prior versions in revision prompt |
| `include_reflection_history` | `bool` | `True` | Include past reflections in context |
| `max_revision_tokens` | `int` | `None` | Token budget for revisions |

### 2.6 Convergence Detection

#### 2.6.1 Convergence Criteria

The pattern must detect when further iterations are unlikely to improve output:

| Criterion | Description |
|-----------|-------------|
| Quality Plateau | Score not improving for N iterations |
| Oscillation | Scores alternating without trend |
| Diminishing Returns | Improvements below threshold |
| Semantic Stability | Output not meaningfully changing |

#### 2.6.2 Convergence Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `plateau_iterations` | `int` | `2` | Iterations without improvement to stop |
| `improvement_threshold` | `float` | `0.05` | Minimum score improvement to continue |
| `detect_oscillation` | `bool` | `True` | Stop if quality oscillates |

### 2.7 Reflection History

#### 2.7.1 Version Tracking

```python
@dataclass
class ReflectionVersion:
    iteration: int
    output: str
    evaluation: ReflectionEvaluation
    improvements_applied: list[str]
    timestamp: datetime
    token_usage: TokenUsage

@dataclass
class ReflectionHistory:
    versions: list[ReflectionVersion]
    best_version: ReflectionVersion
    final_version: ReflectionVersion
    convergence_reason: str
    total_iterations: int
```

#### 2.7.2 Best Version Selection

Options for selecting the "best" version to return:

| Selection | Description |
|-----------|-------------|
| `highest_score` | Version with highest overall quality score |
| `latest` | Most recent version (default) |
| `first_satisfactory` | First version meeting threshold |
| `weighted_recency` | Balance quality and recency |

---

## 3. Interface Requirements

### 3.1 Pattern Class

```python
class ReflectionPattern(AgentPattern):
    """Reflection pattern for iterative self-improvement."""

    def __init__(
        self,
        max_iterations: int = 3,
        quality_criteria: list[QualityCriterion] | None = None,
        quality_threshold: float = 0.8,
        reflection_prompt: str | None = None,
        revision_strategy: Literal["full_rewrite", "targeted_edit", "incremental", "hybrid"] = "hybrid",
        reflection_triggers: list[str] = ["completion"],
        best_version_selection: Literal["highest_score", "latest", "first_satisfactory"] = "highest_score",
        structured_reflection: bool = True,
    ) -> None: ...

    async def execute(
        self,
        agent: Agent,
        query: str,
    ) -> ReflectionResult: ...

    async def reflect(
        self,
        output: str,
        criteria: list[QualityCriterion] | None = None,
    ) -> ReflectionEvaluation: ...

    async def revise(
        self,
        output: str,
        evaluation: ReflectionEvaluation,
    ) -> str: ...

    def get_history(self) -> ReflectionHistory: ...

    def get_best_version(self) -> ReflectionVersion: ...
```

### 3.2 Result Structure

```python
@dataclass
class ReflectionResult:
    success: bool
    final_output: str
    best_output: str                    # May differ from final
    final_evaluation: ReflectionEvaluation
    history: ReflectionHistory
    iterations_used: int
    convergence_reason: ConvergenceReason
    token_usage: TokenUsage
    execution_time: float
```

### 3.3 Convergence Reasons

```python
class ConvergenceReason(Enum):
    QUALITY_MET = "quality_met"              # Met quality threshold
    MAX_ITERATIONS = "max_iterations"        # Hit iteration limit
    PLATEAU = "plateau"                      # No improvement
    OSCILLATION = "oscillation"              # Quality oscillating
    DIMINISHING_RETURNS = "diminishing"      # Improvements too small
    USER_ACCEPTED = "user_accepted"          # User approved output
    TOKEN_BUDGET = "token_budget"            # Out of tokens
```

### 3.4 Callbacks

```python
class ReflectionCallbacks:
    async def on_generation(self, output: str, iteration: int) -> None:
        """Called after each output generation."""
        pass

    async def on_reflection(
        self,
        evaluation: ReflectionEvaluation,
        iteration: int
    ) -> ReflectionDecision:
        """Called after reflection. Can override continue/stop decision."""
        return ReflectionDecision.AUTO

    async def on_revision(
        self,
        old_output: str,
        new_output: str,
        iteration: int
    ) -> None:
        """Called after each revision."""
        pass
```

---

## 4. Behavioral Requirements

### 4.1 Reflection Quality

- Reflections should identify specific, actionable issues
- Evaluations should be consistent across similar outputs
- Criteria scores should correlate with actual output quality
- Self-evaluation should acknowledge uncertainty when appropriate

### 4.2 Revision Effectiveness

- Revisions should address identified issues
- Revisions should not introduce new problems
- Targeted edits should preserve unaffected content
- Each iteration should show measurable progress (when possible)

### 4.3 Efficiency

- Avoid unnecessary reflection iterations
- Recognize when output is already satisfactory
- Stop early when diminishing returns detected
- Use targeted edits over full rewrites when appropriate

### 4.4 Failure Modes to Handle

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Infinite Loop | Same issues repeatedly identified | Force termination, return best |
| Quality Regression | New version worse than previous | Revert, try different approach |
| Overcritical | Never rates output as satisfactory | Adjust threshold or return best |
| Undercritical | Always rates output as satisfactory | May need external validation |

---

## 5. Example Usage

### 5.1 Basic Reflection for Code Generation

```python
from agent_framework import Agent
from agent_framework.patterns import ReflectionPattern, QualityCriterion

criteria = [
    QualityCriterion(name="correctness", weight=2.0),
    QualityCriterion(name="efficiency", weight=1.0),
    QualityCriterion(name="readability", weight=1.0),
    QualityCriterion(name="edge_cases", weight=1.5),
]

pattern = ReflectionPattern(
    max_iterations=4,
    quality_criteria=criteria,
    quality_threshold=0.85,
    revision_strategy="targeted_edit"
)

agent = Agent(
    pattern=pattern,
    system_prompt="You are an expert Python developer."
)

result = await agent.run(
    "Write a function to find the longest palindromic substring"
)

# View the improvement journey
for version in result.history.versions:
    print(f"Iteration {version.iteration}: Score {version.evaluation.overall_score:.2f}")
    print(f"  Issues: {version.evaluation.issues}")
```

### 5.2 Reflection with Custom Validator

```python
import ast

def syntax_validator(code: str) -> tuple[bool, str]:
    """Validate Python syntax."""
    try:
        ast.parse(code)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

criteria = [
    QualityCriterion(
        name="syntax",
        evaluator="function",
        evaluator_config={"function": syntax_validator}
    ),
    QualityCriterion(name="correctness"),  # LLM evaluation
]

pattern = ReflectionPattern(quality_criteria=criteria)
```

### 5.3 Writing with Multiple Drafts

```python
pattern = ReflectionPattern(
    max_iterations=5,
    quality_criteria=[
        QualityCriterion(name="clarity", weight=1.5),
        QualityCriterion(name="engagement", weight=1.0),
        QualityCriterion(name="accuracy", weight=2.0),
        QualityCriterion(name="structure", weight=1.0),
    ],
    revision_strategy="incremental",
    best_version_selection="highest_score"
)

agent = Agent(
    pattern=pattern,
    system_prompt="You are a technical writer."
)

result = await agent.run(
    "Write a tutorial explaining how async/await works in Python"
)

print(f"Final draft after {result.iterations_used} iterations")
print(f"Best score achieved: {result.history.best_version.evaluation.overall_score:.2f}")
```

---

## 6. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Quality Improvement | Final output scores higher than initial in 80%+ of runs |
| Convergence Rate | Reaches satisfactory quality within 3 iterations in 70%+ of runs |
| Issue Resolution | Identified issues addressed in subsequent revision 85%+ of time |
| Evaluation Consistency | Same output receives similar scores across evaluations (within 0.1) |
| Efficient Termination | Stops appropriately when quality met or plateau reached 90%+ of runs |

---

## 7. Open Questions

1. Should reflection use a different model than generation for objectivity?
2. How to handle tasks where quality is highly subjective?
3. Should there be "meta-reflection" on the reflection process itself?
4. How to balance thoroughness of reflection with token efficiency?
