# Agent Pattern Requirements: Meta-Controller (Router/Orchestrator)

## 1. Overview

### 1.1 Pattern Description

The Meta-Controller pattern (also known as Smart Dispatcher, Router, or Orchestrator) provides intelligent routing of queries and tasks to the most appropriate model, agent, or tool based on task characteristics, cost constraints, quality requirements, and available resources. Rather than using a single model for all tasks or rigid rule-based dispatching, the meta-controller dynamically selects the optimal execution path for each input.

### 1.2 Key Characteristics

- Dynamic selection of models, agents, or tools based on query analysis
- Cost-aware routing that balances quality against computational expense
- Multiple routing strategies (semantic, LLM-based, ML classifier, rule-based, hybrid)
- Support for cascading (sequential escalation) and direct routing
- Confidence-based decision making with fallback mechanisms
- Semantic caching to avoid redundant inference

### 1.3 Use Cases

- Multi-model deployments where different models excel at different tasks
- Cost optimization in API-based LLM usage (routing simple queries to cheaper models)
- Latency-sensitive applications requiring fast responses for simple queries
- Multi-agent systems requiring intelligent task delegation
- Customer service routing to specialized agents (billing, technical, sales)
- Hybrid RAG systems choosing between retrieval strategies

### 1.4 References

- [Cascade Routing (ETH Zurich)](https://arxiv.org/abs/2410.10347) - Unified routing and cascading
- [NVIDIA Orchestrator-8B](https://huggingface.co/nvidia/Orchestrator-8B) - RL-trained orchestrator
- [RouteLLM](https://arxiv.org/abs/2406.18665) - Learning to route with preference data
- [Semantic Router](https://github.com/aurelio-labs/semantic-router) - Fast semantic routing library
- [vLLM Semantic Router](https://blog.vllm.ai/2025/09/11/semantic-router.html) - Intent-aware LLM gateway

### 1.5 Related Documents

- [Core Framework Requirements](../core-framework-prd.md)
- [Multi-Agent Systems Pattern](./pattern-multi-agent.md)

---

## 2. Functional Requirements

### 2.1 Routing Architecture

#### 2.1.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pre-Processing                               │
│         (Safety screening, PII detection, normalization)        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Semantic Cache Check                         │
│            (Return cached response if similar query)            │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Cache Miss
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Analysis                               │
│    (Intent classification, complexity estimation, features)     │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Route Selection                              │
│      (Apply routing strategy, select target(s), estimate cost)  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
           ┌──────────────┼──────────────┐
           │              │              │
           ▼              ▼              ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │ Model A │   │ Model B │   │ Agent X │
      │ (Fast)  │   │(Capable)│   │(Special)│
      └────┬────┘   └────┬────┘   └────┬────┘
           │              │              │
           └──────────────┼──────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Response Validation                           │
│        (Quality check, confidence scoring, escalation?)         │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Return Response                            │
│              (Cache result, log routing decision)               │
└─────────────────────────────────────────────────────────────────┘
```

#### 2.1.2 Core Components

| Component | Description |
|-----------|-------------|
| Query Analyzer | Extracts features, classifies intent, estimates complexity |
| Route Selector | Applies routing strategy to choose target(s) |
| Target Registry | Maintains catalog of available models/agents/tools |
| Cost Estimator | Predicts cost (tokens, latency, API cost) for each route |
| Quality Predictor | Estimates expected output quality per route |
| Confidence Scorer | Evaluates response quality post-generation |
| Semantic Cache | Stores and retrieves responses for similar queries |
| Escalation Manager | Handles cascading and fallback logic |

### 2.2 Routing Strategies

#### 2.2.1 Strategy Types

| Strategy | Description | Best For |
|----------|-------------|----------|
| `semantic` | Embedding-based similarity to predefined routes | Well-defined intent categories |
| `llm_classifier` | LLM classifies query and selects route | Flexible, nuanced routing |
| `ml_classifier` | Trained ML model predicts best route | High-volume, low-latency |
| `rule_based` | Pattern matching, keywords, regex | Simple, deterministic routing |
| `hybrid` | Combination of multiple strategies | Production systems |
| `learned` | RL-trained router optimizing cost/quality | Cost-sensitive deployments |

#### 2.2.2 Semantic Routing

Uses embedding similarity to match queries to predefined route definitions:

```python
@dataclass
class SemanticRoute:
    name: str
    description: str
    utterances: list[str]           # Example queries for this route
    target: str                     # Model/agent/tool name
    threshold: float = 0.8          # Minimum similarity to match
    embedding: np.ndarray = None    # Computed from utterances
```

Configuration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoder` | `str` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model |
| `similarity_metric` | `str` | `"cosine"` | Similarity function |
| `top_k` | `int` | `1` | Number of routes to consider |
| `fallback_route` | `str` | `None` | Route when no match exceeds threshold |
| `aggregation` | `str` | `"mean"` | How to combine utterance embeddings |

#### 2.2.3 LLM-Based Routing

Uses an LLM to analyze the query and select the appropriate route:

```python
@dataclass
class LLMRouterConfig:
    model: str                      # Router model (can be smaller/cheaper)
    route_descriptions: dict[str, str]  # Route name -> description
    routing_prompt: str | None      # Custom routing prompt
    output_format: str = "json"     # "json" or "text"
    include_reasoning: bool = True  # Include chain-of-thought
    max_tokens: int = 100           # Limit router output
```

#### 2.2.4 ML Classifier Routing

Uses a trained classifier for fast, deterministic routing:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_type` | `str` | `"bert"` | Classifier architecture |
| `model_path` | `str` | Required | Path to trained model |
| `classes` | `list[str]` | Required | Route class names |
| `confidence_threshold` | `float` | `0.7` | Minimum confidence to route |
| `fallback_route` | `str` | `None` | Route for low-confidence |

#### 2.2.5 Hybrid Routing

Combines multiple strategies with configurable priority:

```python
@dataclass
class HybridRouterConfig:
    strategies: list[RoutingStrategy]   # Ordered by priority
    combination_method: str = "first_match"  # "first_match", "vote", "weighted"
    weights: dict[str, float] = None    # For weighted combination
    agreement_threshold: float = 0.67   # For voting
```

### 2.3 Model Selection Modes

#### 2.3.1 Direct Routing

Select a single target for the query:

```
Query ──▶ Router ──▶ Selected Model ──▶ Response
```

Configuration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `selection_criterion` | `str` | `"best_quality"` | "best_quality", "lowest_cost", "balanced" |
| `quality_weight` | `float` | `0.7` | Weight for quality vs cost |
| `cost_weight` | `float` | `0.3` | Weight for cost vs quality |

#### 2.3.2 Cascading

Process through sequential models until satisfactory response:

```
Query ──▶ Model₁ (cheap) ──[not confident]──▶ Model₂ (medium) ──[not confident]──▶ Model₃ (expensive)
              │                                    │                                    │
              └────────[confident]─────────────────┴────────[confident]────────────────┴──▶ Response
```

Configuration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cascade_order` | `list[str]` | Required | Models ordered cheap to expensive |
| `confidence_threshold` | `float` | `0.8` | Threshold to stop cascading |
| `confidence_method` | `str` | `"self_eval"` | "self_eval", "judge", "entropy" |
| `max_cascade_depth` | `int` | `3` | Maximum models to try |
| `early_exit` | `bool` | `True` | Stop at first confident response |

#### 2.3.3 Cascade Routing (Hybrid)

Combines routing flexibility with cascade cost-efficiency:

```
Query ──▶ Router ──▶ Model_i ──[not confident]──▶ Router ──▶ Model_j ──▶ Response
```

The router can re-route to any model after an unsatisfactory response, not just the next in sequence.

Configuration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_routing` | `str` | `"predicted_best"` | First model selection strategy |
| `reroute_strategy` | `str` | `"next_best"` | How to select on reroute |
| `max_reroutes` | `int` | `2` | Maximum rerouting attempts |
| `exclude_tried` | `bool` | `True` | Don't reroute to already-tried models |

#### 2.3.4 Parallel Ensemble

Query multiple models simultaneously, aggregate results:

```
          ┌──▶ Model A ──┐
Query ────┼──▶ Model B ──┼──▶ Aggregator ──▶ Response
          └──▶ Model C ──┘
```

Configuration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[str]` | Required | Models to query in parallel |
| `aggregation` | `str` | `"best"` | "best", "vote", "blend", "judge" |
| `judge_model` | `str` | `None` | Model to judge/select best response |
| `timeout` | `float` | `30.0` | Max wait for all responses |
| `min_responses` | `int` | `1` | Minimum responses before aggregating |

### 2.4 Target Registry

#### 2.4.1 Target Definition

```python
@dataclass
class RoutingTarget:
    name: str                           # Unique identifier
    type: TargetType                    # "model", "agent", "tool", "chain"
    endpoint: str                       # API endpoint or reference
    capabilities: list[str]             # What this target can do
    description: str                    # Human-readable description

    # Cost information
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    average_latency_ms: float = 0.0

    # Performance information
    quality_scores: dict[str, float] = None  # Task -> score mapping
    benchmark_results: dict[str, float] = None

    # Constraints
    max_context_length: int = 4096
    supports_streaming: bool = True
    rate_limit: float | None = None     # Requests per second

    # Metadata
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0                   # For tie-breaking
```

#### 2.4.2 Target Types

| Type | Description | Example |
|------|-------------|---------|
| `model` | LLM endpoint | GPT-4, Claude, Llama |
| `agent` | Specialized agent | Code agent, research agent |
| `tool` | External tool/API | Calculator, web search |
| `chain` | Predefined workflow | RAG pipeline, analysis chain |
| `router` | Nested router | Sub-router for domain |

#### 2.4.3 Dynamic Target Discovery

```python
class TargetRegistry:
    def register(self, target: RoutingTarget) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> RoutingTarget | None: ...
    def list_by_capability(self, capability: str) -> list[RoutingTarget]: ...
    def list_by_tag(self, tag: str) -> list[RoutingTarget]: ...
    def update_metrics(self, name: str, metrics: TargetMetrics) -> None: ...
    def get_cheapest(self, capabilities: list[str]) -> RoutingTarget | None: ...
    def get_best_quality(self, task: str) -> RoutingTarget | None: ...
```

### 2.5 Cost and Quality Optimization

#### 2.5.1 Cost Model

```python
@dataclass
class CostEstimate:
    input_tokens: int
    estimated_output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    latency_estimate_ms: float
    confidence: float                   # Confidence in estimate
```

#### 2.5.2 Quality Prediction

| Method | Description | Use Case |
|--------|-------------|----------|
| `benchmark_lookup` | Use precomputed benchmark scores | Known task types |
| `ml_predictor` | Trained model predicts quality | Custom deployments |
| `llm_predictor` | LLM estimates expected quality | Flexible, expensive |
| `historical` | Use past performance on similar queries | Production systems |

#### 2.5.3 Optimization Objectives

```python
@dataclass
class OptimizationConfig:
    objective: str = "balanced"         # "quality", "cost", "latency", "balanced"
    quality_weight: float = 0.5
    cost_weight: float = 0.3
    latency_weight: float = 0.2

    # Constraints
    max_cost_per_query: float | None = None
    max_latency_ms: float | None = None
    min_quality_threshold: float | None = None

    # Budget management
    budget_period: str = "hourly"       # "per_query", "hourly", "daily"
    budget_amount: float | None = None
    budget_action: str = "downgrade"    # "downgrade", "reject", "queue"
```

### 2.6 Confidence and Escalation

#### 2.6.1 Confidence Scoring Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| `self_evaluation` | Model rates its own response | Simple, no extra cost | Self-enhancement bias |
| `judge_model` | Separate model evaluates | More objective | Additional cost |
| `entropy` | Token probability entropy | No extra inference | Only for local models |
| `consistency` | Multiple samples, check agreement | Robust | Expensive |
| `verifier` | Task-specific verification | Accurate for specific tasks | Limited scope |

#### 2.6.2 Escalation Rules

```python
@dataclass
class EscalationRule:
    trigger: EscalationTrigger          # What triggers escalation
    threshold: float                    # Threshold value
    action: EscalationAction            # What to do
    target: str | None = None           # Specific escalation target

class EscalationTrigger(Enum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_COMPLEXITY = "high_complexity"
    SAFETY_FLAG = "safety_flag"
    USER_REQUEST = "user_request"
    ERROR = "error"
    TIMEOUT = "timeout"

class EscalationAction(Enum):
    ROUTE_TO_STRONGER = "route_to_stronger"
    ROUTE_TO_SPECIFIC = "route_to_specific"
    HUMAN_REVIEW = "human_review"
    REJECT = "reject"
    RETRY = "retry"
```

### 2.7 Semantic Caching

#### 2.7.1 Cache Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable semantic caching |
| `similarity_threshold` | `float` | `0.95` | Minimum similarity for cache hit |
| `encoder` | `str` | Same as router | Embedding model for cache |
| `max_entries` | `int` | `10000` | Maximum cache size |
| `ttl_seconds` | `int` | `3600` | Time-to-live for entries |
| `cache_backend` | `str` | `"memory"` | "memory", "redis", "faiss" |

#### 2.7.2 Cache Operations

```python
class SemanticCache:
    async def get(self, query: str) -> CacheResult | None: ...
    async def set(self, query: str, response: str, metadata: dict) -> None: ...
    async def invalidate(self, pattern: str) -> int: ...
    def get_stats(self) -> CacheStats: ...
```

### 2.8 Safety and Preprocessing

#### 2.8.1 Safety Screening

| Check | Description | Action |
|-------|-------------|--------|
| PII Detection | Detect personal information | Mask, route to secure model, or reject |
| Jailbreak Detection | Detect prompt injection attempts | Reject or route to hardened model |
| Content Policy | Check against content policies | Reject or flag for review |
| Rate Limiting | Prevent abuse | Queue or reject |

#### 2.8.2 Preprocessing Pipeline

```python
@dataclass
class PreprocessingConfig:
    normalize_whitespace: bool = True
    detect_language: bool = True
    extract_entities: bool = False
    classify_safety: bool = True
    pii_detection: bool = True
    pii_action: str = "mask"            # "mask", "reject", "flag"
    max_input_length: int = 10000
```

---

## 3. Interface Requirements

### 3.1 Pattern Class

```python
class MetaControllerPattern(AgentPattern):
    """Meta-Controller pattern for intelligent routing."""

    def __init__(
        self,
        targets: list[RoutingTarget],
        routing_strategy: RoutingStrategy,
        selection_mode: SelectionMode = SelectionMode.DIRECT,
        optimization: OptimizationConfig | None = None,
        confidence_method: str = "self_evaluation",
        confidence_threshold: float = 0.8,
        escalation_rules: list[EscalationRule] | None = None,
        cache_config: CacheConfig | None = None,
        preprocessing: PreprocessingConfig | None = None,
    ) -> None: ...

    async def execute(
        self,
        query: str,
        context: dict | None = None,
        constraints: RoutingConstraints | None = None,
    ) -> MetaControllerResult: ...

    async def route(
        self,
        query: str,
    ) -> RoutingDecision: ...

    async def execute_on_target(
        self,
        target: str,
        query: str,
    ) -> TargetResponse: ...

    def get_registry(self) -> TargetRegistry: ...

    def get_cache(self) -> SemanticCache: ...

    def get_routing_stats(self) -> RoutingStats: ...
```

### 3.2 Result Structure

```python
@dataclass
class MetaControllerResult:
    success: bool
    response: str

    # Routing information
    routing_decision: RoutingDecision
    targets_tried: list[str]
    final_target: str

    # Cost and quality
    total_cost: float
    estimated_quality: float
    confidence_score: float

    # Cache information
    cache_hit: bool
    cache_key: str | None

    # Timing
    routing_time_ms: float
    inference_time_ms: float
    total_time_ms: float

    # Token usage
    token_usage: TokenUsage

@dataclass
class RoutingDecision:
    selected_target: str
    strategy_used: str
    confidence: float
    alternatives: list[tuple[str, float]]  # (target, score) pairs
    reasoning: str | None                   # For LLM-based routing
    estimated_cost: CostEstimate
    features: dict[str, Any]                # Extracted query features
```

### 3.3 Routing Statistics

```python
@dataclass
class RoutingStats:
    total_queries: int
    queries_by_target: dict[str, int]
    cache_hit_rate: float
    average_routing_time_ms: float
    average_confidence: float
    escalation_rate: float
    cost_by_target: dict[str, float]
    quality_by_target: dict[str, float]
    errors_by_target: dict[str, int]
```

### 3.4 Callbacks

```python
class MetaControllerCallbacks:
    async def on_routing_start(self, query: str) -> None:
        """Called before routing decision."""
        pass

    async def on_routing_decision(
        self,
        decision: RoutingDecision
    ) -> RoutingDecision:
        """Called after routing. Can modify decision."""
        return decision

    async def on_target_response(
        self,
        target: str,
        response: TargetResponse,
    ) -> EscalationDecision:
        """Called after each target response. Decide escalation."""
        return EscalationDecision.ACCEPT

    async def on_cache_hit(
        self,
        query: str,
        cached_response: str,
    ) -> bool:
        """Called on cache hit. Return False to bypass cache."""
        return True
```

---

## 4. Behavioral Requirements

### 4.1 Routing Quality

- Routing decisions should be consistent for similar queries
- Router should not exhibit self-enhancement bias (preferring itself)
- Cost estimates should be within 20% of actual costs
- Quality predictions should correlate with actual output quality

### 4.2 Efficiency

- Routing overhead should be < 100ms for semantic/ML routing
- Routing overhead should be < 500ms for LLM-based routing
- Cache hit rate should exceed 30% for typical workloads
- Total latency should not exceed 2x single-model latency

### 4.3 Robustness

- Graceful degradation when targets are unavailable
- Fallback routing when primary strategy fails
- Rate limiting and backpressure handling
- Recovery from transient errors

### 4.4 Avoiding Common Pitfalls

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| Self-enhancement bias | Router overuses itself or favorite models | Use separate router model, train with diverse data |
| Over-routing to expensive | Small models underutilized | Set cost constraints, use cascading |
| Cache poisoning | Bad responses cached | Quality validation before caching |
| Cascading costs | Multiple model calls expensive | Set max cascade depth, optimize thresholds |
| Single point of failure | Router failure breaks system | Fallback rules, circuit breakers |

---

## 5. Example Usage

### 5.1 Basic Semantic Router

```python
from agent_framework.patterns import MetaControllerPattern
from agent_framework.routing import SemanticRouter, RoutingTarget

# Define routes
routes = [
    SemanticRoute(
        name="code",
        description="Code generation and debugging",
        utterances=[
            "write a function that",
            "debug this code",
            "explain this algorithm",
        ],
        target="code_model"
    ),
    SemanticRoute(
        name="creative",
        description="Creative writing tasks",
        utterances=[
            "write a story about",
            "compose a poem",
            "create a marketing tagline",
        ],
        target="creative_model"
    ),
    SemanticRoute(
        name="analysis",
        description="Data analysis and reasoning",
        utterances=[
            "analyze this data",
            "compare these options",
            "what are the implications of",
        ],
        target="reasoning_model"
    ),
]

# Define targets
targets = [
    RoutingTarget(
        name="code_model",
        type=TargetType.MODEL,
        endpoint="http://localhost:11434/v1",
        model="codellama:34b",
        cost_per_input_token=0.0001,
    ),
    RoutingTarget(
        name="creative_model",
        type=TargetType.MODEL,
        endpoint="http://localhost:11434/v1",
        model="llama3.2:latest",
        cost_per_input_token=0.00005,
    ),
    RoutingTarget(
        name="reasoning_model",
        type=TargetType.MODEL,
        endpoint="http://localhost:11434/v1",
        model="qwen2.5:72b",
        cost_per_input_token=0.0002,
    ),
]

router = SemanticRouter(routes=routes, fallback_route="reasoning_model")

pattern = MetaControllerPattern(
    targets=targets,
    routing_strategy=router,
)

result = await pattern.execute("Write a Python function to calculate fibonacci numbers")
print(f"Routed to: {result.final_target}")  # code_model
```

### 5.2 Cost-Optimized Cascading

```python
from agent_framework.patterns import MetaControllerPattern, CascadeMode
from agent_framework.routing import CascadeConfig

cascade_config = CascadeConfig(
    cascade_order=["llama3.2:8b", "llama3.2:70b", "qwen2.5:72b"],
    confidence_threshold=0.85,
    confidence_method="self_eval",
    max_cascade_depth=3,
)

optimization = OptimizationConfig(
    objective="balanced",
    quality_weight=0.6,
    cost_weight=0.4,
    max_cost_per_query=0.05,
)

pattern = MetaControllerPattern(
    targets=targets,
    routing_strategy=cascade_config,
    selection_mode=SelectionMode.CASCADE,
    optimization=optimization,
)

# Simple query stops at first model
result = await pattern.execute("What is 2 + 2?")
print(f"Targets tried: {result.targets_tried}")  # ["llama3.2:8b"]

# Complex query cascades to stronger model
result = await pattern.execute("Prove that there are infinitely many primes")
print(f"Targets tried: {result.targets_tried}")  # ["llama3.2:8b", "llama3.2:70b"]
```

### 5.3 Hybrid Router with Caching

```python
from agent_framework.routing import (
    HybridRouter,
    SemanticRouter,
    MLClassifierRouter,
    CacheConfig,
)

# Combine semantic and ML routing
hybrid = HybridRouter(
    strategies=[
        SemanticRouter(routes=routes, threshold=0.9),  # High-confidence semantic
        MLClassifierRouter(model_path="./router_model.pt"),  # ML fallback
    ],
    combination_method="first_match",
)

cache_config = CacheConfig(
    enabled=True,
    similarity_threshold=0.95,
    ttl_seconds=3600,
    max_entries=50000,
)

pattern = MetaControllerPattern(
    targets=targets,
    routing_strategy=hybrid,
    cache_config=cache_config,
)

# First call: cache miss, routes via hybrid strategy
result1 = await pattern.execute("Explain how transformers work")
print(f"Cache hit: {result1.cache_hit}")  # False

# Second call: semantically similar, cache hit
result2 = await pattern.execute("How do transformer models work?")
print(f"Cache hit: {result2.cache_hit}")  # True
```

### 5.4 Multi-Agent Routing

```python
from agent_framework.patterns import MetaControllerPattern
from agent_framework.routing import LLMRouter

# Route to specialized agents
targets = [
    RoutingTarget(
        name="research_agent",
        type=TargetType.AGENT,
        capabilities=["research", "fact_finding", "summarization"],
        description="Searches and synthesizes information from multiple sources",
    ),
    RoutingTarget(
        name="code_agent",
        type=TargetType.AGENT,
        capabilities=["coding", "debugging", "code_review"],
        description="Writes, reviews, and debugs code",
    ),
    RoutingTarget(
        name="writing_agent",
        type=TargetType.AGENT,
        capabilities=["writing", "editing", "creative"],
        description="Creates and edits written content",
    ),
]

llm_router = LLMRouter(
    model="llama3.2:8b",  # Small, fast model for routing
    route_descriptions={t.name: t.description for t in targets},
    include_reasoning=True,
)

pattern = MetaControllerPattern(
    targets=targets,
    routing_strategy=llm_router,
)

result = await pattern.execute(
    "Research the latest developments in quantum computing and write a summary"
)
print(f"Routed to: {result.final_target}")
print(f"Routing reasoning: {result.routing_decision.reasoning}")
```

---

## 6. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Routing Accuracy | Correct target selected in 90%+ of cases |
| Cost Reduction | 30%+ cost savings vs always using best model |
| Latency Overhead | < 100ms for semantic/ML, < 500ms for LLM routing |
| Cache Effectiveness | > 30% hit rate, < 5% stale responses |
| Escalation Accuracy | Escalation improves response in 80%+ of cases |
| Quality Preservation | Output quality within 5% of oracle routing |

---

## 7. Open Questions

1. How to handle queries that genuinely require multiple model capabilities?
2. Should routing decisions be explainable to end users?
3. How to efficiently update router when new models are added?
4. What's the optimal balance between routing accuracy and routing cost?
5. How to handle routing for multi-turn conversations (route per turn vs session)?
6. Should there be "negative routing" (routes that explicitly avoid certain models)?
