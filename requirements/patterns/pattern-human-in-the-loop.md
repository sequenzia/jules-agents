# Agent Pattern Requirements: Human-in-the-Loop (HITL)

## 1. Overview

### 1.1 Pattern Description

The Human-in-the-Loop (HITL) pattern introduces human oversight and intervention points throughout agent execution. Humans can approve, reject, or modify agent actions before they execute, provide guidance when the agent is uncertain, and correct course when the agent makes mistakes. This pattern is essential for high-stakes operations where autonomous action carries risk.

### 1.2 Key Characteristics

- Configurable approval gates at various stages of execution
- Support for both synchronous and asynchronous human interaction
- Graceful handling of human unavailability (timeouts, defaults)
- Feedback incorporation for improved future decisions

### 1.3 Use Cases

- Financial operations requiring human authorization
- Content moderation with human review
- System administration tasks with destructive potential
- Learning scenarios where human feedback improves agent behavior
- Compliance-sensitive workflows requiring audit trails

### 1.4 Related Documents

- [Core Framework Requirements](../core-framework-prd.md)

---

## 2. Functional Requirements

### 2.1 Approval Gates

#### 2.1.1 Gate Types

| Gate Type | Description | Triggered By |
|-----------|-------------|--------------|
| `tool_execution` | Approve before any tool runs | All tool calls |
| `specific_tools` | Approve only for designated tools | Configured tool list |
| `destructive_actions` | Approve actions that modify state | Tool metadata |
| `final_response` | Approve agent's final answer | Completion |
| `plan_approval` | Approve generated plans | Planning pattern |
| `threshold_based` | Approve when confidence is low | Confidence score |
| `budget_based` | Approve when cost exceeds threshold | Token/cost tracking |
| `custom` | Custom approval logic | User-defined function |

#### 2.1.2 Gate Configuration

```python
@dataclass
class ApprovalGate:
    name: str
    gate_type: GateType
    tools: list[str] | None = None           # For specific_tools type
    confidence_threshold: float | None = None  # For threshold_based
    cost_threshold: float | None = None        # For budget_based
    condition: Callable | None = None          # For custom type
    timeout_seconds: float = 300.0
    timeout_action: TimeoutAction = TimeoutAction.REJECT
    allow_modification: bool = True
    require_reason: bool = False               # Require human to provide reason
```

### 2.2 Interaction Flow

#### 2.2.1 Approval Request Flow

```
┌─────────────────────────────────────────────────────┐
│              Agent Reaches Gate Point               │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            Generate Approval Request                │
│   (Action details, context, risk assessment)        │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│            Send to Human Interface                  │
│      (UI callback, webhook, queue, etc.)            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Await Human  │
              │   Response    │
              └───────┬───────┘
                      │
         ┌────────────┼────────────┬────────────┐
         │            │            │            │
         ▼            ▼            ▼            ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Approve │ │ Reject  │ │ Modify  │ │ Timeout │
    └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │           │
         ▼           ▼           ▼           ▼
    Execute     Handle      Execute      Apply
    Action     Rejection   Modified    Timeout
                           Action      Action
```

#### 2.2.2 Human Response Options

| Response | Description | Data |
|----------|-------------|------|
| `APPROVE` | Proceed with proposed action | None or confirmation note |
| `REJECT` | Do not execute action | Required: rejection reason |
| `MODIFY` | Execute with modifications | Modified action parameters |
| `DEFER` | Ask agent to try different approach | Guidance/feedback |
| `ESCALATE` | Route to higher authority | Escalation target |
| `ABORT` | Stop entire agent execution | Abort reason |

### 2.3 Approval Request Content

#### 2.3.1 Request Structure

```python
@dataclass
class ApprovalRequest:
    request_id: str
    gate: ApprovalGate
    action: ProposedAction
    context: ExecutionContext
    risk_assessment: RiskAssessment | None
    suggested_response: HumanResponse | None
    timeout_at: datetime
    metadata: dict[str, Any]

@dataclass
class ProposedAction:
    action_type: str                    # "tool_call", "response", "plan"
    tool_name: str | None
    parameters: dict[str, Any] | None
    expected_outcome: str
    reversible: bool
    estimated_cost: float | None

@dataclass
class ExecutionContext:
    task_description: str
    conversation_summary: str
    steps_completed: list[str]
    current_reasoning: str
    relevant_history: list[Message]
```

#### 2.3.2 Risk Assessment

```python
@dataclass
class RiskAssessment:
    risk_level: Literal["low", "medium", "high", "critical"]
    risk_factors: list[str]
    potential_consequences: list[str]
    reversibility: Literal["fully", "partially", "irreversible"]
    confidence: float
```

### 2.4 Timeout Handling

#### 2.4.1 Timeout Actions

| Action | Description |
|--------|-------------|
| `REJECT` | Treat as rejection, do not execute |
| `APPROVE` | Treat as approval (use with caution) |
| `RETRY` | Re-send approval request |
| `ESCALATE` | Route to backup approver |
| `ABORT` | Stop agent execution entirely |
| `SKIP` | Skip this action, continue execution |
| `DEFAULT` | Use configured default action |

#### 2.4.2 Timeout Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeout_seconds` | `float` | `300.0` | Time to wait for response |
| `timeout_action` | `TimeoutAction` | `REJECT` | Action on timeout |
| `reminder_interval` | `float` | `None` | Send reminders at interval |
| `max_reminders` | `int` | `3` | Maximum reminder count |
| `escalation_timeout` | `float` | `None` | Time before escalation |
| `escalation_target` | `str` | `None` | Who to escalate to |

### 2.5 Human Interface Adapters

#### 2.5.1 Required Interface

```python
class HumanInterface(Protocol):
    """Protocol for human interaction adapters."""

    async def request_approval(
        self,
        request: ApprovalRequest
    ) -> HumanResponse:
        """Send approval request and await response."""
        ...

    async def send_notification(
        self,
        notification: Notification
    ) -> None:
        """Send informational notification (no response needed)."""
        ...

    async def request_input(
        self,
        prompt: str,
        input_type: InputType,
        options: list[str] | None = None
    ) -> str:
        """Request free-form or structured input."""
        ...

    def is_available(self) -> bool:
        """Check if human is available for interaction."""
        ...
```

#### 2.5.2 Built-in Adapters

| Adapter | Description | Use Case |
|---------|-------------|----------|
| `ConsoleAdapter` | Terminal-based interaction | Development, CLI apps |
| `CallbackAdapter` | Custom callback functions | Embedding in applications |
| `WebhookAdapter` | HTTP webhook notifications | Integration with external systems |
| `QueueAdapter` | Message queue integration | Async workflows |

#### 2.5.3 Adapter Configuration Example

```python
# Console adapter for development
console_interface = ConsoleAdapter(
    prompt_style="detailed",
    color_enabled=True
)

# Webhook adapter for production
webhook_interface = WebhookAdapter(
    approval_url="https://app.example.com/api/approvals",
    notification_url="https://app.example.com/api/notifications",
    auth_header="Bearer ${APPROVAL_API_KEY}",
    response_poll_interval=5.0
)

# Callback adapter for embedding
def handle_approval(request: ApprovalRequest) -> HumanResponse:
    # Custom logic, e.g., show in UI
    return my_ui.show_approval_dialog(request)

callback_interface = CallbackAdapter(
    approval_callback=handle_approval
)
```

### 2.6 Feedback Incorporation

#### 2.6.1 Feedback Types

| Type | Description | Usage |
|------|-------------|-------|
| Rejection Reason | Why action was rejected | Avoid similar actions |
| Modification Details | How action was changed | Learn preferences |
| Guidance | Direction for alternative approach | Improve reasoning |
| Correction | Fix for incorrect output | Error learning |

#### 2.6.2 Feedback Storage

```python
@dataclass
class HumanFeedback:
    request_id: str
    response: HumanResponse
    reason: str | None
    modifications: dict[str, Any] | None
    guidance: str | None
    timestamp: datetime
    human_id: str | None            # For multi-approver scenarios
```

#### 2.6.3 Feedback Application

- Rejected actions added to "avoid" context for current session
- Modifications inform parameter preferences
- Guidance incorporated into subsequent reasoning prompts
- Patterns detected across feedback used for proactive adjustment

### 2.7 Audit Trail

#### 2.7.1 Audit Log Structure

```python
@dataclass
class HITLAuditEntry:
    timestamp: datetime
    request_id: str
    gate_name: str
    proposed_action: ProposedAction
    human_response: HumanResponse
    response_time_seconds: float
    human_id: str | None
    ip_address: str | None          # If applicable
    session_id: str
    outcome: str                    # What happened after
```

#### 2.7.2 Audit Requirements

- All approval requests logged regardless of outcome
- Tamper-evident logging (append-only, checksummed)
- Configurable retention period
- Export capability for compliance review

---

## 3. Interface Requirements

### 3.1 Pattern Class

```python
class HumanInTheLoopPattern(AgentPattern):
    """Human-in-the-Loop pattern for supervised agent execution."""

    def __init__(
        self,
        interface: HumanInterface,
        gates: list[ApprovalGate] | None = None,
        default_timeout: float = 300.0,
        default_timeout_action: TimeoutAction = TimeoutAction.REJECT,
        require_approval_for_tools: list[str] | None = None,
        require_approval_for_destructive: bool = True,
        require_final_approval: bool = False,
        risk_assessment_enabled: bool = True,
        feedback_incorporation: bool = True,
        audit_enabled: bool = True,
    ) -> None: ...

    async def execute(
        self,
        agent: Agent,
        query: str,
    ) -> HITLResult: ...

    async def request_approval(
        self,
        action: ProposedAction,
        gate: ApprovalGate,
    ) -> HumanResponse: ...

    async def request_guidance(
        self,
        situation: str,
        options: list[str] | None = None,
    ) -> str: ...

    def get_feedback_history(self) -> list[HumanFeedback]: ...

    def get_audit_log(self) -> list[HITLAuditEntry]: ...
```

### 3.2 Result Structure

```python
@dataclass
class HITLResult:
    success: bool
    final_output: str | None
    approval_requests: list[ApprovalRequest]
    human_responses: list[HumanResponse]
    rejections: list[tuple[ApprovalRequest, HumanResponse]]
    modifications_applied: list[dict]
    audit_log: list[HITLAuditEntry]
    token_usage: TokenUsage
    execution_time: float
    human_interaction_time: float
```

### 3.3 Gate Presets

```python
# Common gate configurations
GATES_MINIMAL = [
    ApprovalGate(name="destructive", gate_type=GateType.DESTRUCTIVE_ACTIONS)
]

GATES_STANDARD = [
    ApprovalGate(name="destructive", gate_type=GateType.DESTRUCTIVE_ACTIONS),
    ApprovalGate(name="external", gate_type=GateType.SPECIFIC_TOOLS,
                 tools=["web_search", "send_email", "api_call"]),
]

GATES_STRICT = [
    ApprovalGate(name="all_tools", gate_type=GateType.TOOL_EXECUTION),
    ApprovalGate(name="final", gate_type=GateType.FINAL_RESPONSE),
]

GATES_PARANOID = [
    ApprovalGate(name="all_tools", gate_type=GateType.TOOL_EXECUTION),
    ApprovalGate(name="final", gate_type=GateType.FINAL_RESPONSE),
    ApprovalGate(name="plans", gate_type=GateType.PLAN_APPROVAL),
]
```

---

## 4. Behavioral Requirements

### 4.1 Approval Request Quality

- Requests should provide sufficient context for informed decisions
- Risk assessments should be accurate and actionable
- Expected outcomes should be clearly stated
- Modification options should be clearly presented

### 4.2 Graceful Degradation

- Agent should function (with limitations) if human unavailable
- Timeout handling should never leave agent in undefined state
- Partial execution results preserved on abort
- Clear communication of what was and wasn't completed

### 4.3 User Experience

- Minimize unnecessary approval requests (batching, smart filtering)
- Provide clear, concise approval prompts
- Support quick approval for routine actions
- Remember preferences within session

### 4.4 Security

- Validate human identity when required
- Prevent approval spoofing
- Secure transmission of approval requests/responses
- Rate limiting on approval endpoints

---

## 5. Example Usage

### 5.1 Basic HITL with Console

```python
from agent_framework import Agent
from agent_framework.patterns import HumanInTheLoopPattern
from agent_framework.hitl import ConsoleAdapter, ApprovalGate, GateType

interface = ConsoleAdapter()

pattern = HumanInTheLoopPattern(
    interface=interface,
    gates=[
        ApprovalGate(
            name="file_writes",
            gate_type=GateType.SPECIFIC_TOOLS,
            tools=["write_file", "delete_file"],
            require_reason=True
        )
    ]
)

agent = Agent(
    pattern=pattern,
    tools=[read_file, write_file, delete_file],
    system_prompt="You are a file management assistant."
)

# When agent tries to write/delete, user will be prompted
result = await agent.run("Clean up all .tmp files in the current directory")
```

### 5.2 HITL with Webhook Integration

```python
from agent_framework.hitl import WebhookAdapter

interface = WebhookAdapter(
    approval_url="https://myapp.com/api/agent/approvals",
    notification_url="https://myapp.com/api/agent/notifications",
    auth_header="Bearer ${AGENT_API_KEY}",
    response_poll_interval=10.0,
    response_timeout=600.0
)

pattern = HumanInTheLoopPattern(
    interface=interface,
    require_approval_for_destructive=True,
    default_timeout_action=TimeoutAction.ESCALATE,
    escalation_target="ops-team@example.com"
)

# Approval requests will be sent to webhook, responses polled
agent = Agent(pattern=pattern, ...)
```

### 5.3 Combining HITL with Other Patterns

```python
from agent_framework.patterns import ReActPattern, HumanInTheLoopPattern

# HITL wraps the inner pattern
react_pattern = ReActPattern(max_iterations=10)

hitl_pattern = HumanInTheLoopPattern(
    interface=console_interface,
    inner_pattern=react_pattern,  # Compose patterns
    gates=[
        ApprovalGate(name="costly", gate_type=GateType.BUDGET_BASED, cost_threshold=1.0)
    ]
)

agent = Agent(pattern=hitl_pattern, ...)
```

### 5.4 Custom Approval Logic

```python
def custom_gate_condition(action: ProposedAction, context: ExecutionContext) -> bool:
    """Require approval for actions on production systems."""
    if action.tool_name == "run_bash":
        command = action.parameters.get("command", "")
        if "prod" in command or "production" in command:
            return True
    return False

pattern = HumanInTheLoopPattern(
    interface=interface,
    gates=[
        ApprovalGate(
            name="production_safety",
            gate_type=GateType.CUSTOM,
            condition=custom_gate_condition,
            timeout_action=TimeoutAction.REJECT
        )
    ]
)
```

---

## 6. Success Criteria

| Criterion | Measurement |
|-----------|-------------|
| Gate Triggering | Correct gates triggered 100% of time |
| Timeout Handling | Timeouts handled per configuration 100% of time |
| Feedback Integration | Rejected actions not repeated within session 95%+ |
| Audit Completeness | All interactions logged with full context 100% |
| UX Efficiency | Average approval time < 30s for routine actions |
| Adapter Compatibility | All built-in adapters pass integration tests |

---

## 7. Open Questions

1. Should there be "auto-approve" rules that learn from human patterns?
2. How to handle conflicting approvals in multi-approver scenarios?
3. Should risk assessment be pluggable with custom models?
4. How to balance security with UX for frequent approval requests?
5. Should there be approval delegation (e.g., "approve all similar actions for next hour")?
