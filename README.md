# 📧 Email Triage Environment

> **OpenEnv-compliant benchmark for evaluating LLM agents on real-world customer support triage.**

An agent must read incoming support emails and perform the same workflow a trained support specialist does every day: classify tickets, assign priorities, route to the right team, extract key entities, detect customer sentiment, flag policy violations, and draft professional first-response emails.

---

## Why This Environment?

Customer support triage is one of the highest-volume knowledge-work tasks in the world. Every company with users operates a support queue. Automating or augmenting triage with AI agents is an immediate, practical use case — yet it requires:

- **Natural language understanding** (what is the customer's actual problem?)
- **Multi-label classification** (category + priority + routing simultaneously)
- **Information extraction** (customer IDs, order numbers, error codes)
- **Tone/sentiment awareness** (a billing question from an angry customer ≠ a curious one)
- **Policy reasoning** (is this reporting abuse? does it require escalation?)
- **Professional writing** (draft a response that doesn't make things worse)

No existing OpenEnv environment covers this domain.

---

## Environment Overview

| Property | Value |
|---|---|
| **Domain** | Customer Support Triage |
| **Tasks** | 3 (Easy → Medium → Hard) |
| **Reward** | Continuous [0.0, 1.0] per step |
| **Episode length** | Up to 10 steps |
| **Action type** | Structured JSON (Pydantic typed) |
| **Observation type** | Email + inbox snapshot + instruction |

---

## Action Space

```python
class Action(BaseModel):
    # Task 1 fields
    category: Optional[EmailCategory]     # billing | technical | general | abuse | refund | account
    priority: Optional[Priority]          # low | medium | high | critical

    # Task 2 fields
    routing: Optional[RoutingTeam]        # billing_team | tech_support | ... | general_support
    summary: Optional[str]               # 1-2 sentence issue summary
    extracted_entities: Optional[Dict]   # {"customer_id": "...", "product": "...", ...}

    # Task 3 fields
    sentiment: Optional[Sentiment]       # positive | neutral | frustrated | angry
    policy_violation: Optional[bool]     # True if abuse/fraud/phishing reported
    draft_response: Optional[str]        # Professional first-response email
```

## Observation Space

```python
class Observation(BaseModel):
    task_id: str                   # Active task name
    step: int                      # Current step number
    current_email: Optional[Email] # The email to triage right now
    inbox: List[Email]             # Full inbox snapshot
    instruction: str               # Task-specific agent instruction
    history: List[Dict]            # Previous steps with rewards
    done: bool                     # Episode complete?
```

---

## Tasks

### Task 1 — Classify & Prioritize (Easy)

**Objective:** Given one support email, output the correct `category` and `priority`.

**Grader:**
- `category` correct → +0.50
- `priority` correct → +0.50 (partial credit for adjacent levels, e.g. high vs critical = 0.60)

**Example email:** *"I was charged twice for my subscription!"* → `billing`, `high`

**Baseline expectation:** A capable 7B model should score 0.7+.

---

### Task 2 — Route & Summarize (Medium)

**Objective:** Route the ticket to the correct internal team, write a concise summary, and extract key entities.

**Grader:**
- `routing` correct → +0.40
- `summary` quality (heuristic: relevance, length, information density) → 0–0.30
- `extracted_entities` coverage vs ground truth → 0–0.30

**Example email:** *"App keeps crashing on iOS 17, order #ORD-2291, error ERR_DASH_503"*
→ `tech_support`, extract `{order_id, error_code, platform}`

**Baseline expectation:** Frontier models score 0.65–0.85. Smaller models struggle with entity extraction.

---

### Task 3 — Full Triage Pipeline (Hard)

**Objective:** Complete end-to-end triage across 7 weighted sub-criteria.

| Sub-criterion | Weight |
|---|---|
| category | 15% |
| priority | 15% |
| routing | 15% |
| extracted_entities | 10% |
| sentiment | 10% |
| policy_violation | 10% |
| draft_response | **25%** |

The `draft_response` is scored on: greeting present, issue acknowledged, next steps offered, professional closing, appropriate length (100–600 chars), no unfilled placeholders.

**Baseline expectation:** GPT-4 class models score 0.70–0.85. This task genuinely challenges models on tone calibration and professional writing under constraints.

---

## Setup & Usage

### Local (Python)

```bash
git clone https://github.com/swayum1004/OpenEnv.git
cd email-triage-env
pip install -r requirements.txt
```

**Start the API server:**
```bash
python server.py
# → http://localhost:7860
```

**Run tests:**
```bash
pip install pytest
python -m pytest tests/test_env.py -v
```

**Run baseline inference:**
```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  email-triage-env
```

### Python SDK

```python
from email_triage_env import EmailTriageEnv, Action, EmailCategory, Priority

env = EmailTriageEnv(task_id="classify_and_prioritize")
obs = env.reset()

action = Action(category=EmailCategory.BILLING, priority=Priority.HIGH)
obs, reward, done, info = env.step(action)

print(f"Reward: {reward.value:.4f}")
print(f"Feedback: {reward.feedback}")
print(f"Final score: {env.final_score}")
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks with metadata |
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit triage action |
| `/state` | GET | Current environment state |
| `/score` | GET | Final score for session |

**Reset:**
```json
POST /reset
{"task_id": "classify_and_prioritize", "session_id": "my-run-1"}
```

**Step:**
```json
POST /step
{
  "session_id": "my-run-1",
  "action": {
    "category": "billing",
    "priority": "high"
  }
}
```

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task | Difficulty | Baseline Score |
|---|---|---|
| classify_and_prioritize | Easy | ~0.82 |
| route_and_summarize | Medium | ~0.68 |
| full_triage_pipeline | Hard | ~0.61 |
| **Overall Average** | — | **~0.70** |

*(Scores are reproducible — set `TEMPERATURE=0.2` in inference.py)*

---

## OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

All three checks should pass:
- ✅ `openenv.yaml` present and valid
- ✅ `step()` / `reset()` / `state()` implemented with typed models
- ✅ Rewards in [0.0, 1.0]

---

## Project Structure

```
email-triage-env/
├── email_triage_env.py   # Core environment (models, graders, EmailTriageEnv)
├── server.py             # FastAPI HTTP server
├── inference.py          # Baseline inference script (mandatory)
├── openenv.yaml          # OpenEnv metadata
├── requirements.txt
├── Dockerfile
├── tests/
│   └── test_env.py       # Pytest unit + integration tests
└── README.md
```

---

## License

Swayum Hastwala
Farhan Ansari
Bhavana Koli
