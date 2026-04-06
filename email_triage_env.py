"""
Email Triage Environment — OpenEnv-compliant implementation.

Simulates a real customer support inbox where an agent must:
  1. Classify emails by category and priority
  2. Route tickets to the correct team
  3. Extract key entities and summarize
  4. Draft professional first-response emails
  5. Detect policy violations and sentiment

All models are Pydantic-typed. step()/reset()/state() follow the OpenEnv spec.
"""

from __future__ import annotations

import copy
import json
import random
import re
import textwrap
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EmailCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    ABUSE = "abuse"
    REFUND = "refund"
    ACCOUNT = "account"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RoutingTeam(str, Enum):
    BILLING_TEAM = "billing_team"
    TECH_SUPPORT = "tech_support"
    CUSTOMER_SUCCESS = "customer_success"
    TRUST_AND_SAFETY = "trust_and_safety"
    ACCOUNT_MANAGEMENT = "account_management"
    GENERAL_SUPPORT = "general_support"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


# ---------------------------------------------------------------------------
# Typed Models
# ---------------------------------------------------------------------------

class Email(BaseModel):
    email_id: str
    subject: str
    body: str
    sender: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """What the agent sees at each step."""
    task_id: str
    step: int
    current_email: Optional[Email] = None
    inbox: List[Email] = Field(default_factory=list)
    instruction: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    done: bool = False


class Action(BaseModel):
    """Structured triage action the agent submits."""
    # Task 1: Classify & Prioritize
    category: Optional[EmailCategory] = None
    priority: Optional[Priority] = None

    # Task 2: Route & Summarize
    routing: Optional[RoutingTeam] = None
    summary: Optional[str] = None
    extracted_entities: Optional[Dict[str, str]] = None  # e.g. {"customer_id": "...", "product": "..."}

    # Task 3: Full Pipeline
    sentiment: Optional[Sentiment] = None
    policy_violation: Optional[bool] = None
    draft_response: Optional[str] = None

    # Free-form raw action string for logging
    raw: Optional[str] = None


class Reward(BaseModel):
    value: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""


class EnvironmentState(BaseModel):
    task_id: str
    step: int
    max_steps: int
    done: bool
    total_reward: float
    emails_processed: int
    history: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Email Dataset
# ---------------------------------------------------------------------------

EMAILS: List[Dict] = [
    {
        "email_id": "E001",
        "subject": "I was charged twice for my subscription!",
        "body": (
            "Hi there,\n\nI noticed that my credit card was charged $49.99 twice this month "
            "for my Pro subscription. My customer ID is CID-84821. This is really frustrating "
            "and I need a refund immediately. Please look into this ASAP.\n\nThanks,\nJohn"
        ),
        "sender": "john.doe@example.com",
        "timestamp": "2024-11-01T09:14:00Z",
        "metadata": {
            "customer_id": "CID-84821",
            "product": "Pro Subscription",
            "ground_truth": {
                "category": "billing",
                "priority": "high",
                "routing": "billing_team",
                "sentiment": "frustrated",
                "policy_violation": False,
                "key_entities": {"customer_id": "CID-84821", "product": "Pro Subscription", "amount": "$49.99"},
            },
        },
    },
    {
        "email_id": "E002",
        "subject": "App keeps crashing on iOS 17",
        "body": (
            "Hello Support,\n\nYour mobile app crashes every time I try to open the dashboard on "
            "my iPhone 15 running iOS 17.0.3. I've tried reinstalling it three times. "
            "Error code: ERR_DASH_503. Order #ORD-2291 is affected. "
            "This is blocking my entire workflow.\n\nBest,\nSarah"
        ),
        "sender": "sarah.m@company.org",
        "timestamp": "2024-11-01T10:02:00Z",
        "metadata": {
            "customer_id": None,
            "product": "Mobile App",
            "ground_truth": {
                "category": "technical",
                "priority": "high",
                "routing": "tech_support",
                "sentiment": "frustrated",
                "policy_violation": False,
                "key_entities": {"order_id": "ORD-2291", "error_code": "ERR_DASH_503", "platform": "iOS 17"},
            },
        },
    },
    {
        "email_id": "E003",
        "subject": "How do I export my data?",
        "body": (
            "Hi,\n\nI'm trying to find the data export feature but can't locate it in the settings. "
            "Can you point me in the right direction? I use the Starter plan.\n\nThanks!"
        ),
        "sender": "curious_user@gmail.com",
        "timestamp": "2024-11-01T11:30:00Z",
        "metadata": {
            "customer_id": None,
            "product": "Starter Plan",
            "ground_truth": {
                "category": "general",
                "priority": "low",
                "routing": "general_support",
                "sentiment": "neutral",
                "policy_violation": False,
                "key_entities": {"plan": "Starter Plan"},
            },
        },
    },
    {
        "email_id": "E004",
        "subject": "URGENT: Security breach — someone accessed my account",
        "body": (
            "I received a login notification from an IP in Russia at 3AM. I did NOT log in. "
            "My account has sensitive financial data. Customer ID: CID-00129. "
            "You need to lock this account RIGHT NOW. I will sue if my data is compromised. "
            "This is an emergency!!!"
        ),
        "sender": "alarmed.user@finance.co",
        "timestamp": "2024-11-01T03:47:00Z",
        "metadata": {
            "customer_id": "CID-00129",
            "product": "Enterprise Account",
            "ground_truth": {
                "category": "account",
                "priority": "critical",
                "routing": "trust_and_safety",
                "sentiment": "angry",
                "policy_violation": False,
                "key_entities": {"customer_id": "CID-00129", "ip_location": "Russia"},
            },
        },
    },
    {
        "email_id": "E005",
        "subject": "I want a refund — this product is garbage",
        "body": (
            "I purchased the annual plan (order #ORD-5512) 10 days ago and it simply does not work "
            "as advertised. The AI features are broken and customer support has ignored me twice. "
            "I want a full refund of $299 processed within 24 hours or I'm disputing with my bank."
        ),
        "sender": "angry.customer@hotmail.com",
        "timestamp": "2024-11-01T14:20:00Z",
        "metadata": {
            "customer_id": None,
            "product": "Annual Plan",
            "ground_truth": {
                "category": "refund",
                "priority": "high",
                "routing": "billing_team",
                "sentiment": "angry",
                "policy_violation": False,
                "key_entities": {"order_id": "ORD-5512", "amount": "$299", "plan": "Annual Plan"},
            },
        },
    },
    {
        "email_id": "E006",
        "subject": "How to add a team member?",
        "body": (
            "Hey,\n\nQuick question — how do I invite a colleague to my workspace? "
            "I'm on the Team plan. Checked the docs but couldn't find it.\n\nCheers,\nAlex"
        ),
        "sender": "alex@startup.io",
        "timestamp": "2024-11-01T15:00:00Z",
        "metadata": {
            "customer_id": None,
            "product": "Team Plan",
            "ground_truth": {
                "category": "general",
                "priority": "low",
                "routing": "general_support",
                "sentiment": "neutral",
                "policy_violation": False,
                "key_entities": {"plan": "Team Plan"},
            },
        },
    },
    {
        "email_id": "E007",
        "subject": "Spam and phishing links being sent from your platform",
        "body": (
            "Your platform is being used to send phishing emails targeting my employees. "
            "I have the originating account email: spammer123@yourdomain.com. "
            "We've had 3 employees click malicious links already. "
            "This is a serious abuse of your service and I demand immediate action."
        ),
        "sender": "ciso@enterprise.com",
        "timestamp": "2024-11-01T08:05:00Z",
        "metadata": {
            "customer_id": None,
            "product": "Platform",
            "ground_truth": {
                "category": "abuse",
                "priority": "critical",
                "routing": "trust_and_safety",
                "sentiment": "angry",
                "policy_violation": True,
                "key_entities": {"abuser_account": "spammer123@yourdomain.com"},
            },
        },
    },
]


def _get_email(email_id: str) -> Email:
    for e in EMAILS:
        if e["email_id"] == email_id:
            return Email(**{k: v for k, v in e.items() if k != "metadata"},
                         metadata=e["metadata"])
    raise ValueError(f"Email {email_id} not found")


def _all_emails() -> List[Email]:
    return [Email(**{k: v for k, v in e.items() if k != "metadata"},
                  metadata=e["metadata"]) for e in EMAILS]


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _score_category(predicted: Optional[str], truth: str) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted.lower() == truth.lower() else 0.0


def _score_priority(predicted: Optional[str], truth: str) -> float:
    if predicted is None:
        return 0.0
    priority_order = ["low", "medium", "high", "critical"]
    if predicted.lower() == truth.lower():
        return 1.0
    try:
        pred_idx = priority_order.index(predicted.lower())
        truth_idx = priority_order.index(truth.lower())
        diff = abs(pred_idx - truth_idx)
        return max(0.0, 1.0 - diff * 0.4)
    except ValueError:
        return 0.0


def _score_routing(predicted: Optional[str], truth: str) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted.lower() == truth.lower() else 0.0


def _score_summary(summary: Optional[str], email: Email) -> float:
    if not summary or len(summary.strip()) < 10:
        return 0.0
    score = 0.0
    text = summary.lower()
    # Check key subject words appear
    subject_words = [w.lower() for w in email.subject.split() if len(w) > 3]
    hits = sum(1 for w in subject_words if w in text)
    score += min(0.5, hits / max(len(subject_words), 1) * 0.5)
    # Length heuristic: good summaries are 20-200 chars
    l = len(summary.strip())
    if 20 <= l <= 200:
        score += 0.3
    elif l < 20:
        score += 0.1
    else:
        score += 0.2
    # Penalize if it's just the subject verbatim
    if summary.strip().lower() == email.subject.lower():
        score *= 0.3
    return min(1.0, score + 0.2)


def _score_entities(predicted: Optional[Dict], truth: Dict) -> float:
    if not predicted or not truth:
        return 0.0 if truth else 1.0
    hits = 0
    for k, v in truth.items():
        pred_val = predicted.get(k, "")
        if pred_val and str(v).lower() in str(pred_val).lower():
            hits += 1
    return hits / len(truth) if truth else 1.0


def _score_sentiment(predicted: Optional[str], truth: str) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted.lower() == truth.lower() else 0.2


def _score_policy_violation(predicted: Optional[bool], truth: bool) -> float:
    if predicted is None:
        return 0.0
    return 1.0 if predicted == truth else 0.0


def _score_draft_response(draft: Optional[str], email: Email, category: str) -> float:
    """Heuristic scoring for draft response quality."""
    if not draft or len(draft.strip()) < 30:
        return 0.0
    score = 0.0
    text = draft.lower()

    # Professional greeting
    if any(g in text for g in ["dear", "hello", "hi ", "thank you for"]):
        score += 0.15
    # Acknowledges the issue
    subject_kws = [w.lower() for w in email.subject.split() if len(w) > 4]
    if any(kw in text for kw in subject_kws):
        score += 0.2
    # Offers help / next steps
    if any(w in text for w in ["will", "team", "resolve", "help", "look into", "investigate", "contact"]):
        score += 0.2
    # Professional closing
    if any(c in text for c in ["regards", "sincerely", "best", "thank you", "support team"]):
        score += 0.15
    # Appropriate length
    if 100 <= len(draft) <= 600:
        score += 0.2
    elif 50 <= len(draft) < 100:
        score += 0.1
    # Does not contain placeholder text like [NAME]
    if "[" not in draft and "{" not in draft:
        score += 0.1

    return min(1.0, score)


def grade_task1(action: Action, email: Email) -> Reward:
    """Easy: Classify by category and assign priority."""
    gt = email.metadata.get("ground_truth", {})
    cat_score = _score_category(action.category, gt.get("category", "general"))
    pri_score = _score_priority(action.priority, gt.get("priority", "low"))
    total = cat_score * 0.5 + pri_score * 0.5
    return Reward(
        value=round(total, 4),
        breakdown={"category": cat_score, "priority": pri_score},
        feedback=f"Category: {'✓' if cat_score == 1.0 else '✗'} | Priority: {'✓' if pri_score == 1.0 else f'partial ({pri_score:.2f})'}",
    )


def grade_task2(action: Action, email: Email) -> Reward:
    """Medium: Route to correct team, summarize, extract entities."""
    gt = email.metadata.get("ground_truth", {})
    route_score = _score_routing(action.routing, gt.get("routing", "general_support"))
    summary_score = _score_summary(action.summary, email)
    entity_score = _score_entities(action.extracted_entities, gt.get("key_entities", {}))
    total = route_score * 0.4 + summary_score * 0.3 + entity_score * 0.3
    return Reward(
        value=round(total, 4),
        breakdown={"routing": route_score, "summary": summary_score, "entities": entity_score},
        feedback=f"Routing: {'✓' if route_score == 1.0 else '✗'} | Summary: {summary_score:.2f} | Entities: {entity_score:.2f}",
    )


def grade_task3(action: Action, email: Email) -> Reward:
    """Hard: Full pipeline — all 7 sub-criteria."""
    gt = email.metadata.get("ground_truth", {})

    cat_score = _score_category(action.category, gt.get("category", "general"))
    pri_score = _score_priority(action.priority, gt.get("priority", "low"))
    route_score = _score_routing(action.routing, gt.get("routing", "general_support"))
    entity_score = _score_entities(action.extracted_entities, gt.get("key_entities", {}))
    sentiment_score = _score_sentiment(action.sentiment, gt.get("sentiment", "neutral"))
    policy_score = _score_policy_violation(action.policy_violation, gt.get("policy_violation", False))
    draft_score = _score_draft_response(action.draft_response, email, gt.get("category", "general"))

    weights = {
        "category": 0.15,
        "priority": 0.15,
        "routing": 0.15,
        "entities": 0.10,
        "sentiment": 0.10,
        "policy_violation": 0.10,
        "draft_response": 0.25,
    }
    breakdown = {
        "category": cat_score,
        "priority": pri_score,
        "routing": route_score,
        "entities": entity_score,
        "sentiment": sentiment_score,
        "policy_violation": policy_score,
        "draft_response": draft_score,
    }
    total = sum(breakdown[k] * weights[k] for k in breakdown)
    return Reward(
        value=round(total, 4),
        breakdown=breakdown,
        feedback=" | ".join(f"{k}: {v:.2f}" for k, v in breakdown.items()),
    )


GRADERS = {
    "classify_and_prioritize": grade_task1,
    "route_and_summarize": grade_task2,
    "full_triage_pipeline": grade_task3,
}

TASK_INSTRUCTIONS = {
    "classify_and_prioritize": (
        "You are a support triage agent. Read the incoming email and output:\n"
        "1. category — one of: billing, technical, general, abuse, refund, account\n"
        "2. priority — one of: low, medium, high, critical\n"
        "Respond with a valid Action JSON."
    ),
    "route_and_summarize": (
        "You are a support triage agent. For the given email:\n"
        "1. routing — which team should handle this: billing_team, tech_support, customer_success, "
        "trust_and_safety, account_management, general_support\n"
        "2. summary — a concise 1-2 sentence summary of the issue\n"
        "3. extracted_entities — key facts as a dict (e.g. customer_id, product, order_id, amount)\n"
        "Respond with a valid Action JSON."
    ),
    "full_triage_pipeline": (
        "You are a senior support triage agent. Perform a complete triage of this email:\n"
        "1. category — billing, technical, general, abuse, refund, account\n"
        "2. priority — low, medium, high, critical\n"
        "3. routing — billing_team, tech_support, customer_success, trust_and_safety, account_management, general_support\n"
        "4. extracted_entities — dict of key facts\n"
        "5. sentiment — positive, neutral, frustrated, angry\n"
        "6. policy_violation — true if this reports abuse/fraud/phishing, else false\n"
        "7. draft_response — a professional first-response email (2-4 sentences)\n"
        "Respond with a valid Action JSON."
    ),
}

TASK_EMAIL_IDS = {
    "classify_and_prioritize": ["E001", "E003", "E007"],
    "route_and_summarize": ["E002", "E004", "E005"],
    "full_triage_pipeline": ["E004", "E007", "E001", "E002", "E005"],
}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnv:
    """
    OpenEnv-compliant Email Triage Environment.

    Usage:
        env = EmailTriageEnv(task_id="classify_and_prioritize")
        obs = env.reset()
        action = Action(category="billing", priority="high")
        obs, reward, done, info = env.step(action)
    """

    VALID_TASKS = list(GRADERS.keys())

    def __init__(self, task_id: str = "classify_and_prioritize", max_steps: int = 10):
        if task_id not in self.VALID_TASKS:
            raise ValueError(f"task_id must be one of {self.VALID_TASKS}")
        self.task_id = task_id
        self.max_steps = max_steps
        self._reset_state()

    def _reset_state(self):
        self._step = 0
        self._done = False
        self._history: List[Dict[str, Any]] = []
        self._total_reward = 0.0
        self._email_queue = [_get_email(eid) for eid in TASK_EMAIL_IDS[self.task_id]]
        self._email_idx = 0
        self._rewards: List[float] = []

    def reset(self) -> Observation:
        """Reset to initial state and return the first observation."""
        self._reset_state()
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process one triage action.

        Returns:
            observation: next state
            reward: Reward object with value in [0, 1]
            done: whether the episode is finished
            info: auxiliary info dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        current_email = self._current_email()
        grader = GRADERS[self.task_id]
        reward = grader(action, current_email)

        self._step += 1
        self._email_idx += 1
        self._total_reward += reward.value
        self._rewards.append(reward.value)

        self._history.append({
            "step": self._step,
            "email_id": current_email.email_id,
            "action": action.model_dump(exclude_none=True),
            "reward": reward.value,
            "breakdown": reward.breakdown,
            "feedback": reward.feedback,
        })

        # Episode ends when all emails processed or max_steps reached
        self._done = (self._email_idx >= len(self._email_queue)) or (self._step >= self.max_steps)

        obs = self._make_observation()
        info = {
            "email_id": current_email.email_id,
            "reward_breakdown": reward.breakdown,
            "feedback": reward.feedback,
            "total_reward": self._total_reward,
            "avg_reward": self._total_reward / self._step,
        }
        return obs, reward, self._done, info

    def state(self) -> EnvironmentState:
        """Return current environment state (OpenEnv spec)."""
        return EnvironmentState(
            task_id=self.task_id,
            step=self._step,
            max_steps=self.max_steps,
            done=self._done,
            total_reward=self._total_reward,
            emails_processed=self._email_idx,
            history=self._history,
        )

    def _current_email(self) -> Email:
        if self._email_idx >= len(self._email_queue):
            raise IndexError("No more emails in queue.")
        return self._email_queue[self._email_idx]

    def _make_observation(self) -> Observation:
        current = None
        if not self._done and self._email_idx < len(self._email_queue):
            current = self._current_email()
        return Observation(
            task_id=self.task_id,
            step=self._step,
            current_email=current,
            inbox=self._email_queue,
            instruction=TASK_INSTRUCTIONS[self.task_id],
            history=self._history,
            done=self._done,
        )

    def close(self):
        """Clean up resources."""
        pass

    @property
    def episode_rewards(self) -> List[float]:
        return self._rewards

    @property
    def final_score(self) -> float:
        if not self._rewards:
            return 0.0
        return round(sum(self._rewards) / len(self._rewards), 4)