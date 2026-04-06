"""
Inference Script — Email Triage Environment
===========================================

Runs an LLM agent against all 3 tasks and emits structured stdout logs in
the mandatory [START] / [STEP] / [END] format required by the OpenEnv spec.

Environment variables:
    API_BASE_URL   LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       HuggingFace / API key (required)
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

from email_triage_env import (
    Action,
    EmailCategory,
    EmailTriageEnv,
    Priority,
    RoutingTeam,
    Sentiment,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "email-triage-env"
MAX_STEPS = 10
TEMPERATURE = 0.2

TASKS = [
    "classify_and_prioritize",
    "route_and_summarize",
    "full_triage_pipeline",
]

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(task_id: str, email_subject: str, email_body: str, sender: str, instruction: str) -> str:
    schema_hint = {
        "classify_and_prioritize": {
            "category": "billing | technical | general | abuse | refund | account",
            "priority": "low | medium | high | critical",
        },
        "route_and_summarize": {
            "routing": "billing_team | tech_support | customer_success | trust_and_safety | account_management | general_support",
            "summary": "1-2 sentence summary of the issue",
            "extracted_entities": {"customer_id": "...", "product": "...", "order_id": "..."},
        },
        "full_triage_pipeline": {
            "category": "billing | technical | general | abuse | refund | account",
            "priority": "low | medium | high | critical",
            "routing": "billing_team | tech_support | customer_success | trust_and_safety | account_management | general_support",
            "extracted_entities": {"customer_id": "...", "product": "..."},
            "sentiment": "positive | neutral | frustrated | angry",
            "policy_violation": "true | false",
            "draft_response": "Professional 2-4 sentence first-response email",
        },
    }

    return textwrap.dedent(f"""
        {instruction}

        --- EMAIL ---
        From: {sender}
        Subject: {email_subject}

        {email_body}
        --- END EMAIL ---

        Respond ONLY with a valid JSON object matching this schema (no extra keys, no markdown):
        {json.dumps(schema_hint[task_id], indent=2)}
    """).strip()


# ---------------------------------------------------------------------------
# LLM call + parse
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert customer support triage agent. "
                        "Always respond with valid JSON only — no markdown, no explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f'{{"error": "{str(e)}"}}'


def parse_action(raw: str, task_id: str) -> Action:
    """Parse LLM JSON output into a typed Action."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to extract JSON substring
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                data = {}
        else:
            data = {}

    # Coerce string "true"/"false" for policy_violation
    if "policy_violation" in data:
        pv = data["policy_violation"]
        if isinstance(pv, str):
            data["policy_violation"] = pv.lower() == "true"

    # Only pass fields relevant to this task
    allowed = {
        "classify_and_prioritize": ["category", "priority"],
        "route_and_summarize": ["routing", "summary", "extracted_entities"],
        "full_triage_pipeline": [
            "category", "priority", "routing", "extracted_entities",
            "sentiment", "policy_violation", "draft_response",
        ],
    }
    filtered = {k: v for k, v in data.items() if k in allowed.get(task_id, [])}
    filtered["raw"] = raw
    return Action(**filtered)


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> Dict[str, Any]:
    env = EmailTriageEnv(task_id=task_id, max_steps=MAX_STEPS)
    obs = env.reset()
    step_num = 0
    rewards: List[float] = []
    success = False
    last_error = None

    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        while not obs.done:
            if obs.current_email is None:
                break

            email = obs.current_email
            prompt = build_prompt(
                task_id=task_id,
                email_subject=email.subject,
                email_body=email.body,
                sender=email.sender,
                instruction=obs.instruction,
            )

            raw_response = call_llm(prompt)
            action = parse_action(raw_response, task_id)
            action_str = json.dumps(action.model_dump(exclude_none=True, exclude={"raw"}))

            obs, reward, done, info = env.step(action)
            step_num += 1
            rewards.append(reward.value)
            last_error = None

            print(
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward={reward.value:.2f} "
                f"done={'true' if done else 'false'} "
                f"error=null",
                flush=True,
            )

        final_score = env.final_score
        success = final_score >= 0.5
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={'true' if success else 'false'} "
            f"steps={step_num} "
            f"score={final_score:.2f} "
            f"rewards={rewards_str}",
            flush=True,
        )

        return {"task_id": task_id, "score": final_score, "steps": step_num, "rewards": rewards}

    except Exception as exc:
        last_error = str(exc)
        traceback.print_exc(file=sys.stderr)
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
        print(
            f"[END] success=false steps={step_num} score=0.00 rewards={rewards_str}",
            flush=True,
        )
        return {"task_id": task_id, "score": 0.0, "steps": step_num, "rewards": rewards, "error": last_error}
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("ERROR: Set HF_TOKEN or API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    print(f"# Email Triage Env — Baseline Inference", flush=True)
    print(f"# Model: {MODEL_NAME}", flush=True)
    print(f"# Endpoint: {API_BASE_URL}", flush=True)
    print(f"# Tasks: {TASKS}", flush=True)
    print("", flush=True)

    all_results = []
    for task_id in TASKS:
        result = run_task(task_id)
        all_results.append(result)
        print("", flush=True)

    # Summary
    print("# ---- SUMMARY ----", flush=True)
    total = 0.0
    for r in all_results:
        score = r["score"]
        total += score
        print(f"# {r['task_id']}: score={score:.4f} steps={r['steps']}", flush=True)
    avg = total / len(all_results)
    print(f"# OVERALL AVERAGE SCORE: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()