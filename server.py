"""
FastAPI server — exposes the Email Triage Environment as an HTTP API.
Endpoints: POST /reset, POST /step, GET /state, GET /tasks, GET /health
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from email_triage_env import (
    Action,
    EmailCategory,
    EmailTriageEnv,
    EnvironmentState,
    Observation,
    Priority,
    Reward,
    RoutingTeam,
    Sentiment,
)

app = FastAPI(
    title="Email Triage Environment",
    description="OpenEnv-compliant email and customer support triage benchmark.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store (single-session for simplicity; extend for multi-user)
# ---------------------------------------------------------------------------
_sessions: Dict[str, EmailTriageEnv] = {}


def _get_env(session_id: str) -> EmailTriageEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "classify_and_prioritize"
    max_steps: int = 10
    session_id: str = "default"


class StepRequest(BaseModel):
    action: Action
    session_id: str = "default"


class ResetResponse(BaseModel):
    observation: Observation
    session_id: str
    task_id: str


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "email-triage-env"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "id": "classify_and_prioritize",
                "name": "Classify & Prioritize",
                "difficulty": "easy",
                "description": "Classify email by category and assign priority level.",
            },
            {
                "id": "route_and_summarize",
                "name": "Route & Summarize",
                "difficulty": "medium",
                "description": "Route ticket to correct team, summarize, extract entities.",
            },
            {
                "id": "full_triage_pipeline",
                "name": "Full Triage Pipeline",
                "difficulty": "hard",
                "description": "Complete triage: classify, route, extract, detect sentiment, flag violations, draft response.",
            },
        ]
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: Optional[ResetRequest] = Body(default=None)):
    
    # ✅ fallback if evaluator sends empty body
    if req is None:
        req = ResetRequest(
            task_id="classify_and_prioritize",
            session_id="default-session",
            max_steps=10
        )

    valid_tasks = EmailTriageEnv.VALID_TASKS
    if req.task_id not in valid_tasks:
        raise HTTPException(status_code=400, detail=f"task_id must be one of {valid_tasks}")

    env = EmailTriageEnv(task_id=req.task_id, max_steps=req.max_steps)
    obs = env.reset()
    _sessions[req.session_id] = env

    return ResetResponse(
        observation=obs,
        session_id=req.session_id,
        task_id=req.task_id
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.session_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvironmentState)
def state(session_id: str = Query("default")):
    env = _get_env(session_id)
    return env.state()


@app.get("/score")
def score(session_id: str = Query("default")):
    env = _get_env(session_id)
    s = env.state()
    return {
        "session_id": session_id,
        "task_id": s.task_id,
        "final_score": env.final_score,
        "total_reward": s.total_reward,
        "emails_processed": s.emails_processed,
        "episode_rewards": env.episode_rewards,
        "done": s.done,
    }

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
if __name__ == "__main__":
    main()
