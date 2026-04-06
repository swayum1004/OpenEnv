"""
Tests for Email Triage Environment.
Run with: python -m pytest tests/test_env.py -v
"""

import pytest
from email_triage_env import (
    Action,
    EmailCategory,
    EmailTriageEnv,
    Priority,
    RoutingTeam,
    Sentiment,
    _get_email,
    grade_task1,
    grade_task2,
    grade_task3,
)


# ---------------------------------------------------------------------------
# Grader unit tests
# ---------------------------------------------------------------------------

class TestGradeTask1:
    def test_perfect_score(self):
        email = _get_email("E001")
        action = Action(category=EmailCategory.BILLING, priority=Priority.HIGH)
        reward = grade_task1(action, email)
        assert reward.value == 1.0

    def test_wrong_category(self):
        email = _get_email("E001")
        action = Action(category=EmailCategory.TECHNICAL, priority=Priority.HIGH)
        reward = grade_task1(action, email)
        assert reward.breakdown["category"] == 0.0
        assert reward.breakdown["priority"] == 1.0
        assert reward.value == 0.5

    def test_adjacent_priority_partial(self):
        email = _get_email("E001")  # ground truth: high
        action = Action(category=EmailCategory.BILLING, priority=Priority.MEDIUM)
        reward = grade_task1(action, email)
        assert reward.breakdown["priority"] == pytest.approx(0.6, abs=0.05)

    def test_none_action(self):
        email = _get_email("E003")
        action = Action()
        reward = grade_task1(action, email)
        assert reward.value == 0.0

    def test_critical_email(self):
        email = _get_email("E007")  # abuse, critical
        action = Action(category=EmailCategory.ABUSE, priority=Priority.CRITICAL)
        reward = grade_task1(action, email)
        assert reward.value == 1.0


class TestGradeTask2:
    def test_perfect_routing(self):
        email = _get_email("E002")  # tech_support
        action = Action(
            routing=RoutingTeam.TECH_SUPPORT,
            summary="User reports mobile app crashing on iOS 17 with error ERR_DASH_503.",
            extracted_entities={"order_id": "ORD-2291", "error_code": "ERR_DASH_503", "platform": "iOS 17"},
        )
        reward = grade_task2(action, email)
        assert reward.breakdown["routing"] == 1.0
        assert reward.value > 0.7

    def test_wrong_routing_zero(self):
        email = _get_email("E002")
        action = Action(routing=RoutingTeam.BILLING_TEAM, summary="Some issue")
        reward = grade_task2(action, email)
        assert reward.breakdown["routing"] == 0.0

    def test_empty_summary_penalty(self):
        email = _get_email("E002")
        action = Action(routing=RoutingTeam.TECH_SUPPORT, summary="")
        reward = grade_task2(action, email)
        assert reward.breakdown["summary"] == 0.0


class TestGradeTask3:
    def test_full_perfect(self):
        email = _get_email("E007")  # abuse, critical, trust_and_safety
        action = Action(
            category=EmailCategory.ABUSE,
            priority=Priority.CRITICAL,
            routing=RoutingTeam.TRUST_AND_SAFETY,
            extracted_entities={"abuser_account": "spammer123@yourdomain.com"},
            sentiment=Sentiment.ANGRY,
            policy_violation=True,
            draft_response=(
                "Dear valued customer, thank you for reporting this serious security concern. "
                "Our Trust & Safety team will investigate the reported account immediately and take "
                "appropriate action. We take abuse of our platform extremely seriously and will contact "
                "you with an update within 2 hours. Best regards, Support Team"
            ),
        )
        reward = grade_task3(action, email)
        assert reward.value > 0.8

    def test_missing_draft_penalty(self):
        email = _get_email("E004")
        action = Action(
            category=EmailCategory.ACCOUNT,
            priority=Priority.CRITICAL,
            routing=RoutingTeam.TRUST_AND_SAFETY,
            sentiment=Sentiment.ANGRY,
            policy_violation=False,
        )
        reward = grade_task3(action, email)
        assert reward.breakdown["draft_response"] == 0.0
        assert reward.value < 0.8


# ---------------------------------------------------------------------------
# Environment integration tests
# ---------------------------------------------------------------------------

class TestEmailTriageEnv:
    def test_reset_returns_observation(self):
        env = EmailTriageEnv("classify_and_prioritize")
        obs = env.reset()
        assert obs.task_id == "classify_and_prioritize"
        assert obs.step == 0
        assert obs.current_email is not None
        assert not obs.done

    def test_step_advances(self):
        env = EmailTriageEnv("classify_and_prioritize")
        env.reset()
        action = Action(category=EmailCategory.BILLING, priority=Priority.HIGH)
        obs, reward, done, info = env.step(action)
        assert obs.step == 1
        assert 0.0 <= reward.value <= 1.0

    def test_episode_completes(self):
        env = EmailTriageEnv("classify_and_prioritize")
        obs = env.reset()
        step = 0
        while not obs.done:
            action = Action(category=EmailCategory.GENERAL, priority=Priority.LOW)
            obs, reward, done, info = env.step(action)
            step += 1
            if step > 20:
                raise AssertionError("Episode did not terminate")
        assert obs.done

    def test_step_after_done_raises(self):
        env = EmailTriageEnv("classify_and_prioritize")
        obs = env.reset()
        while not obs.done:
            obs, _, _, _ = env.step(Action(category=EmailCategory.GENERAL, priority=Priority.LOW))
        with pytest.raises(RuntimeError):
            env.step(Action(category=EmailCategory.GENERAL, priority=Priority.LOW))

    def test_state(self):
        env = EmailTriageEnv("route_and_summarize")
        env.reset()
        state = env.state()
        assert state.task_id == "route_and_summarize"
        assert state.step == 0

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            EmailTriageEnv("nonexistent_task")

    def test_final_score_in_range(self):
        env = EmailTriageEnv("full_triage_pipeline")
        obs = env.reset()
        while not obs.done:
            action = Action(
                category=EmailCategory.BILLING,
                priority=Priority.HIGH,
                routing=RoutingTeam.BILLING_TEAM,
                sentiment=Sentiment.FRUSTRATED,
                policy_violation=False,
                draft_response="Thank you for reaching out. Our team will investigate and respond shortly. Best regards.",
            )
            obs, _, _, _ = env.step(action)
        score = env.final_score
        assert 0.0 <= score <= 1.0

    def test_reward_breakdown_keys(self):
        env = EmailTriageEnv("full_triage_pipeline")
        env.reset()
        action = Action(
            category=EmailCategory.ACCOUNT,
            priority=Priority.CRITICAL,
            routing=RoutingTeam.TRUST_AND_SAFETY,
            sentiment=Sentiment.ANGRY,
            policy_violation=True,
            draft_response="We have received your security report and are acting immediately.",
        )
        _, reward, _, _ = env.step(action)
        expected_keys = {"category", "priority", "routing", "entities", "sentiment", "policy_violation", "draft_response"}
        assert set(reward.breakdown.keys()) == expected_keys


if __name__ == "__main__":
    pytest.main([__file__, "-v"])