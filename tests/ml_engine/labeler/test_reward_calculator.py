"""
test_reward_calculator.py — Tests for RewardCalculator.__init__ and calculate_reward.

Group 5: Construction — payout validation, boundary conditions, repr.
Group 6: calculate_reward — SKIP, invalid actions, all direction×gate combinations,
         return type.
"""

import pytest
from src.ml_engine.labeler import RewardCalculator


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 5: RewardCalculator.__init__
# ═══════════════════════════════════════════════════════════════════════════════


# ── T32 ───────────────────────────────────────────────────────────────────────


def test_rc_init_valid_payout():
    """A payout_ratio of 0.85 constructs without error and is stored."""
    rc = RewardCalculator(payout_ratio=0.85)
    assert rc.payout_ratio == pytest.approx(0.85)


# ── T33 ───────────────────────────────────────────────────────────────────────


def test_rc_init_zero_payout_raises():
    """0.0 is outside (0, 1] — raises ValueError."""
    with pytest.raises(ValueError, match="payout_ratio"):
        RewardCalculator(payout_ratio=0.0)


# ── T34 ───────────────────────────────────────────────────────────────────────


def test_rc_init_negative_payout_raises():
    """Negative payout_ratio raises ValueError."""
    with pytest.raises(ValueError):
        RewardCalculator(payout_ratio=-0.5)


# ── T35 ───────────────────────────────────────────────────────────────────────


def test_rc_init_above_one_raises():
    """payout_ratio > 1.0 raises ValueError."""
    with pytest.raises(ValueError):
        RewardCalculator(payout_ratio=1.01)


# ── T36 ───────────────────────────────────────────────────────────────────────


def test_rc_init_exactly_one_valid():
    """1.0 is the valid upper boundary — must not raise."""
    rc = RewardCalculator(payout_ratio=1.0)
    assert rc.payout_ratio == pytest.approx(1.0)


# ── T37 ───────────────────────────────────────────────────────────────────────


def test_rc_repr():
    """repr() includes payout_ratio formatted to 2 decimal places."""
    rc = RewardCalculator(payout_ratio=0.85)
    r = repr(rc)
    assert "0.85" in r


# ═══════════════════════════════════════════════════════════════════════════════
# GROUP 6: RewardCalculator.calculate_reward
# ═══════════════════════════════════════════════════════════════════════════════


# ── T38 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_skip_returns_zero(rc):
    """action=2 (SKIP) always returns 0.0 — no financial outcome."""
    assert rc.calculate_reward(action=2, is_correct=True, gate_passed=True) == pytest.approx(0.0)


# ── T39 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_invalid_action_high(rc):
    """action=3 is outside the action space — raises ValueError."""
    with pytest.raises(ValueError, match="Invalid action"):
        rc.calculate_reward(action=3, is_correct=True, gate_passed=True)


# ── T40 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_invalid_action_negative(rc):
    """action=-1 raises ValueError — only 0, 1, 2 are valid."""
    with pytest.raises(ValueError):
        rc.calculate_reward(action=-1, is_correct=True, gate_passed=True)


# ── T41 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_call_correct_gate_passed(rc):
    """CALL, correct, gate passed: payout_ratio + 0.1 = 0.85 + 0.1 = 0.95."""
    reward = rc.calculate_reward(action=0, is_correct=True, gate_passed=True)
    assert reward == pytest.approx(0.95)


# ── T42 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_call_correct_gate_failed(rc):
    """CALL, correct, gate failed: payout_ratio - 0.2 = 0.85 - 0.2 = 0.65."""
    reward = rc.calculate_reward(action=0, is_correct=True, gate_passed=False)
    assert reward == pytest.approx(0.65)


# ── T43 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_call_wrong_gate_passed(rc):
    """CALL, wrong, gate passed: -1.0 + 0.1 = -0.9."""
    reward = rc.calculate_reward(action=0, is_correct=False, gate_passed=True)
    assert reward == pytest.approx(-0.9)


# ── T44 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_call_wrong_gate_failed(rc):
    """CALL, wrong, gate failed: -1.0 - 0.2 = -1.2."""
    reward = rc.calculate_reward(action=0, is_correct=False, gate_passed=False)
    assert reward == pytest.approx(-1.2)


# ── T45 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_put_correct_gate_passed(rc):
    """PUT uses the same reward formula as CALL — direction is symmetric."""
    reward = rc.calculate_reward(action=1, is_correct=True, gate_passed=True)
    assert reward == pytest.approx(0.95)


# ── T46 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_put_wrong_gate_failed(rc):
    """PUT, wrong, gate failed: -1.0 - 0.2 = -1.2."""
    reward = rc.calculate_reward(action=1, is_correct=False, gate_passed=False)
    assert reward == pytest.approx(-1.2)


# ── T47 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_returns_float(rc):
    """Return type is float — not int, even when the value is whole."""
    reward = rc.calculate_reward(action=2, is_correct=False, gate_passed=False)
    assert isinstance(reward, float)


# ── T48 ───────────────────────────────────────────────────────────────────────


def test_calculate_reward_skip_ignores_is_correct_and_gate(rc):
    """SKIP returns 0.0 regardless of is_correct and gate_passed values."""
    assert rc.calculate_reward(action=2, is_correct=False, gate_passed=False) == pytest.approx(0.0)
    assert rc.calculate_reward(action=2, is_correct=True, gate_passed=False) == pytest.approx(0.0)
    assert rc.calculate_reward(action=2, is_correct=False, gate_passed=True) == pytest.approx(0.0)
