"""
test_evaluate_gates.py — Tests for FeatureEngineer.evaluate_gates().

Group 10: Individual gate failures, all-pass, gate values, settings thresholds.
"""

import pandas as pd
import pytest

from src.ml_engine.features import TradeEligibility


def _gate_df(**overrides):
    """
    Build a minimal feature DataFrame where ALL four gates pass by default.
    Override individual fields to trigger specific gate failures.
    """
    defaults = {
        "BB_WIDTH": 0.020,       # > BB_WIDTH_MA (0.010) → pass
        "BB_WIDTH_MA": 0.010,
        "ATR": 0.0010,           # > ATR_MA (0.0005) → pass
        "ATR_MA": 0.0005,
        "RVOL": 2.0,             # > gate_min_rvol (1.5) → pass
        "SPREAD_NORMALIZED": 0.0001,  # < gate_max_spread (0.0005) → pass
    }
    defaults.update(overrides)
    return pd.DataFrame({k: [v] for k, v in defaults.items()})


# ── T68 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_empty_df_returns_ineligible(engineer):
    result = engineer.evaluate_gates(pd.DataFrame())
    assert result.is_eligible is False
    assert not any(result.gate_results.values())


# ── T69 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_all_pass_returns_eligible(engineer):
    result = engineer.evaluate_gates(_gate_df())
    assert result.is_eligible is True
    assert all(result.gate_results.values())


# ── T70 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_bb_fail_blocks_trade(engineer):
    fe = _gate_df(BB_WIDTH=0.005, BB_WIDTH_MA=0.020)  # BB_WIDTH < MA → fail
    result = engineer.evaluate_gates(fe)
    assert result.is_eligible is False


# ── T71 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_atr_fail_blocks_trade(engineer):
    fe = _gate_df(ATR=0.0001, ATR_MA=0.001)  # ATR < MA → fail
    result = engineer.evaluate_gates(fe)
    assert result.is_eligible is False


# ── T72 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_rvol_fail_blocks_trade(engineer):
    fe = _gate_df(RVOL=0.5)  # < gate_min_rvol (1.5) → fail
    result = engineer.evaluate_gates(fe)
    assert result.is_eligible is False


# ── T73 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_spread_fail_blocks_trade(engineer):
    fe = _gate_df(SPREAD_NORMALIZED=0.001)  # > gate_max_spread (0.0005) → fail
    result = engineer.evaluate_gates(fe)
    assert result.is_eligible is False


# ── T74 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_gate_values_populated(engineer):
    result = engineer.evaluate_gates(_gate_df())
    assert len(result.gate_values) > 0
    for v in result.gate_values.values():
        assert isinstance(v, float)


# ── T75 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_uses_settings_thresholds(engineer, mock_settings):
    """Gate key strings must embed the configured threshold value."""
    result = engineer.evaluate_gates(_gate_df())
    rvol_threshold = mock_settings.gate_min_rvol
    matching = [k for k in result.gate_results if str(rvol_threshold) in k]
    assert len(matching) == 1, f"Expected gate key containing {rvol_threshold}"


# ── T76 ───────────────────────────────────────────────────────────────────────


def test_evaluate_gates_returns_trade_eligibility_type(engineer):
    result = engineer.evaluate_gates(_gate_df())
    assert isinstance(result, TradeEligibility)
