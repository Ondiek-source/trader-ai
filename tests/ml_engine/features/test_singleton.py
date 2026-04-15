"""
test_singleton.py — Tests for get_feature_engineer() singleton.

Group 13: Lazy init, identity across calls, fresh instance after reset.

All tests reset _engineer to None via monkeypatch and also patch get_settings()
so the constructor never hits the real filesystem.
"""

import pytest
from unittest.mock import MagicMock


def _mock_settings():
    s = MagicMock()
    s.feat_price_action_enabled = True
    s.feat_momentum_enabled = True
    s.feat_volatility_enabled = True
    s.feat_micro_enabled = True
    s.feat_context_enabled = True
    s.gate_min_rvol = 1.5
    s.gate_max_spread = 0.0005
    return s


# ── T93 ───────────────────────────────────────────────────────────────────────


def test_get_feature_engineer_returns_engineer(monkeypatch):
    import src.ml_engine.features as feat_mod
    from src.ml_engine.features import FeatureEngineer

    monkeypatch.setattr(feat_mod, "_engineer", None)
    monkeypatch.setattr(feat_mod, "get_settings", _mock_settings)
    result = feat_mod.get_feature_engineer()
    assert isinstance(result, FeatureEngineer)


# ── T94 ───────────────────────────────────────────────────────────────────────


def test_get_feature_engineer_returns_same_instance(monkeypatch):
    import src.ml_engine.features as feat_mod

    monkeypatch.setattr(feat_mod, "_engineer", None)
    monkeypatch.setattr(feat_mod, "get_settings", _mock_settings)
    a = feat_mod.get_feature_engineer()
    b = feat_mod.get_feature_engineer()
    assert a is b


# ── T95 ───────────────────────────────────────────────────────────────────────


def test_get_feature_engineer_after_reset_creates_new(monkeypatch):
    import src.ml_engine.features as feat_mod

    monkeypatch.setattr(feat_mod, "get_settings", _mock_settings)

    # First call — creates instance
    monkeypatch.setattr(feat_mod, "_engineer", None)
    first = feat_mod.get_feature_engineer()

    # Simulate reset (e.g. after config reload)
    monkeypatch.setattr(feat_mod, "_engineer", None)
    second = feat_mod.get_feature_engineer()

    assert second is not first
