"""
test_trade_eligibility.py — Tests for the TradeEligibility dataclass.

Group 2: to_dict, repr, frozen enforcement, gate aggregation.
"""

import pytest
from dataclasses import FrozenInstanceError
from src.ml_engine.features import TradeEligibility


def _make_te(is_eligible=True, gate_results=None, gate_values=None):
    if gate_results is None:
        gate_results = {"BB_WIDTH > MA": True, "ATR > MA": True}
    if gate_values is None:
        gate_values = {"BB_WIDTH": 0.02, "ATR": 0.001}
    return TradeEligibility(
        is_eligible=is_eligible,
        gate_results=gate_results,
        gate_values=gate_values,
    )


# ── T08 ───────────────────────────────────────────────────────────────────────


def test_trade_eligibility_to_dict_keys():
    te = _make_te()
    d = te.to_dict()
    assert set(d.keys()) == {"is_eligible", "gate_results", "gate_values"}


# ── T09 ───────────────────────────────────────────────────────────────────────


def test_trade_eligibility_repr_shows_gate_count():
    te = _make_te(gate_results={"g1": True, "g2": False, "g3": True})
    r = repr(te)
    assert "2/3" in r


# ── T10 ───────────────────────────────────────────────────────────────────────


def test_trade_eligibility_frozen_raises_on_setattr():
    te = _make_te()
    with pytest.raises(FrozenInstanceError):
        te.is_eligible = False  # type: ignore[misc]


# ── T11 ───────────────────────────────────────────────────────────────────────


def test_trade_eligibility_eligible_when_all_pass():
    te = _make_te(
        is_eligible=True,
        gate_results={"g1": True, "g2": True},
    )
    assert te.is_eligible is True
    assert all(te.gate_results.values())


# ── T12 ───────────────────────────────────────────────────────────────────────


def test_trade_eligibility_ineligible_when_one_fails():
    te = _make_te(
        is_eligible=False,
        gate_results={"g1": True, "g2": False},
    )
    assert te.is_eligible is False
