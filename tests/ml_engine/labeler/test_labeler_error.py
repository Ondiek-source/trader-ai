"""
test_labeler_error.py — Tests for the LabelerError exception.

Group 1: stage attribute, default stage, str representation.
"""

import pytest
from src.ml_engine.labeler import LabelerError


# ── T01 ───────────────────────────────────────────────────────────────────────


def test_labeler_error_stage_attribute():
    """stage is stored on the exception and retrievable after construction."""
    err = LabelerError("something broke", stage="compute_labels")
    assert err.stage == "compute_labels"


# ── T02 ───────────────────────────────────────────────────────────────────────


def test_labeler_error_default_stage_empty():
    """stage defaults to '' when not provided."""
    err = LabelerError("something broke")
    assert err.stage == ""


# ── T03 ───────────────────────────────────────────────────────────────────────


def test_labeler_error_str_is_message():
    """str(err) returns the message, not the repr — consistent with Exception."""
    err = LabelerError("pipeline failure", stage="compute_labels")
    assert str(err) == "pipeline failure"
