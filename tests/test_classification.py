import numpy as np
import pandas as pd
import pytest

from scripts.evaluation import (
    binary_threshold_metrics,
    greenlight_label,
    revenue_tier,
    tier_metrics,
    tune_greenlight_offset,
)


def test_greenlight_label_threshold():
    rev = pd.Series([50e6, 20e6, 2.5e9])
    bud = pd.Series([20e6, 20e6, 1e9])
    assert greenlight_label(rev, bud).tolist() == [True, False, True]


def test_revenue_tier_boundaries():
    out = revenue_tier(np.array([0.0, 24_999_999, 25e6, 99_999_999, 100e6, 499_999_999, 500e6, 2e9]))
    assert out.tolist() == ["< $25M"] * 2 + ["$25-100M"] * 2 + ["$100-500M"] * 2 + [">= $500M"] * 2


def test_binary_threshold_metrics_structure_and_range():
    rev = np.array([50e6, 200e6, 30e6])
    bud = np.array([10e6, 40e6, 100e6])  # third movie is a flop vs its budget
    pred_log = np.log1p(rev)  # perfect
    m = binary_threshold_metrics(np.log1p(rev), pred_log, rev, bud)
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert k in m
        assert 0.0 <= m[k] <= 1.0
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0


def test_binary_threshold_metrics_random_scoring():
    rng = np.random.default_rng(0)
    rev = 10e6 * np.exp(rng.normal(0, 1, 500))
    bud = np.full(500, 5e6)
    pred_log = rng.normal(0, 1, 500)
    m = binary_threshold_metrics(np.log1p(rev), pred_log, rev, bud)
    assert m["roc_auc"] == pytest.approx(0.5, abs=0.06)


def test_tune_greenlight_offset_finds_better_cutoff():
    rng = np.random.default_rng(1)
    rev = 10e6 * np.exp(rng.normal(0.0, 1.0, 600))
    bud = np.full(600, 5e6)
    pred_log = np.log1p(rev) - 0.3 + rng.normal(0, 0.1, 600)  # systematically low -> needs +0.3 offset
    offset, rows = tune_greenlight_offset(np.log1p(rev), pred_log, rev, bud)
    assert abs(offset - 0.3) < 0.05
    assert rows["f1"] > 0.9


def test_tier_metrics_confusion_frame_shape():
    rev = np.array([5e6, 50e6, 300e6, 900e6])
    y_log = np.log1p(rev)
    tm = tier_metrics(y_log, y_log, rev)
    assert tm["confusion"].shape == (4, 4)
    assert tm["accuracy"] == 1.0
    assert tm["macro_f1"] == 1.0