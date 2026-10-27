"""Five paper metrics and Validation macro-MCC threshold selection."""
from __future__ import annotations
import math
import numpy as np

def _inputs(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or labels.shape != scores.shape:
        raise ValueError("labels and scores must be non-empty arrays with the same shape")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    positives = int(labels.sum())
    if positives == 0 or positives == labels.size:
        raise ValueError("ranking metrics require both positive and negative labels")
    return labels, scores


def auroc(labels, scores) -> float:
    labels, scores = _inputs(labels, scores)
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    positives = int(labels.sum())
    negatives = len(labels) - positives
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def binary_metrics(labels, scores, threshold: float) -> dict[str, float]:
    labels, scores = _inputs(labels, scores)
    predicted = scores >= threshold
    tp = int(np.sum(labels & predicted))
    tn = int(np.sum(~labels & ~predicted))
    fp = int(np.sum(~labels & predicted))
    fn = int(np.sum(labels & ~predicted))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = tp / (tp + fn)
    return {
        "mcc": 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator,
        "auroc": auroc(labels, scores),
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall),
    }


def macro_mean(per_complex: list[dict[str, float]]) -> dict[str, float]:
    if not per_complex:
        raise ValueError("per_complex cannot be empty")
    keys = set(per_complex[0])
    if any(set(row) != keys for row in per_complex):
        raise ValueError("all per-complex metric dictionaries must share the same keys")
    return {key: float(np.mean([row[key] for row in per_complex])) for key in sorted(keys)}


def select_global_mcc_threshold(examples, grid=None) -> tuple[float, float]:
    """Select one Validation threshold by macro MCC; prefer the higher threshold on ties."""
    grid = np.linspace(0.0, 1.0, 1001) if grid is None else np.asarray(grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("grid must be a non-empty finite one-dimensional array")
    if np.any(np.diff(grid) <= 0) or grid[0] < 0 or grid[-1] > 1:
        raise ValueError("grid must be strictly increasing within [0, 1]")
    totals = np.zeros(len(grid), dtype=np.float64)
    count = 0
    for labels, scores in examples:
        labels, scores = _inputs(labels, scores)
        order = np.argsort(scores, kind="stable")
        sorted_scores = scores[order]
        cumulative_positive = np.cumsum(labels[order], dtype=np.int64)
        index = np.searchsorted(sorted_scores, grid, side="left")
        positive_below = np.where(index, cumulative_positive[np.maximum(index - 1, 0)], 0)
        total_positive = int(labels.sum())
        tp = total_positive - positive_below
        fn = positive_below
        fp = (len(labels) - index) - tp
        tn = index - fn
        denominator = np.sqrt((tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn))
        totals += np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(denominator), where=denominator != 0)
        count += 1
    if count == 0:
        raise ValueError("examples cannot be empty")
    macro = totals / count
    best_score = float(macro.max())
    best_index = int(np.flatnonzero(macro == best_score)[-1])
    return float(grid[best_index]), best_score
