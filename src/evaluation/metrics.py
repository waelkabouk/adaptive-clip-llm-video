"""Evaluation metrics."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    """Container for evaluation metrics."""
    
    # Accuracy metrics
    accuracy: float = 0.0
    exact_match: float = 0.0
    
    # Retrieval metrics
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mean_rank: float = 0.0
    
    # Temporal metrics
    temporal_iou: float = 0.0
    
    # Efficiency metrics
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    avg_frame_reduction: float = 0.0
    avg_frames_to_llm: float = 0.0
    
    # Counts
    total_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0
    
    # Raw data for analysis
    per_sample_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "exact_match": self.exact_match,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mean_rank": self.mean_rank,
            "temporal_iou": self.temporal_iou,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_frame_reduction": self.avg_frame_reduction,
            "avg_frames_to_llm": self.avg_frames_to_llm,
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "failed_samples": self.failed_samples,
        }


def compute_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Compute accuracy metrics for QA/captioning.
    
    Args:
        predictions: Model predictions
        ground_truths: Ground truth answers
        normalize: Whether to normalize text before comparison
        
    Returns:
        Dictionary with accuracy metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
        
    if not predictions:
        return {"accuracy": 0.0, "exact_match": 0.0}
        
    def normalize_text(text: str) -> str:
        """Normalize text for comparison."""
        if not normalize:
            return text
        # Lowercase, strip whitespace, remove punctuation
        text = text.lower().strip()
        text = "".join(c for c in text if c.isalnum() or c.isspace())
        return " ".join(text.split())  # Normalize whitespace
        
    # Exact match
    exact_matches = sum(
        1 for p, g in zip(predictions, ground_truths)
        if normalize_text(p) == normalize_text(g)
    )
    
    # Fuzzy accuracy (prediction contains ground truth or vice versa)
    fuzzy_matches = sum(
        1 for p, g in zip(predictions, ground_truths)
        if normalize_text(g) in normalize_text(p) or
           normalize_text(p) in normalize_text(g)
    )
    
    n = len(predictions)
    
    return {
        "accuracy": fuzzy_matches / n,
        "exact_match": exact_matches / n,
    }


def compute_retrieval_metrics(
    rankings: List[int],  # Rank of correct item (1-indexed)
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute retrieval metrics.
    
    Args:
        rankings: List of ranks for correct items (1 = top result)
        k_values: K values for recall@K
        
    Returns:
        Dictionary with retrieval metrics
    """
    if not rankings:
        return {f"recall_at_{k}": 0.0 for k in k_values} | {"mean_rank": 0.0}
        
    rankings = np.array(rankings)
    n = len(rankings)
    
    results = {}
    
    # Recall@K
    for k in k_values:
        recall = np.sum(rankings <= k) / n
        results[f"recall_at_{k}"] = float(recall)
        
    # Mean rank
    results["mean_rank"] = float(np.mean(rankings))
    
    # Mean reciprocal rank
    results["mrr"] = float(np.mean(1.0 / rankings))
    
    return results


def compute_temporal_iou(
    pred_start: float,
    pred_end: float,
    gt_start: float,
    gt_end: float,
) -> float:
    """
    Compute temporal IoU for temporal grounding.
    
    Args:
        pred_start: Predicted start time
        pred_end: Predicted end time
        gt_start: Ground truth start time
        gt_end: Ground truth end time
        
    Returns:
        IoU score
    """
    # Calculate intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    
    if union <= 0:
        return 0.0
        
    return intersection / union


def aggregate_metrics(
    sample_results: List[Dict[str, Any]],
    task_type: str,
) -> Metrics:
    """
    Aggregate per-sample results into overall metrics.
    
    Args:
        sample_results: List of per-sample result dictionaries
        task_type: Task type for metric selection
        
    Returns:
        Aggregated Metrics object
    """
    metrics = Metrics()
    metrics.total_samples = len(sample_results)
    metrics.per_sample_results = sample_results
    
    if not sample_results:
        return metrics
        
    # Filter successful samples
    successful = [r for r in sample_results if not r.get("error")]
    failed = [r for r in sample_results if r.get("error")]
    
    metrics.successful_samples = len(successful)
    metrics.failed_samples = len(failed)
    
    if not successful:
        return metrics
        
    # Efficiency metrics (always computed)
    metrics.avg_latency_ms = np.mean([
        r.get("latency_ms", 0) for r in successful
    ])
    metrics.avg_cost_usd = np.mean([
        r.get("cost_usd", 0) for r in successful
    ])
    metrics.avg_frame_reduction = np.mean([
        r.get("frame_reduction", 0) for r in successful
    ])
    metrics.avg_frames_to_llm = np.mean([
        r.get("frames_to_llm", 0) for r in successful
    ])
    
    # Task-specific metrics
    if task_type in ["qa", "caption"]:
        predictions = [r.get("prediction", "") for r in successful]
        ground_truths = [r.get("ground_truth", "") for r in successful]
        
        acc_metrics = compute_accuracy(predictions, ground_truths)
        metrics.accuracy = acc_metrics["accuracy"]
        metrics.exact_match = acc_metrics["exact_match"]
        
    elif task_type == "retrieval":
        rankings = [r.get("rank", 1000) for r in successful]
        ret_metrics = compute_retrieval_metrics(rankings)
        
        metrics.recall_at_1 = ret_metrics.get("recall_at_1", 0)
        metrics.recall_at_5 = ret_metrics.get("recall_at_5", 0)
        metrics.recall_at_10 = ret_metrics.get("recall_at_10", 0)
        metrics.mean_rank = ret_metrics.get("mean_rank", 0)
        
    elif task_type == "temporal":
        ious = [r.get("temporal_iou", 0) for r in successful]
        metrics.temporal_iou = np.mean(ious) if ious else 0
        
    return metrics

