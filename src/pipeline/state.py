"""Pipeline state definitions."""

from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Supported task types."""
    QA = "qa"
    CAPTION = "caption"
    RETRIEVAL = "retrieval"
    TEMPORAL = "temporal"


class PipelineMetrics(BaseModel):
    """Pipeline execution metrics."""
    
    # Timing
    total_time_ms: float = 0.0
    extraction_time_ms: float = 0.0
    encoding_time_ms: float = 0.0
    dedup_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    
    # Frame statistics
    original_frame_count: int = 0
    sampled_frame_count: int = 0
    deduplicated_frame_count: int = 0
    frames_sent_to_llm: int = 0
    frame_reduction_ratio: float = 0.0
    
    # Cost (for API inference)
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    
    # Model info
    encoder_model: str = ""
    llm_model: str = ""
    llm_provider: str = ""


class PipelineState(TypedDict, total=False):
    """
    State object passed through the LangGraph pipeline.
    
    Using TypedDict for LangGraph compatibility.
    """
    
    # Input
    video_path: str
    query: str
    task_type: str  # TaskType value
    
    # Video metadata
    video_duration: float
    video_fps: float
    total_frames: int
    
    # Frame extraction
    frame_indices: List[int]
    timestamps: List[float]
    frames: List[Any]  # numpy arrays
    
    # Embeddings
    embeddings: Any  # numpy array
    embedding_dim: int
    
    # Deduplication
    key_frame_indices: List[int]
    key_frame_timestamps: List[float]
    key_frames: List[Any]  # numpy arrays
    
    # Context for LLM
    context: Dict[str, Any]
    prompt: str
    
    # Response
    response: str
    
    # Metrics
    metrics: Dict[str, Any]
    
    # Error handling
    error: Optional[str]
    
    # Configuration (passed through)
    config: Dict[str, Any]

