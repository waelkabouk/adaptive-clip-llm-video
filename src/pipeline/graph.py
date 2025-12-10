"""LangGraph pipeline definition."""

import logging
import time
from typing import Any, Dict, Optional

from pathlib import Path
from omegaconf import OmegaConf

from langgraph.graph import StateGraph, END

from .state import PipelineState, TaskType
from .nodes import (
    extract_frames_node,
    encode_frames_node,
    deduplicate_node,
    task_router_node,
    context_builder_node,
    inference_node,
)

logger = logging.getLogger(__name__)


def create_pipeline() -> StateGraph:
    """
    Create the LangGraph video processing pipeline.
    
    Pipeline flow:
    extract_frames -> encode_frames -> deduplicate -> task_router -> 
    context_builder -> inference -> END
    
    Returns:
        Compiled StateGraph
    """
    # Create graph with state schema
    workflow = StateGraph(PipelineState)
    
    # Add nodes
    workflow.add_node("extract_frames", extract_frames_node)
    workflow.add_node("encode_frames", encode_frames_node)
    workflow.add_node("deduplicate", deduplicate_node)
    workflow.add_node("task_router", task_router_node)
    workflow.add_node("context_builder", context_builder_node)
    workflow.add_node("inference", inference_node)
    
    # Define edges (linear flow)
    workflow.set_entry_point("extract_frames")
    workflow.add_edge("extract_frames", "encode_frames")
    workflow.add_edge("encode_frames", "deduplicate")
    workflow.add_edge("deduplicate", "task_router")
    workflow.add_edge("task_router", "context_builder")
    workflow.add_edge("context_builder", "inference")
    workflow.add_edge("inference", END)
    
    return workflow.compile()


class VideoPipeline:
    """
    High-level wrapper for the video understanding pipeline.
    
    Provides a simple interface for processing videos with
    configuration management and metrics tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config or {}
        self.graph = create_pipeline()
        
    def process(
        self,
        video_path: str,
        query: str,
        task_type: str = "qa",
    ) -> Dict[str, Any]:
        """
        Process a video with a query.
        
        Args:
            video_path: Path to video file
            query: User query or instruction
            task_type: Task type (qa, caption, retrieval, temporal)
            
        Returns:
            Dictionary with response and metrics
        """
        start_time = time.time()
        
        # Validate task type
        if task_type not in [t.value for t in TaskType]:
            raise ValueError(f"Invalid task type: {task_type}")
        
        # Task-specific overrides (optional)
        task_config = self._load_task_config(task_type)

        # Build initial state
        initial_state: PipelineState = {
            "video_path": video_path,
            "query": query,
            "task_type": task_type,
            "config": {**self.config, "task_config": task_config},
            "metrics": {},
        }
        
        # Run pipeline
        logger.info(f"Processing video: {video_path} (task={task_type})")
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Calculate total time
            total_time_ms = (time.time() - start_time) * 1000
            final_state["metrics"]["total_time_ms"] = total_time_ms
            
            logger.info(f"Pipeline completed in {total_time_ms:.0f}ms")
            
            return {
                "response": final_state.get("response", ""),
                "metrics": final_state.get("metrics", {}),
                "error": final_state.get("error"),
                "key_frame_timestamps": final_state.get("key_frame_timestamps", []),
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                "response": "",
                "metrics": {"total_time_ms": (time.time() - start_time) * 1000},
                "error": str(e),
            }
            
    def process_qa(self, video_path: str, question: str) -> Dict[str, Any]:
        """Process video for question answering."""
        return self.process(video_path, question, TaskType.QA.value)
        
    def process_caption(self, video_path: str) -> Dict[str, Any]:
        """Generate video caption."""
        return self.process(video_path, "", TaskType.CAPTION.value)
        
    def process_retrieval(self, video_path: str, query: str) -> Dict[str, Any]:
        """Check if video matches query."""
        return self.process(video_path, query, TaskType.RETRIEVAL.value)
        
    def process_temporal(self, video_path: str, query: str) -> Dict[str, Any]:
        """Find temporal location of event in video."""
        return self.process(video_path, query, TaskType.TEMPORAL.value)
        
    @classmethod
    def from_config_file(cls, config_path: str) -> "VideoPipeline":
        """Create pipeline from Hydra config file."""
        config = OmegaConf.load(config_path)
        return cls(OmegaConf.to_container(config, resolve=True))

    def _load_task_config(self, task_type: str) -> Dict[str, Any]:
        """
        Load task-specific overrides from configs/task/{task}.yaml if present.
        """
        base_dir = self.config.get("task_config_dir", "configs/task")
        task_path = Path(base_dir) / f"{task_type}.yaml"
        if not task_path.exists():
            return {}
        try:
            cfg = OmegaConf.load(task_path)
            return OmegaConf.to_container(cfg, resolve=True)
        except Exception as exc:  # pragma: no cover - file/parse issues
            logger.warning("Failed to load task config %s: %s", task_path, exc)
            return {}

