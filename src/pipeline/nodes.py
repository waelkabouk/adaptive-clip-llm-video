"""LangGraph pipeline nodes."""

import base64
import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..video import VideoLoader, create_sampler
from ..encoders import CLIPEncoder
from ..dedup import create_deduplicator
from .state import PipelineState, TaskType

logger = logging.getLogger(__name__)


def extract_frames_node(state: PipelineState) -> PipelineState:
    """
    Extract frames from video.
    
    Node that loads video, samples frames according to strategy,
    and updates state with frames and timestamps.
    """
    start_time = time.time()
    
    config = state.get("config", {})
    task_config = config.get("task_config", {})
    video_path = state["video_path"]
    
    # Get sampling config
    sampling_config = config.get("sampling", {})
    strategy = sampling_config.get("strategy", "uniform")
    max_frames = sampling_config.get("max_frames", 64)
    
    # Create sampler
    sampler_kwargs = {}
    if strategy == "uniform":
        sampler_kwargs["num_frames"] = sampling_config.get("uniform_count", 32)
    elif strategy == "fps_cap":
        sampler_kwargs["target_fps"] = sampling_config.get("fps_cap", 2.0)
        sampler_kwargs["max_frames"] = max_frames
        
    sampler = create_sampler(strategy, **sampler_kwargs)
    
    # Load video and sample
    loader = VideoLoader(
        cache_dir=config.get("video", {}).get("cache_dir")
    )
    
    try:
        metadata = loader.load(video_path)
        sampling_result = sampler.sample(metadata, max_frames=max_frames)
        
        # Extract actual frames
        frames, timestamps = loader.get_frames(sampling_result.frame_indices)
        
        # Update metrics
        metrics = state.get("metrics", {})
        metrics["original_frame_count"] = metadata.total_frames
        metrics["sampled_frame_count"] = len(frames)
        metrics["extraction_time_ms"] = (time.time() - start_time) * 1000
        
        return {
            **state,
            "video_duration": metadata.duration,
            "video_fps": metadata.fps,
            "total_frames": metadata.total_frames,
            "frame_indices": sampling_result.frame_indices,
            "timestamps": timestamps,
            "frames": frames,
            "metrics": metrics,
        }
        
    finally:
        loader.close()


def encode_frames_node(state: PipelineState) -> PipelineState:
    """
    Encode frames using CLIP.
    
    Node that encodes sampled frames into embeddings.
    """
    start_time = time.time()
    
    config = state.get("config", {})
    frames = state["frames"]
    
    # Get encoder config
    encoder_config = config.get("encoder", {})
    
    # Create encoder
    encoder = CLIPEncoder(
        model_name=encoder_config.get("model_name", "ViT-B-32"),
        pretrained=encoder_config.get("pretrained", "openai"),
        device=encoder_config.get("device", "auto"),
        batch_size=encoder_config.get("batch_size", 16),
        auto_batch_scale=encoder_config.get("auto_batch_scale", True),
    )
    
    # Encode frames
    output = encoder.encode_images(frames)
    
    # Update metrics
    metrics = state.get("metrics", {})
    metrics["encoding_time_ms"] = (time.time() - start_time) * 1000
    metrics["encoder_model"] = encoder.model_name
    
    return {
        **state,
        "embeddings": output.embeddings,
        "embedding_dim": output.embedding_dim,
        "metrics": metrics,
    }


def deduplicate_node(state: PipelineState) -> PipelineState:
    """
    Deduplicate frames using semantic similarity.
    
    Node that reduces redundant frames based on CLIP embeddings.
    """
    start_time = time.time()
    
    config = state.get("config", {})
    task_config = config.get("task_config", {})
    embeddings = state["embeddings"]
    timestamps = state["timestamps"]
    frames = state["frames"]
    
    # Get dedup config with task override
    dedup_config = {**config.get("dedup", {}), **task_config.get("dedup", {})}
    
    if not dedup_config.get("enabled", True):
        # Skip deduplication
        metrics = state.get("metrics", {})
        metrics["dedup_time_ms"] = 0
        metrics["deduplicated_frame_count"] = len(frames)
        metrics["frame_reduction_ratio"] = 0.0
        
        return {
            **state,
            "key_frame_indices": list(range(len(frames))),
            "key_frame_timestamps": timestamps,
            "key_frames": frames,
            "metrics": metrics,
        }
    
    # Create deduplicator
    method = dedup_config.get("method", "greedy_cosine")
    deduplicator = create_deduplicator(
        method=method,
        threshold=dedup_config.get("threshold", 0.85),
        min_frames=dedup_config.get("min_frames", 4),
        max_frames=dedup_config.get("max_frames", 16),
        preserve_temporal_order=dedup_config.get("preserve_temporal_order", True),
    )
    
    # Deduplicate
    result = deduplicator.deduplicate(
        embeddings,
        timestamps=timestamps,
        max_frames=dedup_config.get("max_frames"),
    )
    
    # Select key frames
    key_frame_indices = result.selected_indices
    key_frame_timestamps = [timestamps[i] for i in key_frame_indices]
    key_frames = [frames[i] for i in key_frame_indices]
    
    # Update metrics
    metrics = state.get("metrics", {})
    metrics["dedup_time_ms"] = (time.time() - start_time) * 1000
    metrics["deduplicated_frame_count"] = len(key_frames)
    metrics["frame_reduction_ratio"] = result.reduction_ratio
    
    logger.info(
        f"Deduplication: {result.original_count} -> {result.deduplicated_count} frames "
        f"({result.reduction_ratio:.1%} reduction)"
    )
    
    return {
        **state,
        "key_frame_indices": key_frame_indices,
        "key_frame_timestamps": key_frame_timestamps,
        "key_frames": key_frames,
        "metrics": metrics,
    }


def task_router_node(state: PipelineState) -> PipelineState:
    """
    Route to appropriate task handler.
    
    This node prepares task-specific configuration.
    """
    task_type = state.get("task_type", TaskType.QA.value)
    
    # Task-specific adjustments could be made here
    # For now, just pass through with task type validated
    if task_type not in [t.value for t in TaskType]:
        logger.warning(f"Unknown task type: {task_type}, defaulting to QA")
        task_type = TaskType.QA.value
        
    return {
        **state,
        "task_type": task_type,
    }


def context_builder_node(state: PipelineState) -> PipelineState:
    """
    Build context for LLM based on task type.
    
    Formats frames, timestamps, and query into appropriate prompt.
    """
    task_type = state.get("task_type", TaskType.QA.value)
    query = state.get("query", "")
    key_frames = state["key_frames"]
    key_frame_timestamps = state["key_frame_timestamps"]
    config = state.get("config", {})
    task_config = config.get("task_config", {})
    
    # Get max images for LLM (task override respected)
    inference_config = config.get("inference", {})
    task_inference = task_config.get("inference", {})
    max_images = task_inference.get(
        "max_images", inference_config.get("max_images", 8)
    )
    
    # Limit frames if needed
    if len(key_frames) > max_images:
        # Uniformly sample from key frames
        indices = np.linspace(0, len(key_frames) - 1, max_images, dtype=int)
        selected_frames = [key_frames[i] for i in indices]
        selected_timestamps = [key_frame_timestamps[i] for i in indices]
    else:
        selected_frames = key_frames
        selected_timestamps = key_frame_timestamps
        
    # Convert frames to base64 for API
    frame_data = []
    for frame in selected_frames:
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame)
        else:
            img = frame
        
        # Resize if too large (for API efficiency)
        max_dim = 768
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64_data = base64.b64encode(buffer.getvalue()).decode()
        frame_data.append(b64_data)
        
    # Format timestamps
    timestamp_str = ", ".join([f"{t:.1f}s" for t in selected_timestamps])
    
    # Build prompt based on task (with task-specific overrides)
    prompt, system_prompt = _build_prompt(
        task_type,
        query,
        len(selected_frames),
        timestamp_str,
        task_config,
    )
    
    # Build context object
    context = {
        "task_type": task_type,
        "query": query,
        "num_frames": len(selected_frames),
        "timestamps": selected_timestamps,
        "timestamp_str": timestamp_str,
        "frame_data": frame_data,  # Base64 encoded
    }
    
    # Update metrics
    metrics = state.get("metrics", {})
    metrics["frames_sent_to_llm"] = len(selected_frames)
    
    return {
        **state,
        "context": context,
        "prompt": prompt,
        "metrics": metrics,
        "system_prompt": system_prompt,
    }


def _build_prompt(
    task_type: str,
    query: str,
    num_frames: int,
    timestamp_str: str,
    task_config: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build user and system prompts based on task type and overrides.
    """
    prompt_cfg = task_config.get("prompt", {})
    system_prompt = prompt_cfg.get("system", _get_system_prompt(task_type))
    user_template = prompt_cfg.get("user_template")

    if user_template:
        user_prompt = user_template.format(
            num_frames=num_frames,
            timestamps=timestamp_str,
            question=query,
            query=query,
        )
        return user_prompt, system_prompt

    # Fallback defaults
    defaults = {
        TaskType.QA.value: f"""Analyze these {num_frames} key frames from a video (timestamps: {timestamp_str}).

Question: {query}

Please provide a clear, accurate answer based on what you observe in the frames.""",
        TaskType.CAPTION.value: f"""Here are {num_frames} key frames from a video in temporal order (timestamps: {timestamp_str}).

Please generate a comprehensive caption describing what happens in this video.
Focus on the main subjects, actions, and any notable events.""",
        TaskType.RETRIEVAL.value: f"""Here are {num_frames} frames from a video (timestamps: {timestamp_str}).

Query: {query}

Does this video match the query? Explain why or why not.""",
        TaskType.TEMPORAL.value: f"""Here are {num_frames} frames from a video with their timestamps: {timestamp_str}

Query: {query}

Please identify the time range(s) when the described event/action occurs.
Provide your answer in the format: [start_time - end_time]""",
    }
    return defaults.get(task_type, defaults[TaskType.QA.value]), system_prompt


def inference_node(state: PipelineState) -> PipelineState:
    """
    Run LLM inference.
    
    This node calls the LLM API with the prepared context.
    The actual implementation is delegated to the LLM module.
    """
    start_time = time.time()
    
    # Import here to avoid circular imports
    from ..llm import create_llm_client
    
    config = state.get("config", {})
    context = state["context"]
    prompt = state["prompt"]
    
    inference_config = config.get("inference", {})
    
    providers_to_try = [
        (
            inference_config.get("provider", "openai"),
            inference_config.get("model", "gpt-4o"),
        )
    ]
    if inference_config.get("fallback_provider"):
        providers_to_try.append(
            (
                inference_config.get("fallback_provider"),
                inference_config.get("fallback_model", None),
            )
        )

    last_error = None
    for provider_name, model_name in providers_to_try:
        try:
            client = create_llm_client(
                provider=provider_name,
                model=model_name,
                max_tokens=inference_config.get("max_tokens", 1024),
                temperature=inference_config.get("temperature", 0.7),
                api_key=inference_config.get("api_key"),
            )

            result = client.generate_with_images(
                prompt=prompt,
                images=context["frame_data"],
                system_prompt=state.get("system_prompt") or _get_system_prompt(context["task_type"]),
            )

            metrics = state.get("metrics", {})
            metrics["inference_time_ms"] = (time.time() - start_time) * 1000
            metrics["input_tokens"] = result.get("input_tokens", 0)
            metrics["output_tokens"] = result.get("output_tokens", 0)
            metrics["estimated_cost_usd"] = result.get("cost", 0.0)
            metrics["llm_model"] = model_name or inference_config.get("model", "gpt-4o")
            metrics["llm_provider"] = provider_name

            return {
                **state,
                "response": result["text"],
                "metrics": metrics,
            }

        except Exception as e:  # pragma: no cover - network/API failure
            last_error = str(e)
            logger.warning("Inference failed for %s/%s: %s", provider_name, model_name, e)
            continue

    return {
        **state,
        "response": f"Error: {last_error or 'inference failed'}",
        "error": last_error,
    }


def _get_system_prompt(task_type: str) -> str:
    """Get system prompt for task type."""
    system_prompts = {
        TaskType.QA.value: "You are a video understanding assistant. Analyze the provided video frames and answer questions accurately based on what you observe. Be concise and specific.",
        
        TaskType.CAPTION.value: "You are a video captioning assistant. Generate clear, descriptive captions that summarize the main content and actions in videos.",
        
        TaskType.RETRIEVAL.value: "You are a video retrieval assistant. Determine whether videos match given text queries based on visual content.",
        
        TaskType.TEMPORAL.value: "You are a temporal grounding assistant. Identify when specific events or actions occur in videos based on frame timestamps.",
    }
    
    return system_prompts.get(task_type, system_prompts[TaskType.QA.value])

