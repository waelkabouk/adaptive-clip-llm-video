"""Chainlit web UI for video understanding pipeline."""

# Import torch first to ensure DLL loads correctly on Windows
import torch  # noqa: F401

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

import chainlit as cl
from omegaconf import OmegaConf

from src.pipeline import VideoPipeline, TaskType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[VideoPipeline] = None

# Task descriptions for UI
TASK_DESCRIPTIONS = {
    TaskType.QA.value: "Ask questions about the video content",
    TaskType.CAPTION.value: "Generate a description of the video",
    TaskType.RETRIEVAL.value: "Check if video matches a text query",
    TaskType.TEMPORAL.value: "Find when events occur in the video",
}


def load_config() -> dict:
    """Load configuration from file or defaults."""
    config_path = os.environ.get("CONFIG_PATH", "configs/config.yaml")
    
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(config, resolve=True)
    
    # Default configuration
    return {
        "sampling": {
            "strategy": "uniform",
            "max_frames": 32,
            "uniform_count": 16,
        },
        "encoder": {
            "model_name": "ViT-B-32",
            "pretrained": "openai",
            "device": "auto",
            "batch_size": 8,
            "auto_batch_scale": True,
        },
        "dedup": {
            "enabled": True,
            "method": "greedy_cosine",
            "threshold": 0.85,
            "min_frames": 4,
            "max_frames": 12,
            "preserve_temporal_order": True,
        },
        "inference": {
            "provider": "google",
            "model": "gemini-2.0-flash",
            "max_tokens": 1024,
            "temperature": 0.7,
            "max_images": 8,
        },
    }


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    global pipeline
    
    # Load config
    config = load_config()
    
    # Check for API keys
    provider = config.get("inference", {}).get("provider", "openai")
    api_key_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider)
    
    if api_key_env and not os.environ.get(api_key_env):
        await cl.Message(
            content=f"⚠️ **Warning**: `{api_key_env}` environment variable not set. "
                   f"Please set it or configure API key in settings."
        ).send()
    
    # Initialize pipeline
    try:
        pipeline = VideoPipeline(config)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        await cl.Message(
            content=f"❌ Failed to initialize pipeline: {e}"
        ).send()
        return
    
    # Store config in session
    cl.user_session.set("config", config)
    cl.user_session.set("current_video", None)
    cl.user_session.set("task_type", TaskType.QA.value)
    
    # Welcome message
    welcome_msg = """
# 🎬 Video Understanding Pipeline

Upload a video and ask questions about it!

**Supported Tasks:**
- **QA**: Ask questions about video content
- **Caption**: Generate video descriptions
- **Retrieval**: Check if video matches a query
- **Temporal**: Find when events occur

**Getting Started:**
1. Upload a video file (MP4, AVI, WebM)
2. Select a task type
3. Enter your question or query

*Note: Large videos will be sampled and deduplicated for efficient processing.*
"""
    
    await cl.Message(content=welcome_msg).send()
    
    # Task selector
    actions = [
        cl.Action(name="task_qa", payload={"task": "qa"}, label="QA"),
        cl.Action(name="task_caption", payload={"task": "caption"}, label="Caption"),
        cl.Action(name="task_retrieval", payload={"task": "retrieval"}, label="Retrieval"),
        cl.Action(name="task_temporal", payload={"task": "temporal"}, label="Temporal"),
    ]
    
    await cl.Message(
        content="**Select a task:**",
        actions=actions,
    ).send()


@cl.action_callback("task_qa")
@cl.action_callback("task_caption")
@cl.action_callback("task_retrieval")
@cl.action_callback("task_temporal")
async def on_task_select(action: cl.Action):
    """Handle task selection."""
    task = action.payload.get("task", "qa")
    cl.user_session.set("task_type", task)
    
    description = TASK_DESCRIPTIONS.get(task, "")
    await cl.Message(
        content=f"✅ Task set to **{task.upper()}**: {description}"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages."""
    global pipeline
    
    if pipeline is None:
        await cl.Message(content="❌ Pipeline not initialized").send()
        return
    
    # Check for file upload
    video_path = cl.user_session.get("current_video")
    
    if message.elements:
        # Handle file upload
        for element in message.elements:
            if hasattr(element, 'path') and element.path:
                # Check file extension
                ext = Path(element.path).suffix.lower()
                if ext in ['.mp4', '.avi', '.webm', '.mkv', '.mov']:
                    video_path = element.path
                    cl.user_session.set("current_video", video_path)
                    await cl.Message(
                        content=f"📹 Video uploaded: `{Path(element.path).name}`"
                    ).send()
                else:
                    await cl.Message(
                        content=f"⚠️ Unsupported file type: {ext}. Please upload MP4, AVI, WebM, or MKV."
                    ).send()
                    return
    
    # Get query text
    query = message.content.strip()
    
    if not video_path:
        await cl.Message(
            content="📁 Please upload a video file first."
        ).send()
        return
    
    # Get task type
    task_type = cl.user_session.get("task_type", TaskType.QA.value)
    
    # For caption task, no query needed
    if task_type == TaskType.CAPTION.value and not query:
        query = "Generate caption"
    elif not query and task_type != TaskType.CAPTION.value:
        await cl.Message(
            content="💬 Please enter a question or query."
        ).send()
        return
    
    # Process video
    processing_msg = cl.Message(content="⏳ Processing video...")
    await processing_msg.send()
    
    try:
        # Run pipeline
        result = pipeline.process(video_path, query, task_type)
        
        # Format response
        response = result.get("response", "No response generated")
        metrics = result.get("metrics", {})
        error = result.get("error")
        
        if error:
            await cl.Message(
                content=f"❌ Error: {error}"
            ).send()
            return
        
        # Build response message
        response_content = f"**Answer:**\n\n{response}\n\n"
        
        # Add metrics
        response_content += "---\n**📊 Metrics:**\n"
        response_content += f"- Total time: {metrics.get('total_time_ms', 0):.0f}ms\n"
        response_content += f"- Original frames: {metrics.get('original_frame_count', 0)}\n"
        response_content += f"- After sampling: {metrics.get('sampled_frame_count', 0)}\n"
        response_content += f"- After dedup: {metrics.get('deduplicated_frame_count', 0)}\n"
        response_content += f"- Sent to LLM: {metrics.get('frames_sent_to_llm', 0)}\n"
        response_content += f"- Frame reduction: {metrics.get('frame_reduction_ratio', 0):.1%}\n"
        
        if metrics.get('estimated_cost_usd', 0) > 0:
            response_content += f"- Estimated cost: ${metrics.get('estimated_cost_usd', 0):.4f}\n"
        
        # Show key frame timestamps
        timestamps = result.get("key_frame_timestamps", [])
        if timestamps:
            timestamp_str = ", ".join([f"{t:.1f}s" for t in timestamps[:8]])
            if len(timestamps) > 8:
                timestamp_str += f"... ({len(timestamps)} total)"
            response_content += f"- Key timestamps: {timestamp_str}\n"
        
        await cl.Message(content=response_content).send()
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        await cl.Message(
            content=f"❌ Processing error: {str(e)}"
        ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """Handle settings updates."""
    global pipeline
    
    config = cl.user_session.get("config", {})
    
    # Update config from settings
    if "provider" in settings:
        config.setdefault("inference", {})["provider"] = settings["provider"]
    if "model" in settings:
        config.setdefault("inference", {})["model"] = settings["model"]
    if "dedup_threshold" in settings:
        config.setdefault("dedup", {})["threshold"] = settings["dedup_threshold"]
    if "max_frames" in settings:
        config.setdefault("dedup", {})["max_frames"] = settings["max_frames"]
    if "openai_api_key" in settings and settings["openai_api_key"]:
        os.environ["OPENAI_API_KEY"] = settings["openai_api_key"]
    if "anthropic_api_key" in settings and settings["anthropic_api_key"]:
        os.environ["ANTHROPIC_API_KEY"] = settings["anthropic_api_key"]
    if "google_api_key" in settings and settings["google_api_key"]:
        os.environ["GOOGLE_API_KEY"] = settings["google_api_key"]
    
    # Reinitialize pipeline
    try:
        pipeline = VideoPipeline(config)
        cl.user_session.set("config", config)
        await cl.Message(content="✅ Settings updated").send()
    except Exception as e:
        await cl.Message(content=f"❌ Failed to update settings: {e}").send()


if __name__ == "__main__":
    # For local testing
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)

