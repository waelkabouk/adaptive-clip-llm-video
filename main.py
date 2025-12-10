#!/usr/bin/env python
"""Main entry point for CLIP-LLM Video Understanding Pipeline."""

# Import torch first to ensure DLL loads correctly on Windows
import torch  # noqa: F401

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.pipeline import VideoPipeline, TaskType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="CLIP-LLM Video Understanding Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Video QA
  python main.py --video video.mp4 --query "What is happening?" --task qa
  
  # Video captioning
  python main.py --video video.mp4 --task caption
  
  # Run web UI
  python main.py --ui
  
  # Run with custom config
  python main.py --video video.mp4 --query "..." --config configs/api_first.yaml
        """
    )
    
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Query or question about the video",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="qa",
        choices=["qa", "caption", "retrieval", "temporal"],
        help="Task type (default: qa)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Chainlit web UI",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Launch UI if requested
    if args.ui:
        logger.info("Launching Chainlit web UI...")
        import subprocess
        subprocess.run(["chainlit", "run", "chainlit_app.py"])
        return
    
    # Validate arguments
    if not args.video:
        parser.error("--video is required unless --ui is specified")
        
    if args.task != "caption" and not args.query:
        parser.error("--query is required for non-caption tasks")
    
    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    # Load config
    config = {}
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
        config = OmegaConf.to_container(config, resolve=True)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config not found: {args.config}, using defaults")
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = VideoPipeline(config)
    
    # Process video
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Task: {args.task}")
    if args.query:
        logger.info(f"Query: {args.query}")
    
    try:
        result = pipeline.process(
            video_path=str(video_path),
            query=args.query,
            task_type=args.task,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if result.get("error"):
        print(f"Error: {result['error']}")
    else:
        print(f"\nResponse:\n{result['response']}\n")
    
    # Display metrics
    metrics = result.get("metrics", {})
    print("-"*60)
    print("Metrics:")
    print(f"  Total time: {metrics.get('total_time_ms', 0):.0f}ms")
    print(f"  Original frames: {metrics.get('original_frame_count', 0)}")
    print(f"  Sampled frames: {metrics.get('sampled_frame_count', 0)}")
    print(f"  After dedup: {metrics.get('deduplicated_frame_count', 0)}")
    print(f"  Sent to LLM: {metrics.get('frames_sent_to_llm', 0)}")
    print(f"  Frame reduction: {metrics.get('frame_reduction_ratio', 0):.1%}")
    
    if metrics.get("estimated_cost_usd", 0) > 0:
        print(f"  Estimated cost: ${metrics.get('estimated_cost_usd', 0):.4f}")
    
    print("="*60)
    
    # Save results if requested
    if args.output:
        import json
        output_data = {
            "video": str(video_path),
            "query": args.query,
            "task": args.task,
            "response": result.get("response", ""),
            "metrics": metrics,
            "error": result.get("error"),
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

