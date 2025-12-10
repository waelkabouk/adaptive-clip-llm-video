#!/usr/bin/env python
"""Run benchmark evaluation on MSR-VTT or other datasets."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from src.pipeline import VideoPipeline
from src.evaluation import (
    BenchmarkRunner,
    MSRVTTDataset,
    load_msrvtt_qa,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run video understanding benchmark")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory (annotations)",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Path to video files (defaults to data-dir/videos)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="qa",
        choices=["qa", "caption", "retrieval", "temporal"],
        help="Task type to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Limit number of samples (for development)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="default",
        help="Name for this configuration (for result files)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = OmegaConf.load(args.config)
        config = OmegaConf.to_container(config, resolve=True)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = VideoPipeline(config)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(pipeline, output_dir=args.output_dir)
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    
    if args.task == "qa":
        samples = load_msrvtt_qa(
            data_dir=args.data_dir,
            video_dir=args.video_dir,
            split=args.split,
            subset_size=args.subset_size,
        )
        
        if not samples:
            logger.error("No samples loaded. Check dataset path and format.")
            sys.exit(1)
            
        logger.info(f"Loaded {len(samples)} QA samples")
        
        # Run benchmark
        result = runner.run_qa_benchmark(
            samples=samples,
            config_name=args.config_name,
        )
        
    elif args.task == "caption":
        dataset = MSRVTTDataset(
            data_dir=args.data_dir,
            video_dir=args.video_dir,
            split=args.split,
        )
        samples = dataset.load_captions(subset_size=args.subset_size)
        
        if not samples:
            logger.error("No samples loaded. Check dataset path and format.")
            sys.exit(1)
            
        logger.info(f"Loaded {len(samples)} caption samples")
        
        result = runner.run_caption_benchmark(
            samples=samples,
            config_name=args.config_name,
        )
        
    else:
        logger.error(f"Task {args.task} not yet implemented for benchmark")
        sys.exit(1)
    
    # Save results
    output_path = Path(args.output_dir) / f"benchmark_{args.task}_{args.config_name}.json"
    result.save(str(output_path))
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Config: {args.config_name}")
    print(f"Samples: {result.metrics.total_samples}")
    print(f"Successful: {result.metrics.successful_samples}")
    print(f"Failed: {result.metrics.failed_samples}")
    print("-"*60)
    
    metrics = result.metrics.to_dict()
    
    if args.task == "qa":
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print(f"Exact Match: {metrics['exact_match']:.2%}")
    elif args.task == "caption":
        print(f"Accuracy (fuzzy): {metrics['accuracy']:.2%}")
        
    print(f"Avg Latency: {metrics['avg_latency_ms']:.0f}ms")
    print(f"Avg Cost: ${metrics['avg_cost_usd']:.4f}")
    print(f"Avg Frame Reduction: {metrics['avg_frame_reduction']:.1%}")
    print(f"Avg Frames to LLM: {metrics['avg_frames_to_llm']:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()

