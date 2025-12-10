#!/usr/bin/env python
"""Run ablation studies comparing different configurations."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from src.pipeline import VideoPipeline
from src.evaluation import (
    BenchmarkRunner,
    load_msrvtt_qa,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_ablation_configs(base_config: dict) -> tuple:
    """Create configurations for ablation study."""
    configs = []
    names = []
    
    # 1. Baseline: No deduplication
    no_dedup = base_config.copy()
    no_dedup["dedup"] = {"enabled": False}
    configs.append(no_dedup)
    names.append("no_dedup")
    
    # 2. Low threshold (more aggressive dedup)
    low_thresh = base_config.copy()
    low_thresh["dedup"] = base_config.get("dedup", {}).copy()
    low_thresh["dedup"]["threshold"] = 0.70
    configs.append(low_thresh)
    names.append("thresh_0.70")
    
    # 3. Default threshold
    default_thresh = base_config.copy()
    default_thresh["dedup"] = base_config.get("dedup", {}).copy()
    default_thresh["dedup"]["threshold"] = 0.85
    configs.append(default_thresh)
    names.append("thresh_0.85")
    
    # 4. High threshold (less aggressive dedup)
    high_thresh = base_config.copy()
    high_thresh["dedup"] = base_config.get("dedup", {}).copy()
    high_thresh["dedup"]["threshold"] = 0.95
    configs.append(high_thresh)
    names.append("thresh_0.95")
    
    # 5. More frames to LLM
    more_frames = base_config.copy()
    more_frames["inference"] = base_config.get("inference", {}).copy()
    more_frames["inference"]["max_images"] = 16
    more_frames["dedup"] = base_config.get("dedup", {}).copy()
    more_frames["dedup"]["max_frames"] = 20
    configs.append(more_frames)
    names.append("max_frames_16")
    
    # 6. Fewer frames to LLM
    fewer_frames = base_config.copy()
    fewer_frames["inference"] = base_config.get("inference", {}).copy()
    fewer_frames["inference"]["max_images"] = 4
    fewer_frames["dedup"] = base_config.get("dedup", {}).copy()
    fewer_frames["dedup"]["max_frames"] = 6
    configs.append(fewer_frames)
    names.append("max_frames_4")
    
    return configs, names


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Base configuration file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Path to video files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation",
        help="Output directory",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Number of samples for ablation (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Load base config
    if os.path.exists(args.config):
        base_config = OmegaConf.load(args.config)
        base_config = OmegaConf.to_container(base_config, resolve=True)
    else:
        base_config = {}
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_dir}")
    samples = load_msrvtt_qa(
        data_dir=args.data_dir,
        video_dir=args.video_dir,
        subset_size=args.subset_size,
    )
    
    if not samples:
        logger.error("No samples loaded")
        sys.exit(1)
        
    logger.info(f"Loaded {len(samples)} samples for ablation")
    
    # Create ablation configs
    configs, names = create_ablation_configs(base_config)
    logger.info(f"Running ablation with {len(configs)} configurations: {names}")
    
    # Run ablation
    pipeline = VideoPipeline(base_config)
    runner = BenchmarkRunner(pipeline, output_dir=args.output_dir)
    
    results = runner.run_ablation(
        samples=samples,
        configs=configs,
        config_names=names,
        task_type="qa",
    )
    
    # Compare results
    comparison = runner.compare_results(results)
    
    # Save comparison
    output_path = Path(args.output_dir) / "ablation_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    print(f"{'Config':<20} {'Accuracy':>10} {'Latency':>12} {'Cost':>10} {'Reduction':>12}")
    print("-"*80)
    
    for i, name in enumerate(comparison["configs"]):
        acc = comparison["metrics"]["accuracy"][i]
        lat = comparison["metrics"]["avg_latency_ms"][i]
        cost = comparison["metrics"]["avg_cost_usd"][i]
        red = comparison["metrics"]["avg_frame_reduction"][i]
        
        print(f"{name:<20} {acc:>10.2%} {lat:>10.0f}ms ${cost:>9.4f} {red:>11.1%}")
    
    print("="*80)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()

