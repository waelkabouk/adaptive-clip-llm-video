"""Benchmark runner for evaluation."""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from ..pipeline import VideoPipeline, TaskType
from .dataset import (
    QASample,
    CaptionSample,
    RetrievalSample,
    TemporalSample,
    MSRVTTDataset,
)
from .metrics import Metrics, aggregate_metrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    
    task_type: str
    config_name: str
    timestamp: str
    metrics: Metrics
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type,
            "config_name": self.config_name,
            "timestamp": self.timestamp,
            "metrics": self.metrics.to_dict(),
            "config": self.config,
        }
        
    def save(self, output_path: str) -> None:
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
        logger.info(f"Saved benchmark results to {output_path}")


class BenchmarkRunner:
    """
    Run benchmarks on video understanding tasks.
    
    Supports QA, captioning, retrieval, and temporal grounding
    with configurable ablations.
    """
    
    def __init__(
        self,
        pipeline: VideoPipeline,
        output_dir: str = "results",
    ):
        """
        Initialize benchmark runner.
        
        Args:
            pipeline: VideoPipeline instance
            output_dir: Directory for output files
        """
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_qa_benchmark(
        self,
        samples: List[QASample],
        config_name: str = "default",
        progress: bool = True,
    ) -> BenchmarkResult:
        """
        Run QA benchmark.
        
        Args:
            samples: List of QA samples
            config_name: Name for this configuration
            progress: Show progress bar
            
        Returns:
            BenchmarkResult with metrics
        """
        logger.info(f"Running QA benchmark on {len(samples)} samples")
        
        sample_results = []
        iterator = tqdm(samples, desc="QA Benchmark") if progress else samples
        
        for sample in iterator:
            result = self._process_qa_sample(sample)
            sample_results.append(result)
            
        # Aggregate metrics
        metrics = aggregate_metrics(sample_results, "qa")
        
        return BenchmarkResult(
            task_type="qa",
            config_name=config_name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config=self.pipeline.config,
        )
        
    def run_caption_benchmark(
        self,
        samples: List[CaptionSample],
        config_name: str = "default",
        progress: bool = True,
    ) -> BenchmarkResult:
        """
        Run captioning benchmark.
        
        Args:
            samples: List of caption samples
            config_name: Name for this configuration
            progress: Show progress bar
            
        Returns:
            BenchmarkResult with metrics
        """
        logger.info(f"Running caption benchmark on {len(samples)} samples")
        
        sample_results = []
        iterator = tqdm(samples, desc="Caption Benchmark") if progress else samples
        
        for sample in iterator:
            result = self._process_caption_sample(sample)
            sample_results.append(result)
            
        metrics = aggregate_metrics(sample_results, "caption")
        
        return BenchmarkResult(
            task_type="caption",
            config_name=config_name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config=self.pipeline.config,
        )

    def run_retrieval_benchmark(
        self,
        samples: List[RetrievalSample],
        config_name: str = "default",
        progress: bool = True,
    ) -> BenchmarkResult:
        """Run retrieval benchmark (expects rank annotations)."""
        logger.info(f"Running retrieval benchmark on {len(samples)} samples")
        sample_results = []
        iterator = tqdm(samples, desc="Retrieval Benchmark") if progress else samples

        for sample in iterator:
            sample_results.append(self._process_retrieval_sample(sample))

        metrics = aggregate_metrics(sample_results, "retrieval")
        return BenchmarkResult(
            task_type="retrieval",
            config_name=config_name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config=self.pipeline.config,
        )

    def run_temporal_benchmark(
        self,
        samples: List[TemporalSample],
        config_name: str = "default",
        progress: bool = True,
    ) -> BenchmarkResult:
        """Run temporal grounding benchmark (requires GT spans)."""
        logger.info(f"Running temporal benchmark on {len(samples)} samples")
        sample_results = []
        iterator = tqdm(samples, desc="Temporal Benchmark") if progress else samples

        for sample in iterator:
            sample_results.append(self._process_temporal_sample(sample))

        metrics = aggregate_metrics(sample_results, "temporal")
        return BenchmarkResult(
            task_type="temporal",
            config_name=config_name,
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            config=self.pipeline.config,
        )
        
    def _process_qa_sample(self, sample: QASample) -> Dict[str, Any]:
        """Process a single QA sample."""
        result = {
            "video_id": sample.video_id,
            "question": sample.question,
            "ground_truth": sample.answer,
        }
        
        try:
            output = self.pipeline.process_qa(sample.video_path, sample.question)
            
            result.update({
                "prediction": output["response"],
                "latency_ms": output["metrics"].get("total_time_ms", 0),
                "cost_usd": output["metrics"].get("estimated_cost_usd", 0),
                "frame_reduction": output["metrics"].get("frame_reduction_ratio", 0),
                "frames_to_llm": output["metrics"].get("frames_sent_to_llm", 0),
                "error": output.get("error"),
            })
            
        except Exception as e:
            logger.error(f"Error processing {sample.video_id}: {e}")
            result["error"] = str(e)
            result["prediction"] = ""
            
        return result

    def _process_retrieval_sample(self, sample: RetrievalSample) -> Dict[str, Any]:
        """Process a retrieval sample. Uses provided rank as ground truth."""
        result = {
            "video_id": sample.video_id,
            "query": sample.query,
            "rank": sample.rank,
        }
        try:
            output = self.pipeline.process_retrieval(sample.video_path, sample.query)
            result.update(
                {
                    "prediction": output["response"],
                    "latency_ms": output["metrics"].get("total_time_ms", 0),
                    "cost_usd": output["metrics"].get("estimated_cost_usd", 0),
                    "frame_reduction": output["metrics"].get("frame_reduction_ratio", 0),
                    "frames_to_llm": output["metrics"].get("frames_sent_to_llm", 0),
                    "error": output.get("error"),
                }
            )
        except Exception as e:
            logger.error("Error processing %s: %s", sample.video_id, e)
            result["error"] = str(e)
            result["prediction"] = ""
        return result

    def _process_temporal_sample(self, sample: TemporalSample) -> Dict[str, Any]:
        """Process a temporal grounding sample."""
        result = {
            "video_id": sample.video_id,
            "query": sample.query,
            "gt_start": sample.gt_start,
            "gt_end": sample.gt_end,
        }
        try:
            output = self.pipeline.process_temporal(sample.video_path, sample.query)
            result.update(
                {
                    "prediction": output["response"],
                    "latency_ms": output["metrics"].get("total_time_ms", 0),
                    "cost_usd": output["metrics"].get("estimated_cost_usd", 0),
                    "frame_reduction": output["metrics"].get("frame_reduction_ratio", 0),
                    "frames_to_llm": output["metrics"].get("frames_sent_to_llm", 0),
                    "error": output.get("error"),
                }
            )
        except Exception as e:
            logger.error("Error processing %s: %s", sample.video_id, e)
            result["error"] = str(e)
            result["prediction"] = ""
        return result
        
    def _process_caption_sample(self, sample: CaptionSample) -> Dict[str, Any]:
        """Process a single caption sample."""
        result = {
            "video_id": sample.video_id,
            "ground_truth": sample.caption,
        }
        
        try:
            output = self.pipeline.process_caption(sample.video_path)
            
            result.update({
                "prediction": output["response"],
                "latency_ms": output["metrics"].get("total_time_ms", 0),
                "cost_usd": output["metrics"].get("estimated_cost_usd", 0),
                "frame_reduction": output["metrics"].get("frame_reduction_ratio", 0),
                "frames_to_llm": output["metrics"].get("frames_sent_to_llm", 0),
                "error": output.get("error"),
            })
            
        except Exception as e:
            logger.error(f"Error processing {sample.video_id}: {e}")
            result["error"] = str(e)
            result["prediction"] = ""
            
        return result
        
    def run_ablation(
        self,
        samples: List[QASample],
        configs: List[Dict[str, Any]],
        config_names: List[str],
        task_type: str = "qa",
    ) -> List[BenchmarkResult]:
        """
        Run ablation study with multiple configurations.
        
        Args:
            samples: Test samples
            configs: List of configuration dictionaries
            config_names: Names for each configuration
            task_type: Task type
            
        Returns:
            List of BenchmarkResult for each configuration
        """
        results = []
        
        for config, name in zip(configs, config_names):
            logger.info(f"Running ablation: {name}")
            
            # Create pipeline with this config
            pipeline = VideoPipeline(config)
            self.pipeline = pipeline
            
            if task_type == "qa":
                result = self.run_qa_benchmark(samples, config_name=name)
            else:
                raise ValueError(f"Unsupported task type for ablation: {task_type}")
                
            results.append(result)
            
            # Save intermediate results
            result.save(str(self.output_dir / f"ablation_{name}.json"))
            
        return results
        
    def compare_results(
        self,
        results: List[BenchmarkResult],
    ) -> Dict[str, Any]:
        """
        Compare multiple benchmark results.
        
        Args:
            results: List of BenchmarkResult objects
            
        Returns:
            Comparison dictionary
        """
        comparison = {
            "configs": [],
            "metrics": {},
        }
        
        metric_names = [
            "accuracy", "exact_match", "avg_latency_ms", 
            "avg_cost_usd", "avg_frame_reduction"
        ]
        
        for name in metric_names:
            comparison["metrics"][name] = []
            
        for result in results:
            comparison["configs"].append(result.config_name)
            
            metrics_dict = result.metrics.to_dict()
            for name in metric_names:
                comparison["metrics"][name].append(metrics_dict.get(name, 0))
                
        return comparison

