"""Tests for the video pipeline components."""

import pytest
import numpy as np
from pathlib import Path


class TestVideoSampling:
    """Tests for video sampling strategies."""
    
    def test_uniform_sampler(self):
        """Test uniform frame sampling."""
        from src.video.sampler import UniformSampler
        from src.video.loader import VideoMetadata
        
        sampler = UniformSampler(num_frames=8)
        
        # Mock metadata
        metadata = VideoMetadata(
            path="/test/video.mp4",
            duration=10.0,
            fps=30.0,
            total_frames=300,
            width=1920,
            height=1080,
        )
        
        result = sampler.sample(metadata)
        
        assert result.sampled_frame_count == 8
        assert len(result.frame_indices) == 8
        assert result.frame_indices[0] == 0
        assert result.frame_indices[-1] == 299
        assert result.strategy == "uniform"
        
    def test_fps_cap_sampler(self):
        """Test FPS-capped sampling."""
        from src.video.sampler import FPSCapSampler
        from src.video.loader import VideoMetadata
        
        sampler = FPSCapSampler(target_fps=2.0, max_frames=20)
        
        metadata = VideoMetadata(
            path="/test/video.mp4",
            duration=10.0,
            fps=30.0,
            total_frames=300,
            width=1920,
            height=1080,
        )
        
        result = sampler.sample(metadata)
        
        # At 2 FPS for 10 seconds = 20 frames
        assert result.sampled_frame_count == 20
        assert result.strategy == "fps_cap"


class TestDeduplication:
    """Tests for frame deduplication."""
    
    def test_greedy_cosine_dedup(self):
        """Test greedy cosine deduplication."""
        from src.dedup.deduplicator import GreedyCosineDedup
        
        dedup = GreedyCosineDedup(
            threshold=0.9,
            min_frames=2,
            max_frames=10,
        )
        
        # Create embeddings with some duplicates
        # First 3 are similar, next 3 are similar, last 2 are different
        dim = 512
        np.random.seed(42)
        
        base1 = np.random.randn(dim)
        base2 = np.random.randn(dim)
        base3 = np.random.randn(dim)
        base4 = np.random.randn(dim)
        
        embeddings = np.array([
            base1,
            base1 + 0.01 * np.random.randn(dim),  # Similar to base1
            base1 + 0.02 * np.random.randn(dim),  # Similar to base1
            base2,
            base2 + 0.01 * np.random.randn(dim),  # Similar to base2
            base3,
            base4,
        ])
        
        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        result = dedup.deduplicate(embeddings)
        
        # Should reduce from 7 to fewer frames
        assert result.deduplicated_count < result.original_count
        assert result.deduplicated_count >= 2  # min_frames
        assert 0 in result.selected_indices  # Should include first frame
        
    def test_kmeans_dedup(self):
        """Test K-means deduplication."""
        from src.dedup.deduplicator import KMeansDedup
        
        dedup = KMeansDedup(
            n_clusters=4,
            min_frames=2,
            max_frames=10,
        )
        
        # Create embeddings
        dim = 512
        np.random.seed(42)
        embeddings = np.random.randn(20, dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        result = dedup.deduplicate(embeddings.astype(np.float32))
        
        # Should select approximately n_clusters frames
        assert result.deduplicated_count <= 10
        assert result.deduplicated_count >= 2


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_accuracy_computation(self):
        """Test accuracy metric computation."""
        from src.evaluation.metrics import compute_accuracy
        
        predictions = ["a cat", "A DOG", "bird"]
        ground_truths = ["cat", "dog", "fish"]
        
        result = compute_accuracy(predictions, ground_truths)
        
        # "a cat" contains "cat", "A DOG" contains "dog"
        assert result["accuracy"] == 2/3
        
    def test_retrieval_metrics(self):
        """Test retrieval metric computation."""
        from src.evaluation.metrics import compute_retrieval_metrics
        
        rankings = [1, 3, 5, 10, 2]  # Ranks of correct items
        
        result = compute_retrieval_metrics(rankings, k_values=[1, 5, 10])
        
        assert result["recall_at_1"] == 1/5  # Only rank 1 is <= 1
        assert result["recall_at_5"] == 4/5  # Ranks 1, 3, 5, 2 are <= 5
        assert result["recall_at_10"] == 1.0  # All are <= 10


class TestPipelineState:
    """Tests for pipeline state management."""
    
    def test_task_type_validation(self):
        """Test task type enum."""
        from src.pipeline.state import TaskType
        
        assert TaskType.QA.value == "qa"
        assert TaskType.CAPTION.value == "caption"
        assert TaskType.RETRIEVAL.value == "retrieval"
        assert TaskType.TEMPORAL.value == "temporal"


# Skip integration tests if dependencies not available
@pytest.mark.skipif(
    not Path("configs/config.yaml").exists(),
    reason="Config file not found"
)
class TestIntegration:
    """Integration tests (require full setup)."""
    
    def test_pipeline_creation(self):
        """Test pipeline can be created."""
        from src.pipeline import VideoPipeline
        
        pipeline = VideoPipeline({})
        assert pipeline is not None
        assert pipeline.graph is not None

