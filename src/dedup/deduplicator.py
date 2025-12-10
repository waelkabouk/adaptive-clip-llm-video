"""FAISS-based semantic frame deduplication."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of frame deduplication."""
    selected_indices: List[int]
    original_count: int
    deduplicated_count: int
    reduction_ratio: float
    method: str
    threshold: Optional[float] = None


class FrameDeduplicator(ABC):
    """Abstract base class for frame deduplication."""
    
    @abstractmethod
    def deduplicate(
        self,
        embeddings: np.ndarray,
        timestamps: Optional[List[float]] = None,
        max_frames: Optional[int] = None,
    ) -> DeduplicationResult:
        """
        Deduplicate frames based on embeddings.
        
        Args:
            embeddings: Frame embeddings (N, D)
            timestamps: Optional timestamps for temporal ordering
            max_frames: Maximum frames to keep
            
        Returns:
            DeduplicationResult with selected indices
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Deduplication method name."""
        pass


class GreedyCosineDedup(FrameDeduplicator):
    """
    Greedy cosine similarity deduplication.
    
    Iteratively selects frames that are sufficiently different
    from already selected frames based on cosine similarity.
    Uses FAISS for efficient similarity search.
    """
    
    def __init__(
        self,
        threshold: float = 0.85,
        min_frames: int = 4,
        max_frames: int = 16,
        preserve_temporal_order: bool = True,
    ):
        """
        Initialize greedy cosine deduplicator.
        
        Args:
            threshold: Cosine similarity threshold (frames above this are duplicates)
            min_frames: Minimum frames to keep
            max_frames: Maximum frames to keep
            preserve_temporal_order: Whether to preserve temporal ordering
        """
        self.threshold = threshold
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.preserve_temporal_order = preserve_temporal_order
        
    def deduplicate(
        self,
        embeddings: np.ndarray,
        timestamps: Optional[List[float]] = None,
        max_frames: Optional[int] = None,
    ) -> DeduplicationResult:
        """Deduplicate using greedy cosine similarity with FAISS."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not available, using numpy fallback")
            return self._deduplicate_numpy(embeddings, timestamps, max_frames)
            
        n_frames = len(embeddings)
        max_frames = max_frames or self.max_frames
        
        if n_frames <= self.min_frames:
            return DeduplicationResult(
                selected_indices=list(range(n_frames)),
                original_count=n_frames,
                deduplicated_count=n_frames,
                reduction_ratio=0.0,
                method=self.name,
                threshold=self.threshold,
            )
            
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        embeddings_normalized = embeddings_normalized.astype(np.float32)
        
        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine for normalized
        
        selected_indices = []
        
        # Always include first frame
        selected_indices.append(0)
        index.add(embeddings_normalized[0:1])
        
        # Greedy selection
        for i in range(1, n_frames):
            if len(selected_indices) >= max_frames:
                break
                
            # Check similarity to already selected frames
            query = embeddings_normalized[i:i+1]
            similarities, _ = index.search(query, min(len(selected_indices), 10))
            max_similarity = similarities[0].max()
            
            # Add if sufficiently different
            if max_similarity < self.threshold:
                selected_indices.append(i)
                index.add(query)
                
        # Ensure minimum frames
        if len(selected_indices) < self.min_frames:
            # Add frames with lowest similarity to selected set
            remaining = [i for i in range(n_frames) if i not in selected_indices]
            
            while len(selected_indices) < self.min_frames and remaining:
                # Find frame most different from current selection
                best_idx = None
                best_min_sim = float('inf')
                
                for idx in remaining:
                    query = embeddings_normalized[idx:idx+1]
                    similarities, _ = index.search(query, len(selected_indices))
                    min_sim = similarities[0].min()
                    
                    if min_sim < best_min_sim:
                        best_min_sim = min_sim
                        best_idx = idx
                        
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    index.add(embeddings_normalized[best_idx:best_idx+1])
                    remaining.remove(best_idx)
                    
        # Sort by temporal order if required
        if self.preserve_temporal_order:
            selected_indices.sort()
            
        reduction_ratio = 1.0 - (len(selected_indices) / n_frames)
        
        return DeduplicationResult(
            selected_indices=selected_indices,
            original_count=n_frames,
            deduplicated_count=len(selected_indices),
            reduction_ratio=reduction_ratio,
            method=self.name,
            threshold=self.threshold,
        )
        
    def _deduplicate_numpy(
        self,
        embeddings: np.ndarray,
        timestamps: Optional[List[float]] = None,
        max_frames: Optional[int] = None,
    ) -> DeduplicationResult:
        """Fallback deduplication using numpy."""
        n_frames = len(embeddings)
        max_frames = max_frames or self.max_frames
        
        if n_frames <= self.min_frames:
            return DeduplicationResult(
                selected_indices=list(range(n_frames)),
                original_count=n_frames,
                deduplicated_count=n_frames,
                reduction_ratio=0.0,
                method=self.name,
                threshold=self.threshold,
            )
            
        # Normalize
        embeddings_normalized = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        
        selected_indices = [0]
        
        for i in range(1, n_frames):
            if len(selected_indices) >= max_frames:
                break
                
            # Compute similarities to selected frames
            selected_embs = embeddings_normalized[selected_indices]
            similarities = embeddings_normalized[i] @ selected_embs.T
            
            if similarities.max() < self.threshold:
                selected_indices.append(i)
                
        # Ensure minimum frames
        while len(selected_indices) < min(self.min_frames, n_frames):
            remaining = [i for i in range(n_frames) if i not in selected_indices]
            if not remaining:
                break
            # Add evenly spaced frames
            idx = remaining[len(remaining) // 2]
            selected_indices.append(idx)
            
        if self.preserve_temporal_order:
            selected_indices.sort()
            
        reduction_ratio = 1.0 - (len(selected_indices) / n_frames)
        
        return DeduplicationResult(
            selected_indices=selected_indices,
            original_count=n_frames,
            deduplicated_count=len(selected_indices),
            reduction_ratio=reduction_ratio,
            method=self.name,
            threshold=self.threshold,
        )
    
    @property
    def name(self) -> str:
        return "greedy_cosine"


class KMeansDedup(FrameDeduplicator):
    """
    K-means clustering deduplication.
    
    Clusters frames and selects representatives from each cluster.
    """
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        min_frames: int = 4,
        max_frames: int = 16,
        preserve_temporal_order: bool = True,
    ):
        """
        Initialize K-means deduplicator.
        
        Args:
            n_clusters: Number of clusters (default: max_frames)
            min_frames: Minimum frames to keep
            max_frames: Maximum frames to keep
            preserve_temporal_order: Whether to preserve temporal ordering
        """
        self.n_clusters = n_clusters
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.preserve_temporal_order = preserve_temporal_order
        
    def deduplicate(
        self,
        embeddings: np.ndarray,
        timestamps: Optional[List[float]] = None,
        max_frames: Optional[int] = None,
    ) -> DeduplicationResult:
        """Deduplicate using K-means clustering."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not available for K-means, using greedy fallback")
            fallback = GreedyCosineDedup(
                min_frames=self.min_frames,
                max_frames=max_frames or self.max_frames,
                preserve_temporal_order=self.preserve_temporal_order,
            )
            return fallback.deduplicate(embeddings, timestamps, max_frames)
            
        n_frames = len(embeddings)
        max_frames = max_frames or self.max_frames
        n_clusters = self.n_clusters or min(max_frames, n_frames)
        
        if n_frames <= n_clusters:
            return DeduplicationResult(
                selected_indices=list(range(n_frames)),
                original_count=n_frames,
                deduplicated_count=n_frames,
                reduction_ratio=0.0,
                method=self.name,
            )
            
        # Normalize embeddings
        embeddings_normalized = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        embeddings_normalized = embeddings_normalized.astype(np.float32)
        
        # K-means clustering
        dim = embeddings.shape[1]
        kmeans = faiss.Kmeans(
            dim, n_clusters,
            niter=20,
            verbose=False,
            spherical=True,  # For cosine similarity
        )
        kmeans.train(embeddings_normalized)
        
        # Assign frames to clusters
        _, cluster_assignments = kmeans.index.search(embeddings_normalized, 1)
        cluster_assignments = cluster_assignments.flatten()
        
        # Select representative from each cluster (closest to centroid)
        selected_indices = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Find frame closest to centroid
            centroid = kmeans.centroids[cluster_id:cluster_id+1]
            cluster_embeddings = embeddings_normalized[cluster_indices]
            similarities = (cluster_embeddings @ centroid.T).flatten()
            
            best_local_idx = similarities.argmax()
            best_global_idx = cluster_indices[best_local_idx]
            selected_indices.append(int(best_global_idx))
            
        # Sort by temporal order if required
        if self.preserve_temporal_order:
            selected_indices.sort()
            
        reduction_ratio = 1.0 - (len(selected_indices) / n_frames)
        
        return DeduplicationResult(
            selected_indices=selected_indices,
            original_count=n_frames,
            deduplicated_count=len(selected_indices),
            reduction_ratio=reduction_ratio,
            method=self.name,
        )
    
    @property
    def name(self) -> str:
        return "kmeans"


def create_deduplicator(
    method: str,
    **kwargs,
) -> FrameDeduplicator:
    """
    Factory function to create a frame deduplicator.
    
    Args:
        method: Deduplication method (greedy_cosine, kmeans)
        **kwargs: Method-specific parameters
        
    Returns:
        FrameDeduplicator instance
    """
    deduplicators = {
        "greedy_cosine": GreedyCosineDedup,
        "kmeans": KMeansDedup,
    }
    
    if method not in deduplicators:
        raise ValueError(f"Unknown dedup method: {method}. "
                        f"Available: {list(deduplicators.keys())}")
    
    return deduplicators[method](**kwargs)

