"""Frame sampling strategies for video processing."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from .loader import VideoLoader, VideoMetadata


@dataclass
class SamplingResult:
    """Result of frame sampling."""
    frame_indices: List[int]
    timestamps: List[float]
    strategy: str
    original_frame_count: int
    sampled_frame_count: int


class FrameSampler(ABC):
    """Abstract base class for frame sampling strategies."""
    
    @abstractmethod
    def sample(
        self,
        metadata: VideoMetadata,
        max_frames: Optional[int] = None,
    ) -> SamplingResult:
        """
        Sample frame indices from a video.
        
        Args:
            metadata: Video metadata
            max_frames: Maximum number of frames to sample
            
        Returns:
            SamplingResult with selected frame indices
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


class UniformSampler(FrameSampler):
    """
    Uniform temporal sampling.
    
    Samples frames at evenly spaced intervals throughout the video.
    """
    
    def __init__(self, num_frames: int = 32):
        """
        Initialize uniform sampler.
        
        Args:
            num_frames: Default number of frames to sample
        """
        self.num_frames = num_frames
        
    def sample(
        self,
        metadata: VideoMetadata,
        max_frames: Optional[int] = None,
    ) -> SamplingResult:
        """Sample frames uniformly."""
        target_frames = max_frames or self.num_frames
        total_frames = metadata.total_frames
        
        if total_frames <= target_frames:
            # Return all frames if video is short
            indices = list(range(total_frames))
        else:
            # Uniform sampling
            indices = np.linspace(
                0, total_frames - 1, target_frames, dtype=int
            ).tolist()
            
        # Calculate timestamps
        timestamps = [i / metadata.fps for i in indices]
        
        return SamplingResult(
            frame_indices=indices,
            timestamps=timestamps,
            strategy=self.name,
            original_frame_count=total_frames,
            sampled_frame_count=len(indices),
        )
    
    @property
    def name(self) -> str:
        return "uniform"


class FPSCapSampler(FrameSampler):
    """
    FPS-capped sampling.
    
    Samples frames at a target FPS, capping the effective frame rate.
    Useful for reducing redundancy in high-FPS videos.
    """
    
    def __init__(self, target_fps: float = 2.0, max_frames: int = 64):
        """
        Initialize FPS cap sampler.
        
        Args:
            target_fps: Target frames per second
            max_frames: Maximum frames to sample regardless of duration
        """
        self.target_fps = target_fps
        self.max_frames = max_frames
        
    def sample(
        self,
        metadata: VideoMetadata,
        max_frames: Optional[int] = None,
    ) -> SamplingResult:
        """Sample frames at target FPS."""
        max_frames = max_frames or self.max_frames
        
        # Calculate frame interval based on target FPS
        source_fps = metadata.fps
        frame_interval = source_fps / self.target_fps
        
        # Generate indices at target FPS
        indices = []
        current_frame = 0.0
        
        while current_frame < metadata.total_frames and len(indices) < max_frames:
            indices.append(int(current_frame))
            current_frame += frame_interval
            
        # Calculate timestamps
        timestamps = [i / metadata.fps for i in indices]
        
        return SamplingResult(
            frame_indices=indices,
            timestamps=timestamps,
            strategy=self.name,
            original_frame_count=metadata.total_frames,
            sampled_frame_count=len(indices),
        )
    
    @property
    def name(self) -> str:
        return "fps_cap"


class SceneDetectSampler(FrameSampler):
    """
    Scene-based sampling (simplified).
    
    Uses frame difference to detect scene changes and samples
    frames near transitions. Falls back to uniform if no scenes detected.
    
    Note: This is a simplified implementation. For production,
    consider using PySceneDetect or similar libraries.
    """
    
    def __init__(
        self,
        threshold: float = 30.0,
        min_scene_length: int = 10,
        max_frames: int = 32,
    ):
        """
        Initialize scene detect sampler.
        
        Args:
            threshold: Scene change threshold (mean frame diff)
            min_scene_length: Minimum frames between scene changes
            max_frames: Maximum frames to sample
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.max_frames = max_frames
        self._uniform_fallback = UniformSampler(max_frames)
        
    def sample(
        self,
        metadata: VideoMetadata,
        max_frames: Optional[int] = None,
    ) -> SamplingResult:
        """
        Sample frames based on scene changes.
        
        Note: This is a stub that returns uniform sampling.
        Full scene detection requires frame content analysis.
        """
        # For now, fall back to uniform sampling
        # Full implementation would analyze frame differences
        result = self._uniform_fallback.sample(metadata, max_frames or self.max_frames)
        
        return SamplingResult(
            frame_indices=result.frame_indices,
            timestamps=result.timestamps,
            strategy=self.name,
            original_frame_count=result.original_frame_count,
            sampled_frame_count=result.sampled_frame_count,
        )
    
    def sample_with_frames(
        self,
        frames: List[np.ndarray],
        metadata: VideoMetadata,
        max_frames: Optional[int] = None,
    ) -> SamplingResult:
        """
        Sample based on actual frame content (scene detection).
        
        Args:
            frames: Pre-loaded frames for analysis
            metadata: Video metadata
            max_frames: Maximum frames to return
            
        Returns:
            SamplingResult with scene-based indices
        """
        import cv2
        
        max_frames = max_frames or self.max_frames
        
        if len(frames) <= max_frames:
            return SamplingResult(
                frame_indices=list(range(len(frames))),
                timestamps=[i / metadata.fps for i in range(len(frames))],
                strategy=self.name,
                original_frame_count=len(frames),
                sampled_frame_count=len(frames),
            )
        
        # Calculate frame differences
        scene_changes = [0]  # Always include first frame
        
        for i in range(1, len(frames)):
            # Convert to grayscale and calculate difference
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            
            diff = cv2.absdiff(prev_gray, curr_gray)
            mean_diff = np.mean(diff)
            
            # Check if scene change and respect minimum scene length
            if mean_diff > self.threshold:
                if not scene_changes or (i - scene_changes[-1]) >= self.min_scene_length:
                    scene_changes.append(i)
                    
        # If too few scene changes, add uniform samples
        if len(scene_changes) < max_frames:
            uniform_indices = np.linspace(
                0, len(frames) - 1, max_frames, dtype=int
            ).tolist()
            # Merge and deduplicate
            all_indices = sorted(set(scene_changes + uniform_indices))
        else:
            all_indices = scene_changes
            
        # Trim to max_frames if needed (keep evenly distributed)
        if len(all_indices) > max_frames:
            keep_indices = np.linspace(
                0, len(all_indices) - 1, max_frames, dtype=int
            )
            all_indices = [all_indices[i] for i in keep_indices]
            
        timestamps = [i / metadata.fps for i in all_indices]
        
        return SamplingResult(
            frame_indices=all_indices,
            timestamps=timestamps,
            strategy=self.name,
            original_frame_count=len(frames),
            sampled_frame_count=len(all_indices),
        )
    
    @property
    def name(self) -> str:
        return "scene_detect"


def create_sampler(
    strategy: str,
    **kwargs,
) -> FrameSampler:
    """
    Factory function to create a frame sampler.
    
    Args:
        strategy: Sampling strategy name (uniform, fps_cap, scene_detect)
        **kwargs: Strategy-specific parameters
        
    Returns:
        FrameSampler instance
    """
    samplers = {
        "uniform": UniformSampler,
        "fps_cap": FPSCapSampler,
        "scene_detect": SceneDetectSampler,
    }
    
    if strategy not in samplers:
        raise ValueError(f"Unknown sampling strategy: {strategy}. "
                        f"Available: {list(samplers.keys())}")
    
    return samplers[strategy](**kwargs)

