"""Video loading utilities using decord."""

import os
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:
    import decord
    from decord import VideoReader, cpu, gpu
    decord.bridge.set_bridge("native")
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

import cv2


@dataclass
class VideoMetadata:
    """Video metadata container."""
    path: str
    duration: float  # seconds
    fps: float
    total_frames: int
    width: int
    height: int
    

class VideoLoader:
    """
    Video loader with support for decord (preferred) and OpenCV fallback.
    
    Handles video loading, metadata extraction, and frame retrieval.
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize video loader.
        
        Args:
            use_gpu: Whether to use GPU for decoding (requires decord GPU build)
            cache_dir: Directory for caching extracted frames
        """
        self.use_gpu = use_gpu and DECORD_AVAILABLE
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        self._reader: Optional[VideoReader] = None
        self._cv_cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._current_path: Optional[str] = None
        
    def load(self, video_path: str) -> VideoMetadata:
        """
        Load a video file and extract metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata object
        """
        video_path = str(Path(video_path).resolve())
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        self._current_path = video_path
        
        if DECORD_AVAILABLE:
            self._load_with_decord(video_path)
        else:
            self._load_with_opencv(video_path)
            
        return self._metadata
    
    def _load_with_decord(self, video_path: str) -> None:
        """Load video using decord."""
        ctx = gpu(0) if self.use_gpu else cpu(0)
        self._reader = VideoReader(video_path, ctx=ctx)
        
        fps = self._reader.get_avg_fps()
        total_frames = len(self._reader)
        duration = total_frames / fps if fps > 0 else 0
        
        # Get frame dimensions from first frame
        first_frame_tensor = self._reader[0]
        first_frame = first_frame_tensor.asnumpy() if hasattr(first_frame_tensor, 'asnumpy') else np.array(first_frame_tensor)
        height, width = first_frame.shape[:2]
        
        self._metadata = VideoMetadata(
            path=video_path,
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
        )
        
    def _load_with_opencv(self, video_path: str) -> None:
        """Load video using OpenCV (fallback)."""
        self._cv_cap = cv2.VideoCapture(video_path)
        
        if not self._cv_cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        fps = self._cv_cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        self._metadata = VideoMetadata(
            path=video_path,
            duration=duration,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
        )
        
    def get_frames(
        self,
        frame_indices: List[int],
        resize: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Extract specific frames from the video.
        
        Args:
            frame_indices: List of frame indices to extract
            resize: Optional (width, height) to resize frames
            
        Returns:
            Tuple of (frames as numpy arrays, timestamps in seconds)
        """
        if self._metadata is None:
            raise RuntimeError("No video loaded. Call load() first.")

        # Try cache first (only if cache_dir is set and all frames cached)
        cached_frames, cached_ts = self._try_load_cache(frame_indices)
        if cached_frames is not None:
            frames, timestamps = cached_frames, cached_ts
        else:
            frames, timestamps = self._decode_frames(frame_indices)
            self._save_cache(frame_indices, frames)

        # Resize if needed
        if resize:
            frames = [self._resize_frame(f, resize) for f in frames]

        return frames, timestamps

    def _decode_frames(
        self,
        frame_indices: List[int],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Decode frames via decord or OpenCV."""
        if DECORD_AVAILABLE and self._reader is not None:
            return self._get_frames_decord(frame_indices)
        return self._get_frames_opencv(frame_indices)
    
    def _get_frames_decord(
        self,
        frame_indices: List[int],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames using decord."""
        # Clamp indices to valid range
        valid_indices = [
            max(0, min(i, self._metadata.total_frames - 1))
            for i in frame_indices
        ]
        
        # Batch extraction for efficiency
        frames_tensor = self._reader.get_batch(valid_indices)
        # Convert to numpy array first (decord NDArray may not support direct subscript)
        frames_array = frames_tensor.asnumpy()
        frames = [frames_array[i] for i in range(len(valid_indices))]
        
        # Calculate timestamps
        timestamps = [i / self._metadata.fps for i in valid_indices]
        
        return frames, timestamps
    
    def _get_frames_opencv(
        self,
        frame_indices: List[int],
    ) -> Tuple[List[np.ndarray], List[float]]:
        """Extract frames using OpenCV."""
        frames = []
        timestamps = []
        
        for idx in frame_indices:
            idx = max(0, min(idx, self._metadata.total_frames - 1))
            self._cv_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self._cv_cap.read()
            
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
                timestamps.append(idx / self._metadata.fps)
            else:
                # Use black frame as fallback
                frames.append(np.zeros(
                    (self._metadata.height, self._metadata.width, 3),
                    dtype=np.uint8
                ))
                timestamps.append(idx / self._metadata.fps)
                
        return frames, timestamps

    def _cache_path(self, frame_idx: int) -> Optional[Path]:
        """Compute cache path for a frame index."""
        if not self.cache_dir or not self._current_path:
            return None
        safe_name = Path(self._current_path).stem
        video_hash = abs(hash(self._current_path)) % (10**8)
        return self.cache_dir / f"{safe_name}_{video_hash}/frame_{frame_idx:06d}.npy"

    def _try_load_cache(
        self,
        frame_indices: List[int],
    ) -> Tuple[Optional[List[np.ndarray]], Optional[List[float]]]:
        """
        Load all requested frames from cache if available.
        Returns (frames, timestamps) or (None, None).
        """
        if not self.cache_dir:
            return None, None

        frames = []
        timestamps = []
        for idx in frame_indices:
            cache_path = self._cache_path(idx)
            if cache_path is None or not cache_path.exists():
                return None, None
            arr = np.load(cache_path)
            frames.append(arr)
            timestamps.append(idx / self._metadata.fps)
        return frames, timestamps

    def _save_cache(self, frame_indices: List[int], frames: List[np.ndarray]) -> None:
        """Persist decoded frames to cache (best-effort)."""
        if not self.cache_dir:
            return
        if len(frame_indices) != len(frames):
            return

        for idx, frame in zip(frame_indices, frames):
            cache_path = self._cache_path(idx)
            if cache_path is None:
                continue
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                np.save(cache_path, frame)
            except Exception:
                # Cache should not block main path
                continue
    
    def _resize_frame(
        self,
        frame: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Resize a frame to target size."""
        return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)
    
    def get_frame_as_pil(
        self,
        frame_idx: int,
        resize: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Get a single frame as PIL Image."""
        frames, _ = self.get_frames([frame_idx], resize=resize)
        return Image.fromarray(frames[0])
    
    def close(self) -> None:
        """Release video resources."""
        if self._cv_cap is not None:
            self._cv_cap.release()
            self._cv_cap = None
        self._reader = None
        self._metadata = None
        self._current_path = None
        
    def __enter__(self) -> "VideoLoader":
        return self
        
    def __exit__(self, *args) -> None:
        self.close()
        
    @property
    def metadata(self) -> Optional[VideoMetadata]:
        """Get current video metadata."""
        return self._metadata
    
    @property
    def is_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        return self._metadata is not None

