"""Dataset loaders for evaluation."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QASample:
    """QA sample from dataset."""
    video_id: str
    video_path: str
    question: str
    answer: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CaptionSample:
    """Caption sample from dataset."""
    video_id: str
    video_path: str
    caption: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievalSample:
    """Retrieval sample (for text-video retrieval eval)."""
    video_id: str
    video_path: str
    query: str
    # 1-indexed rank of correct video; lower is better
    rank: int = 1000
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TemporalSample:
    """Temporal grounding sample."""
    video_id: str
    video_path: str
    query: str
    gt_start: float
    gt_end: float
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MSRVTTDataset:
    """
    MSR-VTT dataset loader.
    
    Supports QA and captioning tasks from MSR-VTT variants.
    """
    
    def __init__(
        self,
        data_dir: str,
        video_dir: Optional[str] = None,
        split: str = "test",
    ):
        """
        Initialize MSR-VTT dataset.
        
        Args:
            data_dir: Directory containing annotation files
            video_dir: Directory containing video files
            split: Dataset split (train, val, test)
        """
        self.data_dir = Path(data_dir)
        self.video_dir = Path(video_dir) if video_dir else self.data_dir / "videos"
        self.split = split
        
        self._qa_data: Optional[List[Dict]] = None
        self._caption_data: Optional[List[Dict]] = None
        
    def load_qa(self, subset_size: Optional[int] = None) -> List[QASample]:
        """
        Load QA annotations.
        
        Args:
            subset_size: Limit number of samples (for dev)
            
        Returns:
            List of QASample objects
        """
        # Try common MSR-VTT QA file names
        qa_files = [
            self.data_dir / f"{self.split}_qa.json",
            self.data_dir / f"msrvtt_qa_{self.split}.json",
            self.data_dir / f"qa_{self.split}.json",
            self.data_dir / "qa.json",
        ]
        
        qa_file = None
        for f in qa_files:
            if f.exists():
                qa_file = f
                break
                
        if qa_file is None:
            logger.warning(f"No QA file found in {self.data_dir}")
            return []
            
        logger.info(f"Loading QA data from {qa_file}")
        
        with open(qa_file, 'r') as f:
            data = json.load(f)
            
        # Handle different formats
        if isinstance(data, dict):
            # Some formats have nested structure
            data = data.get("annotations", data.get("data", []))
            
        samples = []
        for item in data:
            video_id = item.get("video_id", item.get("video", ""))
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            # Try to construct video path
            video_path = self._find_video(video_id)
            
            if video_path and question and answer:
                samples.append(QASample(
                    video_id=video_id,
                    video_path=str(video_path),
                    question=question,
                    answer=answer,
                    metadata=item,
                ))
                
        if subset_size and len(samples) > subset_size:
            # Sample evenly
            import numpy as np
            indices = np.linspace(0, len(samples) - 1, subset_size, dtype=int)
            samples = [samples[i] for i in indices]
            
        logger.info(f"Loaded {len(samples)} QA samples")
        return samples
        
    def load_captions(self, subset_size: Optional[int] = None) -> List[CaptionSample]:
        """
        Load caption annotations.
        
        Args:
            subset_size: Limit number of samples
            
        Returns:
            List of CaptionSample objects
        """
        # Try common caption file names
        caption_files = [
            self.data_dir / f"{self.split}_captions.json",
            self.data_dir / f"msrvtt_caption_{self.split}.json",
            self.data_dir / "captions.json",
        ]
        
        caption_file = None
        for f in caption_files:
            if f.exists():
                caption_file = f
                break
                
        if caption_file is None:
            logger.warning(f"No caption file found in {self.data_dir}")
            return []
            
        logger.info(f"Loading caption data from {caption_file}")
        
        with open(caption_file, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            data = data.get("sentences", data.get("annotations", []))
            
        samples = []
        for item in data:
            video_id = item.get("video_id", item.get("video", ""))
            caption = item.get("caption", item.get("sentence", ""))
            
            video_path = self._find_video(video_id)
            
            if video_path and caption:
                samples.append(CaptionSample(
                    video_id=video_id,
                    video_path=str(video_path),
                    caption=caption,
                    metadata=item,
                ))
                
        if subset_size and len(samples) > subset_size:
            import numpy as np
            indices = np.linspace(0, len(samples) - 1, subset_size, dtype=int)
            samples = [samples[i] for i in indices]
            
        logger.info(f"Loaded {len(samples)} caption samples")
        return samples
        
    def _find_video(self, video_id: str) -> Optional[Path]:
        """Find video file for a video ID."""
        # Common video extensions
        extensions = [".mp4", ".avi", ".webm", ".mkv"]
        
        for ext in extensions:
            # Try direct match
            video_path = self.video_dir / f"{video_id}{ext}"
            if video_path.exists():
                return video_path
                
            # Try with video prefix
            video_path = self.video_dir / f"video{video_id}{ext}"
            if video_path.exists():
                return video_path
                
        return None
        
    def __len__(self) -> int:
        """Get dataset size (QA samples)."""
        if self._qa_data is None:
            self._qa_data = self.load_qa()
        return len(self._qa_data)


def load_msrvtt_qa(
    data_dir: str,
    video_dir: Optional[str] = None,
    split: str = "test",
    subset_size: Optional[int] = None,
) -> List[QASample]:
    """
    Convenience function to load MSR-VTT QA.
    
    Args:
        data_dir: Directory containing annotations
        video_dir: Directory containing videos
        split: Dataset split
        subset_size: Limit samples
        
    Returns:
        List of QASample objects
    """
    dataset = MSRVTTDataset(data_dir, video_dir, split)
    return dataset.load_qa(subset_size)

