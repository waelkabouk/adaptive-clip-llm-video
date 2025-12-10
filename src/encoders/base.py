"""Base encoder interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class EncoderOutput:
    """Output from an encoder."""
    embeddings: np.ndarray  # Shape: (N, embedding_dim)
    embedding_dim: int
    model_name: str
    device: str
    

class BaseEncoder(ABC):
    """Abstract base class for visual encoders."""
    
    @abstractmethod
    def encode_images(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: Optional[int] = None,
    ) -> EncoderOutput:
        """
        Encode a list of images into embeddings.
        
        Args:
            images: List of images (numpy arrays or PIL Images)
            batch_size: Override default batch size
            
        Returns:
            EncoderOutput with embeddings
        """
        pass
    
    @abstractmethod
    def encode_text(self, texts: List[str]) -> EncoderOutput:
        """
        Encode a list of text strings into embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            EncoderOutput with embeddings
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Get current device."""
        pass

