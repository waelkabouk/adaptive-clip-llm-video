"""CLIP encoder module."""

from .base import BaseEncoder, EncoderOutput
from .clip_encoder import CLIPEncoder

__all__ = [
    "BaseEncoder",
    "EncoderOutput", 
    "CLIPEncoder",
]

