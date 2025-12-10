"""CLIP encoder implementation with memory-aware batching."""

import logging
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .base import BaseEncoder, EncoderOutput

logger = logging.getLogger(__name__)


class CLIPEncoder(BaseEncoder):
    """
    CLIP encoder using OpenCLIP with memory-aware batching.
    
    Automatically reduces batch size on OOM errors to handle
    limited VRAM scenarios.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
        batch_size: int = 16,
        auto_batch_scale: bool = True,
    ):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: CLIP model variant (e.g., ViT-B-32, ViT-L-14)
            pretrained: Pretrained weights source (e.g., openai, laion2b_s34b_b79k)
            device: Device to use (auto, cuda, cpu)
            batch_size: Initial batch size for encoding
            auto_batch_scale: Whether to automatically reduce batch on OOM
        """
        self._model_name = model_name
        self._pretrained = pretrained
        self._batch_size = batch_size
        self._auto_batch_scale = auto_batch_scale
        self._min_batch_size = 1
        
        # Determine device
        if device is None or device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device
            
        # Load model
        self._load_model()
        
    def _load_model(self) -> None:
        """Load CLIP model and preprocessing."""
        try:
            import open_clip
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self._model_name,
                pretrained=self._pretrained,
                device=self._device,
            )
            self._tokenizer = open_clip.get_tokenizer(self._model_name)
            self._model.eval()
            
            # Get embedding dimension
            self._embedding_dim = self._model.visual.output_dim
            
            logger.info(
                f"Loaded CLIP model {self._model_name} on {self._device} "
                f"(embedding_dim={self._embedding_dim})"
            )
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
            
    def encode_images(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: Optional[int] = None,
    ) -> EncoderOutput:
        """
        Encode images with memory-aware batching.
        
        Automatically reduces batch size on OOM errors.
        """
        if not images:
            return EncoderOutput(
                embeddings=np.array([]),
                embedding_dim=self._embedding_dim,
                model_name=self._model_name,
                device=self._device,
            )
            
        batch_size = batch_size or self._batch_size
        all_embeddings = []
        
        # Convert images to PIL if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img))
            else:
                pil_images.append(img)
                
        # Process in batches
        idx = 0
        while idx < len(pil_images):
            batch = pil_images[idx:idx + batch_size]
            
            try:
                embeddings = self._encode_batch(batch)
                all_embeddings.append(embeddings)
                idx += batch_size
                
            except torch.cuda.OutOfMemoryError:
                if self._auto_batch_scale and batch_size > self._min_batch_size:
                    # Reduce batch size and retry
                    new_batch_size = max(self._min_batch_size, batch_size // 2)
                    logger.warning(
                        f"OOM error, reducing batch size: {batch_size} -> {new_batch_size}"
                    )
                    batch_size = new_batch_size
                    torch.cuda.empty_cache()
                else:
                    raise
                    
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        return EncoderOutput(
            embeddings=embeddings,
            embedding_dim=self._embedding_dim,
            model_name=self._model_name,
            device=self._device,
        )
        
    def _encode_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Encode a single batch of images."""
        # Preprocess images
        image_tensors = torch.stack([
            self._preprocess(img) for img in images
        ]).to(self._device)
        
        # Encode
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device == "cuda"):
            features = self._model.encode_image(image_tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            
        return features.cpu().numpy()
        
    def encode_text(self, texts: List[str]) -> EncoderOutput:
        """Encode text strings into embeddings."""
        if not texts:
            return EncoderOutput(
                embeddings=np.array([]),
                embedding_dim=self._embedding_dim,
                model_name=self._model_name,
                device=self._device,
            )
            
        # Tokenize
        tokens = self._tokenizer(texts).to(self._device)
        
        # Encode
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self._device == "cuda"):
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            
        return EncoderOutput(
            embeddings=features.cpu().numpy(),
            embedding_dim=self._embedding_dim,
            model_name=self._model_name,
            device=self._device,
        )
        
    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embeddings: Shape (N, D)
            text_embeddings: Shape (M, D)
            
        Returns:
            Similarity matrix of shape (N, M)
        """
        # Normalize (should already be normalized, but ensure)
        image_norm = image_embeddings / np.linalg.norm(
            image_embeddings, axis=1, keepdims=True
        )
        text_norm = text_embeddings / np.linalg.norm(
            text_embeddings, axis=1, keepdims=True
        )
        
        return image_norm @ text_norm.T
        
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
        
    @property
    def model_name(self) -> str:
        return f"{self._model_name}_{self._pretrained}"
        
    @property
    def device(self) -> str:
        return self._device
        
    def to(self, device: str) -> "CLIPEncoder":
        """Move model to different device."""
        self._device = device
        self._model = self._model.to(device)
        return self

