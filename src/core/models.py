import torch
from pathlib import Path
from typing import Optional, Union, Tuple
from transformers import AutoImageProcessor, AutoModel
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    """Domain layer abstraction for loading models."""
    
    @staticmethod
    def load(
        model_name: str,
        device: str = "cpu",
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[AutoModel, AutoImageProcessor]:
        """
        Loads the DINOv3 model and processor.
        
        Args:
            model_name: The Hugging Face hub ID or local path.
            device: 'cpu', 'cuda', 'mps' etc.
            cache_dir: Optional HF cache directory.
            
        Returns:
            Tuple of (model, processor)
        """
        logger.info(f"Loading model '{model_name}' on {device}")
        
        processor = AutoImageProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
        ).to(device)
        
        model.eval()
        return model, processor
