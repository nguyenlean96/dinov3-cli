import torch
from PIL import Image
import numpy as np


class FeatureExtractor:
    """Core domain logic for DINOv3 feature extraction."""

    def __init__(self, model, processor, device: str = "cpu"):
        self.model = model
        self.processor = processor
        self.device = device

    @torch.inference_mode()
    def extract(self, image: Image.Image, pool: bool = False) -> np.ndarray:
        """
        Extracts features from an image.

        Args:
            image: PIL Image
            pool: If True, returns CLS token embedding (1, hidden_size).
                  If False, returns patch features (1, num_patches, hidden_size),
                  excluding register tokens.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state

        # DINOv3 output structure:
        # [batch_size, 1 (CLS) + num_register_tokens + num_patches, hidden_size]

        if pool:
            # Return CLS token
            features = last_hidden_state[:, 0, :]
        else:
            # Return patch features (skipping CLS and register tokens)
            num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
            features = last_hidden_state[:, 1 + num_register_tokens :, :]

        return features.cpu().numpy()
