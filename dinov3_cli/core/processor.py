from typing import Literal

import numpy as np
import torch
from PIL import Image


class FeatureExtractor:
    """Core domain logic for DINOv3 feature extraction."""

    def __init__(self, model, processor, device: str = "cpu"):
        self.model = model
        self.processor = processor
        self.device = device

    @torch.inference_mode()
    def extract(self, image: Image.Image, pool: Literal["none", "cls", "mean"] = "none") -> np.ndarray:
        """
        Extracts features from an image.

        Args:
            image: PIL Image
            pool: "none" returns patch features (1, num_patches, hidden_size),
                  excluding CLS and register tokens.
                  "cls" returns the CLS token embedding (1, hidden_size).
                  "mean" returns the average of the patch features (1, hidden_size).
        """
        if pool not in ("none", "cls", "mean"):
            raise ValueError(f"Invalid pool {pool!r}; expected 'none', 'cls' or 'mean'")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # DINOv3 output structure (already layer-normed by the HF model):
        # [batch_size, 1 (CLS) + num_register_tokens + num_patches, hidden_size]
        # ConvNeXt variants have no registers; their token 0 is a global-average-pooled token.
        last_hidden_state = outputs.last_hidden_state

        if pool == "cls":
            return outputs.pooler_output.cpu().numpy()

        num_register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        patch_tokens = last_hidden_state[:, 1 + num_register_tokens :, :]

        if pool == "mean":
            return patch_tokens.mean(dim=1).cpu().numpy()

        return patch_tokens.cpu().numpy()
