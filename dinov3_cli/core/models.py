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

        load_kwargs = {}
        processor_kwargs = {}

        if cache_dir != None:
            load_kwargs["cache_dir"] = cache_dir
            processor_kwargs["cache_dir"] = cache_dir

        model_path = Path(model_name)
        resolved_path = str(model_path.resolve())

        # Check if loading from local path
        if model_path.exists() and model_path.is_dir():
            load_kwargs["local_files_only"] = True
            load_kwargs["trust_remote_code"] = True  # Required for DINOv3's custom architecture
            processor_kwargs["local_files_only"] = True

        processor = AutoImageProcessor.from_pretrained(resolved_path, **processor_kwargs)
        model = AutoModel.from_pretrained(
            resolved_path,
            **load_kwargs,
        ).to(device)

        model.eval()
        return model, processor
