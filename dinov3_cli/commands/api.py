import io
import asyncio
import base64
import logging
import torch
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import typer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from dinov3_cli.core.models import ModelLoader
from dinov3_cli.core.processor import FeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for request/response
class ExtractRequest(BaseModel):
    """Request model for feature extraction."""

    image: str = Field(..., description="Image file path or base64-encoded image data")
    model_name: Optional[str] = Field(default=None, description="HuggingFace model name or local path")
    pool: bool = Field(False, description="Whether to pool features into a single embedding")
    device: str = Field("cpu", description="Device to use (cpu, cuda, mps)")


class ExtractResponse(BaseModel):
    """Response model for feature extraction."""

    features: list[float] = Field(..., description="Extracted features as flattened list")
    shape: tuple[int, ...] = Field(..., description="Original shape of the features array")
    dtype: str = Field(..., description="Data type of the features")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    model_loaded: bool
    model_name: str | None


# Global state for model
_model = None
_processor = None
_current_model_name = None
_device = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load model once at startup."""
    global _model, _processor, _current_model_name, _device

    # Startup - model is loaded lazily on first request
    _model, _processor = ModelLoader.load(
        model_name=_current_model_name,
        device=_device,
    )
    
    yield
    # Shutdown cleanup
    _model = None
    _processor = None
    _current_model_name = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# FastAPI app with lifespan
app = FastAPI(
    title="DINOv3 API",
    description="DINOv3 Feature Extraction API Server",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=_model is not None,
        model_name=_current_model_name,
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract_features(request: ExtractRequest) -> ExtractResponse:
    """
    Extract features from an image.

    The model is loaded on the first request and cached for subsequent requests.
    """
    global _model, _processor, _current_model_name, _device

    # Lazy load model if not already loaded or if model changed
    if _model is None or (
        request.model_name is not None and _current_model_name != request.model_name
    ):
        logger.info(f"Loading model '{request.model_name}' on {request.device}")
        _model, _processor = ModelLoader.load(
            model_name=request.model_name,
            device=request.device,
        )
        _current_model_name = request.model_name
        _device = request.device

    # Decode image if base64
    try:
        if request.image.startswith("data:image"):
            # Handle data URL format: data:image/png;base64,xxxxx
            header, data = request.image.split(",", 1)
            img_data = base64.b64decode(data)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
        elif Path(request.image).exists():
            # Handle file path
            image = Image.open(request.image).convert("RGB")
        else:
            # Try as base64 directly
            img_data = base64.b64decode(request.image)
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    # Extract features in thread to avoid blocking
    def do_extract():
        extractor = FeatureExtractor(model=_model, processor=_processor, device=_device)
        return extractor.extract(image=image, pool=request.pool)

    features = await asyncio.to_thread(do_extract)

    return ExtractResponse(
        features=features.flatten().tolist(),
        shape=features.shape,
        dtype=str(features.dtype),
    )


# Typer CLI for running the server
cli = typer.Typer(help="Start DINOv3 API server")


@cli.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    model: Optional[str] = typer.Option(None, help="Model name or path"),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
):
    """Start the DINOv3 API server."""
    import uvicorn

    # Store device globally for lazy loading
    global _device, _current_model_name
    _device = device
    _current_model_name = model

    uvicorn.run(
        "dinov3_cli.commands.api:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    cli()