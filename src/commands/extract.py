import json
from pathlib import Path
from typing import Optional
import typer
import numpy as np
from PIL import Image
from rich.console import Console

import sys

from src.core.models import ModelLoader
from src.core.processor import FeatureExtractor

console = Console()

app = typer.Typer(help="Extract dense features from images using DINOv3.")


@app.callback(invoke_without_command=True)
def extract(
    image: Path = typer.Argument(
        ..., help="Path to input image file", exists=True, file_okay=True, dir_okay=False, readable=True
    ),
    model: str = typer.Option("facebook/dinov3-vitl16-pretrain-lvd1689m", help="Model name or path"),
    pool: bool = typer.Option(False, "--pool/--no-pool", help="Whether to pool features into a single embedding"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file (.npy or .json). Defaults to stdout as JSON if not provided."
    ),
    device: str = typer.Option("cpu", help="Device to use (cpu, cuda, mps)"),
):
    """
    Extract dense features from images using DINOv3 vision backbones.
    """
    try:
        # Load Model
        with console.status(f"[bold green]Loading model '{model}' to {device}..."):
            hf_model, processor = ModelLoader.load(model_name=model, device=device)

        # Load Image (referencing prototype: must convert to RGB)
        img = Image.open(image).convert("RGB")

        # Extract Features
        with console.status("[bold cyan]Extracting features..."):
            extractor = FeatureExtractor(model=hf_model, processor=processor, device=device)
            features = extractor.extract(image=img, pool=pool)

        # Handle Output
        if output:
            if output.suffix == ".npy":
                np.save(output, features)
                console.print(
                    f"[bold green]✓[/bold green] Features saved to [cyan]{output}[/cyan] with shape {features.shape}"
                )
            elif output.suffix == ".json":
                with open(output, "w") as f:
                    json.dump(features.tolist(), f)
                console.print(
                    f"[bold green]✓[/bold green] Features saved to [cyan]{output}[/cyan] with shape {features.shape}"
                )
            else:
                console.print(
                    f"[bold yellow]Warning:[/bold yellow] Unknown extension '{output.suffix}'. Defaulting to numpy format."
                )
                np.save(output, features)
                console.print(
                    f"[bold green]✓[/bold green] Features saved to [cyan]{output}[/cyan] with shape {features.shape}"
                )
        else:
            # Output to stdout as JSON
            # Note: For non-pooled dense features, this can be extremely large.
            sys.stdout.write(json.dumps(features.tolist()))
            sys.stdout.write("\n")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
