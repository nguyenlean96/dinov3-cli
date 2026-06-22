import typer
from rich.console import Console
from typing import Optional

# Import commands
from dinov3_cli.commands.extract import extract as extract_cmd
from dinov3_cli.commands.api import api as api_cli

console = Console()

# Initialize Typer with Rich markup mode
app = typer.Typer(
    name="dinov3",
    help="DINOv3 Command Line Interface",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Register subcommands
# Extract command uses callback directly, so we can wrap it as a command here
app.command(name="extract", help="Extract dense features from images using DINOv3 backbones.")(extract_cmd)
app.command(name="api", help="Start API server")(api_cli)


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", help="Show version and exit."),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose output."),
    device: Optional[str] = typer.Option(None, help="Global device override."),
):
    """
    [bold blue]DINOv3 CLI[/bold blue]

    A command-line tool for feature extraction and dense prediction tasks using DINOv3.
    """
    if version:
        console.print("[bold cyan]dinov3-cli version 0.1.0[/bold cyan]")
        raise typer.Exit()

    # In a full implementation, you'd store `verbose` and `device` in a global context or config.
    # For now, commands accept their own `--device` flag as well.


if __name__ == "__main__":
    app()
