# Development Setup

## Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- Git

## Clone & Initialize

```bash
git clone https://github.com/yourusername/dinov3-cli.git
cd dinov3-cli
uv sync
```

## Environment

Copy `.env.example` to `.env` and fill in required values:

```bash
cp .env.example .env
```

Required variables:
- `HF_TOKEN` - Hugging Face access token (required for gated models)

## Install for Development

```bash
# Install in editable mode
uv pip install -e .

# Or with dev dependencies
uv pip install -e ".[dev]"
```

## Running the CLI

```bash
# Local development
uv run dinov3 --help

# Or use python module directly
uv run python -m dinov3_cli --help
```

## Project Structure

```
dinov3-cli/
├── src/
│   └── dinov3_cli/
│       ├── __init__.py
│       ├── __main__.py           # Entry point
│       ├── core/                 # Domain logic
│       ├── commands/             # CLI commands
│       └── cli.py                # App setup
├── tests/
│   ├── commands/
│   └── core/
├── docs/
├── pyproject.toml
└── README.md
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/dinov3_cli --cov-report=html

# Run specific test file
uv run pytest tests/commands/test_extract.py
```

## Code Quality

```bash
# Format code
uv run ruff format src/

# Lint
uv run ruff check src/

# Type check
uv run mypy src/
```

## Building

```bash
# Build distribution
uv build

# Build and upload to PyPI
uv publish
```

## Common Issues

### `HF_TOKEN` Required

Some DINOv3 models require accepting terms on Hugging Face. Set your token:

```bash
export HF_TOKEN="your_token_here"
```

Or create `~/.netrc`:
```
machine huggingface.co
login your_username
password your_token
```

### CUDA Out of Memory

For large models, reduce batch size or use `--device cpu`:
```bash
dinov3 extract image.jpg --device cpu
```

### Model Not Found

Models are cached to `~/.cache/huggingface/`. To clear cache:
```bash
rm -rf ~/.cache/huggingface/hub/
```

## Dependencies

### Runtime
- `typer` - CLI framework
- `rich` - Terminal output
- `transformers` - Model loading
- `torch` - PyTorch backend
- `pydantic-settings` - Config management

### Development
- `pytest` - Testing
- `pytest-cov` - Coverage
- `ruff` - Linting/formatting
- `mypy` - Type checking

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag
4. Build and publish:
```bash
uv build
uv publish
```