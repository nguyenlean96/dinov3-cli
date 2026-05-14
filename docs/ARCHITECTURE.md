# Architecture

## Overview

This project follows a **layered + feature-based architecture** inspired by Clean Architecture principles, adapted for Python's ecosystem conventions.

## Project Structure

```
dinov3-cli/
├── src/
│   └── dinov3_cli/
│       ├── __init__.py
│       ├── __main__.py           # Entry point: python -m dinov3_cli
│       ├── core/                 # Domain layer: model loading, processing
│       │   ├── __init__.py
│       │   ├── models.py          # Model abstractions
│       │   └── processor.py      # Image processor abstractions
│       ├── commands/             # Feature-based commands
│       │   ├── __init__.py
│       │   ├── extract.py        # Feature extraction command
│       │   ├── depth.py          # Depth estimation command
│       │   ├── segment.py        # Segmentation command
│       │   ├── detect.py         # Object detection command
│       │   └── similarity.py     # Image similarity command
│       └── cli.py                # Typer app setup
├── tests/
│   ├── commands/
│   └── core/
├── pyproject.toml
└── README.md
```

## Layer Responsibilities

### `core/` - Domain Layer

The core layer contains pure domain logic with **no external dependencies** (except PyTorch/transformers for type hints).

```
core/
├── models.py        # AutoModel, AutoImageProcessor abstractions
└── processor.py    # Preprocessing logic
```

**Principles:**
- No CLI framework imports
- No HTTP clients
- Business logic only
- Easily testable in isolation

### `commands/` - Interface Layer

Feature-based commands that expose functionality via CLI. Each command is self-contained.

```
commands/
├── extract.py      # image-feature-extraction pipeline
├── depth.py        # Depth estimation with DPT head
├── segment.py      # Semantic segmentation with M2F head
├── detect.py       # Object detection
└── similarity.py  # PCA-based similarity
```

**Principles:**
- One command per file
- Depends on `core/` for logic
- Handles user I/O (files, images, output)
- Uses Typer decorators for CLI definition

### `cli.py` - Application Layer

Orchestrates all commands into a single Typer application.

## Design Decisions

### Why `src/` Layout?

The `src/` layout is the **2025/2026 industry standard** for Python projects:

1. **Prevents import bugs** - Tests always import the installed package, not local modules
2. **Clear separation** - Source code is clearly distinguished from config/docs/tests
3. **CI/CD friendly** - Build and test without installing
4. **PEP 621 compliant** - Works with all modern packaging tools (uv, poetry, hatch)

### Why Layered + Feature-Based?

| Approach | Pros | Cons |
|----------|------|------|
| Pure feature-based | Fast navigation | Layers get mixed |
| Pure layered | Clean separation | Deep nesting for features |
| **Hybrid** | Best of both | Slightly more structure |

The hybrid approach groups related functionality (commands) while maintaining layer separation (core is independent).

### Technology Choices

| Component | Choice | Rationale |
|-----------|--------|----------|
| CLI Framework | **Typer** | Type hints, async support, built on Click |
| Output Formatting | **Rich** | Beautiful CLI output, progress bars |
| Config Management | **pydantic-settings** | Type-safe, environment-based |
| Model Loading | **transformers** | Official DINOv3 support |
| Package Manager | **uv** | Fast, modern, PEP 621 native |

### Comparison with Rust Architecture

If you're coming from Rust, here's the mapping:

| Rust Concept | Python Equivalent |
|--------------|-------------------|
| `src/domain/` | `core/` - Pure domain logic |
| `src/commands/` | `commands/` - CLI handlers |
| `src/application/` | `cli.py` - Orchestration |
| `src/infrastructure/` | `core/` (for now) - Model loading |
| Feature modules | `commands/extract.py` etc. |

## Extension Points

### Adding a New Command

1. Create `commands/new_feature.py`:
```python
import typer
from typing import Optional
from pathlib import Path

from dinov3_cli.core import ModelLoader

def new_feature(
    image: Path = typer.Argument(..., help="Input image"),
    output: Optional[Path] = typer.Option(None, "-o", "--output"),
):
    """Description shown in help."""
    model = ModelLoader.load("facebook/dinov3-vitl16-pretrain-lvd1689m")
    # ... implementation
```

2. Register in `cli.py`:
```python
from dinov3_cli.commands.new_feature import new_feature

app.command()(new_feature)
```

### Adding a New Model Variant

Update `core/models.py` with new model registry.

## Testing Strategy

```
tests/
├── commands/           # Mirror commands/ structure
│   ├── test_extract.py
│   └── test_depth.py
└── core/
    └── test_models.py
```

- **Unit tests** for `core/` - no external I/O
- **Integration tests** for commands - real model loading (slow)
- Use `pytest` with `pytest-mock` for fixtures