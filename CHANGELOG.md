# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-21

### Added

- PyTorch CUDA 12 support via custom wheel index (`pytorch-cu132`)
- Extract command with improved model handling
- PyInstaller bootstrap for standalone executable (`bootstrap.py`, `dinov3_cli.spec`)
- GitHub Actions CI/CD pipeline
- High-level architecture documentation (`docs/`)
- Unit tests for extract command
- Initial project setup with CLI framework (Typer + Rich)

### Changed

- Restructured from `src/` to `dinov3_cli/` package layout
- Migrated to `pyproject.toml`-based setuptools configuration
