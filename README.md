# DINOv3 CLI

A command-line interface for DINOv3 vision foundation models by Meta AI.

## Overview

DINOv3 is a family of versatile vision foundation models producing high-quality dense features and achieving outstanding performance on various vision tasks, including outperforming specialized state-of-the-art models across a broad range of settings, without fine-tuning.

This CLI provides easy access to DINOv3's capabilities:

- **Feature Extraction** - Extract dense features from images
- **Depth Estimation** - Monocular depth estimation (NYUv2)
- **Semantic Segmentation** - ADE20K semantic segmentation
- **Object Detection** - COCO2017 object detection
- **Image Similarity** - PCA-based image similarity
- **Canopy Height Maps** - CHMv2 canopy height estimation

## Quick Start

```bash
# Install
pip install dinov3-cli

# Extract features from an image
dinov3 extract image.jpg --pool

# Estimate depth
dinov3 depth image.jpg --output depth.png

# Semantic segmentation
dinov3 segment image.jpg --output segmentation.png
```

## Documentation

- [Architecture](ARCHITECTURE.md) - Project structure and design decisions
- [CLI Usage](CLI.md) - Detailed CLI command reference
- [Setup](SETUP.md) - Development setup and installation

## Available Models

| Model | Parameters | Dataset |
|-------|------------|---------|
| ViT-S/16 distilled | 21M | LVD-1689M |
| ViT-S+/16 distilled | 29M | LVD-1689M |
| ViT-B/16 distilled | 86M | LVD-1689M |
| ViT-L/16 distilled | 300M | LVD-1689M |
| ViT-H+/16 distilled | 840M | LVD-1689M |
| ViT-7B/16 | 6,716M | LVD-1689M |

## License

DINOv3 code and model weights are released under the DINOv3 License. See [LICENSE.md](../LICENSE.md) for details.