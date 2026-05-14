# CLI Usage

## Commands

### `extract` - Feature Extraction

Extract dense features from images using DINOv3 vision backbones.

```bash
dinov3 extract IMAGE [OPTIONS]
```

**Arguments:**
- `IMAGE` - Path to input image file

**Options:**
- `--model TEXT` - Model name (default: `facebook/dinov3-vitl16-pretrain-lvd1689m`)
- `--pool / --no-pool` - Whether to pool features (default: no-pool)
- `-o, --output PATH` - Output file (default: stdout as JSON)
- `--device TEXT` - Device to use (`cpu`, `cuda`, `mps`)

**Examples:**

```bash
# Basic feature extraction
dinov3 extract image.jpg

# Extract with pooling
dinov3 extract image.jpg --pool

# Save to file
dinov3 extract image.jpg -o features.npy

# Use specific model
dinov3 extract image.jpg --model facebook/dinov3-vits16-pretrain-lvd1689m
```

**Output shape:**
- Without `--pool`: `(1, N_PATCHES, HIDDEN_SIZE)` - per-patch features
- With `--pool`: `(1, HIDDEN_SIZE)` - pooled image embedding

---

### `depth` - Depth Estimation

Estimate depth from images using DINOv3 + DPT heads.

```bash
dinov3 depth IMAGE [OPTIONS]
```

**Arguments:**
- `IMAGE` - Path to input image file

**Options:**
- `--model TEXT` - Model name (default: `facebook/dinov3-vitl16-pretrain-lvd1689m`)
- `-o, --output PATH` - Output file (default: `depth.png`)
- `--size INT` - Input size (default: 1024)
- `--device TEXT` - Device to use

**Examples:**

```bash
# Basic depth estimation
dinov3 depth image.jpg

# Custom output
dinov3 depth image.jpg -o my_depth.png

# Different input size
dinov3 depth image.jpg --size 768
```

---

### `segment` - Semantic Segmentation

Semantic segmentation on ADE20K using DINOv3 + M2F heads.

```bash
dinov3 segment IMAGE [OPTIONS]
```

**Arguments:**
- `IMAGE` - Path to input image file

**Options:**
- `--model TEXT` - Model name
- `-o, --output PATH` - Output file (default: `segmentation.png`)
- `--size INT` - Input size (default: 896)
- `--device TEXT` - Device to use

**Examples:**

```bash
# Basic segmentation
dinov3 segment image.jpg

# Custom output and size
dinov3 segment image.jpg -o seg.png --size 768
```

---

### `detect` - Object Detection

Object detection on COCO2017 using DINOv3 + detection heads.

```bash
dinov3 detect IMAGE [OPTIONS]
```

**Arguments:**
- `IMAGE` - Path to input image file

**Options:**
- `--model TEXT` - Model name
- `-o, --output PATH` - Output file (default: `detections.json`)
- `--device TEXT` - Device to use

**Examples:**

```bash
# Basic detection
dinov3 detect image.jpg

# Save as JSON
dinov3 detect image.jpg -o detections.json
```

---

### `similarity` - Image Similarity

Compute similarity between two images using PCA features.

```bash
dinov3 similarity IMAGE1 IMAGE2 [OPTIONS]
```

**Arguments:**
- `IMAGE1` - Path to first image
- `IMAGE2` - Path to second image

**Options:**
- `--model TEXT` - Model name
- `--metric TEXT` - Similarity metric (`cosine`, `euclidean`)
- `--device TEXT` - Device to use

**Examples:**

```bash
# Compare two images
dinov3 similarity image1.jpg image2.jpg

# Use cosine similarity
dinov3 similarity image1.jpg image2.jpg --metric cosine
```

---

## Global Options

These options apply to all commands:

- `--help` - Show help message
- `--version` - Show version
- `--verbose / --quiet` - Verbosity control
- `--device TEXT` - Default device for all operations

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token | Required for gated models |
| `DINOV3_CACHE_DIR` | Model cache directory | `~/.cache/huggingface/` |
| `DINOV3_DEVICE` | Default device | `cuda` if available |

## Configuration File

Create `~/.dinov3.toml` or `./.dinov3.toml`:

```toml
[default]
device = "cuda"
cache_dir = "~/.cache/huggingface/"

[extract]
default_model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
pool = false

[depth]
default_model = "facebook/dinov3-vitl16-chmv2-dpt-head"
default_size = 1024
```

## Output Formats

### Feature Extraction (JSON)
```json
{
  "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
  "shape": [1, 201, 1024],
  "features": [[0.123, -0.456, ...]],
  "pooled": false
}
```

### Depth Estimation (PNG)
- Single-channel PNG with depth values
- Use `--output-format float16` for raw float output

### Segmentation (PNG)
- Color-coded segmentation map
- Class labels in metadata JSON sidecar

### Detection (JSON)
```json
{
  "image": "image.jpg",
  "detections": [
    {"class": "cat", "bbox": [x, y, w, h], "score": 0.95}
  ]
}
```