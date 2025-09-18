# Mosaic → Mask (ComfyUI custom node)

A lightweight ComfyUI node that detects mosaic/pixelation grids (TV-style censorship)
and outputs a binary **MASK** with the same resolution — perfect as an automatic mask
generator before inpainting.

- Upstream idea: https://github.com/summer4an/mosaic_detector
- Output type: `MASK` (batched tensor `N × 1 × H × W`, float 0–1)
- Works with single images, lists, and batched tensors (e.g., video frames)

## Installation

**Via ComfyUI Manager**
1. Open **Manager → Install → Add Repo** and paste this repository URL.
2. Click **Install dependencies**.
3. Restart ComfyUI.

**Manual (git)**
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/ComfyUI-Mosaic-Mask-Node.git
pip install -r ComfyUI/custom_nodes/ComfyUI-Mosaic-Mask-Node/requirements.txt
```
Restart ComfyUI. The node appears under **Image / Masks → “Mosaic → Mask”**.

## Usage

| Input | Type  | Notes |
|------:|:-----:|------|
| `images` | IMAGE | single frame, list, or batched tensor (B×C×H×W) |
| `threshold` | FLOAT | 0.05–1.0, lower = more sensitive; default **0.30** |

**Output:** `MASK` (`N × 1 × H × W`) — wire directly into *MaskAreaCondition* or inpainting nodes.

### Minimal example (image)
`examples/minimal_image_mask.json`:
```
LoadImage → Mosaic → Mask → PreviewImage
```

### Video tip
For VRAM-friendly video, iterate per-frame (Impact Pack **ForEach** or VHS Batch Manager) and combine at the end.

## Notes
- Accepts ComfyUI tensors (BCHW/NCHW), numpy arrays, and PIL images.
- Returns a single **batched** mask tensor (not a Python list), so downstream nodes like
  *MaskAreaCondition* work out-of-the-box.

## License
This project is licensed under the MIT License (see `LICENSE`).
It adapts the detection idea from https://github.com/summer4an/mosaic_detector; please see that project’s license and attribution.
