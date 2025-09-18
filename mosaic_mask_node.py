"""Mosaic Mask Node for ComfyUI
================================
Detects mosaic/pixelation grids in an input image and outputs a **MASK** tensor
(N×1×H×W, float32 0–1) suitable for downstream nodes such as MaskAreaCondition
and Inpaint (using Model).

Upstream idea: https://github.com/summer4an/mosaic_detector
"""

from __future__ import annotations

from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image


# -----------------------------------------------------------------------------
# Grid templates generated once (based on summer4an/mosaic_detector idea)
# -----------------------------------------------------------------------------
def _gen_templates() -> list[np.ndarray]:
    out = []
    for size in range(11, 21):
        s = 2 + size + size - 1 + 2
        t = np.full((s, s), 255, np.uint8)
        step = size - 1
        for p in range(2, s, step):
            t[:, p] = 0
            t[p, :] = 0
        out.append(t)
    return out


TEMPLATES = _gen_templates()


# -----------------------------------------------------------------------------
# Helper conversions
# -----------------------------------------------------------------------------
ImageLike = Union[np.ndarray, torch.Tensor, Image.Image]


def _to_pil(img: ImageLike) -> Image.Image:
    """Convert ComfyUI IMAGE variants to PIL RGB (handles batch, CHW/HWC)."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    # If batch (N,C,H,W) take first frame for per-frame processing
    if img.ndim == 4 and img.shape[0] >= 1:
        img = img[0]

    # CHW ⇒ HWC
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)

    # float ⇒ uint8
    if img.dtype.kind == "f":
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8, copy=False)

    return Image.fromarray(img, mode="RGB")


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    """PIL L mask ⇒ torch.FloatTensor (1×H×W) in 0–1."""
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # 1×H×W


# -----------------------------------------------------------------------------
# Node definition
# -----------------------------------------------------------------------------
class MosaicToMask:
    @classmethod
    def LABEL(cls):
        return "Mosaic → Mask"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                "threshold": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05})
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "detect"
    CATEGORY = "Image/Masks"

    def detect(self, images: ImageLike, threshold: float = 0.30):
        """Return a batched mask tensor N×1×H×W (float32)."""
        # Normalize input to a list of frames for per-frame processing.
        if isinstance(images, list):
            frames = images
        elif isinstance(images, torch.Tensor) and images.ndim == 4:
            frames = list(images)  # split batch dim
        else:
            frames = [images]

        masks = []
        for frame in frames:
            pil_img = _to_pil(frame)
            mask_pil = self._process_single(pil_img, threshold)
            masks.append(_mask_to_tensor(mask_pil))

        return (torch.stack(masks, dim=0),)  # N×1×H×W

    @staticmethod
    def _process_single(img: Image.Image, thr: float) -> Image.Image:
        rgb = np.asarray(img.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 10, 20)
        inv_edges = 255 - edges
        blurred = cv2.GaussianBlur(inv_edges, (3, 3), 0)

        mask = np.zeros_like(gray, np.uint8)
        for templ in TEMPLATES:
            res = cv2.matchTemplate(blurred, templ, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= thr)
            h, w = templ.shape
            for pt in zip(*loc[::-1]):
                cv2.rectangle(mask, pt, (pt[0] + w, pt[1] + h), 255, -1)

        return Image.fromarray(mask, mode="L")


NODE_CLASS_MAPPINGS = {"MosaicToMask": MosaicToMask}
NODE_DISPLAY_NAME_MAPPINGS = {"MosaicToMask": "Mosaic → Mask"}
