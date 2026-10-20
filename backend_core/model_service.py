"""OpenCLIP deployment and image-encoding service.

This module owns device selection, project-local weight caching, trusted weight
loading and normalized image embedding.  It deliberately has no dependency on
the command-line or HTTP layers, so the same implementation is used by index
building, incremental import and query search.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


VALID_DEVICES = {"auto", "cpu", "cuda"}


def select_device(requested=None, cuda_available=None):
    """Resolve the inference device with an explicit, testable policy."""

    if requested is None:
        requested = os.environ.get("IMAGE_LIBRARY_DEVICE", "auto")
    requested = str(requested).strip().lower()
    if requested not in VALID_DEVICES:
        raise RuntimeError("IMAGE_LIBRARY_DEVICE 只能是 auto、cpu 或 cuda")

    if cuda_available is None:
        import torch

        cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("已要求使用 CUDA，但当前 PyTorch 未检测到可用 GPU")
    if requested == "cuda" or (requested == "auto" and cuda_available):
        return "cuda"
    return "cpu"


def resolve_weight_source(model_name, weights, cache_dir):
    """Return a verified OpenCLIP weight path, downloading into the D-drive cache."""

    from open_clip.pretrained import download_pretrained_from_url, get_pretrained_url

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = get_pretrained_url(model_name, weights)
    if not url:
        return weights
    return download_pretrained_from_url(url, cache_dir=str(cache_dir.resolve()))


def load_openclip_model(model_name, weights, cache_dir, requested_device=None):
    """Load a trusted OpenCLIP checkpoint and its preprocessing pipeline."""

    import open_clip

    cache_dir = Path(cache_dir)
    weight_source = resolve_weight_source(model_name, weights, cache_dir)
    device = select_device(requested_device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=weight_source,
        device=device,
        cache_dir=str(cache_dir.resolve()),
        # Official OpenAI .pt files are TorchScript archives.  Modern PyTorch
        # otherwise applies weights_only=True and rejects this trusted format.
        weights_only=False,
    )
    model.eval()
    return model, preprocess


def _model_device(model):
    """Read the model device without assuming a specific OpenCLIP class."""

    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return "cpu"


def encode_image_path(path, model, preprocess):
    """Encode one image as a finite, L2-normalized float32 vector."""

    import torch

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = preprocess(image).unsqueeze(0).to(_model_device(model))

    with torch.inference_mode():
        vector = model.encode_image(tensor)
        norm = vector.norm(dim=-1, keepdim=True)
        if torch.any(~torch.isfinite(vector)) or torch.any(norm <= 0):
            raise RuntimeError("模型返回了无效图片向量")
        vector = vector / norm

    result = vector[0].detach().cpu().numpy().astype(np.float32, copy=False)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise RuntimeError("图片向量格式无效")
    return result

