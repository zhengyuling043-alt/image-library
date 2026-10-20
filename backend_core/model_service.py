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


def encode_image_paths(paths, model, preprocess, batch_size=None):
    """Batch-encode images while isolating unreadable files.

    Returns ``(vectors, failures)`` keyed by the original ``Path`` objects.
    The bounded batch avoids loading a whole library into memory and provides
    the high-throughput path used by incremental import.
    """

    import torch

    paths = [Path(path) for path in paths]
    if batch_size is None:
        default = 32 if str(_model_device(model)).startswith("cuda") else 8
        batch_size = int(os.environ.get("IMAGE_LIBRARY_BATCH_SIZE", default))
    batch_size = max(1, int(batch_size))
    vectors, failures = {}, {}

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        tensors, valid_paths = [], []
        for path in batch_paths:
            try:
                with Image.open(path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    tensors.append(preprocess(image))
                valid_paths.append(path)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                failures[path] = str(exc)
        if not tensors:
            continue

        tensor = torch.stack(tensors).to(_model_device(model))
        try:
            with torch.inference_mode():
                encoded = model.encode_image(tensor)
                norms = encoded.norm(dim=-1, keepdim=True)
                if torch.any(~torch.isfinite(encoded)) or torch.any(norms <= 0):
                    raise RuntimeError("模型返回了无效图片向量")
                encoded = encoded / norms
        except RuntimeError as exc:
            # A batch-level model error must not incorrectly mark every file as
            # successful. Fall back to one-by-one inference to identify which
            # image failed; this also reduces peak memory after a CUDA OOM.
            if str(_model_device(model)).startswith("cuda"):
                torch.cuda.empty_cache()
            for path in valid_paths:
                try:
                    vectors[path] = encode_image_path(path, model, preprocess)
                except (OSError, ValueError, RuntimeError) as single_error:
                    failures[path] = str(single_error)
            continue

        array = encoded.detach().cpu().numpy().astype(np.float32, copy=False)
        for path, vector in zip(valid_paths, array):
            if vector.ndim != 1 or not np.isfinite(vector).all():
                failures[path] = "图片向量格式无效"
            else:
                vectors[path] = vector
    return vectors, failures

