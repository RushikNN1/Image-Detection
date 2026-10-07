"""Pretrained M3 image tampering detector.

The model is downloaded from Hugging Face on the first call and cached locally.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Union

ImageInput = Union[str, Path, bytes, bytearray, Any]

MODEL_REPO = "salmanzaman777/image-forgery-m3"
MODEL_FILE = "M3_best.keras"
MODEL_REVISION = "v1"
IMAGE_SIZE = (224, 224)

_MODEL = None
_TF = None


def _dependencies():
    """Import runtime dependencies only when prediction is requested."""
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    try:
        import numpy as np
        import tensorflow as tf
        from huggingface_hub import hf_hub_download
        from PIL import Image, ImageChops, ImageEnhance
    except ImportError as exc:
        raise ImportError(
            "AI detector dependencies are missing. Install them with: "
            "pip3 install tensorflow keras huggingface_hub"
        ) from exc
    return np, tf, hf_hub_download, Image, ImageChops, ImageEnhance


def _open_image(image: ImageInput, Image):
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError("image must be a file path, bytes, or PIL.Image.Image")


def _load_model():
    global _MODEL, _TF
    if _MODEL is not None:
        return _MODEL, _TF

    np, tf, hf_hub_download, *_ = _dependencies()
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            revision=MODEL_REVISION,
        )
        import keras

        _MODEL = keras.models.load_model(model_path, compile=False)
        _TF = tf
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the M3 model from Hugging Face "
            f"({MODEL_REPO}, revision {MODEL_REVISION}). Check internet access, "
            "then retry. If this persists, use the project's ELA/statistical fallback."
        ) from exc
    return _MODEL, _TF


def _ela_input(image, tf, Image, ImageChops, ImageEnhance):
    """Build the M3 model's documented JPEG Error Level Analysis tensor."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    difference = ImageChops.difference(image, recompressed)
    ela = ImageEnhance.Brightness(difference).enhance(15)

    ela_buffer = io.BytesIO()
    ela.save(ela_buffer, format="JPEG", quality=75)
    ela_buffer.seek(0)
    encoded = tf.io.decode_jpeg(ela_buffer.getvalue(), channels=3)
    resized = tf.image.resize(encoded, IMAGE_SIZE)
    return (tf.cast(resized, tf.float32) / 255.0).numpy()[None, ...]


def predict_image(image: ImageInput) -> dict[str, Union[str, int]]:
    """Predict whether an image is authentic or tampered.

    Returns {"label": "tampered"|"real", "confidence": 0..100}.
    """
    np, tf, _download, Image, ImageChops, ImageEnhance = _dependencies()
    pil_image = _open_image(image, Image)
    rgb_image = pil_image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    rgb = np.asarray(rgb_image, dtype=np.float32)[None, ...] / 255.0
    ela = _ela_input(pil_image, tf, Image, ImageChops, ImageEnhance)

    model, _ = _load_model()
    output = np.asarray(model.predict([rgb, ela], verbose=0)).reshape(-1)
    if output.size != 1 or not np.isfinite(output[0]):
        raise RuntimeError(f"Unexpected model output: {output!r}")
    forged_probability = float(np.clip(output[0], 0.0, 1.0))
    if forged_probability >= 0.5:
        label = "tampered"
        confidence = forged_probability
    else:
        label = "real"
        confidence = 1.0 - forged_probability
    return {"label": label, "confidence": int(round(confidence * 100))}
