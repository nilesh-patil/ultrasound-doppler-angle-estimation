"""Grad-CAM interpretability for the trained image→angle pipeline.

A Grad-CAM heatmap over the **last convolutional feature maps** of the trained
pipeline's backbone, showing which pixels drive the Doppler-angle estimate — the
thesis's evidence that the model attends to the carotid walls rather than speckle or
annotation burn-in.

With ``A_k`` the last conv activations and ``g_k = d(angle)/d A_k``, the channel
weight is ``a_k = GAP(g_k)`` (global-average-pool the gradient over ``h, w``) and the
CAM is ``ReLU(Σ_k a_k · A_k)``, resized to the image and min–max normalized into
``[0, 1]``. Two subtleties:

1. The conv activations live **inside the backbone sub-model** (``model`` is
   ``Input → Flatten(backbone) → head``), so the explained layer is fetched from the
   backbone, not the top-level model.
2. The explained scalar is the **angle output** — for a 1-wide raw target it is the
   single output unit; for a 2-wide ``(sin 2θ, cos 2θ)`` target it is the decoded
   angle ``0.5·atan2(sin, cos)`` (differentiable, a single well-defined reduction) —
   so ``d(angle)/d A`` is well-defined.

Activation gradients require :class:`tf.GradientTape`, which exists **only** on the
TensorFlow backend, so :func:`grad_cam` and :func:`cam_figure` run under
``pixi run -e mac-gpu python ...`` (``KERAS_BACKEND=tensorflow``). The computation is
deliberately light — one frozen-backbone forward and one conv-layer gradient for a
single image — so there is no fine-tune and no OOM. :func:`overlay` and the heatmap
:func:`_normalize` math are **pure numpy/matplotlib** and need no backend.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible (matches uda.figures)
import matplotlib.cm as cm  # noqa: E402
import numpy as np  # noqa: E402

__all__ = ["grad_cam", "overlay", "cam_figure"]


# --------------------------------------------------------------------------- #
# Pure-numpy CAM helpers (no Keras / no backend).
# --------------------------------------------------------------------------- #
def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min–max normalize ``arr`` into ``[0, 1]``; a constant array maps to all-zeros.

    The ``max == min`` branch returns zeros instead of dividing by zero, so a flat
    CAM (e.g. a dead/zero-gradient model) yields a defined, finite heatmap.
    """
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _to_gray3(image: np.ndarray) -> np.ndarray:
    """Rescale a background frame to ``[0, 1]`` and broadcast to ``(H, W, 3)``."""
    img = np.asarray(image, dtype=np.float32)
    if img.ndim == 3 and img.shape[-1] == 3:
        gray = img.mean(axis=-1)
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(f"image must be (H, W) or (H, W, 3); got shape {img.shape}")
    lo, hi = float(gray.min()), float(gray.max())
    gray = (gray - lo) / (hi - lo) if hi > lo else np.zeros_like(gray)
    return np.repeat(gray[..., np.newaxis].astype(np.float32), 3, axis=-1)


def overlay(image: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.4) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto a grayscale image as an RGB overlay.

    Renders ``heatmap`` (``[0, 1]``) through the ``jet`` colormap and alpha-blends it
    over the grayscale ``image`` (``(H, W)`` or ``(H, W, 3)``), returning an RGB array
    in ``[0, 1]``. **Pure numpy/matplotlib** — no Keras/backend.

    Parameters
    ----------
    image : numpy.ndarray
        Background frame, ``(H, W)`` or ``(H, W, 3)``, any real dtype (rescaled to
        ``[0, 1]`` for display).
    heatmap : numpy.ndarray
        ``(H, W)`` heatmap in ``[0, 1]`` (resized to ``image`` if needed).
    alpha : float, keyword-only
        Blend weight of the jet heatmap over the grayscale base (default ``0.4``).

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` ``float`` RGB overlay in ``[0, 1]``.
    """
    gray3 = _to_gray3(image)
    h, w = gray3.shape[:2]
    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.shape[:2] != (h, w):
        heat = _resize_np(heat, (h, w))
    jet = cm.get_cmap("jet")(np.clip(heat, 0.0, 1.0))[..., :3].astype(np.float32)
    blended = (1.0 - alpha) * gray3 + alpha * jet
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def _resize_np(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear-resize a 2-D ``(h, w)`` array to ``size`` with numpy (no backend).

    Used only by :func:`overlay` (the pure-numpy path); :func:`grad_cam` resizes the
    raw CAM with ``tf.image.resize`` on the TF backend before normalizing.
    """
    arr = np.asarray(arr, dtype=np.float32)
    sh, sw = arr.shape
    th, tw = size
    if (sh, sw) == (th, tw):
        return arr
    ys = np.linspace(0, sh - 1, th)
    xs = np.linspace(0, sw - 1, tw)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    x1 = np.minimum(x0 + 1, sw - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    top = arr[y0][:, x0] * (1 - wx) + arr[y0][:, x1] * wx
    bot = arr[y1][:, x0] * (1 - wx) + arr[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


# --------------------------------------------------------------------------- #
# Grad-CAM (TF backend only — tf.GradientTape).
# --------------------------------------------------------------------------- #
def _find_backbone(model):
    """Return the backbone sub-model (the ``Functional`` named ``*_features``)."""
    import keras

    for layer in model.layers:
        if isinstance(layer, keras.Model) and layer.name.endswith("_features"):
            return layer
    # Fallback: the first nested Functional sub-model.
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError(
        "grad_cam expects an Input → Flatten(backbone) → head model with a nested "
        "backbone sub-model; none was found in model.layers"
    )


def _autopick_last_conv(backbone) -> str:
    """Auto-pick the **last 4-D conv activation** in the backbone sub-model.

    Walks the backbone's layers in reverse and returns the name of the last layer
    whose output is 4-D ``(B, h, w, c)`` *and* is a convolutional layer (its class
    name contains ``"Conv"`` — Conv2D / SeparableConv2D / DepthwiseConv2D). Pooling
    or flatten layers (which can also be 4-D) are skipped so the explained map is a
    genuine conv activation. Falls back to the last 4-D output if no conv is found.
    """
    last_conv = None
    last_4d = None
    for layer in reversed(backbone.layers):
        out = getattr(layer, "output", None)
        shape = getattr(out, "shape", None)
        if shape is None or len(shape) != 4:
            continue
        if last_4d is None:
            last_4d = layer.name
        if "Conv" in type(layer).__name__:
            last_conv = layer.name
            break
    name = last_conv if last_conv is not None else last_4d
    if name is None:
        raise ValueError("no 4-D conv activation found in the backbone sub-model")
    return name


def _post_backbone_forward(model, backbone, features):
    """Apply the top-level layers *after* the backbone to its features → angle output.

    The pipeline is ``Input → backbone → Flatten → head``; ``features`` is the
    backbone's output tensor, and this re-applies the remaining top-level layers (the
    ``Flatten`` and the ``head`` sub-model) in order so the angle output is rebuilt
    inside the gradient tape.
    """
    x = features
    seen_backbone = False
    for layer in model.layers:
        if layer is backbone:
            seen_backbone = True
            continue
        if not seen_backbone:
            continue  # skip the Input layer(s) ahead of the backbone
        x = layer(x)
    return x


def _angle_scalar(y):
    """Reduce the encoded output to a single differentiable angle scalar per sample.

    * 1-wide (raw degrees): the single output unit ``y[:, 0]``.
    * 2-wide ``(sin 2θ, cos 2θ)``: the decoded angle ``0.5·atan2(sin, cos)`` (in
      radians; the constant scale is irrelevant to the CAM's *direction*).
    """
    import tensorflow as tf

    width = int(y.shape[-1])
    if width == 1:
        return y[:, 0]
    sin = y[:, 0]
    cos = y[:, 1]
    return 0.5 * tf.atan2(sin, cos)


def grad_cam(model, image: np.ndarray, *, last_conv_layer: str | None = None) -> np.ndarray:
    """Grad-CAM heatmap for the angle output over the backbone's last conv maps.

    Builds a sub-model that maps the input to (last-conv activations ``A``, backbone
    features), records the angle output ``y`` under :class:`tf.GradientTape`, takes
    ``g = d y / d A``, channel-weights via global-average-pooled gradients
    ``a_k = GAP(g)[k]``, forms the CAM ``ReLU(Σ_k a_k · A_k)``, resizes it to the
    input image's ``(H, W)``, and min–max normalizes to ``[0, 1]``.

    Parameters
    ----------
    model : keras.Model
        A trained image→angle pipeline (e.g. from
        :func:`uda.deploy.build_trained_pipeline`) whose backbone sub-model exposes
        convolutional layers.
    image : numpy.ndarray
        A single **preprocessed** input frame, ``(H, W, 3)`` or ``(1, H, W, 3)``
        (batch of one), ready for the backbone (use
        :func:`uda.models.backbones.preprocess_for`).
    last_conv_layer : str or None, keyword-only
        Name of the conv layer to explain. ``None`` (default) **auto-picks** the last
        4-D conv activation in the backbone sub-model.

    Returns
    -------
    numpy.ndarray
        An ``(H, W)`` ``float32`` heatmap in ``[0, 1]`` (``1`` = most important),
        ``H``/``W`` matching the input image.
    """
    import keras
    import tensorflow as tf

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"grad_cam takes a single image; got batch {arr.shape}")
        single = arr[0]
    elif arr.ndim == 3:
        single = arr
        arr = arr[np.newaxis, ...]
    else:
        raise ValueError(f"image must be (H, W, 3) or (1, H, W, 3); got {arr.shape}")
    in_h, in_w = single.shape[:2]

    backbone = _find_backbone(model)
    if last_conv_layer is None:
        last_conv_layer = _autopick_last_conv(backbone)
    conv_layer = backbone.get_layer(last_conv_layer)

    # Sub-model: backbone input -> (conv activations, backbone features). Reusing the
    # backbone's own input means the SAME single frozen forward feeds both outputs.
    grad_model = keras.Model(backbone.input, [conv_layer.output, backbone.output])

    x = tf.convert_to_tensor(arr)
    with tf.GradientTape() as tape:
        conv_out, features = grad_model(x, training=False)
        tape.watch(conv_out)
        y = _post_backbone_forward(model, backbone, features)
        scalar = _angle_scalar(y)
    grads = tape.gradient(scalar, conv_out)

    if grads is None:
        # No gradient path (output independent of conv activations) -> zero CAM.
        return np.zeros((in_h, in_w), dtype=np.float32)

    a_k = tf.reduce_mean(grads, axis=(0, 1, 2))  # GAP over (batch, h, w)
    cam = tf.reduce_sum(conv_out[0] * a_k, axis=-1)  # Σ_k a_k · A_k  -> (h, w)
    cam = tf.nn.relu(cam)  # keep only positive influence on the angle
    cam = tf.image.resize(cam[..., tf.newaxis], (in_h, in_w))[..., 0]
    return _normalize(np.asarray(cam.numpy(), dtype=np.float32))


def cam_figure(
    model, images, image_ids, out_dir: str | Path = "results/figures"
) -> list:
    """Save Grad-CAM overlays for a batch of images and return their paths.

    For each ``(image, image_id)`` pair computes :func:`grad_cam`, builds the
    :func:`overlay`, and writes it under ``out_dir`` as ``gradcam_<image_id>.png``
    (one file per image).

    Parameters
    ----------
    model : keras.Model
        The trained pipeline to explain (TF backend).
    images : sequence of numpy.ndarray
        Preprocessed input frames (each ``(H, W, 3)``).
    image_ids : sequence of str
        Stable ids used in the output filenames; ``len`` must match ``images``.
    out_dir : str or pathlib.Path
        Output directory (created if absent); default ``"results/figures"``.

    Returns
    -------
    list of pathlib.Path
        The saved overlay image paths, in input order.
    """
    import matplotlib.pyplot as plt

    images = list(images)
    image_ids = list(image_ids)
    if len(images) != len(image_ids):
        raise ValueError(
            f"len(images)={len(images)} != len(image_ids)={len(image_ids)}"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for image, image_id in zip(images, image_ids):
        heat = grad_cam(model, image)
        blended = overlay(image, heat)
        path = out / f"gradcam_{image_id}.png"
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        ax.imshow(blended)
        ax.set_title(f"Grad-CAM — {image_id}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
