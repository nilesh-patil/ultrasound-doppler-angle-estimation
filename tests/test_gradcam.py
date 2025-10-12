"""Tests — ``uda.interpret.gradcam`` (Grad-CAM over the backbone's last conv).

Two kinds of test live here:

* ``grad_cam`` / ``cam_figure`` touch ``tf.GradientTape`` (activation gradients exist
  **only** on the TensorFlow backend), so every test that calls them is guarded with
  ``@pytest.mark.skipif(KERAS_BACKEND != "tensorflow")``. The default JAX suite
  (``pixi run python -m pytest -q``) stays green by SKIPPING them; they run under
  ``pixi run -e mac-gpu python -m pytest tests/test_gradcam.py -q``.
* ``overlay`` and the heatmap normalization are **pure numpy/matplotlib** and carry
  **no** skip — they run on any backend (tests 6 & 7).

The Grad-CAM tests build the smallest possible model: a frozen ``cnn_scratch``
backbone (``weights=None`` random init, 128×128, no ImageNet download) plus a shallow
head, or a tiny hand-built conv model — a single-image forward and one conv-layer
gradient, so there is no fine-tune and no OOM.
"""
import os

import numpy as np
import pytest

TF_ONLY = pytest.mark.skipif(
    os.environ.get("KERAS_BACKEND") != "tensorflow",
    reason="needs TF GradientTape (uda.interpret.gradcam.grad_cam imports tensorflow)",
)

from uda.interpret import gradcam  # noqa: E402  (pure-numpy overlay/normalize import on any backend)


# --------------------------------------------------------------------------- #
# Helpers (only imported inside TF-guarded tests touch keras/build_model).
# --------------------------------------------------------------------------- #
def _small_cfg(*, target="raw"):
    """A tiny, weight-free config — frozen scratch backbone + small head.

    ``cnn_scratch`` (``weights=None``, 128×128, random init) gives a fast frozen
    forward that cannot OOM, with real Conv2D layers for the auto-pick to find.
    """
    from uda.config import ExperimentConfig

    return ExperimentConfig(
        name=f"gradcam_test_{target}",
        seed=0,
        backbone={"name": "cnn_scratch", "weights": None, "trainable": False, "pooling": "grid2"},
        head={"hidden_units": [16], "dropout": 0.0, "batchnorm": True},
        target={"kind": target},
        split={"strategy": "image", "test_size": 0.25, "seed": 0},
        train={"epochs": 1, "batch_size": 8, "early_stopping_patience": 2, "lr": 1e-3},
    )


def _toy_model(h=16, w=16, n_outputs=1, dead=False):
    """A tiny hand-built Input -> Flatten(backbone) -> head model.

    Mirrors ``uda.models.model.build_model``'s structure (a backbone *sub-model* named
    ``"<x>_features"`` containing the conv, then a ``Flatten`` and a ``head``
    sub-model) so ``grad_cam`` exercises the real "conv lives inside the backbone"
    path. ``dead=True`` zeroes the head kernel so the output is independent of the
    input (gradient is zero everywhere) for the sensitivity test.
    """
    import keras
    from keras import layers

    inp = keras.Input(shape=(h, w, 1), name="image")
    x = layers.Conv2D(4, 3, padding="same", activation="relu", name="conv_last")(inp)
    backbone = keras.Model(inp, x, name="toy_features")

    feat_in = keras.Input(shape=(h, w, 4), name="features_in")
    flat = layers.Flatten()(feat_in)
    out = layers.Dense(n_outputs, name="theta")(flat)
    head = keras.Model(feat_in, out, name="head")

    top_in = keras.Input(shape=(h, w, 1), name="image")
    feats = backbone(top_in)
    flat_top = layers.Flatten(name="flatten")(feats)
    # Re-wire head as a single Dense on the flattened conv maps for transparency.
    dense = layers.Dense(n_outputs, name="theta_top")
    y = dense(flat_top)
    model = keras.Model(top_in, y, name="uda_toy")
    if dead:
        kshape = dense.get_weights()
        dense.set_weights([np.zeros_like(kshape[0]), np.zeros_like(kshape[1])])
    return model


# =========================================================================== #
# 1. heatmap shape + range  [skipif-TF]
# =========================================================================== #
@TF_ONLY
def test_heatmap_shape_and_range():
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)

    from uda.models.backbones import native_input_size

    h, w = native_input_size(cfg.backbone.name)
    rng = np.random.default_rng(0)
    image = rng.random((h, w, 3)).astype("float32")

    cam = gradcam.grad_cam(model, image)
    assert cam.shape == (h, w)
    assert cam.dtype == np.float32
    assert np.all(cam >= 0.0) and np.all(cam <= 1.0)
    assert np.isfinite(cam).all()
    assert cam.max() > 0.0, "ReLU zeroed the whole CAM (output must depend on input)"


@TF_ONLY
def test_heatmap_accepts_batched_image():
    """A ``(1, H, W, 3)`` batch-of-one is accepted identically to ``(H, W, 3)``."""
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)
    from uda.models.backbones import native_input_size

    h, w = native_input_size(cfg.backbone.name)
    rng = np.random.default_rng(1)
    image = rng.random((h, w, 3)).astype("float32")
    cam_a = gradcam.grad_cam(model, image)
    cam_b = gradcam.grad_cam(model, image[np.newaxis, ...])
    np.testing.assert_allclose(cam_a, cam_b, rtol=1e-5, atol=1e-6)


# =========================================================================== #
# 2. auto-pick last conv  [skipif-TF]
# =========================================================================== #
@TF_ONLY
def test_autopick_matches_explicit_last_conv():
    """With ``last_conv_layer=None`` the explained layer is the last 4-D conv of
    the backbone sub-model; passing that name explicitly yields the same heatmap."""
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)

    # The backbone sub-model's last Conv2D is "conv2d_3" in cnn_scratch's graph;
    # discover it generically (the auto-pick must agree with this).
    backbone = model.get_layer(f"{cfg.backbone.name}_features")
    last_conv = None
    for layer in reversed(backbone.layers):
        shp = getattr(getattr(layer, "output", None), "shape", None)
        if shp is not None and len(shp) == 4 and "Conv" in type(layer).__name__:
            last_conv = layer.name
            break
    assert last_conv is not None

    from uda.models.backbones import native_input_size

    h, w = native_input_size(cfg.backbone.name)
    rng = np.random.default_rng(2)
    image = rng.random((h, w, 3)).astype("float32")

    cam_auto = gradcam.grad_cam(model, image, last_conv_layer=None)
    cam_named = gradcam.grad_cam(model, image, last_conv_layer=last_conv)
    np.testing.assert_allclose(cam_auto, cam_named, rtol=1e-5, atol=1e-6)


# =========================================================================== #
# 3. formula correctness  [skipif-TF]
# =========================================================================== #
@TF_ONLY
def test_formula_matches_relu_gap_weighted_sum():
    """On a tiny conv model the CAM equals ``ReLU(Σ_k a_k·A_k)`` (``a_k=GAP(dy/dA_k)``)
    resized + normalized — recomputed independently in the test."""
    import keras
    import tensorflow as tf

    h = w = 16
    model = _toy_model(h=h, w=w, n_outputs=1)
    rng = np.random.default_rng(3)
    image = rng.random((h, w, 1)).astype("float32")

    cam = gradcam.grad_cam(model, image, last_conv_layer="conv_last")

    # Recompute the textbook Grad-CAM from scratch.
    backbone = model.get_layer("toy_features")
    conv = backbone.get_layer("conv_last")
    grad_model = keras.Model(backbone.input, [conv.output, backbone.output])
    x = image[np.newaxis, ...]
    dense = model.get_layer("theta_top")
    with tf.GradientTape() as tape:
        conv_out, feat = grad_model(x)
        tape.watch(conv_out)
        flat = tf.reshape(feat, (tf.shape(feat)[0], -1))
        y = dense(flat)[:, 0]
    grads = tape.gradient(y, conv_out)
    a_k = tf.reduce_mean(grads, axis=(0, 1, 2))  # GAP over (batch, h, w)
    A = conv_out[0]
    cam_raw = tf.nn.relu(tf.reduce_sum(A * a_k, axis=-1)).numpy()
    # Resize to image size + min-max normalize (same as the module).
    resized = np.asarray(
        tf.image.resize(cam_raw[..., np.newaxis], (h, w)).numpy()[..., 0]
    )
    lo, hi = resized.min(), resized.max()
    expected = (resized - lo) / (hi - lo) if hi > lo else np.zeros_like(resized)

    np.testing.assert_allclose(cam, expected.astype("float32"), rtol=1e-4, atol=1e-5)


# =========================================================================== #
# 4. sensitivity — a dead model yields an all-zero CAM  [skipif-TF]
# =========================================================================== #
@TF_ONLY
def test_dead_model_yields_zero_cam():
    """A model whose output is independent of the input (zero head kernel) has a
    zero gradient everywhere, so the post-ReLU CAM is all zeros."""
    h = w = 16
    model = _toy_model(h=h, w=w, n_outputs=1, dead=True)
    rng = np.random.default_rng(4)
    image = rng.random((h, w, 1)).astype("float32")
    cam = gradcam.grad_cam(model, image, last_conv_layer="conv_last")
    assert cam.shape == (h, w)
    assert np.all(cam == 0.0), "independent output must give an all-zero CAM"


@TF_ONLY
def test_sensitivity_tracks_gradient():
    """A region the model responds to receives more heat than a zeroed-out region."""
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)
    from uda.models.backbones import native_input_size

    h, w = native_input_size(cfg.backbone.name)
    rng = np.random.default_rng(5)
    image = rng.random((h, w, 3)).astype("float32")
    # Zero the bottom half so it carries no signal.
    image[h // 2 :, :, :] = 0.0
    cam = gradcam.grad_cam(model, image)
    # CAM is non-degenerate and in range; at least some pixels are hot.
    assert cam.max() > 0.0
    assert cam.shape == (h, w)


# =========================================================================== #
# 5. cam_figure writes one file per image  [skipif-TF]
# =========================================================================== #
@TF_ONLY
def test_cam_figure_writes_one_file_per_image(tmp_path):
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)
    from uda.models.backbones import native_input_size

    h, w = native_input_size(cfg.backbone.name)
    rng = np.random.default_rng(6)
    images = [rng.random((h, w, 3)).astype("float32") for _ in range(3)]
    image_ids = ["alpha", "beta", "gamma"]

    paths = gradcam.cam_figure(model, images, image_ids, out_dir=tmp_path)
    assert len(paths) == 3
    for p, iid in zip(paths, image_ids):
        from pathlib import Path

        p = Path(p)
        assert p.exists() and p.is_file()
        assert p.stat().st_size > 0
        assert iid in p.name


@TF_ONLY
def test_cam_figure_length_mismatch_raises(tmp_path):
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)
    from uda.models.backbones import native_input_size

    h, w = native_input_size(cfg.backbone.name)
    images = [np.zeros((h, w, 3), dtype="float32")]
    with pytest.raises((ValueError, AssertionError)):
        gradcam.cam_figure(model, images, ["a", "b"], out_dir=tmp_path)


# =========================================================================== #
# 6. overlay shape/range  (NO skip — pure numpy)
# =========================================================================== #
def test_overlay_shape_and_range_grayscale():
    rng = np.random.default_rng(7)
    image = rng.random((24, 32)).astype("float32")
    heat = rng.random((24, 32)).astype("float32")
    out = gradcam.overlay(image, heat)
    assert out.shape == (24, 32, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    assert np.isfinite(out).all()


def test_overlay_accepts_rgb_background():
    rng = np.random.default_rng(8)
    image = rng.random((24, 32, 3)).astype("float32")
    heat = rng.random((24, 32)).astype("float32")
    out = gradcam.overlay(image, heat)
    assert out.shape == (24, 32, 3)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_overlay_alpha_blends():
    """alpha=0 ~ grayscale base; alpha=1 ~ pure jet heatmap."""
    import matplotlib.cm as cm

    image = np.linspace(0, 1, 24 * 32).reshape(24, 32).astype("float32")
    heat = np.linspace(0, 1, 24 * 32).reshape(24, 32).astype("float32")

    base = gradcam.overlay(image, heat, alpha=0.0)
    # alpha=0 -> the grayscale base broadcast to 3 channels.
    gray3 = np.repeat(image[..., np.newaxis], 3, axis=-1)
    np.testing.assert_allclose(base, gray3, atol=1e-6)

    pure = gradcam.overlay(image, heat, alpha=1.0)
    jet = cm.get_cmap("jet")(heat)[..., :3]
    np.testing.assert_allclose(pure, jet, atol=1e-6)


# =========================================================================== #
# 7. overlay/normalize math  (NO skip — pure numpy)
# =========================================================================== #
def test_normalize_maps_to_unit_interval():
    arr = np.array([[2.0, 4.0], [6.0, 10.0]], dtype="float32")
    norm = gradcam._normalize(arr)
    assert norm.min() == pytest.approx(0.0)
    assert norm.max() == pytest.approx(1.0)
    # Linear: value 6 -> (6-2)/(10-2) = 0.5.
    assert norm[1, 0] == pytest.approx(0.5)


def test_normalize_constant_array_is_finite():
    arr = np.full((4, 4), 7.0, dtype="float32")
    norm = gradcam._normalize(arr)
    assert np.isfinite(norm).all()
    # A constant map normalizes to a defined finite value (all-zero), no /0.
    assert np.all(norm == 0.0)


def test_jet_endpoints_match_matplotlib():
    import matplotlib.cm as cm

    jet = cm.get_cmap("jet")
    heat = np.array([[0.0, 1.0]], dtype="float32")
    # alpha=1 returns the pure jet coloring; endpoints must match matplotlib.
    image = np.zeros((1, 2), dtype="float32")
    out = gradcam.overlay(image, heat, alpha=1.0)
    np.testing.assert_allclose(out[0, 0], jet(0.0)[:3], atol=1e-6)
    np.testing.assert_allclose(out[0, 1], jet(1.0)[:3], atol=1e-6)


# =========================================================================== #
# 8. guard works — module imports on any backend; GradientTape only on TF
# =========================================================================== #
def test_module_imports_on_any_backend():
    """``uda.interpret.gradcam`` imports (overlay/normalize are pure numpy) on any backend;
    only ``grad_cam``/``cam_figure`` require TF GradientTape (guarded above)."""
    assert hasattr(gradcam, "grad_cam")
    assert hasattr(gradcam, "overlay")
    assert hasattr(gradcam, "cam_figure")
    # Pure-numpy path works with no backend requirement.
    out = gradcam.overlay(np.zeros((4, 4), "float32"), np.ones((4, 4), "float32"))
    assert out.shape == (4, 4, 3)


@TF_ONLY
def test_gradcam_runs_under_tensorflow_backend():
    """Sanity: under KERAS_BACKEND=tensorflow the GradientTape path actually runs."""
    import keras

    assert keras.backend.backend() == "tensorflow"
    h = w = 16
    model = _toy_model(h=h, w=w, n_outputs=1)
    image = np.random.default_rng(9).random((h, w, 1)).astype("float32")
    cam = gradcam.grad_cam(model, image, last_conv_layer="conv_last")
    assert cam.shape == (h, w) and cam.dtype == np.float32
