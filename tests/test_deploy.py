"""Tests — ``uda.deploy`` (trained image→angle pipeline).

Every test touches Keras (build_model / set_weights / predict / model.export), so
the whole module is guarded ``skipif KERAS_BACKEND != "tensorflow"``: the default
JAX suite (``pixi run python -m pytest -q``) stays green by SKIPPING every test
here, and they run under ``pixi run -e mac-gpu python -m pytest tests/test_deploy.py
-q`` (KERAS_BACKEND=tensorflow, Apple Metal GPU).

The pipeline is the paper recipe: a **frozen** backbone forward over a *small*
corpus (``cnn_scratch`` / ``weights=None`` and a tiny ``max_images``) plus a shallow
head fit on the cached features — no fine-tune, no OOM. Each test uses the smallest
config that exercises the contract.
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("KERAS_BACKEND") != "tensorflow",
    reason="needs TF backend (uda.deploy imports keras: build_model/set_weights/export)",
)

from uda.config import ExperimentConfig  # noqa: E402
from uda.models.model import build_model, feature_dim  # noqa: E402
from uda.models.backbones import build_backbone, native_input_size, preprocess_for  # noqa: E402
from uda.models.targets import build_target  # noqa: E402


def _small_cfg(*, name="cnn_scratch", target="raw"):
    """A tiny, weight-free config — frozen scratch backbone + small head.

    ``cnn_scratch`` has ``weights=None`` (no ImageNet download, random init) and a
    128×128 native input, so a frozen forward over a handful of base images is fast
    and cannot OOM on the Metal GPU.
    """
    return ExperimentConfig(
        name=f"deploy_test_{name}_{target}",
        seed=0,
        backbone={"name": name, "weights": None, "trainable": False, "pooling": "grid2"},
        head={"hidden_units": [16], "dropout": 0.0, "batchnorm": True},
        target={"kind": target},
        # "image" split partitions base images (works with a small cap); the deploy
        # contract is split-strategy agnostic, and a small base count keeps the
        # frozen forward fast. (The first ~12 base images all share patient 0, so a
        # "patient" split would leave an empty train set under a tiny max_images.)
        split={"strategy": "image", "test_size": 0.25, "seed": 0},
        train={"epochs": 3, "batch_size": 16, "early_stopping_patience": 2, "lr": 1e-3},
    )


def _flat(weights):
    return np.concatenate([np.asarray(w).ravel() for w in weights]) if weights else np.array([])


# --- 1. pipeline shape + name ------------------------------------------------
def test_pipeline_shape_and_name():
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)

    import keras

    assert isinstance(model, keras.Model)
    assert model.name == f"uda_{cfg.backbone.name}"
    h, w = native_input_size(cfg.backbone.name)
    assert tuple(model.input_shape) == (None, h, w, 3)
    assert model.output_shape[-1] == build_target(cfg.target).n_outputs == 1


def test_pipeline_sincos_output_width():
    from uda import deploy

    cfg = _small_cfg(target="sincos2theta")
    model = deploy.build_trained_pipeline(cfg, max_images=8)
    assert model.output_shape[-1] == 2


# --- predict_angle uses the faithful eager path (not model.predict) ----------
def test_predict_angle_matches_eager_path():
    """``predict_angle`` = decode of ``model(x, training=False)`` with
    ``preprocess_for`` applied internally — the eager path that reproduces the
    trained head (``model.predict`` can diverge with a nested frozen-BN backbone, so
    deployed inference must go through this helper)."""
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)
    h, w = native_input_size(cfg.backbone.name)
    rng = np.random.default_rng(0)
    images = rng.uniform(0.0, 1.0, size=(3, h, w, 3)).astype("float32")

    ang = deploy.predict_angle(cfg, model, images)
    assert ang.shape == (3,)
    assert np.all(np.isfinite(ang))

    x = preprocess_for(cfg.backbone.name, images)
    manual = np.asarray(
        build_target(cfg.target).decode(np.asarray(model(x, training=False)))
    ).ravel()
    assert np.allclose(ang, manual)


# --- 2. head weights are the trained ones (transfer correctness) -------------
def test_head_weights_are_trained_not_random(monkeypatch):
    """The assembled model's head sub-layers hold exactly the arrays that were
    fit on the cached features — captured at the ``set_weights`` boundary — and
    differ from a fresh random ``build_model(cfg)``."""
    from uda import deploy

    cfg = _small_cfg(target="raw")

    # Spy on the head sub-model's set_weights to capture the transferred arrays.
    captured = {}
    import keras

    orig_set_weights = keras.Model.set_weights

    def spy_set_weights(self, weights):
        if self.name == "head":
            captured["weights"] = [np.asarray(w).copy() for w in weights]
        return orig_set_weights(self, weights)

    monkeypatch.setattr(keras.Model, "set_weights", spy_set_weights)

    model = deploy.build_trained_pipeline(cfg, max_images=8)

    assert "weights" in captured, "head weights were never transferred via set_weights"

    head_layer = model.get_layer("head")
    transferred = _flat(captured["weights"])
    loaded = _flat(head_layer.get_weights())
    assert transferred.shape == loaded.shape
    np.testing.assert_allclose(loaded, transferred, rtol=0, atol=0)

    # And they are NOT the random init of a fresh assembled model.
    fresh_head = build_model(cfg).get_layer("head")
    fresh = _flat(fresh_head.get_weights())
    assert fresh.shape == loaded.shape
    assert not np.allclose(loaded, fresh), "head weights look like fresh random init"


# --- 3. cache-once / frozen (no fine-tune) -----------------------------------
def test_backbone_is_frozen_and_features_cached_once(monkeypatch):
    """The backbone sub-model is frozen (no trainable backbone weights) and
    ``_extract_features`` is the SOLE backbone forward — called once per split,
    never per epoch."""
    from uda import deploy
    from uda.training import train as train_mod

    cfg = _small_cfg(target="raw")

    calls = {"n": 0}
    orig = train_mod._extract_features

    def counting_extract(cfg_, backbone, x):
        calls["n"] += 1
        return orig(cfg_, backbone, x)

    # Patch the symbol deploy actually calls (it reuses train._extract_features).
    monkeypatch.setattr(deploy, "_extract_features", counting_extract, raising=False)
    monkeypatch.setattr(train_mod, "_extract_features", counting_extract)

    model = deploy.build_trained_pipeline(cfg, max_images=8)

    # One forward for training features (head fits on cached features, not images).
    assert calls["n"] == 1, f"_extract_features called {calls['n']}x (expected 1, per epoch => leak)"

    backbone = model.get_layer(f"{cfg.backbone.name}_features")
    assert backbone.trainable is False
    assert len(backbone.trainable_weights) == 0
    # The only trained weights in the pipeline are the head's.
    head = model.get_layer("head")
    assert len(head.trainable_weights) > 0


def test_rejects_trainable_backbone():
    """deploy targets the frozen recipe only — a trainable backbone is rejected."""
    from uda import deploy

    cfg = _small_cfg(target="raw")
    cfg = cfg.model_copy(update={"backbone": cfg.backbone.model_copy(update={"trainable": True})})
    with pytest.raises((ValueError, NotImplementedError)):
        deploy.build_trained_pipeline(cfg, max_images=8)


# --- 4. inference runs end-to-end --------------------------------------------
def test_predict_one_image_is_scalar_angle():
    from uda import deploy
    from uda.data.dataset import build_corpus

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)
    target = build_target(cfg.target)

    corpus = build_corpus(cfg, max_images=8)
    x = preprocess_for(cfg.backbone.name, corpus.x_test[:1])
    pred_enc = np.asarray(model.predict(x, verbose=0))
    assert pred_enc.shape == (1, target.n_outputs)
    assert np.all(np.isfinite(pred_enc))

    angle = np.asarray(target.decode(pred_enc)).ravel()
    assert angle.shape == (1,)
    assert np.isfinite(angle[0])
    # raw-degrees decode is the identity; the network is unconstrained but finite.
    assert np.isfinite(float(angle[0]))


def test_predict_batch_finite():
    from uda import deploy
    from uda.data.dataset import build_corpus

    cfg = _small_cfg(target="sincos2theta")
    model = deploy.build_trained_pipeline(cfg, max_images=8)
    target = build_target(cfg.target)
    corpus = build_corpus(cfg, max_images=8)
    k = min(4, corpus.x_test.shape[0])
    x = preprocess_for(cfg.backbone.name, corpus.x_test[:k])
    pred = np.asarray(model.predict(x, verbose=0))
    assert pred.shape == (k, 2)
    assert np.all(np.isfinite(pred))
    angles = np.asarray(target.decode(pred)).ravel()
    assert np.all((angles >= 0.0) & (angles < 180.0))


# --- 5. export success path --------------------------------------------------
def test_export_savedmodel_writes_real_dir(tmp_path):
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)

    out = tmp_path / "savedmodel"
    ret = deploy.export_savedmodel(model, out)
    assert ret == str(out)
    assert out.exists() and out.is_dir()
    # A real SavedModel writes a non-empty directory tree.
    files = list(out.rglob("*"))
    assert any(f.is_file() and f.stat().st_size > 0 for f in files), "export produced no files"


# --- 6. export honesty (never fake) ------------------------------------------
def test_export_raises_when_export_unavailable(tmp_path, monkeypatch):
    """If ``model.export`` raises (e.g. unsupported backend), ``export_savedmodel``
    re-raises ``NotImplementedError`` naming the backend limit — and writes no stub."""
    from uda import deploy

    cfg = _small_cfg(target="raw")
    model = deploy.build_trained_pipeline(cfg, max_images=8)

    def boom(*a, **k):
        raise RuntimeError("export not supported on this backend")

    monkeypatch.setattr(model, "export", boom, raising=False)

    out = tmp_path / "should_not_exist"
    with pytest.raises(NotImplementedError) as exc:
        deploy.export_savedmodel(model, out)
    msg = str(exc.value).lower()
    assert "backend" in msg or "tensorflow" in msg, f"message must name the backend limit: {exc.value!r}"
    # No stub directory was fabricated.
    assert not out.exists(), "export wrote a stub dir on failure (must be loud, not faked)"
