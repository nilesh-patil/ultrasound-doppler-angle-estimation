"""Trained image→angle pipeline — assemble + load weights for inference/export.

This is the deployable artifact: a single
``Input((H, W, 3)) → Flatten(backbone) → head`` model whose head carries the
**trained** weights, so one call maps a preprocessed B-mode frame straight to a
Doppler angle. It is also the model the Grad-CAM module explains.

Training is **not** re-run end-to-end. Following the paper recipe (and
:func:`uda.training.train._fit_frozen`), the frozen backbone's features are cached **once**
via :func:`uda.training.train._extract_features`, a shallow head
(:func:`uda.models.heads.build_head`) is fit on those cached features, and the trained
head weights are **transferred** into the architecturally identical head of
:func:`uda.models.model.build_model`. Because both heads come from the *same*
``build_head(feature_dim(backbone), cfg.head, target.n_outputs)`` call, the layer
architectures match element-for-element and ``set_weights`` is a 1:1 copy.

This module *touches Keras* (``build_model`` / ``set_weights`` / ``Model.export``),
so it runs on the **TensorFlow backend** (``pixi run -e mac-gpu python ...``). It is
light: a frozen-backbone forward over the corpus plus a shallow-head fit — **no**
fine-tune, **no** OOM.
"""
from __future__ import annotations

from pathlib import Path

import keras
import numpy as np

from uda.models.backbones import build_backbone
from uda.config import ExperimentConfig
from uda.models.heads import build_head
from uda.models.model import build_model, feature_dim
from uda.seed import set_seed
from uda.models.targets import build_target
from uda.training.train import _callbacks, _extract_features

__all__ = ["build_trained_pipeline", "predict_angle", "export_savedmodel"]


def build_trained_pipeline(
    cfg: ExperimentConfig, *, max_images: int | None = None
) -> keras.Model:
    """Assemble the trained image→angle model (frozen backbone → trained head).

    Builds the corpus (:func:`uda.data.dataset.build_corpus`), extracts the
    **frozen** backbone features **once** via :func:`uda.training.train._extract_features`,
    fits a shallow head (:func:`uda.models.heads.build_head`) on those cached features
    (fast — avoids re-running the backbone every epoch), then assembles the
    image→angle model with :func:`uda.models.model.build_model` and **transfers** the
    trained head's weights into the assembled model's head sub-layer.

    Because the cached-feature head and the assembled model's head are built from
    the **same** ``build_head(feature_dim(backbone), cfg.head, target.n_outputs)``
    call, their layer architectures are identical and ``set_weights`` matches
    element-for-element.

    Parameters
    ----------
    cfg : uda.config.ExperimentConfig
        Full experiment config (backbone, head, target, optimizer) — e.g. loaded
        from ``configs/tuned_densenet201.yaml``. ``cfg.backbone.trainable`` must be
        ``False`` (deploy targets the frozen recipe only).
    max_images : int or None, keyword-only
        Cap on the number of *base* images consumed when building the corpus
        (passed through to :func:`build_corpus`); ``None`` (default) uses all 84.
        A small cap gives a fast smoke-build for tests.

    Returns
    -------
    keras.Model
        The assembled ``Input((H, W, 3)) → Flatten(backbone) → head`` model named
        ``"uda_<backbone>"``, with the **trained** head weights loaded.

    Raises
    ------
    ValueError
        If ``cfg.backbone.trainable`` is ``True`` — deploy reuses the cache-once
        frozen recipe and never fine-tunes (that would re-run the backbone per
        epoch and risk OOM).
    """
    if cfg.backbone.trainable:
        raise ValueError(
            "build_trained_pipeline targets the frozen paper recipe only "
            "(cache features once, fit the head); got cfg.backbone.trainable=True. "
            "Fine-tuning re-runs the backbone every epoch and is out of scope here."
        )

    set_seed(cfg.seed)
    target = build_target(cfg.target)

    # Import here so the corpus builder (Keras-free) is only pulled in on use and
    # tests can patch it; mirrors uda.training.train's lazy structure.
    from uda.data.dataset import build_corpus

    corpus = build_corpus(cfg, max_images=max_images)

    # 1. Cache the FROZEN backbone's features once (the paper recipe). This is the
    #    only place the backbone is run — never per epoch.
    backbone = build_backbone(cfg.backbone)
    f_train = _extract_features(cfg, backbone, corpus.x_train)

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(f_train.shape[0])
    f_train = f_train[perm]
    y_train = np.asarray(corpus.y_train)[perm]

    # 2. Fit the shallow head on the cached features — the SAME build_head call the
    #    assembled model uses, so the architectures are identical for transfer.
    head = build_head(f_train.shape[1], cfg.head, target.n_outputs)
    head.compile(
        optimizer=keras.optimizers.Adam(cfg.train.lr),
        loss=cfg.train.loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    head.fit(
        f_train,
        y_train,
        validation_split=0.15 if f_train.shape[0] >= 8 else 0.0,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        callbacks=_callbacks(cfg) if f_train.shape[0] >= 8 else [],
        verbose=0,
    )

    # 3. Assemble the deployable image→angle model and TRANSFER the trained head
    #    weights into its head sub-layer (1:1 copy — identical architecture).
    model = build_model(cfg)
    model.get_layer("head").set_weights(head.get_weights())
    return model


def predict_angle(cfg: ExperimentConfig, model: keras.Model, images: np.ndarray) -> np.ndarray:
    """Faithful image→angle inference for a deployed pipeline (degrees).

    Applies ``preprocess_for`` to the raw frames, runs the **eager** forward
    ``model(x, training=False)``, and decodes to degrees via the config's target.

    Uses ``model(x, training=False)`` and **not** ``model.predict()`` on purpose:
    with a nested *frozen*-BatchNorm backbone, the two diverge in Keras 3 (the
    ``predict`` path through the wrapped backbone does not run BN in the same mode as
    the eager call), and only the eager path reproduces the trained head's numbers
    — i.e. the value ``build_trained_pipeline`` actually fit. Always use this helper
    for deployed inference rather than ``model.predict``.

    Parameters
    ----------
    cfg : uda.config.ExperimentConfig
        The same config the pipeline was built from (for ``preprocess_for`` + target).
    model : keras.Model
        A pipeline from :func:`build_trained_pipeline`.
    images : numpy.ndarray
        **Raw** frames at the backbone's native size, shape ``(N, H, W, 3)`` (the
        model itself does *not* preprocess — feeding raw values to ``model`` directly
        is the silent-wrong-angle footgun this helper closes).

    Returns
    -------
    numpy.ndarray
        Decoded Doppler angles in degrees, shape ``(N,)``.
    """
    from uda.models.backbones import preprocess_for

    x = preprocess_for(cfg.backbone.name, np.asarray(images, dtype="float32"))
    encoded = np.asarray(model(x, training=False))
    target = build_target(cfg.target)
    return np.asarray(target.decode(encoded)).ravel()


def export_savedmodel(model: keras.Model, out_dir: str | Path) -> str:
    """Best-effort SavedModel export of a trained pipeline.

    Attempts ``model.export(out_dir)`` (the Keras 3 inference-graph SavedModel
    export). On **any** failure — most importantly a backend that does not support
    ``Model.export`` — it raises :class:`NotImplementedError` with a clear message
    that **names the backend limitation** (SavedModel export requires the
    TensorFlow backend). It never fails silently and never fabricates an export.

    Parameters
    ----------
    model : keras.Model
        The assembled, weight-loaded pipeline from :func:`build_trained_pipeline`.
    out_dir : str or pathlib.Path
        Destination directory for the SavedModel.

    Returns
    -------
    str
        ``str(out_dir)`` on a successful export.

    Raises
    ------
    NotImplementedError
        If ``model.export`` is unavailable or raises — with a message naming the
        backend limitation. (Never a silent or faked export.)
    """
    out = Path(out_dir)
    backend = keras.backend.backend()
    export = getattr(model, "export", None)
    if export is None:
        raise NotImplementedError(
            "SavedModel export requires the TensorFlow backend "
            f"(keras.Model.export is unavailable on the {backend!r} backend)."
        )
    try:
        export(str(out))
    except Exception as exc:  # noqa: BLE001 — re-raise honestly, never swallow.
        # Honesty contract: on any failure raise loudly, naming the backend limit,
        # and do NOT leave a stub/partial directory behind that looks like a real
        # export.
        if out.exists() and not any(out.rglob("*")):
            try:
                out.rmdir()
            except OSError:
                pass
        raise NotImplementedError(
            "SavedModel export requires the TensorFlow backend; "
            f"model.export failed on the {backend!r} backend: {exc}"
        ) from exc
    return str(out)
