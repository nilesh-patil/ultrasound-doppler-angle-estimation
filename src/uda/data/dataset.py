"""Corpus assembly — the integrating step of the data pipeline.

:func:`build_corpus` deterministically turns the 84 base
images into the augmented training corpus the model consumes. The flow order is
**mandatory** and encodes the leakage discipline:

1. **Load labels.** Read ``data/labels.csv`` (``image_id, patient_id,
   theta_deg``); if it is missing, build it from the repo-root ``Results.txt``
   via :func:`uda.data.labels.build_labels_csv`.
2. **Split first.** Partition the *base* ``image_id`` values with the strategy
   named by ``cfg.split`` (:mod:`uda.data.splits`). Augmentation has not yet
   happened, so no rotated copy of a held-out image can leak into training.
3. **Augment within each split.** Expand each base image into its rotation
   sweep with CLAHE + normalization (:func:`uda.data.augment.augment_image`),
   rotating the label with the image.
4. **Resize per backbone.** Resize every augmented frame to the backbone's
   native input size with :func:`skimage.transform.resize`.
5. **Grayscale -> 3-channel.** Stack the single grayscale channel three times so
   ImageNet-pretrained backbones receive an RGB-shaped tensor.

The encoded regression target is produced by the :class:`~uda.models.targets.AngleTarget`
selected by ``cfg.target`` (raw degrees by default), so ``y_train``/``y_test``
have a trailing axis of width ``target.n_outputs``.

This module is Keras-free (numpy, pandas, scikit-image, scikit-learn only) — the
per-backbone native input size is taken from a small local table rather than by
importing ``keras.applications``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.transform import resize as _sk_resize

from uda.config import BackboneConfig, ExperimentConfig
from uda.models.targets import build_target
from uda.data import augment as _augment
from uda.data import images as _images
from uda.data import labels as _labels
from uda.data import splits as _splits

__all__ = ["Corpus", "build_corpus", "native_input_size", "NATIVE_INPUT_SIZES"]

#: Native square input side (pixels) for each supported backbone, mirroring the
#: ``keras.applications`` defaults. Used to resize augmented frames without
#: importing Keras (this module is part of the Keras-free pipeline).
NATIVE_INPUT_SIZES: dict[str, tuple[int, int]] = {
    "vgg19": (224, 224),
    "resnet50": (224, 224),
    "densenet201": (224, 224),
    "xception": (299, 299),
    "inceptionv3": (299, 299),
    # Mirror uda.models.backbones.BACKBONES (drift-guarded by
    # tests/test_backbones_extra.py::test_backbones_and_dataset_tables_agree).
    "efficientnetb0": (224, 224),
    "efficientnetb1": (240, 240),
    "efficientnetb2": (260, 260),
    "efficientnetb3": (300, 300),
    # Modern backbones (mirror uda.models.backbones.BACKBONES).
    "convnext_tiny": (224, 224),
    "convnext_small": (224, 224),
    "convnext_base": (224, 224),
    "efficientnetv2b0": (224, 224),
    "efficientnetv2b1": (240, 240),
    "efficientnetv2b2": (260, 260),
    "efficientnetv2b3": (300, 300),
    "cnn_scratch": (128, 128),
}

#: Columns of :attr:`Corpus.meta`, one row per produced (augmented) sample.
META_COLUMNS = ["image_id", "patient_id", "rotation_deg", "split"]


@dataclass
class Corpus:
    """The assembled, split, augmented corpus.

    Attributes
    ----------
    x_train, x_test : numpy.ndarray
        Image tensors of shape ``(n, H, W, 3)``, ``float32`` in ``[0, 1]``, where
        ``(H, W)`` is the backbone's native input size.
    y_train, y_test : numpy.ndarray
        Encoded regression targets of shape ``(n, target.n_outputs)``,
        ``float32``. With the default raw-degrees target this is the Doppler
        angle in a trailing singleton axis.
    meta : pandas.DataFrame
        One row per produced sample with columns
        ``image_id, patient_id, rotation_deg, split`` (``split`` is ``"train"``
        or ``"test"``). Row order matches the concatenation order of
        ``x_train`` then ``x_test``.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    meta: pd.DataFrame


def native_input_size(name: str) -> tuple[int, int]:
    """Return the ``(height, width)`` native input size for a backbone name.

    Parameters
    ----------
    name : str
        A backbone name (e.g. ``"vgg19"``).

    Returns
    -------
    tuple[int, int]
        The documented native input size.

    Raises
    ------
    KeyError
        If ``name`` is not a known backbone.
    """
    if name not in NATIVE_INPUT_SIZES:
        raise KeyError(f"unknown backbone: {name!r}")
    return NATIVE_INPUT_SIZES[name]


def _resolve_existing(path: Path) -> Path | None:
    """Return the first existing path among ``path`` and repo-root candidates.

    ``DataConfig`` carries repo-relative defaults (``data/images``,
    ``data/labels.csv``). When the process is launched from somewhere other than
    the repo root we still want to find these files, so an absolute/relative
    ``path`` that does not exist is also probed against this package's repo root
    (``src/uda/data`` -> three parents up to the repo).
    """
    path = Path(path)
    if path.exists():
        return path
    if not path.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        candidate = repo_root / path
        if candidate.exists():
            return candidate
    return None


def _results_txt_for(images_dir: Path, labels_csv: Path) -> Path:
    """Locate the source ``Results.txt`` used to build ``labels.csv`` on demand.

    Probes, in order: a ``Results.txt`` beside the data directory
    (``images_dir.parent``), beside the labels CSV, the current working
    directory, and finally this package's repo root.

    Raises
    ------
    FileNotFoundError
        If no ``Results.txt`` can be found.
    """
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        Path(images_dir).parent / "Results.txt",
        Path(labels_csv).parent / "Results.txt",
        Path.cwd() / "Results.txt",
        repo_root / "data" / "Results.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not locate Results.txt to build labels.csv; looked in: "
        + ", ".join(str(c) for c in candidates)
    )


def _load_labels(cfg: ExperimentConfig) -> pd.DataFrame:
    """Load ``data/labels.csv``, building it from ``Results.txt`` if absent.

    Returns a frame with at least ``image_id, patient_id, theta_deg``.
    """
    labels_csv = cfg.data.labels_csv
    found = _resolve_existing(labels_csv)
    if found is not None:
        return pd.read_csv(found)

    results_txt = _results_txt_for(cfg.data.images_dir, labels_csv)
    # Write to the configured location (resolved against the repo root if the
    # configured path is repo-relative and the data dir lives there).
    out_csv = Path(labels_csv)
    if not out_csv.is_absolute() and not out_csv.parent.exists():
        repo_root = Path(__file__).resolve().parents[3]
        if (repo_root / Path(labels_csv).parent).exists():
            out_csv = repo_root / labels_csv
    return _labels.build_labels_csv(results_txt, out_csv)


def _images_dir(cfg: ExperimentConfig) -> Path:
    """Resolve the base-images directory, probing the repo root if needed."""
    found = _resolve_existing(cfg.data.images_dir)
    if found is None:
        raise FileNotFoundError(f"images_dir does not exist: {cfg.data.images_dir}")
    return found


def _to_three_channel(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a 2-D grayscale frame to ``size`` and stack to 3 channels.

    Parameters
    ----------
    img : numpy.ndarray
        2-D grayscale image, ``float`` in ``[0, 1]``.
    size : tuple[int, int]
        Target ``(height, width)``.

    Returns
    -------
    numpy.ndarray
        ``(height, width, 3)`` ``float32`` array in ``[0, 1]``.
    """
    resized = _sk_resize(
        img,
        size,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    resized = np.asarray(resized, dtype=np.float32)
    np.clip(resized, 0.0, 1.0, out=resized)
    return np.stack([resized, resized, resized], axis=-1)


def _select_base_images(
    cfg: ExperimentConfig, max_images: int | None
) -> tuple[list[Path], pd.DataFrame]:
    """List base images (optionally capped) and align labels to them.

    Returns the chosen image paths in deterministic order and the labels frame
    restricted to those ``image_id`` values (indexed by ``image_id``).
    """
    paths = _images.list_base_images(_images_dir(cfg))
    if max_images is not None:
        paths = paths[:max_images]

    labels = _load_labels(cfg)
    chosen_ids = {_images.image_id(p) for p in paths}
    labels = labels[labels["image_id"].isin(chosen_ids)].copy()
    return paths, labels


def build_corpus(
    cfg: ExperimentConfig, max_images: int | None = None
) -> Corpus:
    """Assemble the split, augmented corpus for ``cfg``.

    The flow depends on ``cfg.split.strategy``: the paper's
    ``"augmented"`` protocol augments every base image then splits the samples
    80/20 at random (rotated copies leak across the split); ``"image"`` and
    ``"patient"`` split the base images first, then augment within each split.
    Then resize to the backbone's native input and stack grayscale to 3 channels.

    Parameters
    ----------
    cfg : uda.config.ExperimentConfig
        Full experiment configuration. ``cfg.data`` drives augmentation,
        ``cfg.split`` the partition, ``cfg.target`` the label encoding, and
        ``cfg.backbone.name`` the resize target.
    max_images : int or None, optional
        Cap on the number of *base* images consumed (after deterministic
        sorting). ``None`` (default) uses all 84. With ``r`` rotations per image
        this yields ``max_images * r`` samples — used for fast/smoke runs.

    Returns
    -------
    Corpus
        Train/test tensors, encoded targets, and per-sample ``meta``.

    Notes
    -----
    The total sample count is ``len(base_images) * len(rotation_angles(cfg))``;
    with the defaults that is ``84 * 25 == 2100``.
    """
    paths, labels = _select_base_images(cfg, max_images)
    id_to_path = {_images.image_id(p): p for p in paths}

    size = native_input_size(cfg.backbone.name)
    target = build_target(cfg.target)

    label_rows = labels.set_index("image_id")
    has_patient = "patient_id" in labels.columns
    # Seeded generator for richer augmentation; a no-op unless any
    # richer-aug knob is set, but threaded so the corpus stays deterministic.
    richer_rng = np.random.default_rng(cfg.seed)

    def _augment_split(
        image_ids: np.ndarray, split_name: str
    ) -> tuple[list[np.ndarray], list[float], list[dict[str, object]]]:
        xs: list[np.ndarray] = []
        thetas: list[float] = []
        rows: list[dict[str, object]] = []
        # Deterministic order: sorted base ids, then ascending rotation angle.
        for image_id in sorted(image_ids.tolist()):
            path = id_to_path[image_id]
            gray = _images.load_image_gray(path)
            row = label_rows.loc[image_id]
            base_theta = float(row["theta_deg"])
            patient_id = row["patient_id"] if has_patient else None
            # 3. Augment within this split.
            for frame, new_theta, rotation_deg in _augment.augment_image(
                gray, base_theta, cfg.data, rng=richer_rng
            ):
                # 4 + 5. Resize to native input, stack to 3 channels.
                xs.append(_to_three_channel(frame, size))
                thetas.append(new_theta)
                rows.append(
                    {
                        "image_id": image_id,
                        "patient_id": patient_id,
                        "rotation_deg": int(rotation_deg),
                        "split": split_name,
                    }
                )
        return xs, thetas, rows

    if cfg.split.strategy == "augmented":
        # Paper protocol: augment everything, then split the samples 80/20 at
        # random -> rotated copies of a base image leak across the split.
        xs, thetas, rows = _augment_split(labels["image_id"].to_numpy(), "train")
        n = len(xs)
        n_test = int(round(n * cfg.split.test_size))
        rng = np.random.default_rng(cfg.split.seed)
        for pos in rng.permutation(n)[:n_test]:
            rows[int(pos)]["split"] = "test"
    else:
        # Split base images first, then augment within each split (no rotated-copy
        # leak): every rotation of a held-out base image stays out of training.
        strategy = _splits.build_split(cfg.split)
        train_ids, test_ids = next(iter(strategy.split(labels, cfg.split)))
        xs_tr, th_tr, rows_tr = _augment_split(train_ids, "train")
        xs_te, th_te, rows_te = _augment_split(test_ids, "test")
        xs = xs_tr + xs_te
        thetas = th_tr + th_te
        rows = rows_tr + rows_te

    train_pos = [i for i, r in enumerate(rows) if r["split"] == "train"]
    test_pos = [i for i, r in enumerate(rows) if r["split"] == "test"]

    x_train = _stack_images([xs[i] for i in train_pos], size)
    x_test = _stack_images([xs[i] for i in test_pos], size)
    y_train = _encode_targets(target, [thetas[i] for i in train_pos])
    y_test = _encode_targets(target, [thetas[i] for i in test_pos])
    meta = pd.DataFrame(
        [rows[i] for i in train_pos] + [rows[i] for i in test_pos],
        columns=META_COLUMNS,
    )

    return Corpus(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, meta=meta)


def _stack_images(frames: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """Stack per-sample ``(H, W, 3)`` frames into ``(n, H, W, 3)`` ``float32``."""
    if not frames:
        h, w = size
        return np.empty((0, h, w, 3), dtype=np.float32)
    return np.stack(frames, axis=0).astype(np.float32, copy=False)


def _encode_targets(target, thetas: list[float]) -> np.ndarray:
    """Encode angle labels to ``(n, target.n_outputs)`` ``float32``."""
    if not thetas:
        return np.empty((0, target.n_outputs), dtype=np.float32)
    encoded = target.encode(np.asarray(thetas, dtype=np.float64))
    return np.asarray(encoded, dtype=np.float32)
